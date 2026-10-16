"""DeepSeek V4.1 on the native engine (C++ and Metal, no MLX), behind the runtime interface.

The engine (Native/DeepSeekNative, libWhallmDeepSeekNative.dylib) computes logits and
DSpark draft logits. NativeQwenRuntime's sampling, prompt cache and text streaming are
shared; this module opens the engine and runs DSpark rounds.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
from transformers import AutoTokenizer

from .cancellation import GenerationCancelled, check_cancelled
from .dspark import DraftResult, VerificationMetrics
from .generation import RuntimeMetrics
from .manifest import InstalledModel
from .model import RuntimeConfig
from .model_support import get_support
from .native_qwen import NativeExpertCounters, NativeQwenRuntime

LIBRARY_NAME = "libWhallmDeepSeekNative.dylib"


class _Options(ctypes.Structure):
    _fields_ = [("slots", ctypes.c_int32), ("read_threads", ctypes.c_int32), ("chunk", ctypes.c_int32),
                ("token_map", ctypes.POINTER(ctypes.c_int32)), ("token_map_count", ctypes.c_int32),
                ("dspark", ctypes.c_int32)]


class _Counters(ctypes.Structure):
    _fields_ = [("expert_hits", ctypes.c_int64), ("expert_misses", ctypes.c_int64),
                ("experts_seeded", ctypes.c_int64), ("experts_streamed", ctypes.c_int64),
                ("read_seconds", ctypes.c_double), ("gpu_seconds", ctypes.c_double),
                ("gpu_bytes", ctypes.c_uint64), ("slots", ctypes.c_int32), ("resident", ctypes.c_int32)]


_CANCEL = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_library_lock = threading.Lock()
_library_value = None


def _library_candidates() -> list[Path]:
    candidates = []
    configured = os.environ.get("WHALLM_DEEPSEEK_NATIVE")
    if configured:
        candidates.append(Path(configured))
    executable = Path(sys.executable).resolve()
    candidates.append(executable.parent.parent / "Frameworks" / LIBRARY_NAME)
    candidates.append(Path(__file__).resolve().parents[2] / ".build" / "native" / LIBRARY_NAME)
    return candidates


def _library():
    global _library_value
    with _library_lock:
        if _library_value is not None:
            return _library_value
        path = next((p for p in _library_candidates() if p.is_file()), None)
        if path is None:
            raise RuntimeError("The native DeepSeek engine library is missing. Run Scripts/build-deepseek-native.sh.")
        lib = ctypes.CDLL(str(path))
        P, I32, F = ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_float)
        tokens = ctypes.POINTER(ctypes.c_int32)
        signatures = {
            "wd_open": (P, [ctypes.c_char_p, ctypes.POINTER(_Options), ctypes.c_char_p, ctypes.c_size_t]),
            "wd_close": (None, [P]),
            "wd_error": (ctypes.c_char_p, [P]),
            "wd_set_cancel": (None, [P, _CANCEL, P]),
            "wd_vocab": (I32, [P]),
            "wd_position": (I32, [P]),
            "wd_tokens": (tokens, [P, ctypes.POINTER(I32)]),
            "wd_contract": (ctypes.c_char_p, [P]),
            "wd_counters_get": (None, [P, ctypes.POINTER(_Counters)]),
            "wd_reset": (ctypes.c_int, [P]),
            "wd_eval": (ctypes.c_int, [P, tokens, I32, I32, ctypes.POINTER(F)]),
            "wd_has_dspark": (I32, [P]),
            "wd_draft_block": (I32, [P]),
            "wd_draft": (ctypes.c_int, [P, I32]),
            "wd_draft_step": (ctypes.c_int, [P, I32, I32, ctypes.POINTER(F), ctypes.POINTER(ctypes.c_float)]),
            "wd_verify": (ctypes.c_int, [P, tokens, I32, ctypes.POINTER(F)]),
            "wd_commit": (ctypes.c_int, [P, I32]),
            "wd_snapshot": (P, [P]),
            "wd_restore": (ctypes.c_int, [P, P]),
            "wd_load": (P, [P, ctypes.c_char_p]),
            "wd_save": (ctypes.c_int, [P, P, ctypes.c_char_p]),
            "wd_state_free": (None, [P]),
            "wd_state_token_count": (I32, [P]),
            "wd_state_tokens": (tokens, [P]),
            "wd_state_bytes": (ctypes.c_uint64, [P]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(lib, name)
            function.restype = result
            function.argtypes = arguments
        _library_value = lib
        return lib


def _int32s(values: Sequence[int]):
    return (ctypes.c_int32 * len(values))(*values)


@lru_cache(maxsize=2)
def token_map(root: str) -> np.ndarray:
    """The compressed Engram id of every vocabulary id (deepseek_v41/engram.py)."""
    from transformers import PreTrainedTokenizerFast
    from .deepseek_v41.engram import build_compressed_token_map
    tokenizer = PreTrainedTokenizerFast.from_pretrained(Path(root) / "tokenizer")
    mapping, _ = build_compressed_token_map(tokenizer)
    return np.ascontiguousarray(mapping, dtype=np.int32)


class NativeState:
    """One prompt cache state of the engine (owned; freed with close or garbage collection)."""

    def __init__(self, handle: int):
        self._handle = handle
        lib = _library()
        count = lib.wd_state_token_count(handle)
        self.tokens = tuple(np.ctypeslib.as_array(lib.wd_state_tokens(handle), shape=(count,)).tolist()) if count else ()
        self.nbytes = int(lib.wd_state_bytes(handle))

    @property
    def closed(self) -> bool:
        return self._handle is None

    @property
    def handle(self) -> int:
        if self._handle is None:
            raise ValueError("the prompt cache state is closed")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            _library().wd_state_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


class NativeEngine:
    """The engine's C interface. After any failure the engine is reset to no tokens."""

    def __init__(self, root: Path, *, slots: int, read_threads: int, dspark: bool, mapping: np.ndarray):
        self._lib = _library()
        self._map = mapping  # the engine copies it; kept for the call
        options = _Options(slots, read_threads, 0, mapping.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                           len(mapping), 1 if dspark else 0)
        error = ctypes.create_string_buffer(4096)
        handle = self._lib.wd_open(str(root).encode(), ctypes.byref(options), error, len(error))
        if not handle:
            raise RuntimeError(f"The native DeepSeek engine could not open the model: {error.value.decode(errors='replace')}")
        self._handle = handle
        self._cancel = _CANCEL(self._cancelled)  # kept alive while the engine uses it
        self._lib.wd_set_cancel(handle, self._cancel, None)
        self.vocab = int(self._lib.wd_vocab(handle))
        self.has_dspark = bool(self._lib.wd_has_dspark(handle))
        self.draft_block = int(self._lib.wd_draft_block(handle))
        self.contract = self._lib.wd_contract(handle).decode()

    has_mtp = False

    @staticmethod
    def _cancelled(_context) -> int:
        try:
            check_cancelled()
        except GenerationCancelled:
            return 1
        return 0

    def close(self) -> None:
        if self._handle is not None:
            self._lib.wd_set_cancel(self._handle, _CANCEL(), None)
            self._lib.wd_close(self._handle)
            self._handle = None

    def _check(self, status: int) -> None:
        if status == 0:
            return
        message = self._lib.wd_error(self._handle).decode(errors="replace")
        self._lib.wd_reset(self._handle)
        if status == 2:
            raise GenerationCancelled("Client disconnected")
        raise RuntimeError(f"Native DeepSeek engine: {message}")

    def _logits(self, pointer, rows: int) -> np.ndarray:
        return np.ctypeslib.as_array(pointer, shape=(rows, self.vocab)).copy()

    @property
    def tokens(self) -> tuple[int, ...]:
        count = ctypes.c_int32()
        pointer = self._lib.wd_tokens(self._handle, ctypes.byref(count))
        return tuple(np.ctypeslib.as_array(pointer, shape=(count.value,)).tolist()) if count.value else ()

    def counters(self) -> _Counters:
        counters = _Counters()
        self._lib.wd_counters_get(self._handle, ctypes.byref(counters))
        return counters

    def reset(self) -> None:
        self._check(self._lib.wd_reset(self._handle))

    def eval(self, tokens: Sequence[int], logit_rows: int = 1) -> np.ndarray:
        logits = ctypes.POINTER(ctypes.c_float)()
        self._check(self._lib.wd_eval(self._handle, _int32s(tokens), len(tokens), logit_rows, ctypes.byref(logits)))
        return self._logits(logits, logit_rows) if logit_rows else np.empty((0, self.vocab), np.float32)

    def draft(self, anchor: int) -> None:
        self._check(self._lib.wd_draft(self._handle, anchor))

    def draft_step(self, index: int, previous: int) -> tuple[np.ndarray, float]:
        logits = ctypes.POINTER(ctypes.c_float)()
        confidence = ctypes.c_float()
        self._check(self._lib.wd_draft_step(self._handle, index, previous, ctypes.byref(logits), ctypes.byref(confidence)))
        return self._logits(logits, 1)[0], float(confidence.value)

    def verify(self, tokens: Sequence[int]) -> np.ndarray:
        logits = ctypes.POINTER(ctypes.c_float)()
        self._check(self._lib.wd_verify(self._handle, _int32s(tokens), len(tokens), ctypes.byref(logits)))
        return self._logits(logits, len(tokens))

    def commit(self, keep: int) -> None:
        self._check(self._lib.wd_commit(self._handle, keep))

    def snapshot(self) -> NativeState:
        handle = self._lib.wd_snapshot(self._handle)
        if not handle:
            self._check(1)
        return NativeState(handle)

    def restore(self, state: NativeState) -> None:
        self._check(self._lib.wd_restore(self._handle, state.handle))

    def save(self, state: NativeState, path: Path) -> None:
        self._check(self._lib.wd_save(self._handle, state.handle, str(path).encode()))

    def load(self, path: Path) -> NativeState:
        handle = self._lib.wd_load(self._handle, str(path).encode())
        if not handle:
            self._check(1)
        return NativeState(handle)


class DraftBackoff:
    """After two drafted rounds in a row without an accepted draft, decode 1, then 2,
    4 ... 16 tokens without drafting; an accepted draft resets the wait. Measured
    2026-10-16 on M2 Max (Novel context): DSpark without it decoded 2.65 tok/s
    against 2.71 without DSpark, with it 2.70; on Code it did not change the speed."""

    def __init__(self):
        self.skip = 0
        self.wait = 1
        self.streak = 0

    def draft_now(self) -> bool:
        if self.skip == 0:
            return True
        self.skip -= 1
        return False

    def observe(self, accepted: int) -> None:
        if accepted > 0:
            self.streak = 0
            self.wait = 1
            return
        self.streak += 1
        if self.streak >= 2:
            self.skip = self.wait
            self.wait = min(self.wait * 2, 16)
            self.streak = 0


class NativeDeepSeekRuntime(NativeQwenRuntime):
    """DeepSeek V4.1 on the native engine; requests are serialized."""

    STATE_MAGIC = b"WDSTATE1"
    STATE_SUFFIX = ".wdstate"

    def __init__(self, installed: InstalledModel, config: RuntimeConfig, model_kind: str):
        self.installed = installed
        self.support = get_support(model_kind)
        self.support.validate_config(config)
        self._model_id = installed.model_id
        self.metrics = RuntimeMetrics()
        self._codec = None
        self._lock = threading.Lock()
        slots = 0 if config.native_automatic_expert_cache else self._configured_slots(installed, config)
        dspark = bool(config.dspark_enabled and installed.has_dspark)
        self.engine = NativeEngine(installed.root, slots=slots, read_threads=max(1, int(config.read_workers)),
                                   dspark=dspark, mapping=token_map(str(installed.root)))
        try:
            self.expert_cache = NativeExpertCounters(self.engine, installed.expert_blob_size)
            self.config = replace(config, slots=int(self.engine.counters().slots), dspark_enabled=dspark)
            self.tokenizer = AutoTokenizer.from_pretrained(installed.root / "tokenizer", trust_remote_code=True)
            self._entries = []
            self._persist_later = []
            self._directory = self._open_prompt_cache_directory()
            self._vision_verified = False
        except BaseException:
            self.engine.close()
            raise

    @classmethod
    def open(cls, model_path: str, config: RuntimeConfig, model_kind: str) -> NativeDeepSeekRuntime:
        return cls(InstalledModel.open(model_path), config, model_kind)

    def _speculative(self) -> bool:
        return self.engine.has_dspark

    def _speculative_metrics(self, enabled: bool) -> dict:
        return {"dspark_enabled": enabled}

    def _tokens(self, prompt_tokens, reused, images, options, sampler, speculative) -> Iterator[int]:
        """Sampled tokens, each yielded before the model consumes it. With DSpark, a round
        drafts up to 5 tokens (stopping at the first below the confidence threshold) and
        verifies them in one pass; the sampler keeps the target distribution exact."""
        if images is not None:
            raise ValueError("DeepSeek V4.1 does not take images")
        engine = self.engine
        history = [prompt_tokens[-1]]
        logits = self._prefill(prompt_tokens[reused:])
        anchor = sampler.sample(sampler.distribution(logits, history))
        yield anchor
        history.append(anchor)
        generated = 1
        threshold = float(self.config.dspark_confidence_threshold)
        backoff = DraftBackoff()
        while generated < options.max_tokens:
            check_cancelled()
            budget = min(engine.draft_block, options.max_tokens - generated - 1)
            if not speculative or budget < 1 or not backoff.draft_now():
                anchor = sampler.sample(sampler.distribution(engine.eval([anchor])[0], history))
                yield anchor
                history.append(anchor)
                generated += 1
                continue
            started = time.perf_counter()
            engine.draft(anchor)
            drafts, distributions, confidence = [], [], []
            previous = anchor
            for index in range(budget):
                check_cancelled()
                row, value = engine.draft_step(index, previous)
                if value < threshold:
                    break
                distribution = sampler.distribution(row, history + drafts)
                previous = sampler.sample(distribution)
                drafts.append(previous)
                distributions.append(distribution)
                confidence.append(value)
            verify_started = time.perf_counter()
            rows = engine.verify([anchor, *drafts])
            targets = [sampler.distribution(rows[i], history + drafts[:i]) for i in range(len(drafts) + 1)]
            accepted, next_token = sampler.verify(drafts, distributions, targets, engine.vocab)
            engine.commit(accepted + 1)
            backoff.observe(accepted)
            committed = min(accepted + 1, options.max_tokens - generated)
            self.metrics.record_dspark_round(
                DraftResult(drafts, [], confidence, verify_started - started), accepted,
                time.perf_counter() - verify_started,
                VerificationMetrics(verification_mode="single", committed_tokens=committed))
            for token in drafts[:accepted]:
                if generated >= options.max_tokens:
                    return
                yield token
                history.append(token)
                generated += 1
            if generated >= options.max_tokens:
                return
            anchor = next_token
            yield anchor
            history.append(anchor)
            generated += 1
