"""Qwen3.8 and Swift1.5 Qwen on the native engine (C++ and Metal, no MLX), behind the runtime interface.

The engine (Native/QwenNative, libWhallmQwenNative.dylib) computes logits; this module
samples tokens, runs MTP draft rounds, keeps the prompt cache and turns tokens into
text the way ModelRuntime does, so the server, Chat and Throughput use it unchanged.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

from .cancellation import GenerationCancelled, check_cancelled
from .expert_cache import CacheMetrics
from .generation import (
    ChatInterface, GeneratedPiece, GenerationOptions, RuntimeMetrics, fit_context_window,
)
from .manifest import InstalledModel
from .media import MediaRequest
from .model import RuntimeConfig
from .model_support import get_support
from .qwen_mtp_policy import MTPDraftPolicy

LIBRARY_NAME = "libWhallmQwenNative.dylib"
# Prompts are fed in segments of at most this many tokens: each segment reads every
# expert file once, and the engine's per-call scratch grows with the segment.
PREFILL_SEGMENT = 16_384
# From this many new prompt tokens, the state before the last prompt token is kept
# as a prompt cache entry too, so the same prompt can be answered again.
PROMPT_SNAPSHOT_MIN_TOKENS = 128
_PENALTY_CONTEXT = 20  # as mlx_lm's presence and repetition penalties


# ---------------------------------------------------------------------- library


class _Options(ctypes.Structure):
    _fields_ = [("slots", ctypes.c_int32), ("read_threads", ctypes.c_int32), ("mtp", ctypes.c_int32),
                ("eight_bit", ctypes.c_int32), ("vision_catalog", ctypes.c_char_p)]


class _Counters(ctypes.Structure):
    _fields_ = [("expert_hits", ctypes.c_int64), ("expert_misses", ctypes.c_int64),
                ("experts_seeded", ctypes.c_int64), ("read_seconds", ctypes.c_double),
                ("gpu_seconds", ctypes.c_double), ("gpu_bytes", ctypes.c_uint64),
                ("slots", ctypes.c_int32), ("resident", ctypes.c_int32), ("tensor_ops", ctypes.c_int32)]


_CANCEL = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_library_lock = threading.Lock()
_library_value = None


def _library_candidates() -> list[Path]:
    candidates = []
    configured = os.environ.get("WHALLM_QWEN_NATIVE")
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
            raise RuntimeError("The native Qwen engine library is missing. Run Scripts/build-qwen-native.sh.")
        lib = ctypes.CDLL(str(path))
        P, I32, F = ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_float)
        tokens = ctypes.POINTER(ctypes.c_int32)
        signatures = {
            "wq_open": (P, [ctypes.c_char_p, ctypes.POINTER(_Options), ctypes.c_char_p, ctypes.c_size_t]),
            "wq_close": (None, [P]),
            "wq_error": (ctypes.c_char_p, [P]),
            "wq_set_cancel": (None, [P, _CANCEL, P]),
            "wq_vocab": (I32, [P]),
            "wq_position": (I32, [P]),
            "wq_has_mtp": (I32, [P]),
            "wq_tokens": (tokens, [P, ctypes.POINTER(I32)]),
            "wq_contract": (ctypes.c_char_p, [P]),
            "wq_counters_get": (None, [P, ctypes.POINTER(_Counters)]),
            "wq_reset": (ctypes.c_int, [P]),
            "wq_eval": (ctypes.c_int, [P, tokens, I32, I32, ctypes.POINTER(F)]),
            "wq_draft": (ctypes.c_int, [P, I32, ctypes.POINTER(F)]),
            "wq_verify": (ctypes.c_int, [P, tokens, I32, ctypes.POINTER(F)]),
            "wq_commit": (ctypes.c_int, [P, I32]),
            "wq_eval_images": (ctypes.c_int, [P, tokens, I32, I32, tokens, ctypes.POINTER(F), I32, ctypes.POINTER(F)]),
            "wq_snapshot": (P, [P]),
            "wq_restore": (ctypes.c_int, [P, P]),
            "wq_load": (P, [P, ctypes.c_char_p]),
            "wq_save": (ctypes.c_int, [P, P, ctypes.c_char_p]),
            "wq_state_free": (None, [P]),
            "wq_state_token_count": (I32, [P]),
            "wq_state_tokens": (tokens, [P]),
            "wq_state_bytes": (ctypes.c_uint64, [P]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(lib, name)
            function.restype = result
            function.argtypes = arguments
        _library_value = lib
        return lib


def _int32s(values: Sequence[int]):
    array = (ctypes.c_int32 * len(values))(*values)
    return array


class NativeState:
    """One prompt cache state of the engine (owned; freed with close or garbage collection)."""

    def __init__(self, handle: int):
        self._handle = handle
        lib = _library()
        count = lib.wq_state_token_count(handle)
        self.tokens = tuple(np.ctypeslib.as_array(lib.wq_state_tokens(handle), shape=(count,)).tolist()) if count else ()
        self.nbytes = int(lib.wq_state_bytes(handle))

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
            _library().wq_state_free(self._handle)
            self._handle = None

    def __del__(self):
        self.close()


class NativeEngine:
    """The engine's C interface. After any failure the engine is reset to no tokens."""

    def __init__(self, root: Path, *, slots: int, read_threads: int, mtp: bool, eight_bit: bool,
                 vision_catalog: Path | None):
        self._lib = _library()
        options = _Options(slots, read_threads, 1 if mtp else 0, 1 if eight_bit else 0,
                           str(vision_catalog).encode() if vision_catalog else None)
        error = ctypes.create_string_buffer(4096)
        handle = self._lib.wq_open(str(root).encode(), ctypes.byref(options), error, len(error))
        if not handle:
            raise RuntimeError(f"The native Qwen engine could not open the model: {error.value.decode(errors='replace')}")
        self._handle = handle
        self._cancel = _CANCEL(self._cancelled)  # kept alive while the engine uses it
        self._lib.wq_set_cancel(handle, self._cancel, None)
        self.vocab = int(self._lib.wq_vocab(handle))
        self.has_mtp = bool(self._lib.wq_has_mtp(handle))
        self.contract = self._lib.wq_contract(handle).decode()

    @staticmethod
    def _cancelled(_context) -> int:
        try:
            check_cancelled()
        except GenerationCancelled:
            return 1
        return 0

    def close(self) -> None:
        if self._handle is not None:
            self._lib.wq_set_cancel(self._handle, _CANCEL(), None)
            self._lib.wq_close(self._handle)
            self._handle = None

    def _check(self, status: int) -> None:
        if status == 0:
            return
        message = self._lib.wq_error(self._handle).decode(errors="replace")
        self._lib.wq_reset(self._handle)
        if status == 2:
            raise GenerationCancelled("Client disconnected")
        raise RuntimeError(f"Native Qwen engine: {message}")

    def _logits(self, pointer, rows: int) -> np.ndarray:
        return np.ctypeslib.as_array(pointer, shape=(rows, self.vocab)).copy()

    @property
    def tokens(self) -> tuple[int, ...]:
        count = ctypes.c_int32()
        pointer = self._lib.wq_tokens(self._handle, ctypes.byref(count))
        return tuple(np.ctypeslib.as_array(pointer, shape=(count.value,)).tolist()) if count.value else ()

    def counters(self) -> _Counters:
        counters = _Counters()
        self._lib.wq_counters_get(self._handle, ctypes.byref(counters))
        return counters

    def reset(self) -> None:
        self._check(self._lib.wq_reset(self._handle))

    def eval(self, tokens: Sequence[int], logit_rows: int = 1) -> np.ndarray:
        logits = ctypes.POINTER(ctypes.c_float)()
        self._check(self._lib.wq_eval(self._handle, _int32s(tokens), len(tokens), logit_rows, ctypes.byref(logits)))
        return self._logits(logits, logit_rows) if logit_rows else np.empty((0, self.vocab), np.float32)

    def draft(self, token: int) -> np.ndarray:
        logits = ctypes.POINTER(ctypes.c_float)()
        self._check(self._lib.wq_draft(self._handle, token, ctypes.byref(logits)))
        return self._logits(logits, 1)[0]

    def verify(self, tokens: Sequence[int]) -> np.ndarray:
        logits = ctypes.POINTER(ctypes.c_float)()
        self._check(self._lib.wq_verify(self._handle, _int32s(tokens), len(tokens), ctypes.byref(logits)))
        return self._logits(logits, len(tokens))

    def commit(self, keep: int) -> None:
        self._check(self._lib.wq_commit(self._handle, keep))

    def eval_images(self, tokens: Sequence[int], images: Sequence[Any]) -> np.ndarray:
        grids, patches = [], []
        for image in images:
            data = np.ascontiguousarray(image.patches, dtype=np.float32)
            grids += [int(image.grid[1]), int(image.grid[2])]
            patches.append(data)
        pointers = (ctypes.POINTER(ctypes.c_float) * len(patches))(
            *[p.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for p in patches])
        logits = ctypes.POINTER(ctypes.c_float)()
        self._check(self._lib.wq_eval_images(self._handle, _int32s(tokens), len(tokens), len(images),
                                             _int32s(grids), pointers, 1, ctypes.byref(logits)))
        return self._logits(logits, 1)

    def snapshot(self) -> NativeState:
        handle = self._lib.wq_snapshot(self._handle)
        if not handle:
            self._check(1)
        return NativeState(handle)

    def restore(self, state: NativeState) -> None:
        self._check(self._lib.wq_restore(self._handle, state.handle))

    def save(self, state: NativeState, path: Path) -> None:
        self._check(self._lib.wq_save(self._handle, state.handle, str(path).encode()))

    def load(self, path: Path) -> NativeState:
        handle = self._lib.wq_load(self._handle, str(path).encode())
        if not handle:
            self._check(1)
        return NativeState(handle)


def read_state_tokens(path: Path) -> tuple[int, ...] | None:
    """Tokens of a saved state, read from the file header; None for other files."""
    try:
        with path.open("rb") as file:
            header = file.read(32)
            if len(header) != 32 or header[:8] != b"WQSTATE1":
                return None
            contract_size, token_count, _ = np.frombuffer(header[8:], dtype="<u8")
            file.seek(32 + int(contract_size))
            data = file.read(int(token_count) * 4)
            if len(data) != int(token_count) * 4:
                return None
            return tuple(np.frombuffer(data, dtype="<i4").tolist())
    except OSError:
        return None


# ---------------------------------------------------------------------- sampling


@dataclass
class Distribution:
    """Normalized log probabilities of the sampling distribution over `indices`
    (every token when `indices` is None), or raw logits for greedy decoding."""
    logp: np.ndarray
    indices: np.ndarray | None = None
    greedy: bool = False

    def at(self, token: int) -> float:
        if self.indices is None:
            return float(self.logp[token])
        where = np.flatnonzero(self.indices == token)
        return float(self.logp[where[0]]) if where.size else -np.inf

    def argmax(self) -> int:
        best = int(np.argmax(self.logp))
        return best if self.indices is None else int(self.indices[best])

    def probabilities(self, vocab: int) -> np.ndarray:
        dense = np.zeros(vocab, np.float64)
        if self.indices is None:
            dense[:] = np.exp(self.logp)
        else:
            dense[self.indices] = np.exp(self.logp)
        return dense


class Sampler:
    """Token sampling with mlx_lm's semantics (make_sampler and its logits processors),
    as ModelRuntime uses them: presence and repetition penalties over the last 20
    tokens, then top-p, min-p and top-k on log probabilities, then temperature."""

    def __init__(self, options: GenerationOptions, rng: np.random.Generator):
        self.temperature = float(options.temperature)
        self.top_p = float(options.top_p)
        self.top_k = int(options.top_k)
        self.min_p = float(options.min_p)
        self.presence = float(options.presence_penalty)
        self.repetition = float(options.repetition_penalty)
        self.rng = rng

    def _penalized(self, logits: np.ndarray, history: Sequence[int]) -> np.ndarray:
        x = np.asarray(logits, dtype=np.float32)
        recent = list(history[-_PENALTY_CONTEXT:])
        if not recent or (self.repetition == 1 and self.presence == 0):
            return x
        x = x.copy()
        tokens = np.asarray(recent, dtype=np.int64)
        if self.repetition != 1:
            selected = x[tokens]
            x[tokens] = np.where(selected < 0, selected * self.repetition, selected / self.repetition)
        if self.presence != 0:
            x[tokens] -= self.presence
        return x

    def distribution(self, logits: np.ndarray, history: Sequence[int]) -> Distribution:
        x = self._penalized(logits, history)
        if self.temperature == 0:
            return Distribution(x, greedy=True)
        top = float(x.max())
        log_total = top + float(np.log(np.exp(x - top, dtype=np.float64).sum()))
        filtered = (0 < self.top_p < 1) or self.min_p != 0 or 0 < self.top_k < x.size
        if not filtered:
            logp = (x.astype(np.float64) - log_total) / max(self.temperature, 1e-5)
            return Distribution(logp - _logsumexp(logp))
        # Candidates: the top-k tokens, or enough of the best tokens to cover top-p.
        if 0 < self.top_k < x.size:
            count = self.top_k
        else:
            count = 64
            while count < x.size:
                best = np.argpartition(-x, count - 1)[:count]
                if np.exp(x[best].astype(np.float64) - log_total).sum() >= self.top_p:
                    break
                count *= 4
            count = min(count, x.size)
        candidates = np.argpartition(-x, count - 1)[:count] if count < x.size else np.arange(x.size)
        candidates = candidates[np.argsort(-x[candidates], kind="stable")]
        logp = x[candidates].astype(np.float64) - log_total
        keep = np.ones(candidates.size, dtype=bool)
        if 0 < self.top_p < 1:
            # mlx_lm keeps a token while the total of strictly more likely tokens is below top-p.
            greater = np.concatenate([[0.0], np.cumsum(np.exp(logp))[:-1]])
            keep &= greater < self.top_p
        if self.min_p != 0:
            keep &= logp >= logp[0] + np.log(self.min_p)
        keep[0] = True
        candidates, logp = candidates[keep], logp[keep]
        scaled = logp / max(self.temperature, 1e-5)
        return Distribution(scaled - _logsumexp(scaled), candidates)

    def sample(self, distribution: Distribution) -> int:
        if distribution.greedy:
            return int(np.argmax(distribution.logp))
        probabilities = np.exp(distribution.logp - distribution.logp.max())
        probabilities /= probabilities.sum()
        choice = int(self.rng.choice(probabilities.size, p=probabilities))
        return choice if distribution.indices is None else int(distribution.indices[choice])

    def verify(self, drafts: Sequence[int], draft_distributions: Sequence[Distribution],
               target_distributions: Sequence[Distribution], vocab: int) -> tuple[int, int]:
        """Accepted draft count and the next token, keeping the target distribution exact
        (dspark._verify: greedy matching, or rejection sampling with the residual)."""
        if self.temperature == 0:
            for index, token in enumerate(drafts):
                target = target_distributions[index].argmax()
                if token != target:
                    return index, target
            return len(drafts), target_distributions[len(drafts)].argmax()
        for index, token in enumerate(drafts):
            target, draft = target_distributions[index], draft_distributions[index]
            accept = min(1.0, float(np.exp(target.at(token) - draft.at(token))))
            if self.rng.uniform() > accept:
                residual = np.maximum(target.probabilities(vocab) - draft.probabilities(vocab), 0)
                total = residual.sum()
                if total > 0:
                    return index, int(self.rng.choice(vocab, p=residual / total))
                return index, self.sample(target)
        return len(drafts), self.sample(target_distributions[len(drafts)])


def _logsumexp(values: np.ndarray) -> float:
    top = float(values.max())
    return top + float(np.log(np.exp(values - top).sum()))


# ---------------------------------------------------------------------- runtime


class NativeExpertCounters:
    """The engine's expert cache totals, in the shape ModelRuntime's expert cache reports."""

    def __init__(self, engine: NativeEngine, blob_bytes: int):
        self._engine = engine
        self._blob = blob_bytes

    def metrics_snapshot(self) -> CacheMetrics:
        c = self._engine.counters()
        return CacheMetrics(hits=int(c.expert_hits), misses=int(c.expert_misses),
                            bytes_read=int(c.expert_misses) * self._blob, read_seconds=float(c.read_seconds),
                            prefill_seeded_experts=int(c.experts_seeded))

    @property
    def metrics(self) -> CacheMetrics:
        return self.metrics_snapshot()

    @property
    def resident_count(self) -> int:
        return int(self._engine.counters().resident)


@dataclass
class _CacheEntry:
    state: NativeState
    used: float


class NativeQwenRuntime(ChatInterface):
    """One loaded native engine; requests are serialized."""

    def __init__(self, installed: InstalledModel, config: RuntimeConfig, model_kind: str):
        self.installed = installed
        self.support = get_support(model_kind)
        self.support.validate_config(config)
        self._model_id = installed.model_id
        self.metrics = RuntimeMetrics()
        self._codec = None
        self._lock = threading.Lock()
        from .qwen_vision.artifact import catalog_path
        slots = 0 if config.native_automatic_expert_cache else self._configured_slots(installed, config)
        self.engine = NativeEngine(
            installed.root, slots=slots, read_threads=max(1, int(config.read_workers)),
            mtp=bool(config.mtp_enabled and installed.mtp is not None),
            eight_bit=bool(config.qwen_8bit_common_tensors), vision_catalog=catalog_path())
        try:
            self.expert_cache = NativeExpertCounters(self.engine, installed.expert_blob_size)
            # Report the cache size the engine chose; Throughput shows it as Slots.
            self.config = replace(config, slots=int(self.engine.counters().slots))
            self.tokenizer = AutoTokenizer.from_pretrained(installed.root / "tokenizer", trust_remote_code=True)
            self._entries: list[_CacheEntry] = []
            self._persist_later: list[NativeState] = []
            self._directory = self._open_prompt_cache_directory()
            self._vision_verified = False
        except BaseException:
            self.engine.close()
            raise

    @classmethod
    def open(cls, model_path: str, config: RuntimeConfig, model_kind: str) -> NativeQwenRuntime:
        return cls(InstalledModel.open(model_path), config, model_kind)

    @property
    def model_id(self) -> str:
        return self._model_id

    @staticmethod
    def _configured_slots(installed: InstalledModel, config: RuntimeConfig) -> int:
        if config.expert_cache_bytes:
            return max(installed.selected_expert_count, int(config.expert_cache_bytes) // installed.expert_blob_size)
        return int(config.slots)

    def close(self) -> None:
        with self._lock:
            for entry in self._entries:
                entry.state.close()
            self._entries.clear()
            self.engine.close()

    # ------------------------------------------------------------------ requests

    def stream(self, prompt: str | list[int] | MediaRequest, options: GenerationOptions) -> Iterator[GeneratedPiece]:
        check_cancelled()
        with self._lock:
            sampler = Sampler(options, np.random.default_rng(
                options.seed if options.seed is not None else secrets.randbits(32)))
            images = None
            if isinstance(prompt, MediaRequest):
                from .qwen_vision.artifact import weight_path
                from .qwen_vision.inputs import expand_prompt
                if self._codec is None:
                    self._codec = self.support.open_codec(self.installed.root, self.tokenizer)
                weight_path(self.installed, verify=not self._vision_verified)
                self._vision_verified = True
                tokens, _, images = expand_prompt(self, prompt)
            else:
                tokens = list(prompt) if isinstance(prompt, list) else self._encode_prompt(prompt)
            if not tokens:
                raise ValueError("the prompt has no tokens")
            options = fit_context_window(
                len(tokens), options, getattr(self.installed, "maximum_context", 262_144))
            reused = 0 if images is not None else self._continue_from_cache(tokens)
            if images is not None:
                self.engine.reset()
            mtp = self.engine.has_mtp and bool(self.config.mtp_enabled)
            new_tokens = len(tokens) - reused
            self.metrics.start(len(tokens), reused, min(new_tokens, 1024), new_tokens > 512,
                               self.expert_cache.metrics_snapshot(), mtp_enabled=mtp)
            completed = False
            self._persist_later = []
            try:
                yield from self._pieces(tokens, reused, images, options, sampler, mtp)
                completed = True
            finally:
                self.metrics.finish(self.expert_cache.metrics_snapshot())
                if images is not None:
                    # Image prompts are not cached; text requests start without them.
                    self.engine.reset()
                elif completed and self._cache_enabled():
                    self._store(self.engine.snapshot(), persist=True)
                    if self._directory is not None:
                        for state in self._persist_later:
                            if not state.closed:
                                self._persist(state)
                self._persist_later = []

    def _pieces(self, prompt_tokens, reused, images, options, sampler, mtp) -> Iterator[GeneratedPiece]:
        # Text and finish handling as ModelRuntime._stream_mtp.
        tokenizer = TokenizerWrapper(self.tokenizer)
        detokenizer = tokenizer.detokenizer
        tokens = self._tokens(prompt_tokens, reused, images, options, sampler, mtp)
        generation_tokens = 0
        last_token = 0
        pending_text = ""
        finish_reason = "length"
        while generation_tokens < options.max_tokens:
            check_cancelled()
            started = time.perf_counter()
            try:
                token = next(tokens)
            except StopIteration:
                break
            check_cancelled()
            step_seconds = time.perf_counter() - started
            generation_tokens += 1
            last_token = token
            if token in tokenizer.eos_token_ids:
                finish_reason = "stop"
                break
            detokenizer.add_token(token)
            self.metrics.record_token(generation_tokens, step_seconds, 0.0)
            if generation_tokens == options.max_tokens:
                pending_text = detokenizer.last_segment
                break
            yield GeneratedPiece(text=detokenizer.last_segment, token=token, prompt_tokens=len(prompt_tokens),
                                 generation_tokens=generation_tokens, finish_reason=None)
        tokens.close()
        detokenizer.finalize()
        final_text = detokenizer.last_segment
        if pending_text and not final_text.startswith(pending_text):
            final_text = pending_text + final_text
        self.metrics.record_token(generation_tokens, 0.0, 0.0)
        yield GeneratedPiece(text=final_text, token=last_token, prompt_tokens=len(prompt_tokens),
                             generation_tokens=generation_tokens, finish_reason=finish_reason)

    def _tokens(self, prompt_tokens, reused, images, options, sampler, mtp) -> Iterator[int]:
        """Sampled tokens, each yielded before the model consumes it (as generate_mtp_tokens)."""
        engine = self.engine
        history = [prompt_tokens[-1]]
        if images is not None:
            logits = engine.eval_images(prompt_tokens, images)[0]
        else:
            logits = self._prefill(prompt_tokens[reused:])
        anchor = sampler.sample(sampler.distribution(logits, history))
        yield anchor
        history.append(anchor)
        generated = 1
        policy = MTPDraftPolicy(int(self.config.qwen_mtp_draft_tokens), int(self.config.qwen_mtp_zero_acceptance_limit))
        fallback = not mtp
        while generated < options.max_tokens:
            check_cancelled()
            remaining = options.max_tokens - generated
            if fallback or remaining == 1:
                anchor = sampler.sample(sampler.distribution(engine.eval([anchor])[0], history))
                yield anchor
                history.append(anchor)
                generated += 1
                continue
            draft_started = time.perf_counter()
            drafts: list[int] = []
            draft_distributions: list[Distribution] = []
            token = anchor
            for _ in range(policy.limit(remaining)):
                check_cancelled()
                distribution = sampler.distribution(engine.draft(token), history + drafts)
                token = sampler.sample(distribution)
                drafts.append(token)
                draft_distributions.append(distribution)
            verify_started = time.perf_counter()
            rows = engine.verify([anchor, *drafts])
            targets = [sampler.distribution(rows[i], history + drafts[:i]) for i in range(len(drafts) + 1)]
            accepted, next_token = sampler.verify(drafts, draft_distributions, targets, engine.vocab)
            engine.commit(accepted + 1)
            fallback = policy.observe(accepted)
            self.metrics.record_mtp_round(len(drafts), accepted, verify_started - draft_started,
                                          time.perf_counter() - verify_started, 0.0, fallback)
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

    def _prefill(self, new_tokens: list[int]) -> np.ndarray:
        """Consume the new prompt tokens and return the last token's logits. A long new
        part also leaves a prompt cache entry for everything before the last token, so
        the same prompt can be answered again (one extra one-token pass)."""
        snapshot = self._cache_enabled() and len(new_tokens) >= PROMPT_SNAPSHOT_MIN_TOKENS
        head = new_tokens[:-1] if snapshot else new_tokens
        logits = None
        for start in range(0, len(head), PREFILL_SEGMENT):
            segment = head[start:start + PREFILL_SEGMENT]
            last = not snapshot and start + PREFILL_SEGMENT >= len(head)
            logits = self.engine.eval(segment, 1 if last else 0)
        if snapshot:
            entry = self._store(self.engine.snapshot(), persist=False)
            if entry is not None:
                self._persist_later.append(entry.state)
            logits = self.engine.eval(new_tokens[-1:])
        return logits[0]

    def warm_prompt(self, prompt: str) -> int:
        tokens = self._encode_prompt(prompt)
        if len(tokens) < 2 or not self._cache_enabled():
            return 0
        with self._lock:
            self.engine.reset()
            for start in range(0, len(tokens) - 1, PREFILL_SEGMENT):
                self.engine.eval(tokens[start:min(start + PREFILL_SEGMENT, len(tokens) - 1)], 0)
            self._store(self.engine.snapshot(), persist=True)
        return len(tokens) - 1

    # ------------------------------------------------------------------ prompt cache

    def _cache_enabled(self) -> bool:
        return int(getattr(self.config, "prompt_cache_entries", 0)) > 0

    def _continue_from_cache(self, tokens: list[int]) -> int:
        """Set the engine to the longest cached prefix of the prompt (leaving at least
        one token to read) and return its length."""
        if not self._cache_enabled():
            self.engine.reset()
            return 0
        limit = len(tokens) - 1
        current = self.engine.tokens
        best_length = len(current) if 0 < len(current) <= limit and tuple(tokens[:len(current)]) == current else 0
        best: _CacheEntry | None = None
        for entry in self._entries:
            length = len(entry.state.tokens)
            if best_length < length <= limit and tuple(tokens[:length]) == entry.state.tokens:
                best, best_length = entry, length
        disk = self._best_disk_prefix(tokens, best_length)
        if disk is not None:
            started = time.perf_counter()
            try:
                state = self.engine.load(disk)
            except (RuntimeError, OSError):
                state = None
            self.metrics.record_prompt_cache_snapshot(time.perf_counter() - started)
            if state is not None:
                best = self._store(state, persist=False)
                best_length = len(state.tokens)
        if best is not None:
            best.used = time.monotonic()
            self.engine.restore(best.state)
        elif best_length == 0:
            self.engine.reset()
        return best_length

    def _store(self, state: NativeState, *, persist: bool) -> _CacheEntry | None:
        if not state.tokens:
            state.close()
            return None
        for entry in self._entries:
            if entry.state.tokens == state.tokens:
                state.close()
                entry.used = time.monotonic()
                return entry
        entry = _CacheEntry(state, time.monotonic())
        self._entries.append(entry)
        # Each request may keep two states: before its last prompt token and at its end.
        limit = 2 * int(self.config.prompt_cache_entries)
        budget = int(self.config.prompt_cache_memory_gib) * 1024**3
        self._entries.sort(key=lambda item: item.used)
        while len(self._entries) > limit or (len(self._entries) > 1 and sum(e.state.nbytes for e in self._entries) > budget):
            self._entries.pop(0).state.close()
        if persist and self._directory is not None and entry in self._entries:
            self._persist(entry.state)
        return entry if entry in self._entries else None

    def _open_prompt_cache_directory(self) -> Path | None:
        if not self._cache_enabled() or not getattr(self.config, "persistent_prompt_cache", False):
            return None
        configured = getattr(self.config, "prompt_cache_directory", None)
        root = Path(configured).expanduser() if configured else Path.home() / ".dsmodel" / "prompt-cache"
        contract = hashlib.sha256(self.engine.contract.encode()).hexdigest()[:16]
        directory = (root / self.installed.model_id.replace("/", "--") / self.installed.revision
                     / f"native-{contract}")
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return directory

    def _persist(self, state: NativeState) -> None:
        name = hashlib.sha256(np.asarray(state.tokens, dtype="<i4").tobytes()).hexdigest() + ".wqstate"
        path = self._directory / name
        started = time.perf_counter()
        try:
            if not path.exists():
                self.engine.save(state, path)
            path.touch()
        except (RuntimeError, OSError):
            return
        self.metrics.record_prompt_cache_write(time.perf_counter() - started)
        files = sorted(self._directory.glob("*.wqstate"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[max(1, int(self.config.persistent_prompt_cache_entries)):]:
            old.unlink(missing_ok=True)

    def _best_disk_prefix(self, tokens: list[int], longer_than: int) -> Path | None:
        if self._directory is None:
            return None
        best, best_length = None, longer_than
        for path in self._directory.glob("*.wqstate"):
            saved = read_state_tokens(path)
            if saved and best_length < len(saved) <= len(tokens) - 1 and tuple(tokens[:len(saved)]) == saved:
                best, best_length = path, len(saved)
        return best
