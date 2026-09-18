from __future__ import annotations

import copy
import hashlib
import importlib
import itertools
import json
import os
import secrets
import threading
import time
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import CacheList, make_prompt_cache
from mlx_lm.models.base import create_ssm_mask
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper
from transformers import AutoTokenizer

from .cancellation import check_cancelled

from .dspark import DraftResult, VerificationMetrics, generate_tokens
from .expert_cache import CacheMetrics
from .fp8_cache import MXFP8PoolingCache
from .io_metrics import ProcessDiskIO, process_disk_io_snapshot
from .manifest import InstalledModel
from .model_support import get_support, support_for_installed, support_for_runtime
from .model_support.state import (
    _RawEvalCacheList,
    _make_prompt_cache, _clone_cache_state, _cache_state_arrays, _cache_state_nbytes,
    _encode_cache_state, _decode_cache_state, _persistence_item, _persistence_cache_state,
    _restore_persistence_item, _restore_persistence_cache,
)
# Compatibility imports for existing research scripts; runtime uses its package.
from .model_support.qwen import _qwen_layer_major_prefill
from .model_support.deepseek_v41 import _deepseek_v41_prefill
from .model import (
    RuntimeConfig,
    _cache_arrays,
    _select_prefill_step_size,
    eval_prompt_cache,
    layer_major_prefill,
    load_model,
)
from .tool_codec import (
    AssistantTurn,
    DeepSeekV41ToolStreamParser,
    QwenToolStreamParser,
    ToolChoice,
    ToolCodec,
    ToolStreamParser,
)

_mlx_lm_generate = importlib.import_module("mlx_lm.generate")
_MLX_LM_GENERATION_LOCK = threading.Lock()

THINK_START = "<think>"
THINK_END = "</think>"
EXACT_APPROXIMATION_MODE = "exact"
LEARNED_ROUTE_DROP_LOWEST_1 = "learned-route-drop-lowest-1"
APPROXIMATION_MODES = frozenset(
    (EXACT_APPROXIMATION_MODE, LEARNED_ROUTE_DROP_LOWEST_1)
)


def _uses_layer_major_prefill(
    config: Any,
    *,
    is_qwen: bool,
    is_deepseek_v41: bool = False,
    token_count: int,
) -> bool:
    kind = "deepseek-v4.1" if is_deepseek_v41 else (
        "qwen3.8-flash-next" if is_qwen else "deepseek-v4"
    )
    return get_support(kind).uses_layer_major_prefill(config, token_count)


@contextmanager
def _use_mlx_lm_generation_stream(stream):
    with _MLX_LM_GENERATION_LOCK:
        previous = _mlx_lm_generate.generation_stream
        _mlx_lm_generate.generation_stream = stream
        try:
            yield
        finally:
            _mlx_lm_generate.generation_stream = previous


def _route_phase(expert_cache, phase: str):
    memory_phase = getattr(expert_cache, "set_memory_phase", None)
    if callable(memory_phase):
        memory_phase(phase)
    trace_routes = getattr(expert_cache, "trace_routes", None)
    return trace_routes(phase) if callable(trace_routes) else nullcontext()


@contextmanager
def _approximation_mode(runtime: Any, mode: str):
    if mode not in APPROXIMATION_MODES:
        raise ValueError(f"unsupported approximation mode: {mode}")
    if mode == EXACT_APPROXIMATION_MODE:
        yield
        return
    with support_for_runtime(runtime).approximation(runtime.model, mode):
        yield


@dataclass(frozen=True)
class GenerationOptions:
    max_tokens: int = 272_000
    temperature: float = 0.2
    top_p: float = 0.98
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    approximation_mode: str = EXACT_APPROXIMATION_MODE
    seed: int | None = None


@dataclass(frozen=True)
class GeneratedPiece:
    text: str
    token: int
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None


@dataclass
class _PromptCacheEntry:
    cache: Any
    tokens: list[int]
    approximation_mode: str = EXACT_APPROXIMATION_MODE
    # Entries stored by one request share a number. The entry limit counts
    # requests, so a prompt-end snapshot survives next to its final state.
    request: int | None = None


@dataclass(frozen=True)
class _PersistentPromptCacheEntry:
    tokens: list[int]
    path: Path
    format: int
    cache_key: str = ""
    metadata_path: Path | None = None
    access_path: Path | None = None
    reuse_count: int = 0
    last_access_ns: int = 0


@dataclass(frozen=True)
class _PromptCacheSnapshot:
    state: Any
    tokens: list[int]


@dataclass
class _DSparkPromptCacheEntry:
    cache: Any
    context_state: tuple[Any, ...]
    tokens: list[int]
    revision: str
    target_layers: tuple[int, ...]


def _dspark_context_count(dspark) -> int:
    # V4 DSpark keeps one draft context per layer; V4.1 keeps one per MTP stage.
    stages = getattr(dspark, "layers", None) or getattr(dspark, "mtp", None)
    return len(stages or ())


@dataclass(frozen=True)
class _PersistentDSparkPromptCacheEntry:
    tokens: list[int]
    path: Path
    format: int
    revision: str
    target_layers: tuple[int, ...]


_PROMPT_CACHE_REQUESTS = itertools.count(1)
_PROMPT_CACHE_FORMAT = 5
_SUPPORTED_PROMPT_CACHE_FORMATS = frozenset((_PROMPT_CACHE_FORMAT,))
_PROMPT_CACHE_BLOCK_SIZE = 128
_PROMPT_CACHE_CONTRACT_FORMAT = 1
_DSPARK_PROMPT_CACHE_FORMAT = 3


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_prompt_cache_file(path: Path):
    try:
        return mx.load(path, return_metadata=True)
    except RuntimeError as error:
        # MLX reports malformed safetensors as RuntimeError. Only translate
        # file-format errors; allocation and GPU execution failures must escape.
        if not str(error).startswith(("[load_safetensors]", "[read] Unable to read ")):
            raise
        raise ValueError(f"Invalid prompt cache file: {path}") from error


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _prompt_cache_contract(installed: InstalledModel, config: RuntimeConfig) -> dict:
    """Return every model/runtime input that can change target KV semantics."""
    raw_config: dict[str, Any] = {}
    config_path = Path(getattr(installed, "root", "")) / "config.json"
    try:
        loaded = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw_config = loaded
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass

    rope_keys = (
        "rope_theta",
        "rope_scaling",
        "qk_rope_head_dim",
        "compress_rope_theta",
        "max_position_embeddings",
    )
    attention_keys = (
        "attention_bias",
        "attention_dropout",
        "num_attention_heads",
        "sliding_window",
        "compress_ratios",
        "use_cache",
    )
    fp8 = bool(getattr(config, "fp8_kv_cache", True))
    fp4_index = bool(getattr(config, "fp4_index_cache", True))
    return {
        "format": _PROMPT_CACHE_CONTRACT_FORMAT,
        "revision": str(getattr(installed, "revision", "")),
        "modelID": str(getattr(installed, "model_id", "")),
        **({"v41PackedCache": [getattr(config, "v41_packed_kv", False), getattr(config, "v41_packed_index", False)]}
           if getattr(config, "v41_packed_kv", False) or getattr(config, "v41_packed_index", False) else {}),
        **({"qwenQuantizedCache": [getattr(config, "qwen_quantized_kv", False), getattr(config, "qwen_quantized_index", False)]}
           if getattr(config, "qwen_quantized_kv", False) or getattr(config, "qwen_quantized_index", False) else {}),
        **({"deepseekANE": config.ane_prefill_ratio} if getattr(config, "deepseek_ane_prefill", False) else {}),
        # Keep legacy cache contracts unchanged when the experiment is off.
        "modelConfigSHA256": _sha256_json(raw_config),
        "rope": {key: raw_config.get(key) for key in rope_keys},
        "kvFormat": {
            "compressedAttention": "mxfp8" if fp8 else "bfloat16",
            "index": "mxfp4" if fp8 and fp4_index else "same-as-kv",
            "stateSchema": "target-only-v2",
        },
        "attentionMode": {
            "implementation": "sparse-pooled-correct-overlap-v1",
            **{key: raw_config.get(key) for key in attention_keys},
        },
        "tokenBlockSize": _PROMPT_CACHE_BLOCK_SIZE,
    }


def _prompt_cache_block_identity(
    contract: dict[str, Any],
    tokens: list[int],
) -> tuple[str, list[dict[str, Any]], str]:
    """Build a content-addressed chain over fixed-size token blocks."""
    contract_sha256 = _sha256_json(contract)
    parent = contract_sha256
    blocks: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(tokens), _PROMPT_CACHE_BLOCK_SIZE)):
        token_block = [int(token) for token in tokens[start : start + _PROMPT_CACHE_BLOCK_SIZE]]
        token_sha256 = _sha256_json(token_block)
        block_key = _sha256_json(
            {
                "contractSHA256": contract_sha256,
                "index": index,
                "parent": parent,
                "tokens": token_block,
            }
        )
        blocks.append(
            {
                "index": index,
                "start": start,
                "length": len(token_block),
                "tokenSHA256": token_sha256,
                "parent": parent,
                "key": block_key,
            }
        )
        parent = block_key
    return contract_sha256, blocks, parent


class RuntimeMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._time_to_first_token_seconds = 0.0
        self._decode_seconds = 0.0
        self._decode_cache_eval_seconds = 0.0
        self._cache_state_eval_seconds = 0.0
        self._cache_state_eval_count = 0
        self._prompt_cache_snapshot_seconds = 0.0
        self._prompt_cache_serialize_seconds = 0.0
        self._prompt_cache_write_seconds = 0.0
        self._prompt_cache_write_errors = 0
        self._prompt_cache_write_error = ""
        self._decode_latency_seconds: list[float] = []
        self._prompt_tokens = 0
        self._generation_tokens = 0
        self._accumulated_generation_tokens = 0
        self._prompt_cache_reused_tokens = 0
        self._dspark_prompt_cache_source = "disabled"
        self._approximation_mode = EXACT_APPROXIMATION_MODE
        self._prefill_step_size = 0
        self._layer_major_prefill = False
        self._request_started = 0.0
        self._request_seconds = 0.0
        self._request_active = False
        self._completed_request_count = 0
        self._expert_before = CacheMetrics()
        self._expert_request = CacheMetrics()
        self._process_disk_io_before: ProcessDiskIO | None = None
        self._request_process_disk_io: ProcessDiskIO | None = None
        self._mtp_enabled = False
        self._mtp_rounds = 0
        self._mtp_proposed_tokens = 0
        self._mtp_accepted_tokens = 0
        self._mtp_committed_tokens = 0
        self._mtp_draft_seconds = 0.0
        self._mtp_verification_seconds = 0.0
        self._mtp_replay_seconds = 0.0
        self._mtp_fallback = False
        self._mtp_fallback_rounds = 0
        self._mtp_cache = None
        self._mtp_expert_before = CacheMetrics()
        self._mtp_expert_request = CacheMetrics()
        self._dspark_enabled = False
        self._dspark_rounds = 0
        self._dspark_proposed_tokens = 0
        self._dspark_accepted_tokens = 0
        self._dspark_committed_tokens = 0
        self._dspark_output_budget_trimmed_tokens = 0
        self._dspark_draft_seconds = 0.0
        self._dspark_verification_seconds = 0.0
        self._dspark_cache_fork_seconds = 0.0
        self._dspark_cache_replay_seconds = 0.0
        self._dspark_cache_eval_count = 0
        self._dspark_cache_eval_bytes = 0
        self._dspark_cache_fork_layers = 0
        self._dspark_per_position_cache_copies = 0
        self._dspark_state_fetch_count = 0
        self._dspark_block_attention_layers = 0
        self._dspark_block_verification_rounds = 0
        self._dspark_sequential_verification_rounds = 0
        self._dspark_last_verification_mode = ""
        self._dspark_last_sequential_position_seconds: tuple[float, ...] = ()
        self._dspark_last_layer_seconds: tuple[float, ...] = ()
        self._dspark_confidence_sum = 0.0
        self._dspark_confidence_count = 0
        self._dspark_last_proposed_tokens = 0
        self._dspark_last_accepted_tokens = 0
        self._dspark_last_draft_seconds = 0.0
        self._dspark_last_verification_seconds = 0.0
        self._dspark_fallback_cost_ratios: list[float] = []
        self._dspark_fallback_would_trigger_rounds = 0
        self._dspark_fallback_triggered_rounds = 0
        self._dspark_last_fallback_target_step_seconds = 0.0
        self._dspark_last_fallback_speculative_seconds = 0.0
        self._dspark_last_fallback_break_even_seconds = 0.0
        self._dspark_last_fallback_cost_ratio = 0.0
        self._dspark_round_trace: list[dict[str, object]] = []
        self._dspark_verification_expert_union_calls = 0
        self._dspark_verification_routed_expert_assignments = 0
        self._dspark_verification_expert_union_experts = 0
        self._dspark_verification_expert_union_misses = 0
        self._dspark_target_expert_bytes_read = 0
        self._dspark_verification_expert_bytes_read = 0
        self._dspark_replay_expert_bytes_read = 0
        self._dspark_verification_expert_read_seconds = 0.0
        self._dspark_last_expert_union_layer_ids: tuple[int, ...] = ()
        self._dspark_last_layer_routed_expert_assignments: tuple[int, ...] = ()
        self._dspark_last_layer_expert_union_counts: tuple[int, ...] = ()
        self._dspark_last_layer_expert_union_misses: tuple[int, ...] = ()
        self._dspark_last_verification_expert_bytes_read = 0
        self._dspark_fallback = False
        self._dspark_cache = None
        self._dspark_expert_before = CacheMetrics()
        self._dspark_expert_request = CacheMetrics()

    def start(
        self,
        prompt_tokens: int,
        reused_tokens: int,
        prefill_step_size: int,
        layer_major_prefill_enabled: bool,
        expert_before: CacheMetrics,
        mtp_enabled: bool = False,
        mtp_cache=None,
        dspark_enabled: bool = False,
        dspark_cache=None,
        dspark_prompt_cache_source: str = "disabled",
        approximation_mode: str = EXACT_APPROXIMATION_MODE,
    ) -> None:
        with self._lock:
            self._time_to_first_token_seconds = 0.0
            self._decode_seconds = 0.0
            self._decode_cache_eval_seconds = 0.0
            self._cache_state_eval_seconds = 0.0
            self._cache_state_eval_count = 0
            self._prompt_cache_snapshot_seconds = 0.0
            self._prompt_cache_serialize_seconds = 0.0
            self._prompt_cache_write_seconds = 0.0
            self._prompt_cache_write_errors = 0
            self._prompt_cache_write_error = ""
            self._decode_latency_seconds = []
            self._prompt_tokens = prompt_tokens
            self._generation_tokens = 0
            self._prompt_cache_reused_tokens = reused_tokens
            self._dspark_prompt_cache_source = dspark_prompt_cache_source
            self._approximation_mode = approximation_mode
            self._prefill_step_size = prefill_step_size
            self._layer_major_prefill = layer_major_prefill_enabled
            self._process_disk_io_before = process_disk_io_snapshot()
            self._request_process_disk_io = None
            self._request_started = time.perf_counter()
            self._request_seconds = 0.0
            self._request_active = True
            self._expert_before = expert_before
            self._expert_request = CacheMetrics()
            self._mtp_enabled = mtp_enabled
            self._mtp_rounds = 0
            self._mtp_proposed_tokens = 0
            self._mtp_accepted_tokens = 0
            self._mtp_committed_tokens = 0
            self._mtp_draft_seconds = 0.0
            self._mtp_verification_seconds = 0.0
            self._mtp_replay_seconds = 0.0
            self._mtp_fallback = False
            self._mtp_fallback_rounds = 0
            self._mtp_cache = mtp_cache
            self._mtp_expert_before = (
                mtp_cache.metrics_snapshot()
                if mtp_cache is not None
                else CacheMetrics()
            )
            self._mtp_expert_request = CacheMetrics()
            self._dspark_enabled = dspark_enabled
            self._dspark_rounds = 0
            self._dspark_proposed_tokens = 0
            self._dspark_accepted_tokens = 0
            self._dspark_committed_tokens = 0
            self._dspark_output_budget_trimmed_tokens = 0
            self._dspark_draft_seconds = 0.0
            self._dspark_verification_seconds = 0.0
            self._dspark_cache_fork_seconds = 0.0
            self._dspark_cache_replay_seconds = 0.0
            self._dspark_cache_eval_count = 0
            self._dspark_cache_eval_bytes = 0
            self._dspark_cache_fork_layers = 0
            self._dspark_per_position_cache_copies = 0
            self._dspark_state_fetch_count = 0
            self._dspark_block_attention_layers = 0
            self._dspark_block_verification_rounds = 0
            self._dspark_sequential_verification_rounds = 0
            self._dspark_last_verification_mode = ""
            self._dspark_last_sequential_position_seconds = ()
            self._dspark_last_layer_seconds = ()
            self._dspark_confidence_sum = 0.0
            self._dspark_confidence_count = 0
            self._dspark_last_proposed_tokens = 0
            self._dspark_last_accepted_tokens = 0
            self._dspark_last_draft_seconds = 0.0
            self._dspark_last_verification_seconds = 0.0
            self._dspark_fallback_cost_ratios = []
            self._dspark_fallback_would_trigger_rounds = 0
            self._dspark_fallback_triggered_rounds = 0
            self._dspark_last_fallback_target_step_seconds = 0.0
            self._dspark_last_fallback_speculative_seconds = 0.0
            self._dspark_last_fallback_break_even_seconds = 0.0
            self._dspark_last_fallback_cost_ratio = 0.0
            self._dspark_round_trace = []
            self._dspark_verification_expert_union_calls = 0
            self._dspark_verification_routed_expert_assignments = 0
            self._dspark_verification_expert_union_experts = 0
            self._dspark_verification_expert_union_misses = 0
            self._dspark_target_expert_bytes_read = 0
            self._dspark_verification_expert_bytes_read = 0
            self._dspark_replay_expert_bytes_read = 0
            self._dspark_verification_expert_read_seconds = 0.0
            self._dspark_last_expert_union_layer_ids = ()
            self._dspark_last_layer_routed_expert_assignments = ()
            self._dspark_last_layer_expert_union_counts = ()
            self._dspark_last_layer_expert_union_misses = ()
            self._dspark_last_verification_expert_bytes_read = 0
            self._dspark_fallback = False
            self._dspark_cache = dspark_cache
            self._dspark_expert_before = (
                dspark_cache.metrics_snapshot()
                if dspark_cache is not None
                else CacheMetrics()
            )
            self._dspark_expert_request = CacheMetrics()

    def record(
        self,
        response,
        step_seconds: float,
        cache_state_eval_seconds: float,
    ) -> None:
        self.record_token(
            response.generation_tokens,
            step_seconds,
            cache_state_eval_seconds,
        )

    def record_token(
        self,
        generation_tokens: int,
        step_seconds: float,
        cache_state_eval_seconds: float,
    ) -> None:
        with self._lock:
            if generation_tokens == 1:
                self._time_to_first_token_seconds = (
                    time.perf_counter() - self._request_started
                )
            else:
                self._decode_seconds += step_seconds
                self._decode_cache_eval_seconds += cache_state_eval_seconds
                self._decode_latency_seconds.append(
                    step_seconds + cache_state_eval_seconds
                )
            self._cache_state_eval_seconds += cache_state_eval_seconds
            self._cache_state_eval_count += 1
            self._generation_tokens = generation_tokens

    def record_prompt_cache_snapshot(self, seconds: float) -> None:
        with self._lock:
            self._prompt_cache_snapshot_seconds += seconds

    def record_prompt_cache_serialize(self, seconds: float) -> None:
        with self._lock:
            self._prompt_cache_serialize_seconds += seconds

    def record_prompt_cache_write(self, seconds: float) -> None:
        with self._lock:
            self._prompt_cache_write_seconds += seconds

    def record_prompt_cache_write_error(self, error: Exception) -> None:
        with self._lock:
            self._prompt_cache_write_errors += 1
            self._prompt_cache_write_error = f"{type(error).__name__}: {error}"

    def record_dspark_round(
        self,
        draft: DraftResult,
        accepted: int,
        verification_seconds: float,
        verification: VerificationMetrics,
    ) -> None:
        with self._lock:
            self._dspark_rounds += 1
            self._dspark_proposed_tokens += len(draft.tokens)
            self._dspark_accepted_tokens += accepted
            self._dspark_committed_tokens += (
                accepted + 1
                if verification.committed_tokens is None
                else verification.committed_tokens
            )
            self._dspark_output_budget_trimmed_tokens += (
                verification.output_budget_trimmed_tokens
            )
            self._dspark_draft_seconds += draft.seconds
            self._dspark_verification_seconds += verification_seconds
            self._dspark_cache_fork_seconds += verification.cache_fork_seconds
            self._dspark_cache_replay_seconds += verification.cache_replay_seconds
            self._dspark_cache_eval_count += verification.cache_eval_count
            self._dspark_cache_eval_bytes += verification.cache_eval_bytes
            self._dspark_cache_fork_layers += verification.cache_fork_layers
            self._dspark_per_position_cache_copies += (
                verification.per_position_cache_copies
            )
            self._dspark_state_fetch_count += verification.state_fetch_count
            self._dspark_block_attention_layers += (
                verification.block_attention_layers
            )
            if verification.verification_mode == "block":
                self._dspark_block_verification_rounds += 1
            elif verification.verification_mode == "sequential":
                self._dspark_sequential_verification_rounds += 1
            self._dspark_last_verification_mode = verification.verification_mode
            self._dspark_last_sequential_position_seconds = (
                verification.sequential_position_seconds
            )
            self._dspark_last_layer_seconds = verification.layer_seconds
            self._dspark_confidence_sum += sum(draft.confidence)
            self._dspark_confidence_count += len(draft.confidence)
            self._dspark_last_proposed_tokens = len(draft.tokens)
            self._dspark_last_accepted_tokens = accepted
            self._dspark_last_draft_seconds = draft.seconds
            self._dspark_last_verification_seconds = verification_seconds
            self._dspark_fallback_cost_ratios.append(
                verification.fallback_cost_ratio
            )
            if verification.fallback_would_trigger:
                self._dspark_fallback_would_trigger_rounds += 1
            if verification.fallback_triggered:
                self._dspark_fallback_triggered_rounds += 1
            self._dspark_last_fallback_target_step_seconds = (
                verification.fallback_target_step_seconds
            )
            self._dspark_last_fallback_speculative_seconds = (
                verification.fallback_speculative_seconds
            )
            self._dspark_last_fallback_break_even_seconds = (
                verification.fallback_break_even_seconds
            )
            self._dspark_last_fallback_cost_ratio = (
                verification.fallback_cost_ratio
            )
            self._dspark_round_trace.append(
                {
                    "proposed_tokens": len(draft.tokens),
                    "accepted_tokens": accepted,
                    "committed_tokens": (
                        accepted + 1
                        if verification.committed_tokens is None
                        else verification.committed_tokens
                    ),
                    "verification_mode": verification.verification_mode,
                    "sequential_verification_positions": (
                        verification.sequential_verification_positions
                    ),
                    "fallback_cost_ratio": verification.fallback_cost_ratio,
                    "fallback_would_trigger": (
                        verification.fallback_would_trigger
                    ),
                    "fallback_triggered": verification.fallback_triggered,
                }
            )
            self._dspark_verification_routed_expert_assignments += (
                verification.routed_expert_assignments
            )
            self._dspark_verification_expert_union_calls += len(
                verification.layer_expert_union_layer_ids
            )
            self._dspark_verification_expert_union_experts += (
                verification.expert_union_experts
            )
            self._dspark_verification_expert_union_misses += (
                verification.expert_union_misses
            )
            self._dspark_target_expert_bytes_read += (
                verification.expert_bytes_read
            )
            self._dspark_verification_expert_bytes_read += (
                verification.verification_expert_bytes_read
            )
            self._dspark_replay_expert_bytes_read += (
                verification.replay_expert_bytes_read
            )
            self._dspark_verification_expert_read_seconds += (
                verification.expert_read_seconds
            )
            self._dspark_last_expert_union_layer_ids = (
                verification.layer_expert_union_layer_ids
            )
            self._dspark_last_layer_routed_expert_assignments = (
                verification.layer_routed_expert_assignments
            )
            self._dspark_last_layer_expert_union_counts = (
                verification.layer_expert_union_counts
            )
            self._dspark_last_layer_expert_union_misses = (
                verification.layer_expert_union_misses
            )
            self._dspark_last_verification_expert_bytes_read = (
                verification.expert_bytes_read
            )

    def record_dspark_fallback(self) -> None:
        with self._lock:
            self._dspark_fallback = True

    def record_mtp_round(
        self,
        proposed: int,
        accepted: int,
        draft_seconds: float,
        verification_seconds: float,
        replay_seconds: float,
        fallback: bool,
    ) -> None:
        with self._lock:
            self._mtp_rounds += 1
            self._mtp_proposed_tokens += proposed
            self._mtp_accepted_tokens += accepted
            self._mtp_committed_tokens += accepted + 1
            self._mtp_draft_seconds += draft_seconds
            self._mtp_verification_seconds += verification_seconds
            self._mtp_replay_seconds += replay_seconds
            if fallback:
                self._mtp_fallback = True
                self._mtp_fallback_rounds += 1

    def finish(self, expert_after: CacheMetrics) -> None:
        with self._lock:
            self._request_seconds = time.perf_counter() - self._request_started
            self._request_active = False
            self._expert_request = expert_after.delta(self._expert_before)
            process_disk_io_after = process_disk_io_snapshot()
            self._request_process_disk_io = (
                process_disk_io_after.delta(self._process_disk_io_before)
                if process_disk_io_after is not None
                and self._process_disk_io_before is not None
                else None
            )
            if self._dspark_cache is not None:
                self._dspark_expert_request = self._dspark_cache.metrics_snapshot().delta(
                    self._dspark_expert_before
                )
            if self._mtp_cache is not None:
                self._mtp_expert_request = self._mtp_cache.metrics_snapshot().delta(
                    self._mtp_expert_before
                )
            if self._generation_tokens > 0:
                self._accumulated_generation_tokens += self._generation_tokens
                self._completed_request_count += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            decode_tokens = max(0, self._generation_tokens - 1)
            request_seconds = self._request_seconds
            if self._request_active:
                request_seconds = time.perf_counter() - self._request_started
            dspark_expert = self._dspark_expert_request
            if self._request_active and self._dspark_cache is not None:
                dspark_expert = self._dspark_cache.metrics_snapshot().delta(
                    self._dspark_expert_before
                )
            mtp_expert = self._mtp_expert_request
            if self._request_active and self._mtp_cache is not None:
                mtp_expert = self._mtp_cache.metrics_snapshot().delta(
                    self._mtp_expert_before
                )
            process_disk_io = self._request_process_disk_io
            if self._request_active and self._process_disk_io_before is not None:
                current_process_disk_io = process_disk_io_snapshot()
                process_disk_io = (
                    current_process_disk_io.delta(self._process_disk_io_before)
                    if current_process_disk_io is not None
                    else None
                )
            prefill_tokens = max(
                0, self._prompt_tokens - self._prompt_cache_reused_tokens
            )
            decode_end_to_end_seconds = (
                self._decode_seconds + self._decode_cache_eval_seconds
            )
            decode_latencies = sorted(self._decode_latency_seconds)

            def percentile(fraction: float) -> float:
                if not decode_latencies:
                    return 0.0
                position = (len(decode_latencies) - 1) * fraction
                lower = int(position)
                upper = min(lower + 1, len(decode_latencies) - 1)
                weight = position - lower
                return (
                    decode_latencies[lower] * (1 - weight)
                    + decode_latencies[upper] * weight
                )

            return {
                "runtime_prompt_tokens": self._prompt_tokens,
                "runtime_generation_tokens": self._generation_tokens,
                "accumulated_generation_tokens": (
                    self._accumulated_generation_tokens
                    + (self._generation_tokens if self._request_active else 0)
                ),
                "prompt_cache_reused_tokens": self._prompt_cache_reused_tokens,
                "dspark_prompt_cache_source": self._dspark_prompt_cache_source,
                "approximation_mode": self._approximation_mode,
                "completed_request_count": self._completed_request_count,
                "request_seconds": request_seconds,
                "time_to_first_token_seconds": self._time_to_first_token_seconds,
                "prefill_tokens_per_second": (
                    prefill_tokens / self._time_to_first_token_seconds
                    if self._time_to_first_token_seconds
                    else 0.0
                ),
                "layer_major_prefill_tokens": (
                    max(0, prefill_tokens - 1)
                    if self._layer_major_prefill
                    else 0
                ),
                "decode_seconds": self._decode_seconds,
                "decode_model_step_seconds": self._decode_seconds,
                "decode_cache_eval_seconds": self._decode_cache_eval_seconds,
                "decode_end_to_end_seconds": decode_end_to_end_seconds,
                "decode_tokens_per_second": (
                    decode_tokens / decode_end_to_end_seconds
                    if decode_end_to_end_seconds
                    else 0.0
                ),
                "decode_end_to_end_tokens_per_second": (
                    decode_tokens / decode_end_to_end_seconds
                    if decode_end_to_end_seconds
                    else 0.0
                ),
                "decode_model_step_tokens_per_second": (
                    decode_tokens / self._decode_seconds if self._decode_seconds else 0.0
                ),
                "decode_latency_p50_seconds": percentile(0.5),
                "decode_latency_p95_seconds": percentile(0.95),
                "cache_state_eval_seconds": self._cache_state_eval_seconds,
                "cache_state_eval_count": self._cache_state_eval_count,
                "prompt_cache_snapshot_seconds": (
                    self._prompt_cache_snapshot_seconds
                ),
                "prompt_cache_serialize_seconds": (
                    self._prompt_cache_serialize_seconds
                ),
                "prompt_cache_write_seconds": self._prompt_cache_write_seconds,
                "prompt_cache_write_errors": self._prompt_cache_write_errors,
                "prompt_cache_write_error": self._prompt_cache_write_error,
                "request_prefill_step_size": self._prefill_step_size,
                "layer_major_prefill": self._layer_major_prefill,
                "request_expert_cache_hit_rate": self._expert_request.hit_rate,
                "request_expert_cache_hits": self._expert_request.hits,
                "request_expert_cache_misses": self._expert_request.misses,
                "request_expert_evictions": self._expert_request.evictions,
                "request_expert_bytes_read": self._expert_request.bytes_read,
                "request_expert_page_cache_probe_calls": (
                    self._expert_request.page_cache_probe_calls
                ),
                "request_expert_page_cache_probe_failures": (
                    self._expert_request.page_cache_probe_failures
                ),
                "request_expert_page_cache_classified_bytes": (
                    self._expert_request.page_cache_classified_bytes
                ),
                "request_expert_page_cache_resident_bytes_before_read": (
                    self._expert_request.page_cache_resident_bytes_before_read
                ),
                "request_expert_page_cache_nonresident_bytes_before_read": (
                    self._expert_request.page_cache_nonresident_bytes_before_read
                ),
                "request_expert_page_cache_unclassified_bytes": (
                    self._expert_request.page_cache_unclassified_bytes
                ),
                "request_expert_page_cache_resident_fraction_before_read": (
                    self._expert_request.page_cache_resident_bytes_before_read
                    / self._expert_request.page_cache_classified_bytes
                    if self._expert_request.page_cache_classified_bytes
                    else None
                ),
                "request_expert_page_cache_nonresident_fraction_before_read": (
                    self._expert_request.page_cache_nonresident_bytes_before_read
                    / self._expert_request.page_cache_classified_bytes
                    if self._expert_request.page_cache_classified_bytes
                    else None
                ),
                "request_expert_page_cache_nonresident_bytes_per_generated_token": (
                    self._expert_request.page_cache_nonresident_bytes_before_read
                    / self._generation_tokens
                    if self._generation_tokens
                    else 0.0
                ),
                "request_process_disk_bytes_read": (
                    process_disk_io.bytes_read
                    if process_disk_io is not None
                    else None
                ),
                "request_process_disk_bytes_written": (
                    process_disk_io.bytes_written
                    if process_disk_io is not None
                    else None
                ),
                "request_expert_read_seconds": self._expert_request.read_seconds,
                "request_expert_upload_seconds": self._expert_request.upload_seconds,
                "request_expert_pack_seconds": self._expert_request.pack_seconds,
                "request_ssd_read_bytes_per_second": (
                    self._expert_request.bytes_read
                    / self._expert_request.read_seconds
                    if self._expert_request.read_seconds
                    else 0.0
                ),
                "request_routing_sync_seconds": (
                    self._expert_request.routing_sync_seconds
                ),
                "request_batched_expert_layers": self._expert_request.batched_layers,
                "request_gather_qmm_calls": self._expert_request.gather_qmm_calls,
                "request_prefetched_layer_hits": (
                    self._expert_request.prefetched_layer_hits
                ),
                "request_expert_union_calls": (
                    self._expert_request.expert_union_calls
                ),
                "request_routed_expert_assignments": (
                    self._expert_request.routed_expert_assignments
                ),
                "request_expert_union_experts": (
                    self._expert_request.expert_union_experts
                ),
                "request_expert_union_reused_assignments": max(
                    0,
                    self._expert_request.routed_expert_assignments
                    - self._expert_request.expert_union_experts,
                ),
                "request_expert_union_reuse_rate": (
                    1
                    - self._expert_request.expert_union_experts
                    / self._expert_request.routed_expert_assignments
                    if self._expert_request.routed_expert_assignments
                    else 0.0
                ),
                "request_expert_union_misses": (
                    self._expert_request.expert_union_misses
                ),
                "request_staged_expert_reads": (
                    self._expert_request.staged_expert_reads
                ),
                "request_staged_w13_bytes_read": (
                    self._expert_request.staged_w13_bytes_read
                ),
                "request_staged_w2_bytes_read": (
                    self._expert_request.staged_w2_bytes_read
                ),
                "request_staged_read_seconds": (
                    self._expert_request.staged_read_seconds
                ),
                "request_staged_w2_wait_seconds": (
                    self._expert_request.staged_w2_wait_seconds
                ),
                "request_staged_first_stage_submit_seconds": (
                    self._expert_request.staged_first_stage_submit_seconds
                ),
                "mtp_enabled": self._mtp_enabled,
                "mtp_fallback": self._mtp_fallback,
                "mtp_fallback_rounds": self._mtp_fallback_rounds,
                "mtp_rounds": self._mtp_rounds,
                "mtp_proposed_tokens": self._mtp_proposed_tokens,
                "mtp_accepted_tokens": self._mtp_accepted_tokens,
                "mtp_committed_tokens": self._mtp_committed_tokens,
                "mtp_rejected_tokens": (
                    self._mtp_proposed_tokens - self._mtp_accepted_tokens
                ),
                "mtp_acceptance_rate": (
                    self._mtp_accepted_tokens / self._mtp_proposed_tokens
                    if self._mtp_proposed_tokens
                    else 0.0
                ),
                "mtp_average_accepted_length": (
                    self._mtp_accepted_tokens / self._mtp_rounds
                    if self._mtp_rounds
                    else 0.0
                ),
                "mtp_draft_seconds": self._mtp_draft_seconds,
                "mtp_verification_seconds": self._mtp_verification_seconds,
                "mtp_replay_seconds": self._mtp_replay_seconds,
                "mtp_expert_cache_hit_rate": mtp_expert.hit_rate,
                "mtp_expert_cache_hits": mtp_expert.hits,
                "mtp_expert_cache_misses": mtp_expert.misses,
                "mtp_expert_evictions": mtp_expert.evictions,
                "mtp_expert_bytes_read": mtp_expert.bytes_read,
                "mtp_expert_read_seconds": mtp_expert.read_seconds,
                "mtp_expert_resident_slots": (
                    self._mtp_cache.resident_count
                    if self._mtp_cache is not None
                    else 0
                ),
                "mtp_expert_capacity_slots": (
                    self._mtp_cache.slots if self._mtp_cache is not None else 0
                ),
                "dspark_enabled": self._dspark_enabled,
                "dspark_fallback": self._dspark_fallback,
                "dspark_fallback_would_trigger_rounds": (
                    self._dspark_fallback_would_trigger_rounds
                ),
                "dspark_fallback_triggered_rounds": (
                    self._dspark_fallback_triggered_rounds
                ),
                "dspark_fallback_cost_ratios": tuple(
                    self._dspark_fallback_cost_ratios
                ),
                "dspark_last_fallback_target_step_seconds": (
                    self._dspark_last_fallback_target_step_seconds
                ),
                "dspark_last_fallback_speculative_seconds": (
                    self._dspark_last_fallback_speculative_seconds
                ),
                "dspark_last_fallback_break_even_seconds": (
                    self._dspark_last_fallback_break_even_seconds
                ),
                "dspark_last_fallback_cost_ratio": (
                    self._dspark_last_fallback_cost_ratio
                ),
                "dspark_round_trace": tuple(
                    dict(round_metrics)
                    for round_metrics in self._dspark_round_trace
                ),
                "dspark_rounds": self._dspark_rounds,
                "dspark_proposed_tokens": self._dspark_proposed_tokens,
                "dspark_accepted_tokens": self._dspark_accepted_tokens,
                "dspark_committed_tokens": self._dspark_committed_tokens,
                "dspark_output_budget_trimmed_tokens": (
                    self._dspark_output_budget_trimmed_tokens
                ),
                "dspark_rejected_tokens": (
                    self._dspark_proposed_tokens - self._dspark_accepted_tokens
                ),
                "dspark_acceptance_rate": (
                    self._dspark_accepted_tokens / self._dspark_proposed_tokens
                    if self._dspark_proposed_tokens
                    else 0.0
                ),
                "dspark_average_accepted_length": (
                    self._dspark_accepted_tokens / self._dspark_rounds
                    if self._dspark_rounds
                    else 0.0
                ),
                "dspark_average_committed_length": (
                    self._dspark_committed_tokens / self._dspark_rounds
                    if self._dspark_rounds
                    else 0.0
                ),
                "dspark_average_confidence": (
                    self._dspark_confidence_sum / self._dspark_confidence_count
                    if self._dspark_confidence_count
                    else 0.0
                ),
                "dspark_draft_seconds": self._dspark_draft_seconds,
                "dspark_verification_seconds": self._dspark_verification_seconds,
                "dspark_cache_fork_seconds": self._dspark_cache_fork_seconds,
                "dspark_cache_replay_seconds": self._dspark_cache_replay_seconds,
                "dspark_verification_cache_eval_count": (
                    self._dspark_cache_eval_count
                ),
                "dspark_verification_cache_eval_bytes": (
                    self._dspark_cache_eval_bytes
                ),
                "dspark_cache_fork_layers": self._dspark_cache_fork_layers,
                "dspark_per_position_cache_copies": (
                    self._dspark_per_position_cache_copies
                ),
                "dspark_verification_state_fetch_count": (
                    self._dspark_state_fetch_count
                ),
                "dspark_block_attention_layers": (
                    self._dspark_block_attention_layers
                ),
                "dspark_block_verification_rounds": (
                    self._dspark_block_verification_rounds
                ),
                "dspark_sequential_verification_rounds": (
                    self._dspark_sequential_verification_rounds
                ),
                "dspark_last_verification_mode": (
                    self._dspark_last_verification_mode
                ),
                "dspark_last_sequential_position_seconds": (
                    self._dspark_last_sequential_position_seconds
                ),
                "dspark_last_verification_layer_seconds": (
                    self._dspark_last_layer_seconds
                ),
                "dspark_last_proposed_tokens": self._dspark_last_proposed_tokens,
                "dspark_last_accepted_tokens": self._dspark_last_accepted_tokens,
                "dspark_last_draft_seconds": self._dspark_last_draft_seconds,
                "dspark_last_verification_seconds": (
                    self._dspark_last_verification_seconds
                ),
                "dspark_verification_routed_expert_assignments": (
                    self._dspark_verification_routed_expert_assignments
                ),
                "dspark_verification_expert_union_calls": (
                    self._dspark_verification_expert_union_calls
                ),
                "dspark_verification_expert_union_experts": (
                    self._dspark_verification_expert_union_experts
                ),
                "dspark_verification_expert_union_reused_assignments": max(
                    0,
                    self._dspark_verification_routed_expert_assignments
                    - self._dspark_verification_expert_union_experts,
                ),
                "dspark_verification_expert_union_reuse_rate": (
                    1
                    - self._dspark_verification_expert_union_experts
                    / self._dspark_verification_routed_expert_assignments
                    if self._dspark_verification_routed_expert_assignments
                    else 0.0
                ),
                "dspark_verification_expert_union_misses": (
                    self._dspark_verification_expert_union_misses
                ),
                "dspark_target_expert_bytes_read": (
                    self._dspark_target_expert_bytes_read
                ),
                "dspark_verification_expert_bytes_read": (
                    self._dspark_verification_expert_bytes_read
                ),
                "dspark_replay_expert_bytes_read": (
                    self._dspark_replay_expert_bytes_read
                ),
                "dspark_target_expert_read_seconds": (
                    self._dspark_verification_expert_read_seconds
                ),
                "dspark_draft_expert_bytes_read": dspark_expert.bytes_read,
                "dspark_draft_page_cache_probe_calls": (
                    dspark_expert.page_cache_probe_calls
                ),
                "dspark_draft_page_cache_probe_failures": (
                    dspark_expert.page_cache_probe_failures
                ),
                "dspark_draft_page_cache_classified_bytes": (
                    dspark_expert.page_cache_classified_bytes
                ),
                "dspark_draft_page_cache_resident_bytes_before_read": (
                    dspark_expert.page_cache_resident_bytes_before_read
                ),
                "dspark_draft_page_cache_nonresident_bytes_before_read": (
                    dspark_expert.page_cache_nonresident_bytes_before_read
                ),
                "dspark_draft_page_cache_unclassified_bytes": (
                    dspark_expert.page_cache_unclassified_bytes
                ),
                "dspark_draft_page_cache_nonresident_bytes_per_committed_token": (
                    dspark_expert.page_cache_nonresident_bytes_before_read
                    / self._dspark_committed_tokens
                    if self._dspark_committed_tokens
                    else 0.0
                ),
                "dspark_speculative_expert_bytes_read": (
                    dspark_expert.bytes_read
                    + self._dspark_target_expert_bytes_read
                ),
                "dspark_draft_expert_bytes_per_committed_token": (
                    dspark_expert.bytes_read / self._dspark_committed_tokens
                    if self._dspark_committed_tokens
                    else 0.0
                ),
                "dspark_target_expert_bytes_per_committed_token": (
                    self._dspark_target_expert_bytes_read
                    / self._dspark_committed_tokens
                    if self._dspark_committed_tokens
                    else 0.0
                ),
                "dspark_speculative_expert_bytes_per_committed_token": (
                    (
                        dspark_expert.bytes_read
                        + self._dspark_target_expert_bytes_read
                    )
                    / self._dspark_committed_tokens
                    if self._dspark_committed_tokens
                    else 0.0
                ),
                "dspark_last_verification_expert_union_layer_ids": (
                    self._dspark_last_expert_union_layer_ids
                ),
                "dspark_last_verification_expert_assignments_by_layer": (
                    self._dspark_last_layer_routed_expert_assignments
                ),
                "dspark_last_verification_expert_union_by_layer": (
                    self._dspark_last_layer_expert_union_counts
                ),
                "dspark_last_verification_expert_misses_by_layer": (
                    self._dspark_last_layer_expert_union_misses
                ),
                "dspark_last_verification_expert_bytes_read": (
                    self._dspark_last_verification_expert_bytes_read
                ),
                "dspark_seconds_per_output_token": (
                    (self._dspark_draft_seconds + self._dspark_verification_seconds)
                    / (self._dspark_accepted_tokens + self._dspark_rounds)
                    if self._dspark_rounds
                    else 0.0
                ),
                "dspark_expert_cache_hit_rate": dspark_expert.hit_rate,
                "dspark_expert_cache_hits": dspark_expert.hits,
                "dspark_expert_cache_misses": dspark_expert.misses,
                "dspark_expert_evictions": dspark_expert.evictions,
                "dspark_expert_bytes_read": dspark_expert.bytes_read,
                "dspark_expert_read_seconds": dspark_expert.read_seconds,
                "dspark_expert_upload_seconds": dspark_expert.upload_seconds,
                "dspark_expert_pack_seconds": dspark_expert.pack_seconds,
                "dspark_expert_resident_slots": (
                    self._dspark_cache.resident_count
                    if self._dspark_cache is not None
                    else 0
                ),
                "dspark_expert_capacity_slots": (
                    self._dspark_cache.slots
                    if self._dspark_cache is not None
                    else 0
                ),
                "active_memory_bytes": mx.get_active_memory(),
                "cache_memory_bytes": mx.get_cache_memory(),
                "peak_memory_bytes": mx.get_peak_memory(),
            }


class ModelRuntime:
    """Keep one installed model resident and serialize all generation."""

    support = get_support("deepseek-v4")

    def __init__(self, installed: InstalledModel, config: RuntimeConfig):
        self.installed = installed
        self.support = support_for_installed(installed)
        self._closed = False
        self._model_id = getattr(installed, "model_id", "deepseek-v4")
        self._revision = getattr(installed, "revision", "")
        self._manifest_format = getattr(installed, "format_version", 1)
        from .memory_budget import resolve_cache_budgets
        config = resolve_cache_budgets(installed, config)
        self.config = config
        self.support.validate_config(config)
        self.metrics = RuntimeMetrics()
        self._codec: ToolCodec | None = None
        self._prompt_caches: list[_PromptCacheEntry] = []
        self._persistent_prompt_caches: list[_PersistentPromptCacheEntry] = []
        self._dspark_prompt_caches: list[_DSparkPromptCacheEntry] = []
        self._persistent_dspark_prompt_caches: list[
            _PersistentDSparkPromptCacheEntry
        ] = []
        self._failed_prompt_cache_paths: set[Path] = set()
        self._prompt_cache_directory: Path | None = None
        self._generation_lock = threading.Lock()
        self._generation_stream = mx.new_thread_unsafe_stream(mx.gpu)
        with mx.stream(self._generation_stream):
            self.model, self.expert_cache = load_model(installed, config)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    installed.root / "tokenizer",
                    trust_remote_code=True,
                )
            except Exception:
                self.support.close(self.model, self.expert_cache)
                raise
        self._prompt_cache_directory = self._open_prompt_cache_directory()
        if self._prompt_cache_directory is not None:
            dspark = getattr(self.model, "dspark", None)
            mtp = getattr(self.model, "mtp", None)
            if self._dspark_prompt_cache_enabled(dspark):
                self._persistent_dspark_prompt_caches = (
                    self._scan_persistent_dspark_prompt_caches(dspark)
                )
            elif dspark is None and mtp is None:
                self._persistent_prompt_caches = (
                    self._scan_persistent_prompt_caches()
                )
        # ponytail: the lock serializes graph evaluation as required by the
        # cross-thread MLX stream and remains correct for batch size 1.

    @classmethod
    def open(
        cls,
        model_path: str,
        config: RuntimeConfig = RuntimeConfig(),
    ) -> ModelRuntime:
        return cls(InstalledModel.open(model_path), config)

    @property
    def model_id(self) -> str:
        return self._model_id

    def encode_chat(
        self,
        messages: list[dict[str, Any]],
        thinking_mode: str = "chat",
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = ToolChoice(),
        reasoning_effort: str = "low",
    ) -> str:
        if self._codec is None:
            self._codec = self.support.open_codec(self.installed.root, self.tokenizer)
        return self._codec.encode(
            messages,
            thinking_mode,
            tools,
            tool_choice,
            reasoning_effort,
        )

    def parse_chat(self, text: str, thinking_mode: str) -> AssistantTurn:
        if self._codec is None:
            self._codec = self.support.open_codec(self.installed.root, self.tokenizer)
        return self._codec.parse(text, thinking_mode)

    def make_tool_stream_parser(self, thinking_mode: str):
        return self.support.make_tool_stream_parser(thinking_mode)

    def stream(
        self,
        prompt: str | list[int],
        options: GenerationOptions,
    ) -> Iterator[GeneratedPiece]:
        check_cancelled()
        sampler = make_sampler(
            temp=options.temperature,
            top_p=options.top_p,
            top_k=options.top_k,
            min_p=options.min_p,
        )
        processor_options = {}
        if options.presence_penalty != 0:
            processor_options["presence_penalty"] = options.presence_penalty
        if options.repetition_penalty != 1:
            processor_options["repetition_penalty"] = options.repetition_penalty
        logits_processors = (
            make_logits_processors(**processor_options) if processor_options else []
        )
        with self._generation_lock, _approximation_mode(
            self, options.approximation_mode
        ):
            memory_request = getattr(self.expert_cache, "phase_memory_request", nullcontext)
            with mx.stream(self._generation_stream), memory_request():
                # MLX gives fresh request threads the same initial random state.
                # Reset on the generating thread, after model loading and before
                # any normal, DSpark, or MTP sampling (including cached prompts).
                mx.random.seed(options.seed if options.seed is not None else secrets.randbits(32))
                prompt_tokens = list(prompt) if isinstance(prompt, list) else self._encode_prompt(prompt)
                available_tokens = getattr(
                    self.installed, "maximum_context", 1_048_576
                ) - len(prompt_tokens)
                if available_tokens < 1:
                    raise ValueError("prompt exceeds the checkpoint context limit")
                if options.max_tokens > available_tokens:
                    options = replace(options, max_tokens=available_tokens)
                dspark = getattr(self.model, "dspark", None)
                mtp = getattr(self.model, "mtp", None)
                for cache in (self.expert_cache, getattr(dspark, "expert_cache", None),
                              getattr(mtp, "expert_cache", None)):
                    begin_request = getattr(cache, "begin_route_request", None)
                    if callable(begin_request):
                        begin_request()
                dspark_prompt_cache_enabled = self._dspark_prompt_cache_enabled(dspark)
                dspark_prompt_cache_source = "disabled"
                if dspark_prompt_cache_enabled:
                    dspark_entry, dspark_prompt_cache_source = (
                        self._acquire_dspark_prompt_cache(prompt_tokens, dspark)
                    )
                    entry = _PromptCacheEntry(
                        dspark_entry.cache,
                        list(dspark_entry.tokens),
                        options.approximation_mode,
                    )
                    dspark.restore_cache_state(dspark_entry.context_state)
                elif mtp is not None:
                    entry = _PromptCacheEntry(
                        self.support.new_cache(self.model),
                        [],
                        options.approximation_mode,
                    )
                else:
                    entry = (
                        _PromptCacheEntry(
                            self.support.new_cache(self.model),
                            [],
                            options.approximation_mode,
                        )
                        if dspark is not None
                        else self._acquire_prompt_cache(
                            prompt_tokens,
                            options.approximation_mode,
                        )
                    )
                request = entry.request = next(_PROMPT_CACHE_REQUESTS)
                prompt_cache = entry.cache
                cache_tokens = entry.tokens
                reused_tokens = len(cache_tokens)
                generation_prompt = prompt_tokens[reused_tokens:]
                if dspark is None and mtp is None:
                    cache_tokens.extend(generation_prompt)
                step_size = _select_prefill_step_size(
                    getattr(self.config, "prefill_step_size", 128),
                    len(generation_prompt),
                )
                use_layer_major = self.support.uses_layer_major_prefill(
                    self.config, len(generation_prompt),
                )
                self.metrics.start(
                    len(prompt_tokens),
                    reused_tokens,
                    step_size,
                    use_layer_major,
                    self._expert_metrics(),
                    mtp_enabled=mtp is not None,
                    mtp_cache=getattr(self.model, "mtp_expert_cache", None),
                    dspark_enabled=dspark is not None,
                    dspark_cache=(dspark.expert_cache if dspark is not None else None),
                    dspark_prompt_cache_source=dspark_prompt_cache_source,
                    approximation_mode=options.approximation_mode,
                )
                completed = False
                prefill_persist_entry = None
                prefill_persist_snapshots: dict[int, _PromptCacheSnapshot] = {}

                def record_prefill_checkpoint(processed: int, total: int) -> None:
                    check_cancelled()
                    if (
                        not self._prompt_cache_enabled()
                        or options.approximation_mode != EXACT_APPROXIMATION_MODE
                        or use_layer_major
                        or processed <= 0
                        or processed >= total
                    ):
                        return
                    is_first = not prefill_persist_snapshots
                    is_last = processed == total - 1
                    if not (is_first or is_last):
                        return
                    token_count = reused_tokens + processed
                    if token_count <= reused_tokens:
                        return
                    snapshot_started = time.perf_counter()
                    state = _clone_cache_state(
                        self.support.snapshot_cache(prompt_cache)
                    )
                    arrays = _cache_state_arrays(state)
                    if arrays:
                        mx.eval(*arrays)
                    prefill_persist_snapshots[token_count] = _PromptCacheSnapshot(
                        state,
                        list(prompt_tokens[:token_count]),
                    )
                    self.metrics.record_prompt_cache_snapshot(
                        time.perf_counter() - snapshot_started
                    )

                try:
                    if dspark is not None:
                        dspark_prefilled = reused_tokens
                        if use_layer_major and self.installed.model_kind == "deepseek-v4.1":
                            from .model_support.deepseek_v41 import _deepseek_v41_layer_major_prefill
                            if reused_tokens == 0:
                                dspark.reset_cache()
                            with _route_phase(self.expert_cache, "prefill"):
                                hidden = _deepseek_v41_layer_major_prefill(
                                    self.model, generation_prompt[:-1], prompt_cache,
                                    step_size, self.expert_cache, self.config, dspark.target_layers)
                                if hidden is not None:
                                    dspark.prefill_context(hidden, reused_tokens)
                                    del hidden
                            dspark_prefilled = len(prompt_tokens) - 1
                            if dspark_prompt_cache_enabled and dspark_prefilled > reused_tokens:
                                self._snapshot_dspark_prompt_cache(
                                    prompt_tokens, dspark_prefilled, prompt_cache,
                                    dspark.cache_state(), dspark,
                                )
                        yield from self._stream_dspark(
                            prompt_tokens,
                            prompt_cache,
                            dspark,
                            options,
                            step_size,
                            dspark_prefilled,
                            dspark_prompt_cache_enabled,
                        )
                        completed = True
                        return
                    if mtp is not None:
                        prefilled_hidden = None
                        if use_layer_major:
                            with _route_phase(self.expert_cache, "prefill"):
                                prefilled_hidden = self.support.prefill(
                                    self.model, prompt_tokens[:-1], prompt_cache,
                                    step_size, self.expert_cache, self.config,
                                )
                        yield from self._stream_mtp(
                            prompt_tokens,
                            prompt_cache,
                            mtp,
                            options,
                            step_size,
                            prefilled_hidden,
                            logits_processors,
                        )
                        completed = True
                        return
                    if use_layer_major:
                        with _route_phase(self.expert_cache, "prefill"):
                            self.support.prefill(
                                self.model, generation_prompt[:-1], prompt_cache,
                                step_size, self.expert_cache, self.config,
                            )
                        if self._prompt_cache_enabled():
                            snapshot_started = time.perf_counter()
                            prefill_persist_entry = _PromptCacheEntry(
                                self.support.clone_cache(prompt_cache),
                                list(prompt_tokens[:-1]),
                                options.approximation_mode,
                                request,
                            )
                            self.metrics.record_prompt_cache_snapshot(
                                time.perf_counter() - snapshot_started
                            )
                            self._store_prompt_cache(
                                prefill_persist_entry,
                                persist=False,
                            )
                        generation_prompt = generation_prompt[-1:]
                    with _use_mlx_lm_generation_stream(self._generation_stream):
                        responses = iter(
                            stream_generate(
                                self.model,
                                self.tokenizer,
                                generation_prompt,
                                max_tokens=options.max_tokens,
                                sampler=sampler,
                                logits_processors=logits_processors,
                                prompt_cache=prompt_cache,
                                prefill_step_size=step_size,
                                prompt_progress_callback=record_prefill_checkpoint,
                            )
                        )
                        with closing(responses):
                            first_response = True
                            while True:
                                check_cancelled()
                                started = time.perf_counter()
                                try:
                                    phase = "prefill" if first_response else "decode"
                                    with _route_phase(self.expert_cache, phase):
                                        response = next(responses)
                                except StopIteration:
                                    break
                                check_cancelled()
                                first_response = False
                                step_seconds = time.perf_counter() - started
                                cache_started = time.perf_counter()
                                self.support.evaluate_cache(prompt_cache)
                                cache_seconds = time.perf_counter() - cache_started
                                self.metrics.record(response, step_seconds, cache_seconds)
                                # mlx_lm feeds each token, EOS included, to the
                                # cache before yielding it.
                                cache_tokens.append(int(response.token))
                                if response.finish_reason is not None:
                                    completed = True
                                yield GeneratedPiece(
                                    text=response.text,
                                    token=response.token,
                                    prompt_tokens=len(prompt_tokens),
                                    generation_tokens=response.generation_tokens,
                                    finish_reason=response.finish_reason,
                                )
                finally:
                    if not completed:
                        discard = getattr(self.expert_cache, "discard_prefetched_layers", None)
                        if discard is not None:
                            discard()
                        # Drain submitted work before another request can reuse
                        # expert buffers. Never run MLX cleanup on the observer.
                        mx.synchronize(self._generation_stream)
                    self.metrics.finish(self._expert_metrics())
                    if completed and dspark is None and mtp is None:
                        if (
                            options.approximation_mode == EXACT_APPROXIMATION_MODE
                            and prefill_persist_entry is not None
                        ):
                            self._persist_prompt_cache(prefill_persist_entry)
                        if options.approximation_mode == EXACT_APPROXIMATION_MODE:
                            for snapshot in prefill_persist_snapshots.values():
                                if self._prompt_cache_directory is not None:
                                    self._persist_prompt_cache_snapshot(snapshot)
                                cache = self.support.new_cache(self.model)
                                self.support.restore_cache(cache, snapshot.state)
                                self._store_prompt_cache(_PromptCacheEntry(
                                    cache, list(snapshot.tokens), request=request,
                                ))
                        self._store_prompt_cache(entry, persist=True)

    def _stream_mtp(
        self,
        prompt_tokens: list[int],
        prompt_cache,
        mtp,
        options: GenerationOptions,
        step_size: int,
        prefilled_hidden: mx.array | None,
        logits_processors,
    ) -> Iterator[GeneratedPiece]:
        from .qwen4_exp import generate_mtp_tokens

        tokenizer = TokenizerWrapper(self.tokenizer)
        detokenizer = tokenizer.detokenizer
        responses = iter(
            generate_mtp_tokens(
                prompt_tokens,
                self.model,
                mtp,
                prompt_cache,
                max_tokens=options.max_tokens,
                prefill_step_size=step_size,
                temperature=options.temperature,
                top_p=options.top_p,
                top_k=options.top_k,
                min_p=options.min_p,
                logits_processors=logits_processors,
                prefilled_hidden=prefilled_hidden,
                record_round=self.metrics.record_mtp_round,
                draft_tokens=getattr(self.config, "qwen_mtp_draft_tokens", 5),
                zero_acceptance_limit=getattr(self.config, "qwen_mtp_zero_acceptance_limit", 1),
            )
        )
        with closing(responses):
            generation_tokens = 0
            last_token = 0
            pending_text = ""
            finish_reason = "length"
            while generation_tokens < options.max_tokens:
                check_cancelled()
                started = time.perf_counter()
                try:
                    phase = "prefill" if generation_tokens == 0 else "decode"
                    with _route_phase(self.expert_cache, phase), _route_phase(
                        getattr(mtp, "expert_cache", None), phase
                    ):
                        token, _ = next(responses)
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
                yield GeneratedPiece(
                    text=detokenizer.last_segment,
                    token=token,
                    prompt_tokens=len(prompt_tokens),
                    generation_tokens=generation_tokens,
                    finish_reason=None,
                )
            detokenizer.finalize()
            final_text = detokenizer.last_segment
            if pending_text and not final_text.startswith(pending_text):
                final_text = pending_text + final_text
            self.metrics.record_token(generation_tokens, 0.0, 0.0)
            yield GeneratedPiece(
                text=final_text,
                token=last_token,
                prompt_tokens=len(prompt_tokens),
                generation_tokens=generation_tokens,
                finish_reason=finish_reason,
            )

    def _stream_dspark(
        self,
        prompt_tokens: list[int],
        prompt_cache,
        dspark,
        options: GenerationOptions,
        step_size: int,
        prefilled_tokens: int,
        prompt_cache_enabled: bool,
    ) -> Iterator[GeneratedPiece]:
        tokenizer = TokenizerWrapper(self.tokenizer)
        detokenizer = tokenizer.detokenizer
        responses = iter(
            generate_tokens(
                prompt_tokens,
                self.model,
                dspark,
                prompt_cache,
                max_tokens=options.max_tokens,
                prefill_step_size=step_size,
                temperature=options.temperature,
                top_p=options.top_p,
                confidence_threshold=getattr(
                    self.config,
                    "dspark_confidence_threshold",
                    0.6,
                ),
                record_round=self.metrics.record_dspark_round,
                record_fallback=self.metrics.record_dspark_fallback,
                target_expert_cache=self.expert_cache,


                fallback_enabled=getattr(
                    self.config,
                    "dspark_fallback_enabled",
                    True,
                ),
                sequential_verification=getattr(
                    self.config,
                    "dspark_sequential_verification",
                    False,
                ),

                prefilled_tokens=prefilled_tokens,
                record_prefill_snapshot=(
                    lambda processed, target_cache, context_state: (
                        self._snapshot_dspark_prompt_cache(
                            prompt_tokens,
                            processed,
                            target_cache,
                            context_state,
                            dspark,
                        )
                    )
                    if prompt_cache_enabled
                    else None
                ),
            )
        )
        with closing(responses):
            generation_tokens = 0
            last_token = 0
            pending_text = ""
            finish_reason = "length"
            while generation_tokens < options.max_tokens:
                check_cancelled()
                started = time.perf_counter()
                try:
                    phase = "prefill" if generation_tokens == 0 else "decode"
                    with _route_phase(self.expert_cache, phase), _route_phase(
                        getattr(dspark, "expert_cache", None), phase
                    ):
                        token, _, _ = next(responses)
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
                self.metrics.record_token(
                    generation_tokens,
                    step_seconds,
                    0.0,
                )
                if generation_tokens == options.max_tokens:
                    pending_text = detokenizer.last_segment
                    break
                yield GeneratedPiece(
                    text=detokenizer.last_segment,
                    token=token,
                    prompt_tokens=len(prompt_tokens),
                    generation_tokens=generation_tokens,
                    finish_reason=None,
                )
            detokenizer.finalize()
            final_text = detokenizer.last_segment
            if pending_text and not final_text.startswith(pending_text):
                final_text = pending_text + final_text
            self.metrics.record_token(generation_tokens, 0.0, 0.0)
            yield GeneratedPiece(
                text=final_text,
                token=last_token,
                prompt_tokens=len(prompt_tokens),
                generation_tokens=generation_tokens,
                finish_reason=finish_reason,
            )

    def warm_prompt(self, prompt: str) -> int:
        if getattr(self.model, "mtp", None) is not None:
            return 0
        tokens = self._encode_prompt(prompt)
        if len(tokens) < 2:
            return 0
        with self._generation_lock:
            memory_request = getattr(self.expert_cache, "phase_memory_request", nullcontext)
            with mx.stream(self._generation_stream), memory_request():
                cache = self.support.new_cache(self.model)
                step_size = _select_prefill_step_size(
                    getattr(self.config, "prefill_step_size", 128),
                    len(tokens) - 1,
                )
                begin_request = getattr(self.expert_cache, "begin_route_request", None)
                if callable(begin_request):
                    begin_request()
                with _route_phase(self.expert_cache, "prefill"):
                    self.support.prefill(
                        self.model, tokens[:-1], cache, step_size, self.expert_cache, self.config,
                    )
                self._store_prompt_cache(
                    _PromptCacheEntry(cache, tokens[:-1]),
                    persist=True,
                )
        return len(tokens) - 1

    def _prompt_cache_enabled(self) -> bool:
        return (
            self.support.descriptor.supports("promptCache")
            and getattr(self.config, "prompt_cache_entries", 2) > 0
        )

    def _dspark_prompt_cache_enabled(self, dspark) -> bool:
        return bool(
            dspark is not None
            and self._prompt_cache_enabled()
            and (
                self.support.dspark_reuses_prompt_cache
                or getattr(self.config, "dspark_prompt_cache", False)
            )
        )

    def _acquire_prompt_cache(
        self,
        prompt_tokens: list[int],
        approximation_mode: str = EXACT_APPROXIMATION_MODE,
    ) -> _PromptCacheEntry:
        if not self._prompt_cache_enabled():
            return _PromptCacheEntry(
                self.support.new_cache(self.model),
                [],
                approximation_mode,
            )
        matches = [
            entry
            for entry in self._prompt_caches
            if entry.approximation_mode == approximation_mode
            and len(entry.tokens) < len(prompt_tokens)
            and prompt_tokens[: len(entry.tokens)] == entry.tokens
        ]
        if matches:
            entry = max(matches, key=lambda item: len(item.tokens))
            return _PromptCacheEntry(
                self.support.clone_cache(entry.cache),
                list(entry.tokens),
                approximation_mode,
            )
        if approximation_mode != EXACT_APPROXIMATION_MODE:
            return _PromptCacheEntry(
                self.support.new_cache(self.model),
                [],
                approximation_mode,
            )
        persistent = [
            entry
            for entry in self._persistent_prompt_caches
            if len(entry.tokens) < len(prompt_tokens)
            and prompt_tokens[: len(entry.tokens)] == entry.tokens
            and entry.path not in getattr(self, "_failed_prompt_cache_paths", ())
        ]
        for entry in sorted(persistent, key=lambda item: len(item.tokens), reverse=True):
            check_cancelled()
            loaded = self._load_persistent_prompt_cache(entry)
            check_cancelled()
            if loaded is not None:
                self._record_persistent_prompt_cache_hit(entry)
                return loaded
            self._exclude_persistent_prompt_cache(entry.path)
        check_cancelled()
        return _PromptCacheEntry(
            self.support.new_cache(self.model),
            [],
            approximation_mode,
        )

    def _dspark_prompt_cache_contract_matches(
        self,
        entry: _DSparkPromptCacheEntry,
        dspark,
    ) -> bool:
        return bool(
            entry.tokens
            and entry.revision == str(getattr(self.installed, "revision", ""))
            and entry.target_layers == tuple(dspark.target_layers)
            and len(entry.context_state) == _dspark_context_count(dspark)
            and all(
                state is not None and _cache_state_arrays(state)
                for state in entry.context_state
            )
        )

    def _clone_dspark_prompt_cache_entry(
        self,
        entry: _DSparkPromptCacheEntry,
    ) -> _DSparkPromptCacheEntry:
        target_cache = self.support.clone_cache(entry.cache)
        context_state = _clone_cache_state(entry.context_state)
        self.support.evaluate_cache(target_cache)
        context_arrays = _cache_state_arrays(context_state)
        if context_arrays:
            mx.eval(*context_arrays)
        return _DSparkPromptCacheEntry(
            target_cache,
            context_state,
            list(entry.tokens),
            entry.revision,
            entry.target_layers,
        )

    def _acquire_dspark_prompt_cache(
        self,
        prompt_tokens: list[int],
        dspark,
    ) -> tuple[_DSparkPromptCacheEntry, str]:
        matches = [
            entry
            for entry in self._dspark_prompt_caches
            if self._dspark_prompt_cache_contract_matches(entry, dspark)
            and len(entry.tokens) < len(prompt_tokens)
            and prompt_tokens[: len(entry.tokens)] == entry.tokens
        ]
        if matches:
            entry = max(matches, key=lambda item: len(item.tokens))
            return self._clone_dspark_prompt_cache_entry(entry), "memory"
        persistent = [
            entry
            for entry in self._persistent_dspark_prompt_caches
            if entry.revision == str(getattr(self.installed, "revision", ""))
            and entry.target_layers == tuple(dspark.target_layers)
            and len(entry.tokens) < len(prompt_tokens)
            and prompt_tokens[: len(entry.tokens)] == entry.tokens
            and entry.path not in getattr(self, "_failed_prompt_cache_paths", ())
        ]
        for descriptor in sorted(persistent, key=lambda item: len(item.tokens), reverse=True):
            check_cancelled()
            loaded = self._load_persistent_dspark_prompt_cache(descriptor, dspark)
            check_cancelled()
            if loaded is not None:
                self._store_dspark_prompt_cache(loaded, persist=False)
                return self._clone_dspark_prompt_cache_entry(loaded), "persistent"
            self._exclude_persistent_prompt_cache(descriptor.path)
        check_cancelled()
        return (
            _DSparkPromptCacheEntry(
                self.support.new_cache(self.model),
                (None,) * _dspark_context_count(dspark),
                [],
                str(getattr(self.installed, "revision", "")),
                tuple(dspark.target_layers),
            ),
            "none",
        )

    def _snapshot_dspark_prompt_cache(
        self,
        prompt_tokens: list[int],
        processed: int,
        target_cache,
        context_state,
        dspark,
    ) -> None:
        if not 0 < processed < len(prompt_tokens):
            return
        snapshot_started = time.perf_counter()
        entry = _DSparkPromptCacheEntry(
            self.support.clone_cache(target_cache),
            _clone_cache_state(context_state),
            list(prompt_tokens[:processed]),
            str(getattr(self.installed, "revision", "")),
            tuple(dspark.target_layers),
        )
        if not self._dspark_prompt_cache_contract_matches(entry, dspark):
            return
        self.support.evaluate_cache(entry.cache)
        context_arrays = _cache_state_arrays(entry.context_state)
        if context_arrays:
            mx.eval(*context_arrays)
        self._store_dspark_prompt_cache(entry, persist=False)
        self.metrics.record_prompt_cache_snapshot(
            time.perf_counter() - snapshot_started
        )
        self._persist_dspark_prompt_cache(entry)

    def _store_dspark_prompt_cache(
        self,
        entry: _DSparkPromptCacheEntry,
        *,
        persist: bool = False,
    ) -> None:
        dspark = getattr(self.model, "dspark", None)
        if dspark is None or not self._dspark_prompt_cache_contract_matches(
            entry, dspark
        ):
            return
        self._dspark_prompt_caches = [
            cached
            for cached in self._dspark_prompt_caches
            if cached.tokens != entry.tokens
        ]
        self._dspark_prompt_caches.insert(0, entry)
        maximum = max(1, int(getattr(self.config, "prompt_cache_entries", 2)))
        memory_limit = max(
            1,
            int(getattr(self.config, "prompt_cache_memory_gib", 8)),
        ) * 1024**3
        while len(self._dspark_prompt_caches) > maximum:
            self._dspark_prompt_caches.pop()
        while (
            len(self._dspark_prompt_caches) > 1
            and self._dspark_prompt_cache_bytes() > memory_limit
        ):
            self._dspark_prompt_caches.pop()
        if persist:
            self._persist_dspark_prompt_cache(entry)

    def _dspark_prompt_cache_bytes(self) -> int:
        return sum(
            _cache_state_nbytes(self.support.snapshot_cache(entry.cache))
            + _cache_state_nbytes(entry.context_state)
            for entry in self._dspark_prompt_caches
        )

    def _store_prompt_cache(
        self,
        entry: _PromptCacheEntry,
        *,
        persist: bool = False,
    ) -> None:
        if not self._prompt_cache_enabled():
            return
        self._prompt_caches = [
            cached
            for cached in self._prompt_caches
            if not (
                cached.tokens == entry.tokens
                and cached.approximation_mode == entry.approximation_mode
            )
        ]
        self._prompt_caches.insert(0, entry)
        maximum = max(1, int(getattr(self.config, "prompt_cache_entries", 2)))
        memory_limit = max(
            1,
            int(getattr(self.config, "prompt_cache_memory_gib", 8)),
        ) * 1024**3

        def request(cached: _PromptCacheEntry) -> int:
            return cached.request if cached.request is not None else id(cached)

        recent = list(dict.fromkeys(map(request, self._prompt_caches)))[:maximum]
        self._prompt_caches = [
            cached for cached in self._prompt_caches if request(cached) in recent
        ]
        while len(self._prompt_caches) > 1 and self._prompt_cache_bytes() > memory_limit:
            self._prompt_caches.pop()
        if persist and entry.approximation_mode == EXACT_APPROXIMATION_MODE:
            self._persist_prompt_cache(entry)

    def _prompt_cache_bytes(self) -> int:
        return sum(
            sum(int(getattr(cache, "nbytes", 0)) for cache in entry.cache)
            for entry in self._prompt_caches
        )

    def _expert_metrics(self) -> CacheMetrics:
        snapshot = getattr(self.expert_cache, "metrics_snapshot", None)
        return snapshot() if snapshot is not None else CacheMetrics()

    def _encode_prompt(self, prompt: str) -> list[int]:
        add_special_tokens = self.tokenizer.bos_token is None or not prompt.startswith(
            self.tokenizer.bos_token
        )
        return list(
            self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        )

    def _open_prompt_cache_directory(self) -> Path | None:
        if not self._prompt_cache_enabled():
            return None
        if not getattr(self.config, "persistent_prompt_cache", False):
            return None
        revision = getattr(self.installed, "revision", None)
        if not revision:
            return None
        configured = getattr(self.config, "prompt_cache_directory", None)
        root = (
            Path(configured).expanduser()
            if configured
            else Path.home() / ".dsmodel" / "prompt-cache"
        )
        model_key = getattr(self.installed, "model_id", "deepseek-v4").replace("/", "--")
        manifest_format = getattr(self.installed, "format_version", 1)
        directory = root / model_key / revision / f"format-{manifest_format}"
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return directory

    def _normal_prompt_cache_contract(self) -> dict[str, Any]:
        cached = getattr(self, "_prompt_cache_contract_value", None)
        if cached is None:
            cached = _prompt_cache_contract(self.installed, self.config)
            self._prompt_cache_contract_value = cached
        return cached

    @staticmethod
    def _read_prompt_cache_access(
        access_path: Path,
        fallback_ns: int,
    ) -> tuple[int, int]:
        try:
            access = json.loads(access_path.read_text(encoding="utf-8"))
            reuse_count = max(0, int(access.get("reuseCount", 0)))
            last_access_ns = max(0, int(access.get("lastAccessUnixNs", 0)))
            return reuse_count, last_access_ns
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return 0, fallback_ns

    def _record_persistent_prompt_cache_hit(
        self,
        entry: _PersistentPromptCacheEntry,
    ) -> None:
        access_path = entry.access_path
        if access_path is None:
            return
        reuse_count, _ = self._read_prompt_cache_access(
            access_path,
            entry.last_access_ns,
        )
        temporary = access_path.with_name(
            f"{access_path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
        )
        try:
            temporary.write_text(
                _canonical_json(
                    {
                        "reuseCount": reuse_count + 1,
                        "lastAccessUnixNs": time.time_ns(),
                    }
                ),
                encoding="utf-8",
            )
            os.replace(temporary, access_path)
        except OSError:
            try:
                temporary.unlink()
            except OSError:
                pass

    def _scan_persistent_prompt_caches(self) -> list[_PersistentPromptCacheEntry]:
        directory = self._prompt_cache_directory
        if directory is None:
            return []
        contract = self._normal_prompt_cache_contract()
        contract_sha256 = _sha256_json(contract)
        entries: list[_PersistentPromptCacheEntry] = []
        for metadata_path in sorted(
            directory.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                data_path = directory / metadata["data"]
                if data_path in getattr(self, "_failed_prompt_cache_paths", ()):
                    continue
                cache_format = int(metadata.get("format", 0))
                tokens = [int(token) for token in metadata["tokens"]]
                expected_contract_sha256, blocks, cache_key = (
                    _prompt_cache_block_identity(contract, tokens)
                )
                access_path = (
                    directory
                    / f"{cache_key}.normal.v{_PROMPT_CACHE_FORMAT}.access"
                )
                modified_ns = metadata_path.stat().st_mtime_ns
                reuse_count, last_access_ns = self._read_prompt_cache_access(
                    access_path,
                    modified_ns,
                )
                if (
                    cache_format in _SUPPORTED_PROMPT_CACHE_FORMATS
                    and metadata.get("mode", "normal") == "normal"
                    and metadata.get("revision") == self.installed.revision
                    and metadata.get("contract") == contract
                    and metadata.get("contractSHA256") == contract_sha256
                    and expected_contract_sha256 == contract_sha256
                    and metadata.get("blocks") == blocks
                    and metadata.get("cacheKey") == cache_key
                    and bool(tokens)
                    and data_path.parent == directory
                    and data_path.name
                    == f"{cache_key}.normal.v{_PROMPT_CACHE_FORMAT}.safetensors"
                    and metadata_path.name
                    == f"{cache_key}.normal.v{_PROMPT_CACHE_FORMAT}.json"
                    and data_path.is_file()
                ):
                    entries.append(
                        _PersistentPromptCacheEntry(
                            tokens,
                            data_path,
                            cache_format,
                            cache_key,
                            metadata_path,
                            access_path,
                            reuse_count,
                            last_access_ns,
                        )
                    )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        entries.sort(
            key=lambda entry: (
                entry.reuse_count,
                entry.last_access_ns,
                len(entry.tokens),
            ),
            reverse=True,
        )
        maximum = max(
            1,
            int(getattr(self.config, "persistent_prompt_cache_entries", 8)),
        )
        return entries[:maximum]

    def _scan_persistent_dspark_prompt_caches(
        self,
        dspark,
    ) -> list[_PersistentDSparkPromptCacheEntry]:
        directory = self._prompt_cache_directory
        if directory is None:
            return []
        entries = []
        expected_revision = str(getattr(self.installed, "revision", ""))
        expected_layers = tuple(dspark.target_layers)
        for metadata_path in sorted(
            directory.glob("*.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                data_path = directory / metadata["data"]
                if data_path in getattr(self, "_failed_prompt_cache_paths", ()):
                    continue
                revision = str(metadata.get("revision", ""))
                target_layers = tuple(
                    int(layer) for layer in metadata["targetLayers"]
                )
                tokens = [int(token) for token in metadata["tokens"]]
                if (
                    int(metadata.get("format", 0))
                    == _DSPARK_PROMPT_CACHE_FORMAT
                    and metadata.get("mode") == "dspark"
                    and revision == expected_revision
                    and target_layers == expected_layers
                    and bool(tokens)
                    and data_path.parent == directory
                    and data_path.is_file()
                ):
                    entries.append(
                        _PersistentDSparkPromptCacheEntry(
                            tokens,
                            data_path,
                            _DSPARK_PROMPT_CACHE_FORMAT,
                            revision,
                            target_layers,
                        )
                    )
            except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        maximum = max(
            1,
            int(getattr(self.config, "persistent_prompt_cache_entries", 8)),
        )
        return entries[:maximum]

    def _exclude_persistent_prompt_cache(self, path: Path) -> None:
        # Keep the original files, but do not repeatedly load a failed payload
        # during this model's lifetime, including after a persistence rescan.
        if not hasattr(self, "_failed_prompt_cache_paths"):
            self._failed_prompt_cache_paths = set()
        self._failed_prompt_cache_paths.add(path)
        self._persistent_prompt_caches = [
            entry for entry in self._persistent_prompt_caches if entry.path != path
        ]
        self._persistent_dspark_prompt_caches = [
            entry for entry in self._persistent_dspark_prompt_caches if entry.path != path
        ]

    def _load_persistent_prompt_cache(
        self,
        entry: _PersistentPromptCacheEntry,
    ) -> _PromptCacheEntry | None:
        try:
            arrays, metadata = _load_prompt_cache_file(entry.path)
            schema = json.loads(metadata["state"])
            state = _decode_cache_state(schema, arrays)
            cache = self.support.new_cache(self.model)
            if entry.format == 1:
                if len(cache) != len(state):
                    return None
                for target, saved in zip(cache, state):
                    target.state = saved
            else:
                self.support.restore_cache(cache, state)
            self.support.evaluate_cache(cache)
            return _PromptCacheEntry(cache, list(entry.tokens))
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _load_persistent_dspark_prompt_cache(
        self,
        entry: _PersistentDSparkPromptCacheEntry,
        dspark,
    ) -> _DSparkPromptCacheEntry | None:
        try:
            arrays, metadata = _load_prompt_cache_file(entry.path)
            schema = json.loads(metadata["state"])
            state = _decode_cache_state(schema, arrays)
            if not isinstance(state, dict) or set(state) != {"target", "context"}:
                return None
            context_state = state["context"]
            if not isinstance(context_state, tuple):
                return None
            cache = self.support.new_cache(self.model)
            self.support.restore_cache(cache, state["target"])
            loaded = _DSparkPromptCacheEntry(
                cache,
                context_state,
                list(entry.tokens),
                entry.revision,
                entry.target_layers,
            )
            if not self._dspark_prompt_cache_contract_matches(loaded, dspark):
                return None
            self.support.evaluate_cache(cache)
            context_arrays = _cache_state_arrays(context_state)
            if context_arrays:
                mx.eval(*context_arrays)
            return loaded
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def _persist_prompt_cache(self, entry: _PromptCacheEntry) -> None:
        self._persist_prompt_cache_state(
            entry.tokens,
            self.support.snapshot_cache(entry.cache),
        )

    def _persist_prompt_cache_snapshot(
        self,
        snapshot: _PromptCacheSnapshot,
    ) -> None:
        self._persist_prompt_cache_state(snapshot.tokens, snapshot.state)

    def _persist_prompt_cache_state(
        self,
        tokens: list[int],
        state: Any,
    ) -> None:
        directory = self._prompt_cache_directory
        if directory is None or not tokens:
            return
        contract = self._normal_prompt_cache_contract()
        contract_sha256, blocks, cache_key = _prompt_cache_block_identity(
            contract,
            tokens,
        )
        stem = f"{cache_key}.normal.v{_PROMPT_CACHE_FORMAT}"
        data_path = directory / f"{stem}.safetensors"
        metadata_path = directory / f"{stem}.json"
        access_path = directory / f"{stem}.access"
        if data_path.exists() and metadata_path.exists():
            return
        serialize_started = time.perf_counter()
        arrays: dict[str, mx.array] = {}
        schema = _encode_cache_state(state, arrays)
        mx.eval(*arrays.values())
        self.metrics.record_prompt_cache_serialize(
            time.perf_counter() - serialize_started
        )
        metadata = {
            "format": _PROMPT_CACHE_FORMAT,
            "mode": "normal",
            "revision": self.installed.revision,
            "contract": contract,
            "contractSHA256": contract_sha256,
            "cacheKey": cache_key,
            "blocks": blocks,
            "tokens": tokens,
            "data": data_path.name,
        }

        def save() -> None:
            write_started = time.perf_counter()
            temporary_suffix = f".tmp-{os.getpid()}-{threading.get_ident()}"
            temporary_data = data_path.with_name(
                f"{data_path.stem}{temporary_suffix}.safetensors"
            )
            temporary_metadata = metadata_path.with_name(
                f"{metadata_path.name}{temporary_suffix}"
            )
            temporary_access = access_path.with_name(
                f"{access_path.name}{temporary_suffix}"
            )
            try:
                mx.save_safetensors(
                    temporary_data,
                    arrays,
                    metadata={"state": json.dumps(schema, separators=(",", ":"))},
                )
                temporary_metadata.write_text(
                    _canonical_json(metadata),
                    encoding="utf-8",
                )
                temporary_access.write_text(
                    _canonical_json(
                        {
                            "reuseCount": 0,
                            "lastAccessUnixNs": time.time_ns(),
                        }
                    ),
                    encoding="utf-8",
                )
                os.replace(temporary_data, data_path)
                os.replace(temporary_metadata, metadata_path)
                if not access_path.exists():
                    os.replace(temporary_access, access_path)
                else:
                    temporary_access.unlink(missing_ok=True)
                self._prune_persistent_cache_files("normal")
                self._persistent_prompt_caches = (
                    self._scan_persistent_prompt_caches()
                )
            except Exception as error:
                self.metrics.record_prompt_cache_write_error(error)
                for path in (
                    temporary_data,
                    temporary_metadata,
                    temporary_access,
                ):
                    try:
                        path.unlink()
                    except OSError:
                        pass
            finally:
                self.metrics.record_prompt_cache_write(
                    time.perf_counter() - write_started
                )

        save()

    def _persist_dspark_prompt_cache(
        self,
        entry: _DSparkPromptCacheEntry,
    ) -> None:
        directory = self._prompt_cache_directory
        dspark = getattr(self.model, "dspark", None)
        if (
            directory is None
            or dspark is None
            or not self._dspark_prompt_cache_contract_matches(entry, dspark)
        ):
            return
        digest = hashlib.sha256(
            json.dumps(entry.tokens, separators=(",", ":")).encode()
        ).hexdigest()
        stem = f"{digest}.dspark.v{_DSPARK_PROMPT_CACHE_FORMAT}"
        data_path = directory / f"{stem}.safetensors"
        metadata_path = directory / f"{stem}.json"
        if data_path.exists() and metadata_path.exists():
            return
        serialize_started = time.perf_counter()
        arrays: dict[str, mx.array] = {}
        state = {
            "target": self.support.snapshot_cache(entry.cache),
            "context": entry.context_state,
        }
        schema = _encode_cache_state(state, arrays)
        mx.eval(*arrays.values())
        self.metrics.record_prompt_cache_serialize(
            time.perf_counter() - serialize_started
        )
        metadata = {
            "format": _DSPARK_PROMPT_CACHE_FORMAT,
            "mode": "dspark",
            "revision": entry.revision,
            "targetLayers": list(entry.target_layers),
            "tokens": entry.tokens,
            "data": data_path.name,
        }
        write_started = time.perf_counter()
        temporary_data = data_path.with_name(data_path.stem + ".tmp.safetensors")
        temporary_metadata = metadata_path.with_name(
            metadata_path.stem + ".tmp.json"
        )
        try:
            mx.save_safetensors(
                temporary_data,
                arrays,
                metadata={"state": json.dumps(schema, separators=(",", ":"))},
            )
            temporary_metadata.write_text(
                json.dumps(metadata, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(temporary_data, data_path)
            os.replace(temporary_metadata, metadata_path)
            self._prune_persistent_cache_files("dspark")
        except Exception as error:
            self.metrics.record_prompt_cache_write_error(error)
            for path in (temporary_data, temporary_metadata):
                try:
                    path.unlink()
                except OSError:
                    pass
        finally:
            self.metrics.record_prompt_cache_write(
                time.perf_counter() - write_started
            )

    def _prune_persistent_cache_files(self, mode: str) -> None:
        directory = self._prompt_cache_directory
        if directory is None:
            return
        saved: list[tuple[tuple[int, int, int], Path, dict[str, Any]]] = []
        for metadata_path in directory.glob("*.json"):
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("mode", "normal") != mode:
                    continue
                modified_ns = metadata_path.stat().st_mtime_ns
                reuse_count = 0
                last_access_ns = modified_ns
                if (
                    mode == "normal"
                    and int(metadata.get("format", 0)) == _PROMPT_CACHE_FORMAT
                ):
                    cache_key = str(metadata["cacheKey"])
                    access_path = (
                        directory
                        / f"{cache_key}.normal.v{_PROMPT_CACHE_FORMAT}.access"
                    )
                    reuse_count, last_access_ns = self._read_prompt_cache_access(
                        access_path,
                        modified_ns,
                    )
                saved.append(
                    (
                        (reuse_count, last_access_ns, modified_ns),
                        metadata_path,
                        metadata,
                    )
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        saved.sort(key=lambda item: item[0], reverse=True)
        maximum = max(
            1,
            int(getattr(self.config, "persistent_prompt_cache_entries", 8)),
        )
        for _, stale_metadata, stale in saved[maximum:]:
            try:
                stale_data = directory / stale["data"]
                if stale_data.parent == directory:
                    stale_data.unlink(missing_ok=True)
                cache_key = stale.get("cacheKey")
                if mode == "normal" and isinstance(cache_key, str):
                    stale_format = int(stale.get("format", 0))
                    stale_access = (
                        directory
                        / f"{cache_key}.normal.v{stale_format}.access"
                    )
                    if stale_access.parent == directory:
                        stale_access.unlink(missing_ok=True)
                stale_metadata.unlink(missing_ok=True)
            except (KeyError, OSError, TypeError):
                continue

    def close(self) -> None:
        self._prompt_caches.clear()
        self._persistent_prompt_caches.clear()
        self._dspark_prompt_caches.clear()
        self._persistent_dspark_prompt_caches.clear()
        if hasattr(self, "_failed_prompt_cache_paths"):
            self._failed_prompt_cache_paths.clear()
        if not getattr(self, "_closed", False):
            self._closed = True
            self.support.close(self.model, self.expert_cache)

    def __enter__(self) -> ModelRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
