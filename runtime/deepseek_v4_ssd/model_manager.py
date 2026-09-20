from __future__ import annotations

import gc
import json
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Iterator

import mlx.core as mx

from .cancellation import check_cancelled

from .generation import ModelRuntime, RuntimeMetrics
from .io_metrics import EXPERT_FILE_CACHE_POLICIES
from .model import (
    RuntimeConfig,
    _POWER_SAVING_LIMITS_GBPS,
)

CATALOG_VERSION = 1
MAX_GENERATION_TOKENS = 272_000
from .model_support import get_support
from .model_support.catalog import DESCRIPTORS

MODEL_IDS = {item.kind: item.api_model_id for item in DESCRIPTORS}
MODEL_OWNERS = {item.kind: item.owner for item in DESCRIPTORS}


class ModelCatalogError(ValueError):
    pass


class ModelNotFound(Exception):
    pass


class ModelLoadFailed(Exception):
    def __init__(self, model: str, cause: Exception):
        super().__init__(f"Unable to load model '{model}'.")
        self.model = model
        self.cause = cause


@dataclass(frozen=True)
class ModelDefaults:
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    approximation_mode: str = "exact"
    qwen_adaptive_sampling: bool = True


@dataclass(frozen=True)
class ModelSpec:
    id: str
    alias: str | None
    path: str
    model_kind: str
    runtime: RuntimeConfig
    defaults: ModelDefaults
    warmup_prompt_path: str | None = None

    @property
    def public_name(self) -> str:
        return self.alias or self.id

    @property
    def owner(self) -> str:
        return MODEL_OWNERS[self.model_kind]

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "alias": self.alias,
            "path": self.path,
            "model_kind": self.model_kind,
            "runtime": asdict(self.runtime),
            "defaults": asdict(self.defaults),
            "warmup_prompt_path": self.warmup_prompt_path,
        }


@dataclass(frozen=True)
class ModelRequest:
    name: str
    model_id: str
    runtime: Any
    defaults: ModelDefaults


def load_model_catalog(path: str | Path) -> tuple[ModelSpec, ...]:
    try:
        with Path(path).expanduser().open("rb") as file:
            value = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ModelCatalogError(f"cannot read model catalog: {error}") from error
    return parse_model_catalog(value)


def parse_model_catalog(value: Any) -> tuple[ModelSpec, ...]:
    if not isinstance(value, dict):
        raise ModelCatalogError("model catalog must be a JSON object")
    if set(value) != {"version", "models"}:
        raise ModelCatalogError("model catalog must contain only version and models")
    if value["version"] != CATALOG_VERSION:
        raise ModelCatalogError(
            f"model catalog version must be {CATALOG_VERSION}"
        )
    raw_models = value["models"]
    if not isinstance(raw_models, list):
        raise ModelCatalogError("model catalog models must be an array")

    specs = tuple(_parse_model(item, index) for index, item in enumerate(raw_models))
    ids: dict[str, ModelSpec] = {}
    for spec in specs:
        if spec.id in ids:
            raise ModelCatalogError(f"duplicate model ID: {spec.id}")
        ids[spec.id] = spec

    aliases: dict[str, ModelSpec] = {}
    for spec in specs:
        alias = spec.alias
        if alias is None or alias == spec.id:
            continue
        owner = ids.get(alias)
        if owner is not None and owner.id != spec.id:
            raise ModelCatalogError(
                f"alias '{alias}' conflicts with model ID '{owner.id}'"
            )
        owner = aliases.get(alias)
        if owner is not None:
            raise ModelCatalogError(
                f"alias '{alias}' conflicts with alias for '{owner.id}'"
            )
        aliases[alias] = spec
    return specs


def _parse_model(value: Any, index: int) -> ModelSpec:
    prefix = f"models.{index}"
    required = {
        "id",
        "alias",
        "path",
        "model_kind",
        "runtime",
        "defaults",
        "warmup_prompt_path",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ModelCatalogError(f"{prefix} has invalid fields")

    model_kind = value["model_kind"]
    if model_kind not in MODEL_IDS:
        raise ModelCatalogError(f"{prefix}.model_kind is not supported")
    model_id = value["id"]
    if model_id != MODEL_IDS[model_kind]:
        raise ModelCatalogError(
            f"{prefix}.id must be '{MODEL_IDS[model_kind]}'"
        )
    path = value["path"]
    if not isinstance(path, str) or not path.strip():
        raise ModelCatalogError(f"{prefix}.path must be a non-empty string")

    raw_alias = value["alias"]
    if raw_alias is not None and not isinstance(raw_alias, str):
        raise ModelCatalogError(f"{prefix}.alias must be a string or null")
    alias = raw_alias.strip() if isinstance(raw_alias, str) else None
    alias = alias or None

    warmup = value["warmup_prompt_path"]
    if warmup is not None and (not isinstance(warmup, str) or not warmup.strip()):
        raise ModelCatalogError(
            f"{prefix}.warmup_prompt_path must be a non-empty string or null"
        )

    runtime = _parse_runtime(value["runtime"], f"{prefix}.runtime", model_kind)
    try:
        get_support(model_kind).validate_config(runtime)
    except ValueError as error:
        raise ModelCatalogError(str(error)) from error
    defaults = _parse_defaults(value["defaults"], f"{prefix}.defaults")
    if defaults.approximation_mode != "exact" and not get_support(model_kind).descriptor.supports("approximation"):
        raise ModelCatalogError(f"{prefix}.defaults.approximation_mode is not supported by this model")
    return ModelSpec(
        id=model_id,
        alias=alias,
        path=path,
        model_kind=model_kind,
        runtime=runtime,
        defaults=defaults,
        warmup_prompt_path=warmup,
    )


def _parse_runtime(value: Any, prefix: str, model_kind: str) -> RuntimeConfig:
    if isinstance(value, dict):
        value = dict(value)
        for removed in ("qwen_grouped_decode", "qwen_short_block",
                        "dspark_hash_prefetch", "dspark_adaptive_block", "dspark_hybrid_verification"):
            if removed in value:
                if value.pop(removed) is not False:
                    raise ModelCatalogError(f"{prefix}.{removed} has been removed; remove this setting")
        if "adaptive_expert_prefill_threshold" in value:
            if value.pop("adaptive_expert_prefill_threshold") is not None:
                raise ModelCatalogError(f"{prefix}.adaptive_expert_prefill_threshold has been removed; remove this setting")
    names = {field.name for field in fields(RuntimeConfig)}
    required = names - {'qwen_expert_wave_slots', 'qwen_ngram_io', 'qwen_ngram_cache_bytes', 'qwen_sparse_sdpa', 'qwen_phase_memory', 'qwen_pooled_index_cache', 'qwen_ngram_lookup_optimized', 'qwen_compile_tensor_ops', 'qwen_mtp_draft_tokens', 'qwen_mtp_zero_acceptance_limit', 'expert_cache_bytes', 'mtp_cache_bytes', 'dspark_cache_bytes', 'separate_prefill_io', 'qwen_quantized_kv', 'qwen_quantized_index', 'v41_ced_prefill', 'v41_packed_kv', 'v41_packed_index', 'deepseek_ane_prefill', 'qwen_grouped_experts', 'expert_eviction_policy', 'v41_layer_major_prefill', 'v41_candidate_index', 'v41_next_layer_prefetch'}
    if not isinstance(value, dict) or not required <= set(value) <= names:
        raise ModelCatalogError(f"{prefix} must contain every required RuntimeConfig field")
    try:
        config = RuntimeConfig(**{
            "qwen_grouped_experts": get_support(model_kind).descriptor.supports("groupedExperts"),
            **value,
        })
    except TypeError as error:
        raise ModelCatalogError(f"{prefix} is invalid: {error}") from error
    try:
        validate_runtime_config(config)
    except ValueError as error:
        raise ModelCatalogError(f"{prefix}.{error}") from error
    return config


def validate_runtime_config(config: RuntimeConfig) -> None:
    from .qwen_flash_config import validate_flash_config
    validate_flash_config(config)
    from .memory_budget import validate_cache_budgets
    validate_cache_budgets(config)
    integer_minimums = {
        "slots": 6,
        "read_workers": 1,
        "prefetch_read_workers": 1,
        "prefill_step_size": 0,
        "memory_limit_gib": 0,
        "layer_major_prefill_threshold": 1,
        "prompt_cache_entries": 0,
        "prompt_cache_memory_gib": 1,
        "persistent_prompt_cache_entries": 1,
        "moe_prefill_step_size": 0,
        "dspark_slots": 30,
        "mtp_slots": 10,
    }
    for name, minimum in integer_minimums.items():
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")

    boolean_names = {
        "qwen_pooled_index_cache",
        "qwen_ngram_lookup_optimized",
        "qwen_compile_tensor_ops",
        "qwen_phase_memory",
        "fp8_kv_cache",
        "layer_major_prefill",
        "persistent_prompt_cache",
        "batched_expert_prefill",
        "qwen_grouped_experts",
        "ane_prefill",
        "deepseek_ane_prefill",
        "fp4_index_cache",
        "dspark_enabled",
        "mtp_enabled",
        "dspark_prompt_cache",
        "dspark_fallback_enabled",
        "dspark_sequential_verification",
        "expert_page_cache_probe",
        "separate_prefill_io",
        "ready_expert_decode",
        "v41_next_layer_prefetch",
        "v41_candidate_index",
        "v41_ced_prefill",
        "v41_layer_major_prefill",
        "v41_packed_kv",
        "qwen_quantized_kv",
        "qwen_quantized_index",
        "v41_packed_index",
        "qwen_next_layer_prefetch",
        "staged_expert_streaming",
    }
    for name in boolean_names:
        if type(getattr(config, name)) is not bool:
            raise ValueError(f"{name} must be a boolean")

    for name, maximum in (("qwen_mtp_draft_tokens", 5), ("qwen_mtp_zero_acceptance_limit", 32)):
        value = getattr(config, name)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"{name} must be an integer from 1 through {maximum}")

    if (
        isinstance(config.ane_prefill_ratio, bool)
        or not isinstance(config.ane_prefill_ratio, (int, float))
        or not 0 <= config.ane_prefill_ratio <= 1
    ):
        raise ValueError("ane_prefill_ratio must be between zero and one")
    if not isinstance(config.dspark_confidence_threshold, (int, float)) or isinstance(
        config.dspark_confidence_threshold, bool
    ) or not 0 <= config.dspark_confidence_threshold <= 1:
        raise ValueError("dspark_confidence_threshold must be between zero and one")
    if config.power_saving_limit_gbps is not None and (
        isinstance(config.power_saving_limit_gbps, bool)
        or config.power_saving_limit_gbps not in _POWER_SAVING_LIMITS_GBPS
    ):
        raise ValueError("power_saving_limit_gbps is not supported")
    for name in ("prompt_cache_directory", "expert_route_trace"):
        value = getattr(config, name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string or null")
    if config.expert_file_cache_policy not in EXPERT_FILE_CACHE_POLICIES:
        raise ValueError("expert_file_cache_policy is not supported")
    if config.expert_eviction_policy not in ("lfu", "lru", "route"):
        raise ValueError("expert_eviction_policy must be lfu, lru or route")
    for name in (
        "dspark_prompt_cache",
        "dspark_sequential_verification",
    ):
        if getattr(config, name) and not config.dspark_enabled:
            raise ValueError(f"{name} requires dspark_enabled")
    if not config.dspark_fallback_enabled and not config.dspark_enabled:
        raise ValueError("dspark_fallback_enabled requires dspark_enabled when false")
    if config.staged_expert_streaming and not config.ready_expert_decode:
        raise ValueError("staged_expert_streaming requires ready_expert_decode")
    if config.staged_expert_streaming and config.dspark_enabled:
        raise ValueError("staged_expert_streaming does not support dspark_enabled")


def _parse_defaults(value: Any, prefix: str) -> ModelDefaults:
    required = {"max_tokens", "temperature", "top_p", "top_k"}
    optional = {"approximation_mode", "qwen_adaptive_sampling"}
    if not isinstance(value, dict) or not required <= set(value) <= required | optional:
        raise ModelCatalogError(f"{prefix} has invalid fields")
    approximation = value.get("approximation_mode", "exact")
    if approximation not in ("exact", "learned-route-drop-lowest-1"):
        raise ModelCatalogError(f"{prefix}.approximation_mode is not supported")
    adaptive = value.get("qwen_adaptive_sampling", True)
    if type(adaptive) is not bool:
        raise ModelCatalogError(f"{prefix}.qwen_adaptive_sampling must be a boolean")
    max_tokens = value["max_tokens"]
    temperature = value["temperature"]
    top_p = value["top_p"]
    top_k = value["top_k"]
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not (
        1 <= max_tokens <= MAX_GENERATION_TOKENS
    ):
        raise ModelCatalogError(
            f"{prefix}.max_tokens must be between 1 and {MAX_GENERATION_TOKENS}"
        )
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not (
        0 <= temperature <= 2
    ):
        raise ModelCatalogError(f"{prefix}.temperature must be between 0 and 2")
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)) or not (
        0.000001 <= top_p <= 1
    ):
        raise ModelCatalogError(f"{prefix}.top_p must be between 0.000001 and 1")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not (
        0 <= top_k <= 248_320
    ):
        raise ModelCatalogError(f"{prefix}.top_k must be between 0 and 248320")
    return ModelDefaults(max_tokens, float(temperature), float(top_p), top_k, approximation, adaptive)


def _zero_runtime_metrics() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in RuntimeMetrics().snapshot().items():
        if isinstance(value, bool):
            result[name] = False
        elif isinstance(value, int):
            result[name] = 0
        elif isinstance(value, float):
            result[name] = 0.0
        elif isinstance(value, str):
            result[name] = ""
        elif isinstance(value, tuple):
            result[name] = ()
        elif isinstance(value, list):
            result[name] = []
        elif isinstance(value, dict):
            result[name] = {}
        else:
            result[name] = None
    return result


class ModelManager:
    """Load one catalog model at a time and serialize generation requests."""

    def __init__(
        self,
        models: tuple[ModelSpec, ...] | list[ModelSpec],
        *,
        runtime_loader: Callable[[ModelSpec], Any] | None = None,
        clear_cache: Callable[[], None] = mx.clear_cache,
    ):
        self._models: tuple[ModelSpec, ...] = ()
        self._by_name: dict[str, ModelSpec] = {}
        self._set_models(models)
        self._runtime_loader = runtime_loader or (
            lambda spec: ModelRuntime.open(spec.path, spec.runtime)
        )
        self._clear_cache = clear_cache
        self._generation_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._runtime: Any | None = None
        self._loaded: ModelSpec | None = None
        self._loading: ModelSpec | None = None
        self._accumulated_generation_tokens = 0
        self._completed_request_count = 0
        self._empty_runtime_metrics = _zero_runtime_metrics()

    def models(self) -> list[dict[str, Any]]:
        result = []
        for spec in self._models:
            for name in (spec.id, spec.alias):
                if name is None or (name == spec.id and result and result[-1]["id"] == name):
                    continue
                result.append(
                    {
                        "id": name,
                        "object": "model",
                        "created": 0,
                        "owned_by": spec.owner,
                    }
                )
        return result

    def codex_models(self) -> list[dict[str, Any]]:
        result = []
        for spec in self._models:
            for name in (spec.id, spec.alias):
                if name is None or (name == spec.id and result and result[-1]["slug"] == name):
                    continue
                result.append(
                    {
                        "slug": name,
                        "display_name": name,
                        "description": f"Local {spec.owner} model served by Whallm.",
                        "default_reasoning_level": None,
                        "supported_reasoning_levels": [],
                        "shell_type": "unified_exec",
                        "visibility": "none",
                        "supported_in_api": True,
                        "priority": 99,
                        "availability_nux": None,
                        "upgrade": None,
                        "include_apps_usage_instructions": False,
                        "support_verbosity": False,
                        "default_verbosity": None,
                        "apply_patch_tool_type": None,
                        "truncation_policy": {"mode": "bytes", "limit": 10_000},
                        "context_window": spec.defaults.max_tokens,
                        "max_context_window": spec.defaults.max_tokens,
                        "experimental_supported_tools": [],
                        "input_modalities": ["text"],
                        "base_instructions": (
                            "You are a coding agent running in Codex CLI. Follow the "
                            "developer and user instructions, use the available tools to "
                            "inspect and modify the workspace, and continue until the "
                            "request is complete."
                        ),
                    }
                )
        return result

    @contextmanager
    def request(self, name: Any) -> Iterator[ModelRequest]:
        while not self._generation_lock.acquire(timeout=0.05):
            check_cancelled()
        try:
            check_cancelled()
            spec = self._by_name.get(name) if isinstance(name, str) else None
            if spec is None:
                raise ModelNotFound(str(name) if name is not None else "")
            runtime = self._ensure_loaded(spec, name)
            check_cancelled()
            yield ModelRequest(name, spec.id, runtime, spec.defaults)
        finally:
            self._generation_lock.release()

    def configure(self, value: Any) -> None:
        with self._generation_lock:
            self._configure(value)

    def load(self, name: Any, configuration: Any = None) -> None:
        with self._generation_lock:
            if configuration is not None:
                self._configure(configuration)
            spec = self._by_name.get(name) if isinstance(name, str) else None
            if spec is None:
                raise ModelNotFound(str(name) if name is not None else "")
            self._ensure_loaded(spec, name)

    def unload(self, name: Any) -> None:
        spec = self._by_name.get(name) if isinstance(name, str) else None
        if spec is None:
            raise ModelNotFound(str(name) if name is not None else "")
        with self._generation_lock:
            with self._state_lock:
                if self._loaded != spec:
                    return
            runtime = self._detach_runtime()
            if runtime is not None:
                try:
                    runtime.close()
                finally:
                    del runtime
                    gc.collect()
                    self._clear_cache()

    def status_snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            runtime = self._runtime
            loaded = self._loaded
            loading = self._loading
            accumulated = self._accumulated_generation_tokens
            completed = self._completed_request_count
            if runtime is None or loaded is None:
                performance = {
                    **self._empty_runtime_metrics,
                    "generating": False,
                    "generation_tokens": 0,
                    "tokens_per_second": 0,
                    "ssd_bytes_read": 0,
                    "ssd_read_seconds": 0,
                    "expert_pack_seconds": 0,
                    "expert_eviction_seconds": 0,
                    "routing_sync_seconds": 0,
                    "active_parameters_cache": {
                        "hit_rate": 0,
                        "hits": 0,
                        "misses": 0,
                        "resident_slots": 0,
                        "capacity_slots": 0,
                    },
                }
                return {
                    "model": None,
                    "source_model": None,
                    "model_path": None,
                    "runtime": None,
                    "loaded_model": None,
                    "loading_model": loading.id if loading else None,
                    "performance": performance,
                }

            runtime_metrics = runtime.metrics.snapshot()
            runtime_metrics["accumulated_generation_tokens"] = (
                accumulated + runtime_metrics["accumulated_generation_tokens"]
            )
            runtime_metrics["completed_request_count"] = (
                completed + runtime_metrics["completed_request_count"]
            )
            config = runtime.config
            installed = runtime.installed
            cache = runtime.expert_cache
            cache_metrics = cache.metrics
            ane_controller = getattr(
                getattr(runtime, "model", None), "ane_prefill", None
            )
            ane_status = (
                ane_controller.snapshot()
                if ane_controller is not None
                else {
                    "requested": False,
                    "requested_ratio": 0.0,
                    "active_ratio": 0.0,
                    "ane_channels": 0,
                    "gpu_channels": 0,
                    "active": False,
                    "error": None,
                    "evaluations": 0,
                    "fallbacks": 0,
                }
            )
            return {
                "model": loaded.public_name,
                "source_model": runtime.model_id,
                "model_path": str(installed.root),
                "loaded_model": loaded.id,
                "loading_model": loading.id if loading else None,
                "runtime": {
                    "slots": config.slots,
                    "read_workers": config.read_workers,
                    "prefetch_read_workers": config.prefetch_read_workers,
                    "power_saving_limit_gbps": config.power_saving_limit_gbps,
                    "prefill_step_size": config.prefill_step_size,
                    "moe_prefill_step_size": config.moe_prefill_step_size,
                    "layer_major_prefill": config.layer_major_prefill,
                    "layer_major_prefill_threshold": (
                        config.layer_major_prefill_threshold
                    ),
                    "batched_expert_prefill": config.batched_expert_prefill,
                    "ane_prefill": getattr(config, "ane_prefill", True),
                    "ane_prefill_ratio": getattr(config, "ane_prefill_ratio", 0.0),
                    "ane_prefill_status": ane_status,
                    "prompt_cache_entries": config.prompt_cache_entries,
                    "prompt_cache_memory_gib": config.prompt_cache_memory_gib,
                    "persistent_prompt_cache": config.persistent_prompt_cache,
                    "fp4_index_cache": config.fp4_index_cache,
                    "expert_page_cache_probe": config.expert_page_cache_probe,
                    "expert_file_cache_policy": config.expert_file_cache_policy,
                    "separate_prefill_io": config.separate_prefill_io,
                    "expert_eviction_policy": config.expert_eviction_policy,
                    "expert_file_direct_io_alignment_bytes": getattr(
                        cache, "direct_io_alignment", 0
                    ),
                    "dspark_available": installed.has_dspark,
                    "dspark_enabled": bool(
                        getattr(getattr(runtime, "model", None), "dspark", None)
                    ),
                    "dspark_prompt_cache": config.dspark_prompt_cache,
                    "dspark_fallback_enabled": config.dspark_fallback_enabled,
                    "dspark_sequential_verification": (
                        config.dspark_sequential_verification
                    ),
                    "dspark_confidence_threshold": config.dspark_confidence_threshold,
                    "dspark_slots": config.dspark_slots,
                    "mtp_available": bool(getattr(installed, "has_mtp", False)),
                    "mtp_enabled": bool(
                        getattr(getattr(runtime, "model", None), "mtp", None)
                    ),
                    "mtp_slots": getattr(config, "mtp_slots", 32),
                    "kv_cache": "MXFP8" if config.fp8_kv_cache else "BF16",
                },
                "performance": {
                    "generating": False,
                    "generation_tokens": 0,
                    "tokens_per_second": 0,
                    **runtime_metrics,
                    "ssd_bytes_read": cache_metrics.bytes_read,
                    "ssd_read_seconds": cache_metrics.read_seconds,
                    "expert_pack_seconds": cache_metrics.pack_seconds,
                    "expert_eviction_seconds": cache_metrics.eviction_seconds,
                    "routing_sync_seconds": cache_metrics.routing_sync_seconds,
                    "active_parameters_cache": {
                        "hit_rate": cache_metrics.hit_rate,
                        "hits": cache_metrics.hits,
                        "misses": cache_metrics.misses,
                        "resident_slots": cache.resident_count,
                        "capacity_slots": config.slots,
                    },
                },
            }

    def close(self) -> None:
        with self._generation_lock:
            runtime = self._detach_runtime()
            if runtime is not None:
                try:
                    runtime.close()
                finally:
                    del runtime
                    gc.collect()
                    self._clear_cache()

    def _configure(self, value: Any) -> None:
        model_id = value.get("id") if isinstance(value, dict) else None
        if not isinstance(model_id, str):
            raise ModelCatalogError("configuration.id must be a string")
        with self._state_lock:
            active = self._loading or self._loaded
            if active is not None and active.id == model_id:
                raise ModelCatalogError("the loaded or loading model cannot be configured")
        found = False
        raw_models = []
        for spec in self._models:
            if spec.id == model_id:
                raw_models.append(value)
                found = True
            else:
                raw_models.append(spec.to_json())
        if not found:
            raise ModelNotFound(model_id)
        self._set_models(
            parse_model_catalog({"version": CATALOG_VERSION, "models": raw_models})
        )

    def _set_models(
        self, models: tuple[ModelSpec, ...] | list[ModelSpec]
    ) -> None:
        self._models = tuple(models)
        self._by_name = {}
        for spec in self._models:
            self._by_name[spec.id] = spec
            if spec.alias:
                self._by_name[spec.alias] = spec

    def _ensure_loaded(self, spec: ModelSpec, requested_name: str) -> Any:
        with self._state_lock:
            if self._loaded == spec and self._runtime is not None:
                return self._runtime

        old_runtime = self._detach_runtime(loading=spec)
        if old_runtime is not None:
            try:
                old_runtime.close()
            except Exception as error:
                self._finish_loading()
                raise ModelLoadFailed(requested_name, error) from error
            finally:
                del old_runtime
                gc.collect()
                self._clear_cache()

        runtime = None
        try:
            runtime = self._runtime_loader(spec)
            if spec.warmup_prompt_path:
                with open(spec.warmup_prompt_path, encoding="utf-8") as file:
                    runtime.warm_prompt(file.read())
        except Exception as error:
            if runtime is not None:
                try:
                    runtime.close()
                except Exception:
                    pass
                del runtime
            gc.collect()
            self._clear_cache()
            self._finish_loading()
            raise ModelLoadFailed(requested_name, error) from error

        with self._state_lock:
            self._runtime = runtime
            self._loaded = spec
            self._loading = None
        return runtime

    def _detach_runtime(self, loading: ModelSpec | None = None) -> Any | None:
        with self._state_lock:
            runtime = self._runtime
            if runtime is not None:
                snapshot = runtime.metrics.snapshot()
                self._accumulated_generation_tokens += snapshot.get(
                    "accumulated_generation_tokens", 0
                )
                self._completed_request_count += snapshot.get(
                    "completed_request_count", 0
                )
            self._runtime = None
            self._loaded = None
            self._loading = loading
            return runtime

    def _finish_loading(self) -> None:
        with self._state_lock:
            self._loading = None
