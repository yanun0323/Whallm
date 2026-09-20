from __future__ import annotations

import copy
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, replace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models import deepseek_v4
from mlx_lm.models.cache import CacheList
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from .dspark import VerificationMetrics, load_dspark_model
from .cancellation import check_cancelled
from .expert_cache import BatchedExperts, ExpertCache, _ReadLimiter
from .fp8_cache import CorrectPoolingCache, MXFP8PoolingCache
from .manifest import InstalledModel, Tensor

_ORIGINAL_SPARSE_POOLED_ATTENTION = deepseek_v4._sparse_pooled_attention
_CACHE_CLEAR_THRESHOLD_BYTES = 512 * 1024**2
_POWER_SAVING_LIMITS_GBPS = (0.5, 1, 2, 3, 5, 10, 25)


def _clear_memory_cache() -> None:
    if mx.get_cache_memory() >= _CACHE_CLEAR_THRESHOLD_BYTES:
        mx.clear_cache()


@dataclass(frozen=True)
class RuntimeConfig:
    slots: int = 1_152
    read_workers: int = 4
    prefetch_read_workers: int = 2
    prefill_step_size: int = 0
    fp8_kv_cache: bool = True
    memory_limit_gib: int = 0
    layer_major_prefill: bool = True
    layer_major_prefill_threshold: int = 1_024
    prompt_cache_entries: int = 2
    prompt_cache_memory_gib: int = 8
    persistent_prompt_cache: bool = False
    persistent_prompt_cache_entries: int = 8
    prompt_cache_directory: str | None = None
    moe_prefill_step_size: int = 0
    batched_expert_prefill: bool = True
    qwen_quantized_kv: bool = False
    qwen_quantized_index: bool = False
    v41_packed_kv: bool = False
    v41_packed_index: bool = False
    v41_layer_major_prefill: bool = True
    v41_ced_prefill: bool = False
    v41_candidate_index: bool = False
    v41_next_layer_prefetch: bool = True
    qwen_next_layer_prefetch: bool = False
    qwen_grouped_experts: bool = True
    deepseek_ane_prefill: bool = False
    ane_prefill: bool = True
    ane_prefill_ratio: float = 0.0
    fp4_index_cache: bool = True
    dspark_enabled: bool = False
    dspark_prompt_cache: bool = False
    dspark_confidence_threshold: float = 0.6
    dspark_slots: int = 768
    mtp_enabled: bool = False
    mtp_slots: int = 32
    dspark_fallback_enabled: bool = True
    dspark_sequential_verification: bool = False
    expert_route_trace: str | None = None
    expert_page_cache_probe: bool = False
    expert_file_cache_policy: str = "cached"
    separate_prefill_io: bool = True
    expert_eviction_policy: str = "lfu"
    ready_expert_decode: bool = True
    staged_expert_streaming: bool = False
    power_saving_limit_gbps: float | None = None
    expert_cache_bytes: int | None = None
    mtp_cache_bytes: int | None = None
    dspark_cache_bytes: int | None = None
    # Opt-in Qwen research candidates; no App defaults change.
    qwen_pooled_index_cache: bool = False
    qwen_ngram_lookup_optimized: bool = False
    qwen_compile_tensor_ops: bool = False
    qwen_phase_memory: bool = False
    qwen_expert_wave_slots: int = 0
    qwen_ngram_io: str = "mmap"
    qwen_ngram_cache_bytes: int = 0
    qwen_sparse_sdpa: bool = False
    qwen_mtp_draft_tokens: int = 5
    qwen_mtp_zero_acceptance_limit: int = 1


def _apply_prompt_cache_mode(config: RuntimeConfig, mode: str | None) -> RuntimeConfig:
    if mode is None:
        return config
    return replace(
        config,
        prompt_cache_entries=0 if mode == "off" else max(1, config.prompt_cache_entries),
        persistent_prompt_cache=mode == "disk",
    )


def _configure_memory_limits(
    config: RuntimeConfig,
    *,
    automatic_cap_gib: int = 0,
) -> int:
    maximum = mx.device_info()["max_recommended_working_set_size"]
    requested = config.memory_limit_gib * 1024**3
    automatic = (
        min(maximum, automatic_cap_gib * 1024**3)
        if automatic_cap_gib > 0
        else maximum
    )
    memory_limit = requested if requested > 0 else automatic
    mx.set_memory_limit(memory_limit)
    mx.set_wired_limit(min(memory_limit, maximum))
    return memory_limit


def _select_prefill_step_size(configured: int, prompt_tokens: int) -> int:
    if configured > 0:
        return configured
    if prompt_tokens < 1_024:
        return 128
    if prompt_tokens < 4_096:
        return 256
    return 1_024


def _select_moe_step_size(configured: int, prompt_tokens: int) -> int:
    if configured > 0:
        return configured
    return 4_096 if prompt_tokens >= 4_096 else max(1, prompt_tokens)


def layer_major_prefill(
    model,
    token_ids: list[int],
    prompt_cache,
    step_size: int,
    expert_cache: ExpertCache,
    moe_step_size: int = 0,
    batched_experts: bool = True,
) -> None:
    """Populate the prompt cache while keeping one layer's experts resident."""
    if not token_ids:
        return
    check_cancelled()
    core = model.model
    if len(prompt_cache) != len(core.pipeline_layers):
        raise ValueError("prompt cache does not match the main model layers")
    if getattr(core, "pipeline_size", 1) != 1:
        raise ValueError("layer-major prefill supports one Apple Silicon device")

    # Whole-layer prefill has its own expert buffers. Retaining decode slots
    # here would stack both allocations, including on subsequent requests.
    release_slots = getattr(expert_cache, "release_prefill_slots", None)
    if batched_experts and callable(release_slots):
        release_slots()
    reuse = getattr(expert_cache, "reuse_layer_buffers", nullcontext)
    with reuse() if batched_experts else nullcontext():
        inputs = mx.array(token_ids)[None]
        hidden = core.embed_tokens(inputs)
        hidden = mx.broadcast_to(
            hidden[:, :, None, :],
            (hidden.shape[0], hidden.shape[1], core.args.hc_mult, hidden.shape[2]),
        )
        hidden = mx.contiguous(hidden)

        last_layer = len(core.pipeline_layers) - 1
        moe_step_size = _select_moe_step_size(moe_step_size, len(token_ids))
        for layer_index, (layer, layer_cache) in enumerate(
            zip(core.pipeline_layers, prompt_cache)
        ):
            check_cancelled()
            if not all(
                hasattr(layer, name)
                for name in ("attn_hc", "attn_norm", "attn", "ffn_hc", "ffn_norm", "ffn")
            ):
                outputs = []
                with expert_cache.pin_layer(layer_index):
                    for start in range(0, len(token_ids), step_size):
                        check_cancelled()
                        end = min(start + step_size, len(token_ids))
                        chunk = hidden[:, start:end]
                        chunk_ids = inputs[:, start:end]
                        mask_cache = (
                            layer_cache[0]
                            if isinstance(layer_cache, CacheList)
                            else layer_cache
                        )
                        mask = deepseek_v4.create_attention_mask(
                            chunk[:, :, 0, :],
                            mask_cache,
                            window_size=core.args.sliding_window,
                            return_array=True,
                        )
                        output = layer(chunk, mask, layer_cache, chunk_ids)
                        eval_prompt_cache([layer_cache], output)
                        if layer_index != last_layer:
                            outputs.append(output)
                if layer_index == last_layer:
                    _clear_memory_cache()
                    return
                hidden = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)
                mx.eval(hidden)
                _clear_memory_cache()
                continue

            prefetch = getattr(expert_cache, "prefetch_layer", None)
            record_compute_submit = getattr(
                expert_cache,
                "record_compute_submit",
                None,
            )
            use_batched = bool(
                batched_experts
                and callable(prefetch)
                and hasattr(expert_cache, "batched_layer")
                and layer_index != last_layer
            )
            if use_batched:
                prefetch(layer_index)

            outputs = []
            for start in range(0, len(token_ids), step_size):
                check_cancelled()
                end = min(start + step_size, len(token_ids))
                chunk = hidden[:, start:end]
                mask_cache = (
                    layer_cache[0] if isinstance(layer_cache, CacheList) else layer_cache
                )
                mask = deepseek_v4.create_attention_mask(
                    chunk[:, :, 0, :],
                    mask_cache,
                    window_size=core.args.sliding_window,
                    return_array=True,
                )
                residual = chunk
                value, post, combine = layer.attn_hc(chunk)
                value = layer.attn(layer.attn_norm(value), mask=mask, cache=layer_cache)
                output = deepseek_v4.hc_expand(value, residual, post, combine)
                eval_prompt_cache([layer_cache], output)
                if layer_index != last_layer:
                    outputs.append(output)
            if layer_index == last_layer:
                _clear_memory_cache()
                return

            attention_output = (
                outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)
            )
            mx.eval(attention_output)
            check_cancelled()
            _clear_memory_cache()
            batch_context = (
                expert_cache.batched_layer(layer_index)
                if use_batched
                else nullcontext()
            )
            outputs = []
            with expert_cache.pin_layer(layer_index), batch_context:
                if use_batched and layer_index + 1 < last_layer:
                    prefetch(layer_index + 1)
                for start in range(0, len(token_ids), moe_step_size):
                    check_cancelled()
                    end = min(start + moe_step_size, len(token_ids))
                    residual = attention_output[:, start:end]
                    value, post, combine = layer.ffn_hc(residual)
                    value = layer.ffn(
                        layer.ffn_norm(value),
                        inputs[:, start:end],
                    )
                    output = deepseek_v4.hc_expand(value, residual, post, combine)
                    if callable(record_compute_submit):
                        record_compute_submit(layer_index)
                    mx.eval(output)
                    outputs.append(output)
            hidden = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=1)
            mx.eval(hidden)
            _clear_memory_cache()

class _EmptySwitchGLU(nn.Module):
    def __init__(self, *_: object, activation: nn.Module, **__: object):
        super().__init__()
        self.activation = activation


class _StreamingSwitchGLU(nn.Module):
    def __init__(self, layer: int, cache: ExpertCache, activation: nn.Module):
        super().__init__()
        self.layer = layer
        self.cache = cache
        self.activation = activation

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        selected_array = None
        if getattr(self.cache, "route_trace_enabled", False):
            selected_array = np.asarray(indices, dtype=np.int32)
            self.cache.record_routes(self.layer, selected_array)
        current_batched = getattr(self.cache, "current_batched", None)
        batched = current_batched(self.layer) if callable(current_batched) else None
        if batched is not None:
            if getattr(self.cache, "route_cache_enabled", False):
                self.cache.observe_batched_routes(self.layer, np.asarray(indices, dtype=np.int32))
            return self._gather_qmm(x, indices, batched)

        selected = (
            selected_array
            if selected_array is not None
            else np.asarray(indices, dtype=np.int32)
        )
        if (
            self._is_single_token(x)
            and getattr(self.cache, "ready_expert_decode", False)
        ):
            outputs = {}
            selected_experts = selected.reshape(-1).tolist()
            if getattr(self.cache, "staged_expert_streaming", False):
                for ready in self.cache.iter_staged_ready(
                    self.layer,
                    selected_experts,
                ):
                    submit_started = time.perf_counter()
                    hidden = _mxfp4_swiglu(
                        x,
                        ready.weights,
                        self.activation,
                    )
                    mx.async_eval(hidden)
                    self.cache.record_staged_first_stage_submit(
                        time.perf_counter() - submit_started
                    )
                    ready.finish_w2()
                    output = _mxfp4(
                        hidden,
                        ready.weights.w2,
                        ready.weights.w2_scales,
                    )
                    mx.async_eval(output)
                    outputs[ready.expert] = output
            else:
                for expert, weights in self.cache.iter_ready(
                    self.layer,
                    selected_experts,
                ):
                    hidden = _mxfp4_swiglu(x, weights, self.activation)
                    output = _mxfp4(hidden, weights.w2, weights.w2_scales)
                    mx.async_eval(output)
                    outputs[expert] = output
            return mx.stack(
                [outputs[int(expert)] for expert in selected.reshape(-1)],
                axis=-2,
            )
        resident = self.cache.get_many(self.layer, selected.reshape(-1).tolist())
        return self._forward_with_resident(x, selected, resident)

    def forward_with_resident(
        self,
        x: mx.array,
        indices: mx.array,
        resident,
    ) -> mx.array:
        """Run token-shaped expert math from one pre-acquired layer union."""
        selected = np.asarray(indices, dtype=np.int32)
        return self._forward_with_resident(x, selected, resident)

    def _forward_with_resident(
        self,
        x: mx.array,
        selected: np.ndarray,
        resident,
    ) -> mx.array:
        if self._is_single_token(x):
            outputs = []
            for expert in selected.reshape(-1):
                weights = resident.individual_weights[resident.slots[int(expert)]]
                hidden = _mxfp4_swiglu(x, weights, self.activation)
                outputs.append(_mxfp4(hidden, weights.w2, weights.w2_scales))
            return mx.stack(outputs, axis=-2)
        flat_selected = selected.reshape(-1)
        order = np.argsort(flat_selected, kind="stable")
        boundaries = np.flatnonzero(np.diff(flat_selected[order])) + 1
        flat_x = x.reshape(-1, x.shape[-1])
        outputs = []
        for positions in np.split(order, boundaries):
            expert = int(flat_selected[positions[0]])
            weights = resident.individual_weights[resident.slots[expert]]
            source = mx.take(
                flat_x,
                mx.array(positions // selected.shape[-1]),
                axis=0,
            )
            hidden = _mxfp4_swiglu(source, weights, self.activation)
            outputs.append(_mxfp4(hidden, weights.w2, weights.w2_scales))
        grouped = mx.concatenate(outputs, axis=0)
        restored = mx.take(grouped, mx.array(np.argsort(order)), axis=0)
        return restored.reshape(*selected.shape, -1)

    @staticmethod
    def _is_single_token(x: mx.array) -> bool:
        return (x.ndim == 2 and x.shape[0] == 1) or (
            x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == 1
        )

    def _gather_qmm(
        self,
        x: mx.array,
        indices: mx.array,
        batched: BatchedExperts,
    ) -> mx.array:
        source = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        selected = indices
        inverse = None
        if do_sort:
            source, selected, inverse = _gather_sort(source, indices)
        if batched.w13 is not None and batched.w13_scales is not None:
            projected = mx.gather_qmm(
                source,
                batched.w13,
                batched.w13_scales,
                rhs_indices=selected,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
                sorted_indices=do_sort,
            )
            up, gate = mx.split(projected, 2, axis=-1)
            qmm_calls = 2
        else:
            up = mx.gather_qmm(
                source,
                batched.w3,
                batched.w3_scales,
                rhs_indices=selected,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
                sorted_indices=do_sort,
            )
            gate = mx.gather_qmm(
                source,
                batched.w1,
                batched.w1_scales,
                rhs_indices=selected,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
                sorted_indices=do_sort,
            )
            qmm_calls = 3
        output = mx.gather_qmm(
            self.activation(up, gate),
            batched.w2,
            batched.w2_scales,
            rhs_indices=selected,
            transpose=True,
            group_size=32,
            bits=4,
            mode="mxfp4",
            sorted_indices=do_sort,
        )
        if do_sort:
            output = _scatter_unsort(output, inverse, indices.shape)
        record = getattr(self.cache, "record_gather_qmm", None)
        if callable(record):
            record(qmm_calls)
        return output.squeeze(-2)


def _mxfp4(x: mx.array, weight: mx.array, scales: mx.array) -> mx.array:
    return mx.quantized_matmul(
        x,
        weight,
        scales,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )


def _mxfp4_swiglu(x: mx.array, weights, activation) -> mx.array:
    if weights.w13 is not None and weights.w13_scales is not None:
        projected = _mxfp4(x, weights.w13, weights.w13_scales)
        up, gate = mx.split(projected, 2, axis=-1)
    else:
        up = _mxfp4(x, weights.w3, weights.w3_scales)
        gate = _mxfp4(x, weights.w1, weights.w1_scales)
    return activation(up, gate)


def _streaming_moe(self, x: mx.array, input_ids: mx.array) -> mx.array:
    if getattr(self, "sharding_group", None) is not None:
        raise ValueError("SSD expert streaming supports one Apple Silicon device")
    indices, scores = self.gate(x, input_ids)
    shared = self.shared_experts(x)
    mx.async_eval(shared)
    cache = self.switch_mlp.cache
    current_batched = getattr(cache, "current_batched", None)
    batched = current_batched(self.switch_mlp.layer) if callable(current_batched) else None
    if batched is None or getattr(cache, "route_trace_enabled", False):
        started = time.perf_counter()
        mx.eval(indices)
        cache.record_routing_sync(time.perf_counter() - started)
    routed = self.switch_mlp(x, indices)
    routed = _route_reduce(routed, scores)
    return routed + shared


@mx.compile
def _route_reduce(routed: mx.array, scores: mx.array) -> mx.array:
    return (routed * scores[..., None].astype(routed.dtype)).sum(-2)


def _correct_compressor(self, x: mx.array, pool_cache, offset) -> mx.array:
    batch = x.shape[0]
    kv = self.wkv(x)
    gate = self.wgate(x)
    if pool_cache is None:
        usable = kv.shape[1] // self.compress_ratio * self.compress_ratio
        ready_kv, ready_gate = kv[:, :usable], gate[:, :usable]
        pool_base = offset
    else:
        ready_kv, ready_gate, pool_base = pool_cache.accumulate_windows(kv, gate, offset)

    if ready_kv.size == 0:
        new_pooled = mx.zeros((batch, 0, self.head_dim), dtype=x.dtype)
    else:
        compress = (
            deepseek_v4._overlap_compress_kv
            if self.overlap
            else deepseek_v4._simple_compress_kv
        )
        windows_kv = mx.unflatten(ready_kv, 1, (-1, self.compress_ratio))
        windows_gate = mx.unflatten(ready_gate, 1, (-1, self.compress_ratio))
        previous_kv = previous_gate = None
        if self.overlap and pool_cache is not None:
            previous_kv, previous_gate = pool_cache.previous_window()
        if previous_kv is not None:
            windows_kv = mx.concatenate([previous_kv, windows_kv], axis=1)
            windows_gate = mx.concatenate([previous_gate, windows_gate], axis=1)
            new_pooled = compress(windows_kv, windows_gate, self.ape, self.head_dim)[:, 1:]
        else:
            new_pooled = compress(windows_kv, windows_gate, self.ape, self.head_dim)
        if self.overlap and pool_cache is not None:
            pool_cache.store_previous_window(windows_kv, windows_gate)
        new_pooled = self.norm(new_pooled)
        new_pooled = self.rope(new_pooled[:, None], offset=pool_base).squeeze(1)

    if isinstance(pool_cache, MXFP8PoolingCache):
        pool_cache.update(new_pooled)
        return pool_cache
    return pool_cache.update_and_fetch(new_pooled) if pool_cache is not None else new_pooled


@mx.compile
def _stable_topk_indices(scores: mx.array, count: int) -> mx.array:
    selected = mx.argpartition(-scores, kth=count - 1, axis=-1)[..., :count]
    threshold = mx.min(mx.take_along_axis(scores, selected, axis=-1), axis=-1, keepdims=True)
    size = scores.shape[-1]
    positions = mx.arange(size, dtype=mx.uint32)
    region = mx.where(
        scores > threshold,
        0,
        mx.where(scores == threshold, 1, 2),
    ).astype(mx.uint32)
    keys = region * size + positions
    return mx.sort(mx.argpartition(keys, kth=count - 1, axis=-1)[..., :count], axis=-1)


def _correct_indexer(
    self,
    x: mx.array,
    q_residual: mx.array,
    position_rope,
    pool_cache,
    offset,
):
    batch, length, _ = x.shape
    pooled = self.compressor(x, pool_cache, offset)
    pooled_length = pooled.offset if isinstance(pooled, MXFP8PoolingCache) else pooled.shape[1]
    if pooled_length == 0:
        return None
    query = self.wq_b(q_residual).reshape(batch, length, self.n_heads, self.head_dim)
    query = position_rope(query.transpose(0, 2, 1, 3), offset)
    query = query.astype(mx.float32)
    scores = (
        pooled.index_matmul(query)
        if isinstance(pooled, MXFP8PoolingCache)
        else query @ pooled[:, None].swapaxes(-1, -2).astype(mx.float32)
    )
    scores = mx.maximum(scores, 0) * self.scale
    weights = self.weights_proj(x).astype(mx.float32) * (self.n_heads**-0.5)
    scores = (scores * weights.swapaxes(-1, -2)[..., None]).sum(axis=1)
    mask = pool_cache.make_mask(length, offset) if pool_cache is not None else None
    if mask is not None:
        scores = mx.where(
            mask if mask.ndim == 3 else mask[None],
            scores,
            mx.finfo(scores.dtype).min,
        )
    count = min(self.index_topk, pooled_length)
    return _stable_topk_indices(scores, count)


def _sparse_pooled_attention(
    q: mx.array,
    local_kv: mx.array,
    pooled,
    topk: mx.array,
    local_mask: mx.array | None,
    pooled_mask: mx.array | None,
    scale: float,
    sinks: mx.array | None,
) -> mx.array:
    if not isinstance(pooled, MXFP8PoolingCache):
        return _ORIGINAL_SPARSE_POOLED_ATTENTION(
            q,
            local_kv,
            pooled,
            topk,
            local_mask,
            pooled_mask,
            scale,
            sinks,
        )

    pooled_selected = pooled.gather(topk)
    q_scaled = q * scale
    local_scores = q_scaled @ local_kv.swapaxes(-1, -2)
    local_scores = deepseek_v4._apply_score_mask(local_scores, local_mask)
    normalizer = mx.logsumexp(local_scores, -1, keepdims=True)

    query = q_scaled.transpose(0, 2, 1, 3)
    pooled_scores = query @ pooled_selected.swapaxes(-1, -2)
    pooled_scores = pooled_scores.transpose(0, 2, 1, 3)
    pooled_scores = deepseek_v4._apply_score_mask(pooled_scores, pooled_mask)
    normalizer = mx.logaddexp(
        normalizer,
        mx.logsumexp(pooled_scores, -1, keepdims=True),
    )

    local_weights, pooled_weights = deepseek_v4._split_softmax(
        normalizer,
        local_scores,
        pooled_scores,
        sinks[None, :, None, None] if sinks is not None else None,
    )
    output = local_weights @ local_kv
    pooled_weights = pooled_weights.transpose(0, 2, 1, 3)
    output += (pooled_weights @ pooled_selected).transpose(0, 2, 1, 3)
    return output.astype(q.dtype)


def load_model(
    installed_model: InstalledModel,
    config: RuntimeConfig = RuntimeConfig(),
):
    from .model_support import support_for_installed
    from .memory_budget import resolve_cache_budgets
    config = resolve_cache_budgets(installed_model, config)
    support = support_for_installed(installed_model)
    support.validate_config(config)
    if (
        config.power_saving_limit_gbps is not None
        and config.power_saving_limit_gbps not in _POWER_SAVING_LIMITS_GBPS
    ):
        raise ValueError(
            "power saving limit must be 0.5, 1, 2, 3, 5, 10, or 25 GB/s"
        )
    _configure_memory_limits(
        config,
        automatic_cap_gib=support.descriptor.automatic_memory_gib,
    )
    mx.set_cache_limit(1024**3)

    with (installed_model.root / "config.json").open("rb") as file:
        raw_config = json.load(file)
    read_limiter = (
        _ReadLimiter(int(config.power_saving_limit_gbps * 1_000_000_000))
        if config.power_saving_limit_gbps is not None
        else None
    )
    return support.load(
        installed_model, config, raw_config, _load_common_weights(installed_model), read_limiter,
    )


def forward_with_hidden(
    model,
    inputs: mx.array,
    cache,
    target_layers: tuple[int, ...],
) -> tuple[mx.array, mx.array]:
    """Run the main model and return DSpark target-layer hidden states."""
    if hasattr(model, "forward_with_hidden"):
        return model.forward_with_hidden(inputs, cache, target_layers)
    core = model.model
    hidden = core.embed_tokens(inputs)
    hidden = mx.broadcast_to(
        hidden[:, :, None, :],
        (*hidden.shape[:2], core.args.hc_mult, hidden.shape[-1]),
    )
    hidden = mx.contiguous(hidden)
    if getattr(core, "pipeline_size", 1) != 1:
        raise ValueError("DSpark supports one Apple Silicon device")
    if cache is None:
        cache = [None] * len(core.pipeline_layers)
    first_cache = cache[0]
    mask_cache = first_cache[0] if isinstance(first_cache, CacheList) else first_cache
    mask = deepseek_v4.create_attention_mask(
        hidden[:, :, 0, :],
        mask_cache,
        window_size=core.args.sliding_window,
        return_array=True,
    )
    captured = []
    target_set = set(target_layers)
    for index, (layer, layer_cache) in enumerate(zip(core.pipeline_layers, cache)):
        hidden = layer(hidden, mask, layer_cache, inputs)
        if index in target_set:
            captured.append(hidden.mean(axis=2))
    if len(captured) != len(target_layers):
        raise ValueError("DSpark target layers do not match the main model")
    output = core.norm(core.hc_head(hidden))
    return model.lm_head(output), mx.concatenate(captured, axis=-1)


def verification_forward_with_hidden(
    model,
    inputs: mx.array,
    cache,
    target_layers: tuple[int, ...],
) -> tuple[mx.array, mx.array, list, VerificationMetrics]:
    """Verify one token block on a round-level cache fork."""
    core = model.model
    if inputs.shape[0] != 1:
        raise ValueError("DSpark verification supports batch size one")
    if getattr(core, "pipeline_size", 1) != 1:
        raise ValueError("DSpark supports one Apple Silicon device")
    if len(cache) != len(core.pipeline_layers):
        raise ValueError("prompt cache does not match the main model layers")

    fork_started = time.perf_counter()
    fork, fork_arrays = _fork_prompt_cache(cache)
    if fork_arrays:
        mx.eval(*fork_arrays)
    fork_seconds = time.perf_counter() - fork_started

    hidden = core.embed_tokens(inputs)
    hidden = mx.broadcast_to(
        hidden[:, :, None, :],
        (*hidden.shape[:2], core.args.hc_mult, hidden.shape[-1]),
    )
    hidden = mx.contiguous(hidden)
    captured = []
    target_set = set(target_layers)
    first_cache = fork[0]
    mask_cache = first_cache[0] if isinstance(first_cache, CacheList) else first_cache
    mask = deepseek_v4.create_attention_mask(
        hidden[:, :, 0, :],
        mask_cache,
        window_size=core.args.sliding_window,
        return_array=True,
    )
    layer_seconds = []
    cache_eval_bytes = 0
    for index, (layer, layer_cache) in enumerate(zip(core.pipeline_layers, fork)):
        layer_started = time.perf_counter()
        hidden = layer(hidden, mask, layer_cache, inputs)
        _, evaluated_bytes = eval_prompt_cache([layer_cache], hidden)
        cache_eval_bytes += evaluated_bytes
        layer_seconds.append(time.perf_counter() - layer_started)
        if index in target_set:
            captured.append(hidden.mean(axis=2))

    if len(captured) != len(target_layers):
        raise ValueError("DSpark target layers do not match the main model")
    output = core.norm(core.hc_head(hidden))
    metrics = VerificationMetrics(
        verification_mode="block",
        cache_fork_seconds=fork_seconds,
        layer_seconds=tuple(layer_seconds),
        cache_eval_count=len(layer_seconds),
        cache_eval_bytes=cache_eval_bytes,
        cache_fork_layers=len(fork),
        block_attention_layers=len(layer_seconds),
    )
    return model.lm_head(output), mx.concatenate(captured, axis=-1), fork, metrics


def _sequential_forward_with_hidden(
    model,
    inputs: mx.array,
    cache,
    target_layers: tuple[int, ...],
) -> tuple[mx.array, mx.array, tuple[float, ...], int, int]:
    """Advance an existing cache one token at a time and evaluate every step."""
    if inputs.shape[0] != 1 or inputs.shape[1] < 1:
        raise ValueError("sequential target verification requires non-empty batch one")
    logits_by_position = []
    hidden_by_position = []
    position_seconds = []
    cache_eval_count = 0
    cache_eval_bytes = 0
    for position in range(inputs.shape[1]):
        started = time.perf_counter()
        logits, hidden = forward_with_hidden(
            model,
            inputs[:, position : position + 1],
            cache,
            target_layers,
        )
        evaluated_count, evaluated_bytes = eval_prompt_cache(
            cache,
            logits,
            hidden,
        )
        cache_eval_count += evaluated_count
        cache_eval_bytes += evaluated_bytes
        position_seconds.append(time.perf_counter() - started)
        logits_by_position.append(logits)
        hidden_by_position.append(hidden)
    return (
        mx.concatenate(logits_by_position, axis=1),
        mx.concatenate(hidden_by_position, axis=1),
        tuple(position_seconds),
        cache_eval_count,
        cache_eval_bytes,
    )


def sequential_verification_forward_with_hidden(
    model,
    inputs: mx.array,
    cache,
    target_layers: tuple[int, ...],
) -> tuple[mx.array, mx.array, list, VerificationMetrics]:
    """Verify a draft on one cache fork using sequential target-shaped steps."""
    if inputs.shape[0] != 1 or inputs.shape[1] < 1:
        raise ValueError("sequential target verification requires non-empty batch one")
    if len(cache) != len(model.model.pipeline_layers):
        raise ValueError("prompt cache does not match the main model layers")

    fork_started = time.perf_counter()
    fork, fork_arrays = _fork_prompt_cache(cache)
    if fork_arrays:
        mx.eval(*fork_arrays)
    fork_seconds = time.perf_counter() - fork_started
    logits, hidden, position_seconds, cache_eval_count, cache_eval_bytes = (
        _sequential_forward_with_hidden(
            model,
            inputs,
            fork,
            target_layers,
        )
    )
    metrics = VerificationMetrics(
        verification_mode="sequential",
        sequential_verification_positions=inputs.shape[1],
        sequential_position_seconds=position_seconds,
        cache_fork_seconds=fork_seconds,
        cache_eval_count=cache_eval_count,
        cache_eval_bytes=cache_eval_bytes,
        cache_fork_layers=len(fork),
    )
    return logits, hidden, fork, metrics


def eval_prompt_cache(cache, *dependencies: mx.array) -> tuple[int, int]:
    """Evaluate cache arrays without materializing persistence state."""
    arrays = _cache_arrays(cache)
    mx.eval(*dependencies, *arrays)
    return len(arrays), sum(array.nbytes for array in arrays)


def _cache_arrays(cache) -> list[mx.array]:
    arrays = []
    seen = set()
    for layer_cache in cache:
        items = (
            layer_cache.caches
            if isinstance(layer_cache, CacheList)
            else (layer_cache,)
        )
        for item in items:
            values = [
                getattr(item, name, None)
                for name in (
                    "keys",
                    "values",
                    "buf_kv",
                    "buf_gate",
                    "previous_window_kv",
                    "previous_window_gate",
                    "_pending",
                    "pooled_keys",
                )
            ]
            for name in ("_chunks", "_index_chunks"):
                for pair in getattr(item, name, ()):
                    values.extend(pair)
            while values:
                value = values.pop()
                if isinstance(value, (list, tuple)):
                    values.extend(value)
                elif isinstance(value, mx.array) and id(value) not in seen:
                    seen.add(id(value))
                    arrays.append(value)
    return arrays


def _fork_prompt_cache(cache) -> tuple[list, list[mx.array]]:
    fork = []
    arrays = []
    for layer_cache in cache:
        checkpoint, copied = _clone_layer_cache(layer_cache)
        fork.append(checkpoint)
        arrays.extend(copied)
    return fork, arrays


def _clone_layer_cache(layer_cache):
    checkpoint = copy.deepcopy(layer_cache)
    arrays = []
    pending = (
        list(checkpoint.caches)
        if isinstance(checkpoint, CacheList)
        else [checkpoint]
    )
    for item in pending:
        for name in (
            "keys",
            "values",
            "buf_kv",
            "buf_gate",
            "previous_window_kv",
            "previous_window_gate",
        ):
            value = getattr(item, name, None)
            if isinstance(value, mx.array):
                # Copy as bytes: floating addition changes signed zero and can
                # alter subnormal/NaN payloads in a speculative checkpoint.
                copied = (
                    value.reshape(-1).view(mx.uint8) + mx.zeros((), mx.uint8)
                ).view(value.dtype).reshape(value.shape)
                setattr(item, name, copied)
                arrays.append(copied)
    return checkpoint, arrays


def _load_dspark(
    installed_model: InstalledModel,
    main_model,
    args,
    config: RuntimeConfig,
    read_limiter: _ReadLimiter | None,
):
    minimum_slots = (
        installed_model.selected_expert_count * installed_model.dspark_block_size
    )
    if config.dspark_slots < minimum_slots:
        raise ValueError(f"DSpark needs at least {minimum_slots} expert slots")
    common = _load_tensor_file(
        installed_model.root / "dspark/common.bin",
        installed_model.dspark_common_tensors,
    )
    expert_cache = ExpertCache(
        installed_model,
        config.dspark_slots,
        config.read_workers,
        config.prefetch_read_workers,
        layer_count=installed_model.dspark_layer_count,
        expert_directory=installed_model.root / "dspark/experts",
        read_limiter=read_limiter,
        page_cache_probe=config.expert_page_cache_probe,
        file_cache_policy=config.expert_file_cache_policy,
        separate_prefill_io=getattr(config, "separate_prefill_io", True),
        eviction_policy=config.expert_eviction_policy,
    )
    try:
        dspark = load_dspark_model(
            main_model,
            args,
            common,
            expert_cache,
            block_size=installed_model.dspark_block_size,
            noise_token_id=installed_model.dspark_noise_token_id,
            target_layers=installed_model.dspark_target_layer_ids,
            markov_rank=installed_model.dspark_markov_rank,
        )
        mx.eval(dspark.parameters())
    except Exception:
        expert_cache.close()
        raise
    return dspark


def _load_qwen_mtp(
    installed_model: InstalledModel,
    args,
    config: RuntimeConfig,
    read_limiter: _ReadLimiter | None,
):
    if installed_model.mtp is None:
        raise ValueError("Qwen MTP requires an installed MTP sidecar")
    from .qwen4_exp import MTPModel

    mtp_installed = replace(
        installed_model,
        root=installed_model.root / "mtp",
        layer_count=1,
        common_tensors=installed_model.mtp.common_tensors,
        ngram=None,
        mtp=None,
    )
    expert_cache = ExpertCache(
        mtp_installed,
        config.mtp_slots,
        config.read_workers,
        config.prefetch_read_workers,
        ready_expert_decode=config.ready_expert_decode,
        read_limiter=read_limiter,
        page_cache_probe=config.expert_page_cache_probe,
        file_cache_policy=config.expert_file_cache_policy,
        separate_prefill_io=getattr(config, "separate_prefill_io", True),
        eviction_policy=config.expert_eviction_policy,
    )
    try:
        model = MTPModel(args, expert_cache)
        weights = _load_tensor_file(
            installed_model.root / "mtp/common.bin",
            installed_model.mtp.common_tensors,
        )
        model.load_weights(list(model.sanitize(weights).items()), strict=True)
        model.eval()
        mx.eval(model.parameters())
        return model, expert_cache
    except Exception:
        expert_cache.close()
        raise


def _load_common_weights(installed_model: InstalledModel) -> dict[str, mx.array]:
    return _load_tensor_file(
        installed_model.root / "common.bin",
        installed_model.common_tensors,
    )


def _load_tensor_file(path, tensors: tuple[Tensor, ...]) -> dict[str, mx.array]:
    mapped = np.memmap(path, mode="r", dtype=np.uint8)
    weights: dict[str, mx.array] = {}
    for tensor in tensors:
        weights[tensor.name] = _tensor_from_buffer(mapped, tensor)
    return weights


def _tensor_from_buffer(buffer: np.memmap, tensor: Tensor) -> mx.array:
    numpy_dtype, mlx_view = {
        "BOOL": (np.bool_, None),
        "F16": (np.float16, None),
        "BF16": (np.uint16, mx.bfloat16),
        "F32": (np.float32, None),
        "F64": (np.float64, None),
        "I8": (np.int8, None),
        "I16": (np.int16, None),
        "I32": (np.int32, None),
        "I64": (np.int64, None),
        "U8": (np.uint8, None),
        "U16": (np.uint16, None),
        "U32": (np.uint32, None),
        "U64": (np.uint64, None),
        "F8_E4M3": (np.uint8, None),
        "F8_E8M0": (np.uint8, None),
    }.get(tensor.dtype, (None, None))
    if numpy_dtype is None:
        raise ValueError(f"unsupported tensor dtype {tensor.dtype} for {tensor.name}")
    item_size = np.dtype(numpy_dtype).itemsize
    if tensor.length != int(np.prod(tensor.shape)) * item_size:
        raise ValueError(f"tensor byte count does not match its shape: {tensor.name}")
    view = np.ndarray(
        tensor.shape,
        dtype=numpy_dtype,
        buffer=buffer,
        offset=tensor.offset,
    )
    result = mx.array(view)
    return result.view(mlx_view) if mlx_view is not None else result
