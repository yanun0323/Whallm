from __future__ import annotations

from .cancellation import check_cancelled

import math
import time
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Callable, Iterator

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.base import create_ssm_mask
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet
from mlx_lm.models.rope_utils import initialize_rope
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

from .expert_cache import ExpertCache, QwenBatchedExperts, QwenExpertWeights
from .manifest import InstalledModel, NGram
# Kept as module-level names for existing callers and checkpoint-contract tests.
from .qwen_ngram_hash import ngram_ids, shift_right_ignore_eos as _shift_right_ignore_eos


@dataclass
class ModelArgs:
    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2_560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    linear_num_value_heads: int = 48
    linear_num_key_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    full_attention_interval: int = 4
    num_experts: int = 512
    num_experts_per_tok: int = 10
    norm_topk_prob: bool = True
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    rms_norm_eps: float = 1e-6
    vocab_size: int = 248_320
    max_position_embeddings: int = 262_144
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000_000
    rope_parameters: dict[str, Any] | None = None
    indexer_n_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128
    indexer_budget: int = 2_048
    indexer_compress_ratio: int = 4
    hc_count: int = 4
    hc_lowrank: int = 320
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ple_embed_dim: int = 2_560
    ple_conv_kernel_size: int = 4
    ple_layer_ids: tuple[int, ...] = (2,)
    eos_token_id: int = 248_044
    layer_types: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> ModelArgs:
        values = dict(values)
        rope = values.get("rope_parameters") or {}
        values.setdefault("partial_rotary_factor", rope.get("partial_rotary_factor", 0.25))
        values.setdefault("rope_theta", rope.get("rope_theta", 10_000_000))
        if isinstance(values.get("eos_token_id"), list):
            values["eos_token_id"] = values["eos_token_id"][0]
        allowed = {item.name for item in fields(cls)}
        values = {key: value for key, value in values.items() if key in allowed}
        for key in ("layer_types", "ple_layer_ids"):
            if key in values:
                values[key] = tuple(values[key])
        args = cls(**values)
        if not args.layer_types:
            args.layer_types = tuple(
                "full_attention" if (layer + 1) % args.full_attention_interval == 0
                else "linear_attention"
                for layer in range(args.num_hidden_layers)
            )
        return args


class GroupRMSNorm(nn.Module):
    def __init__(self, dimensions: int, group_size: int | None, eps: float):
        super().__init__()
        self.weight = mx.ones((dimensions,))
        self.group_size = group_size
        self.eps = eps
        self.compiled = False

    def __call__(self, value: mx.array) -> mx.array:
        original_shape = value.shape
        if self.group_size is not None:
            value = value.reshape(*value.shape[:-1], -1, self.group_size)
            weight = self.weight.reshape(-1, self.group_size)
            if self.compiled:
                from .qwen_tensor_ops import compiled_group_rms_norm
                output = compiled_group_rms_norm(value, weight, self.eps)
            else:
                normalized = value.astype(mx.float32) * mx.rsqrt(
                    mx.mean(mx.square(value.astype(mx.float32)), axis=-1, keepdims=True)
                    + self.eps
                )
                output = (normalized * weight).astype(value.dtype)
        else:
            output = mx.fast.rms_norm(value, self.weight, self.eps)
        return output.reshape(original_shape)


class RMSNormGated(nn.Module):
    def __init__(self, dimensions: int, eps: float):
        super().__init__()
        self.weight = mx.ones((dimensions,))
        self.eps = eps

    def __call__(self, value: mx.array, gate: mx.array | None = None) -> mx.array:
        output = mx.fast.rms_norm(value, self.weight, self.eps)
        return output if gate is None else output * mx.sigmoid(gate)


class GatedResidual(nn.Module):
    def __init__(self, args: ModelArgs, *, combine: bool = True):
        super().__init__()
        dimensions = args.hc_count * args.hidden_size
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        self.compiled = False
        self.hc_norm = GroupRMSNorm(dimensions, args.hidden_size, args.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(dimensions, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, dimensions, bias=False)
        self.block_inject_weight = (
            nn.Linear(dimensions, args.hc_count, bias=False) if combine else None
        )

    def __call__(self, hyper_input: mx.array):
        normalized = self.hc_norm(hyper_input)
        if self.compiled:
            from .qwen_tensor_ops import (
                compiled_scaled_silu, compiled_hyper_mix, compiled_injection_gate,
            )
            mixing = compiled_scaled_silu(
                self.input_mix_weight_down(normalized), self.hc_count)
            mixed = compiled_hyper_mix(
                normalized, self.input_mix_weight_up(mixing), self.hc_count)
            if self.block_inject_weight is None:
                return mixed
            injection = compiled_injection_gate(
                self.block_inject_weight(normalized), self.hc_count)
        else:
            mixing = nn.silu(self.input_mix_weight_down(normalized) / self.hc_count)
            mixing = mx.sigmoid(self.input_mix_weight_up(mixing))
            mixing = mixing.reshape(*mixing.shape[:-1], self.hc_count, self.hidden_size)
            streams = normalized.reshape(
                *normalized.shape[:-1], self.hc_count, self.hidden_size
            )
            mixed = (mixing * streams).mean(axis=-2)
            if self.block_inject_weight is None:
                return mixed
            injection = 2 * mx.sigmoid(self.block_inject_weight(normalized) / self.hc_count)
        return mixed, hyper_input, injection

    def inject(self, residual: mx.array, result: mx.array, injection: mx.array) -> mx.array:
        if self.compiled:
            from .qwen_tensor_ops import compiled_hyper_inject
            return compiled_hyper_inject(residual, result, injection)
        return residual + (result[..., None, :] * injection[..., None]).reshape(residual.shape)


def mtp_prefill_pairs(
    target_hidden: mx.array,
    prompt_token_ids: mx.array,
) -> tuple[mx.array, mx.array]:
    """Pair target hidden[S] with prompt token[S+1]."""
    if target_hidden.ndim != 3 or prompt_token_ids.ndim != 2:
        raise ValueError("Qwen MTP Prefill requires batched hidden states and tokens")
    if target_hidden.shape[:2] != prompt_token_ids.shape:
        raise ValueError("Qwen MTP Prefill hidden state and token shape do not match")
    return target_hidden[:, :-1], prompt_token_ids[:, 1:]


def rollback_mtp_cache(cache: CacheList, checkpoint: int) -> None:
    """Restore both QSA cache branches to one earlier token count."""
    if checkpoint < 0:
        raise ValueError("Qwen MTP cache checkpoint must not be negative")
    current = cache.size()
    if checkpoint > current:
        raise ValueError("Qwen MTP cache checkpoint is ahead of the cache")
    trimmed = cache.trim(current - checkpoint)
    if trimmed != current - checkpoint or cache.size() != checkpoint:
        raise ValueError("Qwen MTP cache branches do not share one token count")


class NGramStore:
    """Read and decode only the requested FP8 N-gram rows."""

    def __init__(self, path: Path, descriptor: NGram, weight_scale: float = 1.0, *,
                 optimized: bool = False, io_backend: str = "mmap", cache_bytes: int = 0):
        self.path = path
        self.descriptor = descriptor
        expected = descriptor.shard_count * descriptor.shard_row_count * descriptor.row_bytes
        if path.stat().st_size != expected:
            raise ValueError("N-gram file size does not match the manifest")
        if descriptor.dtype != "F8_E4M3":
            raise ValueError("Qwen N-gram store must use F8_E4M3")
        if not math.isfinite(weight_scale) or weight_scale <= 0:
            raise ValueError("Qwen N-gram weight scale must be finite and positive")
        self.weight_scale = weight_scale
        self.optimized = optimized
        if optimized:
            from .qwen_ngram_lookup import fp8_table
            self._fp8_table = fp8_table(weight_scale)
        self._row_count = descriptor.shard_count * descriptor.shard_row_count
        self._reader = None
        self._rows = None
        self._closed = False
        if io_backend not in ("mmap", "pread"):
            raise ValueError("N-gram I/O backend must be mmap or pread")
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError("N-gram cache bytes must be a nonnegative integer")
        if io_backend == "pread":
            from .qwen_flash_io import NGramRowReader
            self._reader = NGramRowReader(
                path, self._row_count, descriptor.row_bytes,
                cache_bytes=cache_bytes, check_cancelled=check_cancelled,
            )
            return
        if cache_bytes:
            raise ValueError("N-gram row cache requires the pread backend")
        self._rows = np.memmap(
            path,
            mode="r",
            dtype=np.uint8,
            shape=(
                descriptor.shard_count * descriptor.shard_row_count,
                descriptor.row_bytes,
            ),
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._reader is not None:
                self._reader.close()
            if self._rows is not None:
                self._rows._mmap.close()
                self._rows = None

    def io_snapshot(self) -> dict:
        if self._reader is not None:
            return {"backend": "pread", **self._reader.snapshot()}
        return {"backend": "mmap"}

    def lookup(self, row_ids: np.ndarray) -> mx.array:
        if self._closed:
            raise ValueError("N-gram store is closed")
        check_cancelled()
        row_ids = np.asarray(row_ids, dtype=np.int64)
        if row_ids.size and (
            row_ids.min() < 0 or row_ids.max() >= self._row_count
        ):
            raise ValueError("N-gram row is outside ngram.bin")
        if self.optimized and self._reader is None:
            from .qwen_ngram_lookup import lookup_rows
            return lookup_rows(self._rows, row_ids, self._fp8_table)
        copied = (self._reader.lookup(row_ids) if self._reader is not None
                  else np.array(self._rows[row_ids], copy=True))
        if self.optimized:
            decoded = self._fp8_table[copied]
            if np.isnan(decoded).any():
                raise ValueError("Qwen N-gram store contains NaN")
            return mx.array(decoded).astype(mx.bfloat16)
        exponent = (copied >> 3) & 0x0F
        mantissa = copied & 0x07
        sign = np.where(copied & 0x80, -1.0, 1.0)
        normal = np.ldexp(
            1.0 + mantissa.astype(np.float32) / 8.0,
            exponent.astype(np.int16) - 7,
        )
        subnormal = np.ldexp(mantissa.astype(np.float32), -9)
        decoded = sign * np.where(exponent == 0, subnormal, normal)
        if np.any((exponent == 15) & (mantissa == 7)):
            raise ValueError("Qwen N-gram store contains NaN")
        return mx.array(decoded * self.weight_scale).astype(mx.bfloat16)


class NGramEmbedding(nn.Module):
    def __init__(self, args: ModelArgs, store: NGramStore):
        super().__init__()
        self.args = args
        self.store = store
        self.layer_multipliers = mx.zeros((3,), dtype=mx.int64)

    def __call__(self, input_ids: mx.array, cache: ArraysCache | None) -> mx.array:
        current = np.asarray(input_ids, dtype=np.int64)
        context = None if cache is None else cache[2]
        if context is None:
            previous = np.full(
                (current.shape[0], self.args.ngram_size - 1),
                self.args.eos_token_id,
                dtype=np.int64,
            )
        else:
            previous = np.asarray(context, dtype=np.int64)
        history = np.concatenate([previous, current], axis=1)
        if cache is not None:
            cache[2] = mx.array(history[:, -(self.args.ngram_size - 1) :])
        rows = ngram_ids(
            history,
            np.asarray(self.layer_multipliers, dtype=np.int64),
            self.store.descriptor,
            eos_token_id=self.args.eos_token_id,
        )[:, -current.shape[1] :]
        return self.store.lookup(rows).reshape(*current.shape, self.args.ple_embed_dim)


class PLELayer(nn.Module):
    def __init__(self, args: ModelArgs, store: NGramStore):
        super().__init__()
        dimensions = args.hc_count * args.hidden_size
        self.args = args
        self.ple_embedding = NGramEmbedding(args, store)
        self.key_proj = nn.Linear(args.ple_embed_dim, dimensions, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, args.hidden_size, bias=False)
        self.norm_key = GroupRMSNorm(dimensions, args.hidden_size, args.rms_norm_eps)
        self.norm_query = GroupRMSNorm(dimensions, args.hidden_size, args.rms_norm_eps)
        self.norm_conv = GroupRMSNorm(dimensions, args.hidden_size, args.rms_norm_eps)
        self.short_state = (args.ple_conv_kernel_size - 1) * args.ngram_size
        self.conv1d = nn.Conv1d(
            dimensions,
            dimensions,
            args.ple_conv_kernel_size,
            dilation=args.ngram_size,
            groups=dimensions,
            bias=False,
        )

    def __call__(
        self,
        hidden: mx.array,
        input_ids: mx.array,
        cache: ArraysCache | None,
    ) -> mx.array:
        embeddings = self.ple_embedding(input_ids, cache)
        shape = (*hidden.shape[:-1], self.args.hc_count, self.args.hidden_size)
        key = self.norm_key(self.key_proj(embeddings)).reshape(shape)
        query = self.norm_query(hidden).reshape(shape)
        gate = (key * query).sum(axis=-1, keepdims=True) / math.sqrt(
            self.args.hidden_size
        )
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        value = mx.sigmoid(gate) * self.value_proj(embeddings)[..., None, :]
        value = value.reshape(*hidden.shape)
        normalized = self.norm_conv(value)
        previous = None if cache is None else cache[3]
        if previous is None:
            previous = mx.zeros(
                (hidden.shape[0], self.short_state, hidden.shape[-1]),
                dtype=hidden.dtype,
            )
        convolution_input = mx.concatenate([previous, normalized], axis=1)
        if cache is not None:
            cache[3] = mx.contiguous(convolution_input[:, -self.short_state :])
        return value + nn.silu(self.conv1d(convolution_input))


def qsa_causal_block_mask(
    query_positions: np.ndarray,
    block_count: int,
    compress_ratio: int = 4,
) -> np.ndarray:
    """Mark complete QSA blocks that are visible to each causal query."""
    queries = np.asarray(query_positions, dtype=np.int64)
    block_ends = np.arange(block_count, dtype=np.int64) * compress_ratio + compress_ratio - 1
    return block_ends[None, :] <= queries[:, None]


def _apply_partial_rope(
    value: mx.array,
    positions: mx.array,
    rotary_dim: int,
    theta: float,
) -> mx.array:
    if rotary_dim == 0:
        return value
    frequencies = mx.exp(
        -math.log(theta) * mx.arange(0, rotary_dim, 2) / rotary_dim
    )
    angles = positions[..., None] * frequencies
    cosine = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=-1)
    sine = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=-1)
    rotary = value[..., :rotary_dim]
    first, second = mx.split(rotary, 2, axis=-1)
    rotated = mx.concatenate([-second, first], axis=-1)
    return mx.concatenate([rotary * cosine + rotated * sine, value[..., rotary_dim:]], axis=-1)


class QSAIndexer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.index_qk_proj = nn.Linear(
            args.hidden_size,
            (args.indexer_n_heads + args.indexer_kv_heads) * args.indexer_head_dim,
            bias=False,
        )
        self.q_layernorm = GroupRMSNorm(args.indexer_head_dim, None, args.rms_norm_eps)
        self.k_layernorm = GroupRMSNorm(args.indexer_head_dim, None, args.rms_norm_eps)

    def project(
        self,
        hidden: mx.array,
        cache: KVCache | None,
        offset: int,
    ) -> tuple[mx.array, mx.array]:
        projected = self.index_qk_proj(hidden)
        split = self.args.indexer_n_heads * self.args.indexer_head_dim
        query, key = mx.split(projected, [split], axis=-1)
        query = query.reshape(
            hidden.shape[0], hidden.shape[1], self.args.indexer_n_heads,
            self.args.indexer_head_dim,
        )
        key = key.reshape(hidden.shape[0], hidden.shape[1], self.args.indexer_head_dim)
        positions = mx.arange(hidden.shape[1]) + offset
        query = _apply_partial_rope(
            self.q_layernorm(query),
            positions[None, :, None],
            int(self.args.head_dim * self.args.partial_rotary_factor),
            self.args.rope_theta,
        )
        raw = key[:, None]
        if cache is not None:
            raw, _ = cache.update_and_fetch(raw, raw)
        return query, raw[:, 0]


class QSAAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.sparse_sdpa = False
        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * args.head_dim * 2,
            bias=args.attention_bias,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * args.head_dim,
            bias=args.attention_bias,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * args.head_dim,
            bias=args.attention_bias,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * args.head_dim,
            args.hidden_size,
            bias=args.attention_bias,
        )
        self.q_norm = GroupRMSNorm(args.head_dim, None, args.rms_norm_eps)
        self.k_norm = GroupRMSNorm(args.head_dim, None, args.rms_norm_eps)
        self.indexer = QSAIndexer(args)
        self.rope = initialize_rope(
            int(args.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=False,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, hidden: mx.array, cache: CacheList | None) -> mx.array:
        batch, length, _ = hidden.shape
        main_cache = None if cache is None else cache[0]
        index_cache = None if cache is None else cache[1]
        offset = 0 if main_cache is None else main_cache.offset
        projected = self.q_proj(hidden).reshape(
            batch, length, self.args.num_attention_heads, -1
        )
        query, gate = mx.split(projected, 2, axis=-1)
        gate = gate.reshape(batch, length, -1)
        key = self.k_proj(hidden).reshape(
            batch, length, self.args.num_key_value_heads, self.args.head_dim
        )
        value = self.v_proj(hidden).reshape(
            batch, length, self.args.num_key_value_heads, self.args.head_dim
        )
        query = self.rope(self.q_norm(query).transpose(0, 2, 1, 3), offset=offset)
        key = self.rope(self.k_norm(key).transpose(0, 2, 1, 3), offset=offset)
        value = value.transpose(0, 2, 1, 3)
        index_query, raw_index_keys = self.indexer.project(hidden, index_cache, offset)
        if main_cache is not None:
            key, value = main_cache.update_and_fetch(key, value)
        output = self._bounded_attention(
            query, key, value, index_query, raw_index_keys, offset, index_cache
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output * mx.sigmoid(gate))

    def _bounded_attention(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        index_query: mx.array,
        raw_index_keys: mx.array,
        offset: int,
        index_cache=None,
    ) -> mx.array:
        if query.shape[0] != 1:
            raise ValueError("Qwen SSD runtime supports batch size one")
        if self.sparse_sdpa and key.shape[2] <= self.args.indexer_budget:
            from .qwen_tensor_ops import dense_causal_attention
            # The selection is every visible key here. Do not duplicate K/V once
            # per query or build unused index pools. Raw index cache was updated
            # by __call__, so crossing the sparse threshold remains valid.
            return dense_causal_attention(query, key, value, offset)
        ratio = self.args.indexer_compress_ratio
        blocks = raw_index_keys.shape[1] // ratio
        def pool_rows(raw, first_block):
            count = raw.shape[1] // ratio
            pooled = raw.reshape(1, count, ratio, self.args.indexer_head_dim
                ).astype(mx.float32).mean(axis=2).astype(raw.dtype)
            pooled = self.indexer.k_layernorm(pooled)
            if count:
                positions = (mx.arange(count) + first_block) * ratio
                pooled = _apply_partial_rope(pooled, positions[None],
                    int(self.args.head_dim * self.args.partial_rotary_factor), self.args.rope_theta)
            return pooled

        if callable(getattr(index_cache, "pooled", None)):
            pooled = index_cache.pooled(raw_index_keys, ratio, pool_rows)
        else:
            pooled = pool_rows(raw_index_keys[:, :blocks * ratio], 0)
        outputs = []
        query_chunk = 4
        for start in range(0, query.shape[2], query_chunk):
            end = min(start + query_chunk, query.shape[2])
            absolute = mx.arange(start, end) + offset
            if key.shape[2] <= self.args.indexer_budget:
                selected = mx.broadcast_to(
                    mx.arange(key.shape[2])[None], (end - start, key.shape[2])
                )
                selected_valid = mx.ones(selected.shape, dtype=mx.bool_)
            else:
                scores = (
                    index_query[:, start:end].astype(mx.float32)
                    @ pooled.swapaxes(-1, -2).astype(mx.float32)
                )
                scores = mx.maximum(scores, 0).sum(axis=2)[0]
                visible = (
                    mx.arange(blocks)[None] * ratio + ratio - 1
                    <= absolute[:, None]
                )
                scores = mx.where(visible, scores, mx.finfo(scores.dtype).min)
                count = min(self.args.indexer_budget // ratio, blocks)
                chosen = mx.argpartition(-scores, kth=count - 1, axis=-1)[..., :count]
                selected = (
                    chosen[..., None] * ratio + mx.arange(ratio)
                ).reshape(end - start, -1)
                complete_length = ((absolute + 1) // ratio) * ratio
                selected_valid = selected < complete_length[:, None]
                tail_base = (absolute // ratio) * ratio
                tail = tail_base[:, None] + mx.arange(ratio)
                selected = mx.concatenate([selected, tail], axis=-1)
                tail_valid = (tail >= complete_length[:, None]) & (
                    tail <= absolute[:, None]
                )
                selected_valid = mx.concatenate(
                    [selected_valid, tail_valid], axis=-1
                )
            selected = mx.clip(selected, 0, key.shape[2] - 1)
            selected_key = mx.take(key[0].transpose(1, 0, 2), selected, axis=0)
            selected_value = mx.take(value[0].transpose(1, 0, 2), selected, axis=0)
            selected_key = selected_key.transpose(0, 2, 1, 3)
            selected_value = selected_value.transpose(0, 2, 1, 3)
            if self.args.num_attention_heads % selected_key.shape[1] != 0:
                raise ValueError("Qwen query heads must be divisible by KV heads")
            repeats = self.args.num_attention_heads // selected_key.shape[1]
            current_query = query[0, :, start:end].transpose(1, 0, 2)
            causal = selected_valid & (selected <= absolute[:, None])
            if self.sparse_sdpa:
                # Each query is a separate batch; GQA shares its selected KV rows.
                # Keep the baseline indexer, selected cells and causal membership.
                # Different reduction precision can change logits: opt-in only.
                current = mx.fast.scaled_dot_product_attention(
                    current_query[:, :, None, :], selected_key, selected_value,
                    scale=self.args.head_dim**-0.5,
                    mask=causal[:, None, None, :],
                ).squeeze(-2)
                outputs.append(current.transpose(1, 0, 2)[None])
                continue
            grouped_query = current_query.reshape(
                end - start,
                selected_key.shape[1],
                repeats,
                self.args.head_dim,
            )
            scores = (
                grouped_query[..., None, :]
                @ selected_key[:, :, None].swapaxes(-1, -2)
            ).squeeze(-2) * (self.args.head_dim**-0.5)
            scores = mx.where(
                causal[:, None, None, :],
                scores,
                mx.finfo(scores.dtype).min,
            )
            weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(
                query.dtype
            )
            current = (
                weights[..., None, :] @ selected_value[:, :, None]
            ).squeeze(-2).reshape(
                end - start,
                self.args.num_attention_heads,
                self.args.head_dim,
            )
            outputs.append(current.transpose(1, 0, 2)[None])
        return mx.concatenate(outputs, axis=2)


class MLP(nn.Module):
    def __init__(self, dimensions: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(dimensions, intermediate, bias=False)
        self.up_proj = nn.Linear(dimensions, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, dimensions, bias=False)

    def __call__(self, value: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(value)) * self.up_proj(value))


def _qmm(value: mx.array, weight: mx.array, scales: mx.array) -> mx.array:
    return mx.quantized_matmul(
        value,
        weight,
        scales,
        transpose=True,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )


class StreamingExperts(nn.Module):
    def __init__(self, layer: int, cache: ExpertCache):
        super().__init__()
        self.layer = layer
        self.cache = cache
        self.grouped_prefill = False
        self.wave_slots = 0
        self.wave_count = 0
        self.wave_pairs = 0
        self.wave_peak_experts = 0

    @staticmethod
    def _one(value: mx.array, weights: QwenExpertWeights) -> mx.array:
        projected = _qmm(value, weights.gate_up, weights.gate_up_scales)
        gate, up = mx.split(projected, 2, axis=-1)
        return _qmm(nn.silu(gate) * up, weights.down, weights.down_scales)

    def _waves(self, value: mx.array, selected: np.ndarray) -> mx.array:
        from .qwen_expert_waves import expert_waves
        capacity = min(self.wave_slots, self.cache.slots)
        flat = selected.reshape(-1)
        flat_value = value.reshape(-1, value.shape[-1])
        outputs, order = [], []
        for wave in expert_waves(selected, capacity, self.cache.model.expert_count):
            check_cancelled()
            positions = np.concatenate([positions for _, positions in wave.groups])
            # Retain repeated IDs for route-frequency accounting in ExpertCache.
            resident = self.cache.get_many(self.layer, flat[positions].tolist())
            current = []
            try:
                for expert, pair_positions in wave.groups:
                    check_cancelled()
                    weights = resident.individual_weights[resident.slots[expert]]
                    source = mx.take(flat_value, mx.array(pair_positions // selected.shape[-1]), axis=0)
                    current.append(self._one(source, weights))
                # Slot-backed weights MUST finish being read before the next wave
                # can evict/reuse them. async_eval alone is not a lifetime fence.
                mx.eval(*current)
            except BaseException:
                mx.synchronize()
                raise
            outputs.extend(current)
            order.append(positions)
            self.wave_count += 1
            self.wave_pairs += positions.size
            self.wave_peak_experts = max(self.wave_peak_experts, len(wave.groups))
        grouped = mx.concatenate(outputs, axis=0)
        inverse = np.argsort(np.concatenate(order))
        return mx.take(grouped, mx.array(inverse), axis=0).reshape(*selected.shape, -1)

    def __call__(self, value: mx.array, indices: mx.array) -> mx.array:
        # Whole-layer batching is disabled by QwenSupport when waves are enabled.
        if self.wave_slots and value.size // value.shape[-1] > 1:
            return self._waves(value, np.asarray(indices, dtype=np.int32))
        batched = self.cache.current_batched(self.layer)
        if isinstance(batched, QwenBatchedExperts):
            if getattr(self.cache, "route_cache_enabled", False):
                self.cache.observe_batched_routes(self.layer, np.asarray(indices, dtype=np.int32))
            source = mx.expand_dims(value, (-2, -3))
            grouped = self.grouped_prefill and indices.size >= 64
            selected = indices
            if grouped:
                source, selected, inverse = _gather_sort(source, indices)
            # Preserve the tested QMM path even when expert indices are sorted.
            projected = mx.gather_qmm(
                source,
                batched.gate_up,
                batched.gate_up_scales,
                rhs_indices=selected,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
                sorted_indices=False,
            )
            gate, up = mx.split(projected, 2, axis=-1)
            output = mx.gather_qmm(
                nn.silu(gate) * up,
                batched.down,
                batched.down_scales,
                rhs_indices=selected,
                transpose=True,
                group_size=32,
                bits=4,
                mode="mxfp4",
                sorted_indices=False,
            )
            self.cache.record_gather_qmm(2)
            if grouped:
                output = _scatter_unsort(output, inverse, indices.shape)
            return output.squeeze(-2)

        selected = np.asarray(indices, dtype=np.int32)
        if value.size // value.shape[-1] == 1 and getattr(self.cache, "ready_expert_decode", False):
            outputs = {}
            ready = self.cache.iter_ready(self.layer, selected.reshape(-1).tolist())
            try:
                for expert, weights in ready:
                    output = self._one(value, weights)
                    mx.async_eval(output)
                    outputs[expert] = output
            finally:
                close = getattr(ready, "close", None)
                if close is not None:
                    close()
            return mx.stack([outputs[int(expert)] for expert in selected.reshape(-1)], axis=-2)
        resident = self.cache.get_many(self.layer, selected.reshape(-1).tolist())
        flat = selected.reshape(-1)
        if value.size // value.shape[-1] == 1:
            route = flat.tolist()
            if len(set(route)) == len(route):
                # Top-k routing has unique experts. Keep each GEMV's [1, H]
                # shape, but avoid sorting, taking and unsorting one-row groups.
                source = value.reshape(1, value.shape[-1])
                output = [self._one(source, resident.individual_weights[resident.slots[expert]])
                          for expert in route]
                return mx.stack(output, axis=1).reshape(*selected.shape, -1)
        order = np.argsort(flat, kind="stable")
        boundaries = np.flatnonzero(np.diff(flat[order])) + 1
        flat_value = value.reshape(-1, value.shape[-1])
        outputs = []
        for positions in np.split(order, boundaries):
            expert = int(flat[positions[0]])
            weights = resident.individual_weights[resident.slots[expert]]
            source = mx.take(
                flat_value,
                mx.array(positions // selected.shape[-1]),
                axis=0,
            )
            outputs.append(self._one(source, weights))
        grouped = mx.concatenate(outputs, axis=0)
        restored = mx.take(grouped, mx.array(np.argsort(order)), axis=0)
        return restored.reshape(*selected.shape, -1)


class SparseMoE(nn.Module):
    def __init__(self, args: ModelArgs, layer: int, cache: ExpertCache):
        super().__init__()
        self.gate = nn.Linear(args.hidden_size, args.num_experts, bias=False)
        self.experts = StreamingExperts(layer, cache)
        self.shared_expert = MLP(args.hidden_size, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.cache = cache
        self.layer = layer

    def __call__(self, value: mx.array) -> mx.array:
        probabilities = mx.softmax(self.gate(value), axis=-1, precise=True)
        indices = mx.argpartition(probabilities, kth=-self.top_k, axis=-1)[..., -self.top_k :]
        scores = mx.take_along_axis(probabilities, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        if getattr(self.cache, "route_trace_enabled", False):
            self.cache.record_routes(self.layer, np.asarray(indices, dtype=np.int32))
        shared = mx.sigmoid(self.shared_expert_gate(value)) * self.shared_expert(value)
        routed = self.experts(value, indices)
        routed = (routed * scores[..., None].astype(routed.dtype)).sum(axis=-2)
        return routed + shared


class DecoderLayer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer: int,
        cache: ExpertCache,
        ngram_store: NGramStore,
    ):
        super().__init__()
        self.layer_type = args.layer_types[layer]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(args)
            self.linear_attn.norm = RMSNormGated(
                args.linear_value_head_dim, args.rms_norm_eps
            )
        else:
            self.self_attn = QSAAttention(args)
        self.mlp = SparseMoE(args, layer, cache)
        self.ple = PLELayer(args, ngram_store) if layer + 1 in args.ple_layer_ids else None
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def __call__(
        self,
        hidden: mx.array,
        input_ids: mx.array,
        mask: mx.array | None,
        cache,
    ) -> mx.array:
        if self.ple is not None:
            hidden = hidden + self.ple(hidden, input_ids, cache)
        mixed, residual, injection = self.attn_hyper_connection(hidden)
        if self.layer_type == "linear_attention":
            result = self.linear_attn(mixed, mask, cache)
        else:
            result = self.self_attn(mixed, cache)
        hidden = self.attn_hyper_connection.inject(residual, result, injection)
        mixed, residual, injection = self.mlp_hyper_connection(hidden)
        result = self.mlp(mixed)
        return self.mlp_hyper_connection.inject(residual, result, injection)


class TextModel(nn.Module):
    def __init__(self, args: ModelArgs, cache: ExpertCache, ngram_store: NGramStore):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, layer, cache, ngram_store)
            for layer in range(args.num_hidden_layers)
        ]
        self.hyper_connection_mixer = GatedResidual(args, combine=False)

    def hidden_states(self, input_ids: mx.array, cache=None) -> mx.array:
        hidden = self.embed_tokens(input_ids)
        hidden = mx.tile(hidden, (1, 1, self.args.hc_count))
        if cache is None:
            cache = [None] * len(self.layers)
        mask = create_ssm_mask(hidden[..., : self.args.hidden_size], cache[0])
        for layer, layer_cache in zip(self.layers, cache):
            check_cancelled()
            hidden = layer(hidden, input_ids, mask, layer_cache)
        return hidden

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        return self.hyper_connection_mixer(self.hidden_states(input_ids, cache))


class Model(nn.Module):
    def __init__(self, args: ModelArgs, cache: ExpertCache, ngram_store: NGramStore):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = TextModel(args, cache, ngram_store)
        self.ngram_store = ngram_store
        self._expert_cache = cache
        self.quantize_kv = False
        self.quantize_index = False
        self.pooled_index_cache = False
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        return self.lm_head(self.model(input_ids, cache))

    def forward_with_hidden(
        self, input_ids: mx.array, cache=None
    ) -> tuple[mx.array, mx.array]:
        hidden = self.model.hidden_states(input_ids, cache)
        logits = self.lm_head(self.model.hyper_connection_mixer(hidden))
        return logits, hidden

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        from .qwen_quantized_cache import QSAQuantizedCache
        from .qwen_pooled_cache import QSAPooledIndexCache, QSAPooledQuantizedIndexCache
        def index_cache():
            if self.pooled_index_cache:
                return (QSAPooledQuantizedIndexCache(4, self.args.indexer_head_dim)
                        if self.quantize_index else QSAPooledIndexCache())
            return QSAQuantizedCache(4, self.args.indexer_head_dim) if self.quantize_index else KVCache()
        return [
            ArraysCache(size=4)
            if layer.layer_type == "linear_attention"
            else CacheList(QSAQuantizedCache(8, self.args.head_dim) if self.quantize_kv else KVCache(),
                           index_cache())
            for layer in self.layers
        ]

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}
        norm_suffixes = (
            ".hc_norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".q_layernorm.weight",
            ".k_layernorm.weight",
            ".norm_key.weight",
            ".norm_query.weight",
            ".norm_conv.weight",
        )
        for key, value in weights.items():
            if key.startswith("model.language_model."):
                key = "model." + key.removeprefix("model.language_model.")
            if key.startswith("model.visual.") or key.startswith("mtp."):
                continue
            if "ngram_embedding.shard_" in key:
                continue
            if key.endswith("ngram_embedding.weight_scale"):
                continue
            if key.endswith("ngram_heads_offsets") or key.endswith("ngram_heads_vocab_sizes"):
                continue
            if "conv1d.weight" in key and value.ndim == 3 and value.shape[-1] != 1:
                value = value.moveaxis(2, 1)
            if key.endswith(norm_suffixes):
                value = value + 1
            sanitized[key] = value
        return sanitized


class MTPModel(nn.Module):
    """One checkpoint-faithful Qwen MTP layer without speculative control flow."""

    def __init__(self, args: ModelArgs, cache: ExpertCache):
        super().__init__()
        self.expert_cache = cache
        self.args = replace(
            args,
            num_hidden_layers=1,
            layer_types=("full_attention",),
            ple_layer_ids=(),
        )
        wide_size = self.args.hc_count * self.args.hidden_size
        self.pre_fc_norm_embedding = GroupRMSNorm(
            self.args.hidden_size, None, self.args.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GroupRMSNorm(
            wide_size, None, self.args.rms_norm_eps
        )
        self.fc_embedding = nn.Linear(
            self.args.hidden_size, self.args.hidden_size, bias=False
        )
        self.fc_hidden = nn.Linear(
            self.args.hidden_size, self.args.hidden_size, bias=False
        )
        self.layers = [DecoderLayer(self.args, 0, cache, None)]
        self.hyper_connection_mixer = GatedResidual(self.args, combine=False)

    def __call__(
        self,
        target_hidden: mx.array,
        next_token_ids: mx.array,
        embedding_weight: mx.array,
        lm_head_weight: mx.array,
        cache: CacheList | None,
    ) -> tuple[mx.array, mx.array]:
        wide_hidden = self.advance(target_hidden, next_token_ids, embedding_weight, cache)
        output = self.hyper_connection_mixer(wide_hidden)
        logits = output @ lm_head_weight.T
        return logits, wide_hidden

    def advance(
        self,
        target_hidden: mx.array,
        next_token_ids: mx.array,
        embedding_weight: mx.array,
        cache: CacheList | None,
    ) -> mx.array:
        """Advance native MTP state without unused mixer/vocabulary projection."""
        expected = self.args.hc_count * self.args.hidden_size
        if target_hidden.shape[-1] != expected:
            raise ValueError("Qwen MTP target hidden state has an invalid width")
        if target_hidden.shape[:2] != next_token_ids.shape:
            raise ValueError("Qwen MTP hidden state and token shape do not match")
        embedded = mx.take(embedding_weight, next_token_ids, axis=0)
        embedded = self.fc_embedding(self.pre_fc_norm_embedding(embedded))
        hidden = self.pre_fc_norm_hidden(target_hidden).reshape(
            *target_hidden.shape[:-1], self.args.hc_count, self.args.hidden_size
        )
        hidden = self.fc_hidden(hidden)
        mixed = (hidden + embedded[..., None, :]).reshape(*target_hidden.shape)
        wide_hidden = self.layers[0](mixed, next_token_ids, None, cache)
        return wide_hidden

    def make_cache(self) -> CacheList:
        return CacheList(KVCache(), KVCache())

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        sanitized = {}
        norm_suffixes = (
            ".hc_norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            ".q_layernorm.weight",
            ".k_layernorm.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
        )
        for key, value in weights.items():
            if not key.startswith("mtp."):
                continue
            key = key.removeprefix("mtp.")
            if key.endswith(norm_suffixes):
                value = value + 1
            sanitized[key] = value
        return sanitized


def generate_mtp_tokens(
    prompt: list[int],
    main_model: Model,
    mtp_model: MTPModel,
    target_cache,
    *,
    max_tokens: int,
    prefill_step_size: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None,
    prefilled_hidden: mx.array | None = None,
    record_round: Callable[[int, int, float, float, float, bool], None] | None = None,
    draft_tokens: int = 5,
    zero_acceptance_limit: int = 1,
) -> Iterator[tuple[int, bool]]:
    """Yield tokens from exact target-distribution MTP verification."""
    from .qwen_mtp_policy import MTPDraftPolicy
    policy = MTPDraftPolicy(draft_tokens, zero_acceptance_limit)
    if not prompt or max_tokens < 1:
        return
    if prefill_step_size < 1:
        raise ValueError("Qwen MTP Prefill step size must be positive")

    from .dspark import DraftResult, _sample, _verify, sampling_logprobs
    from .model import _fork_prompt_cache, eval_prompt_cache

    logits_processors = logits_processors or []
    sampling_tokens = [prompt[-1]]

    def adjusted_logprobs(logits: mx.array, tokens: list[int]) -> mx.array:
        for processor in logits_processors:
            logits = processor(mx.array(tokens, dtype=mx.int32), logits)
        return sampling_logprobs(
            logits,
            temperature,
            top_p,
            top_k,
            min_p,
        )[0]

    embedding_weight = main_model.model.embed_tokens.weight
    lm_head_weight = main_model.lm_head.weight
    mtp_cache = mtp_model.make_cache()
    cache_slots = int(getattr(mtp_model.expert_cache, "slots", 0))
    selected_experts = int(getattr(mtp_model.args, "num_experts_per_tok", 0))
    mtp_prefill_step = (
        max(1, cache_slots // selected_experts)
        if cache_slots and selected_experts
        else prefill_step_size
    )

    def advance_mtp(hidden: mx.array, token_ids: mx.array) -> None:
        advance = getattr(mtp_model, "advance", None)
        if callable(advance):
            state_hidden = advance(hidden, token_ids, embedding_weight, mtp_cache)
        else:
            _, state_hidden = mtp_model(
                hidden, token_ids, embedding_weight, lm_head_weight, mtp_cache)
        # Evaluate the whole layer before shared expert slots may be recycled;
        # evaluating only cache arrays would not fence the trailing MoE work.
        eval_prompt_cache([mtp_cache], state_hidden)

    def prefill_mtp(paired_hidden: mx.array, paired_tokens: mx.array) -> None:
        for start in range(0, paired_tokens.shape[1], mtp_prefill_step):
            check_cancelled()
            end = min(start + mtp_prefill_step, paired_tokens.shape[1])
            advance_mtp(paired_hidden[:, start:end], paired_tokens[:, start:end])

    if prefilled_hidden is not None:
        if prefilled_hidden.shape[1] != len(prompt) - 1:
            raise ValueError("Qwen MTP Prefill hidden state length does not match")
        mtp_prefill_window = min(prefilled_hidden.shape[1], mtp_prefill_step)
        prefill_mtp(
            prefilled_hidden[:, -mtp_prefill_window:],
            mx.array([prompt[1:]], dtype=mx.int32)[:, -mtp_prefill_window:],
        )
        final_logits, final_hidden = main_model.forward_with_hidden(
            mx.array([[prompt[-1]]], dtype=mx.int32),
            target_cache,
        )
        eval_prompt_cache(target_cache, final_logits, final_hidden)
        processed = len(prompt)
    else:
        final_logits = final_hidden = None
        processed = 0
    previous_hidden = None
    while processed < len(prompt):
        check_cancelled()
        count = min(prefill_step_size, len(prompt) - processed)
        token_ids = mx.array([prompt[processed : processed + count]], dtype=mx.int32)
        final_logits, final_hidden = main_model.forward_with_hidden(
            token_ids,
            target_cache,
        )
        if previous_hidden is None:
            paired_hidden = final_hidden[:, :-1]
            paired_tokens = token_ids[:, 1:]
        else:
            paired_hidden = mx.concatenate(
                [previous_hidden, final_hidden[:, :-1]],
                axis=1,
            )
            paired_tokens = token_ids
        if paired_tokens.shape[1]:
            prefill_mtp(paired_hidden, paired_tokens)
            mx.eval(final_hidden)
        else:
            mx.eval(final_logits, final_hidden)
        previous_hidden = final_hidden[:, -1:]
        processed += count

    assert final_logits is not None and final_hidden is not None
    final_logprobs = adjusted_logprobs(final_logits[:, -1], sampling_tokens)
    anchor = _sample(final_logprobs, temperature)
    predecessor_hidden = final_hidden[:, -1:]
    yield anchor, False
    sampling_tokens.append(anchor)
    generated = 1

    while generated < max_tokens:
        check_cancelled()
        remaining = max_tokens - generated
        if remaining == 1:
            logits, predecessor_hidden = main_model.forward_with_hidden(
                mx.array([[anchor]], dtype=mx.int32),
                target_cache,
            )
            eval_prompt_cache(target_cache, logits, predecessor_hidden)
            logprobs = adjusted_logprobs(logits[:, -1], sampling_tokens)
            anchor = _sample(logprobs, temperature)
            yield anchor, False
            sampling_tokens.append(anchor)
            return

        draft_limit = policy.limit(remaining)
        checkpoint = mtp_cache.size()
        draft_started = time.perf_counter()
        draft_tokens: list[int] = []
        draft_logprobs: list[mx.array] = []
        draft_hidden = predecessor_hidden
        draft_input = anchor
        for _ in range(draft_limit):
            check_cancelled()
            draft_logits, draft_hidden = mtp_model(
                draft_hidden,
                mx.array([[draft_input]], dtype=mx.int32),
                embedding_weight,
                lm_head_weight,
                mtp_cache,
            )
            eval_prompt_cache([mtp_cache], draft_logits, draft_hidden)
            draft_distribution = adjusted_logprobs(
                draft_logits[:, -1],
                sampling_tokens + draft_tokens,
            )
            draft_input = _sample(draft_distribution, temperature)
            draft_tokens.append(draft_input)
            draft_logprobs.append(draft_distribution)
        draft_seconds = time.perf_counter() - draft_started

        check_cancelled()
        verification_started = time.perf_counter()
        verified_cache, copied = _fork_prompt_cache(target_cache)
        if copied:
            mx.eval(*copied)
        verified_logits, verified_hidden = main_model.forward_with_hidden(
            mx.array([[anchor, *draft_tokens]], dtype=mx.int32),
            verified_cache,
        )
        eval_prompt_cache(verified_cache, verified_logits, verified_hidden)
        target_logprobs = mx.stack(
            [
                adjusted_logprobs(
                    verified_logits[:, index],
                    sampling_tokens + draft_tokens[:index],
                )
                for index in range(len(draft_tokens) + 1)
            ]
        )
        mx.eval(target_logprobs)
        draft = DraftResult(
            tokens=draft_tokens,
            logprobs=draft_logprobs,
            confidence=[1.0] * len(draft_tokens),
            seconds=draft_seconds,
        )
        accepted, next_token, _ = _verify(draft, target_logprobs, temperature)
        verification_seconds = time.perf_counter() - verification_started
        replay_seconds = 0.0

        if accepted == len(draft_tokens):
            target_cache[:] = verified_cache
            predecessor_hidden = verified_hidden[:, -1:]
            advance_mtp(draft_hidden, mx.array([[draft_tokens[-1]]], dtype=mx.int32))
        else:
            rollback_mtp_cache(mtp_cache, checkpoint + accepted + 1)
            replay_started = time.perf_counter()
            replay_logits = replay_hidden = None
            for input_token in [anchor, *draft_tokens[:accepted]]:
                check_cancelled()
                replay_logits, replay_hidden = main_model.forward_with_hidden(
                    mx.array([[input_token]], dtype=mx.int32),
                    target_cache,
                )
                eval_prompt_cache(target_cache, replay_logits, replay_hidden)
            assert replay_logits is not None and replay_hidden is not None
            predecessor_hidden = replay_hidden[:, -1:]
            replay_seconds = time.perf_counter() - replay_started

        fallback = policy.observe(accepted)
        if record_round is not None:
            record_round(
                len(draft_tokens),
                accepted,
                draft_seconds,
                verification_seconds,
                replay_seconds,
                fallback,
            )

        for token in draft_tokens[:accepted]:
            if generated >= max_tokens:
                return
            yield token, True
            sampling_tokens.append(token)
            generated += 1
        if generated >= max_tokens:
            return
        anchor = next_token
        yield anchor, False
        sampling_tokens.append(anchor)
        generated += 1

        if fallback:
            # Default remains one zero-acceptance round. A higher opt-in
            # threshold permits retries, without changing rejection sampling.
            while generated < max_tokens:
                check_cancelled()
                logits, predecessor_hidden = main_model.forward_with_hidden(
                    mx.array([[anchor]], dtype=mx.int32),
                    target_cache,
                )
                eval_prompt_cache(target_cache, logits, predecessor_hidden)
                logprobs = adjusted_logprobs(logits[:, -1], sampling_tokens)
                anchor = _sample(logprobs, temperature)
                yield anchor, False
                sampling_tokens.append(anchor)
                generated += 1
            return


def load(
    installed: InstalledModel,
    config,
    raw_config: dict[str, Any],
    weights: dict[str, mx.array],
    read_limiter=None,
):
    if installed.ngram is None:
        raise ValueError("Qwen installed model has no N-gram descriptor")
    args = ModelArgs.from_dict(raw_config["text_config"])
    cache = ExpertCache(
        installed,
        config.slots,
        config.read_workers,
        config.prefetch_read_workers,
        route_trace_path=config.expert_route_trace,
        ready_expert_decode=config.ready_expert_decode,
        read_limiter=read_limiter,
        page_cache_probe=getattr(config, "expert_page_cache_probe", False),
        file_cache_policy=getattr(config, "expert_file_cache_policy", "cached"),
        separate_prefill_io=getattr(config, "separate_prefill_io", True),
        eviction_policy=getattr(config, "expert_eviction_policy", "lfu"),
    )
    ngram_store = None
    try:
        if getattr(config, "qwen_phase_memory", False):
            cache.enable_phase_memory()
        scale_name = (
            "model.language_model.layers.1.ple.ple_embedding."
            "ngram_embedding.weight_scale"
        )
        if scale_name not in weights or weights[scale_name].shape != (1,):
            raise ValueError("Qwen common tensors have no valid N-gram weight scale")
        weight_scale = float(weights[scale_name].astype(mx.float32).item())
        ngram_store = NGramStore(
            installed.root / installed.ngram.file,
            installed.ngram,
            weight_scale,
            optimized=getattr(config, "qwen_ngram_lookup_optimized", False),
            io_backend=getattr(config, "qwen_ngram_io", "mmap"),
            cache_bytes=getattr(config, "qwen_ngram_cache_bytes", 0),
        )
        model = Model(args, cache, ngram_store)
        model.quantize_kv = config.qwen_quantized_kv
        model.quantize_index = config.qwen_quantized_index
        model.pooled_index_cache = getattr(config, "qwen_pooled_index_cache", False)
        if getattr(config, "qwen_compile_tensor_ops", False):
            for _, module in model.named_modules():
                if isinstance(module, (GroupRMSNorm, GatedResidual)):
                    module.compiled = True
        grouped_prefill = bool(
            getattr(config, "qwen_grouped_experts", True)
            and not getattr(config, "mtp_enabled", False)
        )
        for layer in model.model.layers:
            layer.mlp.experts.grouped_prefill = grouped_prefill
            layer.mlp.experts.wave_slots = getattr(config, "qwen_expert_wave_slots", 0)
            attention = getattr(layer, "self_attn", None)
            if isinstance(attention, QSAAttention):
                attention.sparse_sdpa = getattr(config, "qwen_sparse_sdpa", False)
        model.eval()
        model.load_weights(list(model.sanitize(weights).items()), strict=True)
        model.dspark = None
        mx.eval(model.parameters())
        from .ane_prefill import install_qwen_ane_prefill

        model.ane_prefill = install_qwen_ane_prefill(
            model,
            getattr(config, "ane_prefill", True),
            getattr(config, "ane_prefill_ratio", 0.0),
        )
        return model, cache
    except BaseException:
        try:
            cache.close()
        finally:
            if ngram_store is not None:
                ngram_store.close()
        raise
