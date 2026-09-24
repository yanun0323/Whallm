"""Native MiMo weight conversion for a single Apple Silicon device.

The pinned checkpoint stores QKV as [Q0,K0,V0,Q1,K1,V1,...] for TP=4.
FP8 scales are separately block-padded within each source TP shard. Applying
one global 128-row scale grid silently assigns incorrect scales to GA rows.

Only common tensors are expanded to BF16. Routed experts remain native MXFP4
and use the existing Whallm quantized operators, without full BF16 staging.
"""
from __future__ import annotations

import math

import mlx.core as mx


def decode_fp8_blocks(weight: mx.array, scales: mx.array, *, source_shards: int = 1) -> mx.array:
    """Decode one F8_E4M3 byte tensor, respecting source-local 128x128 blocks."""
    if weight.ndim != 2 or scales.ndim != 2 or weight.dtype != mx.uint8:
        raise ValueError("FP8 weights must be a 2-D uint8 tensor with 2-D scales")
    if scales.dtype != mx.float32:
        raise ValueError("FP8 block scales must be float32")
    rows, cols = weight.shape
    if (type(source_shards) is not int or source_shards < 1 or rows < 1 or cols < 1
            or rows % source_shards):
        raise ValueError("FP8 tensor cannot be divided into source shards")
    local_rows = rows // source_shards
    scale_rows, scale_cols = math.ceil(local_rows / 128), math.ceil(cols / 128)
    if scales.shape != (source_shards * scale_rows, scale_cols):
        raise ValueError("FP8 scale shape does not match source-shard block padding")
    # MLX's byte converter maps E4M3FN's reserved NaN codes to finite values;
    # reject them before conversion rather than silently accepting bad weights.
    if mx.any((weight & 127) == 127).item():
        raise ValueError("FP8 checkpoint contains reserved NaN codes")
    if mx.any(~mx.isfinite(scales) | (scales < 0)).item():
        raise ValueError("FP8 checkpoint contains invalid block scales")
    parts = []
    for shard in range(source_shards):
        values = mx.from_fp8(weight[shard * local_rows:(shard + 1) * local_rows], dtype=mx.float32)
        local_scales = scales[shard * scale_rows:(shard + 1) * scale_rows]
        expanded = mx.repeat(mx.repeat(local_scales, 128, axis=0), 128, axis=1)
        parts.append((values * expanded[:local_rows, :cols]).astype(mx.bfloat16))
    return mx.concatenate(parts, axis=0)


def decode_qkv(
    weight: mx.array,
    scales: mx.array,
    *,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    value_dim: int,
    source_shards: int,
) -> mx.array:
    """Return canonical [all Q, all K, all V] BF16 weights, without requantizing.

Source topology is required, never inferred from the runtime GPU count. The
pinned MiMo checkpoint has no replicated KV heads (GA 4, SWA 8, source TP 4).
"""
    dims = (query_heads, kv_heads, head_dim, value_dim, source_shards)
    if any(type(n) is not int or n < 1 for n in dims):
        raise ValueError("QKV dimensions and source shards must be positive integers")
    if query_heads % source_shards or kv_heads % source_shards or query_heads % kv_heads:
        raise ValueError("unsupported QKV source topology (replicated KV is not supported)")
    q, k, v = (query_heads // source_shards * head_dim,
               kv_heads // source_shards * head_dim,
               kv_heads // source_shards * value_dim)
    if weight.ndim != 2 or weight.shape[0] != source_shards * (q + k + v):
        raise ValueError("QKV weight shape does not match source head dimensions")
    decoded = decode_fp8_blocks(weight, scales, source_shards=source_shards)
    shards = mx.split(decoded, source_shards, axis=0)
    parts = [mx.split(shard, [q, q + k], axis=0) for shard in shards]
    return mx.concatenate([parts[s][projection]
                           for projection in range(3) for s in range(source_shards)], axis=0)


def native_expert_views(weight: mx.array, scales: mx.array) -> tuple[mx.array, mx.array]:
    """Reinterpret packed native bytes for MLX QMM. No dequantize or requantize."""
    if (weight.dtype != mx.uint8 or scales.dtype != mx.uint8
            or weight.ndim != 2 or scales.ndim != 2
            or weight.shape[0] < 1 or weight.shape[1] < 16 or weight.shape[1] % 16
            or scales.shape != (weight.shape[0], weight.shape[1] // 16)):
        raise ValueError("invalid native MXFP4 weight/scale layout")
    return weight.view(mx.uint32), scales
