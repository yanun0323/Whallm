"""Opt-in direct-indexed split-K attention for narrow Qwen QSA decode/verify.

Original implementation inspired by direct sparse-cache access in oMLX and
Rapid-MLX. No upstream kernel bodies are copied. Selection and causality are
provided by the existing QSA indexer. FP32 online-softmax reductions are NOT
bit-identical to the gathered BF16 path. No cache tensors are mutated here.
"""
from __future__ import annotations

from functools import lru_cache

import mlx.core as mx

SPLITS = 32
MAX_QUERIES = 8


def supported(query, key, value, selected, valid) -> bool:
    return bool(
        mx.metal.is_available() and mx.default_device() == mx.gpu
        and query.ndim == key.ndim == value.ndim == 4
        and query.shape[0] == key.shape[0] == value.shape[0] == 1
        and key.shape == value.shape and key.shape[2] > 0
        and 1 <= query.shape[2] <= MAX_QUERIES
        and query.shape[3] == key.shape[3] and query.shape[3] in (32, 64, 128, 256)
        and key.shape[1] > 0 and query.shape[1] % key.shape[1] == 0
        and query.dtype == key.dtype == value.dtype
        and query.dtype in (mx.float32, mx.float16, mx.bfloat16)
        and selected.ndim == valid.ndim == 2 and selected.shape == valid.shape
        and selected.shape[0] == query.shape[2] and selected.shape[1] > 0
        and selected.dtype == mx.int32 and valid.dtype == mx.bool_
    )


@lru_cache(maxsize=1)
def _partials_kernel():
    return mx.fast.metal_kernel(
        name="whallm_qsa_indexed_partials",
        input_names=["q", "k", "v", "ids", "valid", "dims", "scale"],
        output_names=["partials"],
        ensure_row_contiguous=False,
        source=r"""
        const uint lane = thread_index_in_simdgroup;
        const uint query_head = threadgroup_position_in_grid.y;
        const uint h = query_head % HQ;
        const uint t = query_head / HQ;
        const uint split = threadgroup_position_in_grid.z;
        const uint kh = h / (HQ / HK);
        const int width = dims[0], kv_length = dims[1], offset = dims[2];
        constexpr int N = D / 32;
        float qr[N], accum[N];
        for (int i = 0; i < N; ++i) {
            const uint d = lane + 32 * i;
            qr[i] = float(q[h * q_strides[1] + t * q_strides[2] + d * q_strides[3]]);
            accum[i] = 0.0f;
        }
        float maximum = -INFINITY, denominator = 0.0f;
        for (int r = int(split); r < width; r += SPLITS) {
            const int row = ids[t * ids_strides[0] + r * ids_strides[1]];
            const bool visible = valid[t * valid_strides[0] + r * valid_strides[1]]
                && row >= 0 && row < kv_length && row <= offset + int(t);
            if (!visible) continue;
            float dot = 0.0f;
            for (int i = 0; i < N; ++i) {
                const uint d = lane + 32 * i;
                const size_t pos = kh * k_strides[1] + size_t(row) * k_strides[2]
                    + d * k_strides[3];
                dot += qr[i] * float(k[pos]);
            }
            const float score = simd_sum(dot) * scale;
            const float next_max = metal::max(maximum, score);
            const float old_scale = metal::exp(maximum - next_max);
            const float weight = metal::exp(score - next_max);
            denominator = denominator * old_scale + weight;
            for (int i = 0; i < N; ++i) {
                const uint d = lane + 32 * i;
                const size_t pos = kh * v_strides[1] + size_t(row) * v_strides[2]
                    + d * v_strides[3];
                accum[i] = accum[i] * old_scale + weight * float(v[pos]);
            }
            maximum = next_max;
        }
        const size_t base = (size_t(query_head) * SPLITS + split) * (D + 2);
        for (int i = 0; i < N; ++i) partials[base + lane + 32 * i] = accum[i];
        if (lane == 0) {
            partials[base + D] = maximum;
            partials[base + D + 1] = denominator;
        }
        """,
    )


@lru_cache(maxsize=1)
def _merge_kernel():
    return mx.fast.metal_kernel(
        name="whallm_qsa_indexed_merge",
        input_names=["partials", "length"], output_names=["out"],
        source=r"""
        const uint lane = thread_index_in_simdgroup;
        const uint query_head = threadgroup_position_in_grid.y;
        const uint h = query_head % HQ, t = query_head / HQ;
        const size_t first = size_t(query_head) * SPLITS * (D + 2);
        float maximum = -INFINITY;
        for (int s = 0; s < SPLITS; ++s)
            maximum = metal::max(maximum, partials[first + s * (D + 2) + D]);
        float denom = 0.0f;
        constexpr int N = D / 32;
        float accum[N];
        for (int i = 0; i < N; ++i) accum[i] = 0.0f;
        for (int s = 0; s < SPLITS; ++s) {
            const size_t base = first + s * (D + 2);
            const float sum = partials[base + D + 1];
            if (sum == 0.0f) continue;
            const float factor = metal::exp(partials[base + D] - maximum);
            denom += factor * sum;
            for (int i = 0; i < N; ++i)
                accum[i] += factor * partials[base + lane + 32 * i];
        }
        for (int i = 0; i < N; ++i) {
            const size_t dest = (size_t(h) * length[0] + t) * D + lane + 32 * i;
            out[dest] = T(denom > 0.0f ? accum[i] / denom : 0.0f);
        }
        """,
    )


def indexed_attention(query, key, value, selected, valid, offset: int):
    """Return None for unsupported layouts before dispatch; never retry GPU errors.

    Tensor strides are passed directly, so sliced KV-cache views are not copied
    to a token-major full-cache temporary. Outputs and scratch are newly owned.
    An all-masked row returns zero, and invalid/future row IDs are never read.
    """
    if not supported(query, key, value, selected, valid):
        return None
    if type(offset) is not int or offset < 0 or offset + query.shape[2] > key.shape[2]:
        raise ValueError("QSA query offsets must be inside the KV prefix")
    heads, length, dim = query.shape[1:]
    params = mx.array([selected.shape[1], key.shape[2], offset], dtype=mx.int32)
    partials, = _partials_kernel()(
        inputs=[query, key, value, selected, valid, params, dim ** -0.5],
        template=[("D", dim), ("HQ", heads), ("HK", key.shape[1]),
                  ("SPLITS", SPLITS)],
        grid=(32, heads * length, SPLITS), threadgroup=(32, 1, 1),
        output_shapes=[(length * heads, SPLITS, dim + 2)],
        output_dtypes=[mx.float32],
    )
    out, = _merge_kernel()(
        inputs=[partials, mx.array([length], dtype=mx.int32)],
        template=[("D", dim), ("HQ", heads), ("SPLITS", SPLITS), ("T", query.dtype)],
        grid=(32, heads * length, 1), threadgroup=(32, 1, 1),
        output_shapes=[query.shape], output_dtypes=[query.dtype],
    )
    return out
