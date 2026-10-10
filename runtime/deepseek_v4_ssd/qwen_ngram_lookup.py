"""Opt-in row deduplication and FP8 lookup-table decoding for Qwen.

Uses the same NumPy arithmetic as the baseline, including scale application
before the MLX/bfloat16 cast. This is not a speculative SSD prefetcher.
"""
import numpy as np
import mlx.core as mx


def fp8_table(weight_scale):
    values = np.arange(256, dtype=np.uint8)
    exponent = (values >> 3) & 0x0F
    mantissa = values & 0x07
    sign = np.where(values & 0x80, -1.0, 1.0)
    normal = np.ldexp(1.0 + mantissa.astype(np.float32) / 8.0,
                      exponent.astype(np.int16) - 7)
    subnormal = np.ldexp(mantissa.astype(np.float32), -9)
    decoded = sign * np.where(exponent == 0, subnormal, normal)
    decoded[(exponent == 15) & (mantissa == 7)] = np.nan
    return decoded * weight_scale


def lookup_rows(rows, row_ids, table):
    if row_ids.size == 0:
        return mx.zeros((*row_ids.shape, rows.shape[1]), dtype=mx.bfloat16)
    unique, inverse = np.unique(row_ids.reshape(-1), return_inverse=True)
    # Advanced indexing already owns a copy; avoid a second host allocation.
    packed = rows[unique]
    decoded = table[packed]
    if np.isnan(decoded).any():
        raise ValueError("Qwen N-gram store contains NaN")
    values = mx.array(decoded).astype(mx.bfloat16)
    restored = mx.take(values, mx.array(inverse.astype(np.int32)), axis=0)
    return restored.reshape(*row_ids.shape, rows.shape[1])
