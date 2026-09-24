from __future__ import annotations

import math
import unittest

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd.mimo.weights import decode_fp8_blocks, decode_qkv, native_expert_views
from deepseek_v4_ssd.model import _mxfp4


def fp8_reference(raw):
    """Independent IEEE E4M3FN scalar decoding, not MLX's conversion."""
    sign = np.where(raw & 128, -1.0, 1.0)
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    values = np.where(exponent == 0, mantissa * 2.0**-9,
                      (1 + mantissa / 8.0) * np.exp2(exponent.astype(int) - 7))
    values = sign * values
    return np.where((exponent == 15) & (mantissa == 7), np.nan, values).astype(np.float32)


def bf16_reference(values):
    values = np.asarray(values, dtype=np.float32)
    words = values.view(np.uint32)
    # Round to nearest even, then expand to float32 for assertions.
    words = (words + 0x7FFF + ((words >> 16) & 1)) & np.uint32(0xFFFF0000)
    return words.view(np.float32)


def mxfp4_reference(raw, scales):
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6,
                    -0., -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    codes = np.stack((raw & 15, raw >> 4), axis=-1).reshape(raw.shape[0], -1)
    factor = np.exp2(scales.astype(np.int32) - 127).astype(np.float32)
    return lut[codes] * np.repeat(factor, 32, axis=-1)


class MiMoNativeWeightTests(unittest.TestCase):
    def test_all_fp8_codes_against_independent_decoder(self):
        raw = np.arange(256, dtype=np.uint8).reshape(1, 256)
        with self.assertRaisesRegex(ValueError, "NaN codes"):
            decode_fp8_blocks(mx.array(raw), mx.ones((1, 2), mx.float32))
        raw[0, [127, 255]] = 0
        actual = decode_fp8_blocks(mx.array(raw), mx.ones((1, 2), mx.float32))
        np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), fp8_reference(raw))

    def test_global_attention_tp4_padding_and_deinterleaving(self):
        # Exact checkpoint QKV row count, narrow columns to keep this a unit test.
        for kv_heads in (4, 8):
            q, k, v = 64 // 4 * 192, kv_heads // 4 * 192, kv_heads // 4 * 128
            rows = q + k + v
            raw = np.full((4 * rows, 32), 56, dtype=np.uint8)  # E4M3 1.0
            scales = np.arange(1, 4 * math.ceil(rows / 128) + 1, dtype=np.float32)[:, None]
            expected_shards = []
            for s in range(4):
                local = np.repeat(scales[s * math.ceil(rows / 128):(s + 1) * math.ceil(rows / 128)], 128, axis=0)[:rows]
                expected_shards.append(np.broadcast_to(local, (rows, 32)))
            expected = np.concatenate(
                [shard[:q] for shard in expected_shards]
                + [shard[q:q + k] for shard in expected_shards]
                + [shard[q + k:] for shard in expected_shards])
            actual = decode_qkv(mx.array(raw), mx.array(scales), query_heads=64,
                                kv_heads=kv_heads, head_dim=192, value_dim=128, source_shards=4)
            np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), expected)
            if kv_heads == 4:
                with self.assertRaisesRegex(ValueError, "padding"):
                    decode_fp8_blocks(mx.array(raw), mx.array(scales), source_shards=1)

    def test_non_power_of_two_scales_and_bf16_rounding(self):
        rng = np.random.default_rng(71)
        raw = rng.integers(0, 119, (516, 137), dtype=np.uint8)
        scales = rng.uniform(.003, .07, (8, 2)).astype(np.float32)
        actual = decode_fp8_blocks(mx.array(raw), mx.array(scales), source_shards=4)
        expected = []
        for s in range(4):
            expanded = scales[s * 2:(s + 1) * 2].repeat(128, 0).repeat(128, 1)[:129, :137]
            expected.append(bf16_reference(fp8_reference(raw[s * 129:(s + 1) * 129]) * expanded))
        np.testing.assert_array_equal(np.asarray(actual.astype(mx.float32)), np.concatenate(expected))

    def test_native_mxfp4_all_nibbles_and_qmm_without_requantization(self):
        raw = np.arange(256, dtype=np.uint8).reshape(8, 32)
        scales = np.arange(123, 139, dtype=np.uint8).reshape(8, 2)
        weight, scale = native_expert_views(mx.array(raw), mx.array(scales))
        np.testing.assert_array_equal(np.asarray(weight).view(np.uint8), raw)
        expected_weights = mxfp4_reference(raw, scales)
        dequantized = mx.dequantize(weight, scale, group_size=32, bits=4, mode="mxfp4")
        np.testing.assert_array_equal(np.asarray(dequantized.astype(mx.float32)), expected_weights)
        x = np.random.default_rng(42).normal(0, .1, (3, 64)).astype(np.float32)
        actual = _mxfp4(mx.array(x), weight, scale)
        np.testing.assert_allclose(np.asarray(actual), x @ expected_weights.T, rtol=2e-5, atol=1e-4)

    def test_invalid_layouts_fail_before_compute(self):
        w, s = mx.zeros((8, 32), mx.uint8), mx.ones((8, 2), mx.uint8)
        for weight, scales in ((w.astype(mx.float32), s), (w, s.astype(mx.float32)),
                               (w, s[:, :1]), (w[:, :31], s), (w[0], s)):
            with self.subTest(shape=weight.shape), self.assertRaises(ValueError):
                native_expert_views(weight, scales)
        with self.assertRaises(ValueError):
            decode_fp8_blocks(w, s)
        with self.assertRaises(ValueError):
            decode_fp8_blocks(w, s.astype(mx.float32), source_shards=True)
        with self.assertRaises(ValueError):
            decode_qkv(w, s.astype(mx.float32), query_heads=4, kv_heads=1,
                       head_dim=8, value_dim=8, source_shards=4)


if __name__ == "__main__":
    unittest.main()
