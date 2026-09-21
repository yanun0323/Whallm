"""Extra shape, numerical-stability and documented-profile qualification."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import mlx.core as mx
import numpy as np
from deepseek_v4_ssd import qwen_qsa_indexed as native
from deepseek_v4_ssd.model import RuntimeConfig
from runtime.tests.test_qwen_qsa_throughput import reference


class QSAThroughputQualificationTests(unittest.TestCase):
    def test_empty_query_heads_decline_before_kernel_dispatch(self):
        query = mx.zeros((1, 0, 1, 32))
        key = mx.ones((1, 2, 20, 32))
        ids = mx.array([[0]], dtype=mx.int32)
        valid = mx.ones((1, 1), dtype=mx.bool_)
        with patch.object(native, "_partials_kernel", side_effect=AssertionError("invalid dispatch")):
            self.assertIsNone(native.indexed_attention(query, key, key, ids, valid, 19))

    def test_online_softmax_across_all_splits_with_large_logits(self):
        mx.random.seed(927)
        query = mx.random.normal((1, 24, 3, 256)) * 4
        key = mx.random.normal((1, 2, 2055, 256)) * 4
        value = mx.random.normal(key.shape)
        ids = mx.broadcast_to(mx.arange(2055, dtype=mx.int32)[None], (3, 2055))
        valid = mx.ones(ids.shape, dtype=mx.bool_)
        actual = native.indexed_attention(query, key, value, ids, valid, 2052)
        self.assertIsNotNone(actual)
        mx.eval(actual)
        self.assertTrue(mx.all(mx.isfinite(actual)).item())
        np.testing.assert_allclose(np.asarray(actual),
            reference(query, key, value, ids, valid, 2052), rtol=3e-4, atol=3e-4)

    def test_documented_starting_profiles_match_runtime_contract(self):
        from deepseek_v4_ssd.model_support import get_support
        path = Path(__file__).resolve().parents[2] / "docs/qwen-recommended-settings.json"
        data = json.loads(path.read_text())
        self.assertFalse(data["full_model_validated"])
        for profile in data["profiles"]:
            with self.subTest(profile=profile["name"]):
                config = RuntimeConfig(**profile["runtime"])
                get_support("qwen3.8-flash-next").validate_config(config)
                self.assertEqual(config.slots, config.expert_cache_bytes // 2_611_200)
                self.assertLess(config.expert_cache_bytes, config.memory_limit_gib * 1024**3)
                self.assertLessEqual(config.memory_limit_gib, profile["physical_memory_gib"] - 16)
                self.assertFalse(config.mtp_enabled)
                self.assertFalse(config.qwen_quantized_index)
                self.assertFalse(config.qwen_quantized_kv)
                self.assertEqual(config.qwen_expert_wave_slots, 0)
                self.assertEqual(config.qwen_ngram_io, "mmap")

    def test_intermediate_head_dimensions_with_strided_inputs(self):
        mx.random.seed(932)
        for dtype, tolerance in ((mx.float32, 2e-5), (mx.float16, .002), (mx.bfloat16, .012)):
            for dim in (64, 128):
                for length in (1, 3, 8):
                    with self.subTest(dtype=dtype, dim=dim, length=length):
                        q = mx.random.normal((1, 4, length * 2, dim)).astype(dtype)[:, :, ::2]
                        k = mx.random.normal((1, 2, 138, dim)).astype(dtype)[:, :, ::2]
                        v = mx.random.normal((1, 2, 138, dim)).astype(dtype)[:, :, ::2]
                        ids = mx.broadcast_to(mx.array([[0, 5, 64, 68, -1, 99, 5]], dtype=mx.int32), (length, 7))
                        valid = mx.ones(ids.shape, dtype=mx.bool_)
                        actual = native.indexed_attention(q, k, v, ids, valid, 69 - length)
                        self.assertIsNotNone(actual)
                        mx.eval(actual)
                        np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)),
                            reference(q, k, v, ids, valid, 69 - length),
                            rtol=tolerance, atol=tolerance)
