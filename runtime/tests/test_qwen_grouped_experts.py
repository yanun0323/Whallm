import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pathlib import Path

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.expert_cache import QwenBatchedExperts, QwenExpertWeights
from deepseek_v4_ssd.model import RuntimeConfig


class GroupedExpertsTests(unittest.TestCase):
    def weights(self):
        rng = np.random.default_rng(20260906)
        gate, gate_scales = mx.quantize(mx.array(rng.normal(0, .1, (16, 128, 64)), dtype=mx.bfloat16),
            group_size=32, bits=4, mode="mxfp4")
        down, down_scales = mx.quantize(mx.array(rng.normal(0, .1, (16, 64, 64)), dtype=mx.bfloat16),
            group_size=32, bits=4, mode="mxfp4")
        return QwenBatchedExperts(gate, gate_scales, down, down_scales)

    def test_grouping_preserves_each_expert_and_short_input_fallback(self):
        weights = self.weights()
        calls = []
        cache = SimpleNamespace(current_batched=lambda layer: weights, record_gather_qmm=calls.append)
        experts = qwen.StreamingExperts(0, cache)
        self.assertTrue(RuntimeConfig().qwen_grouped_experts)
        self.assertFalse(experts.grouped_prefill)
        rng = np.random.default_rng(61)
        for length, top_k in ((1, 10), (6, 10), (8, 8), (128, 10), (1024, 10)):
            with self.subTest(length=length, top_k=top_k):
                value = mx.array(rng.normal(0, .2, (1, length, 64)), dtype=mx.bfloat16)
                indices = mx.array((np.arange(length)[:, None] * 7 + np.arange(top_k)[None, :] * 3) % 16, dtype=mx.int32)[None]
                experts.grouped_prefill = False
                expected = experts(value, indices)
                mx.eval(expected)
                experts.grouped_prefill = True
                with patch.object(qwen, "_gather_sort", wraps=qwen._gather_sort) as sort:
                    actual = experts(value, indices)
                    mx.eval(actual)
                    self.assertEqual(sort.call_count, int(length * top_k >= 64))
                self.assertEqual(actual.shape, (1, length, top_k, 64))
                self.assertTrue(mx.array_equal(actual, expected).item())
        self.assertEqual(calls, [2] * 10)

    def test_sorted_prefill_is_opt_in_close_and_only_for_grouped_rows(self):
        weights = self.weights()
        cache = SimpleNamespace(current_batched=lambda layer: weights, record_gather_qmm=lambda count: None)
        experts = qwen.StreamingExperts(0, cache)
        self.assertFalse(experts.sorted_prefill)
        self.assertFalse(RuntimeConfig().qwen_sorted_expert_prefill)
        rng = np.random.default_rng(62)
        for length, top_k in ((6, 10), (128, 10), (1024, 10)):
            with self.subTest(length=length):
                value = mx.array(rng.normal(0, .2, (1, length, 64)), dtype=mx.bfloat16)
                indices = mx.array(rng.integers(0, 16, (1, length, top_k)), dtype=mx.int32)
                experts.grouped_prefill, experts.sorted_prefill = True, False
                expected = experts(value, indices)
                flags = []
                original = mx.gather_qmm
                def record(*args, **kwargs):
                    flags.append(kwargs["sorted_indices"])
                    return original(*args, **kwargs)
                experts.sorted_prefill = True
                with patch.object(qwen.mx, "gather_qmm", side_effect=record):
                    actual = experts(value, indices)
                    mx.eval(expected, actual)
                # Short inputs are not grouped, so they keep the unsorted kernel.
                self.assertEqual(flags, [length * top_k >= 64] * 2)
                self.assertEqual(actual.shape, expected.shape)
                difference = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)).max().item()
                self.assertLess(difference, 1e-2)
                experts.grouped_prefill = False
                with patch.object(qwen.mx, "gather_qmm", side_effect=record):
                    flags.clear()
                    mx.eval(experts(value, indices))
                self.assertEqual(flags, [False, False])

    def test_sorted_prefill_config_cli_and_cache_contract(self):
        import argparse
        from deepseek_v4_ssd.generation import _prompt_cache_contract
        from deepseek_v4_ssd.qwen_flash_config import (
            add_flash_arguments, flash_arguments, validate_flash_config)
        parser = argparse.ArgumentParser()
        add_flash_arguments(parser)
        self.assertFalse(flash_arguments(parser.parse_args([]))["qwen_sorted_expert_prefill"])
        self.assertTrue(flash_arguments(parser.parse_args(["--qwen-sorted-expert-prefill"]))[
            "qwen_sorted_expert_prefill"])
        validate_flash_config(RuntimeConfig(qwen_sorted_expert_prefill=True))
        for invalid in (RuntimeConfig(qwen_sorted_expert_prefill=1),
                        RuntimeConfig(qwen_sorted_expert_prefill=True, qwen_grouped_experts=False)):
            with self.subTest(config=invalid), self.assertRaises(ValueError):
                validate_flash_config(invalid)
        installed = SimpleNamespace(root=Path("/nonexistent"), revision="r", model_id="m")
        off = _prompt_cache_contract(installed, RuntimeConfig())
        self.assertNotIn("qwenSortedExpertPrefill", off)
        self.assertEqual(off, _prompt_cache_contract(installed, RuntimeConfig(
            qwen_sorted_expert_prefill=True, qwen_grouped_experts=False)))
        self.assertEqual(_prompt_cache_contract(installed, RuntimeConfig(qwen_sorted_expert_prefill=True))[
            "qwenSortedExpertPrefill"], "mlx-v1")

    def test_load_applies_default_off_override_with_or_without_mtp(self):
        installed = SimpleNamespace(root=Path("/unused"), ngram=SimpleNamespace(file="ngram.bin"))
        scale = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
        for config, expected, sort in (
            (RuntimeConfig(), True, False),
            (RuntimeConfig(qwen_grouped_experts=False), False, False),
            # Grouping is bit-identical, so MTP no longer disables it.
            (RuntimeConfig(mtp_enabled=True), True, False),
            (RuntimeConfig(qwen_sorted_expert_prefill=True), True, True),
        ):
            with self.subTest(config=config):
                model = MagicMock()
                model.model.layers = [SimpleNamespace(mlp=SimpleNamespace(experts=SimpleNamespace())) for _ in range(2)]
                model.parameters.return_value = []
                model.sanitize.return_value = {}
                with patch.object(qwen.ModelArgs, "from_dict"), \
                     patch.object(qwen, "ExpertCache"), patch.object(qwen, "NGramStore"), \
                     patch.object(qwen, "Model", return_value=model), \
                     patch("deepseek_v4_ssd.ane_prefill.install_qwen_ane_prefill", return_value=None):
                    qwen.load(installed, config, {"text_config": {}}, {scale: mx.array([1.0])})
                self.assertEqual([layer.mlp.experts.grouped_prefill for layer in model.model.layers], [expected] * 2)
                self.assertEqual([layer.mlp.experts.sorted_prefill for layer in model.model.layers], [sort] * 2)

    def test_individual_expert_decode_keeps_original_path(self):
        batched = self.weights()
        individual = [QwenExpertWeights(batched.gate_up[i], batched.gate_up_scales[i],
                      batched.down[i], batched.down_scales[i]) for i in range(16)]
        acquisitions = []
        def get_many(layer, ids):
            acquisitions.append((layer, ids))
            return SimpleNamespace(individual_weights=individual, slots={i: i for i in range(16)})
        cache = SimpleNamespace(current_batched=lambda layer: None, get_many=get_many)
        experts = qwen.StreamingExperts(3, cache)
        value = mx.full((1, 1, 64), .25, dtype=mx.bfloat16)
        indices = mx.array([[[9, 2, 7, 1]]])
        expected = experts(value, indices)
        mx.eval(expected)
        experts.grouped_prefill = True
        with patch.object(qwen, "_gather_sort", side_effect=AssertionError("Decode was grouped")):
            actual = experts(value, indices)
            mx.eval(actual)
        self.assertTrue(mx.array_equal(actual, expected).item())
        self.assertEqual(acquisitions, [(3, [9, 2, 7, 1])] * 2)


if __name__ == "__main__":
    unittest.main()
