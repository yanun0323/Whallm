"""Opt-in QSA Prefill kernel that reads only the selected key blocks. Synthetic tensors."""
import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd import qwen4_exp
from deepseek_v4_ssd.generation import _prompt_cache_contract
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import validate_runtime_config
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.qwen4_exp import QSAAttention
from deepseek_v4_ssd.qwen_flash_config import add_flash_arguments, flash_arguments
from runtime.tests.test_qwen_speed import tiny_args


def attention():
    # The kernel serves 256-wide heads, two query heads per SIMD group. The index
    # rotary width (head_dim * factor) fits the index key, as in the checkpoint.
    return QSAAttention(tiny_args(head_dim=256, num_attention_heads=4, num_key_value_heads=2,
                                  indexer_head_dim=128, partial_rotary_factor=0.25, indexer_budget=16))


def inputs(key_length, queries, zero_query=False, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((1, 4, queries, 256)).astype(mx.bfloat16)
    if zero_query:
        q = mx.zeros_like(q)
    k = mx.random.normal((1, 2, key_length, 256)).astype(mx.bfloat16)
    v = mx.random.normal((1, 2, key_length, 256)).astype(mx.bfloat16)
    iq = mx.random.normal((1, queries, 2, 128))
    raw = mx.random.normal((1, key_length, 128))
    return q, k, v, iq, raw


class SelectedKernelTests(unittest.TestCase):
    def test_kernel_reads_the_gathered_paths_key_set(self):
        attn = attention()
        # Lengths cover incomplete last blocks, fewer visible blocks than the
        # budget, and a chunk that starts at position zero.
        for key_length, queries in ((40, 16), (64, 33), (63, 17), (37, 37), (200, 64)):
            for zero_query in (True, False):
                with self.subTest(key_length=key_length, queries=queries, zero_query=zero_query):
                    q, k, v, iq, raw = inputs(key_length, queries, zero_query, seed=key_length + queries)
                    offset = key_length - queries
                    attn.selected_kernel = False
                    # FP32 gathered path over the same BF16 values: the reference.
                    expected = attn._bounded_attention(q.astype(mx.float32), k.astype(mx.float32),
                                                       v.astype(mx.float32), iq, raw, offset)
                    attn.selected_kernel = True
                    with patch.object(attn, "_selected_attention", wraps=attn._selected_attention) as spy:
                        actual = attn._bounded_attention(q, k, v, iq, raw, offset)
                    mx.eval(expected, actual)
                    self.assertTrue(spy.called)
                    self.assertEqual(actual.dtype, mx.bfloat16)
                    # BF16 output rounding only; one wrong key moves a value by ~0.05.
                    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)), np.asarray(expected),
                                               rtol=1e-2, atol=1e-2)

    def test_dispatch_follows_query_count_budget_and_masked_threshold(self):
        attn = attention()
        attn.selected_kernel = True
        with patch.object(qwen4_exp, "SELECTED_KERNEL_MIN_KEYS", 48):
            for masked, key_length, queries, selected in (
                    (True, 47, 16, False), (True, 48, 16, True), (True, 64, 33, True),
                    (False, 17, 16, True), (False, 64, 15, False), (True, 16, 16, False),
                    (False, 16, 16, False)):
                with self.subTest(masked=masked, key_length=key_length, queries=queries):
                    attn.masked_prefill = masked
                    q, k, v, iq, raw = inputs(key_length, queries)
                    with patch.object(attn, "_selected_attention", wraps=attn._selected_attention) as spy:
                        mx.eval(attn._bounded_attention(q, k, v, iq, raw, key_length - queries))
                    self.assertEqual(spy.called, selected)
        self.assertEqual(qwen4_exp.SELECTED_KERNEL_MIN_KEYS, 16_384)
        # Off by default: the masked path keeps every length it served before.
        attn = attention()
        attn.masked_prefill = True
        q, k, v, iq, raw = inputs(64, 16)
        with patch.object(attn, "_selected_attention") as spy:
            mx.eval(attn._bounded_attention(q, k, v, iq, raw, 48))
        spy.assert_not_called()

    def test_setting_is_off_by_default_qwen_only_and_keeps_caches_apart(self):
        self.assertFalse(RuntimeConfig().qwen_qsa_selected_kernel)
        config = RuntimeConfig(qwen_qsa_selected_kernel=True)
        validate_runtime_config(config)
        for kind in ("qwen3.8-flash-next", "swift1.5-qwen3.8-flash-next"):
            with self.assertRaises(ValueError):
                get_support(kind).validate_config(config)
        for kind in ("deepseek-v4", "deepseek-v4.1"):
            with self.assertRaises(ValueError):
                get_support(kind).validate_config(config)
        with self.assertRaises(ValueError):
            validate_runtime_config(RuntimeConfig(qwen_qsa_selected_kernel=1))
        parser = argparse.ArgumentParser()
        add_flash_arguments(parser)
        self.assertIs(flash_arguments(parser.parse_args([]))["qwen_qsa_selected_kernel"], False)
        self.assertIs(flash_arguments(parser.parse_args(["--qwen-qsa-selected-kernel"]))
                      ["qwen_qsa_selected_kernel"], True)
        installed = SimpleNamespace(root="/missing", revision="r", model_id="m")
        off = _prompt_cache_contract(installed, RuntimeConfig())
        on = _prompt_cache_contract(installed, config)
        self.assertNotIn("qwenQSASelectedKernel", off)
        self.assertEqual(on.pop("qwenQSASelectedKernel"), "metal-v1-min16384")
        self.assertEqual(on, off)


if __name__ == "__main__":
    unittest.main()
