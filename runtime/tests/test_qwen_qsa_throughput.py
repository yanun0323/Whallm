"""Synthetic cache-lifecycle checks for QSA throughput controls."""
import unittest
from unittest.mock import patch
import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import CacheList, KVCache
from deepseek_v4_ssd.qwen4_exp import QSAAttention
from deepseek_v4_ssd.qwen_quantized_cache import QSAQuantizedCache
from deepseek_v4_ssd.qwen_pooled_cache import QSAPooledIndexCache
from deepseek_v4_ssd.model import RuntimeConfig, eval_prompt_cache, _fork_prompt_cache
from deepseek_v4_ssd.model_support.state import persistence_cache_state, restore_persistence_cache
from runtime.tests.test_qwen_speed import tiny_args, bits, close


class QSAThroughputTests(unittest.TestCase):
    def test_stored_axis_gather_is_byte_exact(self):
        mx.random.seed(26)
        for dtype in (mx.float32,mx.bfloat16):
            kv=mx.random.normal((1,2,29,32)).astype(dtype)
            ids=mx.array([[28,0,1,1],[2,3,20,28]])
            a=mx.take(kv[0].transpose(1,0,2),ids,axis=0).transpose(0,2,1,3)
            b=mx.take(kv[0],ids,axis=1).transpose(1,0,2,3)
            np.testing.assert_array_equal(bits(a),bits(b))

    def test_chunks_threshold_quantized_cache_fork_restore_and_rollback(self):
        for dtype,tol in ((mx.float32,4e-5),(mx.bfloat16,.015)):
            for packed in (False,True):
                for chunk in (1,4,16,32):
                    with self.subTest(dtype=dtype,packed=packed,chunk=chunk):
                        mx.random.seed(631)
                        attn=QSAAttention(tiny_args());attn.set_dtype(dtype)
                        hidden=(mx.random.normal((1,31,64))*.15).astype(dtype)
                        def cache():return CacheList(QSAQuantizedCache(8,32) if packed else KVCache(),QSAPooledIndexCache())
                        baseline,fast=cache(),cache()
                        start=0
                        for length in (5,8,1,3,8):
                            x=hidden[:,start:start+length]
                            attn.query_chunk=4
                            expected=attn(x,baseline)
                            attn.query_chunk=chunk
                            actual=attn(x,fast)
                            eval_prompt_cache([baseline,fast],actual,expected)
                            close(actual,expected,tol)
                            start+=length
                        saved=persistence_cache_state([fast]);forked,dependencies=_fork_prompt_cache([fast]);mx.eval(*dependencies)
                        restored=cache();restore_persistence_cache([restored],saved)
                        expected=attn(hidden[:,25:28],fast);eval_prompt_cache([fast],expected)
                        for other in (forked[0],restored):
                            actual=attn(hidden[:,25:28],other);eval_prompt_cache([other],actual)
                            np.testing.assert_array_equal(bits(actual),bits(expected))
                        for other in (fast,restored):
                            for branch in other.caches:branch.trim(3)
                        close(attn(hidden[:,25:28],restored),expected,tol)

    def test_masked_prefill_keeps_the_selected_key_set(self):
        args = tiny_args(indexer_budget=16)
        attn = QSAAttention(args)
        mx.random.seed(1007)
        for key_length, queries in ((40, 16), (64, 33), (64, 64), (16, 16), (40, 15)):
            offset = key_length - queries
            # Equal scores weight every selected key alike, so the output is the
            # mean of the selected values and exposes any change in the key set.
            for zero_query in (True, False):
                with self.subTest(key_length=key_length, queries=queries, zero_query=zero_query):
                    q = mx.random.normal((1, 4, queries, 32))
                    if zero_query:
                        q = mx.zeros_like(q)
                    k = mx.random.normal((1, 2, key_length, 32))
                    v = mx.random.normal((1, 2, key_length, 32))
                    iq = mx.random.normal((1, queries, 2, 32))
                    raw = mx.random.normal((1, key_length, 32))
                    attn.masked_prefill = False
                    expected = attn._bounded_attention(q, k, v, iq, raw, offset)
                    attn.masked_prefill = True
                    if queries < 16:
                        with patch.object(attn, "_masked_attention", side_effect=AssertionError("decode used mask")):
                            actual = attn._bounded_attention(q, k, v, iq, raw, offset)
                    else:
                        actual = attn._bounded_attention(q, k, v, iq, raw, offset)
                    mx.eval(expected, actual)
                    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-5)

    def test_masked_prefill_config_and_cache_contract(self):
        from deepseek_v4_ssd.generation import _prompt_cache_contract
        from deepseek_v4_ssd.qwen_flash_config import validate_flash_config
        from types import SimpleNamespace
        with self.assertRaises(ValueError):
            validate_flash_config(SimpleNamespace(qwen_qsa_masked_prefill=1))
        installed = SimpleNamespace(root="/missing", revision="r", model_id="m")
        off = _prompt_cache_contract(installed, RuntimeConfig())
        on = _prompt_cache_contract(installed, RuntimeConfig(qwen_qsa_masked_prefill=True))
        self.assertNotIn("qwenQSAMaskedPrefill", off)
        self.assertNotEqual(off, on)

    def test_runtime_catalog_legacy_fields_and_qwen_only(self):
        from deepseek_v4_ssd.model_support import get_support
        for kind in ('deepseek-v4','deepseek-v4.1'):
            with self.assertRaises(ValueError): get_support(kind).validate_config(RuntimeConfig(qwen_qsa_query_chunk=32))
        get_support('qwen3.8-flash-next').validate_config(RuntimeConfig(qwen_qsa_query_chunk=32))

    def test_removed_qsa_and_overlap_fields(self):
        from dataclasses import asdict
        from deepseek_v4_ssd.model_manager import ModelCatalogError, _parse_runtime
        removed = ("qwen_shared_expert_overlap", "qwen_sparse_sdpa", "qwen_qsa_indexed")
        raw = asdict(RuntimeConfig())
        for name in removed:
            self.assertNotIn(name, raw)
            # Catalogs written before the removal still load when the switch was off.
            self.assertEqual(_parse_runtime(raw | {name: False}, "runtime", "qwen3.8-flash-next"), RuntimeConfig())
            with self.assertRaisesRegex(ModelCatalogError, "has been removed"):
                _parse_runtime(raw | {name: True}, "runtime", "qwen3.8-flash-next")
        attention = QSAAttention(tiny_args())
        self.assertFalse(hasattr(attention, "sparse_sdpa") or hasattr(attention, "indexed_decode"))


class NativeMTPSettingsTests(unittest.TestCase):
    def test_new_qsa_controls_reach_native_mtp_without_changing_old_defaults(self):
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import MagicMock
        from deepseek_v4_ssd import model as loader
        installed = SimpleNamespace(root=Path("/synthetic-model"),
            mtp=SimpleNamespace(common_tensors=()))
        for chunk in (4, 16, 32):
            with self.subTest(chunk=chunk):
                attention = SimpleNamespace(query_chunk=4,
                    dense_within_budget=False, dense_threshold=0, skip_complete_gather=False)
                draft = MagicMock()
                draft.layers = [SimpleNamespace(self_attn=attention)]
                draft.sanitize.return_value = {}
                draft.parameters.return_value = []
                cache = MagicMock()
                config = RuntimeConfig(qwen_qsa_query_chunk=chunk)
                with patch.object(loader,"replace",return_value=installed), \
                     patch.object(loader,"ExpertCache",return_value=cache), \
                     patch.object(loader,"_load_tensor_file",return_value={}), \
                     patch("deepseek_v4_ssd.qwen4_exp.MTPModel",return_value=draft):
                    actual, actual_cache = loader._load_qwen_mtp(installed,tiny_args(),config,None)
                self.assertIs(actual,draft)
                self.assertIs(actual_cache,cache)
                self.assertEqual(attention.query_chunk,chunk)
                self.assertTrue(attention.skip_complete_gather)
                self.assertFalse(attention.dense_within_budget)
                self.assertEqual(attention.dense_threshold,0)
                draft.load_weights.assert_called_once_with([],strict=True)
                cache.close.assert_not_called()
