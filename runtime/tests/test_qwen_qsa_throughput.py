"""Synthetic Metal and cache-lifecycle checks for QSA throughput experiments."""
import unittest
from unittest.mock import patch
import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import CacheList, KVCache
from deepseek_v4_ssd import qwen_qsa_indexed as native
from deepseek_v4_ssd.qwen4_exp import QSAAttention
from deepseek_v4_ssd.qwen_quantized_cache import QSAQuantizedCache
from deepseek_v4_ssd.qwen_pooled_cache import QSAPooledIndexCache
from deepseek_v4_ssd.model import RuntimeConfig, eval_prompt_cache, _fork_prompt_cache
from deepseek_v4_ssd.model_support.state import persistence_cache_state, restore_persistence_cache
from runtime.tests.test_qwen_speed import tiny_args, bits, close


def reference(q, k, v, selected, valid, offset):
    q, k, v = [np.asarray(a.astype(mx.float32)).astype(np.float64) for a in (q,k,v)]
    ids, mask = np.asarray(selected), np.asarray(valid)
    out = np.zeros_like(q)
    for t in range(q.shape[2]):
        visible = ids[t][mask[t] & (ids[t]>=0) & (ids[t]<k.shape[2]) & (ids[t]<=offset+t)]
        if not len(visible):
            continue
        for h in range(q.shape[1]):
            kh = h // (q.shape[1]//k.shape[1])
            scores = k[0,kh,visible] @ q[0,h,t] / np.sqrt(q.shape[-1])
            weights = np.exp(scores - scores.max()); weights /= weights.sum()
            out[0,h,t] = weights @ v[0,kh,visible]
    return out


class IndexedQSATests(unittest.TestCase):
    def test_metal_precision_masks_strides_and_gqa(self):
        self.assertTrue(mx.metal.is_available(), "This test requires real Metal")
        mx.random.seed(624)
        for dtype,tol in ((mx.float32, 2e-5),(mx.float16, .002),(mx.bfloat16,.012)):
            for length in (1,3,8):
                for dim in (32,256):
                    with self.subTest(dtype=dtype,length=length,dim=dim):
                        # All inputs include non-contiguous cache/query views.
                        q = mx.random.normal((1,4,length*2,dim)).astype(dtype)[:,:,::2]
                        k = mx.random.normal((1,2,138,dim)).astype(dtype)[:,:,::2]
                        v = mx.random.normal((1,2,138,dim)).astype(dtype)[:,:,::2]
                        ids = np.tile(np.array([0,3,5,10,30,64,65,67,68,-1,99,5],np.int32),(length,1))
                        mask = np.ones(ids.shape,bool); mask[:,2] = False
                        if length>1: mask[1]=False
                        ids, mask = mx.array(ids),mx.array(mask)
                        actual = native.indexed_attention(q,k,v,ids,mask,69-length)
                        self.assertIsNotNone(actual)
                        mx.eval(actual)
                        np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)),
                            reference(q,k,v,ids,mask,69-length),rtol=tol,atol=tol)
                        self.assertTrue(mx.all(mx.isfinite(actual)).item())

    def test_narrow_admission_and_invalid_offset(self):
        q=mx.ones((1,4,1,32)); k=mx.ones((1,2,20,32))
        ids=mx.array([[1,2]],dtype=mx.int32); valid=mx.ones((1,2),dtype=mx.bool_)
        with self.assertRaises(ValueError): native.indexed_attention(q,k,k,ids,valid,-1)
        with self.assertRaises(ValueError): native.indexed_attention(q,k,k,ids,valid,20)
        self.assertIsNone(native.indexed_attention(mx.ones((1,4,9,32)),k,k,ids,valid,0))
        with patch.object(native.mx, 'default_device', return_value=mx.cpu):
            self.assertIsNone(native.indexed_attention(q,k,k,ids,valid,0))

    def test_kernel_cache_does_not_specialize_on_context_length(self):
        self.assertIs(native._partials_kernel(), native._partials_kernel())
        self.assertIs(native._merge_kernel(), native._merge_kernel())

    def test_direct_route_avoids_kv_gather_and_preserves_inputs(self):
        args=tiny_args(); attn=QSAAttention(args); attn.sparse_sdpa=True; attn.indexed_decode=True
        q=mx.ones((1,4,3,32));kv=mx.ones((1,2,17,32))
        raw=mx.ones((1,17,32)); iq=mx.ones((1,3,2,32))
        before=bits(kv).copy()
        with patch('deepseek_v4_ssd.qwen4_exp.mx.take', side_effect=AssertionError('KV gather')):
            out=attn._bounded_attention(q,kv,kv,iq,raw,14)
            mx.eval(out)
        np.testing.assert_array_equal(bits(kv),before)
        np.testing.assert_allclose(np.asarray(out),1,atol=1e-6)

    def test_prefill_does_not_use_narrow_kernel(self):
        attn=QSAAttention(tiny_args());attn.sparse_sdpa=True;attn.indexed_decode=True;attn.query_chunk=16
        with patch.object(native,'indexed_attention',side_effect=AssertionError('prefill entered decode kernel')):
            mx.eval(attn(mx.ones((1,17,64)),None))

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
                            attn.sparse_sdpa=False;attn.indexed_decode=False;attn.query_chunk=4
                            expected=attn(x,baseline)
                            attn.sparse_sdpa=True;attn.indexed_decode=True;attn.query_chunk=chunk
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

    def test_runtime_catalog_legacy_fields_and_qwen_only(self):
        from deepseek_v4_ssd.model_support import get_support
        for kind in ('deepseek-v4','deepseek-v4.1'):
            with self.assertRaises(ValueError): get_support(kind).validate_config(RuntimeConfig(qwen_qsa_query_chunk=32))
        get_support('qwen3.8-flash-next').validate_config(RuntimeConfig(qwen_qsa_query_chunk=32,qwen_sparse_sdpa=True,qwen_qsa_indexed=True))
