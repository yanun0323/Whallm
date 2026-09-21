"""Tests for Qwen speed paths. No full model or network required."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import CacheList, KVCache

from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.expert_cache import QwenExpertWeights
from deepseek_v4_ssd.model import eval_prompt_cache
from deepseek_v4_ssd.qwen_tensor_ops import dense_causal_attention, hyper_inject
from deepseek_v4_ssd.qwen_pooled_cache import QSAPooledIndexCache
from runtime.tests.test_qwen import FakeGreedyTarget, FakeGreedyMTP, FakeTargetCache


def bits(value):
    mx.eval(value)
    return np.asarray(value.view(mx.uint8))


def close(actual, expected, tolerance):
    np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)),
                               np.asarray(expected.astype(mx.float32)),
                               atol=tolerance, rtol=tolerance)


def tiny_args(**kwargs):
    fields = dict(hidden_size=64, hc_count=4, hc_lowrank=16, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=32, indexer_n_heads=2,
                  indexer_head_dim=32, indexer_budget=8, indexer_compress_ratio=4,
                  partial_rotary_factor=0.5, num_experts=8, num_experts_per_tok=2,
                  moe_intermediate_size=64, shared_expert_intermediate_size=64,
                  vocab_size=32, max_position_embeddings=128)
    fields.update(kwargs)
    return qwen.ModelArgs(**fields)


def experts_fixture(dtype=mx.bfloat16):
    mx.random.seed(18)
    gate, gs = mx.quantize(mx.random.normal((8, 128, 64)).astype(mx.bfloat16) * .03,
                           group_size=32, bits=4, mode='mxfp4')
    down, ds = mx.quantize(mx.random.normal((8, 64, 64)).astype(mx.bfloat16) * .03,
                           group_size=32, bits=4, mode='mxfp4')
    mx.eval(gate, gs, down, ds)
    weights = [QwenExpertWeights(gate[i], gs[i], down[i], ds[i]) for i in range(8)]
    cache = SimpleNamespace(slots=8, model=SimpleNamespace(expert_count=8),
                            current_batched=lambda _: None,
                            get_many=lambda layer, ids: SimpleNamespace(individual_weights=weights, slots={i: i for i in ids}))
    return weights, cache


class DenseQSASpeedTests(unittest.TestCase):
    def test_dense_matches_explicit_attention_with_prefix_offsets_and_gqa(self):
        mx.random.seed(19)
        for dtype, tol in ((mx.float32, 2e-5), (mx.float16, 3e-3), (mx.bfloat16, 1.5e-2)):
            for head_dim in (32, 256):
                for length, total in ((1, 1), (1, 8), (3, 3), (3, 8), (8, 8)):
                    with self.subTest(dtype=dtype, head=head_dim, shape=(length, total)):
                        query = (mx.random.normal((1, 4, length, head_dim)) * .1).astype(dtype)
                        key = (mx.random.normal((1, 2, total, head_dim)) * .1).astype(dtype)
                        value = (mx.random.normal(key.shape) * .1).astype(dtype)
                        offset = total - length
                        scores = query @ mx.repeat(key, 2, axis=1).swapaxes(-1, -2) * head_dim**-.5
                        mask = mx.arange(total)[None] <= (mx.arange(length) + offset)[:, None]
                        scores = mx.where(mask, scores, mx.finfo(dtype).min)
                        expected = mx.softmax(scores.astype(mx.float32), axis=-1).astype(dtype) @ mx.repeat(value, 2, axis=1)
                        close(dense_causal_attention(query, key, value, offset), expected, tol)

    def test_dense_route_does_not_gather_or_pool(self):
        attention = qwen.QSAAttention(tiny_args())
        attention.sparse_sdpa = True
        query = mx.ones((1, 4, 3, 32))
        kv = mx.ones((1, 2, 7, 32))
        bad_pool = SimpleNamespace(pooled=lambda *args: self.fail('dense route pooled keys'))
        with patch.object(qwen.mx, 'take', side_effect=AssertionError('dense route gathered KV')):
            output = attention._bounded_attention(query, kv, kv, None, None, 4, bad_pool)
            mx.eval(output)
        self.assertEqual(output.shape, query.shape)

    def test_dense_to_sparse_pooled_cache_trim_and_continuation(self):
        mx.random.seed(38)
        attention = qwen.QSAAttention(tiny_args())
        attention.sparse_sdpa = True
        hidden = mx.random.normal((1, 17, 64)) * .1
        cache = CacheList(KVCache(), QSAPooledIndexCache())
        baseline = CacheList(KVCache(), KVCache())
        start = 0
        for length in (3, 5, 1, 4, 4):
            chunk = hidden[:, start:start + length]
            actual = attention(chunk, cache)
            attention.sparse_sdpa = False
            expected = attention(chunk, baseline)
            attention.sparse_sdpa = True
            close(actual, expected, 2e-5)
            start += length
        for branch in (*cache.caches, *baseline.caches):
            branch.trim(11)
        actual = attention(hidden[:, 6:10], cache)
        attention.sparse_sdpa = False
        close(actual, attention(hidden[:, 6:10], baseline), 2e-5)


class HyperCompilationTests(unittest.TestCase):
    def test_multiple_tokens_keep_norm_only_path(self):
        from deepseek_v4_ssd import qwen_tensor_ops as ops
        layer = qwen.GatedResidual(tiny_args())
        layer.compiled = True
        layer.hc_norm.compiled = True
        hidden = mx.ones((1, 3, 256))
        with patch.object(ops, 'compiled_scaled_silu', side_effect=AssertionError('batched fusion')):
            with patch.object(ops, 'compiled_hyper_inject', side_effect=AssertionError('batched injection')):
                mixed, residual, injection = layer(hidden)
                mx.eval(layer.inject(residual, mixed, injection))

    def test_compiled_hyper_connection_and_live_weights(self):
        mx.random.seed(12)
        for dtype, tol in ((mx.float32, 3e-5), (mx.float16, 3e-3), (mx.bfloat16, 2e-2)):
            for length in (1, 3, 32):
                for combine in (True, False):
                    layer = qwen.GatedResidual(tiny_args(), combine=combine)
                    layer.set_dtype(dtype)
                    x = (mx.random.normal((1, length, 256)) * .1).astype(dtype)
                    for updated in (False, True):
                        if updated:
                            layer.input_mix_weight_up.weight = layer.input_mix_weight_up.weight * -2
                        layer.compiled = False
                        expected = layer(x)
                        layer.compiled = True
                        actual = layer(x)
                        if combine:
                            for got, want in zip(actual, expected):
                                close(got, want, tol)
                            close(layer.inject(actual[1], actual[0], actual[2]),
                                  hyper_inject(expected[1], expected[0], expected[2]), tol)
                        else:
                            close(actual, expected, tol)


class SingletonExpertTests(unittest.TestCase):
    def test_singleton_matches_grouped_reference_bits_without_sort_or_take(self):
        weights, cache = experts_fixture()
        layer = qwen.StreamingExperts(0, cache)
        for dtype in (mx.float32, mx.float16, mx.bfloat16):
            x = mx.random.normal((1, 1, 64)).astype(dtype)
            selected = mx.array([[[5, 1, 7, 0]]])
            # Preserve the baseline per-expert [1,H] shape and route order.
            expected = mx.stack([layer._one(mx.take(x.reshape(1, 64), mx.array([0]), axis=0), weights[i])
                                 for i in (5, 1, 7, 0)], axis=1).reshape(1, 1, 4, 64)
            mx.eval(expected)
            with patch.object(qwen.np, 'argsort', side_effect=AssertionError('sorted singleton')):
                actual = layer(x, selected)
            np.testing.assert_array_equal(bits(actual), bits(expected))

    def test_repeated_routes_preserve_existing_gemm_shape(self):
        weights, cache = experts_fixture()
        layer = qwen.StreamingExperts(0, cache)
        value = mx.ones((1, 1, 64), dtype=mx.bfloat16)
        with patch.object(qwen.StreamingExperts, '_one', wraps=layer._one) as one:
            actual = layer(value, mx.array([[[2, 2, 1]]]))
            mx.eval(actual)
            self.assertEqual([call.args[0].shape for call in one.call_args_list], [(1, 64), (2, 64)])
        np.testing.assert_array_equal(bits(actual[:, :, 0]), bits(actual[:, :, 1]))


class MTPAdvanceTests(unittest.TestCase):
    def test_native_advance_matches_full_forward_hidden_and_following_logits(self):
        _, cache = experts_fixture()
        model = qwen.MTPModel(tiny_args(), cache)
        model.set_dtype(mx.float32)
        hidden = mx.random.normal((1, 3, 256)) * .1
        embedding = mx.random.normal((32, 64)) * .1
        output_weight = mx.random.normal((32, 64)) * .1
        tokens = mx.array([[1, 2, 3]])
        full_cache, advance_cache = model.make_cache(), model.make_cache()
        logits, expected = model(hidden, tokens, embedding, output_weight, full_cache)
        eval_prompt_cache([full_cache], logits, expected)
        original = model.hyper_connection_mixer
        model.hyper_connection_mixer = lambda _: (_ for _ in ()).throw(AssertionError('final mixer called'))
        actual = model.advance(hidden, tokens, embedding, advance_cache)
        eval_prompt_cache([advance_cache], actual)
        model.hyper_connection_mixer = original
        np.testing.assert_array_equal(bits(actual), bits(expected))
        for got, want in zip(advance_cache.caches, full_cache.caches):
            self.assertEqual(got.offset, want.offset)
            for a, b in zip(got.state, want.state):
                np.testing.assert_array_equal(bits(a), bits(b))
        next_hidden = hidden[:, -1:]
        a, _ = model(next_hidden, mx.array([[4]]), embedding, output_weight, advance_cache)
        b, _ = model(next_hidden, mx.array([[4]]), embedding, output_weight, full_cache)
        np.testing.assert_array_equal(bits(a), bits(b))

    def test_prefill_and_full_accept_sync_use_advance_without_changing_tokens(self):
        class AdvancingMTP(FakeGreedyMTP):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.advanced = 0
            def advance(self, hidden, ids, embedding, cache):
                self.advanced += ids.shape[1]
                return super().__call__(hidden, ids, embedding, None, cache)[1]
        for reject in (None, 3, 4):
            results, caches = [], []
            for cls in (FakeGreedyMTP, AdvancingMTP):
                model = cls(reject_input_token=reject)
                target = FakeGreedyTarget()
                cache = [FakeTargetCache()]
                results.append(list(qwen.generate_mtp_tokens([1, 2], target, model, cache,
                                      max_tokens=11, prefill_step_size=4, draft_tokens=2)))
                caches.append((cache[0].offset, model.cache.size()))
            self.assertGreater(model.advanced, 0)
            self.assertEqual(results[0], results[1])
            self.assertEqual(caches[0], caches[1])


if __name__ == '__main__':
    unittest.main()
