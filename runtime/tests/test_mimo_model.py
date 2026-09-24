from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd.cancellation import GenerationCancelled, cancellation_scope
from deepseek_v4_ssd.expert_cache import ExpertCache, ExpertWeights, ResidentExperts
from deepseek_v4_ssd.manifest import InstalledModel, Tensor
from deepseek_v4_ssd.mimo.model import Attention, MiMoArgs, Model, Router
from runtime.tests.test_mimo_weights import bf16_reference


def tiny_args():
    return MiMoArgs(hidden_size=64, intermediate_size=96, num_hidden_layers=3,
                    num_attention_heads=4, num_key_value_heads=2,
                    swa_num_attention_heads=4, swa_num_key_value_heads=2,
                    head_dim=32, v_head_dim=32, swa_head_dim=32, swa_v_head_dim=32,
                    partial_rotary_factor=.5, rope_theta=10000000., swa_rope_theta=10000.,
                    sliding_window_size=8, hybrid_layer_pattern=(0, 1, 0), moe_layer_freq=(0, 1, 1),
                    n_routed_experts=4, num_experts_per_tok=2, layernorm_epsilon=1e-6, vocab_size=128)


class TinyExpertCache:
    """In-memory fixture exercising the existing streaming expert compute path."""
    route_trace_enabled = False
    ready_expert_decode = True
    staged_expert_streaming = False

    def __init__(self):
        self.layers_seen = []
        rng = np.random.default_rng(82)
        self.experts = []
        for _ in range(4):
            tensors = []
            for shape in ((32, 64), (64, 32), (32, 64)):
                tensors.extend(mx.quantize(mx.array(rng.normal(0, .03, shape).astype(np.float32)),
                                           group_size=32, bits=4, mode="mxfp4"))
            self.experts.append(ExpertWeights(*tensors))

    def iter_ready(self, layer, selected):
        self.layers_seen.append(layer)
        return iter((i, self.experts[i]) for i in dict.fromkeys(selected))

    def get_many(self, layer, selected):
        self.layers_seen.append(layer)
        return ResidentExperts(tuple(self.experts), {i: i for i in range(4)})

    def pin_layer(self, layer):
        return nullcontext()


def attention_reference(attention, x, *, window=None):
    x = np.asarray(x)
    qkv = x @ np.asarray(attention.qkv_proj.weight).T
    q, k, v = np.split(qkv, [attention.q_size, attention.q_size + attention.k_size], axis=-1)
    batch, length, _ = x.shape
    q = q.reshape(batch, length, attention.heads, attention.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(batch, length, attention.kv_heads, attention.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(batch, length, attention.kv_heads, attention.value_dim).transpose(0, 2, 1, 3)
    v = v * attention.value_scale
    dim = attention.rope.dims
    base = attention.rope.base
    angle = np.arange(length)[:, None] * base ** (-np.arange(0, dim, 2, dtype=np.float32) / dim)
    cosine, sine = np.cos(angle), np.sin(angle)
    for array in (q, k):
        first, second = array[..., :dim // 2].copy(), array[..., dim // 2:dim].copy()
        array[..., :dim // 2] = first * cosine - second * sine
        array[..., dim // 2:dim] = first * sine + second * cosine
    k = np.repeat(k, attention.heads // attention.kv_heads, axis=1)
    v = np.repeat(v, attention.heads // attention.kv_heads, axis=1)
    scores = (q @ k.swapaxes(-1, -2)) * attention.head_dim**-.5
    positions = np.arange(length)
    mask = positions[:, None] >= positions[None, :]
    if window is not None:
        mask &= positions[:, None] - positions[None, :] < window
    scores = np.where(mask, scores, -np.inf)
    sinks = attention.attention_sink_bias
    if sinks is not None:
        scores = np.concatenate((scores, np.broadcast_to(np.asarray(sinks)[None, :, None, None],
                                                        (*scores.shape[:-1], 1))), axis=-1)
    probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    if sinks is not None:
        probs = probs[..., :-1]
    output = (probs @ v).transpose(0, 2, 1, 3).reshape(batch, length, -1)
    return output @ np.asarray(attention.o_proj.weight).T


class MiMoBackboneTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(71)

    def test_attention_matches_independent_reference_with_asymmetric_qk_v(self):
        from mlx_lm.models.base import create_attention_mask
        args = replace(tiny_args(), head_dim=192, v_head_dim=128,
                       swa_head_dim=192, swa_v_head_dim=128, partial_rotary_factor=.334)
        x = mx.random.normal((1, 11, 64))
        for sliding in (False, True):
            module = Attention(args, sliding)
            if sliding:
                module.attention_sink_bias = mx.array([-2., .5, 1., 3.])
            mask = create_attention_mask(x, None, window_size=args.sliding_window_size if sliding else None)
            actual = module(x, mask, None)
            expected = attention_reference(module, x, window=args.sliding_window_size if sliding else None)
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-4, atol=2e-5)

    def test_checkpoint_bf16_attention_accepts_preserved_fp32_sinks(self):
        from mlx_lm.models.base import create_attention_mask
        args = replace(tiny_args(), swa_head_dim=192, swa_v_head_dim=128,
                       partial_rotary_factor=.334)
        module = Attention(args, True)
        module.qkv_proj.weight = module.qkv_proj.weight.astype(mx.bfloat16)
        module.o_proj.weight = module.o_proj.weight.astype(mx.bfloat16)
        module.attention_sink_bias = mx.array([-2., .5, 1., 3.], mx.float32)
        x = mx.random.normal((1, 3, 64)).astype(mx.bfloat16)
        result = module(x, create_attention_mask(x, None, window_size=8), None)
        self.assertEqual(result.dtype, mx.bfloat16)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(mx.all(mx.isfinite(result)).item())
        self.assertEqual(module.attention_sink_bias.dtype, mx.float32)

    def test_router_uses_bias_for_selection_not_weight_and_bf16_operands(self):
        args = tiny_args()
        router = Router(args)
        rng = np.random.default_rng(12)
        weight = rng.normal(0, .2, (4, 64)).astype(np.float32)
        x = rng.normal(0, .3, (1, 3, 64)).astype(np.float32)
        router.weight = mx.array(weight)
        router.e_score_correction_bias = mx.array([0., -3., 2., .3])
        indices, actual = router(mx.array(x))
        logits = bf16_reference(x) @ bf16_reference(weight).T
        scores = 1 / (1 + np.exp(-logits))
        corrected = scores + np.array([0., -3., 2., .3])
        expected_indices = np.argsort(-corrected, axis=-1)[..., :2]
        indices = np.asarray(indices)
        np.testing.assert_array_equal(np.sort(indices, axis=-1), np.sort(expected_indices, axis=-1))
        expected = np.take_along_axis(scores, indices, axis=-1)
        expected /= expected.sum(axis=-1, keepdims=True)
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-6, atol=1e-7)

    def test_chunked_prefill_decode_state_and_backbone_expert_mapping(self):
        cache = TinyExpertCache()
        model = Model(tiny_args(), cache)
        # Distinct routing weights avoid unstable top-k ties in this fixture.
        for layer in model.layers[1:]:
            layer.mlp.gate.weight = mx.random.normal((4, 64)).astype(mx.bfloat16) * .1
        tokens = mx.array([list(range(23))])
        expected = model(tokens)
        state = model.make_cache()
        parts = []
        for start, end in ((0, 3), (3, 10), (10, 17), (17, 18), (18, 23)):
            parts.append(model(tokens[:, start:end], cache=state))
            mx.eval(parts[-1], *[c.state for c in state])
        actual = mx.concatenate(parts, axis=1)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-4, atol=2e-5)
        self.assertEqual(set(cache.layers_seen), {0, 1})
        self.assertEqual([c.offset for c in state], [23, 23, 23])
        fork = copy.deepcopy(state)
        next_token = mx.array([[24]])
        np.testing.assert_allclose(np.asarray(model(next_token, cache=state)),
                                   np.asarray(model(next_token, cache=fork)), rtol=1e-6, atol=1e-6)

    def test_layer_major_preserves_chunk_math_and_continuation(self):
        from deepseek_v4_ssd.model import RuntimeConfig, eval_prompt_cache
        from deepseek_v4_ssd.model_support import get_support
        support = get_support("mimo-v2.6-flash-rl")
        expert_cache = TinyExpertCache()
        model = Model(tiny_args(), expert_cache)
        for layer in model.layers[1:]:
            layer.mlp.gate.weight = mx.random.normal((4, 64)).astype(mx.bfloat16) * .1
        # Exercise window rollover, a partial final chunk, and a nonzero prefix.
        for prefix in ([], [31, 32, 33]):
            baseline = model.make_cache()
            candidate = model.make_cache()
            for state in (baseline, candidate):
                if prefix:
                    eval_prompt_cache(state, model(mx.array([prefix]), cache=state))
            tokens = list(range(23))
            for start in range(0, len(tokens), 3):
                eval_prompt_cache(baseline, model(mx.array([tokens[start:start + 3]]), cache=baseline))
            expert_cache.layers_seen.clear()
            support.prefill(model, tokens, candidate, 3, expert_cache, RuntimeConfig())
            self.assertEqual(expert_cache.layers_seen, [0] * 8)
            self.assertEqual([c.offset for c in candidate], [len(prefix) + 23] * 3)
            for left, right in zip(baseline, candidate):
                for x, y in zip(left.state, right.state):
                    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
            for token in (24, 25, 26):
                left = model(mx.array([[token]]), cache=baseline)
                right = model(mx.array([[token]]), cache=candidate)
                np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    def test_cache_only_attention_requires_state(self):
        attention = Attention(tiny_args(), False)
        with self.assertRaisesRegex(ValueError, 'KV cache'):
            attention(mx.zeros((1, 3, 64)), None, None, cache_only=True)

    def test_cache_only_attention_preserves_global_and_sliding_state(self):
        from mlx_lm.models.base import create_attention_mask
        from mlx_lm.models.cache import KVCache, RotatingKVCache
        from deepseek_v4_ssd.model import eval_prompt_cache
        for sliding in (False, True):
            attention = Attention(tiny_args(), sliding)
            baseline, candidate = [RotatingKVCache(max_size=8) if sliding else KVCache() for _ in range(2)]
            for length in (3, 7, 1, 5):
                x = mx.random.normal((1, length, 64))
                mask = create_attention_mask(x, baseline, window_size=8 if sliding else None)
                eval_prompt_cache([baseline], attention(x, mask, baseline))
                mask = create_attention_mask(x, candidate, window_size=8 if sliding else None)
                self.assertIsNone(attention(x, mask, candidate, cache_only=True))
                eval_prompt_cache([candidate])
                self.assertEqual(candidate.offset, baseline.offset)
                for left, right in zip(baseline.state, candidate.state):
                    np.testing.assert_array_equal(np.asarray(left), np.asarray(right))

    def test_layer_major_last_layer_only_populates_kv(self):
        from deepseek_v4_ssd.model import RuntimeConfig, eval_prompt_cache
        from deepseek_v4_ssd.model_support import get_support
        model = Model(tiny_args(), TinyExpertCache())
        baseline, candidate = model.make_cache(), model.make_cache()
        tokens = list(range(23))
        for start in range(0, len(tokens), 3):
            eval_prompt_cache(baseline, model(mx.array([tokens[start:start + 3]]), cache=baseline))
        last = model.layers[-1]
        mlp, output_projection = last.mlp, last.self_attn.o_proj
        def unused(*args, **kwargs):
            raise AssertionError('last Prefill layer must not compute discarded outputs')
        try:
            last.mlp = unused
            last.self_attn.o_proj = unused
            get_support("mimo-v2.6-flash-rl").prefill(
                model, tokens, candidate, 3, model.layers[1].mlp.switch_mlp.cache, RuntimeConfig())
        finally:
            last.mlp, last.self_attn.o_proj = mlp, output_projection
        for left, right in zip(baseline, candidate):
            for x, y in zip(left.state, right.state):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        np.testing.assert_array_equal(np.asarray(model(mx.array([[24]]), cache=baseline)),
                                      np.asarray(model(mx.array([[24]]), cache=candidate)))

    def test_layer_major_block_boundary_preserves_cache(self):
        from deepseek_v4_ssd.model import RuntimeConfig, eval_prompt_cache
        from deepseek_v4_ssd.model_support import get_support
        expert_cache = TinyExpertCache()
        model = Model(tiny_args(), expert_cache)
        for layer in model.layers[1:]:
            layer.mlp.gate.weight = mx.random.normal((4, 64)).astype(mx.bfloat16) * .1
        # 4096 is not divisible by 1000: block boundaries must stay aligned.
        for count, step in ((4103, 1000), (11, 8192)):
            tokens = [i % 128 for i in range(count)]
            baseline, candidate = model.make_cache(), model.make_cache()
            for start in range(0, count, step):
                eval_prompt_cache(baseline, model(mx.array([tokens[start:start + step]]), cache=baseline))
            get_support("mimo-v2.6-flash-rl").prefill(
                model, tokens, candidate, step, expert_cache, RuntimeConfig())
            for left, right in zip(baseline, candidate):
                self.assertEqual(left.offset, right.offset)
                for x, y in zip(left.state, right.state):
                    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
            np.testing.assert_array_equal(
                np.asarray(model(mx.array([[1]]), cache=baseline)),
                np.asarray(model(mx.array([[1]]), cache=candidate)))

    def test_layer_major_mid_prefill_cancel_and_fresh_request(self):
        from unittest.mock import patch
        from deepseek_v4_ssd.model import RuntimeConfig
        from deepseek_v4_ssd.model_support import get_support
        support = get_support("mimo-v2.6-flash-rl")
        cache = TinyExpertCache()
        model = Model(tiny_args(), cache)
        event = Event()
        original = cache.get_many
        def cancel_after_read(layer, selected):
            result = original(layer, selected)
            event.set()
            return result
        with patch.object(cache, "get_many", side_effect=cancel_after_read):
            with cancellation_scope(event), self.assertRaises(GenerationCancelled):
                support.prefill(model, list(range(10)), model.make_cache(), 3, cache, RuntimeConfig())
        event.clear()
        state = model.make_cache()
        with cancellation_scope(event):
            support.prefill(model, list(range(10)), state, 3, cache, RuntimeConfig())
            actual = model(mx.array([[10]]), cache=state)
            self.assertTrue(mx.all(mx.isfinite(actual)).item())
        self.assertEqual([c.offset for c in state], [11] * 3)

    def test_layer_major_validates_and_checks_cancellation(self):
        from deepseek_v4_ssd.model import RuntimeConfig
        from deepseek_v4_ssd.model_support import get_support
        support = get_support("mimo-v2.6-flash-rl")
        cache = TinyExpertCache()
        model = Model(tiny_args(), cache)
        state = model.make_cache()
        support.prefill(model, [], state, 3, cache, RuntimeConfig())
        self.assertEqual(cache.layers_seen, [])
        with self.assertRaisesRegex(ValueError, "step"):
            support.prefill(model, [1], state, 0, cache, RuntimeConfig())
        with self.assertRaisesRegex(ValueError, "cache"):
            support.prefill(model, [1], [], 3, cache, RuntimeConfig())
        event = Event()
        event.set()
        with cancellation_scope(event), self.assertRaises(GenerationCancelled):
            support.prefill(model, [1, 2, 3], state, 3, cache, RuntimeConfig())
        self.assertEqual(cache.layers_seen, [])

    def test_real_ssd_slots_eviction_and_continuation(self):
        from mlx.utils import tree_flatten
        args = tiny_args()
        memory_cache = TinyExpertCache()
        reference = Model(args, memory_cache)
        for layer in reference.layers[1:]:
            layer.mlp.gate.weight = mx.random.normal((4, 64)).astype(mx.bfloat16) * .1
        regions = []
        offset = 0
        for name, rows, cols in (("w1", 32, 64), ("w2", 64, 32), ("w3", 32, 64)):
            for suffix, dtype, shape in (("weight", "I8", (rows, cols // 2)),
                                          ("scale", "F8_E8M0", (rows, cols // 32))):
                size = shape[0] * shape[1]
                regions.append(Tensor(f"{name}.{suffix}", dtype, shape, offset, size))
                offset += size
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experts").mkdir()
            blob = bytearray()
            for expert in memory_cache.experts:
                for array in (expert.w1, expert.w1_scales, expert.w2, expert.w2_scales,
                              expert.w3, expert.w3_scales):
                    blob.extend(np.asarray(array).tobytes())
            for layer in range(2):
                (root / f"experts/layer_{layer:02d}.bin").write_bytes(blob)
            # Reuse the existing DeepSeek packed layout for this synthetic I/O
            # fixture only. No production MiMo manifest is accepted yet.
            installed = InstalledModel(root, "fixture", "fixture", 2, 4, 2, offset,
                                       (), tuple(regions))
            for ready in (False, True):
                disk_cache = ExpertCache(installed, slots=4, read_workers=2,
                                         ready_expert_decode=ready, eviction_policy="lru")
                try:
                    actual_model = Model(args, disk_cache)
                    actual_model.load_weights(tree_flatten(reference.parameters()), strict=True)
                    from deepseek_v4_ssd.model import RuntimeConfig, eval_prompt_cache
                    from deepseek_v4_ssd.model_support import get_support
                    state = actual_model.make_cache()
                    expected_state = reference.make_cache()
                    tokens = list(range(23))
                    for start in range(0, len(tokens), 3):
                        eval_prompt_cache(expected_state, reference(
                            mx.array([tokens[start:start + 3]]), cache=expected_state))
                    get_support("mimo-v2.6-flash-rl").prefill(
                        actual_model, tokens, state, 3, disk_cache, RuntimeConfig())
                    self.assertLessEqual(disk_cache.metrics.misses, 2 * args.n_routed_experts)
                    self.assertFalse(disk_cache._pinned_layers)
                    np.testing.assert_array_equal(
                        np.asarray(actual_model(mx.array([[24]]), cache=state)),
                        np.asarray(reference(mx.array([[24]]), cache=expected_state)))
                    state = actual_model.make_cache()
                    for i in range(16):
                        actual = actual_model(mx.array([[i]]), cache=state)
                        expected = reference(mx.array([list(range(i + 1))]))[:, -1:]
                        mx.eval(actual, expected)
                        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected),
                                                   rtol=3e-4, atol=2e-5)
                    self.assertGreater(disk_cache.metrics.misses, 0)
                finally:
                    mx.synchronize()
                    disk_cache.close()

    def test_input_embeddings_and_invalid_state(self):
        model = Model(tiny_args(), TinyExpertCache())
        tokens = mx.array([[1, 2, 3]])
        baseline = model(tokens)
        embedded = model(tokens, input_embeddings=model.model.embed_tokens(tokens))
        np.testing.assert_array_equal(np.asarray(baseline), np.asarray(embedded))
        with self.assertRaisesRegex(ValueError, "embeddings"):
            model(tokens, input_embeddings=mx.zeros((1, 2, 64)))
        with self.assertRaisesRegex(ValueError, "cache layer"):
            model(tokens, cache=[])
        with self.assertRaisesRegex(ValueError, "token IDs"):
            model(mx.array([1, 2]))

    def test_mlx_lm_generation_uses_embeddings_through_last_prompt_token(self):
        from mlx_lm.generate import generate_step
        model = Model(tiny_args(), TinyExpertCache())
        tokens = mx.array([1, 2, 3, 4])
        embeddings = model.model.embed_tokens(tokens)
        sampler = lambda logits: mx.argmax(logits, axis=-1)
        ordinary = list(generate_step(tokens, model, max_tokens=3, sampler=sampler,
                                      prefill_step_size=2))
        embedded = list(generate_step(tokens, model, max_tokens=3, sampler=sampler,
                                      prefill_step_size=2, input_embeddings=embeddings))
        self.assertEqual([int(t[0]) for t in ordinary], [int(t[0]) for t in embedded])
        for left, right in zip(ordinary, embedded):
            np.testing.assert_array_equal(np.asarray(left[1]), np.asarray(right[1]))

    def test_cancelled_request_does_not_enter_expert_cache(self):
        cache = TinyExpertCache()
        model = Model(tiny_args(), cache)
        event = Event()
        event.set()
        with cancellation_scope(event), self.assertRaises(GenerationCancelled):
            model(mx.array([[1]]), cache=model.make_cache())
        self.assertEqual(cache.layers_seen, [])
        event.clear()
        with cancellation_scope(event):
            output = model(mx.array([[1]]), cache=model.make_cache())
            self.assertTrue(mx.all(mx.isfinite(output)).item())


if __name__ == "__main__":
    unittest.main()
