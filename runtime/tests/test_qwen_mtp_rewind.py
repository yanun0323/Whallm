"""Qwen MTP verification rewind on a tiny full model. No checkpoint required."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.manifest import NGram
from deepseek_v4_ssd.model import _fork_prompt_cache, eval_prompt_cache
from runtime.tests.test_qwen import FakeGreedyMTP
from runtime.tests.test_qwen_speed import experts_fixture, tiny_args


def arrays(cache):
    items = cache.caches if hasattr(cache, "caches") else [cache]
    for item in items:
        for value in item.state:
            if isinstance(value, mx.array):
                yield value


def fork(cache):
    copied, values = _fork_prompt_cache(cache)
    if values:
        mx.eval(*values)
    return copied


class ScriptedDraft(FakeGreedyMTP):
    """Drafts a known continuation, except at the listed sequence positions."""

    def __init__(self, sequence, wrong):
        super().__init__()
        self.sequence, self.wrong = sequence, wrong

    def __call__(self, hidden, token_ids, embedding, head, cache):
        start, count = cache.offset, token_ids.shape[1]
        cache.offset += count
        logits = np.full((1, count, 32), -1_000.0, dtype=np.float32)
        for position in range(count):
            # The pair at draft offset i holds sequence[i + 1] and predicts the next.
            index = start + position + 2
            token = self.sequence[index] if index < len(self.sequence) else 0
            logits[0, position, (token + (index in self.wrong)) % 32] = 1_000.0
        return mx.array(logits), mx.zeros((1, count, 1))


class VerificationRewindTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / "ngram.bin"
        np.resize(np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8), 200).tofile(path)
        self.store = qwen.NGramStore(path, NGram("ngram.bin", "F8_E4M3", 2, 1, 100, (0,) * 16, (100,) * 16))
        _, self.experts = experts_fixture()

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def model(self, pooled):
        mx.random.seed(23)
        args = tiny_args(num_hidden_layers=4, linear_num_value_heads=4, linear_num_key_heads=2,
                         linear_key_head_dim=16, linear_value_head_dim=16, ple_embed_dim=32,
                         ple_conv_kernel_size=2, ple_layer_ids=(2,), eos_token_id=31,
                         indexer_budget=4,
                         layer_types=("linear_attention",) * 3 + ("full_attention",))
        model = qwen.Model(args, self.experts, self.store)
        model.set_dtype(mx.float32)
        model.pooled_index_cache = pooled
        self.assertEqual([layer.layer_type for layer in model.layers],
                         ["linear_attention"] * 3 + ["full_attention"])
        self.assertIsNotNone(model.layers[1].ple)
        self.assertTrue(model.supports_verification_rewind)
        return model

    def test_rewind_matches_direct_prefix_forward_without_routed_experts(self):
        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        verified_ids = mx.array([[10, 11, 12, 13]])
        for pooled in (False, True):
            model = self.model(pooled)
            prefilled = model.make_cache()
            eval_prompt_cache(prefilled, *model.forward_with_hidden(prompt, prefilled))
            for keep, in_place in ((keep, in_place) for keep in (1, 2, 3, 4) for in_place in (False, True)):
                with self.subTest(pooled=pooled, keep=keep, in_place=in_place):
                    target, inputs = fork(prefilled), []
                    verified = model.verification_cache(target) if in_place else fork(prefilled)
                    if in_place:
                        # Attention rows are appended to the target's own KV; only
                        # linear-attention state, which cannot be trimmed, is cloned.
                        self.assertEqual([a is b for a, b in zip(verified, target)],
                                         [False] * 3 + [True])
                    _, verified_hidden = model.forward_with_hidden(verified_ids, verified, inputs)
                    eval_prompt_cache(verified, verified_hidden)
                    with patch.object(qwen.StreamingExperts, "__call__",
                                      side_effect=AssertionError("routed experts used")):
                        model.rewind_verification(target, verified, inputs, verified_ids, keep)
                    eval_prompt_cache(target)

                    reference = fork(prefilled)
                    _, reference_hidden = model.forward_with_hidden(verified_ids[:, :keep], reference)
                    eval_prompt_cache(reference, reference_hidden)
                    np.testing.assert_allclose(np.asarray(verified_hidden[:, keep - 1]),
                                               np.asarray(reference_hidden[:, -1]), atol=2e-5, rtol=2e-5)
                    for got, want in zip(target, reference):
                        self.assertEqual(got.size() if hasattr(got, "size") else None,
                                         want.size() if hasattr(want, "size") else None)
                        for a, b in zip(arrays(got), arrays(want), strict=True):
                            self.assertEqual(a.shape, b.shape)
                            np.testing.assert_allclose(np.asarray(a.astype(mx.float32)),
                                                       np.asarray(b.astype(mx.float32)), atol=2e-5, rtol=2e-5)
                    # Continuing from either cache yields the same next distribution.
                    following = mx.array([[14, 15]])
                    a, _ = model.forward_with_hidden(following, target)
                    b, _ = model.forward_with_hidden(following, reference)
                    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=5e-5, rtol=5e-5)

    def test_generation_verifies_in_place_with_the_tokens_and_state_of_a_copied_fork(self):
        prompt, count = [1, 2, 3, 4, 5, 6, 7, 8, 9], 14
        model = self.model(True)
        # Plain greedy decoding supplies the continuation the drafts are scripted from.
        cache = model.make_cache()
        logits, _ = model.forward_with_hidden(mx.array([prompt]), cache)
        sequence = list(prompt)
        for _ in range(count):
            sequence.append(int(mx.argmax(logits[0, -1]).item()))
            logits, _ = model.forward_with_hidden(mx.array([sequence[-1:]]), cache)
            eval_prompt_cache(cache, logits)

        def generate():
            target, rounds = model.make_cache(), []
            # Rounds draft positions (10, 11), (13, 14), (15, 16), (16, 17), (19, 20):
            # a wrong 14 is a partial acceptance, a wrong 15 a zero acceptance.
            tokens = [token for token, _ in qwen.generate_mtp_tokens(
                prompt, model, ScriptedDraft(sequence, {14, 15}), target, max_tokens=count,
                prefill_step_size=4, draft_tokens=2, zero_acceptance_limit=32,
                record_round=lambda proposed, accepted, *_: rounds.append(accepted))]
            eval_prompt_cache(target)
            return tokens, target, rounds

        tokens, target, rounds = generate()
        with patch.object(qwen.Model, "verification_cache", lambda _, cache: fork(cache)):
            copied_tokens, copied_target, copied_rounds = generate()
        self.assertEqual(tokens, sequence[len(prompt):])
        self.assertEqual((tokens, rounds), (copied_tokens, copied_rounds))
        self.assertEqual(set(rounds), {0, 1, 2})
        for got, want in zip(target, copied_target):
            self.assertEqual(got.size() if hasattr(got, "size") else None,
                             want.size() if hasattr(want, "size") else None)
            for a, b in zip(arrays(got), arrays(want), strict=True):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))

    def test_in_place_verification_does_not_write_into_a_stored_clone(self):
        model = self.model(True)
        target = model.make_cache()
        eval_prompt_cache(target, *model.forward_with_hidden(
            mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]]), target))
        stored = copy.deepcopy(target)  # how clone_cache keeps a prompt-end state

        def buffers():
            # Whole allocations, including the spare rows an append writes.
            return [np.array(getattr(branch, name))
                    for branch in stored[3].caches for name in ("keys", "values")]

        before, inputs, verified_ids = buffers(), [], mx.array([[10, 11, 12, 13]])
        verified = model.verification_cache(target)
        _, hidden = model.forward_with_hidden(verified_ids, verified, inputs)
        eval_prompt_cache(verified, hidden)
        model.rewind_verification(target, verified, inputs, verified_ids, 2)
        eval_prompt_cache(target)
        self.assertEqual((stored[3].size(), target[3].size()), (9, 11))
        for got, want in zip(buffers(), before, strict=True):
            np.testing.assert_array_equal(got, want)

    def test_rewind_eval_waits_for_all_linear_state(self):
        prompt = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        verified_ids = mx.array([[10, 11, 12, 13]])
        for pooled in (False, True):
            with self.subTest(pooled=pooled):
                model = self.model(pooled)
                prefilled = model.make_cache()
                eval_prompt_cache(prefilled, *model.forward_with_hidden(prompt, prefilled))
                target, verified, inputs = fork(prefilled), fork(prefilled), []
                _, hidden = model.forward_with_hidden(verified_ids, verified, inputs)
                eval_prompt_cache(verified, hidden)
                model.rewind_verification(target, verified, inputs, verified_ids, 2)
                linear_state = [value for cache in target[:3] for value in arrays(cache)]
                self.assertEqual(len(linear_state), 8)

                # Use the same wait as generate_mtp_tokens, before stopping its timer.
                with patch("deepseek_v4_ssd.model.mx.eval", wraps=mx.eval) as evaluate:
                    count, size = eval_prompt_cache(target, hidden[:, 1:2])
                evaluated = {id(value) for value in evaluate.call_args.args}
                self.assertTrue({id(value) for value in linear_state} <= evaluated)
                self.assertGreaterEqual(count, len(linear_state))
                self.assertGreaterEqual(size, sum(value.nbytes for value in linear_state))

    def test_rewind_rejects_mismatched_inputs(self):
        model = self.model(False)
        cache = model.make_cache()
        with self.assertRaises(ValueError):
            model.rewind_verification(cache, cache, [], mx.array([[1, 2]]), 1)
        with self.assertRaises(ValueError):
            model.rewind_verification(cache, cache, [mx.zeros((1, 2, 1))] * 4, mx.array([[1, 2]]), 0)


if __name__ == "__main__":
    unittest.main()
