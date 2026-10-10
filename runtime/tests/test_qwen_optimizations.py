import mmap
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from threading import Event
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import CacheList, KVCache

from deepseek_v4_ssd.cancellation import GenerationCancelled, cancellation_scope
from deepseek_v4_ssd.manifest import NGram
from deepseek_v4_ssd.model import RuntimeConfig, _fork_prompt_cache, _cache_arrays, eval_prompt_cache
from deepseek_v4_ssd.model_manager import validate_runtime_config, _parse_runtime
from deepseek_v4_ssd.model_support.state import persistence_cache_state, restore_persistence_cache
from deepseek_v4_ssd.qwen4_exp import ModelArgs, QSAAttention, NGramStore, GroupRMSNorm, generate_mtp_tokens
from deepseek_v4_ssd.qwen_ngram_lookup import request_rows
from deepseek_v4_ssd.qwen_pooled_cache import QSAPooledIndexCache, QSAPooledQuantizedIndexCache
from deepseek_v4_ssd.qwen_quantized_cache import QSAQuantizedCache
from deepseek_v4_ssd.qwen_mtp_policy import MTPDraftPolicy
from deepseek_v4_ssd.qwen_phase_budget import QwenPhaseBudget
from runtime.tests.test_qwen import FakeGreedyTarget, FakeGreedyMTP, FakeTargetCache


def bits(value):
    mx.eval(value)
    return np.asarray(value.view(mx.uint8))


class QwenOptimizationTests(unittest.TestCase):
    def test_defaults_and_catalog_backward_compatibility(self):
        from dataclasses import asdict
        config = RuntimeConfig()
        raw = asdict(config)
        for name in ("qwen_pooled_index_cache", "qwen_ngram_lookup_optimized", "qwen_compile_tensor_ops", "qwen_phase_memory"):
            self.assertFalse(raw.pop(name))
        self.assertEqual(raw.pop("qwen_mtp_draft_tokens"), 2)
        self.assertEqual(raw.pop("qwen_mtp_zero_acceptance_limit"), 32)
        self.assertEqual(raw.pop("qwen_mtp_draft_min_probability"), 0.0)
        self.assertEqual(raw.pop("qwen_mtp_first_draft_min_probability"), 0.0)
        self.assertEqual(_parse_runtime(raw, "runtime", "qwen3.8-flash-next"), config)

    def test_invalid_research_settings(self):
        for name, values in {
            "qwen_mtp_draft_tokens": (0, 6, True, 2.0),
            "qwen_mtp_zero_acceptance_limit": (0, 33, True, 2.0),
            "qwen_mtp_draft_min_probability": (-0.1, 1, 1.5, True, "0.5", None),
            "qwen_mtp_first_draft_min_probability": (-0.1, 1, 1.5, True, "0.5", None),
            "qwen_pooled_index_cache": (1, None),
            "qwen_ngram_lookup_optimized": (1, None),
            "qwen_compile_tensor_ops": (1, None),
            "qwen_phase_memory": (1, None),
        }.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    validate_runtime_config(replace(RuntimeConfig(), **{name: value}))

    def test_pooled_keys_compute_only_new_complete_blocks(self):
        cache = QSAPooledIndexCache()
        calls = []
        def pool(raw, first):
            calls.append((first, raw.shape[1]))
            return raw.reshape(1, -1, 2, 4).mean(axis=2)
        for count in (1, 2, 1, 2, 1):
            raw = mx.ones((1, 1, count, 4))
            keys, _ = cache.update_and_fetch(raw, raw)
            pooled = cache.pooled(keys[:, 0], 2, pool)
            mx.eval(pooled)
        self.assertEqual(calls, [(0, 2), (1, 2), (2, 2)])
        self.assertEqual(cache.nbytes, cache.keys.nbytes + cache.values.nbytes + pooled.nbytes)
        self.assertTrue(any(array is pooled for array in _cache_arrays([cache])))

    def test_qsa_append_fork_trim_restore_match_baseline(self):
        for quantized in (False, True):
            for dtype in (mx.float32, mx.bfloat16, mx.float16):
                with self.subTest(quantized=quantized, dtype=dtype):
                    self._check_qsa(quantized, dtype)

    def _check_qsa(self, quantized, dtype):
        mx.random.seed(44)
        args = ModelArgs(hidden_size=32, num_attention_heads=2, num_key_value_heads=1,
                         head_dim=32, indexer_n_heads=2, indexer_kv_heads=1,
                         indexer_head_dim=32, indexer_budget=4, indexer_compress_ratio=2,
                         partial_rotary_factor=0.5, max_position_embeddings=128)
        attention = QSAAttention(args)
        attention.set_dtype(dtype)
        hidden = mx.random.normal((1, 16, 32)).astype(dtype)
        index = QSAPooledQuantizedIndexCache(4, 32) if quantized else QSAPooledIndexCache()
        baseline = CacheList(KVCache(), QSAQuantizedCache(4, 32) if quantized else KVCache())
        candidate = CacheList(KVCache(), index)
        start = 0
        for count in (1, 2, 3, 1, 2):
            chunk = hidden[:, start:start + count]
            expected, actual = attention(chunk, baseline), attention(chunk, candidate)
            np.testing.assert_array_equal(bits(actual), bits(expected))
            eval_prompt_cache([candidate])
            start += count
        saved = persistence_cache_state([candidate])
        fork, copied = _fork_prompt_cache([candidate])
        mx.eval(*copied)
        original = bits(index.pooled_keys).copy()
        attention(hidden[:, 9:13], fork[0])
        eval_prompt_cache(fork)
        np.testing.assert_array_equal(bits(index.pooled_keys), original)
        np.testing.assert_array_equal(bits(attention(hidden[:, 9:10], candidate)),
                                      bits(attention(hidden[:, 9:10], baseline)))
        # Roll back across both complete and incomplete blocks, then overwrite.
        before_trim = bits(index.pooled_keys).copy()
        for item in (*candidate.caches, *baseline.caches):
            item.trim(3)
        # Only complete blocks before the new offset survive, unchanged.
        kept = index.offset // args.indexer_compress_ratio
        self.assertEqual(index.pooled_keys.shape[1], kept)
        np.testing.assert_array_equal(bits(index.pooled_keys), before_trim[:, :kept])
        np.testing.assert_array_equal(bits(attention(hidden[:, 12:15], candidate)),
                                      bits(attention(hidden[:, 12:15], baseline)))
        # Restore into a used cache: no stale derived keys may survive.
        restore_persistence_cache([candidate], saved)
        restore_persistence_cache([baseline], saved)
        self.assertIsNone(index.pooled_keys)
        np.testing.assert_array_equal(bits(attention(hidden[:, 9:12], candidate)),
                                      bits(attention(hidden[:, 9:12], baseline)))

    def test_ngram_all_fp8_values_reordering_and_scale_are_bit_exact(self):
        descriptor = NGram("ngram.bin", "F8_E4M3", 1, 1, 256, (0,) * 16, (256,) * 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ngram.bin"
            np.arange(256, dtype=np.uint8).tofile(path)
            valid = np.array([i for i in range(256) if i not in (127, 255)])
            rows = np.concatenate([valid[::-1], valid, valid]).reshape(3, -1)
            for scale in (0.5, 0.123456789, 5.3):
                baseline = NGramStore(path, descriptor, scale)
                candidate = NGramStore(path, descriptor, scale, optimized=True)
                for ids in (rows, np.array(3), np.empty((1, 0, 16), dtype=np.int64)):
                    np.testing.assert_array_equal(bits(candidate.lookup(ids)), bits(baseline.lookup(ids)))
                for invalid in (127, 255, -1, 256):
                    with self.assertRaises(ValueError):
                        candidate.lookup(np.array([invalid]))

    def test_ngram_rows_request_each_of_their_pages_once(self):
        class Mapped:
            def __init__(self):
                self.requests = []

            def madvise(self, option, start, length):
                self.requests.append((option, start // mmap.PAGESIZE, length))

        # Rows of three quarters of a page: row 0 is on page 0, row 1 on pages
        # 0 and 1, row 5 on pages 3 and 4.
        mapped = Mapped()
        request_rows(mapped, np.array([[5, 0], [1, 5]]), 3 * mmap.PAGESIZE // 4)
        self.assertEqual(mapped.requests,
                         [(mmap.MADV_WILLNEED, page, mmap.PAGESIZE) for page in (0, 1, 3, 4)])
        request_rows(mapped, np.empty((1, 0, 16), dtype=np.int64), 160)
        self.assertEqual(len(mapped.requests), 4)

    def test_ngram_mapped_lookups_request_their_rows_and_return_the_same_values(self):
        descriptor = NGram("ngram.bin", "F8_E4M3", 1, 1, 256, (0,) * 16, (256,) * 16)
        ids = np.array([[3, 200], [3, 64]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ngram.bin"
            np.arange(256, dtype=np.uint8).tofile(path)
            for options in ({}, {"optimized": True}, {"io_backend": "pread"}):
                with self.subTest(**options):
                    store = NGramStore(path, descriptor, 0.5, **options)
                    self.addCleanup(store.close)
                    with patch("deepseek_v4_ssd.qwen_ngram_lookup.request_rows",
                               wraps=request_rows) as request:
                        values = store.lookup(ids)
                    if "io_backend" in options:
                        request.assert_not_called()
                    else:
                        (_, requested, row_bytes), _ = request.call_args
                        np.testing.assert_array_equal(requested, ids)
                        self.assertEqual((request.call_count, row_bytes), (1, 1))
                    # FP8 0x03, 0xC8 and 0x40, scaled by one half.
                    np.testing.assert_array_equal(
                        np.asarray(values.astype(mx.float32)).reshape(2, 2),
                        np.array([[0.5 * 3 / 512, -0.5 * 4.0], [0.5 * 3 / 512, 0.5 * 2.0]],
                                 dtype=np.float32))

    def test_mtp_policy_consecutive_rejections_and_limits(self):
        policy = MTPDraftPolicy(2, 2)
        self.assertEqual([policy.limit(i) for i in (0, 1, 2, 8)], [0, 0, 1, 2])
        self.assertFalse(policy.observe(0))
        self.assertFalse(policy.observe(1))
        self.assertFalse(policy.observe(0))
        self.assertTrue(policy.observe(0))
        self.assertTrue(MTPDraftPolicy().observe(0))
        for depth, limit in ((True, 1), (0, 1), (6, 1), (2, 0), (2, 33)):
            with self.assertRaises(ValueError):
                MTPDraftPolicy(depth, limit)

    def test_mtp_depth_retry_outputs_and_cache_offsets(self):
        for depth in (1, 2, 3, 5):
            for retries in (1, 2, 3):
                for reject in (None, 3, 4):
                    with self.subTest(depth=depth, retries=retries, reject=reject):
                        target, mtp = FakeGreedyTarget(), FakeGreedyMTP(reject_input_token=reject)
                        cache, rounds = [FakeTargetCache()], []
                        result = list(generate_mtp_tokens([1, 2], target, mtp, cache,
                            max_tokens=15, prefill_step_size=2, draft_tokens=depth,
                            zero_acceptance_limit=retries, record_round=lambda *args: rounds.append(args)))
                        self.assertEqual([token for token, _ in result], list(range(3, 18)))
                        self.assertEqual(cache[0].offset, 16)
                        self.assertTrue(all(round_[0] <= depth for round_ in rounds))
                        if reject == 3:
                            self.assertEqual(rounds[0][-1], retries == 1)
                            self.assertEqual(len(rounds) > 1, retries > 1)

    def test_mtp_confidence_gate_keeps_outputs_and_cache_offsets(self):
        class UncertainMTP(FakeGreedyMTP):
            """Right about the next token, but nearly flat after the listed inputs."""
            def __init__(self, uncertain):
                super().__init__()
                self.uncertain, self.calls = uncertain, 0

            def __call__(self, hidden, token_ids, *rest):
                self.calls += 1
                logits, hidden = super().__call__(hidden, token_ids, *rest)
                if int(np.asarray(token_ids)[0, -1]) in self.uncertain:
                    logits = logits * 1e-4
                return logits, hidden

        expected = list(range(3, 18))
        for uncertain, proposed in (
            (set(), [3, 3, 3, 1]),
            ({5}, [2, 3, 3, 2]),            # stops before the third draft of round one
            ({3}, [0, 3, 3, 3]),            # no draft at all: one ordinary target step
            (set(range(32)), [0] * 13),   # the 15th token takes the ordinary last-token step
        ):
            with self.subTest(uncertain=sorted(uncertain)[:3]):
                target, mtp = FakeGreedyTarget(), UncertainMTP(uncertain)
                cache, rounds, state = [FakeTargetCache()], [], {}
                result = list(generate_mtp_tokens([1, 2], target, mtp, cache, max_tokens=15,
                    prefill_step_size=2, draft_tokens=3, zero_acceptance_limit=32,
                    draft_min_probability=0.5, record_round=lambda *args: rounds.append(args),
                    state=state))
                self.assertEqual([token for token, _ in result], expected)
                self.assertEqual([round_[0] for round_ in rounds], proposed)
                self.assertTrue(all(round_[1] == round_[0] and not round_[-1] for round_ in rounds))
                # After the deferred pairs are flushed, the target holds every token
                # before the last yielded one and the draft layer one pair fewer.
                snapshot = state["finish"]()
                self.assertEqual(snapshot.tokens, [1, 2, *expected[:-1]])
                self.assertEqual(cache[0].offset, 16)
                self.assertEqual(mtp.cache.offset, 15)
        with self.assertRaises(ValueError):
            list(generate_mtp_tokens([1, 2], FakeGreedyTarget(), FakeGreedyMTP(), [FakeTargetCache()],
                max_tokens=2, prefill_step_size=2, draft_min_probability=1.0))

    def test_mtp_first_draft_limit_is_separate_from_the_chain_limit(self):
        class GradedMTP(FakeGreedyMTP):
            """Right about the next token with top probability 0.6 at every position."""
            def __call__(self, hidden, token_ids, *rest):
                logits, hidden = super().__call__(hidden, token_ids, *rest)
                # One logit at log(46.5) against 31 zeros: 46.5 / 77.5 = 0.6.
                return mx.where(logits > 0, 3.8395, 0.0), hidden

        for chain, first, proposed in (
            (0.0, 0.0, [3, 3, 3, 1]),      # gate off
            (0.5, 0.0, [1] * 7),           # 0.6 passes, 0.36 does not
            (0.7, 0.0, [0] * 13),          # the first draft already fails
            (0.7, 0.5, [1] * 7),           # looser first position, strict chain
            (0.3, 0.7, [0] * 13),          # stricter first position
            (0.0, 0.5, [3, 3, 3, 1]),      # first-position limit only
            (0.2, 0.5, [3, 3, 3, 1]),      # 0.216 still passes at depth three
        ):
            with self.subTest(chain=chain, first=first):
                target, mtp = FakeGreedyTarget(), GradedMTP()
                cache, rounds, state = [FakeTargetCache()], [], {}
                result = list(generate_mtp_tokens([1, 2], target, mtp, cache, max_tokens=15,
                    prefill_step_size=2, draft_tokens=3, zero_acceptance_limit=32,
                    draft_min_probability=chain, first_draft_min_probability=first,
                    record_round=lambda *args: rounds.append(args), state=state))
                self.assertEqual([token for token, _ in result], list(range(3, 18)))
                self.assertEqual([round_[0] for round_ in rounds], proposed)
                state["finish"]()
                self.assertEqual((cache[0].offset, mtp.cache.offset), (16, 15))

    def test_mtp_confidence_gate_is_off_by_default_and_keeps_rejections_exact(self):
        for gate in (0.0, 0.5):
            for reject in (3, 4, 6):
                with self.subTest(gate=gate, reject=reject):
                    target, mtp = FakeGreedyTarget(), FakeGreedyMTP(reject_input_token=reject)
                    cache = [FakeTargetCache()]
                    result = list(generate_mtp_tokens([1, 2], target, mtp, cache, max_tokens=15,
                        prefill_step_size=2, draft_tokens=3, zero_acceptance_limit=32,
                        draft_min_probability=gate))
                    self.assertEqual([token for token, _ in result], list(range(3, 18)))
                    self.assertEqual(cache[0].offset, 16)

    def test_mtp_cancel_after_anchor_does_not_advance_target(self):
        event = Event()
        cache = [FakeTargetCache()]
        with cancellation_scope(event):
            generator = generate_mtp_tokens([1, 2], FakeGreedyTarget(), FakeGreedyMTP(), cache,
                max_tokens=15, prefill_step_size=2, draft_tokens=2, zero_acceptance_limit=2)
            self.assertEqual(next(generator), (3, False))
            event.set()
            with self.assertRaises(GenerationCancelled):
                next(generator)
        self.assertEqual(cache[0].offset, 2)

    def test_compiled_norm_outputs_and_updated_weights(self):
        mx.random.seed(71)
        norm = GroupRMSNorm(32, 8, 1e-6)
        for dtype in (mx.float32, mx.bfloat16, mx.float16):
            for count in (1, 7, 32):
                for scale in (1.0, 0.13):
                    norm.weight = mx.random.normal((32,)).astype(dtype) * scale
                    value = mx.random.normal((1, count, 32)).astype(dtype)
                    norm.compiled = False
                    baseline = norm(value)
                    norm.compiled = True
                    candidate = norm(value)
                    mx.eval(baseline, candidate)
                    np.testing.assert_array_equal(bits(candidate), bits(baseline))

    def test_phase_planner_never_increases_existing_ceiling(self):
        for reserve in (0, 7, 300):
            plan = QwenPhaseBudget(1027, 10, 6, reserve)
            self.assertEqual(plan.decode_slots, 102)
            self.assertLessEqual(plan.prefill_expert_bytes + reserve, plan.expert_ceiling_bytes)
            self.assertLessEqual(plan.decode_expert_bytes, plan.expert_ceiling_bytes)
            self.assertGreaterEqual(plan.prefill_slots, 6)
        for args in ((100, 10, 6, 41), (100, 10, 6, 101), (100, 10, 6, -1), (100, 10, 6, True)):
            with self.assertRaises(ValueError):
                QwenPhaseBudget(*args)


if __name__ == "__main__":
    unittest.main()
