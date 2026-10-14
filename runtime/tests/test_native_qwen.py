"""The native Qwen runtime: sampling, MTP rounds, prompt cache choice and catalog dispatch.

Engine-free tests use a fake engine. One test runs the real engine on the installed
Swift1.5 model and is skipped when the library or the model is missing.
"""
from __future__ import annotations

import os
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
import numpy as np
from mlx_lm.sample_utils import make_logits_processors

from deepseek_v4_ssd import native_qwen
from deepseek_v4_ssd.dspark import sampling_logprobs
from deepseek_v4_ssd.generation import GenerationOptions, RuntimeMetrics
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import open_runtime, parse_model_catalog
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.native_qwen import NativeQwenRuntime, Sampler

NATIVE = "swift1.5-qwen3.8-flash-next-native"
QWEN_NATIVE = "qwen3.8-flash-next-native"


def mlx_distribution(logits, history, options):
    """The MLX runtime's distribution: logits processors, then sampling_logprobs."""
    processors = make_logits_processors(**{
        key: value for key, value in (("presence_penalty", options.presence_penalty),
                                      ("repetition_penalty", options.repetition_penalty))
        if value not in (0.0, 1.0)})
    x = mx.array(logits[None])
    for processor in processors:
        x = processor(mx.array(history, dtype=mx.int32), x)
    result = sampling_logprobs(x, options.temperature, options.top_p, options.top_k, options.min_p)[0]
    return np.asarray(result.astype(mx.float32))


class SamplerTest(unittest.TestCase):
    def test_distribution_matches_mlx_sampling(self):
        rng = np.random.default_rng(3)
        vocab = 4_000
        cases = [
            GenerationOptions(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5),
            GenerationOptions(temperature=1.0, top_p=0.95, top_k=20),
            GenerationOptions(temperature=0.6, top_p=0.9, top_k=0),
            GenerationOptions(temperature=0.8, top_p=1.0, top_k=0, min_p=0.05, repetition_penalty=1.2),
            GenerationOptions(temperature=0.5, top_p=1.0, top_k=0),
        ]
        for options in cases:
            with self.subTest(options=options):
                logits = (rng.standard_normal(vocab) * 3).astype(np.float32)
                history = [int(v) for v in rng.integers(0, vocab, 30)]
                expected = np.exp(mlx_distribution(logits, history, options))
                ours = Sampler(options, rng).distribution(logits, history).probabilities(vocab)
                np.testing.assert_allclose(ours, expected, atol=2e-5)

    def test_greedy_uses_penalized_argmax(self):
        logits = np.zeros(10, np.float32)
        logits[3], logits[5] = 2.0, 1.9
        sampler = Sampler(GenerationOptions(temperature=0.0, presence_penalty=0.5), np.random.default_rng(0))
        self.assertEqual(sampler.sample(sampler.distribution(logits, [3])), 5)
        self.assertEqual(sampler.sample(sampler.distribution(logits, [7])), 3)

    def test_rejection_sampling_keeps_the_target_distribution(self):
        # One draft position: whatever the draft distribution, the emitted first
        # token must follow the target distribution.
        rng = np.random.default_rng(11)
        options = GenerationOptions(temperature=1.0, top_p=1.0, top_k=0)
        sampler = Sampler(options, rng)
        vocab = 5
        target_logits = np.log(np.array([0.5, 0.2, 0.15, 0.1, 0.05], np.float32))
        draft_logits = np.log(np.array([0.1, 0.1, 0.6, 0.1, 0.1], np.float32))
        target = sampler.distribution(target_logits, [])
        draft = sampler.distribution(draft_logits, [])
        counts = np.zeros(vocab)
        for _ in range(20_000):
            token = sampler.sample(draft)
            accepted, following = sampler.verify([token], [draft], [target, target], vocab)
            counts[token if accepted else following] += 1
        np.testing.assert_allclose(counts / counts.sum(), np.exp(target.logp), atol=0.015)


class FakeState:
    def __init__(self, tokens, nbytes=100):
        self.tokens = tuple(tokens)
        self.nbytes = nbytes
        self.closed = False

    def close(self):
        self.closed = True


class FakeEngine:
    """A deterministic model: the next token is (last + 1) % vocab; the draft layer
    predicts (last + 1) except every third position, where it is wrong."""
    vocab = 50
    has_mtp = True
    contract = "fake"

    def __init__(self):
        self.consumed: list[int] = []
        self.verified: list[int] = []
        self.drafted = 0
        self.all_drafts = 0
        self.calls: list[str] = []

    def _logits(self, last):
        logits = np.zeros(self.vocab, np.float32)
        logits[(last + 1) % self.vocab] = 5.0
        return logits

    @property
    def tokens(self):
        return tuple(self.consumed)

    def reset(self):
        self.calls.append("reset")
        self.consumed = []

    def restore(self, state):
        self.calls.append(f"restore {len(state.tokens)}")
        self.consumed = list(state.tokens)

    def snapshot(self):
        return FakeState(self.consumed)

    def eval(self, tokens, logit_rows=1):
        self.consumed += list(tokens)
        return np.stack([self._logits(self.consumed[-1])]) if logit_rows else np.empty((0, self.vocab))

    def draft(self, token):
        self.drafted += 1
        self.all_drafts += 1
        logits = self._logits(token)
        if self.all_drafts % 3 == 0:  # every third draft of the run is wrong
            logits = self._logits(token + 1)
        return logits

    def verify(self, tokens):
        self.verified = list(tokens)
        rows = []
        context = list(self.consumed)
        for token in tokens:
            context.append(token)
            rows.append(self._logits(context[-1]))
        return np.stack(rows)

    def commit(self, keep):
        self.consumed += self.verified[:keep]
        self.verified, self.drafted = [], 0

    def counters(self):
        return SimpleNamespace(expert_hits=0, expert_misses=0, experts_seeded=0, read_seconds=0.0,
                               resident=0, slots=10)


def fake_runtime(config=None):
    runtime = object.__new__(NativeQwenRuntime)
    runtime.engine = FakeEngine()
    runtime.config = config or RuntimeConfig(prompt_cache_entries=2, mtp_enabled=True)
    runtime.metrics = RuntimeMetrics()
    runtime._entries = []
    runtime._persist_later = []
    runtime._directory = None
    return runtime


class RuntimeTest(unittest.TestCase):
    def run_tokens(self, runtime, prompt, count, mtp):
        options = GenerationOptions(max_tokens=count, temperature=0.0)
        sampler = Sampler(options, np.random.default_rng(0))
        return list(runtime._tokens(prompt, 0, None, options, sampler, mtp))

    def test_mtp_rounds_give_the_tokens_of_plain_decoding(self):
        prompt = [1, 2, 3]
        plain = self.run_tokens(fake_runtime(), prompt, 20, mtp=False)
        self.assertEqual(plain, [(4 + i) % 50 for i in range(20)])
        runtime = fake_runtime()
        self.assertEqual(self.run_tokens(runtime, prompt, 20, mtp=True), plain)
        snapshot = runtime.metrics.snapshot()
        self.assertGreater(snapshot["mtp_rounds"], 0)
        self.assertGreater(snapshot["mtp_accepted_tokens"], 0)
        self.assertLess(snapshot["mtp_accepted_tokens"], snapshot["mtp_proposed_tokens"])
        # Every yielded token but the last was consumed, in order.
        self.assertEqual(runtime.engine.consumed[:len(prompt) + 19], prompt + plain[:19])

    def test_cache_uses_the_longest_prefix_that_leaves_a_token(self):
        runtime = fake_runtime()
        runtime._store(FakeState([1, 2, 3]), persist=False)
        runtime._store(FakeState([1, 2, 3, 4, 5]), persist=False)
        runtime._store(FakeState([9, 9]), persist=False)
        self.assertEqual(runtime._continue_from_cache([1, 2, 3, 4, 5, 6]), 5)
        self.assertEqual(runtime.engine.calls[-1], "restore 5")
        # The whole prompt cached: the shorter entry leaves one token to read.
        self.assertEqual(runtime._continue_from_cache([1, 2, 3, 4, 5]), 3)
        self.assertEqual(runtime._continue_from_cache([7, 8]), 0)
        self.assertEqual(runtime.engine.calls[-1], "reset")

    def test_cache_continues_the_engine_state_without_restoring(self):
        runtime = fake_runtime()
        runtime.engine.consumed = [1, 2, 3, 4]
        self.assertEqual(runtime._continue_from_cache([1, 2, 3, 4, 5]), 4)
        self.assertEqual(runtime.engine.calls, [])

    def test_cache_off_and_eviction(self):
        runtime = fake_runtime(RuntimeConfig(prompt_cache_entries=0))
        runtime.engine.consumed = [1, 2]
        self.assertEqual(runtime._continue_from_cache([1, 2, 3]), 0)
        runtime = fake_runtime(RuntimeConfig(prompt_cache_entries=1, prompt_cache_memory_gib=1))
        states = [FakeState([i, i + 1]) for i in range(3)]
        for state in states:
            runtime._store(state, persist=False)
        self.assertEqual([e.state for e in runtime._entries], states[1:])
        self.assertTrue(states[0].closed)
        duplicate = FakeState([2, 3])
        self.assertIs(runtime._store(duplicate, persist=False).state, states[2])
        self.assertTrue(duplicate.closed)


class CatalogTest(unittest.TestCase):
    def raw(self, **runtime):
        return {"version": 1, "models": [{
            "id": NATIVE, "alias": None, "path": "/tmp/swift", "model_kind": NATIVE,
            "runtime": {**asdict(RuntimeConfig(mtp_enabled=True, qwen_8bit_common_tensors=True,
                                               native_automatic_expert_cache=True)), **runtime},
            "defaults": {"max_tokens": 8192, "temperature": 0.7, "top_p": 0.8, "top_k": 20},
            "warmup_prompt_path": None}]}

    def test_native_kind_opens_the_native_runtime(self):
        spec = parse_model_catalog(self.raw())[0]
        self.assertTrue(spec.runtime.native_automatic_expert_cache)
        with mock.patch.object(NativeQwenRuntime, "open", return_value="native") as opened:
            self.assertEqual(open_runtime(spec), "native")
        opened.assert_called_once_with("/tmp/swift", spec.runtime, NATIVE)

    def test_qwen_native_kind_opens_the_native_runtime(self):
        raw = self.raw()
        raw["models"][0].update(id=QWEN_NATIVE, model_kind=QWEN_NATIVE, path="/tmp/qwen")
        spec = parse_model_catalog(raw)[0]
        self.assertIsInstance(get_support(QWEN_NATIVE), type(get_support(NATIVE)))
        with mock.patch.object(NativeQwenRuntime, "open", return_value="native") as opened:
            self.assertEqual(open_runtime(spec), "native")
        opened.assert_called_once_with("/tmp/qwen", spec.runtime, QWEN_NATIVE)
        with self.assertRaises(Exception):
            raw["models"][0]["runtime"]["qwen_sorted_expert_prefill"] = True
            parse_model_catalog(raw)

    def test_native_kind_rejects_mlx_only_options(self):
        for name, value in (("qwen_sorted_expert_prefill", True), ("qwen_qsa_selected_kernel", True),
                            ("dspark_enabled", True), ("qwen_quantized_kv", True)):
            with self.subTest(name=name):
                with self.assertRaises(Exception):
                    parse_model_catalog(self.raw(**{name: value}))
        with self.assertRaises(ValueError):
            get_support(NATIVE).load(None, None, None, None, None)


MODEL = Path(os.environ.get("WHALLM_SWIFT_MODEL",
                            Path.home() / ".dsmodel" / "swift1.5-qwen3.8-flash-next-mxfp4.dsv4"))


class RealEngineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (MODEL / "manifest.json").is_file():
            raise unittest.SkipTest("The Swift1.5 model is not installed.")
        try:
            native_qwen._library()
        except RuntimeError as error:
            raise unittest.SkipTest(str(error))
        config = RuntimeConfig(slots=1024, read_workers=8, mtp_enabled=True, prompt_cache_entries=2)
        cls.runtime = NativeQwenRuntime.open(str(MODEL), config, NATIVE)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "runtime"):
            cls.runtime.close()

    def test_greedy_tokens_match_the_mlx_engine(self):
        # Recorded from the MLX Qwen engine (BF16 resident weights, MTP off), 2026-10-14.
        prompt = [760, 6511, 314, 9338, 369, 11751, 13, 561, 6511, 314, 6124, 369]
        expected = [25358, 13, 561, 6511, 314, 14898, 369, 21047, 13, 561, 6511, 314]
        options = GenerationOptions(max_tokens=12, temperature=0.0, top_p=1.0, top_k=0)
        self.assertEqual([p.token for p in self.runtime.stream(prompt, options)], expected)

    def test_prompt_cache_answers_the_same_prompt_again(self):
        prompt = [760, 6511, 314, 9338, 369, 11751, 13] * 20
        options = GenerationOptions(max_tokens=8, temperature=0.0, top_p=1.0, top_k=0)
        first = [p.token for p in self.runtime.stream(prompt, options)]
        self.assertEqual(self.runtime.metrics.snapshot()["prompt_cache_reused_tokens"], 0)
        second = [p.token for p in self.runtime.stream(prompt, options)]
        self.assertEqual(second, first)
        self.assertEqual(self.runtime.metrics.snapshot()["prompt_cache_reused_tokens"], len(prompt) - 1)


if __name__ == "__main__":
    unittest.main()
