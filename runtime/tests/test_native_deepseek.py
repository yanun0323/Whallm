"""DeepSeek V4.1 on the native engine: catalog wiring, settings and DSpark rounds.

The DSpark tests drive NativeDeepSeekRuntime._tokens with a fake engine, so they run
without the model or the engine library."""
from __future__ import annotations

import unittest
from dataclasses import asdict
from unittest import mock

import numpy as np

from deepseek_v4_ssd.generation import GenerationOptions, RuntimeMetrics
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import open_runtime, parse_model_catalog
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.native_deepseek import DraftBackoff, NativeDeepSeekRuntime
from deepseek_v4_ssd.native_qwen import Sampler

KIND = "deepseek-v4.1"
VOCAB = 16


def target(token: int) -> int:
    """The fake model's greedy next token."""
    return (3 * token + 1) % VOCAB


def one_hot(token: int) -> np.ndarray:
    row = np.zeros(VOCAB, np.float32)
    row[token] = 10.0
    return row


class FakeEngine:
    """Engine stand-in: the model always predicts target(); drafts follow `draft_rule`."""

    has_dspark = True
    draft_block = 5
    vocab = VOCAB

    def __init__(self, draft_rule, confidence=0.9):
        self.draft_rule = draft_rule
        self.confidence = confidence
        self.tokens: list[int] = []
        self.calls: list[tuple] = []
        self.verifying = None

    def eval(self, tokens, logit_rows=1):
        assert self.verifying is None
        self.tokens += list(tokens)
        self.calls.append(("eval", len(tokens)))
        return np.stack([one_hot(target(t)) for t in tokens[len(tokens) - logit_rows:]])

    def draft(self, anchor):
        self.calls.append(("draft", anchor))

    def draft_step(self, index, previous):
        return one_hot(self.draft_rule(previous)), self.confidence

    def verify(self, tokens):
        assert self.verifying is None
        self.verifying = len(self.tokens)
        self.tokens += list(tokens)
        self.calls.append(("verify", len(tokens)))
        return np.stack([one_hot(target(t)) for t in tokens])

    def commit(self, keep):
        del self.tokens[self.verifying + keep:]
        self.verifying = None


def runtime(engine, threshold=0.6) -> NativeDeepSeekRuntime:
    rt = NativeDeepSeekRuntime.__new__(NativeDeepSeekRuntime)
    rt.engine = engine
    rt.config = RuntimeConfig(dspark_enabled=True, dspark_confidence_threshold=threshold, prompt_cache_entries=0)
    rt.metrics = RuntimeMetrics()
    rt.metrics.start(3, 0, 3, False, mock.MagicMock(), dspark_enabled=True)
    return rt


def generate(engine, count, speculative=True, threshold=0.6):
    rt = runtime(engine, threshold)
    sampler = Sampler(GenerationOptions(temperature=0, top_p=1, top_k=0), np.random.default_rng(0))
    options = GenerationOptions(max_tokens=count, temperature=0)
    return list(rt._tokens([1, 2, 3], 0, None, options, sampler, speculative)), rt


def plain(count):
    out, token = [], target(3)
    for _ in range(count):
        out.append(token)
        token = target(token)
    return out


class DSparkRoundTests(unittest.TestCase):
    def test_greedy_tokens_do_not_depend_on_the_drafts(self):
        rules = {"always right": target, "always wrong": lambda t: (target(t) + 1) % VOCAB,
                 "wrong after 5": lambda t: (target(t) + 1) % VOCAB if t == 5 else target(t)}
        for name, rule in rules.items():
            for count in (1, 2, 7, 23):
                with self.subTest(rule=name, count=count):
                    engine = FakeEngine(rule)
                    tokens, _ = generate(engine, count)
                    self.assertEqual(tokens, plain(count))
                    # The engine consumed every yielded token but the last.
                    self.assertEqual(engine.tokens, [1, 2, 3] + tokens[:-1])

    def test_rounds_verify_the_anchor_and_up_to_five_drafts(self):
        engine = FakeEngine(target)
        tokens, rt = generate(engine, 20)
        verified = [n for kind, n in engine.calls if kind == "verify"]
        self.assertTrue(verified)
        self.assertTrue(all(2 <= n <= 6 for n in verified))
        snapshot = rt.metrics.snapshot()
        self.assertEqual(snapshot["dspark_rounds"], len(verified))
        self.assertEqual(snapshot["dspark_accepted_tokens"], snapshot["dspark_proposed_tokens"])

    def test_low_confidence_stops_drafting(self):
        engine = FakeEngine(target, confidence=0.5)
        tokens, _ = generate(engine, 10, threshold=0.6)
        self.assertEqual(tokens, plain(10))
        # No draft reached the threshold: each round verifies the anchor alone.
        self.assertTrue(all(n == 1 for kind, n in engine.calls if kind == "verify"))

    def test_without_dspark_every_token_is_one_eval(self):
        engine = FakeEngine(target)
        tokens, _ = generate(engine, 8, speculative=False)
        self.assertEqual(tokens, plain(8))
        self.assertFalse([c for c in engine.calls if c[0] in ("draft", "verify")])

    def test_rejected_rounds_back_off(self):
        engine = FakeEngine(lambda t: (target(t) + 1) % VOCAB)
        tokens, _ = generate(engine, 40)
        self.assertEqual(tokens, plain(40))
        drafts = sum(1 for c in engine.calls if c[0] == "draft")
        # Two rejected rounds, then waits of 1, 2, 4, 8 and 16 tokens between pairs.
        self.assertLess(drafts, 15)


class BackoffTests(unittest.TestCase):
    def test_waits_double_up_to_16_and_reset_on_acceptance(self):
        backoff = DraftBackoff()
        skipped = []
        for _ in range(6):
            backoff.observe(0)
            backoff.observe(0)
            count = 0
            while not backoff.draft_now():
                count += 1
            skipped.append(count)
        self.assertEqual(skipped, [1, 2, 4, 8, 16, 16])
        backoff.observe(1)
        backoff.observe(0)
        self.assertTrue(backoff.draft_now())
        backoff.observe(0)
        self.assertFalse(backoff.draft_now())


class CatalogTests(unittest.TestCase):
    def raw(self, **runtime):
        return {"version": 1, "models": [{
            "id": "deepseek-v4.1-flash", "alias": None, "path": "/tmp/v41", "model_kind": KIND,
            "runtime": {**asdict(RuntimeConfig(dspark_enabled=True, native_automatic_expert_cache=True)), **runtime},
            "defaults": {"max_tokens": 8192, "temperature": 0.2, "top_p": 0.98, "top_k": 0},
            "warmup_prompt_path": None}]}

    def test_catalog_opens_the_native_runtime(self):
        self.assertTrue(get_support(KIND).descriptor.supports("nativeEngine"))
        spec = parse_model_catalog(self.raw())[0]
        with mock.patch.object(NativeDeepSeekRuntime, "open", return_value="native") as opened:
            self.assertEqual(open_runtime(spec), "native")
        opened.assert_called_once_with("/tmp/v41", spec.runtime, KIND)

    def test_mlx_only_options_are_refused(self):
        for name in ("v41_packed_kv", "v41_packed_index", "v41_candidate_index", "deepseek_ane_prefill",
                     "dspark_sequential_verification", "qwen_8bit_common_tensors"):
            with self.subTest(name=name), self.assertRaises(Exception):
                parse_model_catalog(self.raw(**{name: True}))
        # MLX tuning that does not change the numbers is accepted and unused.
        parse_model_catalog(self.raw(layer_major_prefill=False, v41_layer_major_prefill=False,
                                     batched_expert_prefill=False, read_workers=4))


if __name__ == "__main__":
    unittest.main()
