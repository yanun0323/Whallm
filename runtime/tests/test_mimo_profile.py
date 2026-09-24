import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd.expert_cache import CacheMetrics
from deepseek_v4_ssd.mimo.model import DecoderLayer, Model
from deepseek_v4_ssd.model import RuntimeConfig, eval_prompt_cache
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.mimo.profile import MiMoProfile, layer_metrics
from runtime.tests.test_mimo_model import TinyExpertCache, tiny_args


class CountingCache(TinyExpertCache):
    def __init__(self):
        super().__init__()
        self.metrics = CacheMetrics()

    def metrics_snapshot(self):
        from dataclasses import replace
        return replace(self.metrics)

    def get_many(self, layer, selected):
        self.metrics.misses += len(set(selected))
        self.metrics.bytes_read += 100 * len(set(selected))
        return super().get_many(layer, selected)

    def iter_ready(self, layer, selected):
        # Deliberately inside consumption, not generator creation.
        self.metrics.misses += len(set(selected))
        self.metrics.bytes_read += 100 * len(set(selected))
        yield from super().iter_ready(layer, selected)


class MiMoProfileTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(17)
        self.cache = CountingCache()
        self.model = Model(tiny_args(), self.cache)
        for layer in self.model.layers[1:]:
            layer.mlp.gate.weight = mx.random.normal((4, 64)).astype(mx.bfloat16) * .1
        self.runtime = SimpleNamespace(model=self.model, expert_cache=self.cache,
                                       support=get_support('mimo-v2.6-flash-rl'))

    def test_host_and_sync_exact_outputs_layer_mapping_and_restoration(self):
        original = DecoderLayer.__call__
        tokens = mx.array([[1, 2, 3]])
        expected = np.asarray(self.model(tokens, cache=self.model.make_cache()))
        for mode in ('host', 'sync'):
            profile = MiMoProfile(mode)
            before = self.cache.metrics.bytes_read
            try:
                profile.attach(self.runtime)
                profile.begin_phase('prefill', 0)
                state = self.model.make_cache()
                actual = self.model(tokens, cache=state)
                eval_prompt_cache(state, actual)
                np.testing.assert_array_equal(np.asarray(actual), expected)
                profile.end_phase()
                profile.begin_phase('decode', 0)
                actual = self.model(mx.array([[4]]), cache=state)
                eval_prompt_cache(state, actual)
                profile.end_phase()
                rows = profile.report()['rows']
                for phase in ('prefill', 'decode'):
                    decoders = [r for r in rows if r['phase'] == phase and r['component'] == 'decoder']
                    self.assertEqual([r['layer'] for r in decoders], [0, 1, 2])
                    self.assertEqual([r['expert_layer'] for r in decoders], [None, 0, 1])
                    self.assertEqual([r['attention_kind'] for r in decoders], ['global', 'sliding', 'global'])
                    self.assertTrue(all(r['calls'] == 1 for r in decoders))
                self.assertEqual(sum(r['counters']['expert_bytes_read'] for r in rows
                                     if r['component'] == 'decoder'), self.cache.metrics.bytes_read - before)
                self.assertTrue(any(r['component'] == 'kv_update' for r in rows))
                flat = layer_metrics(profile.report())
                self.assertEqual(len(flat), 6)
                self.assertEqual(sum(r['logical_expert_bytes'] for r in flat), self.cache.metrics.bytes_read - before)
                self.assertTrue(all(r['miss_fraction'] is None for r in flat if r['layer'] == 0))
                self.assertTrue(all(r['attention_inclusive_seconds'] > 0 for r in flat))
                # Host mode must not accidentally introduce any explicit barriers.
                if mode == 'host':
                    with patch('mlx.core.synchronize', side_effect=AssertionError('host fence')):
                        self.model(mx.array([[5]]), cache=state)
            finally:
                profile.close()
            self.assertIs(DecoderLayer.__call__, original)

    def test_layer_major_and_cancelled_scope_cleanup(self):
        profile = MiMoProfile('host')
        original = DecoderLayer.__call__
        try:
            profile.attach(self.runtime)
            profile.begin_phase('prefill', 2)
            self.runtime.support.prefill(self.model, list(range(11)), self.model.make_cache(),
                                         3, self.cache, RuntimeConfig())
            rows = profile.report()['rows']
            self.assertEqual([r['calls'] for r in rows if r['component'] == 'decoder'], [4, 4, 4])
            self.assertTrue(any(r['component'] == 'allocator_clear' for r in rows))
            self.assertEqual(sum(r['counters']['expert_bytes_read'] for r in rows
                                 if r['component'] == 'decoder'), self.cache.metrics.bytes_read)
            from deepseek_v4_ssd.cancellation import GenerationCancelled, cancellation_scope
            event = threading.Event()
            event.set()
            with cancellation_scope(event), self.assertRaises(GenerationCancelled):
                self.model(mx.array([[1]]), cache=self.model.make_cache())
            self.assertFalse(profile.stack)
            self.assertEqual(profile.layer, -1)
        finally:
            profile.close()
        self.assertIs(DecoderLayer.__call__, original)

    def test_rejects_other_models_before_installing_hooks(self):
        profile = MiMoProfile()
        with self.assertRaisesRegex(ValueError, 'MiMo'):
            profile.attach(SimpleNamespace(model=object()))
        self.assertFalse(profile.patches)


if __name__ == '__main__':
    unittest.main()
