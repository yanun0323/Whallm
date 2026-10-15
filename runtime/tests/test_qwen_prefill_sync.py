"""Small real decoder checks for opt-in prefill synchronization diagnostics."""
from contextlib import contextmanager, redirect_stderr
import copy
import io
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.cache import CacheList, KVCache

from deepseek_v4_ssd.event_log import request_scope
from deepseek_v4_ssd.cancellation import GenerationCancelled, cancellation_scope
from deepseek_v4_ssd.model_support import qwen
from deepseek_v4_ssd.qwen4_exp import DecoderLayer, ModelArgs
from deepseek_v4_ssd.qwen_pooled_cache import QSAIndexCache


class PrefillSyncTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(17)
        args = ModelArgs.from_dict(dict(hidden_size=8, num_hidden_layers=1,
            layer_types=['full_attention'], num_attention_heads=2,
            num_key_value_heads=1, head_dim=4, partial_rotary_factor=0.5,
            indexer_n_heads=1, indexer_kv_heads=1, indexer_head_dim=4,
            indexer_budget=4, indexer_compress_ratio=2, hc_count=2, hc_lowrank=2,
            num_experts=2, num_experts_per_tok=1, shared_expert_intermediate_size=4,
            moe_intermediate_size=4, vocab_size=32, ple_layer_ids=[]))
        self.released = []
        released = self.released

        class Experts:
            @contextmanager
            def reuse_layer_buffers(self):
                try:
                    yield
                finally:
                    released.append('buffers')

            @contextmanager
            def batched_layer(self, index):
                try:
                    yield
                finally:
                    released.append(index)

        self.experts = Experts()
        self.layer = DecoderLayer(args, 0, self.experts, None)
        self.layer.mlp.experts = lambda value, indices: mx.zeros((*indices.shape, value.shape[-1]))
        self.model = SimpleNamespace(model=SimpleNamespace(args=args,
            layers=[self.layer], embed_tokens=nn.Embedding(32, 8)),
            private_prompt='private-prompt-must-not-appear')
        self.cache = [CacheList(KVCache(), QSAIndexCache())]
        self.log = io.StringIO()
        self.environment = patch.dict(os.environ, {'WHALLM_QWEN_PREFILL_SYNC': '1',
            'DEEPSEEK_API_KEY': 'private-key-must-not-appear'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def run_prefill(self, cache=None, tokens=(11, 22), step=2):
        with redirect_stderr(self.log), request_scope("debug"):
            return qwen._qwen_layer_major_prefill(self.model, list(tokens),
                self.cache if cache is None else cache, step, self.experts)

    def test_marker_and_environment_are_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'qwen-prefill-sync.enabled'
            with patch.object(qwen, '_PREFILL_SYNC_MARKER', marker):
                os.environ.pop('WHALLM_QWEN_PREFILL_SYNC')
                self.assertFalse(qwen._prefill_sync_enabled())
                marker.touch()
                self.assertTrue(qwen._prefill_sync_enabled())
                for flag, expected in [('0', False), ('1', True), ('invalid', False)]:
                    os.environ['WHALLM_QWEN_PREFILL_SYNC'] = flag
                    self.assertEqual(qwen._prefill_sync_enabled(), expected)

    def test_disabled_path_has_no_extra_waits_or_logs(self):
        with patch.dict(os.environ, {'WHALLM_QWEN_PREFILL_SYNC': '0'}), \
                patch.object(qwen.mx, 'synchronize') as synchronize:
            self.run_prefill()
        synchronize.assert_not_called()
        self.assertEqual(self.log.getvalue(), '')

    def test_success_preserves_cached_output_and_has_bounded_logs(self):
        with patch.dict(os.environ, {'WHALLM_QWEN_PREFILL_SYNC': '0'}):
            self.run_prefill()
            expected_cache = copy.deepcopy(self.cache)
            expected = self.run_prefill(expected_cache, (3, 4, 5, 6))
        with patch.object(qwen.mx, 'synchronize', wraps=mx.synchronize) as synchronize:
            actual = self.run_prefill(self.cache, (3, 4, 5, 6))
        self.assertEqual(synchronize.call_count, 12)  # Six boundaries, two chunks.
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        np.testing.assert_array_equal(np.asarray(self.cache[0][0].keys),
                                      np.asarray(expected_cache[0][0].keys))
        np.testing.assert_array_equal(np.asarray(self.cache[0][1].keys),
                                      np.asarray(expected_cache[0][1].keys))
        self.assertEqual(len(self.log.getvalue().splitlines()), 2)
        self.assertIn('prefill sync diagnostic enabled', self.log.getvalue())
        self.assertIn('completed', self.log.getvalue())

    def test_eval_and_stream_failures_identify_the_unfinished_stage(self):
        stages = ('layer_input', 'attention_input', 'attention',
                  'experts_input', 'experts', 'layer_output')
        original_eval, original_sync = qwen.eval_prompt_cache, mx.synchronize
        for boundary in ('evaluate', 'synchronize'):
            for call, stage in enumerate(stages, 1):
                with self.subTest(boundary=boundary, stage=stage):
                    self.released.clear()
                    failure = RuntimeError('injected GPU failure')
                    evaluations = 0

                    def evaluate(*args):
                        nonlocal evaluations
                        evaluations += 1
                        if boundary == 'evaluate' and evaluations == call:
                            raise failure
                        return original_eval(*args)

                    def synchronize(*args):
                        if boundary == 'synchronize' and evaluations == call:
                            raise failure
                        return original_sync(*args)

                    cache = [CacheList(KVCache(), QSAIndexCache())]
                    with patch.object(qwen, 'eval_prompt_cache', side_effect=evaluate), \
                            patch.object(qwen.mx, 'synchronize', side_effect=synchronize), \
                            self.assertRaises(RuntimeError) as caught:
                        self.run_prefill(cache)
                    self.assertIs(caught.exception, failure)
                    notes = '\n'.join(failure.__notes__)
                    self.assertIn(f'stage={stage} ', notes)
                    self.assertIn(f'last_completed={stages[call-2] if call > 1 else "none"} ', notes)
                    self.assertIn('layer=0', notes)
                    self.assertIn('token_range=[0,2)', notes)
                    self.assertEqual(self.released, [0, 'buffers'])
                    for secret in ('private-prompt-must-not-appear', 'private-key-must-not-appear', '[11, 22]'):
                        self.assertNotIn(secret, notes + self.log.getvalue())

    def test_construction_failures_keep_the_current_stage(self):
        for stage, attribute in [('attention', 'self_attn'), ('experts', 'mlp')]:
            with self.subTest(stage=stage):
                failure = RuntimeError('injected construction failure')
                with patch.object(self.layer, attribute, side_effect=failure), \
                        self.assertRaises(RuntimeError) as caught:
                    self.run_prefill()
                self.assertIs(caught.exception, failure)
                self.assertIn(f'stage={stage} ', '\n'.join(failure.__notes__))

    def test_cancellation_does_not_get_error_notes_and_releases_scopes(self):
        cancellation = GenerationCancelled('cancelled')
        with patch.object(qwen.mx, 'synchronize', side_effect=cancellation), \
                self.assertRaises(GenerationCancelled) as caught:
            self.run_prefill()
        self.assertIs(caught.exception, cancellation)
        self.assertFalse(getattr(cancellation, '__notes__', []))
        self.assertEqual(self.released, [0, 'buffers'])
        self.assertNotIn('completed', self.log.getvalue())

    def test_cancel_during_wait_stops_before_next_component(self):
        cancelled = threading.Event()
        original_eval = qwen.eval_prompt_cache
        count = 0

        def evaluate(*args):
            nonlocal count
            result = original_eval(*args)
            count += 1
            if count == 2:
                cancelled.set()
            return result

        with cancellation_scope(cancelled), \
                patch.object(qwen, 'eval_prompt_cache', side_effect=evaluate), \
                patch.object(self.layer, 'self_attn') as attention, \
                self.assertRaises(GenerationCancelled):
            self.run_prefill()
        attention.assert_not_called()
        self.assertEqual(self.released, [0, 'buffers'])

    def test_direct_decoder_calls_do_not_enable_prefill_diagnostics(self):
        tokens = mx.array([[11, 22]])
        hidden = mx.tile(self.model.model.embed_tokens(tokens), (1, 1, 2))
        with patch.object(qwen.mx, 'synchronize') as synchronize:
            mx.eval(self.layer(hidden, tokens, None, self.cache[0]))
        synchronize.assert_not_called()


if __name__ == '__main__':
    unittest.main()
