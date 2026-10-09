"""Qwen MTP with image prompts. No full model or network required."""
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.generation import GeneratedPiece, GenerationOptions, ModelRuntime, RuntimeMetrics
from deepseek_v4_ssd.media import ModalSpan
from deepseek_v4_ssd.model import eval_prompt_cache
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.qwen_vision.inputs import ImagePrompt
from deepseek_v4_ssd.qwen_vision.positions import MultimodalRoPE, image_positions
from runtime.tests.test_qwen import FakeGreedyMTP, FakeGreedyTarget, FakeTargetCache
from runtime.tests.test_qwen_speed import bits, experts_fixture, tiny_args


def image_args():
    # Partial rotary width 16 needs sections summing to 8.
    return tiny_args(rope_parameters={'mrope_section': [3, 3, 2]})


class MTPLayerImageInputTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(7)
        _, cache = experts_fixture()
        self.args = image_args()
        self.model = qwen.MTPModel(self.args, cache)
        self.model.set_dtype(mx.float32)
        self.hidden = mx.random.normal((1, 6, 256)) * .1
        self.embedding = mx.random.normal((32, 64)) * .1
        self.tokens = mx.array([[1, 2, 3, 4, 5, 6]])
        # Two text tokens, a 2x2 merged image, then text.
        self.rope = MultimodalRoPE(image_positions(10, [ModalSpan(2, 4, 'a', (1, 4, 4))]), self.args)

    def advance(self, *, chunks=((0, 6),), **kwargs):
        cache = self.model.make_cache()
        outputs = []
        for start, end in chunks:
            extra = dict(kwargs)
            if 'next_embeddings' in extra:
                extra['next_embeddings'] = extra['next_embeddings'][:, start:end]
            outputs.append(self.model.advance(self.hidden[:, start:end], self.tokens[:, start:end],
                                              self.embedding, cache, **extra))
        output = mx.concatenate(outputs, axis=1)
        eval_prompt_cache([cache], output)
        return output

    def test_token_rows_as_embeddings_match_the_token_path_bit_for_bit(self):
        rows = mx.take(self.embedding, self.tokens, axis=0)
        np.testing.assert_array_equal(bits(self.advance(next_embeddings=rows)), bits(self.advance()))

    def test_image_embeddings_and_positions_reach_the_draft_layer(self):
        text = self.advance()
        features = mx.random.normal((1, 6, 64)) * .1
        self.assertGreater(np.max(np.abs(np.array(self.advance(next_embeddings=features) - text))), 1e-5)
        spatial = self.advance(rope_positions=self.rope)
        self.assertGreater(np.max(np.abs(np.array(spatial - text))), 1e-5)
        # Positions follow the draft cache offset, so chunking does not move them.
        chunked = self.advance(rope_positions=self.rope, chunks=((0, 2), (2, 5), (5, 6)))
        np.testing.assert_allclose(np.array(chunked), np.array(spatial), rtol=1e-4, atol=1e-5)

    def test_misaligned_image_embeddings_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'image embeddings'):
            self.model.advance(self.hidden, self.tokens, self.embedding, self.model.make_cache(),
                               next_embeddings=mx.zeros((1, 5, 64)))


class RecordingTarget(FakeGreedyTarget):
    def __init__(self, rope):
        super().__init__()
        self.rope_positions = rope
        self.calls = []

    def forward_with_hidden(self, input_ids, cache, capture=None, *, input_embeddings=None,
                            rope_positions=None):
        self.calls.append((np.asarray(input_ids)[0].tolist(),
                           None if input_embeddings is None else np.asarray(input_embeddings)[0, :, 0].tolist(),
                           rope_positions))
        return super().forward_with_hidden(input_ids, cache)


class RecordingMTP(FakeGreedyMTP):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = []

    def __call__(self, target_hidden, next_token_ids, embedding_weight, lm_head_weight, cache, *,
                 next_embeddings=None, rope_positions=None):
        self.calls.append((np.asarray(next_token_ids)[0].tolist(),
                           None if next_embeddings is None else np.asarray(next_embeddings)[0, :, 0].tolist(),
                           rope_positions))
        return super().__call__(target_hidden, next_token_ids, embedding_weight, lm_head_weight, cache)


class GenerateWithImageTests(unittest.TestCase):
    prompt = [1, 2, 3, 4, 5, 6, 7]

    def run_generation(self, target, mtp, **kwargs):
        return list(qwen.generate_mtp_tokens(self.prompt, target, mtp, [FakeTargetCache()],
                                             max_tokens=12, prefill_step_size=3, draft_tokens=2,
                                             **kwargs))

    def test_prompt_features_and_positions_follow_vllm_pairing_without_changing_tokens(self):
        rope = object()
        features = mx.array([[100.0 + index] for index in range(len(self.prompt))])
        for reject in (None, 10):
            target, mtp = RecordingTarget(rope), RecordingMTP(reject_input_token=reject)
            result = self.run_generation(target, mtp, input_embeddings=features)
            expected = self.run_generation(FakeGreedyTarget(), FakeGreedyMTP(reject_input_token=reject))
            self.assertEqual(result, expected)
            # The target receives the prompt features in order, chunk by chunk.
            prefill = [call for call in target.calls if call[1] is not None]
            self.assertEqual([call[0] for call in prefill], [[1, 2, 3], [4, 5, 6], [7]])
            self.assertEqual(sum((call[1] for call in prefill), []), [100.0 + i for i in range(7)])
            self.assertTrue(all(call[2] is rope for call in target.calls))
            self.assertGreater(len(target.calls), len(prefill))
            # Draft pair i takes the features of prompt[i + 1]; drafts after the prompt are text.
            paired = [call for call in mtp.calls if call[1] is not None]
            self.assertEqual(sum((call[0] for call in paired), []), self.prompt[1:])
            self.assertEqual(sum((call[1] for call in paired), []), [101.0 + i for i in range(6)])
            self.assertTrue(all(call[2] is rope for call in mtp.calls))
            self.assertGreater(len(mtp.calls), len(paired))

    def test_image_input_needs_positions_full_prompt_and_no_cached_prefix(self):
        features = mx.zeros((len(self.prompt), 1))
        with self.assertRaisesRegex(ValueError, 'positions'):
            self.run_generation(FakeGreedyTarget(), FakeGreedyMTP(), input_embeddings=features)
        with self.assertRaisesRegex(ValueError, 'prompt-aligned'):
            self.run_generation(RecordingTarget(object()), FakeGreedyMTP(),
                                input_embeddings=features[:-1])
        with self.assertRaisesRegex(ValueError, 'layer-major'):
            self.run_generation(RecordingTarget(object()), FakeGreedyMTP(), input_embeddings=features,
                                prefilled_hidden=mx.zeros((1, 6, 1)))


class RuntimeImageMTPTests(unittest.TestCase):
    def test_image_request_uses_mtp_with_a_fresh_cache_and_never_touches_prompt_cache(self):
        runtime = object.__new__(ModelRuntime)
        runtime.support = get_support('swift1.5-qwen3.8-flash-next')
        runtime.config = SimpleNamespace(prompt_cache_entries=2, prefill_step_size=128)
        runtime.installed = SimpleNamespace(maximum_context=1024)
        runtime.model = SimpleNamespace(mtp=SimpleNamespace(), dspark=None,
                                        args=SimpleNamespace(hidden_size=4))
        runtime.metrics = RuntimeMetrics()
        runtime.expert_cache = SimpleNamespace()
        runtime._generation_lock = threading.Lock()
        runtime._generation_stream = mx.new_stream(mx.default_device())
        spans = (ModalSpan(1, 4, 'a', (1, 4, 4)),)
        prepared = ImagePrompt((1, 248056, 248056, 248056, 248056, 2), mx.zeros((6, 4)), spans, 'id',
                               image_positions(6, spans))
        seen = {}

        def stream_mtp(_runtime, prompt, cache, mtp, options, step, prefilled, processors, **kwargs):
            seen.update(kwargs, prompt=prompt, cache=cache, prefilled=prefilled)
            kwargs['state']['finish'] = lambda: self.fail('image request stored an MTP prompt state')
            kwargs['on_prompt_end'](SimpleNamespace(tokens=list(prompt)))
            yield GeneratedPiece('ok', 3, len(prompt), 1, 'stop')

        fresh = [object()]
        with patch.object(ModelRuntime, '_stream_mtp', stream_mtp), \
             patch.object(runtime.support, 'new_cache', return_value=fresh), \
             patch.object(ModelRuntime, '_acquire_prompt_cache', side_effect=AssertionError('read prompt cache')), \
             patch.object(ModelRuntime, '_store_prompt_cache', side_effect=AssertionError('stored prompt cache')), \
             patch.object(ModelRuntime, '_expert_metrics', return_value=None), \
             patch.object(RuntimeMetrics, 'start'), patch.object(RuntimeMetrics, 'finish'):
            pieces = list(runtime.stream(prepared, GenerationOptions(max_tokens=4, temperature=0.0)))
        self.assertEqual([piece.text for piece in pieces], ['ok'])
        self.assertIs(seen['prepared'], prepared)
        self.assertIs(seen['cache'], fresh)
        self.assertIsNone(seen['cached'])
        self.assertIsNone(seen['prefilled'])
        self.assertEqual(seen['prompt'], list(prepared.token_ids))


if __name__ == '__main__':
    unittest.main()
