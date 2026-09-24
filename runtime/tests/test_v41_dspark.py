import unittest
from unittest.mock import patch
from dataclasses import replace
import mlx.core as mx
from runtime.tests.test_v41_prompt_cache import tiny_model
from deepseek_v4_ssd.deepseek_v41.dspark import DSpark
from deepseek_v4_ssd.dspark import generate_tokens, _target_sequence
from deepseek_v4_ssd.model import forward_with_hidden


class V41DSparkTests(unittest.TestCase):
    def test_drafts_verify_and_rejected_branches_preserve_committed_state(self):
        main = tiny_model()
        draft = DSpark(main.args, block_size=3, noise_token_id=15,
                       target_layers=(1, 2, 3), markov_rank=8, expert_count=2, topk=1)
        prompt = [1, 2, 3, 4, 5, 6, 7]
        reference = main.make_cache()
        logits = main(mx.array([prompt]), reference)
        expected = []
        for _ in range(8):
            token = int(mx.argmax(logits[:, -1], axis=-1).item())
            expected.append(token)
            logits = main(mx.array([[token]]), reference)
        rounds = []
        cache = main.make_cache()
        actual = list(generate_tokens(prompt, main, draft, cache, max_tokens=8,
                        prefill_step_size=3, temperature=0, top_p=1,
                        fallback_enabled=False,
                        record_round=lambda *args: rounds.append(args)))
        self.assertEqual([int(value[0]) for value in actual], expected)
        self.assertTrue(rounds)
        # A speculative fork leaves the original position and Engram history intact.
        before = cache[0].persistence_state()
        _target_sequence(main, [1, 2], cache, (1, 2, 3))
        self.assertEqual(cache[0].offset, before['offset'])
        self.assertTrue(mx.array_equal(mx.array(cache[0].cache.engram_ids), before['engram_ids']).item())

    def test_hidden_capture_and_draft_context_snapshot(self):
        main = tiny_model()
        draft = DSpark(main.args, block_size=3, noise_token_id=15,
                       target_layers=(1, 2, 3), markov_rank=8, expert_count=2, topk=1)
        cache = main.make_cache()
        logits, hidden = forward_with_hidden(main, mx.array([[1, 2, 3, 4, 5]]), cache, draft.target_layers)
        self.assertEqual(hidden.shape, (1, 5, main.args.dim * 3))
        draft.prefill_context(hidden, 0)
        saved = draft.cache_state()
        _, hidden = forward_with_hidden(main, mx.array([[6, 7]]), cache, draft.target_layers)
        first = draft.draft(main, 8, hidden, 5, 0, 1, 0)
        draft.restore_cache_state(saved)
        second = draft.draft(main, 8, hidden, 5, 0, 1, 0)
        self.assertEqual(first.tokens, second.tokens)
        self.assertEqual(saved[0][1], 5)
        self.assertEqual(draft.mtp[0].attn.offset, 7)
        with self.assertRaisesRegex(ValueError, 'position'):
            draft.prefill_context(hidden, 2)

    def test_runtime_layer_major_matches_chunked_and_recovers_after_cancel(self):
        import json
        from dataclasses import asdict
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from tokenizers import Tokenizer, models
        from transformers import PreTrainedTokenizerFast
        from deepseek_v4_ssd.generation import ModelRuntime, GenerationOptions
        from deepseek_v4_ssd.model import RuntimeConfig
        from deepseek_v4_ssd.model_support import get_support
        from deepseek_v4_ssd.expert_cache import CacheMetrics
        from contextlib import nullcontext
        model = tiny_model()
        model.dspark = DSpark(model.args, block_size=3, noise_token_id=15,
                              target_layers=(1, 2, 3), markov_rank=8, expert_count=2, topk=1)
        backend = Tokenizer(models.WordLevel({str(i): i for i in range(16)}, unk_token='0'))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='0')
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        options = GenerationOptions(max_tokens=8, temperature=0, seed=17)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps(asdict(model.args)))
            installed = SimpleNamespace(root=root, model_kind='deepseek-v4.1',
                model_id='fixture/v41', revision='fixture', format_version=3, maximum_context=32)
            def runtime(layer_major):
                config = RuntimeConfig(dspark_enabled=True, layer_major_prefill_threshold=1,
                    v41_layer_major_prefill=layer_major, layer_major_prefill=layer_major,
                    batched_expert_prefill=False, prompt_cache_entries=0, prefill_step_size=3)
                with patch('deepseek_v4_ssd.generation.load_model',
                           return_value=(model, SimpleNamespace(close=lambda: None, metrics_snapshot=CacheMetrics,
                                                              capture_expert_unions=nullcontext))), \
                     patch('deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained', return_value=tokenizer):
                    return ModelRuntime(installed, config)
            with runtime(False) as control:
                expected = [p.token for p in control.stream(prompt, options)]
            with runtime(True) as candidate:
                for _ in range(2):
                    self.assertEqual([p.token for p in candidate.stream(prompt, options)], expected)
                    self.assertTrue(candidate.metrics.snapshot()['layer_major_prefill'])
                pending = candidate.stream(prompt, options)
                next(pending)
                pending.close()
                self.assertEqual([p.token for p in candidate.stream(prompt, options)], expected)
        support = get_support('deepseek-v4.1')
        support.validate_config(RuntimeConfig(dspark_enabled=True, v41_layer_major_prefill=True,
                                             v41_next_layer_prefetch=True))
        # V4.1 DSpark always reuses the prompt cache, so the experimental flag is accepted.
        support.validate_config(RuntimeConfig(dspark_enabled=True, dspark_prompt_cache=True))
        for setting in ('v41_ced_prefill', 'dspark_sequential_verification'):
            with self.assertRaises(ValueError):
                support.validate_config(RuntimeConfig(dspark_enabled=True, **{setting: True}))
