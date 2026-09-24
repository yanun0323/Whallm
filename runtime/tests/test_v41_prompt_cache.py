from __future__ import annotations

import json
import unittest
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from deepseek_v4_ssd.deepseek_v41.config import ModelArgs
from deepseek_v4_ssd.deepseek_v41.model import Model
from deepseek_v4_ssd.deepseek_v41_ssd import DeepSeekV41ForCausalLM, DeepSeekV41PromptCache
from deepseek_v4_ssd.expert_cache import CacheMetrics
from deepseek_v4_ssd.generation import ModelRuntime, GenerationOptions, _PromptCacheEntry
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.model_support.state import _encode_cache_state, _decode_cache_state


def tiny_model():
    mx.random.seed(41)
    args = ModelArgs(vocab_size=16, dim=64, n_layers=4, moe_inter_dim=32,
                     n_heads=2, head_dim=32, rope_head_dim=4, q_lora_rank=32,
                     o_lora_rank=32, o_groups=2, window_size=4,
                     n_routed_experts=2, n_shared_experts=1, n_activated_experts=1,
                     compress_ratios=(2, 2, 1, 1), kv_source_layers=(0, 2),
                     index_source_layers=(0, 2), index_n_heads=2, index_head_dim=32,
                     index_topk=32, original_seq_len=32, max_seq_len=32, hc_mult=1,
                     engram_layer_ids=(1,), engram_num_embeddings=(59,),
                     engram_vocab_size=16, engram_n_heads=1, engram_head_dim=32,
                     engram_compressed_vocab_size=16)
    core = Model(args, token_map=list(range(16)))
    # Nonzero Engram weights make a lost token history affect continuation logits.
    core.layers[1].engram.embed.weight = mx.random.normal((59, 32)) * 0.1
    core.layers[1].engram.embed.scale = mx.ones((59, 1))
    mx.eval(core.parameters())
    return DeepSeekV41ForCausalLM(core)


class V41PromptCacheTests(unittest.TestCase):
    def setUp(self):
        self.model = tiny_model()
        self.support = get_support('deepseek-v4.1')

    def test_corrupt_disk_prefix_recovers_and_generates_the_cold_output(self):
        backend = Tokenizer(models.WordLevel({str(i): i for i in range(16)}, unk_token='0'))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='0')
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        options = GenerationOptions(max_tokens=3, temperature=0)
        for all_bad in (False, True):
            with self.subTest(all_bad=all_bad), TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'config.json').write_text(json.dumps(asdict(self.model.args)))
                installed = SimpleNamespace(root=root, model_kind='deepseek-v4.1',
                                            model_id='fixture/v41', revision='fixture',
                                            format_version=3, maximum_context=32)
                def open_runtime(disk):
                    config = RuntimeConfig(prompt_cache_entries=2 if disk else 0,
                                           persistent_prompt_cache=disk,
                                           prompt_cache_directory=root / 'cache', prefill_step_size=3)
                    with patch('deepseek_v4_ssd.generation.load_model',
                               return_value=(self.model, SimpleNamespace(close=lambda: None))), \
                         patch('deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained', return_value=tokenizer):
                        return ModelRuntime(installed, config)
                with open_runtime(False) as cold:
                    expected = [p.token for p in cold.stream(prompt, options)]
                with open_runtime(True) as warm:
                    self.assertEqual([p.token for p in warm.stream(prompt, options)], expected)
                with open_runtime(True) as recovered:
                    for entry in recovered._persistent_prompt_caches:
                        if all_bad or len(entry.tokens) > 3:
                            entry.path.write_bytes(b'corrupt fixture')
                    self.assertEqual([p.token for p in recovered.stream(prompt, options)], expected)
                    self.assertEqual(recovered.metrics.snapshot()['prompt_cache_reused_tokens'], 0 if all_bad else 3)
                    # Follow-up requests still work after persistence rescans.
                    self.assertEqual([p.token for p in recovered.stream(prompt, options)], expected)

    def test_clone_and_disk_round_trip_continue_at_partial_group_and_ring_wrap(self):
        original = [DeepSeekV41PromptCache(self.model.model.make_cache(max_seq_len=4, dtype=mx.bfloat16))]
        prefix = mx.array([[1, 2, 3, 4, 5, 6, 7]])
        mx.eval(self.model(prefix, cache=original))
        cloned = self.support.clone_cache(original)
        state = self.support.snapshot_cache(original)
        arrays = {}
        schema = _encode_cache_state(state, arrays)
        with TemporaryDirectory() as directory:
            path = str(Path(directory) / 'cache.safetensors')
            mx.save_safetensors(path, arrays, {'state': json.dumps(schema)})
            saved, metadata = mx.load(path, return_metadata=True)
            restored = self.model.make_cache()
            self.support.restore_cache(restored, _decode_cache_state(json.loads(metadata['state']), saved))
        original_ids = original[0].cache.engram_ids.copy()
        for tokens in ([[8]], [[9, 10, 11]], [[12]]):
            expected = self.model(mx.array(tokens), cache=original)
            for branch in (cloned, restored):
                actual = self.model(mx.array(tokens), cache=branch)
                mx.eval(expected, actual)
                np.testing.assert_array_equal(np.array(actual), np.array(expected))
        # Independent branches must never overwrite the shared saved prefix.
        np.testing.assert_array_equal(np.array(state[0]['engram_ids']), original_ids)
        self.assertEqual(state[0]['offset'], 7)
        self.assertEqual(restored[0].offset, 12)
        self.assertEqual(restored[0].cache.max_seq_len, 16)
        self.assertGreater(restored[0].nbytes, sum(a.nbytes for a in restored[0].state))

    def test_history_is_required_and_branches_are_independent(self):
        prefix = self.model.make_cache()
        mx.eval(self.model(mx.array([[1, 2, 3, 4, 5, 6, 7]]), cache=prefix))
        for component in ('engram', 'compressor'):
            with self.subTest(component=component):
                correct = self.support.clone_cache(prefix)
                broken = self.support.clone_cache(prefix)
                if component == 'engram':
                    broken[0].cache.engram_ids[:] = 0
                else:
                    broken[0].cache.layers[0].comp_state.kv_state[:] = 0
                expected = self.model(mx.array([[8, 9, 10, 11, 12]]), cache=correct)
                actual = self.model(mx.array([[8, 9, 10, 11, 12]]), cache=broken)
                mx.eval(expected, actual)
                self.assertFalse(np.array_equal(np.array(expected), np.array(actual)))
                self.assertEqual(prefix[0].offset, 7)
                self.assertEqual(prefix[0].cache.engram_ids[0, 6], 7)

    def test_runtime_memory_disk_off_and_cancelled_branch(self):
        backend = Tokenizer(models.WordLevel({str(i): i for i in range(16)}, unk_token='0'))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='0')
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        options = GenerationOptions(max_tokens=3, temperature=0)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps(asdict(self.model.args)))
            installed = SimpleNamespace(root=root, model_kind='deepseek-v4.1',
                                        model_id='fixture/v41', revision='fixture',
                                        format_version=3, maximum_context=32)
            def runtime(disk=False, enabled=True):
                config = RuntimeConfig(prompt_cache_entries=2 if enabled else 0,
                                       persistent_prompt_cache=disk,
                                       prompt_cache_directory=root / 'cache', prefill_step_size=3)
                with patch('deepseek_v4_ssd.generation.load_model',
                           return_value=(self.model, SimpleNamespace(close=lambda: None))), \
                     patch('deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained', return_value=tokenizer):
                    return ModelRuntime(installed, config)
            cold = runtime(enabled=False)
            expected = [p.token for p in cold.stream(prompt, options)]
            self.assertEqual(cold.metrics.snapshot()['prompt_cache_reused_tokens'], 0)
            self.assertFalse(cold._prompt_caches)
            cold.close()
            memory = runtime()
            self.assertEqual([p.token for p in memory.stream(prompt, options)], expected)
            self.assertEqual([p.token for p in memory.stream(prompt, options)], expected)
            self.assertGreater(memory.metrics.snapshot()['prompt_cache_reused_tokens'], 0)
            self.assertIsNone(memory._prompt_cache_directory)
            self.assertFalse((root / 'cache').exists())
            memory.close()
            warm = runtime(disk=True)
            self.assertEqual([p.token for p in warm.stream(prompt, options)], expected)
            self.assertEqual([p.token for p in warm.stream(prompt, options)], expected)
            self.assertGreater(warm.metrics.snapshot()['prompt_cache_reused_tokens'], 0)
            interrupted = warm.stream(prompt + [9], GenerationOptions(max_tokens=8, temperature=0))
            next(interrupted)
            interrupted.close()
            self.assertEqual([p.token for p in warm.stream(prompt, options)], expected)
            warm.close()
            reopened = runtime(disk=True)
            self.assertTrue(reopened._persistent_prompt_caches)
            self.assertEqual([p.token for p in reopened.stream(prompt, options)], expected)
            self.assertGreater(reopened.metrics.snapshot()['prompt_cache_reused_tokens'], 0)
            reopened.close()

    def _runtime(self, root, tokenizer, **config):
        (root / 'config.json').write_text(json.dumps(asdict(self.model.args)))
        installed = SimpleNamespace(root=root, model_kind='deepseek-v4.1',
                                    model_id='fixture/v41', revision='fixture',
                                    format_version=3, maximum_context=32)
        experts = SimpleNamespace(close=lambda: None, metrics_snapshot=CacheMetrics,
                                  capture_expert_unions=lambda: nullcontext(None))
        with patch('deepseek_v4_ssd.generation.load_model', return_value=(self.model, experts)), \
             patch('deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained', return_value=tokenizer):
            return ModelRuntime(installed, RuntimeConfig(prefill_step_size=3, batched_expert_prefill=False,
                                                         **config))

    @staticmethod
    def _tokenizer(eos=None):
        backend = Tokenizer(models.WordLevel({str(i): i for i in range(16)}, unk_token='0'))
        return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='0',
                                       **({'eos_token': str(eos)} if eos is not None else {}))

    def _generate(self, runtime, prompt, max_tokens=3):
        pieces = list(runtime.stream(prompt, GenerationOptions(max_tokens=max_tokens, temperature=0)))
        return [p.token for p in pieces], runtime.metrics.snapshot()

    def test_one_entry_keeps_prompt_snapshot_when_history_is_rendered_differently(self):
        # A client that drops reasoning re-renders the last assistant turn, so the
        # next prompt leaves the cached output right after the previous prompt.
        first = [1, 2, 3, 4, 5, 6, 7, 8]
        second = first[:-1] + [9, 10, 11, 12]
        with TemporaryDirectory() as directory:
            cold = self._runtime(Path(directory), self._tokenizer(), prompt_cache_entries=0)
            expected, _ = self._generate(cold, second)
            cold.close()
            runtime = self._runtime(Path(directory), self._tokenizer(), prompt_cache_entries=1)
            self._generate(runtime, first)
            self.assertLessEqual({7, 11}, {len(e.tokens) for e in runtime._prompt_caches})
            actual, metrics = self._generate(runtime, second)
            self.assertEqual(actual, expected)
            self.assertEqual(metrics['prompt_cache_reused_tokens'], 7)
            # The limit counts requests, so the first request's entries are gone.
            self.assertEqual(sorted(len(e.tokens) for e in runtime._prompt_caches), [10, 14])
            runtime.close()

    def test_final_entry_counts_the_eos_token_the_cache_consumed(self):
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        cache = self.model.make_cache()
        logits = self.model(mx.array([prompt]), cache=cache)
        greedy = []
        for _ in range(3):
            greedy.append(int(mx.argmax(logits[:, -1], axis=-1).item()))
            logits = self.model(mx.array([[greedy[-1]]]), cache=cache)
        eos = greedy[-1]
        self.assertNotIn(eos, greedy[:-1])
        follow_up = prompt + greedy + [9, 10]
        with TemporaryDirectory() as directory:
            cold = self._runtime(Path(directory), self._tokenizer(eos), prompt_cache_entries=0)
            expected, _ = self._generate(cold, follow_up)
            cold.close()
            runtime = self._runtime(Path(directory), self._tokenizer(eos), prompt_cache_entries=2)
            tokens, _ = self._generate(runtime, prompt, max_tokens=8)
            self.assertEqual(tokens, greedy)
            final = max(runtime._prompt_caches, key=lambda e: len(e.tokens))
            self.assertEqual(final.tokens, prompt + greedy)
            self.assertEqual(final.cache[0].offset, len(final.tokens))
            actual, metrics = self._generate(runtime, follow_up)
            self.assertEqual(actual, expected)
            self.assertEqual(metrics['prompt_cache_reused_tokens'], len(prompt + greedy))
            runtime.close()

    def test_upgrade_ignores_legacy_eos_state_and_rebuilds_reusable_disk_cache(self):
        prompt = [1, 2, 3, 4, 5, 6, 7, 8]
        cache = self.model.make_cache()
        logits = self.model(mx.array([prompt]), cache=cache)
        greedy = []
        for _ in range(3):
            greedy.append(int(mx.argmax(logits[:, -1], axis=-1).item()))
            logits = self.model(mx.array([[greedy[-1]]]), cache=cache)
        mx.eval(logits)
        eos = greedy[-1]
        self.assertNotIn(eos, greedy[:-1])
        follow_up = prompt + greedy + [9, 10]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(prompt_cache_entries=1, persistent_prompt_cache=True,
                          prompt_cache_directory=str(root / 'cache'))
            # Reproduce the pre-fix disk contract: state consumed EOS, tokens did not.
            with patch('deepseek_v4_ssd.generation._PROMPT_CACHE_CONTRACT_FORMAT', 1):
                with self._runtime(root, self._tokenizer(eos), **config) as legacy:
                    legacy._persist_prompt_cache(_PromptCacheEntry(cache, prompt + greedy[:-1]))
                    self.assertEqual(len(legacy._persistent_prompt_caches), 1)
            with self._runtime(root, self._tokenizer(eos), prompt_cache_entries=0) as cold:
                expected, _ = self._generate(cold, follow_up)
            with self._runtime(root, self._tokenizer(eos), **config) as upgraded:
                self.assertEqual(upgraded._persistent_prompt_caches, [])
                actual, metrics = self._generate(upgraded, follow_up)
                self.assertEqual(actual, expected)
                self.assertEqual(metrics['prompt_cache_reused_tokens'], 0)
                for entry in upgraded._prompt_caches:
                    self.assertEqual(entry.cache[0].offset, len(entry.tokens))
            with self._runtime(root, self._tokenizer(eos), **config) as reopened:
                actual, metrics = self._generate(reopened, follow_up)
                self.assertEqual(actual, expected)
                self.assertGreater(metrics['prompt_cache_reused_tokens'], 0)
                for entry in reopened._prompt_caches:
                    self.assertEqual(entry.cache[0].offset, len(entry.tokens))

    def test_dspark_requests_reuse_the_target_and_draft_prompt_snapshot(self):
        from deepseek_v4_ssd.deepseek_v41.dspark import DSpark
        self.model.dspark = DSpark(self.model.args, block_size=3, noise_token_id=15,
                                   target_layers=(1, 2, 3), markov_rank=8, expert_count=2, topk=1)
        first = [1, 2, 3, 4, 5, 6, 7, 8]
        second = first[:-1] + [9, 10, 11, 12]
        # Threshold 1 sends the 4-token suffix through layer-major prefill, which
        # extends the restored draft context instead of the chunked prefill.
        for threshold in (1_024, 1):
            with self.subTest(threshold=threshold), TemporaryDirectory() as directory:
                config = dict(dspark_enabled=True, layer_major_prefill_threshold=threshold,
                              prompt_cache_entries=1)
                cold = self._runtime(Path(directory), self._tokenizer(), **config)
                expected, metrics = self._generate(cold, second)
                self.assertEqual(metrics['prompt_cache_reused_tokens'], 0)
                runtime = self._runtime(Path(directory), self._tokenizer(), **config)
                _, metrics = self._generate(runtime, first)
                self.assertEqual(metrics['dspark_prompt_cache_source'], 'none')
                self.assertEqual([len(e.tokens) for e in runtime._dspark_prompt_caches], [7])
                actual, metrics = self._generate(runtime, second)
                self.assertEqual(actual, expected)
                self.assertEqual(metrics['prompt_cache_reused_tokens'], 7)
                self.assertEqual(metrics['dspark_prompt_cache_source'], 'memory')
                self.assertEqual([len(e.tokens) for e in runtime._dspark_prompt_caches], [10])
                # The restored draft context continues like a cold prefill. Different
                # chunking can move a fake-quantized FP8 value by one code.
                reused = runtime._dspark_prompt_caches[0].context_state
                for (context, offset), (reference, reference_offset) in zip(
                        reused, cold._dspark_prompt_caches[0].context_state):
                    self.assertEqual((context.shape, offset), (reference.shape, reference_offset))
                    self.assertTrue(mx.allclose(context.astype(mx.float32),
                                                reference.astype(mx.float32), atol=0.25).item())
                cold.close()
                runtime.close()

    def test_dspark_prompt_snapshot_survives_a_restart_in_disk_mode(self):
        from deepseek_v4_ssd.deepseek_v41.dspark import DSpark
        self.model.dspark = DSpark(self.model.args, block_size=3, noise_token_id=15,
                                   target_layers=(1, 2, 3), markov_rank=8, expert_count=2, topk=1)
        first = [1, 2, 3, 4, 5, 6, 7, 8]
        second = first[:-1] + [9, 10, 11, 12]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(dspark_enabled=True, prompt_cache_entries=1,
                          persistent_prompt_cache=True, prompt_cache_directory=str(root / 'cache'))
            cold = self._runtime(root, self._tokenizer(), **{**config, 'prompt_cache_entries': 0})
            expected, _ = self._generate(cold, second)
            cold.close()
            writer = self._runtime(root, self._tokenizer(), **config)
            self._generate(writer, first)
            writer.close()
            reader = self._runtime(root, self._tokenizer(), **config)
            self.assertEqual([len(e.tokens) for e in reader._persistent_dspark_prompt_caches], [7])
            actual, metrics = self._generate(reader, second)
            self.assertEqual(actual, expected)
            self.assertEqual(metrics['prompt_cache_reused_tokens'], 7)
            self.assertEqual(metrics['dspark_prompt_cache_source'], 'persistent')
            reader.close()

    def test_rejects_invalid_state_without_changing_target(self):
        cache = self.model.make_cache()
        for field, value in (('offset', 33), ('capacity', 0), ('version', 2),
                             ('layers', []), ('engram_ids', None)):
            with self.subTest(field=field):
                state = self.support.snapshot_cache(cache)
                state[0][field] = value
                with self.assertRaises(ValueError):
                    self.support.restore_cache(cache, state)
                self.assertEqual(cache[0].offset, 0)


if __name__ == '__main__':
    unittest.main()
