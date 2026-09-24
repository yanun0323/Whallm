from __future__ import annotations

import json
import tempfile
import unittest
from threading import Event
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from deepseek_v4_ssd.cancellation import GenerationCancelled, cancellation_scope

from deepseek_v4_ssd.generation import (
    GenerationOptions,
    ModelRuntime,
    RuntimeMetrics,
    _PromptCacheEntry,
    _prompt_cache_block_identity,
    _prompt_cache_contract,
)


class _FixtureCache:
    def __init__(self, value: int = 0):
        self._state = (mx.array([value], dtype=mx.int32),)
        self.nbytes = 4

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value


class _Closeable:
    def close(self):
        pass


def _runtime(directory: Path, *, entries: int = 8, fp8: bool = True):
    runtime = object.__new__(ModelRuntime)
    runtime.installed = SimpleNamespace(
        root=directory / "model",
        revision="fixture-revision",
        model_id="fixture/model",
    )
    runtime.config = SimpleNamespace(
        fp8_kv_cache=fp8,
        fp4_index_cache=True,
        persistent_prompt_cache_entries=entries,
    )
    runtime.metrics = RuntimeMetrics()
    runtime.model = SimpleNamespace(dspark=None)
    runtime.expert_cache = _Closeable()
    runtime._prompt_caches = []
    runtime._persistent_prompt_caches = []
    runtime._dspark_prompt_caches = []
    runtime._persistent_dspark_prompt_caches = []
    runtime._prompt_cache_directory = directory
    return runtime


class BlockPromptCacheTests(unittest.TestCase):
    def test_invalid_disk_candidate_uses_shorter_prefix_and_stays_excluded(self):
        for damage in ('header', 'short_read', 'schema', 'invalid_item', 'missing'):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as temporary:
                runtime = _runtime(Path(temporary))
                for tokens in ([1], [1, 2]):
                    runtime._persist_prompt_cache(_PromptCacheEntry([_FixtureCache(len(tokens))], tokens))
                bad = max(runtime._persistent_prompt_caches, key=lambda e: len(e.tokens))
                if damage == 'header':
                    bad.path.write_bytes(b'corrupt fixture')
                elif damage == 'short_read':
                    bad.path.write_bytes(b'invalid')
                elif damage == 'schema':
                    mx.save_safetensors(bad.path, {'a': mx.array([9])}, {'state': '{}'})
                elif damage == 'invalid_item':
                    mx.save_safetensors(bad.path, {'a': mx.array([9])}, {'state': '{"items":[null]}'})
                else:
                    bad.path.unlink()
                with patch.object(runtime.support, 'new_cache', side_effect=lambda _: [_FixtureCache()]), \
                     patch.object(runtime, '_load_persistent_prompt_cache', wraps=runtime._load_persistent_prompt_cache) as load:
                    acquired = runtime._acquire_prompt_cache([1, 2, 3])
                    self.assertEqual(acquired.tokens, [1])
                    self.assertEqual(acquired.cache[0].state[0].tolist(), [1])
                    self.assertEqual([call.args[0].tokens for call in load.call_args_list], [[1, 2], [1]])
                    runtime._persistent_prompt_caches = runtime._scan_persistent_prompt_caches()
                    self.assertNotIn(bad.path, [e.path for e in runtime._persistent_prompt_caches])
                    load.reset_mock()
                    self.assertEqual(runtime._acquire_prompt_cache([1, 2, 3]).tokens, [1])
                    self.assertEqual(load.call_count, 1)
                if damage != 'missing':
                    self.assertTrue(bad.path.exists())
                # Exclusion lasts only for this loaded model; original evidence stays on disk.
                runtime.close()
                fresh = _runtime(Path(temporary))
                if damage != 'missing':
                    self.assertIn(bad.path, [e.path for e in fresh._scan_persistent_prompt_caches()])

    def test_all_invalid_candidates_return_fresh_independent_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            for tokens in ([1], [1, 2]):
                runtime._persist_prompt_cache(_PromptCacheEntry([_FixtureCache(99)], tokens))
            for entry in runtime._persistent_prompt_caches:
                entry.path.write_bytes(b'corrupt fixture')
            with patch.object(runtime.support, 'new_cache', side_effect=lambda _: [_FixtureCache()]), \
                 patch.object(runtime, '_record_persistent_prompt_cache_hit') as hit:
                acquired = runtime._acquire_prompt_cache([1, 2, 3])
                self.assertEqual(acquired.tokens, [])
                self.assertEqual(acquired.cache[0].state[0].tolist(), [0])
                self.assertEqual(runtime._persistent_prompt_caches, [])
                self.assertEqual(runtime._scan_persistent_prompt_caches(), [])
                hit.assert_not_called()

    def test_disk_recovery_does_not_hide_execution_errors_or_cancellation(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = _runtime(Path(temporary))
            runtime._persist_prompt_cache(_PromptCacheEntry([_FixtureCache(1)], [1]))
            candidate = runtime._persistent_prompt_caches[0]
            with patch('deepseek_v4_ssd.generation.mx.load', side_effect=RuntimeError('GPU out of memory')):
                with self.assertRaisesRegex(RuntimeError, 'GPU out of memory'):
                    runtime._acquire_prompt_cache([1, 2])
            with patch.object(runtime.support, 'new_cache', side_effect=lambda _: [_FixtureCache()]), \
                 patch.object(runtime.support, 'evaluate_cache', side_effect=RuntimeError('GPU execution failed')):
                with self.assertRaisesRegex(RuntimeError, 'GPU execution failed'):
                    runtime._acquire_prompt_cache([1, 2])
            event = Event()
            event.set()
            with cancellation_scope(event), patch('deepseek_v4_ssd.generation.mx.load') as load:
                with self.assertRaises(GenerationCancelled):
                    runtime._acquire_prompt_cache([1, 2])
                load.assert_not_called()
            self.assertIn(candidate, runtime._persistent_prompt_caches)

    def test_contract_and_token_block_chain_change_every_compatibility_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            config_path = model / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "rope_theta": 10_000,
                        "rope_scaling": {"factor": 16},
                        "sliding_window": 128,
                        "compress_ratios": [4, 128],
                    }
                ),
                encoding="utf-8",
            )
            installed = SimpleNamespace(
                root=model,
                revision="revision-a",
                model_id="fixture/model",
            )
            fp8 = SimpleNamespace(fp8_kv_cache=True, fp4_index_cache=True)
            bf16 = SimpleNamespace(fp8_kv_cache=False, fp4_index_cache=True)

            first = _prompt_cache_contract(installed, fp8)
            first_identity = _prompt_cache_block_identity(first, list(range(129)))
            second_identity = _prompt_cache_block_identity(
                _prompt_cache_contract(installed, bf16),
                list(range(129)),
            )
            config_path.write_text(
                json.dumps(
                    {
                        "rope_theta": 20_000,
                        "rope_scaling": {"factor": 16},
                        "sliding_window": 128,
                        "compress_ratios": [4, 128],
                    }
                ),
                encoding="utf-8",
            )
            third_identity = _prompt_cache_block_identity(
                _prompt_cache_contract(installed, fp8),
                list(range(129)),
            )

        self.assertNotEqual(first_identity[0], second_identity[0])
        self.assertNotEqual(first_identity[2], second_identity[2])
        self.assertNotEqual(first_identity[0], third_identity[0])
        self.assertEqual(first_identity[1][0]["length"], 128)
        self.assertEqual(first_identity[1][1]["length"], 1)
        self.assertEqual(first_identity[1][1]["parent"], first_identity[1][0]["key"])

    def test_previous_contract_cannot_reuse_or_shadow_new_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            # Contract 1 writers could omit a consumed EOS from the token list.
            with patch('deepseek_v4_ssd.generation._PROMPT_CACHE_CONTRACT_FORMAT', 1):
                legacy = _runtime(directory)
                legacy._persist_prompt_cache(_PromptCacheEntry([_FixtureCache(99)], [1, 2]))
                old_path = legacy._persistent_prompt_caches[0].path
            current = _runtime(directory)
            self.assertEqual(current._scan_persistent_prompt_caches(), [])
            current._persist_prompt_cache(_PromptCacheEntry([_FixtureCache(2)], [1, 2]))
            saved = current._scan_persistent_prompt_caches()
            self.assertEqual(len(saved), 1)
            self.assertNotEqual(saved[0].path, old_path)
            with patch.object(current.support, 'new_cache', side_effect=lambda _: [_FixtureCache()]):
                acquired = current._acquire_prompt_cache([1, 2, 3])
            self.assertEqual(acquired.tokens, [1, 2])
            self.assertEqual(acquired.cache[0].state[0].tolist(), [2])

    def test_identical_prefixes_share_one_immutable_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runtime = _runtime(directory)
            tokens = list(range(129))
            runtime._persist_prompt_cache(
                _PromptCacheEntry([_FixtureCache(1)], tokens)
            )
            data_path = next(directory.glob("*.normal.v5.safetensors"))
            metadata_path = next(directory.glob("*.normal.v5.json"))
            first_data_mtime = data_path.stat().st_mtime_ns
            first_metadata = metadata_path.read_bytes()

            runtime._persist_prompt_cache(
                _PromptCacheEntry([_FixtureCache(99)], tokens)
            )

            self.assertEqual(
                len(list(directory.glob("*.normal.v5.safetensors"))),
                1,
            )
            self.assertEqual(len(list(directory.glob("*.normal.v5.json"))), 1)
            self.assertEqual(data_path.stat().st_mtime_ns, first_data_mtime)
            self.assertEqual(metadata_path.read_bytes(), first_metadata)

    def test_restart_reuses_first_prefill_checkpoint_after_suffix_diverges(self):
        first_tokens = list(range(256))
        branch_tokens = [*range(128), *range(10_000, 10_022)]
        prompts = {"first": first_tokens, "branch": branch_tokens}
        received: list[list[int]] = []

        def generate(_model, _tokenizer, prompt, **options):
            prompt = list(prompt)
            received.append(prompt)
            callback = options["prompt_progress_callback"]
            cache = options["prompt_cache"][0]
            callback(0, len(prompt))
            if len(prompt) > 1:
                first = min(128, len(prompt) - 1)
                cache.state = (mx.array([first], dtype=mx.int32),)
                callback(first, len(prompt))
                if first != len(prompt) - 1:
                    cache.state = (
                        mx.array([len(prompt) - 1], dtype=mx.int32),
                    )
                    callback(len(prompt) - 1, len(prompt))
            yield SimpleNamespace(
                text="A",
                token=42,
                generation_tokens=1,
                finish_reason="length",
            )

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            installed = SimpleNamespace(
                root=directory / "model",
                revision="fixture-revision",
                model_id="fixture/model",
            )
            config = SimpleNamespace(
                prefill_step_size=128,
                layer_major_prefill=False,
                prompt_cache_entries=2,
                prompt_cache_memory_gib=1,
                persistent_prompt_cache=True,
                persistent_prompt_cache_entries=8,
                prompt_cache_directory=directory,
                fp8_kv_cache=True,
                fp4_index_cache=True,
            )
            tokenizer = SimpleNamespace(
                bos_token=None,
                encode=lambda prompt, **_options: prompts[prompt],
            )
            with (
                patch(
                    "deepseek_v4_ssd.generation.load_model",
                    return_value=(object(), _Closeable()),
                ),
                patch(
                    "deepseek_v4_ssd.generation.AutoTokenizer.from_pretrained",
                    return_value=tokenizer,
                ),
                patch(
                    "deepseek_v4_ssd.model_support.state.make_prompt_cache",
                    side_effect=lambda _model: [_FixtureCache()],
                ),
                patch(
                    "deepseek_v4_ssd.generation.stream_generate",
                    side_effect=generate,
                ),
            ):
                first = ModelRuntime(installed, config)
                list(first.stream("first", GenerationOptions(max_tokens=1)))
                first.close()

                second = ModelRuntime(installed, config)
                list(second.stream("branch", GenerationOptions(max_tokens=1)))
                reused = second.metrics.snapshot()["prompt_cache_reused_tokens"]
                second.close()

            access = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in directory.rglob("*.normal.v5.access")
            ]

        self.assertEqual(received[0], first_tokens)
        self.assertEqual(received[1], branch_tokens[128:])
        self.assertEqual(reused, 128)
        self.assertIn(1, [item["reuseCount"] for item in access])

    def test_reused_prefix_survives_frequency_aware_eviction(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runtime = _runtime(directory, entries=2)
            for token in (1, 2):
                runtime._persist_prompt_cache(
                    _PromptCacheEntry([_FixtureCache(token)], [token])
                )
            reused = next(
                entry
                for entry in runtime._scan_persistent_prompt_caches()
                if entry.tokens == [1]
            )
            runtime._record_persistent_prompt_cache_hit(reused)
            runtime._persist_prompt_cache(
                _PromptCacheEntry([_FixtureCache(3)], [3])
            )
            remaining = {
                tuple(entry.tokens)
                for entry in runtime._scan_persistent_prompt_caches()
            }

        self.assertEqual(remaining, {(1,), (3,)})

    def test_scanner_rejects_format_four_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            runtime = _runtime(directory)
            runtime._persist_prompt_cache(
                _PromptCacheEntry([_FixtureCache(1)], [1, 2])
            )
            metadata_path = next(directory.glob("*.normal.v5.json"))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["format"] = 4
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

            self.assertEqual(runtime._scan_persistent_prompt_caches(), [])

    def test_scanner_rejects_a_different_kv_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = _runtime(directory, fp8=True)
            first._persist_prompt_cache(
                _PromptCacheEntry([_FixtureCache(1)], [1, 2])
            )
            second = _runtime(directory, fp8=False)

            self.assertEqual(second._scan_persistent_prompt_caches(), [])


if __name__ == "__main__":
    unittest.main()
