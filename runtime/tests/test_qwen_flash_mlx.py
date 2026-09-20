"""MLX integration and numerical checks; no installed model download required.

These are synthetic kernel/lifecycle tests, not model-quality or speed results.
Run on Apple Silicon with the repository's pinned runtime requirements.
"""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import CacheList, KVCache

from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.cancellation import GenerationCancelled
from deepseek_v4_ssd.expert_cache import QwenExpertWeights
from deepseek_v4_ssd.manifest import NGram
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import _parse_runtime, validate_runtime_config
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.qwen_flash_config import DEFAULTS


def bits(value):
    mx.eval(value)
    return np.asarray(value.view(mx.uint8))


class FlashIntegrationTests(unittest.TestCase):
    def test_old_catalog_and_runtime_defaults_are_unchanged(self):
        raw = asdict(RuntimeConfig())
        for name, value in DEFAULTS.items():
            self.assertEqual(raw.pop(name), value)
        self.assertEqual(_parse_runtime(raw, "runtime", "qwen3.8-flash-next"), RuntimeConfig())
        for name, value in {"qwen_expert_wave_slots": -1, "qwen_ngram_io": "direct",
                            "qwen_ngram_cache_bytes": 1, "qwen_sparse_sdpa": 1}.items():
            with self.assertRaises(ValueError):
                validate_runtime_config(replace(RuntimeConfig(), **{name: value}))

    def test_options_are_qwen_only(self):
        for config in (RuntimeConfig(qwen_expert_wave_slots=32),
                       RuntimeConfig(qwen_ngram_io="pread"),
                       RuntimeConfig(qwen_sparse_sdpa=True)):
            get_support("qwen3.8-flash-next").validate_config(config)
            for kind in ("deepseek-v4", "deepseek-v4.1"):
                with self.assertRaises(ValueError):
                    get_support(kind).validate_config(config)

    def test_prefill_waves_do_not_allocate_full_layer_buffers(self):
        support = get_support("qwen3.8-flash-next")
        for slots in (0, 32):
            config = RuntimeConfig(qwen_expert_wave_slots=slots)
            with patch("deepseek_v4_ssd.model_support.qwen._qwen_layer_major_prefill") as prefill:
                support.prefill(None, [1, 2], [], 2, None, config)
                self.assertEqual(prefill.call_args.args[-1], not bool(slots))

    def test_close_releases_ngram_even_when_another_resource_fails(self):
        ngram = SimpleNamespace(close=Mock())
        expert = SimpleNamespace(close=Mock(side_effect=RuntimeError("close failure")))
        model = SimpleNamespace(ngram_store=ngram)
        with self.assertRaises(RuntimeError):
            get_support("qwen3.8-flash-next").close(model, expert)
        ngram.close.assert_called_once()

    def test_failed_load_closes_ngram_descriptor(self):
        installed = SimpleNamespace(root=Path("/unused"), ngram=SimpleNamespace(file="ngram.bin"))
        scale = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
        with patch.object(qwen, "ExpertCache") as cache, patch.object(qwen, "NGramStore") as store, \
             patch.object(qwen, "Model", side_effect=RuntimeError("load failure")):
            with self.assertRaisesRegex(RuntimeError, "load failure"):
                qwen.load(installed, RuntimeConfig(), {"text_config": {}}, {scale: mx.array([1.0])})
            store.return_value.close.assert_called_once()
            cache.return_value.close.assert_called_once()

    def test_pread_fp8_decode_matches_mmap_bits_and_closes(self):
        descriptor = NGram("ngram.bin", "F8_E4M3", 1, 1, 256, (0,) * 16, (256,) * 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ngram.bin"
            np.arange(256, dtype=np.uint8).tofile(path)
            valid = np.array([i for i in range(256) if i not in (127, 255)])
            ids = np.concatenate([valid[::-1], valid, valid]).reshape(3, -1)
            for scale in (0.5, 0.123456789, 5.3):
                baseline = qwen.NGramStore(path, descriptor, scale)
                self.addCleanup(baseline.close)
                for optimized in (False, True):
                    candidate = qwen.NGramStore(path, descriptor, scale, optimized=optimized,
                                                io_backend="pread", cache_bytes=256)
                    self.addCleanup(candidate.close)
                    for rows in (ids, ids, np.array(3), np.empty((1, 0, 16), dtype=np.int64)):
                        np.testing.assert_array_equal(bits(candidate.lookup(rows)), bits(baseline.lookup(rows)))
                    self.assertLessEqual(candidate.io_snapshot()["cached_payload_bytes"], 256)
                    for invalid in (127, 255, -1, 256):
                        with self.assertRaises(ValueError):
                            candidate.lookup(np.array([invalid]))
                    candidate.close()
                    candidate.close()
                    with self.assertRaises(ValueError):
                        candidate.lookup(np.array([3]))


class ExpertWaveIntegrationTests(unittest.TestCase):
    def weights(self):
        rng = np.random.default_rng(31)
        gate, gate_scales = mx.quantize(mx.array(rng.normal(0, .1, (16, 128, 64)), dtype=mx.bfloat16),
            group_size=32, bits=4, mode="mxfp4")
        down, down_scales = mx.quantize(mx.array(rng.normal(0, .1, (16, 64, 64)), dtype=mx.bfloat16),
            group_size=32, bits=4, mode="mxfp4")
        mx.eval(gate, gate_scales, down, down_scales)
        return [QwenExpertWeights(gate[i], gate_scales[i], down[i], down_scales[i]) for i in range(16)]

    def test_waves_preserve_mxfp4_results_and_fence_before_slot_reuse(self):
        individual = self.weights()
        events = []
        def acquire(layer, ids):
            if events:
                self.assertEqual(events[-1], "eval", "slots reused before GPU evaluation")
            self.assertLessEqual(len(set(ids)), cache.slots)
            events.append("load")
            return SimpleNamespace(individual_weights=individual, slots={i: i for i in range(16)})
        cache = SimpleNamespace(model=SimpleNamespace(expert_count=16), slots=16,
                                current_batched=lambda _: None, get_many=acquire)
        experts = qwen.StreamingExperts(0, cache)
        value = mx.random.normal((1, 31, 64)).astype(mx.bfloat16)
        selected = mx.array((np.arange(31)[:, None] * 7 + np.arange(10)[None, :] * 3) % 16)[None]
        baseline = experts(value, selected)
        mx.eval(baseline)
        for capacity in (1, 3, 16):
            events.clear()
            cache.slots = capacity
            experts.wave_slots = 32  # Must also clamp to the physical slot count.
            real_eval = mx.eval
            def evaluate(*values):
                real_eval(*values)
                events.append("eval")
            with patch.object(qwen.mx, "eval", side_effect=evaluate):
                actual = experts(value, selected)
            np.testing.assert_array_equal(bits(actual), bits(baseline))
            self.assertEqual(events, [item for _ in range((16 + capacity - 1) // capacity)
                                      for item in ("load", "eval")])

    def test_cancelled_wave_does_not_start_another_read(self):
        individual = self.weights()
        acquired = []
        def acquire(layer, ids):
            acquired.append(ids)
            return SimpleNamespace(individual_weights=individual, slots={i: i for i in range(16)})
        cache = SimpleNamespace(model=SimpleNamespace(expert_count=16), slots=2,
                                current_batched=lambda _: None, get_many=acquire)
        experts = qwen.StreamingExperts(0, cache)
        experts.wave_slots = 2
        def cancelled():
            if acquired:
                raise GenerationCancelled()
        with patch.object(qwen, "check_cancelled", side_effect=cancelled), self.assertRaises(GenerationCancelled):
            experts(mx.ones((1, 3, 64)), mx.array([[[0, 1], [2, 3], [4, 5]]]))
        self.assertEqual(len(acquired), 1)

    def test_single_token_keeps_existing_ready_decode_path(self):
        cache = SimpleNamespace(current_batched=lambda _: None, ready_expert_decode=True,
                                iter_ready=lambda layer, ids: iter((i, None) for i in ids))
        experts = qwen.StreamingExperts(0, cache)
        experts.wave_slots = 1
        with patch.object(qwen.StreamingExperts, "_one", side_effect=lambda v, _: v), \
             patch.object(qwen.StreamingExperts, "_waves", side_effect=AssertionError("decode used waves")):
            output = experts(mx.ones((1, 1, 8)), mx.array([[[1, 2]]]))
            mx.eval(output)
        self.assertEqual(output.shape, (1, 1, 2, 8))


class SparseSDPATests(unittest.TestCase):
    def attention(self, head_dim=32, dtype=mx.float32):
        mx.random.seed(19)
        args = qwen.ModelArgs(hidden_size=32, num_attention_heads=4, num_key_value_heads=2,
            head_dim=head_dim, indexer_n_heads=2, indexer_head_dim=32,
            indexer_budget=8, indexer_compress_ratio=4,
            partial_rotary_factor=0.125 if head_dim == 256 else 0.5)
        attention = qwen.QSAAttention(args)
        attention.set_dtype(dtype)
        return attention

    def test_same_selected_cells_and_causality_with_fused_attention(self):
        for head_dim in (32, 256):
            for dtype, tolerance in ((mx.float32, 2e-5), (mx.bfloat16, 5e-3)):
                with self.subTest(head_dim=head_dim, dtype=dtype):
                    attention = self.attention(head_dim, dtype)
                    hidden = (mx.random.normal((1, 17, 32)) * 0.1).astype(dtype)
                    for length in (1, 3, 8, 9, 17):
                        attention.sparse_sdpa = False
                        expected = attention(hidden[:, :length], None)
                        attention.sparse_sdpa = True
                        actual = attention(hidden[:, :length], None)
                        np.testing.assert_allclose(np.asarray(actual.astype(mx.float32)),
                            np.asarray(expected.astype(mx.float32)), atol=tolerance, rtol=tolerance)
                    # Alter only future tokens: causal prefix must be identical.
                    changed = mx.concatenate([hidden[:, :9], hidden[:, 9:] * -30], axis=1)
                    original = attention(hidden, None)
                    future_changed = attention(changed, None)
                    np.testing.assert_array_equal(bits(original[:, :9]), bits(future_changed[:, :9]))

    def test_chunked_decode_and_cache_rollback(self):
        attention = self.attention()
        attention.sparse_sdpa = True
        hidden = mx.random.normal((1, 17, 32)) * 0.1
        expected = attention(hidden, None)
        cache = CacheList(KVCache(), KVCache())
        chunks = [attention(hidden[:, :3], cache), attention(hidden[:, 3:9], cache),
                  attention(hidden[:, 9:16], cache), attention(hidden[:, 16:], cache)]
        np.testing.assert_allclose(np.asarray(mx.concatenate(chunks, axis=1)), np.asarray(expected), atol=2e-5, rtol=2e-5)
        for branch in cache.caches:
            branch.trim(5)
        replay = attention(hidden[:, 12:17], cache)
        np.testing.assert_allclose(np.asarray(replay), np.asarray(expected[:, 12:17]), atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
