"""MTP verification computes each expert as soon as it is resident (Qwen main model, always on)."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.manifest import InstalledModel, Tensor
from deepseek_v4_ssd.model import RuntimeConfig

EXPERTS = 64


def installed(root):
    """One layer of real MXFP4 expert blobs: hidden 128, intermediate 128."""
    specs = [("gate_up.weight", "U32", (256, 16)), ("gate_up.scale", "U8", (256, 4)),
             ("down.weight", "U32", (128, 16)), ("down.scale", "U8", (128, 4))]
    regions, offset = [], 0
    for name, dtype, shape in specs:
        length = int(np.prod(shape)) * (4 if dtype == "U32" else 1)
        regions.append(Tensor(name, dtype, shape, offset, length))
        offset += length
    model = InstalledModel(root, "fixture", "fixture", 1, EXPERTS, 4, offset, (), tuple(regions),
                           model_kind="qwen3.8-flash-next")
    (root / "experts").mkdir()
    rng = np.random.default_rng(1009)
    with (root / "experts/layer_00.bin").open("wb") as file:
        for expert in range(EXPERTS):
            for region in regions:
                # Finite packed FP4 values and a nonzero scale that differs between experts.
                file.write(rng.integers(0, 256, region.length, dtype=np.uint8).tobytes()
                           if region.name.endswith(".weight") else bytes([117 + expert % 3]) * region.length)
    return model


def routes(rng, rows):
    return mx.array(np.stack([rng.choice(EXPERTS, 4, replace=False) for _ in range(rows)])[None], dtype=mx.int32)


class ReadyVerificationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.model = installed(Path(directory.name))

    def cache(self):
        cache = ExpertCache(self.model, slots=40, read_workers=2, ready_expert_decode=True)
        self.addCleanup(cache.close)
        return cache

    def experts(self, cache, on=True):
        experts = qwen.StreamingExperts(0, cache)
        experts.ready_verification = on
        return experts

    @staticmethod
    def state(cache):
        return ({key: (entry.slot, entry.frequency, entry.last_access) for key, entry in cache._entries.items()},
                sorted(cache._free_slots),
                {name: getattr(cache.metrics, name) for name in ("hits", "misses", "evictions", "bytes_read")})

    def test_matches_the_waiting_path_bit_for_bit_and_leaves_the_same_cache(self):
        control, candidate = self.cache(), self.cache()
        waiting, ready = self.experts(control, on=False), self.experts(candidate)
        rng = np.random.default_rng(3)
        # More distinct experts than slots over the sequence, so residency and eviction vary per call.
        for rows in (2, 3, 6, 3, 8, 2, 6):
            value = mx.array(rng.normal(0, .3, (1, rows, 128)), dtype=mx.bfloat16)
            indices = routes(rng, rows)
            expected = waiting(value, indices)
            with patch.object(candidate, "get_many", side_effect=AssertionError("waiting path used")):
                actual = ready(value, indices)
            mx.eval(expected, actual)
            with self.subTest(rows=rows):
                self.assertEqual(actual.shape, (1, rows, 4, 128))
                self.assertTrue(mx.array_equal(actual, expected).item())
                self.assertEqual(self.state(candidate), self.state(control))
        self.assertGreater(control.metrics.evictions, 0)

    def test_only_decode_verification_blocks_take_the_new_path(self):
        cache = self.cache()
        experts = self.experts(cache)
        value = mx.ones((1, 3, 128), dtype=mx.bfloat16)
        indices = mx.array([[[0, 1, 2, 3], [4, 5, 6, 7], [1, 3, 5, 7]]], dtype=mx.int32)

        def path(value=value, indices=indices):
            with patch.object(cache, "get_many", wraps=cache.get_many) as waiting, \
                 patch.object(cache, "iter_ready", wraps=cache.iter_ready) as ready:
                mx.eval(experts(value, indices))
            if waiting.called:
                return "waiting"
            return "ready rows" if "on_split" in ready.call_args.kwargs else "single row"
        self.assertEqual(path(), "ready rows")
        self.assertEqual(path(value[:, :1], indices[:, :1]), "single row")
        with cache.trace_routes("prefill"):
            self.assertEqual(path(), "waiting")
        cache.ready_expert_decode = False
        self.assertEqual(path(), "waiting")
        cache.ready_expert_decode = True
        experts.ready_verification = False
        self.assertEqual(path(), "waiting")
        experts.ready_verification = True
        rows = qwen.READY_VERIFICATION_ROWS + 1
        self.assertEqual(path(mx.ones((1, rows, 128), dtype=mx.bfloat16),
                              mx.array(np.tile(np.arange(4), (rows, 1))[None], dtype=mx.int32)), "waiting")

    def test_resident_experts_go_in_one_submission_then_each_late_read(self):
        value = mx.ones((1, 2, 128), dtype=mx.bfloat16)
        indices = mx.array([[[0, 1, 5, 9], [2, 1, 6, 9]]], dtype=mx.int32)
        for preload, expected in (([0, 1, 2], [3, 1, 1, 1]), ([], [1] * 6),
                                  ([0, 1, 2, 5, 6, 9], [6])):
            cache = self.cache()
            if preload:
                cache.get_many(0, preload)
            experts = self.experts(cache)
            with self.subTest(resident=preload), \
                 patch.object(qwen.mx, "async_eval", wraps=mx.async_eval) as submit:
                mx.eval(experts(value, indices))
                self.assertEqual([len(call.args) for call in submit.call_args_list], expected)

    def test_cancelled_or_failed_reads_release_their_reserved_slots(self):
        cache = self.cache()
        experts = self.experts(cache)
        cache.get_many(0, [0, 1])
        value = mx.ones((1, 2, 128), dtype=mx.bfloat16)
        indices = mx.array([[[0, 1, 4, 5], [1, 0, 6, 7]]], dtype=mx.int32)
        untouched = self.state(cache)
        with patch.object(qwen, "check_cancelled", side_effect=RuntimeError("cancelled")):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                experts(value, indices)
        self.assertEqual(self.state(cache), untouched)
        with patch.object(cache, "_read_expert_into_slot", side_effect=EOFError("truncated")):
            with self.assertRaises(EOFError):
                experts(value, indices)
        mx.synchronize()   # resident work may still be in flight, as after any failed request
        self.assertEqual(set(cache._entries), {(0, 0), (0, 1)})
        self.assertEqual(len(cache._entries) + len(cache._free_slots), cache.slots)
        # Stopped after the first late read: completed reads stay, pending ones are released.
        calls, original = [], qwen.StreamingExperts._one

        def one(source, weights):
            calls.append(None)
            if len(calls) == 4:
                raise RuntimeError("cancelled")
            return original(source, weights)
        with patch.object(qwen.StreamingExperts, "_one", staticmethod(one)):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                experts(value, indices)
        mx.synchronize()
        self.assertLessEqual({(0, 0), (0, 1)}, set(cache._entries))
        self.assertEqual(len(cache._entries) + len(cache._free_slots), cache.slots)
        expected = self.experts(self.cache(), on=False)(value, indices)
        self.assertTrue(mx.array_equal(experts(value, indices), expected).item())

    def test_on_for_every_main_layer_and_off_for_the_draft_layer(self):
        installed_model = SimpleNamespace(root=Path("/unused"), ngram=SimpleNamespace(file="ngram.bin"))
        scale = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
        model = MagicMock()
        model.model.layers = [SimpleNamespace(mlp=SimpleNamespace(experts=SimpleNamespace())) for _ in range(2)]
        model.parameters.return_value = []
        model.sanitize.return_value = {}
        with patch.object(qwen.ModelArgs, "from_dict"), patch.object(qwen, "ExpertCache"), \
             patch.object(qwen, "NGramStore"), patch.object(qwen, "Model", return_value=model), \
             patch("deepseek_v4_ssd.ane_prefill.install_qwen_ane_prefill", return_value=None):
            qwen.load(installed_model, RuntimeConfig(), {"text_config": {}}, {scale: mx.array([1.0])})
        self.assertEqual([layer.mlp.experts.ready_verification for layer in model.model.layers], [True, True])
        # The MTP draft layer was not part of the validation and keeps the waiting path.
        args = qwen.ModelArgs(hidden_size=64, hc_count=4, hc_lowrank=16, num_attention_heads=4,
                              num_key_value_heads=2, head_dim=32, indexer_n_heads=2, indexer_head_dim=32,
                              indexer_budget=8, indexer_compress_ratio=4, partial_rotary_factor=0.5, num_experts=8,
                              num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
                              vocab_size=32, max_position_embeddings=128)
        self.assertFalse(qwen.MTPModel(args, None).layers[0].mlp.experts.ready_verification)


if __name__ == "__main__":
    unittest.main()
