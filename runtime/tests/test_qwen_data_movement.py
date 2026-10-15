"""Qwen changes that move or compute less without changing results.

Routed-expert Prefill, merged sorted-expert Prefill calls, whole-block
cache-bypassing slot reads and the cached RoPE frequencies must reproduce the
earlier bytes exactly. Tiny synthetic models; no checkpoint required.
"""
from __future__ import annotations

import argparse
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.expert_cache import ExpertCache, _SlotPool
from deepseek_v4_ssd.generation import _prompt_cache_contract
from deepseek_v4_ssd.manifest import InstalledModel, Tensor
from deepseek_v4_ssd.model import RuntimeConfig, _cache_arrays
from deepseek_v4_ssd.model_manager import _parse_runtime, validate_runtime_config
from deepseek_v4_ssd.model_support.qwen import QwenSupport
from deepseek_v4_ssd.model_support.catalog import BY_KIND
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.model_support.qwen import _expert_spans, _qwen_layer_major_prefill
from deepseek_v4_ssd.qwen_flash_config import add_flash_arguments, flash_arguments

EXPERTS = 16
HIDDEN = 128


def bits(value):
    mx.eval(value)
    return np.asarray(value.view(mx.uint8)).copy()


def write_experts(root: Path, layers: int, seed: int, hidden: int = HIDDEN):
    """MXFP4 experts with 4-byte scale rows; the blob is not a multiple of 4 KiB."""
    mx.random.seed(seed)
    gate, gate_scales = mx.quantize(mx.random.normal((EXPERTS, 2 * hidden, hidden)).astype(mx.bfloat16) * .03,
                                    group_size=32, bits=4, mode="mxfp4")
    down, down_scales = mx.quantize(mx.random.normal((EXPERTS, hidden, hidden)).astype(mx.bfloat16) * .03,
                                    group_size=32, bits=4, mode="mxfp4")
    mx.eval(gate, gate_scales, down, down_scales)
    arrays = (gate, gate_scales, down, down_scales)
    names = ("gate_up.weight", "gate_up.scale", "down.weight", "down.scale")
    regions, offset = [], 0
    for name, array in zip(names, arrays):
        length = array[0].nbytes
        regions.append(Tensor(name, "U32" if name.endswith(".weight") else "U8",
                              tuple(array.shape[1:]), offset, length))
        offset += length
    (root / "experts").mkdir()
    for layer in range(layers):
        with (root / f"experts/layer_{layer:02d}.bin").open("wb") as file:
            for expert in range(EXPERTS):
                for array in arrays:
                    # Distinct weights per layer: rotate the expert order.
                    file.write(np.asarray(array[(expert + layer) % EXPERTS]).tobytes())
    installed = InstalledModel(root, "synthetic", "synthetic", layers, EXPERTS, 2, offset, (),
                               tuple(regions), model_kind="qwen3.8-flash-next")
    weights = [qwen.QwenExpertWeights(gate[e], gate_scales[e], down[e], down_scales[e])
               for e in range(EXPERTS)]
    return installed, weights


def tiny_model(cache, hidden: int = HIDDEN):
    args = qwen.ModelArgs(
        hidden_size=hidden, num_hidden_layers=2, layer_types=("linear_attention", "full_attention"),
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32,
        linear_value_head_dim=32, indexer_n_heads=2, indexer_head_dim=32, indexer_budget=4,
        indexer_compress_ratio=2, hc_count=4, hc_lowrank=16, partial_rotary_factor=0.5,
        num_experts=EXPERTS, num_experts_per_tok=2, moe_intermediate_size=hidden,
        shared_expert_intermediate_size=64, vocab_size=64, max_position_embeddings=256,
        ple_layer_ids=(),
    )
    mx.random.seed(7)
    model = qwen.Model(args, cache, None)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model


class RopeFrequencyTests(unittest.TestCase):
    def test_cached_frequencies_reproduce_the_per_call_rotation(self):
        def per_call(value, positions, rotary_dim, theta):
            frequencies = mx.exp(-math.log(theta) * mx.arange(0, rotary_dim, 2) / rotary_dim)
            angles = positions[..., None] * frequencies
            cosine = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=-1)
            sine = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=-1)
            rotary = value[..., :rotary_dim]
            first, second = mx.split(rotary, 2, axis=-1)
            rotated = mx.concatenate([-second, first], axis=-1)
            return mx.concatenate([rotary * cosine + rotated * sine, value[..., rotary_dim:]], axis=-1)

        mx.random.seed(5)
        for dtype in (mx.float32, mx.bfloat16):
            for shape, positions in (((1, 7, 2, 128), (mx.arange(7) + 1000)[None, :, None]),
                                     ((1, 5, 128), (mx.arange(5) * 4)[None])):
                for theta in (10_000_000, 10_000):
                    with self.subTest(dtype=dtype, shape=shape, theta=theta):
                        value = mx.random.normal(shape).astype(dtype)
                        np.testing.assert_array_equal(
                            bits(qwen._apply_partial_rope(value, positions, 64, theta)),
                            bits(per_call(value, positions, 64, theta)))


class RoutedPrefillTests(unittest.TestCase):
    def run_prefill(self, installed, tokens, routed_max_tokens, step_size):
        with ExpertCache(installed, slots=EXPERTS, read_workers=2) as cache:
            model = tiny_model(cache)
            prompt_cache = model.make_cache()
            with patch.object(cache, "prefetch_layer", wraps=cache.prefetch_layer) as prefetch:
                hidden = _qwen_layer_major_prefill(model, tokens, prompt_cache, step_size, cache,
                                                   routed_max_tokens=routed_max_tokens)
            requested = [call.args[1] if len(call.args) > 1 else call.kwargs.get("experts")
                         for call in prefetch.call_args_list]
            state = [bits(array) for array in _cache_arrays(prompt_cache)]
            return bits(hidden), state, cache.metrics.bytes_read, requested

    def test_routed_reads_reproduce_whole_layer_bytes_and_read_less(self):
        with tempfile.TemporaryDirectory() as directory:
            installed, _ = write_experts(Path(directory), layers=2, seed=11)
            tokens = [3, 17, 5, 9, 41, 2]
            for step_size in (4, 6):
                with self.subTest(step_size=step_size):
                    full = self.run_prefill(installed, tokens, 0, step_size)
                    routed = self.run_prefill(installed, tokens, len(tokens), step_size)
                    np.testing.assert_array_equal(routed[0], full[0])
                    self.assertEqual(len(routed[1]), len(full[1]))
                    for got, want in zip(routed[1], full[1]):
                        np.testing.assert_array_equal(got, want)
                    # Whole-layer reads request every expert; routed reads only the union.
                    self.assertEqual(full[2], 2 * EXPERTS * installed.expert_blob_size)
                    self.assertTrue(all(experts is None for experts in full[3]))
                    self.assertTrue(all(0 < len(experts) <= 2 * len(tokens) for experts in routed[3]))
                    self.assertEqual(routed[2], sum(map(len, routed[3])) * installed.expert_blob_size)
                    self.assertLess(routed[2], full[2])

    def test_routed_prefill_runs_each_router_once(self):
        with tempfile.TemporaryDirectory() as directory:
            installed, _ = write_experts(Path(directory), layers=2, seed=15)
            tokens = [3, 17, 5, 9, 41, 2]
            full = self.run_prefill(installed, tokens, 0, 4)
            with patch.object(qwen.SparseMoE, "route", autospec=True,
                              side_effect=qwen.SparseMoE.route) as route, \
                 patch.object(qwen.GatedResidual, "__call__", autospec=True,
                              side_effect=qwen.GatedResidual.__call__) as mixer:
                routed = self.run_prefill(installed, tokens, len(tokens), 4)
            np.testing.assert_array_equal(routed[0], full[0])
            for got, want in zip(routed[1], full[1]):
                np.testing.assert_array_equal(got, want)
            # Two layers, two spans: one router and two mixers (attention, MoE) each.
            self.assertEqual(route.call_count, 2 * 2)
            self.assertEqual(mixer.call_count, 2 * 2 * 2)

    def test_longer_prompts_keep_whole_layer_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            installed, _ = write_experts(Path(directory), layers=2, seed=12)
            tokens = list(range(1, 8))
            result = self.run_prefill(installed, tokens, len(tokens) - 1, 4)
            self.assertTrue(all(experts is None for experts in result[3]))

    def test_default_512_validation_catalog_and_qwen_only(self):
        self.assertEqual(RuntimeConfig().qwen_prefill_routed_max_tokens, 512)
        from dataclasses import asdict
        raw = asdict(RuntimeConfig())
        raw.pop("qwen_prefill_routed_max_tokens")
        # Catalogs without the field, including the App's, get the default.
        self.assertEqual(_parse_runtime(raw, "runtime", "qwen3.8-flash-next").qwen_prefill_routed_max_tokens, 512)
        for value in (0, 1024, 262_144):
            validate_runtime_config(RuntimeConfig(qwen_prefill_routed_max_tokens=value))
        for value in (-1, 262_145, True, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_runtime_config(RuntimeConfig(qwen_prefill_routed_max_tokens=value))
        # Prefill paths without whole-layer reads ignore the default instead of failing.
        for other in ({"batched_expert_prefill": False}, {"layer_major_prefill": False},
                      {"qwen_expert_wave_slots": 8}):
            with self.subTest(other=other):
                validate_runtime_config(RuntimeConfig(**other))
        with patch("deepseek_v4_ssd.model_support.qwen._qwen_layer_major_prefill") as prefill:
            QwenSupport(BY_KIND["qwen3.8-flash-next"]).prefill(None, [1], [], 1, None, RuntimeConfig(qwen_expert_wave_slots=8))
        self.assertFalse(prefill.call_args.args[6])
        # The default is accepted by every model; a different value is Qwen-only.
        for kind in ("deepseek-v4", "deepseek-v4.1"):
            get_support(kind).validate_config(RuntimeConfig())
            with self.assertRaises(ValueError):
                get_support(kind).validate_config(RuntimeConfig(qwen_prefill_routed_max_tokens=128))
        config = RuntimeConfig(qwen_prefill_routed_max_tokens=128)
        QwenSupport(BY_KIND["swift1.5-qwen3.8-flash-next"]).validate_config(config)
        parser = argparse.ArgumentParser()
        add_flash_arguments(parser)
        self.assertEqual(flash_arguments(parser.parse_args([]))["qwen_prefill_routed_max_tokens"], 512)
        self.assertEqual(flash_arguments(parser.parse_args(["--qwen-prefill-routed-max-tokens", "0"]))
                         ["qwen_prefill_routed_max_tokens"], 0)
        with patch("deepseek_v4_ssd.model_support.qwen._qwen_layer_major_prefill") as prefill:
            QwenSupport(BY_KIND["qwen3.8-flash-next"]).prefill(None, [1], [], 1, None, config)
        self.assertEqual(prefill.call_args.kwargs["routed_max_tokens"], 128)
        # Same output, so cached prompts stay compatible.
        installed = SimpleNamespace(root="/missing", revision="r", model_id="m")
        self.assertEqual(_prompt_cache_contract(installed, config),
                         _prompt_cache_contract(installed, RuntimeConfig()))


class MergedExpertPrefillTests(unittest.TestCase):
    # Two experts per token over 16 experts: chunks of 32+ tokens take MLX's
    # sorted kernel alone, as 1024-token chunks do with 10 of 512 experts.
    TOKENS = [(index * 7 + 3) % 64 for index in range(5 * 32 + 10)]
    # From this width the per-row and sorted expert kernels round differently;
    # at 128 they happen to agree. A layer's residual absorbs that difference,
    # so only the MoE block's own output shows it.
    WIDTH = 1024

    def test_merged_block_matches_each_chunk_from_the_sorted_kernel_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            installed, _ = write_experts(Path(directory), layers=2, seed=15, hidden=self.WIDTH)
            with ExpertCache(installed, slots=EXPERTS, read_workers=2) as cache:
                moe = tiny_model(cache, self.WIDTH).layers[0].mlp
                moe.experts.grouped_prefill = moe.experts.sorted_prefill = True
                mx.random.seed(16)
                # 32 tokens: exactly four rows per expert.
                values = [mx.random.normal((1, tokens, self.WIDTH)).astype(mx.bfloat16)
                          for tokens in (32, 32, 40)]
                with cache.reuse_layer_buffers(), cache.batched_layer(0):
                    separate = [bits(moe(value)) for value in values]
                    merged = [bits(value) for value in moe.merged(values)]
        for got, want in zip(merged, separate):
            np.testing.assert_array_equal(got, want)

    def run_prefill(self, installed, moe_step_size, sorted_prefill=True):
        with ExpertCache(installed, slots=EXPERTS, read_workers=2) as cache:
            model = tiny_model(cache, self.WIDTH)
            for layer in model.layers:
                layer.mlp.experts.grouped_prefill = True
                layer.mlp.experts.sorted_prefill = sorted_prefill
            prompt_cache = model.make_cache()
            hidden = _qwen_layer_major_prefill(model, self.TOKENS, prompt_cache, 32, cache,
                                               next_layer_prefetch=True, moe_step_size=moe_step_size)
            state = [bits(array) for array in _cache_arrays(prompt_cache)]
            return bits(hidden), state, cache.metrics.gather_qmm_calls

    def test_prefill_merges_whole_chunks_with_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            installed, _ = write_experts(Path(directory), layers=2, seed=15, hidden=self.WIDTH)
            per_chunk = self.run_prefill(installed, 32)
            # Six chunks per layer, two expert products each.
            self.assertEqual(per_chunk[2], 2 * 6 * 2)
            # 64: four calls per layer. 0 (the whole 170-token prompt): the five
            # 32-token chunks share one call; the 10-token tail stays alone.
            for moe_step_size, calls in ((64, 4), (0, 2)):
                with self.subTest(moe_step_size=moe_step_size):
                    merged = self.run_prefill(installed, moe_step_size)
                    np.testing.assert_array_equal(merged[0], per_chunk[0])
                    self.assertEqual(len(merged[1]), len(per_chunk[1]))
                    for got, want in zip(merged[1], per_chunk[1]):
                        np.testing.assert_array_equal(got, want)
                    self.assertEqual(merged[2], 2 * calls * 2)
            # Without the sorted kernel a longer call gains nothing; chunks stay apart.
            self.assertEqual(self.run_prefill(installed, 0, sorted_prefill=False)[2], 2 * 6 * 2)

    def test_only_sorted_chunks_that_take_the_sorted_kernel_alone_join(self):
        # Qwen: 10 of 512 experts, so 205 tokens average four rows per expert.
        experts = qwen.StreamingExperts(0, SimpleNamespace(
            model=SimpleNamespace(selected_expert_count=10, expert_count=512)))
        experts.grouped_prefill = experts.sorted_prefill = True
        self.assertFalse(experts.shares_expert_call(204))
        self.assertTrue(experts.shares_expert_call(205))
        for flag in ("grouped_prefill", "sorted_prefill"):
            with self.subTest(flag=flag):
                setattr(experts, flag, False)
                self.assertFalse(experts.shares_expert_call(1024))
                setattr(experts, flag, True)

    def test_spans_hold_whole_joining_chunks_up_to_the_limit(self):
        joins = lambda tokens: tokens >= 205
        self.assertEqual(_expert_spans(5000, 1024, 4096, joins),
                         [[(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)], [(4096, 5000)]])
        self.assertEqual(_expert_spans(3000, 1024, 3000, joins),
                         [[(0, 1024), (1024, 2048), (2048, 3000)]])
        # A tail too short for the sorted kernel stays alone even within the limit.
        self.assertEqual(_expert_spans(2248, 1024, 4096, joins),
                         [[(0, 1024), (1024, 2048)], [(2048, 2248)]])
        # Smaller limits than a chunk, short chunks and no joins keep every chunk alone.
        for spans in (_expert_spans(3000, 1024, 512, joins), _expert_spans(3000, 1024, 4096, None),
                      _expert_spans(600, 128, 4096, joins)):
            self.assertTrue(all(len(span) == 1 for span in spans))

    def test_qwen_passes_the_moe_step_and_keeps_the_prompt_cache_contract(self):
        config = RuntimeConfig(moe_prefill_step_size=8192)
        with patch("deepseek_v4_ssd.model_support.qwen._qwen_layer_major_prefill") as prefill:
            QwenSupport(BY_KIND["qwen3.8-flash-next"]).prefill(None, [1], [], 1, None, config)
        self.assertEqual(prefill.call_args.kwargs["moe_step_size"], 8192)
        # Same output, so cached prompts stay compatible.
        installed = SimpleNamespace(root="/missing", revision="r", model_id="m")
        self.assertEqual(_prompt_cache_contract(installed, config),
                         _prompt_cache_contract(installed, RuntimeConfig()))


class AlignedSlotReadTests(unittest.TestCase):
    def test_bypass_reads_whole_blocks_into_shifted_slots_with_identical_compute(self):
        with tempfile.TemporaryDirectory() as directory:
            installed, weights = write_experts(Path(directory), layers=1, seed=13)
            blob = installed.expert_blob_size
            value = mx.random.normal((1, HIDDEN)).astype(mx.bfloat16) * .1
            outputs = {}
            for policy in ("cached", "bypass"):
                # Two slots: every later read reuses a slot, often at another shift.
                with ExpertCache(installed, slots=2, read_workers=2, file_cache_policy=policy) as cache:
                    alignment = cache.direct_io_alignment
                    if policy == "bypass":
                        self.assertGreater(alignment, 0)
                        self.assertNotEqual(blob % alignment, 0)
                        granule = math.gcd(blob, alignment)
                        self.assertEqual(cache._pool._slot_bytes,
                                         -(-(alignment - granule + blob) // alignment) * alignment)
                    else:
                        self.assertEqual(cache._pool._slot_bytes, blob)
                    module = qwen.StreamingExperts(0, cache)
                    with patch.object(cache, "_pread_views", wraps=cache._pread_views) as reads:
                        results = []
                        for expert in list(range(EXPERTS)) + [1, 0, 3]:
                            resident = cache.get_many(0, [expert])
                            slot = cache._entries[(0, expert)].slot
                            if policy == "bypass":
                                shift = (expert * blob) % alignment
                                self.assertEqual(cache._pool._shifts[slot], shift)
                            loaded = resident.individual_weights[0]
                            for got, want in zip((loaded.gate_up, loaded.gate_up_scales, loaded.down,
                                                  loaded.down_scales), (weights[expert].gate_up,
                                                  weights[expert].gate_up_scales, weights[expert].down,
                                                  weights[expert].down_scales)):
                                np.testing.assert_array_equal(bits(got), bits(want))
                            results.append(bits(module._one(value, loaded)))
                    if policy == "bypass":
                        for call in reads.call_args_list:
                            layer, views, offset = call.args
                            self.assertEqual(len(views), 1)
                            self.assertEqual(offset % alignment, 0)
                            self.assertEqual(len(views[0]) % alignment, 0)
                    outputs[policy] = results
            for got, want in zip(outputs["bypass"], outputs["cached"]):
                np.testing.assert_array_equal(got, want)

    def test_region_writes_and_slot_views_follow_the_shift(self):
        with tempfile.TemporaryDirectory() as directory:
            installed, weights = write_experts(Path(directory), layers=1, seed=14)
            pool = _SlotPool(installed, 1, 4096)
            pool.prepare([0])
            offset, shift, length = pool.aligned_read(1)
            self.assertGreater(shift, 0)
            target = pool.read_target(0, shift, length)
            with open(installed.root / "experts/layer_00.bin", "rb") as file:
                file.seek(offset)
                target[:] = file.read(length)
            target.release()
            pool.mark_loaded(0)
            loaded = pool.select_individual([0])[0]
            np.testing.assert_array_equal(bits(loaded.down), bits(weights[1].down))
            blob = b"".join(bytes(pool.writable_region(0, region.name)) for region in installed.expert_regions)
            with open(installed.root / "experts/layer_00.bin", "rb") as file:
                file.seek(installed.expert_blob_size)
                self.assertEqual(blob, file.read(installed.expert_blob_size))
            # A plain blob store keeps the slot's shift and its views stay consistent.
            with open(installed.root / "experts/layer_00.bin", "rb") as file:
                file.seek(2 * installed.expert_blob_size)
                pool.store([0], [file.read(installed.expert_blob_size)])
            np.testing.assert_array_equal(bits(pool.select_individual([0])[0].down), bits(weights[2].down))


if __name__ == "__main__":
    unittest.main()
