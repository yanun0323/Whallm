from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from deepseek_v4_ssd.deepseek_v41 import moe as v41_moe
from deepseek_v4_ssd.deepseek_v41.cache import ModelCache
from deepseek_v4_ssd.deepseek_v41.config import ModelArgs
from deepseek_v4_ssd.deepseek_v41.attention import Attention
from deepseek_v4_ssd.deepseek_v41.dequant import dequant_fp8, dequant_fp8_rows
from deepseek_v4_ssd.deepseek_v41.model import Model
from deepseek_v4_ssd.deepseek_v41_ssd import (
    SSDEngramEmbedding,
    _load_token_map,
    _prepare_common_weights,
    _validate_common_weight_shapes,
    load as load_deepseek_v41,
)
from deepseek_v4_ssd.expert_cache import ExpertWeights, ResidentExperts
from deepseek_v4_ssd.io_metrics import request_rows
from deepseek_v4_ssd.manifest import (
    DEEPSEEK_V41_EXPERT_REGIONS,
    DEEPSEEK_V41_MODEL_ID,
    DEEPSEEK_V41_REVISION,
    _deepseek_v41_contract,
)
from deepseek_v4_ssd.model import _EmptySwitchGLU, RuntimeConfig


def manifest_contract() -> dict:
    expert_count = 384
    expert_blob_size = 18_800_640
    tables = ((1, 384_006_168), (14, 384_016_682))
    files = [
        {"path": "common.bin", "size": 1},
        {"path": "config.json", "size": 1},
        {"path": "encoding/encoding.py", "size": 1},
        {"path": "tokenizer/tokenizer.json", "size": 1},
        {"path": "tokenizer/tokenizer_config.json", "size": 1},
        *(
            {
                "path": f"experts/layer_{layer:02d}.bin",
                "size": expert_count * expert_blob_size,
            }
            for layer in range(40)
        ),
        *(
            {
                "path": f"engram/layer_{layer:02d}.weight.bin",
                "size": rows * 256,
            }
            for layer, rows in tables
        ),
        *(
            {
                "path": f"engram/layer_{layer:02d}.scale.bin",
                "size": rows * 8,
            }
            for layer, rows in tables
        ),
    ]
    return {
        "formatVersion": 3,
        "modelKind": "deepseek-v4.1",
        "modelID": DEEPSEEK_V41_MODEL_ID,
        "revision": DEEPSEEK_V41_REVISION,
        "layerCount": 40,
        "expertCount": expert_count,
        "selectedExpertCount": 6,
        "expertBlobSize": expert_blob_size,
        "maximumContext": 1_048_576,
        "expertQuantization": None,
        "ngram": None,
        "mtp": None,
        "dspark": None,
        "engram": {
            "tables": [
                {
                    "layer": layer,
                    "weightFile": f"engram/layer_{layer:02d}.weight.bin",
                    "scaleFile": f"engram/layer_{layer:02d}.scale.bin",
                    "rows": rows,
                    "dimension": 256,
                    "blockSize": 32,
                }
                for layer, rows in tables
            ]
        },
        "files": files,
    }


class DeepSeekV41ContractTests(unittest.TestCase):
    def test_exact_release_contract_is_accepted(self):
        contract = _deepseek_v41_contract(manifest_contract())

        self.assertEqual(contract["model_kind"], "deepseek-v4.1")
        self.assertEqual(contract["layer_count"], 40)
        self.assertEqual(contract["expert_count"], 384)
        self.assertEqual(contract["selected_expert_count"], 6)
        self.assertEqual(contract["maximum_context"], 1_048_576)
        self.assertEqual(contract["expert_regions"], DEEPSEEK_V41_EXPERT_REGIONS)

    def test_revision_and_engram_sizes_are_pinned(self):
        invalid_revision = manifest_contract()
        invalid_revision["revision"] = "main"
        with self.assertRaisesRegex(ValueError, "pinned V4.1"):
            _deepseek_v41_contract(invalid_revision)

        invalid_engram = manifest_contract()
        table = invalid_engram["engram"]["tables"][0]
        weight = next(
            item
            for item in invalid_engram["files"]
            if item["path"] == table["weightFile"]
        )
        weight["size"] -= 1
        with self.assertRaisesRegex(ValueError, "engram weight"):
            _deepseek_v41_contract(invalid_engram)

    def test_native_fp8_weights_are_packed_for_mlx_without_changing_values(self):
        rng = np.random.default_rng(7)
        raw = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        raw[raw % 128 == 127] = 0
        scale = rng.integers(122, 133, size=(2, 2), dtype=np.uint8)
        weight = mx.array(raw)
        scale_array = mx.array(scale)

        prepared, modules = _prepare_common_weights(
            {"layers.0.attn.wq_a.weight": weight, "layers.0.attn.wq_a.scale": scale_array}
        )
        packed = prepared["layers.0.attn.wq_a.weight"]
        expanded_scale = prepared["layers.0.attn.wq_a.scales"]
        restored = mx.dequantize(
            packed,
            expanded_scale,
            group_size=32,
            bits=8,
            mode="mxfp8",
        ).astype(mx.float32)
        reference = dequant_fp8(weight, scale_array, mx.float32)
        mx.eval(restored, reference)

        self.assertEqual(modules, {"layers.0.attn.wq_a"})
        self.assertEqual(packed.shape, (64, 16))
        self.assertEqual(expanded_scale.shape, (64, 2))
        np.testing.assert_array_equal(np.array(restored), np.array(reference))

    def test_packed_native_fp8_weights_load_into_quantized_linear(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(64, 64, bias=False)

        raw = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
        raw[raw % 128 == 127] = 0
        scale = np.full((2, 2), 127, dtype=np.uint8)
        prepared, modules = _prepare_common_weights(
            {
                "projection.weight": mx.array(raw),
                "projection.scale": mx.array(scale),
            }
        )
        model = TinyModel()
        nn.quantize(
            model,
            group_size=32,
            bits=8,
            mode="mxfp8",
            class_predicate=lambda path, _: path in modules,
        )

        model.load_weights(list(prepared.items()), strict=True)
        self.assertEqual(model.projection.weight.shape, (64, 16))
        self.assertEqual(model.projection.scales.shape, (64, 2))

    def test_common_weight_shape_mismatch_is_rejected_before_loading(self):
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(4, 3, bias=False)

        with self.assertRaisesRegex(ValueError, "shape_mismatches"):
            _validate_common_weight_shapes(
                TinyModel(),
                {"projection.weight": mx.zeros((2, 4))},
            )

    def test_token_map_is_cached_in_memory_without_mutating_the_install(self):
        _load_token_map.cache_clear()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = SimpleNamespace()
            with (
                patch(
                    "transformers.PreTrainedTokenizerFast.from_pretrained",
                    return_value=tokenizer,
                ) as open_tokenizer,
                patch(
                    "deepseek_v4_ssd.deepseek_v41_ssd.v41_engram.build_compressed_token_map",
                    return_value=([0, 1, 1, 2], {}),
                ) as build_map,
            ):
                first = _load_token_map(str(root), 4, 3)
                second = _load_token_map(str(root), 4, 3)

            self.assertEqual(first, (0, 1, 1, 2))
            self.assertIs(first, second)
            open_tokenizer.assert_called_once_with(root / "tokenizer")
            build_map.assert_called_once_with(tokenizer)
            self.assertFalse((root / "tokenizer" / "engram_token_map.json").exists())

    def test_rope_frequencies_grow_geometrically(self):
        attention = SimpleNamespace(
            _cos=None,
            _sin=None,
            _rope=(4, 0, 10_000.0, 1.0, 32, 1),
            max_seq_len=4_096,
        )

        def frequencies(_dim, length, *_args):
            return mx.zeros((length, 2)), mx.zeros((length, 2))

        with patch(
            "deepseek_v4_ssd.deepseek_v41.attention.precompute_freqs_cis",
            side_effect=frequencies,
        ) as precompute:
            Attention._freqs(attention, 65)
            Attention._freqs(attention, 66)
            Attention._freqs(attention, 1_025)

        self.assertEqual(
            [call.args[1] for call in precompute.call_args_list],
            [1_024, 2_048],
        )

    def test_tiny_adapter_loads_and_runs_a_real_forward(self):
        args = ModelArgs(
            vocab_size=8,
            dim=64,
            n_layers=1,
            moe_inter_dim=32,
            n_heads=2,
            head_dim=32,
            rope_head_dim=4,
            q_lora_rank=32,
            o_lora_rank=32,
            o_groups=2,
            window_size=4,
            n_routed_experts=2,
            n_shared_experts=1,
            n_activated_experts=1,
            compress_ratios=(0,),
            original_seq_len=16,
            max_seq_len=16,
            index_n_heads=2,
            index_head_dim=32,
            index_topk=2,
            hc_mult=1,
            engram_vocab_size=8,
            engram_n_heads=1,
            engram_head_dim=8,
            engram_compressed_vocab_size=4,
        )
        original_switch_glu = v41_moe.SwitchGLU
        try:
            v41_moe.SwitchGLU = _EmptySwitchGLU
            baseline = Model(args, token_map=[0, 1, 2, 3, 0, 1, 2, 3])
        finally:
            v41_moe.SwitchGLU = original_switch_glu
        common = {
            name: mx.zeros(parameter.shape, dtype=parameter.dtype)
            for name, parameter in tree_flatten(baseline.parameters())
        }

        def quantized(rows: int, columns: int, value: float):
            return mx.quantize(
                mx.full((rows, columns), value),
                group_size=32,
                bits=4,
                mode="mxfp4",
            )

        experts = []
        for value in (0.125, 0.25):
            w1, w1_scales = quantized(32, 64, value)
            w2, w2_scales = quantized(64, 32, value + 0.125)
            w3, w3_scales = quantized(32, 64, value + 0.25)
            experts.append(
                ExpertWeights(
                    w1,
                    w1_scales,
                    w2,
                    w2_scales,
                    w3,
                    w3_scales,
                )
            )

        class TinyExpertCache:
            route_trace_enabled = False
            ready_expert_decode = True
            staged_expert_streaming = False

            def __init__(self, *_args, **_kwargs):
                self.closed = False

            @staticmethod
            def current_batched(_layer):
                return None

            @staticmethod
            def speculative_prefetch_active(_layer):
                return False

            @staticmethod
            def iter_ready(_layer, selected):
                return iter(
                    (expert, experts[expert])
                    for expert in dict.fromkeys(selected)
                )

            @staticmethod
            def get_many(_layer, _selected):
                return ResidentExperts(tuple(experts), {0: 0, 1: 1})

            def close(self):
                self.closed = True

        with (
            TemporaryDirectory() as directory,
            patch(
                "deepseek_v4_ssd.deepseek_v41_ssd._load_token_map",
                return_value=(0, 1, 2, 3, 0, 1, 2, 3),
            ),
            patch(
                "deepseek_v4_ssd.deepseek_v41_ssd.ExpertCache",
                TinyExpertCache,
            ),
        ):
            installed = SimpleNamespace(
                root=Path(directory),
                engram=SimpleNamespace(tables=[]),
            )
            model, cache = load_deepseek_v41(
                installed,
                RuntimeConfig(),
                asdict(args),
                common,
                None,
            )
            prompt_cache = model.make_cache()
            logits = model(mx.array([[1, 2]]), cache=prompt_cache)
            mx.eval(logits)

        self.assertEqual(logits.shape, (1, 2, 8))
        self.assertEqual(prompt_cache[0].offset, 2)
        self.assertFalse(cache.closed)

    def test_ssd_engram_reads_only_selected_rows(self):
        rng = np.random.default_rng(11)
        weight = rng.integers(0, 256, size=(4, 32), dtype=np.uint8)
        weight[weight % 128 == 127] = 0
        scale = rng.integers(122, 133, size=(4, 1), dtype=np.uint8)
        indices = mx.array([[3, 1, 3]])

        with TemporaryDirectory() as directory:
            root = Path(directory)
            weight.tofile(root / "weight.bin")
            scale.tofile(root / "scale.bin")
            embedding = SSDEngramEmbedding(
                root / "weight.bin",
                root / "scale.bin",
                rows=4,
                dimension=32,
                block_size=32,
            )
            with patch("deepseek_v4_ssd.deepseek_v41_ssd.request_rows",
                       wraps=request_rows) as request:
                actual = embedding(indices)
            expected = dequant_fp8_rows(
                mx.array(weight[[[3, 1, 3]]]),
                mx.array(scale[[[3, 1, 3]]]),
            )
            mx.eval(actual, expected)

        np.testing.assert_array_equal(np.array(actual), np.array(expected))
        # Both mapped tables are asked for the rows about to be read, by row width.
        self.assertEqual([call.args[2] for call in request.call_args_list], [32, 1])
        for call in request.call_args_list:
            np.testing.assert_array_equal(call.args[1], [[3, 1, 3]])

    def test_model_cache_grows_and_rejects_the_context_ceiling(self):
        class Args:
            window_size = 4
            kv_source_layers = (0,)
            index_source_layers = (0,)
            head_dim = 8
            index_head_dim = 4
            n_layers = 1
            max_seq_len = 16
            engram_layer_ids = (0,)

            @staticmethod
            def compress_ratio(_: int) -> int:
                return 2

        cache = ModelCache(Args(), max_seq_len=4)
        cache.engram_ids[0, :4] = [1, 2, 3, 4]
        cache.ensure_capacity(9)

        self.assertEqual(cache.max_seq_len, 9)
        self.assertEqual(cache.layers[0].comp_kv.shape[1], 4)
        self.assertEqual(cache.layers[0].index_k.shape[1], 4)
        self.assertEqual(cache.engram_ids[0, :4].tolist(), [1, 2, 3, 4])
        with self.assertRaisesRegex(ValueError, "configured context"):
            cache.ensure_capacity(17)


if __name__ == "__main__":
    unittest.main()
