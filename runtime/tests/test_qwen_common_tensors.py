"""Opt-in 8-bit common tensors for the Qwen engine. Synthetic modules and a tiny
model; no checkpoint required."""
from __future__ import annotations

import argparse
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd.generation import _prompt_cache_contract
from deepseek_v4_ssd.manifest import NGram
from deepseek_v4_ssd.model import RuntimeConfig, eval_prompt_cache
from deepseek_v4_ssd.model_manager import validate_runtime_config
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.qwen_common_tensors import CommonQuantizedLinear, quantize_common_tensors
from deepseek_v4_ssd.qwen_flash_config import add_flash_arguments, flash_arguments
from runtime.tests.test_qwen_speed import experts_fixture, tiny_args


class Node(nn.Module):
    def __init__(self, **children):
        super().__init__()
        for name, child in children.items():
            setattr(self, name, child)


def linear(inputs, outputs):
    layer = nn.Linear(inputs, outputs, bias=False)
    layer.weight = (mx.random.normal((outputs, inputs)) * 0.05).astype(mx.bfloat16)
    return layer


class CommonTensorTests(unittest.TestCase):
    def test_projections_with_256_or_more_rows_become_8_bit_except_routers(self):
        mx.random.seed(3)
        model = Node(
            model=Node(embed_tokens=nn.Embedding(512, 64), layers=[Node(
                linear_attn=Node(in_proj_qkv=linear(64, 256), in_proj_a=linear(64, 48)),
                mlp=Node(gate=linear(64, 512), shared_expert_gate=linear(64, 1)))]),
            lm_head=linear(64, 512), mtp=Node(fc_hidden=linear(128, 256)))
        before = dict(model.named_modules())
        packed_paths = ["lm_head", "model.layers.0.linear_attn.in_proj_qkv", "mtp.fc_hidden"]

        self.assertEqual(quantize_common_tensors(model), 3)
        after = dict(model.named_modules())
        self.assertEqual(sorted(path for path, module in after.items()
                                if isinstance(module, CommonQuantizedLinear)), packed_paths)
        for path in ("model.embed_tokens", "model.layers.0.linear_attn.in_proj_a",
                     "model.layers.0.mlp.gate", "model.layers.0.mlp.shared_expert_gate"):
            self.assertIs(after[path], before[path])
        for path in packed_paths:
            packed, source = after[path], before[path]
            self.assertEqual((packed.bits, packed.group_size, packed.mode), (8, 64, "affine"))
            x = mx.random.normal((1, 3, source.weight.shape[1])).astype(mx.bfloat16)
            got, want = packed(x).astype(mx.float32), source(x).astype(mx.float32)
            error = mx.sqrt(mx.mean((got - want) ** 2)) / mx.sqrt(mx.mean(want ** 2))
            self.assertLess(error.item(), 0.02, path)

    def test_decode_and_verification_use_the_8_bit_kernel_and_prefill_a_bf16_copy(self):
        mx.random.seed(4)
        packed = CommonQuantizedLinear.from_linear(linear(128, 256), group_size=64, bits=8)
        weight = mx.dequantize(packed.weight, packed.scales, packed.biases, group_size=64, bits=8)
        for rows in (1, 3, 16, 17, 64):
            x = mx.random.normal((1, rows, 128)).astype(mx.bfloat16)
            expected = (mx.quantized_matmul(x, packed.weight, packed.scales, packed.biases,
                                            transpose=True, group_size=64, bits=8)
                        if rows <= 16 else x @ weight.T)
            self.assertTrue(mx.array_equal(packed(x), expected).item(), rows)

    def test_setting_is_off_by_default_qwen_only_boolean_and_not_with_ane_prefill(self):
        self.assertFalse(RuntimeConfig().qwen_8bit_common_tensors)
        config = RuntimeConfig(qwen_8bit_common_tensors=True)
        validate_runtime_config(config)
        for kind in ("qwen3.8-flash-next", "swift1.5-qwen3.8-flash-next"):
            get_support(kind).validate_config(config)
            with self.assertRaises(ValueError):
                get_support(kind).validate_config(replace(config, ane_prefill_ratio=0.5))
        for kind in ("deepseek-v4", "deepseek-v4.1"):
            with self.assertRaises(ValueError):
                get_support(kind).validate_config(config)
        with self.assertRaises(ValueError):
            validate_runtime_config(replace(config, qwen_8bit_common_tensors=1))
        parser = argparse.ArgumentParser()
        add_flash_arguments(parser)
        self.assertIs(flash_arguments(parser.parse_args([]))["qwen_8bit_common_tensors"], False)
        self.assertIs(flash_arguments(parser.parse_args(["--qwen-8bit-common-tensors"]))
                      ["qwen_8bit_common_tensors"], True)

    def test_prompt_caches_from_bf16_and_8_bit_weights_are_kept_apart(self):
        installed = SimpleNamespace(root="/missing", revision="r", model_id="m")
        off = _prompt_cache_contract(installed, RuntimeConfig())
        on = _prompt_cache_contract(installed, RuntimeConfig(qwen_8bit_common_tensors=True))
        self.assertNotIn("qwenCommonTensors", off)
        self.assertEqual(on.pop("qwenCommonTensors"), "affine8-group64-v1")
        self.assertEqual(on, off)

    def test_loading_packs_the_target_and_mtp_layer_only_when_enabled(self):
        support = get_support("swift1.5-qwen3.8-flash-next")
        model, experts = SimpleNamespace(args=object()), object()
        mtp, mtp_experts = object(), object()
        packed = []
        for enabled in (False, True):
            packed.clear()
            config = RuntimeConfig(mtp_enabled=True, qwen_8bit_common_tensors=enabled)
            with patch("deepseek_v4_ssd.qwen4_exp.load", return_value=(model, experts)), \
                 patch("deepseek_v4_ssd.model._load_qwen_mtp", return_value=(mtp, mtp_experts)), \
                 patch("deepseek_v4_ssd.qwen_common_tensors.quantize_common_tensors",
                       side_effect=lambda target: packed.append((target, target.mtp))):
                self.assertEqual(support.load(object(), config, {}, {}, object()), (model, experts))
            self.assertEqual(packed, [(model, mtp)] if enabled else [])


class PackedHeadMTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / "ngram.bin"
        np.resize(np.array([0x30, 0x38, 0x3C, 0x40], dtype=np.uint8), 200).tofile(path)
        self.store = qwen.NGramStore(path, NGram("ngram.bin", "F8_E4M3", 2, 1, 100, (0,) * 16, (100,) * 16))
        _, self.experts = experts_fixture()

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def test_mtp_drafts_with_a_packed_lm_head_and_keeps_greedy_tokens(self):
        mx.random.seed(23)
        args = tiny_args(num_hidden_layers=4, linear_num_value_heads=4, linear_num_key_heads=2,
                         linear_key_head_dim=16, linear_value_head_dim=16, ple_embed_dim=32,
                         ple_conv_kernel_size=2, ple_layer_ids=(2,), eos_token_id=31,
                         indexer_budget=4, layer_types=("linear_attention",) * 3 + ("full_attention",))
        model = qwen.Model(args, self.experts, self.store)
        model.set_dtype(mx.float32)
        model.lm_head = CommonQuantizedLinear.from_linear(model.lm_head, group_size=64, bits=8)
        draft = qwen.MTPModel(args, self.experts)
        prompt, count = [1, 2, 3, 4, 5, 6, 7, 8, 9], 10

        cache = model.make_cache()
        logits, _ = model.forward_with_hidden(mx.array([prompt]), cache)
        sequence = list(prompt)
        for _ in range(count):
            sequence.append(int(mx.argmax(logits[0, -1]).item()))
            logits, _ = model.forward_with_hidden(mx.array([sequence[-1:]]), cache)
            eval_prompt_cache(cache, logits)

        rounds = []
        tokens = [token for token, _ in qwen.generate_mtp_tokens(
            prompt, model, draft, model.make_cache(), max_tokens=count, prefill_step_size=4,
            draft_tokens=2, zero_acceptance_limit=32,
            record_round=lambda proposed, accepted, *_: rounds.append(proposed))]
        self.assertEqual(tokens, sequence[len(prompt):])
        self.assertTrue(rounds and all(rounds))


if __name__ == "__main__":
    unittest.main()
