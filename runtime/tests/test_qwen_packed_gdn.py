"""Packed GDN experiment: config, byte parity and owned-state regressions.

Synthetic layer tests do not establish full-model quality or speed.
"""
import argparse
import copy
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet

from deepseek_v4_ssd import qwen4_exp as qwen
from deepseek_v4_ssd import qwen_gdn_packed as packed
from deepseek_v4_ssd.cancellation import GenerationCancelled, cancellation_scope
from deepseek_v4_ssd.generation import _prompt_cache_contract
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import _parse_runtime
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.qwen_flash_config import add_flash_arguments, flash_arguments, validate_flash_config

SWIFT = "swift1.5-qwen3.8-flash-next"


def bits(a):
    mx.eval(a)
    return np.asarray(mx.contiguous(a).view(mx.uint8)).tobytes()


class PackedConfigTests(unittest.TestCase):
    def test_legacy_catalog_defaults_off_and_real_boolean_required(self):
        raw = asdict(RuntimeConfig())
        self.assertFalse(raw.pop('qwen_packed_gdn_prefill'))
        self.assertFalse(_parse_runtime(raw, 'runtime', SWIFT).qwen_packed_gdn_prefill)
        for value in (1, 0, None, 'true', []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'boolean'):
                validate_flash_config(replace(RuntimeConfig(), qwen_packed_gdn_prefill=value))

    def test_cli_default_enable_and_disable(self):
        parser = argparse.ArgumentParser(); add_flash_arguments(parser)
        for flags, expected in (([], False), (['--qwen-packed-gdn-prefill'], True),
                                (['--qwen-packed-gdn-prefill', '--no-qwen-packed-gdn-prefill'], False)):
            self.assertEqual(flash_arguments(parser.parse_args(flags))['qwen_packed_gdn_prefill'], expected)

    def test_only_swift_and_no_mtp(self):
        config = RuntimeConfig(qwen_packed_gdn_prefill=True)
        get_support(SWIFT).validate_config(config)
        for kind in ('qwen3.8-flash-next', 'deepseek-v4', 'deepseek-v4.1', 'mimo-v2.6-flash-rl'):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                get_support(kind).validate_config(config)
        with self.assertRaisesRegex(ValueError, 'MTP'):
            get_support(SWIFT).validate_config(replace(config, mtp_enabled=True))
        get_support(SWIFT).validate_config(RuntimeConfig(mtp_enabled=True))

    def test_direct_loader_rejects_wrong_kind_or_mtp_before_io(self):
        for kind, mtp in (('qwen3.8-flash-next', False), (SWIFT, True)):
            with patch.object(qwen, 'ExpertCache') as cache, self.assertRaisesRegex(ValueError, 'Swift'):
                qwen.load(SimpleNamespace(model_kind=kind),
                          RuntimeConfig(qwen_packed_gdn_prefill=True, mtp_enabled=mtp), {}, {})
            cache.assert_not_called()

    def test_prompt_cache_contract_separates_on_and_preserves_legacy_off(self):
        with tempfile.TemporaryDirectory() as folder:
            installed = SimpleNamespace(root=Path(folder), revision='fixture', model_id='fixture')
            off = _prompt_cache_contract(installed, RuntimeConfig())
            legacy = asdict(RuntimeConfig()); legacy.pop('qwen_packed_gdn_prefill')
            self.assertEqual(off, _prompt_cache_contract(installed, SimpleNamespace(**legacy)))
            on = _prompt_cache_contract(installed, RuntimeConfig(qwen_packed_gdn_prefill=True))
            self.assertNotEqual(off, on)
            self.assertEqual(on.pop('qwenPackedGDNPrefill'), 'm5-v1')
            self.assertEqual(off, on)


class PackedLayerTests(unittest.TestCase):
    def layers(self, dtype=mx.bfloat16):
        mx.random.seed(1004)
        args = qwen.ModelArgs(hidden_size=32)
        baseline = GatedDeltaNet(args)
        candidate = packed.PackedGatedDeltaNet(args)
        for layer in (baseline, candidate):
            # Preserve this project's sigmoid, not upstream's SiLU output gate.
            layer.norm = qwen.RMSNormGated(128, 1e-6)
            layer.set_dtype(dtype); layer.eval()
        candidate.load_weights(tree_flatten(baseline.parameters()))
        mx.eval(baseline.parameters(), candidate.parameters())
        return baseline, candidate

    def test_real_layer_chain_is_byte_equal_for_each_activation_type(self):
        if not packed.supported_gpu(): self.skipTest('packed numerical coverage requires M5')
        for dtype in (mx.bfloat16, mx.float16, mx.float32):
            base, candidate = self.layers(dtype)
            a, b = ArraysCache(size=4), ArraysCache(size=4)
            a[2] = b[2] = mx.array([[17]], dtype=mx.int32)
            with patch.object(packed, '_kernel', wraps=packed._kernel) as dispatch:
                for length in (2, 7, 127, 128, 1023, 1024, 1):
                    with self.subTest(dtype=str(dtype), length=length):
                        x = (mx.random.normal((1, length, 32)) * .1).astype(dtype)
                        left, right = base(x, cache=a), candidate(x, cache=b)
                        self.assertEqual(bits(left), bits(right))
                        self.assertEqual(bits(a[0]), bits(b[0]))
                        self.assertEqual(bits(a[1]), bits(b[1]))
                        self.assertEqual(b[1].dtype, mx.float32)
                        self.assertEqual(b[2].tolist(), [[17]])
                        self.assertIsNone(b[3])
                self.assertEqual(dispatch.call_count, 6, 'Decode must not use the packed kernel')

    def test_unsupported_calls_use_original_before_dispatch(self):
        _, candidate = self.layers()
        x = mx.zeros((1, 7, 32), dtype=mx.bfloat16)
        for mode in ('decode', 'long', 'batch', 'mask', 'lengths', 'padding', 'training', 'sharding', 'device', 'excluded'):
            cache = ArraysCache(size=4); values = x; mask = None; kwargs = {}
            if mode == 'decode': values = x[:, :1]
            if mode == 'long': values = mx.zeros((1, 1025, 32), dtype=x.dtype)
            if mode == 'batch': values = mx.zeros((2, 7, 32), dtype=x.dtype)
            if mode == 'mask': mask = mx.ones((1, 7), dtype=mx.bool_)
            if mode == 'lengths': cache.lengths = mx.array([7])
            if mode == 'padding': cache.left_padding = mx.array([0])
            if mode == 'training': candidate.train()
            if mode == 'sharding': candidate.sharding_group = object()
            if mode == 'excluded': kwargs['allow_packed'] = False
            try:
                with self.subTest(mode=mode), patch.object(GatedDeltaNet, '__call__', return_value='original') as original, \
                     patch.object(packed, 'supported_gpu', return_value=mode != 'device'), patch.object(packed, '_kernel') as kernel:
                    self.assertEqual(candidate(values, mask, cache, **kwargs), 'original')
                    original.assert_called_once(); kernel.assert_not_called()
            finally:
                candidate.eval(); candidate.sharding_group = None

    def test_dispatch_failure_and_cancel_do_not_publish_state_or_retry(self):
        if not packed.supported_gpu(): self.skipTest('requires admitted M5 path')
        _, candidate = self.layers()
        x = mx.zeros((1, 7, 32), dtype=mx.bfloat16)
        for mode in ('error', 'cancel'):
            cache = ArraysCache(size=4); before = list(cache.state); event = Event()
            original = packed.packed_gated_delta_kernel
            def dispatch(*args, **kwargs):
                if mode == 'error': raise RuntimeError('injected dispatch failure')
                out = original(*args, **kwargs); event.set(); return out
            with cancellation_scope(event), patch.object(packed, 'packed_gated_delta_kernel', side_effect=dispatch) as called, \
                 patch.object(GatedDeltaNet, '__call__') as fallback:
                with self.assertRaises((RuntimeError, GenerationCancelled)):
                    candidate(x, cache=cache)
                self.assertEqual(cache.state, before)
                called.assert_called_once(); fallback.assert_not_called()
            mx.synchronize()

    def test_instances_do_not_patch_upstream_or_change_parameter_names(self):
        import mlx_lm.models.gated_delta as delta
        old = delta.gated_delta_kernel
        base, candidate = self.layers()
        self.assertIs(delta.gated_delta_kernel, old)
        self.assertEqual([x[0] for x in tree_flatten(base.parameters())],
                         [x[0] for x in tree_flatten(candidate.parameters())])
        self.assertEqual(type(base), GatedDeltaNet)

    def test_text_model_excludes_image_embeddings_and_verification_capture(self):
        args = qwen.ModelArgs(hidden_size=32, vocab_size=64, num_hidden_layers=1,
                              layer_types=['linear_attention'], ple_layer_ids=[])
        model = qwen.TextModel(args, SimpleNamespace(), None, packed_gdn_prefill=True)
        seen = []
        def layer(hidden, *args, **kwargs):
            seen.append(kwargs.get('allow_packed_gdn', True)); return hidden
        model.layers = [layer]
        ids = mx.array([[1, 2, 3]])
        model.hidden_states(ids)
        model.hidden_states(ids, input_embeddings=mx.zeros((1, 3, 32)))
        captured = []; model.hidden_states(ids, capture=captured)
        self.assertEqual(seen, [True, False, False])
        self.assertEqual(len(captured), 1)

    def test_invalid_output_metadata_is_an_error_not_a_fallback(self):
        if not packed.supported_gpu(): self.skipTest('requires admitted M5 path')
        _, candidate = self.layers()
        cache = ArraysCache(size=4)
        with patch.object(packed, '_kernel', return_value=lambda **kw: [mx.zeros((1,))]), \
             patch.object(packed, 'gated_delta_kernel') as fallback:
            with self.assertRaisesRegex(RuntimeError, 'Invalid Packed GDN output'):
                candidate(mx.zeros((1, 2, 32), dtype=mx.bfloat16), cache=cache)
        self.assertEqual(cache.state, [None] * 4)
        fallback.assert_not_called()

    def test_image_and_rollback_layer_calls_explicitly_exclude_packed(self):
        # Exercise DecoderLayer's real dispatch without allocating a full model.
        args = qwen.ModelArgs(hidden_size=32, num_hidden_layers=1,
                              layer_types=['linear_attention'], ple_layer_ids=[])
        layer = qwen.DecoderLayer(args, 0, SimpleNamespace(), None, packed_gdn_prefill=True)
        class Mixer:
            def __call__(self, x): return x, x, x
            def inject(self, residual, result, injection): return result
        layer.attn_hyper_connection = Mixer(); layer.mlp_hyper_connection = Mixer()
        layer.mlp = lambda x: x
        linear = Mock(side_effect=lambda x, *a, **kw: x); layer.linear_attn = linear
        x = mx.zeros((1, 3, 32)); ids = mx.array([[1, 2, 3]])
        for rope, allowed, expected in ((None, True, True), (object(), True, False), (None, False, False)):
            layer(x, ids, None, None, rope, allow_packed_gdn=allowed)
            self.assertEqual(linear.call_args.kwargs['allow_packed'], expected)
        with patch.object(qwen, 'create_ssm_mask', return_value=None):
            layer.advance_state(x, ids, None)
        self.assertFalse(linear.call_args.kwargs['allow_packed'])
