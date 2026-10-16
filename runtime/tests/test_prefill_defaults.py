"""V4.1 prefill defaults stay consistent across Python and both CLI entry points."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from deepseek_v4_ssd import cli, server
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_support import get_support


class PrefillDefaultTests(unittest.TestCase):
    def test_default_config_is_valid_for_every_model_and_preserves_opt_out(self):
        from v41_mlx import mlx_v41_support
        for kind in ('deepseek-v4', 'deepseek-v4.1', 'qwen3.8-flash-next'):
            # The defaults concern V4.1's MLX runtime; the native engine ignores them.
            support = mlx_v41_support() if kind == 'deepseek-v4.1' else get_support(kind)
            for enabled in (True, False):
                with self.subTest(kind=kind, enabled=enabled):
                    config = (RuntimeConfig() if enabled else RuntimeConfig(
                        v41_layer_major_prefill=False, v41_next_layer_prefetch=False))
                    support.validate_config(config)
                    self.assertEqual(config.v41_layer_major_prefill, enabled)
                    self.assertEqual(config.v41_next_layer_prefetch, enabled)
                    if kind == 'deepseek-v4.1':
                        self.assertEqual(support.uses_layer_major_prefill(config, 10_000), enabled)

    def test_both_entry_points_follow_runtime_defaults_and_independent_overrides(self):
        for flags, expected in (
            ([], (True, True)),
            (['--no-v41-layer-major-prefill'], (False, True)),
            (['--no-v41-next-layer-prefetch'], (True, False)),
            (['--no-v41-layer-major-prefill', '--no-v41-next-layer-prefetch'], (False, False)),
            (['--v41-layer-major-prefill', '--v41-next-layer-prefetch'], (True, True)),
        ):
            with self.subTest(flags=flags):
                args = server._parser().parse_args(['--model', '/unused', *flags])
                self.assertEqual((args.v41_layer_major_prefill, args.v41_next_layer_prefetch), expected)
                installed = SimpleNamespace(is_qwen=False, has_mtp=False)
                with patch('sys.argv', ['cli', '--model', '/unused', '--prompt', 'hello', *flags]), \
                     patch.object(cli.InstalledModel, 'open', return_value=installed), \
                     patch.object(cli.ModelRuntime, 'open', side_effect=RuntimeError('captured')) as opened:
                    with self.assertRaisesRegex(RuntimeError, 'captured'):
                        cli.main()
                config = opened.call_args.args[1]
                self.assertEqual((config.v41_layer_major_prefill, config.v41_next_layer_prefetch), expected)

    def test_v41_flags_do_not_enable_qwen_prefetch_or_other_experiments(self):
        config = RuntimeConfig()
        self.assertFalse(config.qwen_next_layer_prefetch)
        self.assertFalse(config.v41_ced_prefill)
        self.assertFalse(config.v41_packed_kv)
        with self.assertRaises(ValueError):
            get_support('qwen3.8-flash-next').validate_config(RuntimeConfig(v41_packed_kv=True))
