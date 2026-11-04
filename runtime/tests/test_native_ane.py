"""Native ANE configuration and stale-library checks; hardware projection check is Native/ANEBridge/check.sh."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest import TestCase, mock

from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd import native_deepseek, native_qwen


class NativeANEConfigurationTests(TestCase):
    def test_both_native_families_accept_opt_in_and_qwen_eight_bit(self):
        deepseek = get_support("deepseek-v4.1")
        qwen = get_support("qwen3.8-flash-next")
        swift = get_support("swift1.5-qwen3.8-flash-next")
        deepseek.validate_config(replace(RuntimeConfig(), deepseek_ane_prefill=True, ane_prefill_ratio=.25))
        selected = replace(RuntimeConfig(), ane_prefill=True, ane_prefill_ratio=.5,
                           qwen_8bit_common_tensors=True)
        qwen.validate_config(selected)
        swift.validate_config(selected)

    def test_invalid_share_and_disabled_switch_are_rejected(self):
        deepseek = get_support("deepseek-v4.1")
        qwen = get_support("qwen3.8-flash-next")
        for ratio in (True, float("nan"), float("inf"), -.01, 1.01):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                deepseek.validate_config(replace(RuntimeConfig(), deepseek_ane_prefill=True,
                                                ane_prefill_ratio=ratio))
            with self.subTest(qwen_ratio=ratio), self.assertRaises(ValueError):
                qwen.validate_config(replace(RuntimeConfig(), ane_prefill=True,
                                             ane_prefill_ratio=ratio))
        with self.assertRaisesRegex(ValueError, "requires deepseek_ane_prefill"):
            deepseek.validate_config(replace(RuntimeConfig(), ane_prefill_ratio=.25))
        with self.assertRaisesRegex(ValueError, "requires ane_prefill"):
            qwen.validate_config(replace(RuntimeConfig(), ane_prefill=False, ane_prefill_ratio=.25))

    def test_old_native_libraries_fail_before_open(self):
        for module in (native_deepseek, native_qwen):
            with self.subTest(module=module.__name__), \
                    mock.patch.object(module, "_library_candidates", return_value=[Path(__file__)]), \
                    mock.patch.object(module.ctypes, "CDLL", return_value=object()), \
                    mock.patch.object(module, "_library_value", None):
                with self.assertRaisesRegex(RuntimeError, "older ABI"):
                    module._library()
