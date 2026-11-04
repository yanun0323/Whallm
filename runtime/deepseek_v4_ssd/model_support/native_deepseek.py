"""DeepSeek V4.1 through the native engine (C++ and Metal, no MLX).

The model kind keeps its installation, chat format and sampling; NativeDeepSeekRuntime
runs it. MLX tuning settings do not apply; settings that would change the numbers are
refused rather than ignored.
"""
from __future__ import annotations

import math
from .deepseek_v41 import DeepSeekV41Support

# MLX V4.1 options without a native counterpart that change outputs or memory use.
_UNSUPPORTED = ("v41_packed_kv", "v41_packed_index", "v41_candidate_index", "v41_ced_prefill",
                "dspark_sequential_verification", "mtp_enabled")


class NativeDeepSeekV41Support(DeepSeekV41Support):
    def validate_config(self, config) -> None:
        super().validate_config(config)
        ratio = getattr(config, "ane_prefill_ratio", 0)
        enabled = getattr(config, "deepseek_ane_prefill", False)
        if type(enabled) is not bool or isinstance(ratio, bool) \
                or not isinstance(ratio, (int, float)) or not math.isfinite(ratio) or not 0 <= ratio <= 1:
            raise ValueError("ANE prefill requires a boolean switch and a finite share from 0 to 1")
        if not enabled and ratio > 0:
            raise ValueError("ANE prefill share requires deepseek_ane_prefill=true")
        for name in _UNSUPPORTED:
            if getattr(config, name, False):
                raise ValueError(f"{name} is not supported by the native DeepSeek engine")
        if type(getattr(config, "native_automatic_expert_cache", False)) is not bool:
            raise ValueError("native_automatic_expert_cache must be a boolean")

    def open_runtime(self, model_path, config):
        from ..native_deepseek import NativeDeepSeekRuntime
        return NativeDeepSeekRuntime.open(model_path, config, self.descriptor.kind)

    def load(self, installed, config, raw_config, weights, read_limiter):
        raise ValueError("The native DeepSeek engine is opened by NativeDeepSeekRuntime")

    def prefill(self, model, tokens, cache, step_size, expert_cache, config):
        raise ValueError("The native DeepSeek engine is opened by NativeDeepSeekRuntime")
