"""DeepSeek V4.1 through the native engine (C++ and Metal, no MLX).

The model kind keeps its installation, chat format and sampling; NativeDeepSeekRuntime
runs it. MLX tuning settings do not apply; settings that would change the numbers are
refused rather than ignored.
"""
from __future__ import annotations

from .deepseek_v41 import DeepSeekV41Support

# MLX V4.1 options without a native counterpart that change outputs or memory use.
_UNSUPPORTED = ("v41_packed_kv", "v41_packed_index", "v41_candidate_index", "v41_ced_prefill",
                "deepseek_ane_prefill", "dspark_sequential_verification", "mtp_enabled")


class NativeDeepSeekV41Support(DeepSeekV41Support):
    def validate_config(self, config) -> None:
        super().validate_config(config)
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
