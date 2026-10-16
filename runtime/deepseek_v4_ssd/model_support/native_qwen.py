"""Qwen3.8 and Swift1.5 Qwen through the native engine (C++ and Metal, no MLX).

The existing model kinds keep their installation, chat format and sampling;
NativeQwenRuntime runs them. Settings of the MLX engine must keep
their defaults here, except 8-bit resident weights, which the native engine also has.
"""
from __future__ import annotations

from ..qwen_flash_config import DEFAULTS, validate_flash_config
from .qwen import QwenSupport


class NativeQwenSupport(QwenSupport):
    def validate_config(self, config) -> None:
        validate_flash_config(config)
        for name, default in DEFAULTS.items():
            if name != "qwen_8bit_common_tensors" and getattr(config, name, default) != default:
                raise ValueError(f"{name} is not used by the native Qwen engine")
        for name in ("dspark_enabled", "staged_expert_streaming", "qwen_quantized_kv",
                     "qwen_quantized_index", "deepseek_ane_prefill", "v41_ced_prefill",
                     "v41_packed_kv", "v41_packed_index", "v41_candidate_index"):
            if getattr(config, name, False):
                raise ValueError(f"{name} is not supported by the native Qwen engine")
        if type(getattr(config, "native_automatic_expert_cache", False)) is not bool:
            raise ValueError("native_automatic_expert_cache must be a boolean")

    def open_runtime(self, model_path, config):
        from ..native_qwen import NativeQwenRuntime
        return NativeQwenRuntime.open(model_path, config, self.descriptor.kind)

    def load(self, installed, config, raw_config, weights, read_limiter):
        raise ValueError("The native Qwen engine is opened by NativeQwenRuntime")

    def prefill(self, model, tokens, cache, step_size, expert_cache, config):
        raise ValueError("The native Qwen engine is opened by NativeQwenRuntime")
