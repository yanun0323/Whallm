from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .event_log import logger as _LOG
_ANE_CHANNEL_GRANULARITY = 256
_ANE_SPATIAL = 1_024
_QWEN_INPUT_CHANNELS = 2_560
_QWEN_OUTPUT_CHANNELS = 12_288


def _library_candidates() -> list[Path]:
    candidates = []
    configured = os.environ.get("WHALLM_ANE_BRIDGE")
    if configured:
        candidates.append(Path(configured))
    executable = Path(sys.executable).resolve()
    candidates.append(
        executable.parent.parent / "Frameworks" / "libWhallmANE.dylib"
    )
    project_root = Path(__file__).resolve().parents[2]
    candidates.append(project_root / ".build" / "native" / "libWhallmANE.dylib")
    return candidates


class _ANELibrary:
    def __init__(self, path: Path):
        self.path = path
        self._library = ctypes.CDLL(str(path))
        self._library.whallm_ane_projection_create.argtypes = (
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        )
        self._library.whallm_ane_projection_create.restype = ctypes.c_void_p
        self._library.whallm_ane_projection_evaluate.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t,
        )
        self._library.whallm_ane_projection_evaluate.restype = ctypes.c_int
        self._library.whallm_ane_projection_destroy.argtypes = (ctypes.c_void_p,)
        self._library.whallm_ane_projection_destroy.restype = None
        self._library.whallm_ane_last_error.argtypes = ()
        self._library.whallm_ane_last_error.restype = ctypes.c_char_p

    @classmethod
    def open(cls) -> _ANELibrary:
        errors = []
        for path in _library_candidates():
            if not path.is_file():
                continue
            try:
                return cls(path)
            except OSError as error:
                errors.append(f"{path}: {error}")
        detail = "; ".join(errors) if errors else "bridge library not found"
        raise RuntimeError(detail)

    def last_error(self) -> str:
        value = self._library.whallm_ane_last_error()
        return value.decode("utf-8", errors="replace") if value else "unknown error"

    def create(self, weight: np.ndarray, spatial: int) -> _ANEProjection:
        weight = np.ascontiguousarray(weight, dtype=np.float16)
        handle = self._library.whallm_ane_projection_create(
            weight.view(np.uint16).ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint16)
            ),
            weight.size,
            weight.shape[1],
            weight.shape[0],
            spatial,
        )
        if not handle:
            raise RuntimeError(self.last_error())
        return _ANEProjection(self, handle, weight.shape[1], weight.shape[0], spatial)


class _ANEProjection:
    def __init__(
        self,
        library: _ANELibrary,
        handle: int,
        input_channels: int,
        output_channels: int,
        spatial: int,
    ):
        self._library = library
        self._handle = handle
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.spatial = spatial

    def evaluate(self, value: np.ndarray) -> np.ndarray:
        value = np.ascontiguousarray(value, dtype=np.float16)
        expected = (self.spatial, self.input_channels)
        if value.shape != expected:
            raise ValueError(f"ANE input shape must be {expected}")
        output = np.empty((self.spatial, self.output_channels), dtype=np.float16)
        result = self._library._library.whallm_ane_projection_evaluate(
            self._handle,
            value.view(np.uint16).ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint16)
            ),
            value.size,
            output.view(np.uint16).ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint16)
            ),
            output.size,
        )
        if result != 0:
            raise RuntimeError(self._library.last_error())
        return output

    def close(self) -> None:
        if self._handle:
            self._library._library.whallm_ane_projection_destroy(self._handle)
            self._handle = None


def _ane_output_channels(ratio: float) -> int:
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not 0 <= ratio <= 1
    ):
        raise ValueError("ANE Prefill ratio must be between zero and one")
    units = int(
        ratio * _QWEN_OUTPUT_CHANNELS / _ANE_CHANNEL_GRANULARITY + 0.5
    )
    return min(_QWEN_OUTPUT_CHANNELS, max(0, units * _ANE_CHANNEL_GRANULARITY))


class ANEPrefillController:
    def __init__(self, requested: bool, ratio: float = 0.0):
        self.output_channels = _QWEN_OUTPUT_CHANNELS
        self.requested = requested
        self.requested_ratio = ratio
        self.ane_channels = _ane_output_channels(ratio) if requested else 0
        self.active = False
        self.error: str | None = None
        self.evaluations = 0
        self.fallbacks = 0
        self._projections: list[_ANEProjection] = []

    def add(self, projection: _ANEProjection) -> None:
        self._projections.append(projection)

    def disable(self, error: object, *, fallback: bool = False) -> None:
        if fallback:
            self.fallbacks += 1
        if self.error is None:
            self.error = str(error)
            _LOG.info("ANE Prefill disabled; using original Prefill; error=%s", type(error).__name__)
        self.active = False
        for projection in self._projections:
            projection.close()
        self._projections.clear()

    def close(self) -> None:
        self.active = False
        for projection in self._projections:
            projection.close()
        self._projections.clear()

    def snapshot(self) -> dict[str, Any]:
        active_ane_channels = self.ane_channels if self.active else 0
        return {
            "requested": self.requested,
            "requested_ratio": self.requested_ratio,
            "active_ratio": active_ane_channels / self.output_channels,
            "ane_channels": active_ane_channels,
            "gpu_channels": self.output_channels - active_ane_channels,
            "active": self.active,
            "error": self.error,
            "evaluations": self.evaluations,
            "fallbacks": self.fallbacks,
        }


class ANEPrefillLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        projection: _ANEProjection,
        controller: ANEPrefillController,
    ):
        super().__init__()
        self.weight = linear.weight
        self.bias = getattr(linear, "bias", None)
        self._projection = projection
        self._controller = controller
        self._gpu_channels = self.weight.shape[0] - projection.output_channels

    def _original(self, value: mx.array) -> mx.array:
        projected = value @ self.weight.T
        return projected if self.bias is None else projected + self.bias

    def __call__(self, value: mx.array) -> mx.array:
        if (
            not self._controller.active
            or value.ndim != 3
            or value.shape[0] != 1
            or value.shape[1] != self._projection.spatial
            or value.shape[2] != self._projection.input_channels
        ):
            return self._original(value)

        try:
            ane_input = np.ascontiguousarray(
                np.asarray(value.astype(mx.float16))[0]
            )
            gpu_output = None
            if self._gpu_channels:
                gpu_output = value @ self.weight[: self._gpu_channels].T
                mx.async_eval(gpu_output)
            ane_output = self._projection.evaluate(ane_input)
            ane_output = mx.array(ane_output[None]).astype(value.dtype)
            self._controller.evaluations += 1
            return (
                ane_output
                if gpu_output is None
                else mx.concatenate([gpu_output, ane_output], axis=-1)
            )
        except Exception as error:
            self._controller.disable(error, fallback=True)
            return self._original(value)


def install_qwen_ane_prefill(
    model: Any,
    requested: bool,
    ratio: float = 0.0,
) -> ANEPrefillController:
    controller = ANEPrefillController(requested, ratio)
    if not requested or controller.ane_channels == 0:
        return controller
    try:
        library = _ANELibrary.open()
        attention_layers = [
            layer.self_attn
            for layer in model.model.layers
            if getattr(layer, "layer_type", None) == "full_attention"
        ]
        if len(attention_layers) != 12:
            raise ValueError("Qwen ANE Prefill requires 12 QSA layers")
        warmup = np.zeros((_ANE_SPATIAL, _QWEN_INPUT_CHANNELS), dtype=np.float16)
        for attention in attention_layers:
            weight = attention.q_proj.weight
            if weight.shape != (_QWEN_OUTPUT_CHANNELS, _QWEN_INPUT_CHANNELS):
                raise ValueError("Qwen ANE Prefill q_proj shape does not match")
            ane_weight = np.asarray(
                weight[-controller.ane_channels :].astype(mx.float16)
            )
            projection = library.create(ane_weight, _ANE_SPATIAL)
            controller.add(projection)
            for _ in range(3):
                projection.evaluate(warmup)
            attention.q_proj = ANEPrefillLinear(
                attention.q_proj,
                projection,
                controller,
            )
        controller.active = True
        return controller
    except Exception as error:
        controller.disable(error)
        return controller


class ANEQuantizedPrefillLinear(nn.Module):
    """Retain the original quantized GPU projection for Decode and all fallbacks."""
    def __init__(self, linear, projection, controller):
        super().__init__()
        import copy
        self._linear = linear
        self._projection = projection
        self._controller = controller
        self._gpu_channels = linear.weight.shape[0] - projection.output_channels
        self._gpu = copy.copy(linear)
        for key in ('weight', 'scales', 'biases', 'bias'):
            value = getattr(linear, key, None)
            if value is not None:
                self._gpu[key] = value[:self._gpu_channels]

    def __call__(self, value):
        if (not self._controller.active or value.ndim != 3 or value.shape[:2] != (1, self._projection.spatial)
                or value.shape[-1] != self._projection.input_channels):
            return self._linear(value)
        try:
            cpu = np.ascontiguousarray(np.asarray(value.astype(mx.float16))[0])
            gpu = self._gpu(value) if self._gpu_channels else None
            if gpu is not None:
                mx.async_eval(gpu)
            ane = mx.array(self._projection.evaluate(cpu)[None]).astype(value.dtype)
            bias = getattr(self._linear, 'bias', None)
            if bias is not None:
                ane = ane + bias[-self._projection.output_channels:]
            self._controller.evaluations += 1
            return mx.concatenate([gpu, ane], axis=-1) if gpu is not None else ane
        except Exception as error:
            self._controller.disable(error, fallback=True)
            return self._linear(value)


def install_deepseek_ane_prefill(model, requested, ratio=0.0):
    """Compile only the selected query-projection rows; leave quantized weights resident."""
    core = getattr(model, 'model', model)
    layers = core.layers
    outputs = layers[0].attn.wq_b.weight.shape[0]
    controller = ANEPrefillController(False, ratio)
    controller.requested = requested
    controller.output_channels = outputs
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not 0 <= ratio <= 1:
        raise ValueError('ANE Prefill ratio must be between zero and one')
    controller.ane_channels = min(outputs, int(ratio * outputs / 256 + 0.5) * 256) if requested else 0
    if not controller.ane_channels:
        return controller
    replacements = []
    try:
        library = _ANELibrary.open()
        for layer in layers:
            linear = layer.attn.wq_b
            rows = controller.ane_channels
            if linear.weight.shape[0] != outputs:
                raise ValueError('DeepSeek query projection widths must match')
            if hasattr(linear, 'scales'):
                weight = mx.dequantize(linear.weight[-rows:], linear.scales[-rows:],
                    biases=(linear.biases[-rows:] if getattr(linear, 'biases', None) is not None else None),
                    group_size=linear.group_size, bits=linear.bits, mode=linear.mode)
            else:
                weight = linear.weight[-rows:]
            projection = library.create(np.asarray(weight.astype(mx.float16)), _ANE_SPATIAL)
            controller.add(projection)
            projection.evaluate(np.zeros((_ANE_SPATIAL, weight.shape[1]), np.float16))
            replacements.append((layer.attn, ANEQuantizedPrefillLinear(linear, projection, controller)))
        # Publish only once every layer compiled successfully.
        for attention, linear in replacements:
            attention.wq_b = linear
        controller.active = True
    except Exception as error:
        controller.disable(error)
    return controller
