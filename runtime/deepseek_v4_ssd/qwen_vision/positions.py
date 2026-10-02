"""Qwen interleaved T/H/W positions, including QSA pooled-key positions.

Equations follow the Qwen4Exp reference, not sequential text-only RoPE.
The position map belongs to one request; it never changes the loaded model.
"""
from __future__ import annotations

import math
import numpy as np
import mlx.core as mx

from ..media import MediaError


def image_positions(token_count, spans, merge=2):
    positions = np.empty((3, token_count), dtype=np.int32)
    end, current = 0, 0
    for span in spans:
        t, h, w = span.grid
        if (span.kind != "image" or t != 1 or h <= 0 or w <= 0 or h % merge or w % merge
                or span.length != h * w // merge**2 or span.start < end
                or span.start + span.length > token_count):
            raise MediaError("Invalid Qwen image positions.")
        length = span.start - end
        positions[:, end:span.start] = np.arange(length)[None] + current
        current += length
        coords = np.indices((1, h // merge, w // merge)).reshape(3, -1)
        positions[:, span.start:span.start + span.length] = coords + current
        current += max(h, w) // merge
        end = span.start + span.length
    positions[:, end:] = np.arange(token_count - end)[None] + current
    return positions


class MultimodalRoPE:
    def __init__(self, positions, args):
        positions = np.asarray(positions)
        if (positions.ndim != 2 or positions.shape[0] != 3 or not positions.shape[1]
                or not np.issubdtype(positions.dtype, np.integer) or np.any(positions < 0)):
            raise MediaError("Invalid Qwen image positions.")
        self.positions = mx.array(positions, dtype=mx.int32)
        self.length = positions.shape[1]
        self.delta = int(positions.max()) + 1 - self.length
        self.dim = int(args.head_dim * args.partial_rotary_factor)
        self.theta = args.rope_theta
        sections = (args.rope_parameters or {}).get("mrope_section", [11, 11, 10])
        if len(sections) != 3 or sum(sections) * 2 != self.dim:
            raise MediaError("Unsupported Qwen image rotary configuration.")
        axes = np.zeros(self.dim // 2, dtype=np.int32)
        for axis in (1, 2):
            axes[axis:sections[axis] * 3:3] = axis
        self.axes = mx.array(axes)

    def at(self, indices):
        selected = mx.take(self.positions, mx.minimum(indices, self.length - 1), axis=1)
        return mx.where(indices[None] < self.length, selected, indices[None] + self.delta)

    def rotate(self, value, indices, sequence_axis):
        positions = self.at(indices)
        frequencies = mx.exp(-math.log(self.theta) * mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        angles = positions[self.axes].T.astype(mx.float32) * frequencies
        # The reference casts cos/sin to the hidden dtype before multiplying.
        shape = [1] * value.ndim
        shape[sequence_axis], shape[-1] = len(indices), self.dim
        cosine = mx.concatenate([mx.cos(angles)] * 2, axis=-1).astype(value.dtype).reshape(shape)
        sine = mx.concatenate([mx.sin(angles)] * 2, axis=-1).astype(value.dtype).reshape(shape)
        first, second = mx.split(value[..., :self.dim], 2, axis=-1)
        rotary = value[..., :self.dim] * cosine + mx.concatenate([-second, first], axis=-1) * sine
        return mx.concatenate([rotary, value[..., self.dim:]], axis=-1)


class ImageGenerationModel:
    """Small per-request adapter for mlx-lm's chunked embedding input interface."""
    def __init__(self, model, positions):
        self._model = model
        self._positions = MultimodalRoPE(positions, model.args)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, tokens, cache=None, *, input_embeddings=None):
        return self._model(tokens, cache=cache, input_embeddings=input_embeddings,
                           rope_positions=self._positions)
