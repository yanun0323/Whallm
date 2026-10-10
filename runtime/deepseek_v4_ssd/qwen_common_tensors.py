"""Opt-in 8-bit common tensors for the Qwen engine.

Every BF16 projection with at least 256 output rows, lm_head and the MTP layer
included, is stored as affine 8-bit with group size 64. The routers
(``mlp.gate``), the embedding, smaller gates, convolutions and norms stay BF16.
Products of up to 16 rows (Decode and MTP verification) use MLX's 8-bit kernel,
which is faster there than its BF16 kernel. Larger products (Prefill) multiply
by a temporary BF16 copy of the 8-bit weights, because MLX's 8-bit kernel is the
slower one at that size. Output tokens can differ from the BF16 model.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

BITS = 8
GROUP_SIZE = 64
MINIMUM_ROWS = 256
DECODE_ROWS = 16


class CommonQuantizedLinear(nn.QuantizedLinear):
    """8-bit weights; products of more than 16 rows use a temporary BF16 copy."""

    def __call__(self, x: mx.array) -> mx.array:
        if x.size // x.shape[-1] <= DECODE_ROWS:
            return super().__call__(x)
        weight = mx.dequantize(self["weight"], self["scales"], self.get("biases"),
                               group_size=self.group_size, bits=self.bits, mode=self.mode)
        if "bias" in self:
            return mx.addmm(self["bias"], x, weight.T)
        return x @ weight.T


def quantize_common_tensors(model: nn.Module) -> int:
    """Replace the chosen BF16 projections in place and return how many changed.

    One projection at a time: its BF16 weight is released as soon as the 8-bit
    form replaces it, so loading never holds a second full copy.
    """
    paths = [path for path, module in model.named_modules()
             if isinstance(module, nn.Linear) and module.weight.shape[0] >= MINIMUM_ROWS
             and not path.endswith("mlp.gate")]
    for path in paths:
        *names, name = path.split(".")
        parent = model
        for part in names:
            parent = parent[int(part)] if part.isdigit() else parent[part]
        key = int(name) if name.isdigit() else name
        packed = CommonQuantizedLinear.from_linear(parent[key], group_size=GROUP_SIZE, bits=BITS)
        mx.eval(packed.parameters())
        parent[key] = packed
    mx.clear_cache()
    return len(paths)
