"""Default-off Swift text Prefill experiment, not an adopted speed improvement.

The per-instance call follows mlx-lm qwen3_5.GatedDeltaNet (Apple Inc., MIT).
Only the recurrent kernel changes. The owning Swift DecoderLayer supplies the
existing sigmoid output norm. Other models never instantiate this subclass.
"""
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache
from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel
from mlx_lm.models.qwen3_5 import GatedDeltaNet

from .cancellation import check_cancelled
from .qwen_gdn_packed_source import PACKED_SOURCE


@lru_cache(maxsize=1)
def supported_gpu():
    # Equality was checked on M5. Do not extend it to other GPU generations by
    # assuming the compiler lowers their old simd_sum in the same way.
    return mx.metal.is_available() and mx.device_info().get("architecture", "").startswith("applegpu_g17")


@lru_cache(maxsize=1)
def _kernel():
    return mx.fast.metal_kernel(
        name="whallm_swift_gdn_packed_v1",
        input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
        output_names=["y", "state_out"], source=PACKED_SOURCE,
        ensure_row_contiguous=True,
    )


def packed_gated_delta_kernel(q, k, v, g, beta, state, mask=None):
    """Use exactly the tested geometry; never retry an admitted GPU failure."""
    length = q.shape[1] if q.ndim == 4 else 0
    supported = (
        mask is None and 2 <= length <= 1024
        and q.shape == k.shape == (1, length, 16, 128)
        and v.shape == (1, length, 48, 128)
        and g.shape == beta.shape == (1, length, 48)
        and state.shape == (1, 48, 128, 128)
        and q.dtype == k.dtype == v.dtype
        and q.dtype in (mx.bfloat16, mx.float16, mx.float32)
        and g.dtype == state.dtype == mx.float32
        and beta.dtype in (q.dtype, mx.float32)
        and mx.default_device() == mx.gpu and supported_gpu()
    )
    if not supported:
        return gated_delta_kernel(q, k, v, g, beta, state, mask)
    outputs = _kernel()(
        inputs=[q, k, v, g, beta, state, length],
        template=[("InT", q.dtype), ("StT", state.dtype), ("Dk", 128),
                  ("Dv", 128), ("Hk", 16), ("Hv", 48)],
        grid=(32, 16, 48), threadgroup=(32, 2, 1),
        output_shapes=[v.shape, state.shape], output_dtypes=[q.dtype, state.dtype],
    )
    if (len(outputs) != 2 or outputs[0].shape != v.shape or outputs[0].dtype != q.dtype
            or outputs[1].shape != state.shape or outputs[1].dtype != state.dtype):
        raise RuntimeError("Invalid Packed GDN output. Disable Packed GDN prefill and reload the model.")
    return outputs


class PackedGatedDeltaNet(GatedDeltaNet):
    """A local override, never a process-wide patch of mlx-lm functions.

Projection, convolution, mean-RMS q/k normalization, gate computation and output
projection retain the pinned qwen3_5 call order. Ragged/masked, training, sharded,
Decode and explicitly excluded image/verification calls use the original class.
"""

    def __call__(self, inputs, mask=None, cache=None, *, allow_packed=True):
        supported_cache = cache is None or (
            isinstance(cache, ArraysCache) and cache.lengths is None
            and cache.left_padding is None
            and (cache[1] is None or cache[1].dtype == mx.float32)
        )
        if (not allow_packed or self.training or self.sharding_group is not None
                or mask is not None or not supported_cache
                or inputs.ndim != 3 or inputs.shape[0] != 1
                or not 2 <= inputs.shape[1] <= 1024
                or (self.num_k_heads, self.num_v_heads, self.head_k_dim, self.head_v_dim,
                    self.conv_kernel_size) != (16, 48, 128, 128, 4)
                or mx.default_device() != mx.gpu or not supported_gpu()):
            return super().__call__(inputs, mask, cache)
        check_cancelled()
        B, S, _ = inputs.shape
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)
        conv_state = cache[0] if cache is not None and cache[0] is not None else mx.zeros(
            (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        next_conv = mx.contiguous(conv_input[:, -(self.conv_kernel_size - 1):, :]) if cache is not None else None
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1] if cache is not None else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        beta = mx.sigmoid(b)
        g = compute_g(self.A_log, a, self.dt_bias)
        if state is None:
            state = mx.zeros((B, self.num_v_heads, self.head_v_dim, self.head_k_dim), dtype=mx.float32)
        out, state = packed_gated_delta_kernel(q, k, v, g, beta, state)
        out = self.norm(out, z)
        out = self.out_proj(out.reshape(B, S, -1))
        # Do not publish partially advanced state on synchronous errors/cancel.
        # Lazy GPU failures remain owned by the enclosing request, which drains
        # its work and discards that request's cache rather than retrying math.
        check_cancelled()
        if cache is not None:
            cache[0] = next_conv
            cache[1] = state
            cache.advance(S)
        return out
