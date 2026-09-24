"""MiMo ViT with row/column windows and denominator sinks (SGLang reference)."""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..cancellation import check_cancelled


def positions(grid, merge=2):
    t, h, w = grid
    if t < 1 or h < 1 or w < 1 or h % merge or w % merge:
        raise ValueError("invalid MiMo vision grid")
    rows, cols = np.indices((h, w))
    def order(x):
        return x.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
    coords = np.tile(np.stack((order(rows), order(cols)), axis=-1), (t, 1))
    columns = np.arange(t * h * w // merge**2).reshape(t, h // merge, w // merge).transpose(0, 2, 1).reshape(-1)
    return mx.array(coords), mx.array(columns)


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(1, 1, bias=False)
        self.proj.weight = mx.zeros((config["hidden_size"], 3, config["temporal_patch_size"],
                                     config["patch_size"], config["patch_size"]), mx.bfloat16)

    def __call__(self, patches):
        w = self.proj.weight
        return patches.astype(w.dtype) @ w.reshape(w.shape[0], -1).T


class VisionMLP(nn.Module):
    def __init__(self, dim, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate)
        self.up_proj = nn.Linear(dim, intermediate)
        self.down_proj = nn.Linear(intermediate, dim)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class VisionAttention(nn.Module):
    def __init__(self, config, full):
        super().__init__()
        self.heads, self.kv = config["num_heads"], config["num_key_value_heads"]
        self.dim = config.get("qk_channels", 64)
        self.qkv = nn.Linear(config["hidden_size"], (self.heads + 2 * self.kv) * self.dim)
        self.proj = nn.Linear(self.heads * self.dim, config["hidden_size"])
        self.sinks = mx.zeros((self.heads,), mx.float32) if config["use_sink"] and not full else None
        self.window = -1 if full else config["visual_token_window_size"]

    def __call__(self, x, angles, grid):
        q, k, v = mx.split(self.qkv(x), [self.heads * self.dim, (self.heads + self.kv) * self.dim], axis=-1)
        q = q.reshape(-1, self.heads, self.dim)
        k, v = (a.reshape(-1, self.kv, self.dim) for a in (k, v))
        def rotate(a):
            first, second = mx.split(a.astype(mx.float32), 2, axis=-1)
            return (a.astype(mx.float32) * mx.cos(angles)[:, None] +
                    mx.concatenate((-second, first), axis=-1) * mx.sin(angles)[:, None]).astype(a.dtype)
        q, k = rotate(q), rotate(k)
        frames, height, width = grid
        per_frame = height * width
        def batch(a, heads):
            return a.reshape(frames, per_frame, heads, self.dim).transpose(0, 2, 1, 3)
        mask = None
        if self.window > 0:
            indices = mx.arange(per_frame)
            mask = mx.abs(indices[:, None] - indices[None, :]) <= self.window
        sinks = self.sinks.astype(q.dtype) if self.sinks is not None else None
        y = mx.fast.scaled_dot_product_attention(batch(q, self.heads), batch(k, self.kv), batch(v, self.kv),
                                                  scale=self.dim**-.5, mask=mask, sinks=sinks)
        return self.proj(y.transpose(0, 2, 1, 3).reshape(x.shape[0], -1))


class VisionBlock(nn.Module):
    def __init__(self, config, full):
        super().__init__()
        hidden = config["hidden_size"]
        self.norm1 = nn.RMSNorm(hidden, eps=1e-6)
        self.norm2 = nn.RMSNorm(hidden, eps=1e-6)
        self.attn = VisionAttention(config, full)
        self.mlp = VisionMLP(hidden, config["intermediate_size"])

    def __call__(self, x, angles, grid):
        x = x + self.attn(self.norm1(x), angles, grid)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, hidden, output, merge):
        super().__init__()
        # Pinned SGLang uses RMSNorm, unlike the upstream HF sketch. Its
        # missing merger biases are initialized to zero; the checkpoint has none.
        self.ln_q = nn.RMSNorm(hidden, eps=1e-6)
        self.mlp = [nn.Linear(hidden * merge**2, hidden * merge**2, bias=False), nn.GELU(),
                    nn.Linear(hidden * merge**2, output, bias=False)]
        self.width = hidden * merge**2

    def __call__(self, x):
        x = self.ln_q(x).reshape(-1, self.width)
        for layer in self.mlp:
            x = layer(x)
        return x


class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config["hidden_act"] != "silu" or config["spatial_merge_size"] != 2:
            raise ValueError("unsupported MiMo vision equations")
        self.patch_embed = PatchEmbed(config)
        self.blocks = [VisionBlock(config, i in config["fullatt_block_indexes"]) for i in range(config["depth"])]
        self.merger = PatchMerger(config["hidden_size"], config.get("out_hidden_size", 4096), 2)
        self.orders = tuple(config["vit_window_attn_types"])
        self.head_dim = config.get("qk_channels", 64)

    def __call__(self, patches, grid):
        coords, columns = positions(grid)
        frequency = 1 / (10000 ** (mx.arange(0, self.head_dim // 2, 2, dtype=mx.float32) / (self.head_dim // 2)))
        angles = (coords[..., None] * frequency).reshape(-1, self.head_dim // 2)
        angles = mx.concatenate((angles, angles), axis=-1)
        def reorder(x, index):
            return x.reshape(-1, 4, x.shape[-1])[index].reshape(x.shape)
        column_angles = reorder(angles, columns)
        reverse = mx.argsort(columns)
        x, column_order = self.patch_embed(patches), False
        for block, order in zip(self.blocks, self.orders):
            check_cancelled()
            if (order == 1) != column_order:
                x = reorder(x, columns if order == 1 else reverse)
                column_order = order == 1
            x = block(x, column_angles if column_order else angles, grid)
            mx.eval(x)
        if column_order:
            x = reorder(x, reverse)
        return self.merger(x)
