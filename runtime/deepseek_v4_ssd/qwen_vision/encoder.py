"""MLX still-image ViT following the Qwen4Exp reference equations.

LayerNorm (not MiMo RMSNorm), learned interpolated positions, full attention,
tanh GELU in blocks and exact GELU in the merger. Source weights stay BF16.
"""
from __future__ import annotations

import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from ..cancellation import check_cancelled


def grid_coordinates(grid, merge=2):
    t, h, w = grid
    if t != 1 or min(h, w) < merge or h % merge or w % merge:
        raise ValueError("Qwen vision accepts still-image grids only")
    coords = np.indices((h, w)).transpose(1, 2, 0)
    return coords.reshape(h // merge, merge, w // merge, merge, 2).transpose(0, 2, 1, 3, 4).reshape(-1, 2)


def interpolation(grid, side, merge=2):
    _, h, w = grid
    coords = grid_coordinates(grid, merge)
    # align_corners=True, with a zero coordinate for a singleton dimension.
    ys = np.linspace(0, side - 1, h, dtype=np.float32)[coords[:, 0]]
    xs = np.linspace(0, side - 1, w, dtype=np.float32)[coords[:, 1]]
    y0, x0 = np.floor(ys).astype(np.int32), np.floor(xs).astype(np.int32)
    y1, x1 = np.minimum(y0 + 1, side - 1), np.minimum(x0 + 1, side - 1)
    dy, dx = ys - y0, xs - x0
    indices = np.stack([y0 * side + x0, y0 * side + x1, y1 * side + x0, y1 * side + x1], axis=1)
    weights = np.stack([(1-dy)*(1-dx), (1-dy)*dx, dy*(1-dx), dy*dx], axis=1).astype(np.float32)
    return mx.array(indices), mx.array(weights)


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(1, config['hidden_size'])
        self.proj.weight = mx.zeros((config['hidden_size'], config['in_channels'],
            config['temporal_patch_size'], config['patch_size'], config['patch_size']), mx.bfloat16)

    def __call__(self, patches):
        weight = self.proj.weight
        return patches.astype(weight.dtype) @ weight.reshape(weight.shape[0], -1).T + self.proj.bias


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config['hidden_size']
        self.heads = config['num_heads']
        self.head_dim = hidden // self.heads
        self.qkv, self.proj = nn.Linear(hidden, hidden * 3), nn.Linear(hidden, hidden)

    def __call__(self, value, angles):
        q, k, v = mx.split(self.qkv(value).reshape(-1, 3, self.heads, self.head_dim), 3, axis=1)
        q, k, v = (x[:, 0] for x in (q, k, v))
        def rotate(x):
            first, second = mx.split(x.astype(mx.float32), 2, axis=-1)
            return (x.astype(mx.float32) * mx.cos(angles)[:, None] +
                    mx.concatenate([-second, first], axis=-1) * mx.sin(angles)[:, None]).astype(x.dtype)
        q, k = rotate(q), rotate(k)
        y = mx.fast.scaled_dot_product_attention(q.transpose(1, 0, 2)[None],
            k.transpose(1, 0, 2)[None], v.transpose(1, 0, 2)[None], scale=self.head_dim**-.5)
        return self.proj(y[0].transpose(1, 0, 2).reshape(value.shape))


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_fc1 = nn.Linear(config['hidden_size'], config['intermediate_size'])
        self.linear_fc2 = nn.Linear(config['intermediate_size'], config['hidden_size'])

    def __call__(self, x):
        return self.linear_fc2(nn.gelu_approx(self.linear_fc1(x)))


class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        self.norm2 = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        self.attn, self.mlp = VisionAttention(config), VisionMLP(config)

    def __call__(self, x, angles):
        x = x + self.attn(self.norm1(x), angles)
        return x + self.mlp(self.norm2(x))


class PatchMerger(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.width = config['hidden_size'] * config['spatial_merge_size']**2
        self.norm = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        self.linear_fc1 = nn.Linear(self.width, self.width)
        self.linear_fc2 = nn.Linear(self.width, config['out_hidden_size'])

    def __call__(self, x):
        return self.linear_fc2(nn.gelu(self.linear_fc1(self.norm(x).reshape(-1, self.width))))


class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if (config['hidden_act'] != 'gelu_pytorch_tanh' or config['spatial_merge_size'] != 2
                or config.get('deepstack_visual_indexes', []) or config['hidden_size'] % config['num_heads']
                or (config['hidden_size'] // config['num_heads']) % 4):
            raise ValueError('Unsupported Qwen vision architecture')
        self.head_dim = config['hidden_size'] // config['num_heads']
        self.side = math.isqrt(config['num_position_embeddings'])
        if self.side**2 != config['num_position_embeddings']:
            raise ValueError('Qwen vision position table must be square')
        self.patch_embed = PatchEmbed(config)
        self.pos_embed = nn.Embedding(config['num_position_embeddings'], config['hidden_size'])
        self.blocks = [VisionBlock(config) for _ in range(config['depth'])]
        self.merger = PatchMerger(config)

    def __call__(self, patches, grid):
        coords = mx.array(grid_coordinates(grid))
        if patches.shape[0] != coords.shape[0]:
            raise ValueError('Image patches do not match the Qwen vision grid')
        indices, weights = interpolation(grid, self.side)
        x = self.patch_embed(patches)
        positions = (self.pos_embed(indices).astype(mx.float32) * weights[:, :, None]).sum(axis=1)
        x = x + positions.astype(x.dtype)
        freq = 1 / (10000 ** (mx.arange(0, self.head_dim // 2, 2, dtype=mx.float32) / (self.head_dim // 2)))
        angles = (coords[..., None] * freq).reshape(-1, self.head_dim // 2)
        angles = mx.concatenate([angles, angles], axis=-1)
        for block in self.blocks:
            check_cancelled()
            x = block(x, angles)
            mx.eval(x)
        return self.merger(x)
