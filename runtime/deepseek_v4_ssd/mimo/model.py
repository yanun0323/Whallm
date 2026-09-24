"""MiMo target backbone for the complete, losslessly preserved installed artifact.

Source equations: XiaomiMiMo/MiMo-V2.6-Flash-RL at
3b38d063180c3e4aed9691fdc735f3d10b266ee4. Checkpoint QKV must first be
converted with weights.decode_qkv. No MTP, DFlash or approximate routing.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.cache import KVCache, RotatingKVCache

from ..cancellation import check_cancelled
from ..model import _StreamingSwitchGLU


@dataclass(frozen=True)
class MiMoArgs:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    swa_num_attention_heads: int
    swa_num_key_value_heads: int
    head_dim: int
    v_head_dim: int
    swa_head_dim: int
    swa_v_head_dim: int
    partial_rotary_factor: float
    rope_theta: float
    swa_rope_theta: float
    sliding_window_size: int
    hybrid_layer_pattern: tuple[int, ...]
    moe_layer_freq: tuple[int, ...]
    n_routed_experts: int
    num_experts_per_tok: int
    layernorm_epsilon: float
    vocab_size: int
    attention_value_scale: float = 0.707

    def __post_init__(self):
        if (len(self.hybrid_layer_pattern) != self.num_hidden_layers
                or len(self.moe_layer_freq) != self.num_hidden_layers
                or set(self.hybrid_layer_pattern) != {0, 1}
                or self.moe_layer_freq != (0,) + (1,) * (self.num_hidden_layers - 1)
                or self.sliding_window_size < 1
                or not 1 <= self.num_experts_per_tok <= self.n_routed_experts):
            raise ValueError("invalid MiMo hybrid/MoE layer contract")
        for heads, kv, dim in ((self.num_attention_heads, self.num_key_value_heads, self.head_dim),
                                (self.swa_num_attention_heads, self.swa_num_key_value_heads, self.swa_head_dim)):
            rotary_dim = int(dim * self.partial_rotary_factor)
            if heads < 1 or kv < 1 or heads % kv or not 0 < rotary_dim <= dim or rotary_dim % 2:
                raise ValueError("invalid MiMo heads or partial rotary dimension")

    @classmethod
    def from_config(cls, config):
        # Reject unsupported equations instead of silently ignoring config flags.
        fixed = {"attention_projection_layout": "fused_qkv", "attention_bias": False,
                 "add_full_attention_sink_bias": False, "add_swa_attention_sink_bias": True,
                 "scoring_func": "sigmoid", "topk_method": "noaux_tc", "n_group": 1,
                 "topk_group": 1, "norm_topk_prob": True, "n_shared_experts": None,
                 "routed_scaling_factor": None, "hidden_act": "silu",
                 "moe_router_dtype": "bfloat16", "tie_word_embeddings": False}
        for name, expected in fixed.items():
            if name not in config or config[name] != expected:
                raise ValueError(f"unsupported MiMo equation setting: {name}")
        values = {name: config[name] for name in cls.__dataclass_fields__}
        for key in ("hybrid_layer_pattern", "moe_layer_freq"):
            values[key] = tuple(values[key])
        return cls(**values)


class Attention(nn.Module):
    def __init__(self, args: MiMoArgs, sliding: bool):
        super().__init__()
        self.heads = args.swa_num_attention_heads if sliding else args.num_attention_heads
        self.kv_heads = args.swa_num_key_value_heads if sliding else args.num_key_value_heads
        self.head_dim = args.swa_head_dim if sliding else args.head_dim
        self.value_dim = args.swa_v_head_dim if sliding else args.v_head_dim
        self.value_scale = args.attention_value_scale
        self.q_size = self.heads * self.head_dim
        self.k_size = self.kv_heads * self.head_dim
        self.v_size = self.kv_heads * self.value_dim
        self.qkv_proj = nn.Linear(args.hidden_size, self.q_size + self.k_size + self.v_size, bias=False)
        self.o_proj = nn.Linear(self.heads * self.value_dim, args.hidden_size, bias=False)
        self.attention_sink_bias = mx.zeros((self.heads,), mx.float32) if sliding else None
        self.rope = nn.RoPE(int(self.head_dim * args.partial_rotary_factor), traditional=False,
                            base=args.swa_rope_theta if sliding else args.rope_theta)

    def __call__(self, x, mask, cache, *, cache_only=False):
        if cache_only and cache is None:
            raise ValueError("MiMo cache-only attention requires a KV cache")
        batch, length, _ = x.shape
        q, k, v = mx.split(self.qkv_proj(x), [self.q_size, self.q_size + self.k_size], axis=-1)
        q = q.reshape(batch, length, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, length, self.kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, length, self.kv_heads, self.value_dim).transpose(0, 2, 1, 3)
        v = v * self.value_scale
        offset = cache.offset if cache is not None else 0
        q, k = self.rope(q, offset=offset), self.rope(k, offset=offset)
        if cache is not None:
            k, v = cache.update_and_fetch(k, v)
        if cache_only:
            # The last Prefill layer needs only K/V for future tokens. Fused
            # QKV and K-RoPE still run; SDPA and the output projection do not.
            return None
        # The artifact preserves sinks as F32; MLX requires the SDPA sink
        # operand to match the attention output dtype (BF16 in real weights).
        sinks = self.attention_sink_bias
        if sinks is not None:
            sinks = sinks.astype(q.dtype)
        y = scaled_dot_product_attention(q, k, v, cache=cache, scale=self.head_dim**-0.5,
                                          mask=mask, sinks=sinks)
        return self.o_proj(y.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class DenseMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gate_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Router(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight = mx.zeros((args.n_routed_experts, args.hidden_size), mx.bfloat16)
        self.e_score_correction_bias = mx.zeros((args.n_routed_experts,), mx.float32)
        self.top_k = args.num_experts_per_tok

    def __call__(self, x):
        # SGLang CUDA uses BF16 operands with FP32 output. FP32 matmul here
        # preserves accumulation instead of rounding logits to BF16 first.
        inputs = x.astype(mx.bfloat16).astype(mx.float32)
        logits = inputs @ self.weight.astype(mx.bfloat16).astype(mx.float32).T
        scores = mx.sigmoid(logits)
        corrected = scores + self.e_score_correction_bias.astype(mx.float32)
        indices = mx.argpartition(-corrected, kth=self.top_k - 1, axis=-1)[..., :self.top_k]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        if self.top_k > 1:
            weights = weights / (mx.sum(weights, axis=-1, keepdims=True) + 1e-20)
        return indices, weights


class StreamingMoE(nn.Module):
    def __init__(self, args, backbone_layer, expert_cache):
        super().__init__()
        self.gate = Router(args)
        # ExpertCache addresses 47 expert-bearing layers (0..46), while model
        # state addresses 48 backbone layers (0..47). The dense block has none.
        self.switch_mlp = _StreamingSwitchGLU(backbone_layer - 1, expert_cache,
                                            activation=lambda up, gate: nn.silu(gate) * up)

    def __call__(self, x):
        indices, scores = self.gate(x)
        routed = self.switch_mlp(x, indices)
        return mx.sum(routed.astype(mx.float32) * scores[..., None], axis=-2).astype(x.dtype)


class DecoderLayer(nn.Module):
    def __init__(self, args, index, expert_cache):
        super().__init__()
        self.sliding = args.hybrid_layer_pattern[index] == 1
        self.self_attn = Attention(args, self.sliding)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        self.mlp = StreamingMoE(args, index, expert_cache) if args.moe_layer_freq[index] else DenseMLP(args)

    def __call__(self, x, mask, cache, *, cache_only=False):
        check_cancelled()
        attention = self.self_attn(self.input_layernorm(x), mask, cache, cache_only=cache_only)
        if cache_only:
            return None
        x = x + attention
        return x + self.mlp(self.post_attention_layernorm(x))


class Backbone(nn.Module):
    def __init__(self, args, expert_cache):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i, expert_cache) for i in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layernorm_epsilon)
        self.swa_index = args.hybrid_layer_pattern.index(1)
        self.ga_index = args.hybrid_layer_pattern.index(0)
        self.window = args.sliding_window_size

    def __call__(self, tokens, cache, *, input_embeddings=None):
        # Strict ownership: callers prepare and validate multimodal spans before
        # entering the model; the model only accepts the resulting embeddings.
        if tokens.ndim != 2:
            raise ValueError("MiMo expects a batch of token IDs")
        if input_embeddings is not None and input_embeddings.shape != (*tokens.shape, self.embed_tokens.weight.shape[1]):
            raise ValueError("MiMo embeddings must align with every prompt token")
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("MiMo cache layer count mismatch")
        full_mask = create_attention_mask(tokens, cache[self.ga_index])
        local_mask = create_attention_mask(tokens, cache[self.swa_index], window_size=self.window)
        hidden = self.embed_tokens(tokens) if input_embeddings is None else input_embeddings
        for layer, state in zip(self.layers, cache):
            hidden = layer(hidden, local_mask if layer.sliding else full_mask, state)
        return self.norm(hidden)


class Model(nn.Module):
    def __init__(self, args: MiMoArgs, expert_cache):
        super().__init__()
        self.args = args
        self.model = Backbone(args, expert_cache)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [RotatingKVCache(max_size=self.args.sliding_window_size) if layer.sliding else KVCache()
                for layer in self.layers]

    def __call__(self, tokens, cache=None, *, input_embeddings=None):
        return self.lm_head(self.model(tokens, cache, input_embeddings=input_embeddings))
