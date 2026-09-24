"""Native MiMo audio input: AudioTokenizer encoder/RVQ and audio patch encoder.

Equations follow SGLang 0c53fec4768a46a003a7afe27c938469db956368's
mimo_audio.py, including alternating local/global tokenizer attention. No
vocoder, audio generation, pretrained ASR, or external inference service.
"""
from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen2 import ModelArgs, TransformerBlock

from ..cancellation import check_cancelled
from ..manifest import Tensor
from ..model import _load_tensor_file
from .install import header


class TokenizerAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["d_model"]
        self.heads = config["encoder_attention_heads"]
        self.head_dim = dim // self.heads
        self.theta = config["rope_theta"]
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def __call__(self, x, mask):
        def heads(a):
            return a.reshape(1, -1, self.heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k, v = (heads(layer(x)) for layer in (self.q_proj, self.k_proj, self.v_proj))
        frequencies = self.theta ** (-mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim)
        angles = mx.arange(x.shape[1], dtype=mx.float32)[:, None] * frequencies
        angles = mx.concatenate((angles, angles), axis=-1)
        cos, sin = mx.cos(angles).astype(q.dtype), mx.sin(angles).astype(q.dtype)
        def rotate(a):
            first, second = mx.split(a, 2, axis=-1)
            return a * cos + mx.concatenate((-second, first), axis=-1) * sin
        y = mx.fast.scaled_dot_product_attention(rotate(q), rotate(k), v, scale=self.head_dim**-.5, mask=mask)
        return self.out_proj(y.transpose(0, 2, 1, 3).reshape(x.shape))


class TokenizerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["d_model"]
        self.self_attn = TokenizerAttention(config)
        self.self_attn_layer_norm = nn.LayerNorm(dim, eps=1e-5)
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = nn.Linear(dim, config["encoder_ffn_dim"])
        self.fc2 = nn.Linear(config["encoder_ffn_dim"], dim)

    def __call__(self, x, mask):
        x = x + self.self_attn(self.self_attn_layer_norm(x), mask)
        return x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))


class AudioTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        required = {"kernel_size": 3, "stride_size": 2, "avg_pooler": 2,
                    "activation_function": "gelu", "ln_type": "LayerNorm",
                    "position_embedding_type": "rope", "rope_type": "default",
                    "hybrid_attention": True, "swa_per_block": 2,
                    "encoder_attn_window_size": [128, 0]}
        if any(config.get(k) != v for k, v in required.items()):
            raise ValueError("unsupported MiMo audio tokenizer equations")
        dim = config["d_model"]
        self.conv1 = nn.Conv1d(config["n_mels"], dim, 3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, 3, stride=2, padding=1)
        self.layers = [TokenizerBlock(config) for _ in range(config["encoder_layers"])]
        self.skip_layer = config["encoder_skip_layer_id"]
        if self.skip_layer is not None and not 1 <= self.skip_layer <= len(self.layers):
            raise ValueError("invalid audio tokenizer skip layer")
        self.layer_norm = nn.LayerNorm(dim, eps=1e-5)
        self.down_sample_layer = [nn.Conv1d(dim, dim, 2, stride=2, bias=False)]
        self.down_sample_norm = nn.LayerNorm(dim, eps=1e-5)
        sizes = config["codebook_size"]
        self.codebooks = [mx.zeros((sizes[min(i, len(sizes) - 1)], dim)) for i in range(config["num_quantizers"])]

    def features(self, mel):
        if mel.ndim != 2 or not 1 <= mel.shape[0] <= 3001 or mel.shape[1] != self.conv1.weight.shape[-1]:
            raise ValueError("invalid or oversized MiMo mel input")
        x = nn.gelu(self.conv2(nn.gelu(self.conv1(mel[None].astype(self.conv1.weight.dtype)))))
        indices = mx.arange(x.shape[1])
        local_mask = (indices[None, :] <= indices[:, None]) & (indices[None, :] >= indices[:, None] - 128)
        skip = None
        for i, layer in enumerate(self.layers):
            check_cancelled()
            # Pinned SGLang ignores encoder_causal on global layers; its actual
            # VisionAttention has no causal flag. Do not copy the HF sketch mask.
            x = layer(x, local_mask if i % 2 == 0 else None)
            if i + 1 == self.skip_layer:
                skip = x
            mx.eval(x)
        x = self.layer_norm(x if skip is None else x + skip)
        if x.shape[1] % 2:
            x = mx.pad(x, ((0, 0), (0, 1), (0, 0)))
        return self.down_sample_norm(nn.gelu(self.down_sample_layer[0](x)))[0]

    def quantize(self, features):
        residual, codes = features.astype(mx.float32), []
        for book in self.codebooks:
            check_cancelled()
            # Euclidean nearest code in F32. Keep residual norm as in reference.
            distances = (mx.sum(residual**2, axis=-1, keepdims=True)
                         - 2 * (residual @ book.T) + mx.sum(book**2, axis=-1))
            indices = mx.argmin(distances, axis=-1)
            residual = residual - book[indices]
            codes.append(indices)
            mx.eval(residual, indices)
        return mx.stack(codes, axis=-1)

    def __call__(self, mel):
        return self.quantize(self.features(mel))


class LocalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        args = ModelArgs(model_type="qwen2", hidden_size=config["input_local_dim"],
                         num_hidden_layers=config["input_local_layers"],
                         intermediate_size=config["input_local_intermediate_size"],
                         num_attention_heads=config["input_local_attn_heads"],
                         num_key_value_heads=config["input_local_attn_heads"],
                         rms_norm_eps=1e-6, vocab_size=1, rope_theta=config["rope_theta"])
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x):
        for layer in self.layers:
            check_cancelled()
            x = layer(x, mask=None)  # Full attention within each four-code group.
            mx.eval(x)
        return self.norm(x)


class Projection(nn.Module):
    def __init__(self, width, output):
        super().__init__()
        self.mlp = [nn.Linear(width, width * 4, bias=False), nn.GELU(), nn.Linear(width * 4, output, bias=False)]

    def __call__(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


class AudioPatchEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if (not config["input_full_attention"] or not config["add_post_norm"] or config["projection_layers"] != 2
                or config["partial_rotary_factor"] != 1.0
                or config["input_local_head_dim"] * config["input_local_attn_heads"] != config["input_local_dim"]):
            raise ValueError("unsupported MiMo audio patch equations")
        self.channels, self.group = config["audio_channels"], config["group_size"]
        self.speech_embeddings = [nn.Embedding(int(config["speech_vocab_size"]), config["input_local_dim"]) for _ in range(self.channels)]
        self.input_local_transformer = LocalTransformer(config)
        self.projection = Projection(config["input_local_dim"] * self.group, config["out_hidden_size"])

    def __call__(self, codes):
        if codes.ndim != 2 or not 1 <= codes.shape[0] <= 751 or codes.shape[1] != self.channels:
            raise ValueError("invalid MiMo audio codes")
        padding = (-codes.shape[0]) % self.group
        if padding:
            codes = mx.concatenate((codes, mx.repeat(codes[-1:], padding, axis=0)))
        codes = codes.reshape(-1, self.group, self.channels)
        x = mx.zeros((*codes.shape[:2], self.speech_embeddings[0].weight.shape[1]), self.speech_embeddings[0].weight.dtype)
        for i, embedding in enumerate(self.speech_embeddings):
            x = x + embedding(codes[:, :, i])
        x = self.input_local_transformer(x)
        return self.projection(x.reshape(x.shape[0], -1))


def load_audio(root, manifest, config, *, dtype=mx.bfloat16):
    """Load only encoder inference tensors, not the preserved decoder/training state.

    Runtime SGLang casts the whole tokenizer (including F32 codebooks) to BF16,
    then restores the rounded books to F32 for RVQ. Reproduce that ordering.
    dtype=F32 is reserved for independent equation validation.
    """
    tokenizer_config = json.loads((root / "checkpoint/audio_tokenizer/config.json").read_text())
    tokenizer = AudioTokenizer(tokenizer_config)
    path = root / "checkpoint/audio_tokenizer/model.safetensors"
    prefix, metadata = header(path)
    start = len(prefix)
    tensors, names = [], {}
    ignored = {f"encoder.quantizer.vq.layers.{i}._codebook.{stat}"
               for i in range(tokenizer_config["num_quantizers"]) for stat in ("inited", "cluster_size", "embed_avg")}
    for name, entry in metadata.items():
        if not name.startswith("encoder.") or name in ignored:
            continue
        target = name.removeprefix("encoder.")
        if target.startswith("quantizer.vq.layers."):
            if not target.endswith("._codebook.embed"):
                raise ValueError(f"unexpected audio tokenizer tensor: {name}")
            target = "codebooks." + target.split(".")[3]
        names[name] = target
        lo, hi = entry["data_offsets"]
        tensors.append(Tensor(name, entry["dtype"], tuple(entry["shape"]), start + lo, hi - lo))
    weights = _load_tensor_file(path, tuple(tensors))
    converted = []
    for name, value in weights.items():
        target = names[name]
        if target in ("conv1.weight", "conv2.weight", "down_sample_layer.0.weight"):
            value = value.transpose(0, 2, 1)
        value = value.astype(dtype)
        if target.startswith("codebooks."):
            value = value.astype(mx.float32)
        converted.append((target, value))
    tokenizer.load_weights(converted, strict=True)
    patch = AudioPatchEncoder(config["audio_config"])
    table = manifest["components"]["audio_patch"]
    tensors = tuple(Tensor(t["name"], t["dtype"], tuple(t["shape"]), t["offset"], t["length"]) for t in table["tensors"])
    weights = _load_tensor_file(root / table["file"], tensors)
    patch.load_weights([(name.removeprefix("audio_encoder."), value.astype(dtype)) for name, value in weights.items()], strict=True)
    return tokenizer, patch
