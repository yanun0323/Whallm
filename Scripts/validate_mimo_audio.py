"""Opt-in real-weight audio equations check (PyTorch CPU, not CUDA/full-model parity).

Reference environment needs torch, torchaudio, numpy and safetensors. Runtime
needs only its existing dependencies. Reviewed SGLang revision:
0c53fec4768a46a003a7afe27c938469db956368, models/mimo_audio.py and
multimodal/processors/mimo_audio.py. Output directories must not already exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def reference(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchaudio
    from safetensors import safe_open
    torch.set_num_threads(8)
    waveform = torch.from_numpy(np.load(args.output / "waveform.npy"))
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=960, win_length=960,
        hop_length=240, f_min=0, f_max=None, n_mels=128, power=1.0, center=True)
    mel = transform(waveform).clamp(min=1e-7).log().T
    np.save(args.output / "reference-mel.npy", mel.numpy())
    raw = safe_open(str(args.model / "checkpoint/audio_tokenizer/model.safetensors"), framework="pt", device="cpu")
    table = json.loads((args.model / "manifest.json").read_text())["components"]["audio_patch"]
    entries = {t["name"].removeprefix("audio_encoder."): t for t in table["tensors"]}
    mapped = np.memmap(args.model / table["file"], mode="r", dtype=np.uint8)
    def w(name):
        if name.startswith("encoder."):
            return raw.get_tensor(name).float()
        t = entries[name]
        data = np.ndarray(t["shape"], dtype=np.uint16, buffer=mapped, offset=t["offset"])
        return torch.from_numpy((data.astype(np.uint32) << 16).view(np.float32).copy())
    def linear(x, name, bias=False):
        return F.linear(x, w(name + ".weight"), w(name + ".bias") if bias else None)
    def norm(x, name, rms=False):
        if rms:
            return F.rms_norm(x, (x.shape[-1],), w(name + ".weight"), eps=1e-6)
        return F.layer_norm(x, (x.shape[-1],), w(name + ".weight"), w(name + ".bias"), eps=1e-5)
    def attention(x, prefix, theta, local=False, patch=False):
        q, k, v = [linear(x, prefix + "." + kind + "_proj", bias=(kind != "k" or patch))
                   .reshape(x.shape[0], x.shape[1], 16, 64).transpose(1, 2) for kind in ("q", "k", "v")]
        angles = torch.arange(x.shape[1]).float()[:, None] / (theta ** (torch.arange(0, 64, 2).float() / 64))
        angles = torch.cat((angles, angles), dim=-1)
        def rotate(a):
            first, second = a.chunk(2, dim=-1)
            return a * angles.cos() + torch.cat((-second, first), dim=-1) * angles.sin()
        scores = rotate(q) @ rotate(k).transpose(-1, -2) / 8
        if local:
            pos = torch.arange(x.shape[1])
            mask = (pos[None, :] <= pos[:, None]) & (pos[None, :] >= pos[:, None] - 128)
            scores = scores.masked_fill(~mask, float("-inf"))
        y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(x.shape)
        return linear(y, prefix + (".o_proj" if patch else ".out_proj"), bias=not patch)
    def conv(x, name, stride, padding, bias=True):
        return F.gelu(F.conv1d(x, w(name + ".weight"), w(name + ".bias") if bias else None, stride, padding))
    x = conv(mel.T[None], "encoder.conv1", 1, 1)
    x = conv(x, "encoder.conv2", 2, 1).transpose(1, 2)
    config = json.loads((args.model / "checkpoint/audio_tokenizer/config.json").read_text())
    skip = None
    for i in range(config["encoder_layers"]):
        p = f"encoder.layers.{i}"
        x = x + attention(norm(x, p + ".self_attn_layer_norm"), p + ".self_attn", 10000, local=(i % 2 == 0))
        x = x + linear(F.gelu(linear(norm(x, p + ".final_layer_norm"), p + ".fc1", True)), p + ".fc2", True)
        if i + 1 == config["encoder_skip_layer_id"]:
            skip = x.clone()
    x = norm(x if skip is None else x + skip, "encoder.layer_norm")
    if x.shape[1] % 2:
        x = F.pad(x, (0, 0, 0, 1))
    x = norm(conv(x.transpose(1, 2), "encoder.down_sample_layer.0", 2, 0, False).transpose(1, 2), "encoder.down_sample_norm")[0]
    np.save(args.output / "reference-features.npy", x.numpy())
    residual, codes = x, []
    for i in range(20):
        book = w(f"encoder.quantizer.vq.layers.{i}._codebook.embed")
        distance = residual.square().sum(-1, keepdim=True) - 2 * residual @ book.T + book.square().sum(-1)
        index = distance.argmin(-1)
        residual = residual - book[index]
        codes.append(index)
    codes = torch.stack(codes, dim=-1)
    np.save(args.output / "reference-codes.npy", codes.numpy())
    if codes.shape[0] % 4:
        codes = torch.cat((codes, codes[-1:].expand(4 - codes.shape[0] % 4, -1)))
    codes = codes.reshape(-1, 4, 20)
    x = torch.zeros(codes.shape[0], 4, 1024)
    for i in range(20):
        x += F.embedding(codes[:, :, i], w(f"speech_embeddings.{i}.weight"))
    for i in range(6):
        p = f"input_local_transformer.layers.{i}"
        x = x + attention(norm(x, p + ".input_layernorm", True), p + ".self_attn", 640000, patch=True)
        n = norm(x, p + ".post_attention_layernorm", True)
        x = x + linear(F.silu(linear(n, p + ".mlp.gate_proj")) * linear(n, p + ".mlp.up_proj"), p + ".mlp.down_proj")
    x = norm(x, "input_local_transformer.norm", True).reshape(x.shape[0], -1)
    x = linear(F.gelu(linear(x, "projection.mlp.0")), "projection.mlp.2")
    np.save(args.output / "reference-patch.npy", x.numpy())
    (args.output / "reference-environment.json").write_text(json.dumps({"torch": torch.__version__, "torchaudio": torchaudio.__version__}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-python", type=Path)
    p.add_argument("--reference", action="store_true")
    args = p.parse_args()
    if args.reference:
        reference(args)
        return
    import numpy as np
    import mlx.core as mx
    from deepseek_v4_ssd.mimo.audio_processor import log_mel
    from deepseek_v4_ssd.mimo.audio import load_audio
    from deepseek_v4_ssd.mimo.install import file_sha, atomic_json
    args.output.mkdir(parents=True, exist_ok=False)
    # Noise + swept tones exercise all mel bands and both sides of the local
    # attention window. Silence separately checks the exact clamp floor.
    rng = np.random.default_rng(260922)
    t = np.arange(3 * 24000 + 73, dtype=np.float32) / 24000
    waveform = (rng.normal(0, .04, len(t)) + .2 * np.sin(2 * np.pi * (150 * t + 500 * t**2))).astype(np.float32)
    np.save(args.output / "waveform.npy", waveform)
    subprocess.run([str(args.reference_python), __file__, "--reference", "--model", str(args.model), "--output", str(args.output)], check=True)
    mel = log_mel(waveform)
    reference_mel = np.load(args.output / "reference-mel.npy")
    np.testing.assert_allclose(mel, reference_mel, rtol=2e-4, atol=2e-4)
    reference_features = np.load(args.output / "reference-features.npy")
    reference_codes = np.load(args.output / "reference-codes.npy")
    reference_patch = np.load(args.output / "reference-patch.npy")
    def metrics(a, b):
        return {"relativeL2": float(np.linalg.norm(a-b) / np.linalg.norm(b)), "maxAbs": float(np.max(np.abs(a-b))),
                "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
    report = {"scope": "independent CPU PyTorch F32 equations, not CUDA or full-model parity",
        "sourceRevision": "0c53fec4768a46a003a7afe27c938469db956368",
        "referenceSources": {"models/mimo_audio.py": "18ec63a91945749afcc0e7fc193bec3b0aa1481c0520c4709ef03d3ffb130753",
                             "multimodal/processors/mimo_audio.py": "a0d3ab2aca95a50172990d17b361750434e57cf42396c6b2c6c9d0f34ab83463"},
        "melMaxAbs": float(np.max(np.abs(mel-reference_mel))),
        "manifestSHA256": file_sha(args.model / "manifest.json"), "waveformSHA256": file_sha(args.output / "waveform.npy"), "results": {}}
    for mode, dtype in (("F32", mx.float32), ("BF16", mx.bfloat16)):
        tokenizer, patch = load_audio(args.model, json.loads((args.model / "manifest.json").read_text()),
                                       json.loads((args.model / "config.json").read_text()), dtype=dtype)
        features = tokenizer.features(mx.array(mel))
        codes = np.asarray(tokenizer.quantize(features))
        # Isolate patch equations from discontinuous RVQ decisions.
        patches = np.asarray(patch(mx.array(reference_codes)).astype(mx.float32))
        features = np.asarray(features.astype(mx.float32))
        result = {"features": metrics(features, reference_features), "patch": metrics(patches, reference_patch),
                  "codeAgreement": float(np.mean(codes == reference_codes)), "codesSHA256": hashlib.sha256(codes.tobytes()).hexdigest()}
        report["results"][mode] = result
        if mode == "F32":
            np.testing.assert_allclose(features, reference_features, atol=2e-3, rtol=3e-3)
            np.testing.assert_allclose(patches, reference_patch, atol=2e-3, rtol=3e-3)
            if result["codeAgreement"] != 1:
                raise ValueError("F32 RVQ reference codes differ")
        if not np.isfinite(features).all() or not np.isfinite(patches).all():
            raise ValueError("non-finite audio output")
        if mode == "BF16":
            boundary_mel = log_mel(np.zeros(720000, np.float32))
            boundary_codes = tokenizer(mx.array(boundary_mel))
            boundary_patch = patch(boundary_codes)
            mx.eval(boundary_patch)
            if boundary_codes.shape != (751, 20) or boundary_patch.shape != (188, 4096) or not mx.all(mx.isfinite(boundary_patch)).item():
                raise ValueError("30-second audio execution boundary failed")
            report["maximumDurationExecution"] = {"seconds": 30, "fixture": "silence, not a quality/reference check",
                "codesShape": boundary_codes.shape, "patchShape": boundary_patch.shape,
                "patchSHA256": hashlib.sha256(np.asarray(boundary_patch.astype(mx.float32)).tobytes()).hexdigest()}
            del boundary_codes, boundary_patch
        del tokenizer, patch
        mx.clear_cache()
    atomic_json(args.output / "validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
