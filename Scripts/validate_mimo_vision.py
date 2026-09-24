"""Opt-in installed MiMo vision check against pinned processor + PyTorch equations.

Does not execute a remote module, require CUDA, or establish full-model parity.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

PROCESSOR_SHA = "713a26ad6080bcd4789c6e6ab94a78d0fb22de02acac283bd7bcda87612ec358"


def reference(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    torch.set_num_threads(8)
    source = args.processor_source.read_bytes()
    if hashlib.sha256(source).hexdigest() != PROCESSOR_SHA:
        raise ValueError("unreviewed processor reference")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiMoProcessor")
    methods = {"smart_resize", "standardize_batch", "get_visual_transform", "_flatten_visual_inputs"}
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    cls.bases, cls.decorator_list = [], []
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes += [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in {"_QWEN2VL_PIXEL_MEAN", "_QWEN2VL_PIXEL_STD"} for t in n.targets)]
    nodes.append(cls)
    namespace = {"torch": torch, "np": np, "Image": Image, "F": F, "math": math, "_mean_std_cache": {}}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "reviewed-processor", "exec"), namespace)
    processor = namespace["MiMoProcessor"]()
    processor.patch_size, processor.merge_size, processor.temporal_patch_size = 16, 2, 2
    pixels, _, _ = processor.get_visual_transform(Image.open(args.output / "fixture.png"), 32, 8192, 8388608)
    patches, grid = processor._flatten_visual_inputs(pixels, "image")
    np.save(args.output / "reference-patches.npy", patches.numpy())
    manifest = json.loads((args.model / "manifest.json").read_text())
    table = {t["name"].removeprefix("visual."): t for t in manifest["components"]["vision"]["tensors"]}
    mapped = np.memmap(args.model / "vision/common.bin", dtype=np.uint8, mode="r")
    def w(name):
        t = table[name]
        dtype = np.uint16 if t["dtype"] == "BF16" else np.float32
        data = np.ndarray(t["shape"], dtype=dtype, buffer=mapped, offset=t["offset"])
        if dtype == np.uint16:
            data = (data.astype(np.uint32) << 16).view(np.float32)
        return torch.from_numpy(data.copy())
    def linear(x, name):
        bias = w(name + ".bias") if name + ".bias" in table else None
        return F.linear(x, w(name + ".weight"), bias)
    def norm(x, name):
        return F.rms_norm(x, (x.shape[-1],), w(name + ".weight"), eps=1e-6)
    config = json.loads((args.model / "config.json").read_text())["vision_config"]
    _, h, width = grid.tolist()
    coords = torch.tensor([(yy, xx) for by in range(0, h, 2) for bx in range(0, width, 2)
                           for yy in (by, by + 1) for xx in (bx, bx + 1)])
    frequency = 1 / (10000 ** (torch.arange(0, 32, 2).float() / 32))
    angle = (coords[..., None] * frequency).flatten(1)
    angle = torch.cat((angle, angle), dim=-1)
    columns = torch.arange(h * width // 4).reshape(h // 2, width // 2).T.flatten()
    def reorder(x, index):
        return x.reshape(-1, 4, x.shape[-1])[index].reshape(x.shape)
    col_angle = reorder(angle, columns)
    reverse = torch.argsort(columns)
    x = F.linear(patches, w("patch_embed.proj.weight").flatten(1))
    column_order = False
    for i, order in enumerate(config["vit_window_attn_types"]):
        if (order == 1) != column_order:
            x = reorder(x, columns if order == 1 else reverse)
            column_order = order == 1
        angles = col_angle if column_order else angle
        p = f"blocks.{i}"
        q, k, v = torch.split(linear(norm(x, p + ".norm1"), p + ".attn.qkv"), [2048, 512, 512], dim=-1)
        def rotate(a, heads):
            a = a.reshape(-1, heads, 64)
            first, second = a.chunk(2, dim=-1)
            return (a * angles.cos()[:, None] + torch.cat((-second, first), dim=-1) * angles.sin()[:, None]).transpose(0, 1)
        q, k = rotate(q, 32), rotate(k, 8).repeat_interleave(4, 0)
        v = v.reshape(-1, 8, 64).transpose(0, 1).repeat_interleave(4, 0)
        scores = q @ k.transpose(-1, -2) / 8
        local = i not in config["fullatt_block_indexes"]
        if local:
            indices = torch.arange(len(x))
            scores = scores.masked_fill((indices[:, None] - indices[None, :]).abs() > 64, float("-inf"))
            sink = w(p + ".attn.sinks")[:, None, None].expand(-1, len(x), 1)
            scores = torch.cat((scores, sink), dim=-1)
        probs = scores.softmax(dim=-1)
        if local:
            probs = probs[..., :-1]
        attention = (probs @ v).transpose(0, 1).reshape(len(x), -1)
        x = x + linear(attention, p + ".attn.proj")
        normalized = norm(x, p + ".norm2")
        x = x + linear(F.silu(linear(normalized, p + ".mlp.gate_proj")) *
                       linear(normalized, p + ".mlp.up_proj"), p + ".mlp.down_proj")
    if column_order:
        x = reorder(x, reverse)
    x = norm(x, "merger.ln_q").reshape(-1, 5120)
    output = linear(F.gelu(linear(x, "merger.mlp.0")), "merger.mlp.2")
    np.save(args.output / "reference-features.npy", output.numpy())
    print("independent PyTorch F32 vision reference complete", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--processor-source", type=Path, required=True)
    p.add_argument("--reference-python", type=Path)
    p.add_argument("--reference", action="store_true")
    args = p.parse_args()
    if args.reference:
        reference(args)
        return
    import numpy as np
    import mlx.core as mx
    from PIL import Image
    from deepseek_v4_ssd.media import ImagePart
    from deepseek_v4_ssd.mimo.processor import image_input
    from deepseek_v4_ssd.mimo.vision import VisionEncoder
    from deepseek_v4_ssd.mimo.install import atomic_json, file_sha
    from deepseek_v4_ssd.manifest import Tensor
    from deepseek_v4_ssd.model import _load_tensor_file
    args.output.mkdir(parents=True, exist_ok=True)
    y, x = np.indices((197, 173))
    image = np.stack(((x * 7 + y) % 256, (y * 3) % 256, (x ^ y) % 256), axis=-1).astype(np.uint8)
    Image.fromarray(image).save(args.output / "fixture.png")
    inp = image_input(ImagePart.from_bytes((args.output / "fixture.png").read_bytes(), "image/png"))
    subprocess.run([str(args.reference_python), __file__, "--reference", "--model", str(args.model),
                    "--output", str(args.output), "--processor-source", str(args.processor_source)], check=True)
    patch_ref = np.load(args.output / "reference-patches.npy")
    np.testing.assert_allclose(inp.patches, patch_ref, rtol=2e-5, atol=3e-5)
    manifest = json.loads((args.model / "manifest.json").read_text())
    config = json.loads((args.model / "config.json").read_text())["vision_config"]
    raw = manifest["components"]["vision"]
    tensors = tuple(Tensor(t["name"], t["dtype"], tuple(t["shape"]), t["offset"], t["length"]) for t in raw["tensors"])
    weights = _load_tensor_file(args.model / raw["file"], tensors)
    reference_features = np.load(args.output / "reference-features.npy")
    report = {"scope": "pinned processor and independent PyTorch F32 encoder equations; not CUDA/full-model parity",
              "processorSourceSHA256": PROCESSOR_SHA, "fixtureSHA256": file_sha(args.output / "fixture.png"),
              "manifestSHA256": file_sha(args.model / "manifest.json"), "grid": inp.grid,
              "patchMaxAbsError": float(np.max(np.abs(inp.patches - patch_ref))), "results": {}}
    for mode in ("F32", "BF16"):
        encoder = VisionEncoder(config)
        encoder.load_weights([(k.removeprefix("visual."), v.astype(mx.float32) if mode == "F32" else v)
                               for k, v in weights.items()], strict=True)
        result = np.asarray(encoder(mx.array(inp.patches), inp.grid).astype(mx.float32))
        if mode == "F32":
            np.testing.assert_allclose(result, reference_features, rtol=3e-3, atol=2e-3)
        relative = float(np.linalg.norm(result - reference_features) / np.linalg.norm(reference_features))
        if not np.isfinite(result).all() or relative > (0.001 if mode == "F32" else 0.08):
            raise ValueError(f"{mode} vision reference tolerance exceeded: {relative}")
        report["results"][mode] = {"relativeL2": relative, "maxAbs": float(np.max(np.abs(result-reference_features))),
                                   "outputSHA256": hashlib.sha256(result.tobytes()).hexdigest()}
        del encoder
        mx.clear_cache()
    atomic_json(args.output / "validation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
