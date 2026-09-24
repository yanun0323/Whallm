"""Validate real checkpoint slices against native MLX and a pinned CPU reference.

This fetches one expert and one GA QKV tensor (about 66 MiB), not the full
checkpoint. Reference PyTorch runs in a separate interpreter; it is not an App
dependency. No safetensors pickle files or arbitrary remote modules are loaded.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Dict
import urllib.request

MODEL_ID = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
REVISION = "3b38d063180c3e4aed9691fdc735f3d10b266ee4"
SGLANG_REVISION = "0c53fec4768a46a003a7afe27c938469db956368"
SGLANG_SHA256 = "a3dd3c7e9d59345c1e435cf43bf569ed1e5dd0ce75a588881283399de8e79656"
MAX_SLICE_BYTES = 64 * 1024**2


def checked_range(url, start, length, *, opener=urllib.request.urlopen):
    if type(start) is not int or type(length) is not int or start < 0 or not 0 < length <= MAX_SLICE_BYTES:
        raise ValueError("invalid checkpoint slice bounds")
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}",
                                                   "Accept-Encoding": "identity"})
    with opener(request, timeout=90) as response:
        if response.status != 206:
            raise ValueError("server ignored Range; refusing a full-shard download")
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        if (not match or (int(match[1]), int(match[2])) != (start, start + length - 1)
                or int(match[3]) < start + length):
            raise ValueError("server returned an unexpected checkpoint byte range")
        if response.headers.get("Content-Encoding", "identity") != "identity":
            raise ValueError("encoded checkpoint range is not supported")
        data = response.read(length + 1)
        if len(data) != length:
            raise ValueError("checkpoint byte range was truncated or oversized")
        return data


def collect(audit_dir: Path, root: Path):
    # Check the audited sources before deriving any byte offsets from them.
    from Scripts.audit_mimo_checkpoint import audit
    report = audit(audit_dir)
    repo = json.loads((audit_dir / "repo.json").read_text())
    index = json.loads((audit_dir / "source/model.safetensors.index.json").read_text())
    names = [f"model.layers.1.mlp.experts.0.{projection}.{suffix}"
             for projection in ("gate_proj", "up_proj", "down_proj")
             for suffix in ("weight", "weight_scale")]
    names += [f"model.layers.0.self_attn.qkv_proj.{suffix}" for suffix in ("weight", "weight_scale_inv")]
    records = []
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        shard = index["weight_map"][name]
        header = json.loads((audit_dir / "headers" / (shard.replace("/", "__") + ".json")).read_text())
        tensors = header["tensors"]
        data_start = repo["files"][shard]["size"] - max(t["data_offsets"][1] for t in tensors.values())
        tensor = tensors[name]
        start, end = tensor["data_offsets"]
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{shard}"
        data = checked_range(url, data_start + start, end - start)
        filename = name + ".bin"
        (root / filename).write_bytes(data)
        records.append({"name": name, "file": filename, "source": shard,
                        "source_offset": data_start + start, "bytes": len(data),
                        "dtype": tensor["dtype"], "shape": tensor["shape"],
                        "sha256": hashlib.sha256(data).hexdigest()})
        print(f"slice {name}: {len(data)} bytes", flush=True)
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "tensors": records,
                "audit_sources_sha256": report["sources_sha256"]}
    (root / "slices.json").write_text(json.dumps(manifest, indent=2) + "\n")


def read_slices(root):
    import numpy as np
    manifest = json.loads((root / "slices.json").read_text())
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION:
        raise ValueError("wrong checkpoint slice identity")
    result = {}
    for tensor in manifest["tensors"]:
        path = (root / tensor["file"]).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("unsafe slice path")
        data = path.read_bytes()
        if len(data) != tensor["bytes"] or hashlib.sha256(data).hexdigest() != tensor["sha256"]:
            raise ValueError("checkpoint slice integrity mismatch")
        dtype = {"U8": np.uint8, "F8_E4M3": np.uint8, "F32": np.float32}[tensor["dtype"]]
        result[tensor["name"]] = np.frombuffer(data, dtype=dtype).reshape(tensor["shape"]).copy()
    return result


def torch_reference(root, reference_source):
    import logging
    import numpy as np
    import torch
    data = reference_source.read_bytes()
    if hashlib.sha256(data).hexdigest() != SGLANG_SHA256:
        raise ValueError("reference code differs from audited SGLang revision")
    # Only these three reviewed weight-conversion functions are executed, not
    # the remote module, imports, initializers, decorators or CUDA model code.
    selected = {"_get_ckpt_qkv_shard_sizes", "_deinterleave_qkv_shards", "_resolve_deferred_qkv_scale_inv"}
    tree = ast.parse(data)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    if {f.name for f in functions} != selected or any(f.decorator_list for f in functions):
        raise ValueError("reference function selection mismatch")
    scope = {"torch": torch, "math": math, "re": re, "Dict": Dict,
             "get_parallel": lambda: SimpleNamespace(attn_tp_size=1, attn_tp_rank=0),
             "ceil_align": lambda n, size: (n + size - 1) // size * size,
             "default_weight_loader": lambda param, value: param.data.copy_(value),
             "logger": logging.getLogger("mimo_reference")}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "pinned_sglang_weight_reference", "exec"), scope)
    tensors = read_slices(root)
    prefix = "model.layers.0.self_attn.qkv_proj"
    raw = torch.from_numpy(tensors[prefix + ".weight"]).view(torch.float8_e4m3fn)
    scales = torch.from_numpy(tensors[prefix + ".weight_scale_inv"])
    params = {prefix + ".weight": torch.nn.Parameter(raw, requires_grad=False),
              prefix + ".weight_scale_inv": torch.nn.Parameter(
                  torch.zeros(math.ceil(raw.shape[0] / 128), math.ceil(raw.shape[1] / 128)), requires_grad=False)}
    config = SimpleNamespace(hybrid_layer_pattern=[0], num_attention_heads=64, num_key_value_heads=4,
                             head_dim=192, v_head_dim=128)
    capture = {}

    def trace(frame, event, arg):
        if (frame.f_code.co_name == "_resolve_deferred_qkv_scale_inv" and event == "return"
                and "merged_bf16" in frame.f_locals):
            capture["canonical"] = frame.f_locals["merged_bf16"].detach().clone()
        return trace

    previous = sys.gettrace()
    try:
        sys.settrace(trace)
        scope["_resolve_deferred_qkv_scale_inv"](
            params, {prefix + ".weight_scale_inv": scales}, 4, config=config)
    finally:
        sys.settrace(previous)
    # Whallm keeps this BF16 common tensor; SGLang subsequently requantizes it
    # for CUDA FP8. Comparison must target the same pre-requantization stage.
    np.save(root / "qkv-reference.npy", capture["canonical"].float().numpy())
    (root / "reference.json").write_text(json.dumps({
        "sglang_revision": SGLANG_REVISION, "source_sha256": SGLANG_SHA256,
        "torch_version": torch.__version__, "stage": "canonical_BF16_before_requantization",
        "reference_sha256": hashlib.sha256((root / "qkv-reference.npy").read_bytes()).hexdigest(),
    }, indent=2) + "\n")


def validate(root):
    import mlx.core as mx
    import numpy as np
    from deepseek_v4_ssd.mimo.weights import decode_qkv, native_expert_views
    from deepseek_v4_ssd.model import _mxfp4
    from runtime.tests.test_mimo_weights import mxfp4_reference
    tensors = read_slices(root)
    report = {"model_id": MODEL_ID, "revision": REVISION,
              "performance_result": False, "full_model_validated": False,
              "checks": [], "reference": json.loads((root / "reference.json").read_text())}
    expected_hash = report["reference"]["reference_sha256"]
    if hashlib.sha256((root / "qkv-reference.npy").read_bytes()).hexdigest() != expected_hash:
        raise ValueError("reference tensor integrity mismatch")
    for projection, width in (("gate_proj", 4096), ("up_proj", 4096), ("down_proj", 2048)):
        prefix = f"model.layers.1.mlp.experts.0.{projection}"
        raw, scales = tensors[prefix + ".weight"], tensors[prefix + ".weight_scale"]
        weights, scale = native_expert_views(mx.array(raw), mx.array(scales))
        decoded = mxfp4_reference(raw, scales)
        x = np.random.default_rng(71).normal(0, .02, (3, width)).astype(np.float32)
        actual = np.asarray(_mxfp4(mx.array(x), weights, scale))
        expected = x @ decoded.T
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
        report["checks"].append({"projection": projection,
                                 "max_absolute_error": float(np.max(np.abs(actual - expected))),
                                 "rtol": 2e-4, "atol": 2e-5})
    prefix = "model.layers.0.self_attn.qkv_proj"
    actual = decode_qkv(mx.array(tensors[prefix + ".weight"]),
                        mx.array(tensors[prefix + ".weight_scale_inv"]),
                        query_heads=64, kv_heads=4, head_dim=192, value_dim=128, source_shards=4)
    actual = np.asarray(actual.astype(mx.float32))
    expected = np.load(root / "qkv-reference.npy", allow_pickle=False)
    np.testing.assert_array_equal(actual, expected)
    if actual.tobytes() != expected.tobytes():
        raise AssertionError("QKV values match but bit patterns differ from reference")
    report["checks"].append({"projection": "global_QKV", "bit_exact_to_BF16_reference": True,
                             "shape": list(actual.shape)})
    report["passed"] = True
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--reference-source", type=Path, required=True)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--torch-reference", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.torch_reference:
        torch_reference(args.output, args.reference_source)
        return
    if not args.reference_python or (not args.offline and not args.audit):
        parser.error("--reference-python and online --audit are required")
    if not args.offline:
        collect(args.audit, args.output)
    subprocess.run([str(args.reference_python), str(Path(__file__).resolve()),
                    "--output", str(args.output), "--reference-source", str(args.reference_source),
                    "--torch-reference"], check=True)
    validate(args.output)


if __name__ == "__main__":
    main()
