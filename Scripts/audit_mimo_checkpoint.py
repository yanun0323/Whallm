"""Audit pinned MiMo metadata without downloading weights or executing remote code.

Run online once, then repeat validation with --offline. This is an installation
prerequisite, not an inference, numerical-parity, or performance acceptance test.
"""
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess

MODEL_ID = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
REVISION = "3b38d063180c3e4aed9691fdc735f3d10b266ee4"
EXPERT_BYTES = 13_369_344
COMPANIONS = ("audio_tokenizer/model.safetensors", "dflash/dflash_draft_model.safetensors")
SMALL_FILES = (
    "README.md", "config.json", "configuration_mimo_v2.py", "modeling_mimo_v2.py",
    "model.safetensors.index.json", "preprocessor_config.json", "chat_template.jinja",
    "tokenizer_config.json", "generation_config.json", "audio_tokenizer/config.json",
    "dflash/config.json", "dflash/dflash.py", "dflash/model.safetensors.index.json",
)
DTYPE_BYTES = {"U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E8M0": 1,
               "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path):
    return json.loads(path.read_text(), object_pairs_hook=_unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def safe_shard(path: str) -> str:
    if not isinstance(path, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+\.safetensors", path):
        raise ValueError("invalid shard name")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or str(pure) != path:
        raise ValueError("unsafe shard path")
    return path


def validate_header(header: dict, file_size: int) -> dict:
    """Check complete payload coverage, dtype/shape bytes and metadata offsets."""
    tensors = header.get("tensors")
    if not isinstance(tensors, dict) or not tensors:
        raise ValueError("empty safetensors header")
    ranges = []
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name or not isinstance(tensor, dict):
            raise ValueError("invalid tensor metadata")
        shape, offsets, dtype = (tensor.get(key) for key in ("shape", "data_offsets", "dtype"))
        if (not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list) or len(offsets) != 2
                or any(type(n) is not int for n in offsets) or dtype not in DTYPE_BYTES):
            raise ValueError(f"invalid tensor shape/dtype/offsets: {name}")
        start, end = offsets
        if start < 0 or end < start or end > file_size - 8:
            raise ValueError(f"out-of-bounds tensor: {name}")
        if end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f"tensor byte count mismatch: {name}")
        ranges.append((start, end, name))
    previous = 0
    for start, end, name in sorted(ranges):
        if start != previous:
            raise ValueError(f"overlapping or non-contiguous tensors: {name}")
        previous = end
    # The remainder is the 8-byte length prefix plus JSON header, not padding
    # in the tensor payload. Hub's parser already verifies the remote header.
    header_bytes = file_size - previous - 8
    if not 0 < header_bytes <= 64 * 1024**2 or header_bytes % 8:
        raise ValueError("invalid safetensors file/header size")
    return tensors


def validate_contract(config: dict, index: dict, headers: dict, files: dict) -> dict:
    expected = {
        "model_type": "mimo_v2", "num_hidden_layers": 48, "hidden_size": 4096,
        "n_routed_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 2048,
        "max_position_embeddings": 1_048_576, "attention_projection_layout": "fused_qkv",
        "moe_layer_freq": [0] + [1] * 47,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"pinned MiMo config mismatch: {name}")
    if config.get("quantization_config", {}).get("store_dtype") != "mxfp4":
        raise ValueError("MiMo expert storage is not MXFP4")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("missing weight map")
    shards = {safe_shard(name) for name in weight_map.values()}
    if set(headers) != shards | set(COMPANIONS):
        raise ValueError("incomplete or unexpected shard audit")
    all_tensors = {}
    categories, dtypes = Counter(), Counter()
    for shard, header in headers.items():
        tensors = validate_header(header, files[shard]["size"])
        if shard not in shards:
            continue
        expected_names = {name for name, source in weight_map.items() if source == shard}
        if set(tensors) != expected_names:
            raise ValueError(f"index/header coverage mismatch: {shard}")
        for name, tensor in tensors.items():
            if name in all_tensors:
                raise ValueError(f"duplicate tensor across shards: {name}")
            all_tensors[name] = tensor
            size = tensor["data_offsets"][1] - tensor["data_offsets"][0]
            category = ("routed_experts" if ".experts." in name else
                        "mtp" if name.startswith("model.mtp.") else
                        "vision" if name.startswith("visual.") else
                        "audio_patch" if name.startswith(("audio_encoder.", "speech_embeddings."))
                        else "backbone_common")
            categories[category] += size
            dtypes[tensor["dtype"]] += size
    if set(all_tensors) != set(weight_map):
        raise ValueError("missing indexed tensors")
    if sum(categories.values()) != index.get("metadata", {}).get("total_size"):
        raise ValueError("index total_size does not match actual payload bytes")
    expected_experts = set()
    for layer in range(1, 48):
        for expert in range(256):
            for projection, rows, cols in (("gate_proj", 2048, 4096),
                                           ("up_proj", 2048, 4096),
                                           ("down_proj", 4096, 2048)):
                for suffix, shape in (("weight", [rows, cols // 2]),
                                      ("weight_scale", [rows, cols // 32])):
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}"
                    expected_experts.add(name)
                    actual = all_tensors.get(name, {})
                    if actual.get("dtype") != "U8" or actual.get("shape") != shape:
                        raise ValueError(f"native MXFP4 expert layout mismatch: {name}")
    if expected_experts != {name for name in all_tensors if ".experts." in name}:
        raise ValueError("unexpected expert tensor or dense layer treated as MoE")
    if categories["routed_experts"] != 47 * 256 * EXPERT_BYTES:
        raise ValueError("expert payload size mismatch")
    mtp_layers = sorted({int(name.split(".")[3]) for name in all_tensors
                         if name.startswith("model.mtp.layers.")})
    if mtp_layers != [0, 1, 2]:
        raise ValueError("MTP checkpoint changed")
    return {"indexed_shards": len(shards), "all_audited_shards": len(headers),
            "indexed_tensors": len(all_tensors), "routed_experts": 47 * 256,
            "expert_blob_bytes": EXPERT_BYTES, "tensor_bytes_by_category": dict(categories),
            "tensor_bytes_by_dtype": dict(dtypes), "mtp_layers": mtp_layers}


def fetch(root: Path) -> None:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    info = api.model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION:
        raise ValueError("Hub revision mismatch")
    files = {f.rfilename: {"size": f.size, "lfs": asdict(f.lfs) if f.lfs else None}
             for f in info.siblings}
    root.mkdir(parents=True, exist_ok=True)
    (root / "repo.json").write_text(json.dumps({"model_id": MODEL_ID, "revision": REVISION,
                                               "files": files}, indent=2) + "\n")
    for name in SMALL_FILES:
        hf_hub_download(MODEL_ID, name, revision=REVISION, local_dir=root / "source")
    index = read_json(root / "source/model.safetensors.index.json")
    shards = sorted({safe_shard(name) for name in index["weight_map"].values()} | set(COMPANIONS))
    (root / "headers").mkdir(exist_ok=True)

    def read_shard(name):
        # Range requests only. Never hf_hub_download a safetensors payload here.
        result = api.parse_safetensors_file_metadata(MODEL_ID, name, revision=REVISION, timeout=45)
        path = root / "headers" / (name.replace("/", "__") + ".json")
        path.write_text(json.dumps(asdict(result), indent=2) + "\n")
        print(f"header {name}: {len(result.tensors)} tensors", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(read_shard, shards))
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache" not in path.parts and path.name not in {"sources.json", "report.json"}:
            records.append({"path": str(path.relative_to(root)), "sha256": digest(path)})
    (root / "sources.json").write_text(json.dumps(records, indent=2) + "\n")


def audit(root: Path) -> dict:
    for record in read_json(root / "sources.json"):
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != record["sha256"]:
            raise ValueError("audit source escaped directory or changed")
    repo = read_json(root / "repo.json")
    if repo["revision"] != REVISION or repo["model_id"] != MODEL_ID:
        raise ValueError("audit repository identity mismatch")
    index = read_json(root / "source/model.safetensors.index.json")
    shards = {safe_shard(name) for name in index["weight_map"].values()} | set(COMPANIONS)
    headers = {name: read_json(root / "headers" / (name.replace("/", "__") + ".json"))
               for name in shards}
    config = read_json(root / "source/config.json")
    contract = validate_contract(config, index, headers, repo["files"])
    findings = []
    try:
        read_json(root / "source/dflash/config.json")
    except ValueError as error:
        findings.append({"scope": "optional_dflash", "code": "invalid_json", "detail": str(error)})
    processor = read_json(root / "source/preprocessor_config.json")
    for source_key, processor_key in (("image_min_pixels", "min_pixels"),
                                      ("image_max_pixels", "max_pixels")):
        if config["processor_config"][source_key] != processor[processor_key]:
            findings.append({"scope": "multimodal", "code": "processor_config_difference",
                             "key": source_key, "config": config["processor_config"][source_key],
                             "preprocessor_config": processor[processor_key]})
    repo_root = Path(__file__).resolve().parents[1]
    return {"schema_version": 1, "evidence_kind": "mimo_checkpoint_metadata_audit",
            "captured_at": datetime.now(timezone.utc).isoformat(), "model_id": MODEL_ID,
            "revision": REVISION, "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip(),
            "environment": {"platform": platform.platform(), "python": platform.python_version()},
            "script_sha256": digest(Path(__file__)), "sources_sha256": digest(root / "sources.json"),
            "contract": contract, "findings": findings, "metadata_validated": True,
            "numerical_parity_validated": False, "full_model_validated": False,
            "performance_result": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="verify previously fetched sources only")
    args = parser.parse_args()
    if not args.offline:
        fetch(args.output)
    report = audit(args.output)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
