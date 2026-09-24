"""Offline, lossless checkpoint repack plus prepared MLX common tensors.

Every upstream file can be reconstructed byte-for-byte from checkpoint-map.json.
The original checkpoint is never modified. Publication is a separate operation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil

MODEL_ID = "XiaomiMiMo/MiMo-V2.6-Flash-RL"
REVISION = "5711b268169967567844e1e560e8a3966da959b1"
KIND = "mimo-v2.6-flash-rl"
FORMAT = 4
CHUNK = 8 * 1024**2
EXPERT_BYTES = 13_369_344
EXPERT_REGIONS = (
    ("w1.weight", "I8", (2048, 2048), 0, 4194304),
    ("w1.scale", "F8_E8M0", (2048, 128), 4194304, 262144),
    ("w2.weight", "I8", (4096, 1024), 4456448, 4194304),
    ("w2.scale", "F8_E8M0", (4096, 64), 8650752, 262144),
    ("w3.weight", "I8", (2048, 2048), 8912896, 4194304),
    ("w3.scale", "F8_E8M0", (2048, 128), 13107200, 262144),
)
DTYPES = {"U8": 1, "I8": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4,
          "F16": 2, "I32": 4, "I64": 8}
EXPERT = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|down_proj|up_proj)\.(weight|weight_scale)$")


def safe_path(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError(f"unsafe artifact path: {name}")
    pure = PurePosixPath(name)
    if (not pure.parts or pure.is_absolute() or ".." in pure.parts
            or str(pure) != name or "\\" in name):
        raise ValueError(f"unsafe artifact path: {name}")
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"artifact path escapes root: {name}")
    return path


def atomic_json(path: Path, value):
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def file_sha(path: Path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def header(path):
    with path.open("rb") as file:
        prefix = file.read(8)
        if len(prefix) != 8:
            raise ValueError("truncated safetensors prefix")
        size = int.from_bytes(prefix, "little")
        if not 0 < size <= 64 * 1024**2 or size % 8:
            raise ValueError("invalid safetensors header size")
        data = file.read(size)
        if len(data) != size:
            raise ValueError("truncated safetensors header")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate tensor metadata: {key}")
            result[key] = value
        return result
    tensors = json.loads(data, object_pairs_hook=unique)
    tensors.pop("__metadata__", None)
    end = 0
    for name, t in sorted(tensors.items(), key=lambda p: p[1]["data_offsets"]):
        start, stop = t["data_offsets"]
        if (type(start) is not int or type(stop) is not int or start != end or stop < start
                or t["dtype"] not in DTYPES or any(type(n) is not int or n < 0 for n in t["shape"])
                or stop - start != math.prod(t["shape"]) * DTYPES[t["dtype"]]):
            raise ValueError(f"invalid tensor layout: {name}")
        end = stop
    if not tensors or 8 + size + end != path.stat().st_size:
        raise ValueError("safetensors payload coverage mismatch")
    return prefix + data, tensors


def make_plan(source: Path, source_info: dict):
    if source_info.get("repository") != MODEL_ID or source_info.get("revision") != REVISION:
        raise ValueError("checkpoint identity does not match pinned MiMo revision")
    index = json.loads((source / "model.safetensors.index.json").read_text())
    if index["metadata"].get("tp_size") != 4 or index["metadata"].get("save_format") != "mxfp4":
        raise ValueError("unsupported MiMo source layout")
    source_files = source_info["files"]
    for name, item in source_files.items():
        # HF snapshots use symlinks to blobs outside the snapshot. Only this
        # trusted local source allows symlinks; installed output paths do not.
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError("unsafe checkpoint source path")
        path = source / name
        if not path.is_file() or path.stat().st_size != item["size"]:
            raise ValueError(f"missing or incomplete checkpoint file: {name}")
    region_by_source = {}
    for projection, canonical in (("gate_proj", "w1"), ("down_proj", "w2"), ("up_proj", "w3")):
        for suffix, installed_suffix in (("weight", "weight"), ("weight_scale", "scale")):
            region_by_source[projection, suffix] = next(r for r in EXPERT_REGIONS if r[0] == f"{canonical}.{installed_suffix}")
    shards = sorted(set(index["weight_map"].values()) - {"model_mtp.safetensors"})
    copies, headers, seen = [], {}, set()
    common_offset = 0
    for shard in shards:
        if shard not in source_files:
            raise ValueError("index refers to an unlisted checkpoint shard")
        raw_header, tensors = header(source / shard)
        expected = {n for n, file in index["weight_map"].items() if file == shard}
        if set(tensors) != expected:
            raise ValueError(f"index/header mismatch: {shard}")
        headers[shard] = {"path": f"checkpoint/headers/{shard}.header", "size": len(raw_header),
                          "sha256": hashlib.sha256(raw_header).hexdigest()}
        for name, t in sorted(tensors.items(), key=lambda p: p[1]["data_offsets"]):
            if name in seen:
                raise ValueError("duplicate checkpoint tensor")
            seen.add(name)
            begin, end = t["data_offsets"]
            match = EXPERT.fullmatch(name)
            if match:
                layer, expert, projection, suffix = match.groups()
                layer, expert = int(layer), int(expert)
                if not 1 <= layer <= 47 or not 0 <= expert < 256:
                    raise ValueError("invalid expert identity")
                region = region_by_source[projection, suffix]
                if t["dtype"] != "U8" or tuple(t["shape"]) != region[2] or end - begin != region[4]:
                    raise ValueError(f"unsupported expert representation: {name}")
                destination = f"experts/layer_{layer - 1:02d}.bin"
                offset = expert * EXPERT_BYTES + region[3]
            else:
                destination, offset = "checkpoint/common.bin", common_offset
                common_offset += end - begin
            copies.append({"name": name, "source": shard, "sourceOffset": len(raw_header) + begin,
                           "length": end - begin, "dtype": t["dtype"], "shape": t["shape"],
                           "destination": destination, "offset": offset})
    expected_experts = 47 * 256 * 6
    if sum(EXPERT.fullmatch(t["name"]) is not None for t in copies) != expected_experts:
        raise ValueError("checkpoint does not contain all 12,032 complete experts")
    whole = {name: {**item, "destination": f"checkpoint/{name}"}
             for name, item in source_files.items() if name not in shards}
    # Retain *all* companions, including both drafts, all encoder weights,
    # original code/config/tokenizers, reports and assets, not a text-only subset.
    sizes = {f"experts/layer_{i:02d}.bin": 256 * EXPERT_BYTES for i in range(47)}
    sizes["checkpoint/common.bin"] = common_offset
    return {"version": 1, "repository": MODEL_ID, "revision": REVISION,
            "sourceFiles": source_files, "headers": headers, "copies": copies,
            "wholeFiles": whole, "outputSizes": sizes}


def _copy_region(src, dst_fd, offset, count, hasher):
    while count:
        data = src.read(min(CHUNK, count))
        if not data:
            raise ValueError("checkpoint read was truncated")
        hasher.update(data)
        view = memoryview(data)
        while view:
            n = os.pwrite(dst_fd, view, offset)
            if n <= 0:
                raise OSError("short artifact write")
            view = view[n:]
            offset += n
        count -= len(data)


def copy_checkpoint(source, root, plan):
    descriptors = {}
    try:
        for name, size in plan["outputSizes"].items():
            target = safe_path(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(target, os.O_RDWR | os.O_CREAT, 0o600)
            os.ftruncate(fd, size)
            descriptors[name] = fd
        def copy_shard(shard):
            raw, _ = header(source / shard)
            info = plan["headers"][shard]
            if hashlib.sha256(raw).hexdigest() != info["sha256"]:
                raise ValueError("source header changed after planning")
            target = safe_path(root, info["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            sha = hashlib.sha256(raw)
            with (source / shard).open("rb") as src:
                src.seek(len(raw))
                for item in (t for t in plan["copies"] if t["source"] == shard):
                    if src.tell() != item["sourceOffset"]:
                        raise ValueError("non-contiguous source copy plan")
                    _copy_region(src, descriptors[item["destination"]], item["offset"], item["length"], sha)
                if src.read(1):
                    raise ValueError("unaccounted checkpoint bytes")
            if sha.hexdigest() != plan["sourceFiles"][shard]["sha256"]:
                raise ValueError(f"checkpoint SHA-256 mismatch: {shard}")
            print(f"repacked {shard}", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(copy_shard, plan["headers"]))
        for fd in descriptors.values():
            os.fsync(fd)
    finally:
        for fd in descriptors.values():
            os.close(fd)
    for name, item in plan["wholeFiles"].items():
        target = safe_path(root, item["destination"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
        sha = file_sha(target)
        if item.get("sha256") and sha != item["sha256"]:
            raise ValueError(f"companion SHA-256 mismatch: {name}")
        if item.get("blobID") and not item.get("sha256"):
            data = target.read_bytes()
            blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if blob != item["blobID"]:
                raise ValueError(f"companion Git blob mismatch: {name}")
        item["sha256"] = sha
        print(f"preserved {name}", flush=True)


def prepare_common(root, plan):
    import mlx.core as mx
    import numpy as np
    from .weights import decode_fp8_blocks, decode_qkv
    config = json.loads((root / "checkpoint/config.json").read_text())
    raw_tensors = {t["name"]: t for t in plan["copies"] if not EXPERT.fullmatch(t["name"])}
    mtp_file = root / "checkpoint/model_mtp.safetensors"
    mtp_header, mtp_tensors = header(mtp_file)
    for name, t in mtp_tensors.items():
        start, end = t["data_offsets"]
        raw_tensors[name] = {"name": name, "destination": "checkpoint/model_mtp.safetensors",
                             "offset": len(mtp_header) + start, "length": end - start,
                             "dtype": t["dtype"], "shape": t["shape"]}
    mapped = {}
    def load(item):
        path = item["destination"]
        if path not in mapped:
            mapped[path] = np.memmap(root / path, mode="r", dtype=np.uint8)
        dtype = {"F8_E4M3": np.uint8, "BF16": np.uint16, "F32": np.float32}[item["dtype"]]
        values = mx.array(np.ndarray(tuple(item["shape"]), dtype=dtype,
                                    buffer=mapped[path], offset=item["offset"]))
        return values.view(mx.bfloat16) if item["dtype"] == "BF16" else values
    components = {}
    for component, prefix, destination in (("backbone", None, "common.bin"),
                                           ("mtp", "model.mtp.", "mtp/common.bin"),
                                           ("vision", "visual.", "vision/common.bin"),
                                           ("audio_patch", ("audio_encoder.", "speech_embeddings."), "audio/common.bin")):
        selected = {n: t for n, t in raw_tensors.items() if
                    (n.startswith(("model.layers.", "model.embed_tokens.", "model.norm.", "lm_head."))
                     if prefix is None else n.startswith(prefix))}
        output = root / destination
        output.parent.mkdir(parents=True, exist_ok=True)
        table = []
        with output.open("wb") as file:
            for name, item in sorted(selected.items()):
                if name.endswith(".weight_scale_inv"):
                    continue
                offset = (file.tell() + 255) // 256 * 256
                file.write(bytes(offset - file.tell()))
                if item["dtype"] == "F8_E4M3":
                    scale_name = name + "_scale_inv"
                    if scale_name not in selected:
                        raise ValueError(f"FP8 common tensor has no scales: {name}")
                    weight, scales = load(item), load(selected[scale_name])
                    if ".self_attn.qkv_proj." in name:
                        layer = int(name.split(".")[3 if component == "mtp" else 2])
                        sliding = component == "mtp" or config["hybrid_layer_pattern"][layer] == 1
                        p = "swa_" if sliding else ""
                        value = decode_qkv(weight, scales, query_heads=config[p + "num_attention_heads"],
                                           kv_heads=config[p + "num_key_value_heads"], head_dim=config[p + "head_dim"],
                                           value_dim=config[p + "v_head_dim"], source_shards=4)
                    else:
                        value = decode_fp8_blocks(weight, scales)
                    mx.eval(value)
                    data = np.asarray(value.view(mx.uint16)).tobytes()
                    dtype = "BF16"
                    del weight, scales, value
                    mx.clear_cache()
                elif name.endswith("attention_sink_bias"):
                    value = load(item).astype(mx.float32)
                    data, dtype = np.asarray(value).tobytes(), "F32"
                    del value
                else:
                    with (root / item["destination"]).open("rb") as src:
                        src.seek(item["offset"])
                        data = src.read(item["length"])
                    dtype = item["dtype"]
                file.write(data)
                table.append({"name": name, "dtype": dtype, "shape": item["shape"],
                              "offset": offset, "length": len(data)})
                del data
            file.flush()
            os.fsync(file.fileno())
        components[component] = {"file": destination, "tensors": table}
        print(f"prepared {component}: {len(table)} tensors", flush=True)
    mapped.clear()
    # Decoder sidecar and AudioTokenizer are already natively readable
    # BF16/F32 safetensors, retained byte-exact with all extra decoder weights.
    components["audio_tokenizer"] = {"file": "checkpoint/audio_tokenizer/model.safetensors"}
    components["dflash"] = {"file": "checkpoint/dflash/dflash_draft_model.safetensors",
                             "mask": "checkpoint/dflash/mask_embedding.pt"}
    for name in ("config.json", "generation_config.json"):
        shutil.copyfile(root / "checkpoint" / name, root / name)
    (root / "tokenizer").mkdir(exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"):
        shutil.copyfile(root / "checkpoint" / name, root / "tokenizer" / name)
    return components


def verify_reconstruction(root: Path, plan: dict):
    """Hash reconstructed source file bytes, not merely our output file hashes."""
    descriptors = {}
    try:
        for shard, info in plan["headers"].items():
            raw = safe_path(root, info["path"]).read_bytes()
            if len(raw) != info["size"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
                raise ValueError("saved checkpoint header is invalid")
            sha = hashlib.sha256(raw)
            position = len(raw)
            for item in (t for t in plan["copies"] if t["source"] == shard):
                if position != item["sourceOffset"]:
                    raise ValueError("source reconstruction coverage mismatch")
                name = item["destination"]
                if name not in descriptors:
                    descriptors[name] = os.open(safe_path(root, name), os.O_RDONLY)
                offset, left = item["offset"], item["length"]
                while left:
                    data = os.pread(descriptors[name], min(left, CHUNK), offset)
                    if not data:
                        raise ValueError("truncated installed tensor")
                    sha.update(data)
                    offset += len(data)
                    left -= len(data)
                position += item["length"]
            if position != plan["sourceFiles"][shard]["size"] or sha.hexdigest() != plan["sourceFiles"][shard]["sha256"]:
                raise ValueError(f"lossless reconstruction failed: {shard}")
            print(f"verified original {shard}", flush=True)
    finally:
        for fd in descriptors.values():
            os.close(fd)
    for name, item in plan["wholeFiles"].items():
        path = safe_path(root, item["destination"])
        if path.stat().st_size != item["size"] or file_sha(path) != item["sha256"]:
            raise ValueError(f"preserved companion integrity failed: {name}")
    return {"sourceFiles": len(plan["sourceFiles"]), "sourceShardsReconstructed": len(plan["headers"]),
            "allSourceBytesPreserved": True}


def convert(source: Path, output: Path, source_info: dict):
    output = output.expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(output.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            raise ValueError("destination already exists; refusing to replace it")
        partial = output.with_suffix(output.suffix + ".partial")
        plan = make_plan(source, source_info)
        identity = hashlib.sha256(json.dumps(source_info, sort_keys=True).encode()).hexdigest()
        receipt = partial / "conversion-receipt.json"
        if partial.exists():
            if not receipt.is_file() or json.loads(receipt.read_text())["sourceIdentity"] != identity:
                raise ValueError("partial artifact belongs to a different checkpoint")
        else:
            required = sum(plan["outputSizes"].values()) + sum(i["size"] for i in plan["wholeFiles"].values()) + 20 * 1024**3
            if shutil.disk_usage(output.parent).free < required:
                raise ValueError(f"insufficient disk space; require {required} free bytes")
            partial.mkdir()
            atomic_json(receipt, {"sourceIdentity": identity, "stage": "new"})
        state = json.loads(receipt.read_text())
        if state["stage"] == "new":
            copy_checkpoint(source, partial, plan)
            atomic_json(partial / "checkpoint-map.json", plan)
            state["stage"] = "copied"
            atomic_json(receipt, state)
        else:
            plan = json.loads((partial / "checkpoint-map.json").read_text())
        components = prepare_common(partial, plan)
        reconstruction = verify_reconstruction(partial, plan)
        atomic_json(partial / "preservation.json", reconstruction)
        files = []
        for path in sorted(partial.rglob("*")):
            if path.is_file() and path != receipt and path.name != "manifest.json":
                files.append({"path": str(path.relative_to(partial)), "size": path.stat().st_size,
                              "sha256": file_sha(path)})
        regions = [{"name": n, "dtype": d, "shape": s, "offset": o, "length": l}
                   for n, d, s, o, l in EXPERT_REGIONS]
        manifest = {"formatVersion": FORMAT, "modelKind": KIND, "modelID": MODEL_ID,
                    "revision": REVISION, "layerCount": 47, "backboneLayerCount": 48,
                    "expertCount": 256, "selectedExpertCount": 8, "expertBlobSize": EXPERT_BYTES,
                    "maximumContext": 1048576, "expertRegions": regions,
                    "commonTensors": components["backbone"]["tensors"], "components": components,
                    "expertQuantization": {"mode": "mxfp4", "bits": 4, "groupSize": 32, "conversionVersion": 1},
                    "files": files, "ngram": None, "mtp": None, "dspark": None,
                    "conversion": {"version": 1, "sourceTP": 4, "qkv": "canonical-bf16", "losslessSource": True}}
        atomic_json(partial / "manifest.json", manifest)
        receipt.unlink()
        os.rename(partial, output)
        print(f"installed complete checkpoint: {output}", flush=True)
        return manifest
