from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEEPSEEK_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEEPSEEK_REVISION = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
DEEPSEEK_V41_MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
DEEPSEEK_V41_REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"
QWEN_MODEL_ID = "Qwen/Qwen3.8-Flash-Next-FP8"
QWEN_REVISION = "bcd9f01ddc9cff2316eb84281bebcd5b058bddce"

# Keep these names for callers that use the DeepSeek format 1 contract.
MODEL_ID = DEEPSEEK_MODEL_ID
REVISION = DEEPSEEK_REVISION
LAYER_COUNT = 43
EXPERT_COUNT = 256
SELECTED_EXPERT_COUNT = 6
EXPERT_BLOB_SIZE = 13_369_344
EXPERT_REGIONS = (
    ("w1.weight", "I8", (2_048, 2_048), 0, 4_194_304),
    ("w1.scale", "F8_E8M0", (2_048, 128), 4_194_304, 262_144),
    ("w2.weight", "I8", (4_096, 1_024), 4_456_448, 4_194_304),
    ("w2.scale", "F8_E8M0", (4_096, 64), 8_650_752, 262_144),
    ("w3.weight", "I8", (2_048, 2_048), 8_912_896, 4_194_304),
    ("w3.scale", "F8_E8M0", (2_048, 128), 13_107_200, 262_144),
)
DEEPSEEK_V41_EXPERT_REGIONS = (
    ("w1.weight", "I8", (2_304, 2_560), 0, 5_898_240),
    ("w1.scale", "F8_E8M0", (2_304, 160), 5_898_240, 368_640),
    ("w2.weight", "I8", (5_120, 1_152), 6_266_880, 5_898_240),
    ("w2.scale", "F8_E8M0", (5_120, 72), 12_165_120, 368_640),
    ("w3.weight", "I8", (2_304, 2_560), 12_533_760, 5_898_240),
    ("w3.scale", "F8_E8M0", (2_304, 160), 18_432_000, 368_640),
)
QWEN_EXPERT_REGIONS = (
    ("gate_up.weight", "U32", (1_280, 320), 0, 1_638_400),
    ("gate_up.scale", "U8", (1_280, 80), 1_638_400, 102_400),
    ("down.weight", "U32", (2_560, 80), 1_740_800, 819_200),
    ("down.scale", "U8", (2_560, 20), 2_560_000, 51_200),
)
QWEN_NGRAM_HEAD_OFFSETS = (
    0, 20_000_003, 40_000_026, 60_000_059, 80_000_106, 100_000_165,
    120_000_228, 140_000_297, 160_000_374, 180_000_455, 200_000_548,
    220_000_655, 240_000_802, 260_000_955, 280_001_114, 300_001_275,
)
QWEN_NGRAM_HEAD_VOCAB_SIZES = (
    20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063,
    20_000_069, 20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147,
    20_000_153, 20_000_159, 20_000_161, 20_000_171,
)


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int


@dataclass(frozen=True)
class ExpertQuantization:
    mode: str
    bits: int
    group_size: int
    conversion_version: int


@dataclass(frozen=True)
class NGram:
    file: str
    dtype: str
    row_bytes: int
    shard_count: int
    shard_row_count: int
    head_offsets: tuple[int, ...]
    head_vocab_sizes: tuple[int, ...]


@dataclass(frozen=True)
class MTP:
    layer_count: int
    use_dedicated_embeddings: bool
    common_tensors: tuple[Tensor, ...]


@dataclass(frozen=True)
class EngramTable:
    layer: int
    weight_file: str
    scale_file: str
    rows: int
    dimension: int
    block_size: int


@dataclass(frozen=True)
class Engram:
    tables: tuple[EngramTable, ...]


@dataclass(frozen=True)
class InstalledModel:
    root: Path
    model_id: str
    revision: str
    layer_count: int
    expert_count: int
    selected_expert_count: int
    expert_blob_size: int
    common_tensors: tuple[Tensor, ...]
    expert_regions: tuple[Tensor, ...]
    model_kind: str = "deepseek-v4"
    format_version: int = 1
    maximum_context: int = 1_048_576
    expert_quantization: ExpertQuantization | None = None
    ngram: NGram | None = None
    mtp: MTP | None = None
    engram: Engram | None = None
    dspark_layer_count: int = 0
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 0
    dspark_common_tensors: tuple[Tensor, ...] = ()

    @property
    def has_dspark(self) -> bool:
        return self.dspark_layer_count > 0

    @property
    def is_qwen(self) -> bool:
        from .model_support.catalog import QWEN_MODEL_KINDS
        return self.model_kind in QWEN_MODEL_KINDS

    @property
    def is_deepseek_v41(self) -> bool:
        return self.model_kind == "deepseek-v4.1"

    @property
    def has_mtp(self) -> bool:
        return self.mtp is not None

    @classmethod
    def open(cls, root: str | Path) -> InstalledModel:
        root = Path(root).expanduser().resolve()
        with (root / "manifest.json").open("rb") as file:
            raw = json.load(file)

        from .model_support import support_for_manifest
        contract = support_for_manifest(raw).manifest_contract(raw)

        files = {item["path"]: item["size"] for item in raw.get("files", [])}
        if len(files) != len(raw.get("files", [])):
            raise ValueError("installed manifest contains duplicate file paths")
        missing = contract["required"].difference(files)
        if missing:
            raise ValueError(f"installed model is missing {sorted(missing)[0]}")
        if set(files) != contract["allowed"]:
            raise ValueError("installed manifest contains an unexpected file")
        for path, size in files.items():
            target = (root / path).resolve()
            if root not in target.parents or not target.is_file():
                raise ValueError(f"installed model has an unsafe or missing file: {path}")
            if target.stat().st_size != size:
                raise ValueError(f"installed file size does not match the manifest: {path}")

        expected_layer_size = contract["expert_count"] * contract["expert_blob_size"]
        for layer in range(contract["layer_count"]):
            path = f"experts/layer_{layer:02d}.bin"
            if files[path] != expected_layer_size:
                raise ValueError(f"installed expert layer has an invalid size: {path}")

        def tensors(key: str) -> tuple[Tensor, ...]:
            return tuple(
                Tensor(
                    name=item["name"],
                    dtype=item["dtype"],
                    shape=tuple(item["shape"]),
                    offset=item["offset"],
                    length=item["length"],
                )
                for item in raw[key]
            )

        common_tensors = tensors("commonTensors")
        expert_regions = tensors("expertRegions")
        actual_regions = tuple(
            (item.name, item.dtype, item.shape, item.offset, item.length)
            for item in expert_regions
        )
        if actual_regions != contract["expert_regions"]:
            raise ValueError("installed expert blob layout does not match the model contract")
        _validate_tensors(common_tensors, files["common.bin"], "common")

        dspark = raw.get("dspark")
        dspark_common_tensors: tuple[Tensor, ...] = ()
        if dspark is not None:
            dspark_common_tensors = tuple(
                Tensor(
                    name=item["name"],
                    dtype=item["dtype"],
                    shape=tuple(item["shape"]),
                    offset=item["offset"],
                    length=item["length"],
                )
                for item in dspark["commonTensors"]
            )
            _validate_tensors(
                dspark_common_tensors,
                files["dspark/common.bin"],
                "DSpark common",
                prefix="mtp.",
            )
            for layer in range(3):
                path = f"dspark/experts/layer_{layer:02d}.bin"
                dspark_layer_size = 128 * contract["expert_blob_size"] if contract["model_kind"] == "deepseek-v4.1" else expected_layer_size
                if files[path] != dspark_layer_size:
                    raise ValueError(f"installed DSpark expert layer has an invalid size: {path}")

        mtp_raw = raw.get("mtp")
        mtp: MTP | None = None
        if mtp_raw is not None:
            mtp_common_tensors = tuple(
                Tensor(
                    name=item["name"],
                    dtype=item["dtype"],
                    shape=tuple(item["shape"]),
                    offset=item["offset"],
                    length=item["length"],
                )
                for item in mtp_raw["commonTensors"]
            )
            _validate_tensors(
                mtp_common_tensors,
                files["mtp/common.bin"],
                "MTP common",
                prefix="mtp.",
            )
            if any(".experts." in item.name for item in mtp_common_tensors):
                raise ValueError("installed MTP common tensor table contains an expert")
            mtp = MTP(
                layer_count=mtp_raw["layerCount"],
                use_dedicated_embeddings=mtp_raw["useDedicatedEmbeddings"],
                common_tensors=mtp_common_tensors,
            )

        engram_raw = raw.get("engram")
        engram: Engram | None = None
        if engram_raw is not None:
            engram = Engram(
                tables=tuple(
                    EngramTable(
                        layer=item["layer"],
                        weight_file=item["weightFile"],
                        scale_file=item["scaleFile"],
                        rows=item["rows"],
                        dimension=item["dimension"],
                        block_size=item["blockSize"],
                    )
                    for item in engram_raw["tables"]
                )
            )

        return cls(
            root=root,
            model_id=raw["modelID"],
            revision=raw["revision"],
            layer_count=contract["layer_count"],
            expert_count=contract["expert_count"],
            selected_expert_count=contract["selected_expert_count"],
            expert_blob_size=contract["expert_blob_size"],
            common_tensors=common_tensors,
            expert_regions=expert_regions,
            model_kind=contract["model_kind"],
            format_version=raw["formatVersion"],
            maximum_context=contract["maximum_context"],
            expert_quantization=contract.get("expert_quantization"),
            ngram=contract.get("ngram"),
            mtp=mtp,
            engram=engram,
            dspark_layer_count=dspark["layerCount"] if dspark is not None else 0,
            dspark_block_size=dspark["blockSize"] if dspark is not None else 0,
            dspark_noise_token_id=dspark["noiseTokenID"] if dspark is not None else 0,
            dspark_target_layer_ids=(
                tuple(dspark["targetLayerIDs"]) if dspark is not None else ()
            ),
            dspark_markov_rank=dspark["markovRank"] if dspark is not None else 0,
            dspark_common_tensors=dspark_common_tensors,
        )


def _validate_tensors(
    tensors: tuple[Tensor, ...],
    file_size: int,
    label: str,
    *,
    prefix: str | None = None,
) -> None:
    names = {item.name for item in tensors}
    if not tensors or len(names) != len(tensors):
        raise ValueError(f"installed {label} tensors are empty or have duplicate names")
    if any(
        (prefix is not None and not item.name.startswith(prefix))
        or item.offset < 0
        or item.length < 0
        or item.offset + item.length > file_size
        for item in tensors
    ):
        raise ValueError(f"installed {label} tensor table is invalid")

# Compatibility entry points for offline contract checks. Runtime dispatches via packages.
def _deepseek_contract(raw):
    from .model_support.deepseek_v4 import _deepseek_contract as validate
    return validate(raw)

def _deepseek_v41_contract(raw):
    from .model_support.deepseek_v41 import _deepseek_v41_contract as validate
    return validate(raw)

def _qwen_contract(raw):
    from .model_support import get_support
    return get_support("qwen3.8-flash-next").manifest_contract(raw)
