"""The trusted model descriptions shared with the Swift installer and App."""

from __future__ import annotations

import json
import math
import re
from typing import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


QWEN_MODEL_KINDS = frozenset({"qwen3.8-flash-next", "swift1.5-qwen3.8-flash-next"})


@dataclass(frozen=True)
class ModelDescriptor:
    kind: str
    display_name: str
    api_model_id: str
    owner: str
    checkpoint_model_id: str
    checkpoint_revision: str
    manifest_version: int
    features: frozenset[str]
    defaults: Mapping[str, int | float | bool]
    automatic_memory_gib: int
    prefill_threshold: int | None

    def supports(self, feature: str) -> bool:
        return feature in self.features


def read_catalog(path: Path) -> tuple[ModelDescriptor, ...]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) != {"version", "models"} or type(raw["version"]) is not int or raw["version"] != 1:
        raise ValueError("unsupported model package catalog")
    if not isinstance(raw["models"], list):
        raise ValueError("model packages must be an array")
    result = []
    for item in raw["models"]:
        required = {
            "kind", "displayName", "kindLabel", "assistantName", "apiModelID", "owner",
            "checkpointModelID", "checkpointRevision", "manifestVersion", "directoryName",
            "requiredPaths", "features", "editableSettings", "defaults",
            "automaticMemoryGiB", "prefillThreshold",
        }
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("model package descriptor has invalid fields")
        for key in required - {
            "manifestVersion", "requiredPaths", "features", "editableSettings", "defaults",
            "automaticMemoryGiB", "prefillThreshold",
        }:
            if not isinstance(item[key], str) or not item[key]:
                raise ValueError(f"model package {key} must be a non-empty string")
        if (type(item["manifestVersion"]) is not int or item["manifestVersion"] < 1
                or type(item["automaticMemoryGiB"]) is not int or item["automaticMemoryGiB"] < 0):
            raise ValueError("invalid model package limits")
        threshold = item["prefillThreshold"]
        if threshold is not None and (type(threshold) is not int or threshold < 1):
            raise ValueError("invalid model package prefill threshold")
        for key in ("requiredPaths", "features", "editableSettings"):
            values = item[key]
            if (not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values)
                    or len(values) != len(set(values))):
                raise ValueError(f"invalid model package {key}")
        for value in [item["directoryName"], *item["requiredPaths"]]:
            if Path(value).is_absolute() or ".." in Path(value).parts:
                raise ValueError("unsafe model package path")
        if not re.fullmatch(r"[0-9a-f]{40}", item["checkpointRevision"]):
            raise ValueError("model package checkpoint must use a fixed commit")
        defaults = item["defaults"]
        limits = {"slots": (6, None), "promptCacheEntries": (1, None),
                  "maxTokens": (1, 272_000), "topK": (0, 248_320)}
        if not isinstance(defaults, dict) or set(defaults) != {*limits, "bf16KVCache", "temperature", "topP"}:
            raise ValueError("invalid model package defaults")
        for key, (minimum, maximum) in limits.items():
            value = defaults[key]
            if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
                raise ValueError(f"invalid model package default {key}")
        if type(defaults["bf16KVCache"]) is not bool:
            raise ValueError("invalid model package cache precision")
        for key, minimum, maximum in (("temperature", 0, 2), ("topP", 0.000001, 1)):
            value = defaults[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"invalid model package default {key}")
        result.append(ModelDescriptor(
            item["kind"], item["displayName"], item["apiModelID"], item["owner"],
            item["checkpointModelID"], item["checkpointRevision"], item["manifestVersion"],
            frozenset(item["features"]), MappingProxyType(item["defaults"]),
            item["automaticMemoryGiB"], threshold,
        ))
    for field in ("kind", "api_model_id"):
        values = [getattr(item, field) for item in result]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate model package {field}")
    if not result:
        raise ValueError("model package catalog is empty")
    return tuple(result)


def _catalog_path() -> Path:
    bundled = Path(__file__).with_name("ModelPackages.json")
    if bundled.is_file():
        return bundled
    # Development checkout; packaging copies the same file beside this module.
    return Path(__file__).resolve().parents[3] / "Sources/DeepSeekRepack/Resources/ModelPackages.json"


DESCRIPTORS = read_catalog(_catalog_path())
BY_KIND = MappingProxyType({item.kind: item for item in DESCRIPTORS})
