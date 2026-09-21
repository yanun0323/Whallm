"""Consume real Swift-generated catalogs with the Python parser; no model loading."""
from __future__ import annotations

import argparse
from pathlib import Path

from deepseek_v4_ssd.model_manager import load_model_catalog
from deepseek_v4_ssd.qwen_flash_config import validate_flash_config


def validate(directory: Path) -> int:
    cases = {
        "qwen-default": (0, "mmap", 0, False),
        "qwen-waves": (32, "mmap", 0, False),
        "qwen-pread": (0, "pread", 64 * 1024**2, False),
        "qwen-sdpa": (0, "mmap", 0, True),
        "qwen-combined": (32, "pread", 64 * 1024**2, True),
        "qwen-mmap-restored": (0, "mmap", 0, False),
        "deepseek-v4": (0, "mmap", 0, False),
        "deepseek-v41": (0, "mmap", 0, False),
    }
    for name, expected in cases.items():
        specs = load_model_catalog(directory / f"{name}.json")
        if len(specs) != 1:
            raise ValueError(f"{name}: expected exactly one model")
        runtime = specs[0].runtime
        validate_flash_config(runtime)
        actual = (runtime.qwen_expert_wave_slots, runtime.qwen_ngram_io,
                  runtime.qwen_ngram_cache_bytes, runtime.qwen_sparse_sdpa)
        if actual != expected:
            raise ValueError(f"{name}: {actual!r} != {expected!r}")
        if expected[0] and any((runtime.qwen_next_layer_prefetch,
                                runtime.batched_expert_prefill,
                                runtime.qwen_grouped_experts)):
            raise ValueError(f"{name}: waves must suppress whole-layer conflicts")
        print(f"PASS {name}: {actual}")
    for name, indexed in (("qwen-throughput", True), ("qwen-throughput-inactive", False)):
        runtime = load_model_catalog(directory / f"{name}.json")[0].runtime
        validate_flash_config(runtime)
        if runtime.qwen_qsa_query_chunk != 32 or runtime.qwen_qsa_indexed != indexed:
            raise ValueError(f"{name}: QSA throughput settings did not survive the catalog")
        print(f"PASS {name}: QSA chunk=32 indexed={indexed}")
    return len(cases) + 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(f"Validated {validate(args.directory)} Swift-to-Python catalogs; no model was loaded.")


if __name__ == "__main__":
    main()
