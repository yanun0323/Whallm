"""Consume real Swift-generated catalogs with the Python parser; no model loading.

Since 2026-10-09 both Qwen models hide their tuning controls, so the App emits one
fixed Qwen configuration and neutral values for other models.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from deepseek_v4_ssd.model_manager import load_model_catalog
from deepseek_v4_ssd.qwen_flash_config import validate_flash_config

FIXED_QWEN = dict(qwen_expert_wave_slots=0, qwen_ngram_io="mmap", qwen_ngram_cache_bytes=0,
    qwen_qsa_query_chunk=16, qwen_qsa_skip_complete_gather=True, qwen_prefill_read_experts=1,
    qwen_prefill_seed_experts=0, qwen_packed_gdn_prefill=False, qwen_sorted_expert_prefill=True,
    qwen_qsa_masked_prefill=True, mtp_enabled=True)
NEUTRAL = dict(FIXED_QWEN, qwen_sorted_expert_prefill=False, qwen_qsa_masked_prefill=False, mtp_enabled=False)


def validate(directory: Path) -> int:
    cases = {"qwen-fixed": FIXED_QWEN, "swift-fixed": FIXED_QWEN,
             "deepseek-v4": NEUTRAL, "deepseek-v41": NEUTRAL}
    for name, expected in cases.items():
        specs = load_model_catalog(directory / f"{name}.json")
        if len(specs) != 1:
            raise ValueError(f"{name}: expected exactly one model")
        runtime = specs[0].runtime
        validate_flash_config(runtime)
        actual = {field: getattr(runtime, field) for field in expected}
        if actual != expected:
            raise ValueError(f"{name}: {actual!r} != {expected!r}")
        print(f"PASS {name}")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(f"Validated {validate(args.directory)} Swift-to-Python catalogs; no model was loaded.")


if __name__ == "__main__":
    main()
