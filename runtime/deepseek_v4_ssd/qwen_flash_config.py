"""Validation and shared CLI switches for opt-in Qwen Flash experiments."""
from __future__ import annotations

import argparse


DEFAULTS = {
    "qwen_prefill_read_experts": 1,
    "qwen_prefill_seed_experts": 0,
    "qwen_shared_expert_overlap": False,
    "qwen_expert_wave_slots": 0,
    "qwen_ngram_io": "mmap",
    "qwen_ngram_cache_bytes": 0,
    "qwen_sparse_sdpa": False,
    "qwen_qsa_query_chunk": 4,
    "qwen_qsa_indexed": False,
}


def validate_flash_config(config) -> None:
    for name, minimum, maximum in (("qwen_prefill_read_experts", 1, 32),
                                   ("qwen_prefill_seed_experts", 0, 128)):
        value = getattr(config, name, DEFAULTS[name])
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    if type(getattr(config, "qwen_shared_expert_overlap", False)) is not bool:
        raise ValueError("qwen_shared_expert_overlap must be a boolean")
    if (getattr(config, "qwen_prefill_read_experts", 1) != 1
            or getattr(config, "qwen_prefill_seed_experts", 0)):
        if (not getattr(config, "layer_major_prefill", True)
                or not getattr(config, "batched_expert_prefill", True)
                or getattr(config, "qwen_expert_wave_slots", 0)):
            raise ValueError("Qwen prefill read/seed experiments require whole-layer prefill without waves")
    for name, maximum in (("qwen_expert_wave_slots", 512),
                          ("qwen_ngram_cache_bytes", 512 * 1024**2)):
        value = getattr(config, name, DEFAULTS[name])
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError(f"{name} must be an integer from 0 through {maximum}")
    chunk = getattr(config, "qwen_qsa_query_chunk", 4)
    if type(chunk) is not int or not 1 <= chunk <= 128:
        raise ValueError("qwen_qsa_query_chunk must be an integer from 1 through 128")
    indexed = getattr(config, "qwen_qsa_indexed", False)
    if type(indexed) is not bool:
        raise ValueError("qwen_qsa_indexed must be a boolean")
    if indexed and not getattr(config, "qwen_sparse_sdpa", False):
        raise ValueError("qwen_qsa_indexed requires qwen_sparse_sdpa")
    backend = getattr(config, "qwen_ngram_io", "mmap")
    if not isinstance(backend, str) or backend not in ("mmap", "pread"):
        raise ValueError("qwen_ngram_io must be mmap or pread")
    if type(getattr(config, "qwen_sparse_sdpa", False)) is not bool:
        raise ValueError("qwen_sparse_sdpa must be a boolean")
    if getattr(config, "qwen_ngram_cache_bytes", 0) and backend != "pread":
        raise ValueError("qwen_ngram_cache_bytes requires qwen_ngram_io=pread")
    if (getattr(config, "qwen_expert_wave_slots", 0)
            and getattr(config, "qwen_next_layer_prefetch", False)):
        raise ValueError("qwen_expert_wave_slots cannot use whole-layer next-layer prefetch")


def add_flash_arguments(parser: argparse.ArgumentParser) -> None:
    _add_qsa_arguments(parser)
    parser.add_argument("--qwen-prefill-read-experts", type=int, default=1,
                        help="1..32 consecutive experts per prefill read; 1 keeps old I/O")
    parser.add_argument("--qwen-prefill-seed-experts", type=int, default=0,
                        help="0..128 hot prompt-tail experts per layer retained in existing decode slots")
    parser.add_argument("--qwen-shared-expert-overlap", action=argparse.BooleanOptionalAction,
                        default=False, help="submit singleton shared-expert work before routed-expert I/O")
    parser.add_argument("--qwen-expert-wave-slots", type=int, default=0,
                        help="experimental exact expert waves; 1..512 experts per wave, 0 disables")
    parser.add_argument("--qwen-ngram-io", choices=("mmap", "pread"), default="mmap",
                        help="experimental positioned N-gram reads instead of mmap")
    parser.add_argument("--qwen-ngram-cache-bytes", type=int, default=0,
                        help="packed N-gram row LRU payload limit; requires pread; max 512 MiB")
    parser.add_argument("--qwen-sparse-sdpa", action=argparse.BooleanOptionalAction, default=False,
                        help="experimental fused attention on unchanged QSA selected cells")


def _add_qsa_arguments(parser):
    parser.add_argument("--qwen-qsa-query-chunk", type=int, default=4,
                        help="QSA queries per chunk, 1..128; bounded estimated workspace")
    parser.add_argument("--qwen-qsa-indexed", action=argparse.BooleanOptionalAction,
                        default=False, help="experimental indexed Metal QSA decode; requires sparse SDPA")


def flash_arguments(arguments: argparse.Namespace) -> dict:
    return {name: getattr(arguments, name) for name in DEFAULTS}
