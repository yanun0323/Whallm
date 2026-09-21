"""Single-request App benchmark. Caller holds ModelManager's request lock."""
from __future__ import annotations

import gzip
import hashlib
import time
from dataclasses import replace
from pathlib import Path

from .cancellation import check_cancelled
from .throughput_diagnostics import cache_counters, make_report, source_files_hash

CONTEXT_LENGTHS = (1024, 4096, 8192, 16384, 32768, 65536, 131072, 204800)
GENERATION_LENGTHS = (128, 512, 1024, 4096)
CONTEXT_TYPES = ("code", "novel")
TEMPERATURE = 0.0
SEED = 42
CORPUS_DIRECTORY = Path(__file__).with_name("benchmark_contexts")


def prompt_tokens(runtime, length, benchmark_context="code"):
    if benchmark_context not in CONTEXT_TYPES:
        raise ValueError("Choose Code or Novel as the benchmark context.")
    with gzip.open(CORPUS_DIRECTORY / f"{benchmark_context}.txt.gz", "rt", encoding="utf-8") as source:
        corpus = source.read()
    tokens = runtime._encode_prompt(corpus)
    if len(tokens) < length:
        raise ValueError(
            f"The bundled {benchmark_context} context contains only {len(tokens)} tokens "
            f"for this model; {length} were requested. Choose a shorter context length."
        )
    # Preserve the original sequence: no repetition or decode/re-encode rounding.
    return tokens[:length], hashlib.sha256(corpus.encode("utf-8")).hexdigest()


def run_trial(runtime, options, context_length, track, progress, benchmark_context="code"):
    # Benchmark controls are fixed even for direct callers. Do not mutate saved
    # model defaults or silently change the remaining sampling parameters.
    options = replace(options, temperature=TEMPERATURE, seed=SEED)
    if context_length + options.max_tokens > runtime.installed.maximum_context:
        raise ValueError("Input and generation lengths exceed this model's context limit.")
    tokens, corpus_hash = prompt_tokens(runtime, context_length, benchmark_context)
    check_cancelled()
    original_config = runtime.config
    cache = getattr(runtime, "expert_cache", None)
    before = cache_counters(cache)
    first = None
    initial_residents = getattr(cache, "resident_count", None)
    source_files_hash()  # File hashing is outside the timed generation interval.
    generated = 0
    finish = None
    output_hash = hashlib.sha256()
    try:
        # The manager lock excludes requests; always restore after any failure.
        runtime.config = replace(original_config, prompt_cache_entries=0,
                                 persistent_prompt_cache=False, dspark_prompt_cache=False)
        started = time.perf_counter()
        last_progress = started
        pieces = track(runtime.stream(tokens, options))
        try:
            for piece in pieces:
                check_cancelled()
                generated = piece.generation_tokens
                if first is None and generated >= 1:
                    first = cache_counters(cache)
                finish = piece.finish_reason
                output_hash.update(f"{piece.token}\n".encode("ascii"))
                now = time.perf_counter()
                if generated == 1 or now - last_progress >= 0.25:
                    progress(generated)
                    last_progress = now
        finally:
            pieces.close()
        elapsed = time.perf_counter() - started
        metrics = runtime.metrics.snapshot()
        decode_tps = metrics["decode_tokens_per_second"]
        diagnostics = make_report(runtime.config, before, first, cache_counters(cache),
                                  metrics, initial_residents, getattr(cache, "resident_count", None))
        return {
            "diagnostics": diagnostics,
            "slots": original_config.slots,
            "benchmark_context": benchmark_context,
            "corpus_sha256": corpus_hash,
            "context_tokens": context_length,
            "generation_tokens": generated,
            "generation_limit": options.max_tokens,
            "temperature": options.temperature,
            "seed": options.seed,
            "top_p": options.top_p,
            "top_k": options.top_k,
            "min_p": options.min_p,
            "presence_penalty": options.presence_penalty,
            "repetition_penalty": options.repetition_penalty,
            "ttft_ms": metrics["time_to_first_token_seconds"] * 1000,
            "tpot_ms": 1000 / decode_tps if decode_tps > 0 else None,
            "prefill_tps": metrics["prefill_tokens_per_second"],
            "decode_tps": decode_tps,
            "elapsed_seconds": elapsed,
            "throughput_tps": (context_length + generated) / elapsed if elapsed > 0 else 0,
            "finish_reason": finish,
            "output_token_sha256": output_hash.hexdigest(),
            "prompt_cache_reused_tokens": metrics["prompt_cache_reused_tokens"],
        }
    finally:
        runtime.config = original_config
