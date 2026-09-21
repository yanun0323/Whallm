#!/usr/bin/env python3
"""Single-process Qwen correctness pilot. Not a formal performance benchmark.

Run each variant in a fresh process and compare output token hashes. Prefill
controls are off as in new App settings; other fixed pilot settings include
LFU and 32 MTP slots (not a claim of matching every App default). No saved App
settings, model files, OS page cache, or expert-byte ceiling are changed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "pooled", "ngram", "compiled", "mtp5", "mtp2", "mtp2-retry", "combined", "phase", "phase-combined", "waves", "pread", "sparse-sdpa", "flash-combined", "qsa-chunk16", "qsa-chunk32", "qsa-indexed", "qsa-throughput"), required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--lifecycle", action="store_true", help="Also check continuation, cancellation, warmup and another request")
    parser.add_argument("--layer-major", action="store_true", help="Exercise layer-major batched prefill instead of default-off controls")
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.output.exists():
        parser.error("output already exists; preserve prior evidence")
    from deepseek_v4_ssd.app_memory import AppMemorySampler
    from deepseek_v4_ssd.generation import ModelRuntime, GenerationOptions
    from deepseek_v4_ssd.model import RuntimeConfig
    from deepseek_v4_ssd.model_manager import validate_runtime_config

    root = Path(__file__).resolve().parents[1]
    sha = lambda data: hashlib.sha256(data).hexdigest()
    config = RuntimeConfig(expert_cache_bytes=int(7.5 * 1024**3), memory_limit_gib=40,
        prompt_cache_entries=0, persistent_prompt_cache=False, layer_major_prefill=False,
        batched_expert_prefill=False, ready_expert_decode=False, ane_prefill=False)
    overrides = {
        "qsa-chunk16": {"qwen_sparse_sdpa": True, "qwen_qsa_query_chunk": 16},
        "qsa-chunk32": {"qwen_sparse_sdpa": True, "qwen_qsa_query_chunk": 32},
        "qsa-indexed": {"qwen_sparse_sdpa": True, "qwen_qsa_indexed": True},
        "qsa-throughput": {"qwen_sparse_sdpa": True, "qwen_qsa_query_chunk": 32,
                           "qwen_qsa_indexed": True},
        "waves": {"qwen_expert_wave_slots": 32},
        "pread": {"qwen_ngram_io": "pread", "qwen_ngram_cache_bytes": 64 * 1024**2},
        "sparse-sdpa": {"qwen_sparse_sdpa": True},
        "flash-combined": {"qwen_expert_wave_slots": 32, "qwen_ngram_io": "pread",
                           "qwen_ngram_cache_bytes": 64 * 1024**2, "qwen_sparse_sdpa": True},
        "phase": {"qwen_phase_memory": True},
        "phase-combined": {"qwen_phase_memory": True, "qwen_pooled_index_cache": True,
                           "qwen_ngram_lookup_optimized": True, "qwen_compile_tensor_ops": True},
        "baseline": {}, "pooled": {"qwen_pooled_index_cache": True},
        "ngram": {"qwen_ngram_lookup_optimized": True},
        "compiled": {"qwen_compile_tensor_ops": True},
        "mtp5": {"mtp_enabled": True},
        "mtp2": {"mtp_enabled": True, "qwen_mtp_draft_tokens": 2},
        "mtp2-retry": {"mtp_enabled": True, "qwen_mtp_draft_tokens": 2, "qwen_mtp_zero_acceptance_limit": 2},
        "combined": {"qwen_pooled_index_cache": True, "qwen_ngram_lookup_optimized": True,
                     "qwen_compile_tensor_ops": True},
    }
    config = replace(config, **overrides[args.variant])
    if args.lifecycle:
        config = replace(config, prompt_cache_entries=2)
    if args.layer_major:
        config = replace(config, layer_major_prefill=True, layer_major_prefill_threshold=1,
                         batched_expert_prefill=True, prefill_step_size=32)
    validate_runtime_config(config)
    prompt = args.prompt.read_text()
    sysctl = lambda name: subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    import deepseek_v4_ssd
    runtime_root = Path(deepseek_v4_ssd.__file__).resolve().parent
    source_hashes = {"runtime/deepseek_v4_ssd/" + str(path.relative_to(runtime_root)): sha(path.read_bytes())
                     for path in sorted(runtime_root.rglob("*.py"))}
    source_hashes[str(Path(__file__).resolve().relative_to(root))] = sha(Path(__file__).read_bytes())
    result = dict(status="running", formal_performance_result=False, variant=args.variant,
        started_at=datetime.now(timezone.utc).isoformat(), config=asdict(config),
        source=dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            imported_runtime_root=str(runtime_root), files_sha256=source_hashes),
        environment=dict(platform=platform.platform(), python=sys.version, chip=sysctl("machdep.cpu.brand_string"),
            memory_bytes=int(sysctl("hw.memsize")), swap_before=sysctl("vm.swapusage"),
            packages={name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "numpy")}),
        model=dict(path=str(args.model.resolve()), manifest_sha256=sha((args.model / "manifest.json").read_bytes())),
        workload=dict(prompt=prompt, prompt_sha256=sha(prompt.encode()), max_tokens=args.max_tokens, temperature=0),
        cache_state="fresh process; initially empty expert/KV/prompt caches; OS page cache uncontrolled and not purged",
        lifecycle_enabled=args.lifecycle, layer_major=args.layer_major,
        limits=["Short greedy correctness pilot; not held-out, stochastic or formal speed validation.",
                "Physical-footprint sampler: this Python process only, nominal 10 ms, loading through generation, excludes unload.",
                "MLX memory ceiling 40 GiB; expert budget 7.5 GiB. Neither is a total physical-footprint guarantee."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    runtime = None
    sampler = AppMemorySampler(pids=(os.getpid(),))
    try:
        with sampler:
            runtime = ModelRuntime.open(args.model, config)
            encoded = runtime._encode_prompt(prompt)
            result["workload"]["prompt_tokens"] = len(encoded)
            result["workload"]["prompt_token_sha256"] = sha(",".join(map(str, encoded)).encode())
            result["effective_config"] = asdict(runtime.config)
            tokens = []
            started = time.perf_counter()
            for piece in runtime.stream(prompt, GenerationOptions(max_tokens=args.max_tokens, temperature=0)):
                tokens.append(int(piece.token))
            result["request_seconds"] = time.perf_counter() - started
            result["generated_token_ids"] = tokens
            result["token_sha256"] = sha(",".join(map(str, tokens)).encode())
            result["metrics"] = runtime.metrics.snapshot()
            result["ngram_io"] = runtime.model.ngram_store.io_snapshot()
            result["expert_waves"] = [dict(layer=i, waves=layer.mlp.experts.wave_count,
                pairs=int(layer.mlp.experts.wave_pairs), peak_experts=layer.mlp.experts.wave_peak_experts)
                for i, layer in enumerate(runtime.model.model.layers)]
            phase_snapshot = getattr(runtime.expert_cache, "phase_memory_snapshot", None)
            result["phase_memory"] = phase_snapshot() if phase_snapshot is not None else None
            if args.lifecycle:
                from threading import Event
                from deepseek_v4_ssd.cancellation import cancellation_scope, GenerationCancelled
                ceiling = runtime.expert_cache.slots
                lifecycle = []
                result["lifecycle"] = lifecycle
                def record(name, ids):
                    state = runtime.expert_cache.phase_memory_snapshot()
                    if state["active_slots"] != ceiling:
                        raise AssertionError("expert capacity not restored at request boundary")
                    lifecycle.append(dict(name=name, generated_token_ids=ids,
                        token_sha256=sha(",".join(map(str, ids)).encode()), phase_memory=state,
                        reused_tokens=runtime.metrics.snapshot().get("prompt_cache_reused_tokens")))
                continuation = encoded + tokens + [198]
                continued = [int(piece.token) for piece in runtime.stream(continuation,
                    GenerationOptions(max_tokens=16, temperature=0))]
                record("continuation", continued)
                event = Event()
                with cancellation_scope(event):
                    responses = runtime.stream(prompt + "\\nExplain one more detail.",
                        GenerationOptions(max_tokens=16, temperature=0))
                    try:
                        first = int(next(responses).token)
                        event.set()
                        try:
                            next(responses)
                        except GenerationCancelled:
                            pass
                        else:
                            raise AssertionError("cancellation was not observed")
                    finally:
                        responses.close()
                record("cancelled_after_first_token", [first])
                warm_tokens = runtime.warm_prompt(prompt)
                record("warmup", [])
                lifecycle[-1]["warmed_tokens"] = warm_tokens
                after = [int(piece.token) for piece in runtime.stream(prompt,
                    GenerationOptions(max_tokens=16, temperature=0))]
                record("after_cancel_and_warmup", after)
                result["lifecycle"] = lifecycle
            result["status"] = "completed"
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result["peak_process_physical_footprint_bytes"] = sampler.peak_bytes
        result["environment"]["swap_after"] = sysctl("vm.swapusage")
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    main()
