#!/usr/bin/env python3
"""Full installed Qwen profiling, offline. Never edits App settings or model bytes.

Run separately with --mode baseline, host, sync. Normal timing is baseline only.
All modes sample system GPU utilization (not per-process/kernel utilization).
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import plistlib
import signal
import subprocess
import sys
import tarfile
import threading
import time

import mlx.core as mx
from deepseek_v4_ssd.generation import ModelRuntime, GenerationOptions
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.qwen_profile import QwenProfile, process_snapshot
from deepseek_v4_ssd.throughput import prompt_tokens
from deepseek_v4_ssd.throughput_diagnostics import cache_counters, difference, source_files_hash


def token_hash(tokens):
    return hashlib.sha256("".join(f"{x}\n" for x in tokens).encode()).hexdigest()


def system_snapshot():
    result = {}
    try:
        devices = plistlib.loads(subprocess.check_output(
            ["ioreg", "-a", "-r", "-c", "AGXAccelerator", "-d", "1"], timeout=3))
        result["gpu_system"] = [d.get("PerformanceStatistics", {}) for d in devices]
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        result["gpu_error"] = str(error)
    result["swap"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True, timeout=3).strip()
    result["vm_stat"] = subprocess.check_output(["vm_stat"], text=True, timeout=3)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "host", "sync"), default="baseline")
    parser.add_argument("--config", type=Path, help="RuntimeConfig JSON overrides")
    parser.add_argument('--timeline', action='store_true', help='Write Mach-clock phase/decoder intervals for external Metal analysis (host/sync only)')
    parser.add_argument("--context", choices=("code", "novel"), default="code")
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--max-footprint-gib", type=float, default=48)
    args = parser.parse_args()
    if min(args.input_tokens, args.max_tokens, args.requests, args.max_seconds, args.max_footprint_gib) <= 0:
        parser.error("limits must be positive")
    if args.timeline and args.mode == 'baseline':
        parser.error('--timeline requires host or sync mode')
    args.output.mkdir(parents=True, exist_ok=False)
    # Current App-style exact compute configuration; pinned explicitly, not
    # theoretical recommended profiles. Prompt Cache is intentionally disabled.
    config = dict(slots=3072, read_workers=16, expert_eviction_policy="lru",
        ready_expert_decode=True, layer_major_prefill=True, prefill_step_size=1024,
        memory_limit_gib=30, batched_expert_prefill=False, qwen_next_layer_prefetch=False,
        qwen_grouped_experts=True, qwen_pooled_index_cache=True,
        qwen_ngram_lookup_optimized=True, qwen_compile_tensor_ops=True, qwen_phase_memory=True,
        ane_prefill=False, prompt_cache_entries=0, persistent_prompt_cache=False)
    if args.config:
        config.update(json.loads(args.config.read_text()))
    config = RuntimeConfig(**config)
    if config.mtp_enabled or config.dspark_enabled or config.prompt_cache_entries or config.persistent_prompt_cache:
        parser.error("this main-model profiler requires MTP/DSpark and Prompt Cache off")
    profile = None if args.mode == "baseline" else QwenProfile(args.mode,
        timeline_path=args.output / 'timeline.jsonl' if args.timeline else None)
    root = Path(__file__).resolve().parents[1]
    with tarfile.open(args.output / "source.tar.gz", "w:gz") as archive:
        for path in sorted((root / "runtime/deepseek_v4_ssd").rglob("*")):
            if path.suffix in {".py", ".metal"}:
                archive.add(path, arcname=str(path.relative_to(root)))
        archive.add(Path(__file__), arcname="Scripts/profile_qwen_runtime.py")
    start = time.perf_counter()
    phase = "load"
    request = -1
    stop = threading.Event()
    runtime = None
    report = dict(schema_version=2, status="running", mode=args.mode, pid=os.getpid(), timeline=args.timeline,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], text=True),
        source_files_sha256=source_files_hash(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((args.model / "manifest.json").read_bytes()).hexdigest(),
        config=asdict(config), workload=dict(context=args.context, input_tokens=args.input_tokens,
            max_tokens=args.max_tokens, requests=args.requests, temperature=0, seed=42),
        environment=dict(platform=platform.platform(), python=sys.version, device=mx.device_info(),
            packages={n: importlib.metadata.version(n) for n in ("mlx", "mlx-lm", "numpy", "transformers")}),
        cache_state="New process and empty expert slots initially; later requests retain slots. Prompt Cache off. OS page cache uncontrolled, not purged; NOT cold SSD.",
        measurement_limits="Baseline has a 0.5s nominal system sampler but no model hooks. Host spans do not force evaluation. Sync spans perturb overlap and are not normal throughput or GPU kernel timings. GPU utilization is device-wide and its driver averaging window is unspecified. Process disk accounting is not device SSD traffic. Phase boundary is first yielded token; pipelined work can straddle it. MLX/RSS/footprint overlap and must not be added. Tokenization covers the whole bundled corpus before slicing. No HTTP/UI timing, GPU bandwidth/power or per-kernel occupancy is measured.",
        stops=dict(seconds=args.max_seconds, footprint_gib=args.max_footprint_gib, swap_growth_mib=256),
        requests=[], phases=[], system_before=system_snapshot())
    initial_swap = float(report["system_before"]["swap"].split("used = ")[1].split("M")[0])
    (args.output / "protocol.json").write_text(json.dumps(report, indent=2) + "\n")
    events = (args.output / "samples.jsonl").open("w", buffering=1)

    def save():
        if profile:
            report["profile"] = profile.report()
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")

    def monitor():
        while not stop.wait(.5):
            try:
                current_phase, current_request = phase, request
                sample = dict(seconds=time.perf_counter() - start, phase=current_phase, request=current_request,
                    process=process_snapshot(), **system_snapshot())
                events.write(json.dumps(sample) + "\n")
                footprint = sample["process"].get("physical_footprint", 0)
                used_swap = float(sample["swap"].split("used = ")[1].split("M")[0])
                reason = ("timeout" if sample["seconds"] > args.max_seconds else
                          "footprint" if footprint > args.max_footprint_gib * 2**30 else
                          "swap_growth" if used_swap - initial_swap > 256 else None)
                if reason:
                    (args.output / "STOP.json").write_text(json.dumps(dict(reason=reason, sample=sample)))
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
            except Exception as error:
                events.write(json.dumps(dict(monitor_error=repr(error))) + "\n")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def snapshot():
        return dict(process=process_snapshot(), expert=cache_counters(runtime.expert_cache) if runtime else None,
            mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
            mlx_peak_bytes=mx.get_peak_memory())

    def mark_phase(name):
        nonlocal phase
        phase = name
        if profile:
            profile.begin_phase(name, request)
        return snapshot(), time.perf_counter()

    def finish_phase(name, before, since):
        after = snapshot()
        elapsed = time.perf_counter() - since
        if profile:
            profile.end_phase()
        report["phases"].append(dict(request=request, phase=name, seconds=elapsed, before=before, after=after,
            process_delta=difference(after["process"], before["process"]),
            expert_delta=difference(after["expert"], before["expert"])))
        print(name, f"{elapsed:.3f}s", flush=True)
        return elapsed

    try:
        if profile:
            from deepseek_v4_ssd import model as model_module
            from deepseek_v4_ssd import qwen4_exp
            profile.hook(model_module, "_load_common_weights", name="common_weights", evaluate=False)
            profile.hook(qwen4_exp, "load", name="construct_and_load", evaluate=False)
        before, since = mark_phase("load")
        with profile.span("load") if profile else nullcontext():
            runtime = ModelRuntime.open(str(args.model), config)
        finish_phase("load", before, since)
        if profile:
            profile.attach(runtime)
        before, since = mark_phase("tokenize")
        ids, corpus_hash = prompt_tokens(runtime, args.input_tokens, args.context)
        finish_phase("tokenize", before, since)
        report.update(input_token_sha256=token_hash(ids), corpus_sha256=corpus_hash)
        (args.output / "input_tokens.json").write_text(json.dumps(ids))
        for request in range(args.requests):
            mx.reset_peak_memory()
            tokens = []
            before, since = mark_phase("prefill_to_first_yield")
            request_start = since
            iterator = runtime.stream(ids, GenerationOptions(max_tokens=args.max_tokens,
                temperature=0, top_p=1, top_k=0, seed=42))
            try:
                for piece in iterator:
                    if not tokens:
                        first_seconds = finish_phase("prefill_to_first_yield", before, since)
                        mx.reset_peak_memory()
                        before, since = mark_phase("decode_after_first_yield")
                    tokens.append(int(piece.token))
                if tokens:
                    finish_phase("decode_after_first_yield", before, since)
            finally:
                iterator.close()
            report["requests"].append(dict(index=request, elapsed_seconds=time.perf_counter() - request_start,
                first_yield_seconds=first_seconds if tokens else None, output_tokens=tokens,
                output_token_sha256=token_hash(tokens), generation_tokens=len(tokens),
                metrics=runtime.metrics.snapshot(), ngram_io=runtime.model.ngram_store.io_snapshot()))
            save()
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        stop.set()
        thread.join(timeout=10)
        if profile:
            profile.close()
        report["system_after"] = system_snapshot()
        report["seconds"] = time.perf_counter() - start
        save()
        if runtime:
            runtime.close()
        events.close()


if __name__ == "__main__":
    main()
