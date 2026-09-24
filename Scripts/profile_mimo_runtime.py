#!/usr/bin/env python3
"""Offline MiMo layer diagnostics and normal timing. Never edits App/model state.

PYTHONPATH=runtime:. .venv/bin/python Scripts/profile_mimo_runtime.py \
  --model ~/.dsmodel/mimo-v2.6-flash-rl.dsv4 --output scratch/mimo-host --mode host

Use baseline for performance comparisons; host/sync are observers, not normal
throughput. Sync spans include CPU and I/O, not just GPU kernel execution.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tarfile
import threading
import time

import mlx.core as mx
from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
from deepseek_v4_ssd.mimo.profile import MiMoProfile, layer_metrics
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.qwen_profile import process_snapshot
from deepseek_v4_ssd.throughput import prompt_tokens, run_trial
from deepseek_v4_ssd.throughput_diagnostics import source_files_hash
from Scripts.profile_qwen_runtime import system_snapshot, token_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('baseline', 'host', 'sync'), default='baseline')
    p.add_argument('--config', type=Path, help='RuntimeConfig JSON overrides')
    p.add_argument('--layer-major', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--timeline', action='store_true', help='Mach-clock phase/decoder intervals (host/sync only)')
    p.add_argument('--context', choices=('code', 'novel'), default='code')
    p.add_argument('--input-tokens', type=int, default=1024)
    p.add_argument('--max-tokens', type=int, default=32)
    p.add_argument('--requests', type=int, default=1)
    p.add_argument('--max-seconds', type=int, default=900)
    p.add_argument('--max-footprint-gib', type=float, default=40)
    args = p.parse_args()
    if min(args.input_tokens, args.max_tokens, args.requests, args.max_seconds, args.max_footprint_gib) <= 0:
        p.error('limits must be positive')
    if args.timeline and args.mode == 'baseline':
        p.error('--timeline requires host or sync')
    config = dict(slots=771, expert_cache_bytes=10307921510, read_workers=4, prefetch_read_workers=2,
        prefill_step_size=0, fp8_kv_cache=False, layer_major_prefill=args.layer_major,
        batched_expert_prefill=False, ane_prefill=False, v41_layer_major_prefill=False,
        v41_next_layer_prefetch=False, qwen_grouped_experts=False, expert_eviction_policy='lru',
        ready_expert_decode=True, separate_prefill_io=True, prompt_cache_entries=0,
        persistent_prompt_cache=False)
    if args.config:
        config.update(json.loads(args.config.read_text()))
    config = RuntimeConfig(**config)
    if config.mtp_enabled or config.dspark_enabled or config.prompt_cache_entries or config.persistent_prompt_cache:
        p.error('this main-model diagnostic requires speculation and Prompt Cache off')
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    with tarfile.open(args.output/'source.tar.gz', 'w:gz') as tar:
        for path in sorted((root/'runtime/deepseek_v4_ssd').rglob('*')):
            if path.suffix in {'.py', '.metal'}:
                tar.add(path, arcname=str(path.relative_to(root)))
        for name in ('Scripts/profile_mimo_runtime.py', 'Scripts/profile_qwen_runtime.py',
                     'Sources/DeepSeekRepack/Resources/ModelPackages.json'):
            tar.add(root/name, arcname=name)
    profile = None if args.mode == 'baseline' else MiMoProfile(args.mode,
        timeline_path=args.output/'timeline.jsonl' if args.timeline else None)
    report = dict(schema_version=1, status='running', mode=args.mode, pid=os.getpid(),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_files_sha256=source_files_hash(),
        catalog_sha256=hashlib.sha256((root/'Sources/DeepSeekRepack/Resources/ModelPackages.json').read_bytes()).hexdigest(),
        manifest_sha256=hashlib.sha256((args.model/'manifest.json').read_bytes()).hexdigest(),
        config=asdict(config), workload=dict(context=args.context, input_tokens=args.input_tokens,
            max_tokens=args.max_tokens, requests=args.requests, seed=42, temperature=0, top_p=.95),
        environment=dict(platform=platform.platform(), python=sys.version, device=mx.device_info(),
            packages={n:importlib.metadata.version(n) for n in ('mlx','mlx-lm','numpy','transformers')}),
        cache_state='New process/empty expert slots initially; later requests retain slots. Prompt Cache off. OS cache not purged, not cold SSD.',
        notes='First yielded token phase boundary; pipelined work can straddle it. Observer results are not normal throughput. Nested spans/counters must not be summed. Sync wall time is not GPU kernel time. Disk accounting is not physical SSD traffic. Memory/GPU samples are process-wide/device-wide, not per-layer ownership.',
        stops=dict(seconds=args.max_seconds, footprint_gib=args.max_footprint_gib, swap_growth_mib=256),
        system_before=system_snapshot(), requests=[])
    start = time.monotonic()
    phase, request = 'load', -1
    stop = threading.Event()
    runtime = None

    def save():
        if profile:
            report['profile'] = profile.report()
            rows = layer_metrics(report['profile'])
            if rows:
                with (args.output/'layers.csv').open('w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')

    save()
    initial_swap = float(report['system_before']['swap'].split('used = ')[1].split('M')[0])

    def monitor():
        with (args.output/'samples.jsonl').open('w', buffering=1) as f:
            while not stop.wait(1):
                try:
                    sample = dict(seconds=time.monotonic()-start, phase=phase, request=request,
                                  process=process_snapshot(), **system_snapshot())
                    f.write(json.dumps(sample)+'\n')
                    swap = float(sample['swap'].split('used = ')[1].split('M')[0])
                    reason = ('timeout' if sample['seconds'] > args.max_seconds else
                        'footprint' if sample['process'].get('physical_footprint', 0) > args.max_footprint_gib*2**30 else
                        'swap_growth' if swap-initial_swap > 256 else None)
                    if reason:
                        (args.output/'STOP.json').write_text(json.dumps(dict(reason=reason, sample=sample)))
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
                except Exception as error:
                    f.write(json.dumps(dict(monitor_error=repr(error)))+'\n')

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        runtime = ModelRuntime.open(str(args.model), config)
        if runtime.installed.model_kind != 'mimo-v2.6-flash-rl':
            raise ValueError('MiMo installed model required')
        report['load_seconds'] = time.monotonic()-start
        phase = 'tokenize'
        ids, corpus_hash = prompt_tokens(runtime, args.input_tokens, args.context)
        report.update(input_token_sha256=token_hash(ids), corpus_sha256=corpus_hash)
        (args.output/'input_tokens.json').write_text(json.dumps(ids)+'\n')
        if profile:
            profile.attach(runtime)
        for request in range(args.requests):
            outputs = []
            mx.reset_peak_memory()

            def track(pieces):
                nonlocal phase
                phase = 'prefill'
                if profile:
                    profile.begin_phase(phase, request)
                completed = False
                try:
                    for piece in pieces:
                        if not outputs:
                            if profile:
                                profile.end_phase()
                            phase = 'decode'
                            if profile:
                                profile.begin_phase(phase, request)
                        outputs.append(int(piece.token))
                        yield piece
                    completed = True
                finally:
                    pieces.close()
                    if profile:
                        profile.end_phase(failed=not completed)

            trial = run_trial(runtime, GenerationOptions(max_tokens=args.max_tokens,
                temperature=0, top_p=.95, top_k=0, seed=42), args.input_tokens, track, lambda n:None, args.context)
            trial.update(output_tokens=outputs, metrics=runtime.metrics.snapshot(),
                         mlx_peak_bytes=mx.get_peak_memory(), system_after=system_snapshot())
            report['requests'].append(trial)
            phase = 'idle'
            save()
            print(json.dumps({k:trial[k] for k in ('ttft_ms','decode_tps','elapsed_seconds','output_token_sha256')}), flush=True)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        stop.set()
        thread.join(10)
        if profile:
            profile.close()
        report.update(system_after=system_snapshot(), seconds=time.monotonic()-start)
        save()
        if runtime:
            runtime.close()


if __name__ == '__main__':
    main()
