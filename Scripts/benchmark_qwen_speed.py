#!/usr/bin/env python3
"""Interleaved component A/B measurements, NOT whole-model tokens/second.

Requires Apple Silicon and the repository's pinned runtime. The baseline is an
immutable Git revision; no installed model, user prompt, network or OS tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', default='113be486422fa88deeeb96fbc29dc19ab7b27414')
    parser.add_argument('--rounds', type=int, default=15)
    parser.add_argument('--inner', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve prior evidence')
    if args.rounds < 3 or args.inner < 1 or not re.fullmatch(r'[0-9a-f]{40}', args.baseline):
        parser.error('require rounds >= 3, inner >= 1 and a full baseline commit SHA')
    import numpy as np
    import mlx.core as mx
    from deepseek_v4_ssd import qwen4_exp as candidate
    from deepseek_v4_ssd.expert_cache import QwenExpertWeights
    if not mx.metal.is_available():
        raise RuntimeError('Metal is required; CPU results cannot pass this benchmark')
    mx.set_default_device(mx.gpu)
    root = Path(__file__).resolve().parents[1]
    baseline_source = subprocess.check_output(
        ['git', 'show', args.baseline + ':runtime/deepseek_v4_ssd/qwen4_exp.py'], cwd=root)
    temporary = tempfile.TemporaryDirectory(prefix='qwen-speed-')
    source = Path(temporary.name) / 'baseline.py'
    source.write_bytes(baseline_source)
    spec = importlib.util.spec_from_file_location('deepseek_v4_ssd._speed_baseline', source)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    sha = lambda data: hashlib.sha256(data).hexdigest()
    report = dict(kind='component_microbenchmarks_not_model_throughput', status='running',
        baseline_commit=args.baseline, baseline_file_sha256=sha(baseline_source),
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        source_sha256={str(p.relative_to(root)): sha(p.read_bytes()) for p in sorted(
            (root / 'runtime/deepseek_v4_ssd').glob('qwen*.py'))},
        benchmark_sha256=sha(Path(__file__).read_bytes()),
        environment=dict(platform=platform.platform(), machine=platform.machine(),
            chip=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
            ram_bytes=int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True)),
            device=mx.metal.device_info(), python=sys.version,
            packages={n: importlib.metadata.version(n) for n in ('mlx', 'mlx-lm', 'numpy')},
            mlx_enable_tf32=os.environ.get('MLX_ENABLE_TF32')),
        methodology=dict(seed=20260921, rounds=args.rounds, inner=args.inner,
            warmup_calls=3, order='reverse order on alternating rounds',
            timer='perf_counter_ns around graph construction plus synchronous mx.eval',
            weights='synthetic random; expert payload resident; no SSD or full model',
            memory='separate untimed call; additional peak MLX allocated bytes above live inputs/weights',
            cache='warm inputs and kernels; no OS page-cache purge'), cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.output.write_text(json.dumps(report, indent=2) + '\n')

    def compare(name, shape, functions, *, host=False, exact=False, tolerance=.02):
        names = list(functions)
        outputs = {}
        for label, function in functions.items():
            out = function()
            if not host:
                mx.eval(out)
                outputs[label] = np.asarray(out.astype(mx.float32))
            else:
                outputs[label] = out
        base = outputs[names[0]]
        errors = {}
        for label, output in outputs.items():
            errors[label] = float(np.max(np.abs(output.astype(np.float64) - base.astype(np.float64)))) if base.size else 0
            if exact:
                np.testing.assert_array_equal(output, base)
            else:
                np.testing.assert_allclose(output, base, atol=tolerance, rtol=tolerance)
        hashes = {label: sha(output.tobytes()) for label, output in outputs.items()}
        del outputs, base, out

        def invoke(function):
            result = function()
            if not host:
                mx.eval(result)
            return result

        for function in functions.values():
            for _ in range(3):
                invoke(function)
        samples = {label: [] for label in names}
        for round_index in range(args.rounds):
            for label in names if round_index % 2 == 0 else names[::-1]:
                start = time.perf_counter_ns()
                for _ in range(args.inner):
                    invoke(functions[label])
                samples[label].append((time.perf_counter_ns() - start) / args.inner / 1e6)
        additional = {}
        if not host:
            for label, function in functions.items():
                mx.synchronize()
                mx.clear_cache()
                before = mx.get_active_memory()
                mx.reset_peak_memory()
                result = invoke(function)
                additional[label] = max(0, mx.get_peak_memory() - before)
                del result
        timings = {label: dict(p50_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                               min_ms=min(values), max_ms=max(values), samples_ms=values)
                   for label, values in samples.items()}
        row = dict(name=name, shape=shape, exact=exact, max_abs_errors=errors, output_float32_sha256=hashes,
                   timings=timings, additional_peak_mlx_bytes=additional)
        report['cases'].append(row)
        persist()
        print(name, {label: round(t['p50_ms'], 4) for label, t in timings.items()}, flush=True)

    try:
        rng = np.random.default_rng(20260921)
        descriptor = SimpleNamespace(head_vocab_sizes=tuple(range(2_500_001, 2_500_017)),
                                     head_offsets=tuple(2_500_100 * i for i in range(16)))
        multipliers = np.array([2**62 + 3, -2**61 + 9, 2**63 - 1], dtype=np.int64)
        for length in (3, 130, 8194):
            tokens = rng.integers(0, 248320, (1, length), dtype=np.int64)
            tokens[:, ::37] = 248044
            compare('ngram-hash', dict(tokens=length, includes_history=2), {
                'baseline': lambda: baseline.ngram_ids(tokens, multipliers, descriptor, eos_token_id=248044),
                'candidate': lambda: candidate.ngram_ids(tokens, multipliers, descriptor, eos_token_id=248044),
            }, host=True, exact=True)
        mx.random.seed(20260921)
        config = candidate.ModelArgs()
        old_attention = baseline.QSAAttention(baseline.ModelArgs())
        new_attention = candidate.QSAAttention(config)
        old_attention.set_dtype(mx.bfloat16)
        new_attention.set_dtype(mx.bfloat16)
        # Only indexer.k_layernorm is used by the bounded-attention component.
        new_attention.indexer.k_layernorm.weight = old_attention.indexer.k_layernorm.weight
        new_attention.sparse_sdpa = True
        for length, total in ((1, 128), (1, 2048), (128, 128), (128, 2048), (1, 8192), (128, 8192)):
            query = (mx.random.normal((1, 24, length, 256)) * .1).astype(mx.bfloat16)
            key = (mx.random.normal((1, 2, total, 256)) * .1).astype(mx.bfloat16)
            value = mx.random.normal(key.shape).astype(mx.bfloat16) * .1
            index = mx.random.normal((1, length, 4, 128)).astype(mx.bfloat16)
            raw = mx.random.normal((1, total, 128)).astype(mx.bfloat16)
            mx.eval(query, key, value, index, raw)
            def original(fused):
                old_attention.sparse_sdpa = fused
                return old_attention._bounded_attention(query, key, value, index, raw, total - length)
            compare('qsa', dict(query_tokens=length, kv_tokens=total, q_heads=24, kv_heads=2, head_dim=256, dtype='bfloat16'), {
                'baseline-manual': lambda: original(False),
                'previous-fused': lambda: original(True),
                'candidate': lambda: new_attention._bounded_attention(query, key, value, index, raw, total - length),
            }, tolerance=.02)
        del old_attention, new_attention, query, key, value, index, raw
        mx.clear_cache()

        old_hc = baseline.GatedResidual(baseline.ModelArgs())
        new_hc = candidate.GatedResidual(config)
        old_hc.set_dtype(mx.bfloat16)
        new_hc.set_dtype(mx.bfloat16)
        from mlx.utils import tree_flatten
        new_hc.load_weights(tree_flatten(old_hc.parameters()))
        old_hc.hc_norm.compiled = True  # Isolate improvements beyond existing norm compilation.
        new_hc.hc_norm.compiled = True
        new_hc.compiled = True
        mx.eval(old_hc.parameters(), new_hc.parameters())
        for length in (1, 3, 128):
            hidden = (mx.random.normal((1, length, 10240)) * .1).astype(mx.bfloat16)
            mx.eval(hidden)
            def old_forward():
                mixed, residual, injection = old_hc(hidden)
                return residual + (mixed[..., None, :] * injection[..., None]).reshape(residual.shape)
            def new_forward():
                mixed, residual, injection = new_hc(hidden)
                return new_hc.inject(residual, mixed, injection)
            compare('hyper-connection', dict(tokens=length, hidden=2560, streams=4, lowrank=320, dtype='bfloat16'),
                    {'previous-compiled-norm': old_forward, 'candidate': new_forward}, tolerance=.02)
        del old_hc, new_hc, hidden
        mx.clear_cache()

        gate, gs = mx.quantize((mx.random.normal((10, 1280, 2560)) * .02).astype(mx.bfloat16),
                               group_size=32, bits=4, mode='mxfp4')
        down, ds = mx.quantize((mx.random.normal((10, 2560, 640)) * .02).astype(mx.bfloat16),
                               group_size=32, bits=4, mode='mxfp4')
        mx.eval(gate, gs, down, ds)
        payloads = [QwenExpertWeights(gate[i], gs[i], down[i], ds[i]) for i in range(10)]
        cache = SimpleNamespace(slots=10, current_batched=lambda _: None,
            get_many=lambda layer, ids: SimpleNamespace(individual_weights=payloads, slots={i: i for i in ids}))
        old_experts, new_experts = baseline.StreamingExperts(0, cache), candidate.StreamingExperts(0, cache)
        hidden = mx.random.normal((1, 1, 2560)).astype(mx.bfloat16)
        indices = mx.array([[[8, 3, 5, 9, 0, 6, 1, 7, 4, 2]]])
        mx.eval(hidden, indices)
        compare('resident-expert-singleton', dict(tokens=1, experts=10, hidden=2560, intermediate=640, quantization='mxfp4', disk_io=False),
                {'baseline': lambda: old_experts(hidden, indices), 'candidate': lambda: new_experts(hidden, indices)}, exact=True)
        report['status'] = 'completed'
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        persist()
        temporary.cleanup()


if __name__ == '__main__':
    main()
