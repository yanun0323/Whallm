"""Interleaved QSA component benchmark; no checkpoint, SSD I/O or model tok/s."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
import types
from importlib.metadata import version

import mlx.core as mx
import numpy as np
from deepseek_v4_ssd import qwen4_exp as current

BASELINE = "0b1cb7a46021b446195716ceb64d9fc1cced4cf4"


def digest(value):
    return hashlib.sha256(np.asarray(value.view(mx.uint8)).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=9)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.rounds < 3:
        raise ValueError('At least three rounds are required')
    if not mx.metal.is_available():
        raise RuntimeError('This benchmark requires Metal, not a CPU substitute')
    mx.set_default_device(mx.gpu)
    source = subprocess.check_output(['git','show',f'{BASELINE}:runtime/deepseek_v4_ssd/qwen4_exp.py'], text=True)
    old = types.ModuleType('deepseek_v4_ssd._qsa_before')
    old.__package__ = 'deepseek_v4_ssd';sys.modules[old.__name__] = old
    exec(compile(source, f'{BASELINE}/qwen4_exp.py', 'exec'), old.__dict__)
    model_args = current.ModelArgs()
    baseline = old.QSAAttention(model_args)
    candidate = current.QSAAttention(model_args)
    baseline.sparse_sdpa = candidate.sparse_sdpa = True
    report = dict(kind='synthetic_qsa_not_model_throughput', baseline=BASELINE,
                  baseline_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                  checkout=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  dirty=bool(subprocess.check_output(['git','status','--porcelain'])),
                  mlx=version('mlx'), platform=platform.platform(), device=mx.device_info(),
                  cpu=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                  physical_memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
                  rounds=args.rounds, seed=628, warmups=3, full_model_benchmark=False, cases=[])
    report['source_hashes'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path('runtime/deepseek_v4_ssd/qwen4_exp.py'),Path('runtime/deepseek_v4_ssd/qwen_qsa_indexed.py'),
                  Path('runtime/deepseek_v4_ssd/qwen_qsa_schedule.py'),Path(__file__))}
    for kv_length in (8192,32768):
        for queries in (1,3,128):
            mx.random.seed(628+kv_length+queries)
            q = (mx.random.normal((1,24,queries,256))*.15).astype(mx.bfloat16)
            k = (mx.random.normal((1,2,kv_length,256))*.15).astype(mx.bfloat16)
            v = (mx.random.normal(k.shape)*.15).astype(mx.bfloat16)
            iq = (mx.random.normal((1,queries,4,128))*.15).astype(mx.bfloat16)
            raw = (mx.random.normal((1,kv_length,128))*.15).astype(mx.bfloat16)
            mx.eval(q,k,v,iq,raw)
            def call(name):
                attn = baseline if name=='previous-fused-4' else candidate
                attn.sparse_sdpa = name!='manual-4'
                attn.query_chunk = 32 if name=='chunk-32' else 16 if name=='chunk-16' else 4
                attn.indexed_decode = name=='indexed-decode'
                return attn._bounded_attention(q,k,v,iq,raw,kv_length-queries)
            names = ['previous-fused-4','stored-axis-4','manual-4']
            names += ['chunk-16','chunk-32'] if queries>8 else ['indexed-decode']
            outputs = {}; samples={name:[] for name in names}
            for name in names:
                for _ in range(3):mx.eval(call(name))
                outputs[name]=call(name);mx.eval(outputs[name])
            for turn in range(args.rounds):
                for name in (names if turn%2==0 else names[::-1]):
                    start=time.perf_counter_ns();out=call(name);mx.eval(out)
                    samples[name].append((time.perf_counter_ns()-start)/1e6)
            reference=np.asarray(outputs['previous-fused-4'].astype(mx.float32))
            results={}
            for name in names:
                array=np.asarray(outputs[name].astype(mx.float32))
                error=float(np.max(np.abs(array-reference)))
                if not np.isfinite(array).all() or error>.025:
                    raise RuntimeError(f'QSA numerical check failed: {kv_length}/{queries}/{name}: {error}')
                mx.clear_cache(); before=mx.get_active_memory(); mx.reset_peak_memory()
                output=call(name);mx.eval(output)
                results[name]=dict(p50_ms=float(np.median(samples[name])),p95_ms=float(np.percentile(samples[name],95)),
                                  samples_ms=samples[name],additional_peak_mlx_bytes=max(0,mx.get_peak_memory()-before),
                                  max_abs_difference=error,output_sha256=digest(outputs[name]))
                del output
            report['cases'].append(dict(kv_tokens=kv_length,query_tokens=queries,results=results))
            print(json.dumps(dict(kv=kv_length,queries=queries,p50_ms={n:round(r['p50_ms'],4) for n,r in results.items()})),flush=True)
            del outputs,q,k,v,iq,raw
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:json.dump(report,handle,indent=2);handle.write('\n')


if __name__=='__main__':
    main()
