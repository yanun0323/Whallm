"""Synthetic native-layout I/O and cache handoff; NOT full-model token throughput."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import statistics
import tempfile
import time

import mlx.core as mx
from deepseek_v4_ssd.expert_cache import ExpertCache
from deepseek_v4_ssd.throughput_diagnostics import source_files_hash
from runtime.tests.test_qwen_streaming_io import fixture, slot_bytes


def trial(model, batch, seeds, trace):
    # Same eight slots in all variants. Files are created locally once; their
    # coldness is not controlled. File bypass does not prove physical SSD traffic.
    with ExpertCache(model, slots=8, read_workers=2, prefetch_read_workers=1,
                     separate_prefill_io=True, prefill_read_experts=batch,
                     prefill_seed_experts=seeds, eviction_policy='lru') as cache:
        started=time.perf_counter()
        with cache.reuse_layer_buffers():
            for layer in range(model.layer_count):
                with cache.batched_layer(layer):
                    cache.capture_prefill_seed_routes(layer,mx.array([[[0],[1],[0],[1]]]))
        prefill=time.perf_counter()-started
        before=cache.metrics_snapshot()
        io=cache.prefill_io_snapshot()
        digest=hashlib.sha256()
        began=time.perf_counter()
        for experts in trace:
            for layer in range(model.layer_count):
                # Real ready reads are exercised; the synthetic consumer hashes
                # bytes rather than pretending that this is a language model.
                ready=cache.iter_ready(layer,experts)
                try:
                    for expert,_ in ready:
                        if expert not in experts: raise RuntimeError('unexpected expert')
                finally: ready.close()
                for expert in experts:
                    data=slot_bytes(cache,layer,expert)
                    if data != bytes([1+layer*model.expert_count+expert])*model.expert_blob_size:
                        raise RuntimeError('payload mismatch')
                    digest.update(data)
        return dict(prefill_seconds=prefill,synthetic_consumer_seconds=time.perf_counter()-began,
                    prefill_metrics=asdict(before),decode_metrics=asdict(cache.metrics_snapshot().delta(before)),
                    prefill_io=io,output_sha256=digest.hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if not mx.metal.is_available(): raise RuntimeError('Apple Metal is required')
    report=dict(kind='synthetic_native_expert_bytes_not_model_inference',full_model_benchmark=False,
                platform=platform.platform(),device=mx.device_info(),source_files_sha256=source_files_hash(),
                notes='2 layers / 8 experts; same 8-slot capacity. Interleaved variants. Warm/cache state not controlled, no physical SSD claim. Hashing cost is in the synthetic consumer timer; no GPU model compute.',traces={})
    with tempfile.TemporaryDirectory() as directory:
        model=fixture(Path(directory))
        variants=[('baseline',1,0),('read-4',4,0),('seed-2',1,2),('combined',4,2)]
        for trace_name,trace in [('reuse',[[0,1]]*8),('unrelated',[[6,7]]*8)]:
            results={name:[] for name,_,_ in variants}
            for rep in range(3):
                for name,batch,seeds in (variants if rep%2==0 else variants[::-1]):
                    results[name].append(trial(model,batch,seeds,trace))
            hashes={sample['output_sha256'] for samples in results.values() for sample in samples}
            if len(hashes)!=1: raise RuntimeError('variant outputs differ')
            report['traces'][trace_name]=dict(samples=results,p50_prefill_seconds={
                name:statistics.median(sample['prefill_seconds'] for sample in samples) for name,samples in results.items()})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({trace:{name:values for name,values in data['p50_prefill_seconds'].items()}
                      for trace,data in report['traces'].items()},indent=2))


if __name__=='__main__': main()
