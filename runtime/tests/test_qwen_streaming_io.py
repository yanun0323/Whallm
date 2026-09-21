"""Byte-exact handoff, actual expert I/O and shared-expert scheduling qualification."""
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from deepseek_v4_ssd.cancellation import GenerationCancelled, cancellation_scope
from deepseek_v4_ssd.expert_cache import ExpertCache, CacheMetrics
from deepseek_v4_ssd.manifest import InstalledModel, Tensor, QWEN_EXPERT_REGIONS
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import _parse_runtime, ModelCatalogError
from deepseek_v4_ssd.qwen_streaming_policy import hot_tail_experts


def fixture(root, layers=2, experts=8):
    (root/'experts').mkdir()
    regions=tuple(Tensor(*spec) for spec in QWEN_EXPERT_REGIONS)
    size=sum(region.length for region in regions)
    for layer in range(layers):
        with (root/f'experts/layer_{layer:02d}.bin').open('wb') as f:
            for expert in range(experts): f.write(bytes([1+layer*experts+expert])*size)
    return InstalledModel(root,'synthetic','synthetic',layers,experts,1,size,(),regions,model_kind='qwen3.8-flash-next')


def slot_bytes(cache, layer, expert):
    slot=cache._entries[(layer,expert)].slot
    views=[cache._pool.writable_region(slot,r.name) for r in cache.model.expert_regions]
    try: return b''.join(bytes(v) for v in views)
    finally:
        for v in views: v.release()


class StreamingIOTests(unittest.TestCase):
    def test_coalesced_prefill_reads_preserve_native_layout_and_skip_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            model=fixture(Path(directory),layers=1)
            for separate in (False,True):
                for batch,expected in ((1,6),(4,2)):
                    with self.subTest(separate=separate,batch=batch), ExpertCache(model,slots=4,read_workers=2,
                            prefetch_read_workers=1,separate_prefill_io=separate,prefill_read_experts=batch) as cache:
                        with patch.object(cache,'_pread_views',wraps=cache._pread_views) as reader:
                            with cache.batched_layer(0,experts=[0,1,2,5,6,7]):
                                packed=memoryview(cache._active_prefetch_trace.job.packed).cast('B')
                                try:
                                    for e in (0,1,2,5,6,7):
                                        parts=cache._pool.layer_write_views(packed,e)
                                        try: self.assertEqual(b''.join(map(bytes,parts)),bytes([e+1])*model.expert_blob_size)
                                        finally:
                                            for part in parts: part.release()
                                finally: packed.release()
                        self.assertEqual(reader.call_count,expected)
                        self.assertEqual(cache.metrics.prefill_read_batches,expected)
                        self.assertEqual(cache.metrics.bytes_read,6*model.expert_blob_size)
                        if separate: self.assertEqual(cache.prefill_io_snapshot()['read_calls'],expected)

    def test_hot_handoff_is_owned_byte_exact_and_saves_demand_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            model=fixture(Path(directory))
            with ExpertCache(model,slots=8,read_workers=2,prefetch_read_workers=1,prefill_seed_experts=2) as cache:
                with cache.reuse_layer_buffers():
                    for layer in range(2):
                        with cache.batched_layer(layer):
                            cache.capture_prefill_seed_routes(layer,mx.array([[[1],[1],[2],[1]]]))
                self.assertEqual(cache.resident_count,4)
                self.assertEqual(cache.metrics.prefill_seeded_experts,4)
                self.assertEqual(cache.metrics.prefill_seed_bytes,4*model.expert_blob_size)
                before=cache.metrics_snapshot()
                # Layer buffers can now be reused; retained slots must own bytes.
                with cache.batched_layer(0): pass
                for layer in range(2):
                    cache.get_many(layer,[1,2])
                    for e in (1,2): self.assertEqual(slot_bytes(cache,layer,e),bytes([1+layer*8+e])*model.expert_blob_size)
                delta=cache.metrics_snapshot().delta(before)
                self.assertEqual(delta.misses,0)
                self.assertEqual(delta.prefill_seed_hits,4)
                cache.get_many(0,[1,2])
                self.assertEqual(cache.metrics.prefill_seed_hits,4,'first demand reuse counts once')
                cache.release_prefill_slots()
                self.assertFalse(cache._seeded_keys)
                self.assertEqual(cache.resident_count,0)

    def test_seed_tail_is_chunk_independent_and_metadata_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            model=fixture(Path(directory),layers=1)
            rows=np.array([[[i%5,(i*3)%7] for i in range(257)]],dtype=np.int32)
            wanted=set(hot_tail_experts(rows,8,3))
            for step in (1,13,128,257):
                with ExpertCache(model,slots=8,read_workers=1,prefill_seed_experts=3) as cache:
                    with cache.batched_layer(0):
                        for start in range(0,257,step):
                            cache.capture_prefill_seed_routes(0,mx.array(rows[:,start:start+step]))
                            self.assertLessEqual(sum(p.shape[1] for p in cache._prefill_seed_routes),128)
                    self.assertEqual({e for _,e in cache._entries},wanted)
                    self.assertIsNone(cache._prefill_seed_routes)

    def test_no_seed_on_failed_layer_and_cancelled_copy_has_no_partial_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            model=fixture(Path(directory),layers=1)
            with ExpertCache(model,slots=4,read_workers=1,prefill_seed_experts=3) as cache:
                with self.assertRaisesRegex(RuntimeError,'layer failed'):
                    with cache.batched_layer(0):
                        cache.capture_prefill_seed_routes(0,mx.array([[[1],[2],[3]]]))
                        raise RuntimeError('layer failed')
                self.assertEqual(cache.resident_count,0)
                event=threading.Event()
                write=cache._pool.writable_region
                def cancelled_write(*args):
                    view=write(*args)
                    event.set()
                    return view
                with cancellation_scope(event), patch.object(cache._pool,'writable_region',side_effect=cancelled_write):
                    with self.assertRaises(GenerationCancelled):
                        with cache.batched_layer(0):
                            cache.capture_prefill_seed_routes(0,mx.array([[[1],[2],[3]]]))
                self.assertEqual(cache.resident_count,0)
                self.assertEqual(len(cache._free_slots),4)
                self.assertFalse(cache._seeded_keys)
                self.assertIsNone(cache._batched_layer)
                cache.get_many(0,[1])  # No poisoned reservations after cancellation.

    def test_seed_quota_never_evicts_demand_residents(self):
        with tempfile.TemporaryDirectory() as directory:
            model=fixture(Path(directory))
            with ExpertCache(model,slots=2,read_workers=1,prefill_seed_experts=128) as cache:
                cache.get_many(0,[7])
                with cache.batched_layer(1):
                    cache.capture_prefill_seed_routes(1,mx.array([[[1],[1],[2]]]))
                self.assertEqual(set(cache._entries),{(0,7),(1,1)})
                self.assertEqual(cache.metrics.evictions,0)
                with cache.batched_layer(0):
                    cache.capture_prefill_seed_routes(0,mx.array([[[1],[2]]]))
                self.assertEqual(set(cache._entries),{(0,7),(1,1)})

    def test_metrics_new_fields_have_exact_snapshot_deltas(self):
        before=CacheMetrics(wait_seconds=2,prefill_seed_hits=4,prefill_seeded_experts=8,prefill_read_batches=16)
        after=replace(before,wait_seconds=2.5,prefill_seed_hits=6,shared_overlap_submissions=1)
        delta=after.delta(before)
        self.assertEqual(delta.wait_seconds,.5)
        self.assertEqual(delta.prefill_seed_hits,2)
        self.assertEqual(delta.prefill_read_batches,0)
        self.assertEqual(delta.shared_overlap_submissions,1)

    def test_catalog_old_defaults_and_model_isolation(self):
        fields=('qwen_prefill_read_experts','qwen_prefill_seed_experts','qwen_shared_expert_overlap')
        old=asdict(RuntimeConfig())
        for field in fields: old.pop(field)
        for kind in ('deepseek-v4','deepseek-v4.1','qwen3.8-flash-next'):
            parsed=_parse_runtime(old,'runtime',kind)
            self.assertEqual(tuple(getattr(parsed,n) for n in fields),(1,0,False))
        updated=old|dict(qwen_prefill_read_experts=4,qwen_prefill_seed_experts=32,qwen_shared_expert_overlap=True,
                        layer_major_prefill=True,batched_expert_prefill=True)
        _parse_runtime(updated,'runtime','qwen3.8-flash-next')
        # Catalog parsing validates scalars before the model-support boundary.
        from deepseek_v4_ssd.model_support import get_support
        for kind in ('deepseek-v4','deepseek-v4.1'):
            parsed = _parse_runtime(updated,'runtime',kind)
            with self.assertRaises(ValueError): get_support(kind).validate_config(parsed)

    def test_shared_overlap_submits_before_expert_fetch_and_preserves_output(self):
        from deepseek_v4_ssd import qwen4_exp as qwen
        from runtime.tests.test_qwen_speed import tiny_args, experts_fixture, bits
        weights,_=experts_fixture()
        events=[]
        def acquire(layer,ids):
            events.append('read')
            return SimpleNamespace(individual_weights=weights,slots={i:i for i in range(8)})
        cache=SimpleNamespace(model=SimpleNamespace(expert_count=8),current_batched=lambda _:None,
                              get_many=acquire,record_shared_overlap=lambda:events.append('submitted'))
        mx.random.seed(589)
        moe=qwen.SparseMoE(tiny_args(),0,cache)
        for length in (1,3):
            x=mx.random.normal((1,length,64)).astype(mx.bfloat16)
            baseline=bits(moe(x)).copy()
            events.clear();moe.shared_overlap=True
            actual=bits(moe(x))
            np.testing.assert_array_equal(actual,baseline)
            self.assertEqual(events[0],'submitted' if length==1 else 'read')
            moe.shared_overlap=False

    def test_shared_overlap_ready_order_is_independent_of_completion_order(self):
        from deepseek_v4_ssd import qwen4_exp as qwen
        from runtime.tests.test_qwen_speed import tiny_args, experts_fixture, bits
        weights,_=experts_fixture()
        events=[]
        def ready(layer, ids):
            events.append('read')
            for expert in reversed(ids):
                yield expert, weights[expert]
        cache=SimpleNamespace(model=SimpleNamespace(expert_count=8),current_batched=lambda _:None,
                              ready_expert_decode=True,iter_ready=ready,
                              record_shared_overlap=lambda:events.append('submitted'))
        mx.random.seed(602)
        moe=qwen.SparseMoE(tiny_args(),0,cache)
        value=mx.random.normal((1,1,64)).astype(mx.bfloat16)
        expected=bits(moe(value)).copy()
        events.clear()
        moe.shared_overlap=True
        np.testing.assert_array_equal(bits(moe(value)),expected)
        self.assertEqual(events[:2],['submitted','read'])

    def test_seeded_slots_preserve_actual_mxfp4_compute_after_layer_reuse(self):
        from deepseek_v4_ssd import qwen4_exp as qwen
        from runtime.tests.test_qwen_speed import bits
        # Scale rows must have four-byte-aligned widths in the batched layout.
        mx.random.seed(604)
        gate, gs = mx.quantize(mx.random.normal((8,256,128)).astype(mx.bfloat16)*.03,
                              group_size=32,bits=4,mode="mxfp4")
        down, ds = mx.quantize(mx.random.normal((8,128,128)).astype(mx.bfloat16)*.03,
                              group_size=32,bits=4,mode="mxfp4")
        mx.eval(gate,gs,down,ds)
        weights=[qwen.QwenExpertWeights(gate[i],gs[i],down[i],ds[i]) for i in range(8)]
        names=('gate_up.weight','gate_up.scale','down.weight','down.scale')
        arrays=lambda w:(w.gate_up,w.gate_up_scales,w.down,w.down_scales)
        regions=[]
        offset=0
        for name,array in zip(names,arrays(weights[0])):
            length=array.nbytes
            regions.append(Tensor(name,'U32' if name.endswith('.weight') else 'U8',tuple(array.shape),offset,length))
            offset+=length
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'experts').mkdir()
            for layer in range(2):
                with (root/f'experts/layer_{layer:02d}.bin').open('wb') as f:
                    for expert in range(8):
                        for array in arrays(weights[(expert+layer)%8]):
                            f.write(np.asarray(array).tobytes())
            model=InstalledModel(root,'synthetic','synthetic',2,8,2,offset,(),tuple(regions),model_kind='qwen3.8-flash-next')
            with ExpertCache(model,slots=8,read_workers=2,prefill_read_experts=4,prefill_seed_experts=2) as cache:
                value=mx.ones((1,1,128),dtype=mx.bfloat16)*.1
                indices=mx.array([[[3,5]]],dtype=mx.int32)
                with cache.reuse_layer_buffers():
                    for layer in range(2):
                        with cache.batched_layer(layer):
                            output=qwen.StreamingExperts(layer,cache)(value,indices)
                            mx.eval(output)
                before=cache.metrics_snapshot()
                for layer in range(2):
                    module=qwen.StreamingExperts(layer,cache)
                    actual=bits(module(value,indices)).copy()
                    expected=mx.stack([module._one(value.reshape(1,128),weights[(expert+layer)%8])
                                       for expert in (3,5)],axis=-2).reshape(1,1,2,128)
                    np.testing.assert_array_equal(actual,bits(expected))
                self.assertEqual(cache.metrics_snapshot().delta(before).misses,0)
                self.assertEqual(cache.metrics.prefill_seed_hits,4)

    def test_overlap_resolves_routes_once_before_shared_submission(self):
        from deepseek_v4_ssd import qwen4_exp as qwen
        from runtime.tests.test_qwen_speed import tiny_args, experts_fixture, bits
        weights,_=experts_fixture()
        events=[]
        def acquire(layer,ids):
            events.append("read")
            return SimpleNamespace(individual_weights=weights,slots={i:i for i in range(8)})
        cache=SimpleNamespace(model=SimpleNamespace(expert_count=8),current_batched=lambda _:None,
                              get_many=acquire,record_routing_sync=lambda _:events.append("route-ready"))
        moe=qwen.SparseMoE(tiny_args(),0,cache)
        moe.shared_overlap=True
        value=mx.ones((1,1,64),dtype=mx.bfloat16)*.1
        original=mx.async_eval
        def submit(*arrays):
            events.append("submit")
            self.assertEqual(events,["route-ready","submit"])
            self.assertEqual(len(arrays),1)
            self.assertEqual(arrays[0].shape,value.shape)
            return original(*arrays)
        with patch.object(qwen.mx,"async_eval",side_effect=submit):
            bits(moe(value))
        self.assertEqual(events,["route-ready","submit","read"])

    def test_darwin_cache_policy_without_python_readahead_constant(self):
        from deepseek_v4_ssd import io_metrics
        calls=[]
        fcntl=SimpleNamespace(F_NOCACHE=48,fcntl=lambda *args:calls.append(args))
        with patch.object(io_metrics,"fcntl",fcntl), patch.object(io_metrics.sys,"platform","darwin"):
            io_metrics.configure_expert_file_cache_policy(7,"bypass")
        self.assertEqual(calls,[(7,48,1),(7,45,0)])
