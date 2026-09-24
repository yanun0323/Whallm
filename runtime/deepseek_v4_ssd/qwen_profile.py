"""Opt-in, single-request Qwen diagnostics; never installed by the server.

Host mode preserves lazy execution: times include construction and existing waits,
NOT GPU kernel time. Sync mode evaluates outputs at each boundary and destroys
normal overlap. Compare output hashes and an uninstrumented baseline separately.
Class hooks are temporary, owner-thread-only, and restored even after failure.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from functools import wraps
import json
import os
from pathlib import Path
import resource
import threading
import time

import mlx.core as mx

from .io_metrics import _RUsageInfoV2, _load_proc_pid_rusage
from .throughput_diagnostics import cache_counters, difference


def process_snapshot():
    function = _load_proc_pid_rusage()
    usage = _RUsageInfoV2()
    if function is None or function(os.getpid(), 2, ctypes.byref(usage)) != 0:
        return {}
    result = {name: int(getattr(usage, name)) for name in (
        "pageins", "resident_size", "physical_footprint",
        "disk_io_bytes_read", "disk_io_bytes_written",
    )}
    # proc_pid_rusage CPU times are Mach ticks, not nanoseconds on Apple Silicon.
    cpu = resource.getrusage(resource.RUSAGE_SELF)
    result.update(cpu_user_seconds=cpu.ru_utime, cpu_system_seconds=cpu.ru_stime,
        minor_faults=cpu.ru_minflt, major_faults=cpu.ru_majflt,
        voluntary_context_switches=cpu.ru_nvcsw, involuntary_context_switches=cpu.ru_nivcsw)
    return result


def arrays(value):
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from arrays(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from arrays(item)


class QwenProfile:
    """Bounded aggregates by phase, layer and component (not one record per token)."""

    def __init__(self, mode="host", *, timeline_path=None):
        if mode not in {"host", "sync"}:
            raise ValueError("profile mode must be host or sync")
        self.mode = mode
        self.phase = "load"
        self.layer = -1
        self.cache = None
        self.rows = {}
        self.stack = []
        self.patches = []
        self.owner = threading.get_ident()
        self.request = -1
        self._phase_started = None
        self._timeline = None
        if timeline_path is not None:
            # Opt-in Apple Silicon diagnostics only, never normal server hooks.
            self._clock = ctypes.CDLL('/usr/lib/libSystem.B.dylib').mach_absolute_time
            self._clock.restype = ctypes.c_uint64
            self._timeline = Path(timeline_path).open('x')
            self._timeline.write(json.dumps(dict(kind='metadata', schema_version=1,
                pid=os.getpid(), clock='mach_absolute_time', mode=mode)) + '\n')

    def begin_phase(self, phase, request):
        if self._phase_started is not None:
            raise RuntimeError('previous profiling phase is still open')
        self.phase, self.request = phase, request
        if self._timeline is not None:
            self._phase_started = int(self._clock())

    def _interval(self, kind, begin, end, *, layer=-1, failed=False):
        self._timeline.write(json.dumps(dict(kind=kind, request=self.request, phase=self.phase,
            layer=layer, begin=begin, end=end, failed=failed)) + '\n')

    def end_phase(self, *, failed=False):
        if self._phase_started is not None:
            self._interval('phase', self._phase_started, int(self._clock()), failed=failed)
            self._phase_started = None

    def snapshot(self):
        result = process_snapshot()
        result.update(mlx_active_bytes=mx.get_active_memory(), mlx_cache_bytes=mx.get_cache_memory(),
                      owner_cpu_seconds=time.thread_time())
        counters = cache_counters(self.cache)
        if counters:
            result.update({"expert_" + k: v for k, v in counters.items()})
        return result

    @contextmanager
    def span(self, name, layer=None):
        old_layer = self.layer
        if layer is not None:
            self.layer = layer
        key = (self.request, self.phase, self.layer, name)
        before = self.snapshot()
        frame = {"children_seconds": 0.0}
        self.stack.append(frame)
        tick = int(self._clock()) if self._timeline is not None and name == 'decoder' else None
        start = time.perf_counter()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            elapsed = time.perf_counter() - start
            end_tick = int(self._clock()) if tick is not None else None
            after = self.snapshot()
            self.stack.pop()
            row = self.rows.setdefault(key, dict(request=key[0], phase=key[1], layer=key[2], component=name,
                calls=0, errors=0, inclusive_seconds=0.0, exclusive_seconds=0.0,
                max_call_seconds=0.0, counters={}, boundary_max={}, boundary_min={}))
            row["calls"] += 1
            row["errors"] += int(failed)
            row["inclusive_seconds"] += elapsed
            row["exclusive_seconds"] += max(0.0, elapsed - frame["children_seconds"])
            row["max_call_seconds"] = max(row["max_call_seconds"], elapsed)
            for field, delta in difference(after, before).items():
                if field.startswith("mlx_") or field in {"resident_size", "physical_footprint"}:
                    row["boundary_max"][field] = max(row["boundary_max"].get(field, 0), before[field], after[field])
                    row["boundary_min"][field] = min(row["boundary_min"].get(field, before[field]), before[field], after[field])
                else:
                    row["counters"][field] = row["counters"].get(field, 0) + delta
            if tick is not None:
                self._interval('decoder', tick, end_tick, layer=key[2], failed=failed)
            if self.stack:
                # Charge hook bookkeeping to its parent, not to child compute.
                self.stack[-1]["children_seconds"] += time.perf_counter() - start
            self.layer = old_layer

    def hook(self, target, method, labels=None, name=None, evaluate=True):
        """labels maps object IDs to (component, layer); None wraps a function."""
        original = getattr(target, method)
        # Preserve descriptors, including classmethods, on restoration.
        raw = vars(target).get(method)

        @wraps(original)
        def wrapped(*args, **kwargs):
            label = labels.get(id(args[0])) if labels is not None and args else None
            if threading.get_ident() != self.owner or (labels is not None and label is None):
                return original(*args, **kwargs)
            component, layer = label if label is not None else (name or method, None)
            if self.mode == 'sync' and component == 'decoder' and self._timeline is not None:
                # Exclude previously submitted GPU work from this decoder window.
                # Lazy dependencies can still execute inside: this is scope activity,
                # not semantic ownership of individual kernels.
                mx.synchronize()
            with self.span(component, layer):
                result = original(*args, **kwargs)
                if self.mode == "sync" and evaluate:
                    outputs = list(arrays(result))
                    if outputs:
                        mx.eval(*outputs)
                    mx.synchronize()
            if component == "ngram_lookup":
                row = self.rows[(self.request, self.phase, self.layer, component)]
                count = int(args[1].size)
                row["counters"]["ngram_requested_rows"] = row["counters"].get("ngram_requested_rows", 0) + count
                row["counters"]["ngram_logical_bytes"] = row["counters"].get("ngram_logical_bytes", 0) + count * args[0].descriptor.row_bytes
            return result

        setattr(target, method, wrapped)
        self.patches.append((target, method, raw))

    def attach(self, runtime):
        """All 48 decoder layers, plus their compute / lookup / cache boundaries."""
        self.cache = runtime.expert_cache
        groups = {}

        def add(obj, name, layer=-1, method="__call__"):
            groups.setdefault((type(obj), method), {})[id(obj)] = (name, layer)

        model = runtime.model
        add(model.model.embed_tokens, "embedding")
        add(model.lm_head, "lm_head")
        add(model.model.hyper_connection_mixer, "final_hyper_connection")
        add(model.ngram_store, "ngram_lookup", None, "lookup")
        for i, layer in enumerate(model.layers):
            add(layer, "decoder", i)
            attention = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
            add(attention, "gated_deltanet" if layer.layer_type == "linear_attention" else "qsa", i)
            if layer.layer_type != "linear_attention":
                add(attention, "qsa_attention", i, "_bounded_attention")
                add(attention.indexer, "qsa_index_projection", i, "project")
            for prefix in ("attn", "mlp"):
                hc = getattr(layer, prefix + "_hyper_connection")
                add(hc, prefix + "_hyper_connection", i)
                add(hc, prefix + "_residual", i, "inject")
            add(layer.mlp, "moe", i)
            add(layer.mlp.gate, "router_projection", i)
            add(layer.mlp.shared_expert, "shared_expert", i)
            add(layer.mlp.experts, "routed_experts", i)
            add(layer.mlp.experts, "routed_compute_io", i, "_routed")
            if layer.ple is not None:
                add(layer.ple, "ple", i)
                add(layer.ple.ple_embedding, "ngram_embedding", i)
        for (cls, method), labels in groups.items():
            self.hook(cls, method, labels)
        # Existing counters also cover iter_ready generator consumption and
        # asynchronous workers; timing generator creation would be incorrect.
        self.hook(type(self.cache), "get_many", {id(self.cache): ("expert_acquire", None)}, evaluate=False)
        from . import model as model_module
        from .model_support import qwen as qwen_support
        self.hook(qwen_support, "_qwen_layer_major_prefill", name="layer_major_prefill", evaluate=False)
        self.hook(qwen_support, "eval_prompt_cache", name="prefill_cache_eval", evaluate=False)
        self.hook(model_module, "eval_prompt_cache", name="decode_cache_eval", evaluate=False)

    def close(self):
        for target, method, original in reversed(self.patches):
            if original is None:
                delattr(target, method)
            else:
                setattr(target, method, original)
        self.patches.clear()
        if self._timeline is not None and not self._timeline.closed:
            self.end_phase(failed=True)
            self._timeline.close()

    def report(self):
        return {"schema_version": 2, "mode": self.mode,
            "rows": sorted(self.rows.values(), key=lambda r: (r['request'], r["phase"], r["layer"], r["component"])),
            "notes": "Nested inclusive times/counters must not be summed. Boundary memory is process-wide, not tensor ownership or transient peak. CPU/disk counters include worker threads; owner_cpu_seconds uses thread_time on the owner thread. Fault/switch values are OS-reported process counters. Async work can cross boundaries. Sync elapsed includes Python, I/O and GPU waits, NOT GPU kernel time. GPU utilization is sampled separately at system/phase level. Optional Mach-clock timeline supports external GPU interval overlap analysis; only sync decoder windows are drained, and lazy dependencies can still execute within a scope."}
