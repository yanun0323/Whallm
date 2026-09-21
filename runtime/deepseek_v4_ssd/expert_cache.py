from __future__ import annotations

import ctypes
import heapq
import os
import threading
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np

from .cancellation import cancel_and_drain, check_cancelled, wait_for_futures

from .io_metrics import (
    EXPERT_FILE_CACHE_POLICIES,
    PageCacheReadClassification,
    configure_expert_file_cache_policy,
    page_cache_residency_snapshot,
)
from .expert_layout import _fused_slot_regions, _qwen_slot_regions, batched_layer_layout
from .model_support import support_for_installed
from .manifest import InstalledModel, Tensor


@dataclass(frozen=True)
class ExpertWeights:
    w1: mx.array
    w1_scales: mx.array
    w2: mx.array
    w2_scales: mx.array
    w3: mx.array
    w3_scales: mx.array
    w13: mx.array | None = None
    w13_scales: mx.array | None = None


@dataclass(frozen=True)
class ResidentExperts:
    individual_weights: tuple[ExpertWeights, ...]
    slots: dict[int, int]


@dataclass(frozen=True)
class BatchedExperts:
    w1: mx.array
    w1_scales: mx.array
    w2: mx.array
    w2_scales: mx.array
    w3: mx.array
    w3_scales: mx.array
    w13: mx.array | None = None
    w13_scales: mx.array | None = None


@dataclass(frozen=True)
class QwenExpertWeights:
    gate_up: mx.array
    gate_up_scales: mx.array
    down: mx.array
    down_scales: mx.array


@dataclass(frozen=True)
class QwenBatchedExperts:
    gate_up: mx.array
    gate_up_scales: mx.array
    down: mx.array
    down_scales: mx.array


@dataclass
class CacheMetrics:
    wait_seconds: float = 0.0
    prefill_read_batches: int = 0
    prefill_seeded_experts: int = 0
    prefill_seed_bytes: int = 0
    prefill_seed_seconds: float = 0.0
    prefill_seed_hits: int = 0
    shared_overlap_submissions: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes_read: int = 0
    read_seconds: float = 0.0
    upload_seconds: float = 0.0
    pack_seconds: float = 0.0
    eviction_seconds: float = 0.0
    routing_sync_seconds: float = 0.0
    batched_layers: int = 0
    gather_qmm_calls: int = 0
    prefetched_layer_hits: int = 0
    expert_union_calls: int = 0
    routed_expert_assignments: int = 0
    expert_union_experts: int = 0
    expert_union_misses: int = 0
    staged_expert_reads: int = 0
    staged_w13_bytes_read: int = 0
    staged_w2_bytes_read: int = 0
    staged_read_seconds: float = 0.0
    staged_w2_wait_seconds: float = 0.0
    staged_first_stage_submit_seconds: float = 0.0
    page_cache_probe_calls: int = 0
    page_cache_probe_failures: int = 0
    page_cache_classified_bytes: int = 0
    page_cache_resident_bytes_before_read: int = 0
    page_cache_nonresident_bytes_before_read: int = 0
    page_cache_unclassified_bytes: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def delta(self, before: CacheMetrics) -> CacheMetrics:
        return CacheMetrics(
            wait_seconds=self.wait_seconds - before.wait_seconds,
            prefill_read_batches=self.prefill_read_batches - before.prefill_read_batches,
            prefill_seeded_experts=self.prefill_seeded_experts - before.prefill_seeded_experts,
            prefill_seed_bytes=self.prefill_seed_bytes - before.prefill_seed_bytes,
            prefill_seed_seconds=self.prefill_seed_seconds - before.prefill_seed_seconds,
            prefill_seed_hits=self.prefill_seed_hits - before.prefill_seed_hits,
            shared_overlap_submissions=self.shared_overlap_submissions - before.shared_overlap_submissions,
            hits=self.hits - before.hits,
            misses=self.misses - before.misses,
            evictions=self.evictions - before.evictions,
            bytes_read=self.bytes_read - before.bytes_read,
            read_seconds=self.read_seconds - before.read_seconds,
            upload_seconds=self.upload_seconds - before.upload_seconds,
            pack_seconds=self.pack_seconds - before.pack_seconds,
            eviction_seconds=self.eviction_seconds - before.eviction_seconds,
            routing_sync_seconds=(
                self.routing_sync_seconds - before.routing_sync_seconds
            ),
            batched_layers=self.batched_layers - before.batched_layers,
            gather_qmm_calls=self.gather_qmm_calls - before.gather_qmm_calls,
            prefetched_layer_hits=(
                self.prefetched_layer_hits - before.prefetched_layer_hits
            ),
            expert_union_calls=(
                self.expert_union_calls - before.expert_union_calls
            ),
            routed_expert_assignments=(
                self.routed_expert_assignments
                - before.routed_expert_assignments
            ),
            expert_union_experts=(
                self.expert_union_experts - before.expert_union_experts
            ),
            expert_union_misses=(
                self.expert_union_misses - before.expert_union_misses
            ),


            staged_expert_reads=(
                self.staged_expert_reads - before.staged_expert_reads
            ),
            staged_w13_bytes_read=(
                self.staged_w13_bytes_read - before.staged_w13_bytes_read
            ),
            staged_w2_bytes_read=(
                self.staged_w2_bytes_read - before.staged_w2_bytes_read
            ),
            staged_read_seconds=(
                self.staged_read_seconds - before.staged_read_seconds
            ),
            staged_w2_wait_seconds=(
                self.staged_w2_wait_seconds - before.staged_w2_wait_seconds
            ),
            staged_first_stage_submit_seconds=(
                self.staged_first_stage_submit_seconds
                - before.staged_first_stage_submit_seconds
            ),


            page_cache_probe_calls=(
                self.page_cache_probe_calls - before.page_cache_probe_calls
            ),
            page_cache_probe_failures=(
                self.page_cache_probe_failures
                - before.page_cache_probe_failures
            ),
            page_cache_classified_bytes=(
                self.page_cache_classified_bytes
                - before.page_cache_classified_bytes
            ),
            page_cache_resident_bytes_before_read=(
                self.page_cache_resident_bytes_before_read
                - before.page_cache_resident_bytes_before_read
            ),
            page_cache_nonresident_bytes_before_read=(
                self.page_cache_nonresident_bytes_before_read
                - before.page_cache_nonresident_bytes_before_read
            ),
            page_cache_unclassified_bytes=(
                self.page_cache_unclassified_bytes
                - before.page_cache_unclassified_bytes
            ),
        )


@dataclass(frozen=True)
class ExpertUnionLayerMetrics:
    layer: int
    routed_expert_assignments: int
    unique_experts: int
    cache_misses: int


@dataclass
class ExpertUnionProfile:
    """Capture the per-layer expert union for one model forward."""

    layers: list[ExpertUnionLayerMetrics] = field(default_factory=list)

    @property
    def routed_expert_assignments(self) -> int:
        return sum(layer.routed_expert_assignments for layer in self.layers)

    @property
    def unique_experts(self) -> int:
        return sum(layer.unique_experts for layer in self.layers)

    @property
    def cache_misses(self) -> int:
        return sum(layer.cache_misses for layer in self.layers)


@dataclass
class _Entry:
    slot: int
    frequency: int
    last_access: int
    version: int = 0
    layer: int = -1
    expert: int = -1


@dataclass(frozen=True)
class _LayerRead:
    packed: mx.array
    futures: tuple[Future[_SpeculativeRead], ...]
    experts: tuple[int, ...]
    submitted: float


@dataclass(frozen=True)
class _SpeculativeRead:
    started: float
    finished: float
    page_cache: PageCacheReadClassification = PageCacheReadClassification()


@dataclass
class _ActivePrefetchTrace:
    layer: int
    job: _LayerRead
    reads: tuple[_SpeculativeRead, ...]
    deadline: float
    future_wait_seconds: float
    used_experts: set[int] = field(default_factory=set)
    compute_submit: float | None = None


class StagedReadyExpert:
    """Expose w13-ready weights while keeping partial slot admission private."""

    def __init__(
        self,
        cache: ExpertCache,
        expert: int,
        weights: ExpertWeights,
        slot: int,
        w13_read: _SpeculativeRead | None = None,
        w2_future: Future[_SpeculativeRead] | None = None,
    ) -> None:
        self._cache = cache
        self.expert = expert
        self.weights = weights
        self.slot = slot
        self.w13_read = w13_read
        self._w2_future = w2_future
        self.w2_read: _SpeculativeRead | None = None
        self.finished = w2_future is None

    def finish_w2(self) -> None:
        if self.finished:
            return
        if self._w2_future is None:
            raise RuntimeError("staged expert has no w2 future")
        wait_started = time.perf_counter()
        read = self._w2_future.result()
        waited = time.perf_counter() - wait_started
        cache = self._cache
        with cache._lock:
            cache._pool.mark_loaded(self.slot)
            cache.metrics.staged_w2_wait_seconds += waited
        self.w2_read = read
        self.finished = True


class _ReadLimiter:
    """Limit aggregate preadv throughput across all read workers."""

    def __init__(self, bytes_per_second: int) -> None:
        if bytes_per_second <= 0:
            raise ValueError("read limit must be greater than zero")
        self._bytes_per_second = bytes_per_second
        self._lock = threading.Lock()

    def preadv(
        self,
        descriptor: int,
        views: list[memoryview],
        offset: int,
    ) -> int:
        with self._lock:
            started = time.perf_counter()
            count = os.preadv(descriptor, views, offset)
            delay = count / self._bytes_per_second - (time.perf_counter() - started)
            if delay > 0:
                time.sleep(delay)
            return count


def _staged_slot_regions(
    model: InstalledModel,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    source = {region.name: region for region in model.expert_regions}
    w13: dict[str, Tensor] = {}
    offset = 0
    for name in ("w3.weight", "w1.weight"):
        region = source[name]
        w13[name] = Tensor(name, region.dtype, region.shape, offset, region.length)
        offset += region.length
    first = w13["w3.weight"]
    second = w13["w1.weight"]
    w13["w13.weight"] = Tensor(
        "w13.weight",
        first.dtype,
        (first.shape[0] + second.shape[0], *first.shape[1:]),
        first.offset,
        first.length + second.length,
    )
    scale_offset = offset
    for name in ("w3.scale", "w1.scale"):
        region = source[name]
        w13[name] = Tensor(name, region.dtype, region.shape, offset, region.length)
        offset += region.length
    first_scale = w13["w3.scale"]
    second_scale = w13["w1.scale"]
    w13["w13.scale"] = Tensor(
        "w13.scale",
        first_scale.dtype,
        (
            first_scale.shape[0] + second_scale.shape[0],
            *first_scale.shape[1:],
        ),
        scale_offset,
        first_scale.length + second_scale.length,
    )

    w2: dict[str, Tensor] = {}
    offset = 0
    for name in ("w2.weight", "w2.scale"):
        region = source[name]
        w2[name] = Tensor(name, region.dtype, region.shape, offset, region.length)
        offset += region.length
    if (
        sum(region.length for name, region in w13.items() if not name.startswith("w13"))
        + sum(region.length for region in w2.values())
        != model.expert_blob_size
    ):
        raise ValueError("staged expert regions do not preserve the blob size")
    return w13, w2


class _SlotPool:
    """Keep each expert slot in a directly writable Metal buffer."""

    def __init__(self, model: InstalledModel, slots: int) -> None:
        self._model = model
        self._source_regions = {
            region.name: region for region in model.expert_regions
        }
        self._layout = support_for_installed(model).expert_layout
        self._regions = self._layout.regions(model)
        self._batched_layout = batched_layer_layout(
            self._regions, self._layout.batched_regions,
            model.expert_blob_size, model.expert_count,
        )
        self._slots: list[mx.array | None] = [None] * slots
        self._views: list[memoryview | None] = [None] * slots
        self._loaded = bytearray(slots)

    def resize(self, slots: int) -> None:
        """Caller has drained readers and synchronized GPU work."""
        previous = len(self._slots)
        if slots < previous:
            del self._views[slots:]
            del self._slots[slots:]
            del self._loaded[slots:]
        elif slots > previous:
            self._slots.extend([None] * (slots - previous))
            self._views.extend([None] * (slots - previous))
            self._loaded.extend(bytes(slots - previous))

    def prepare(self, slots: list[int]) -> None:
        created = []
        for slot in slots:
            if self._slots[slot] is None:
                array = mx.empty((self._model.expert_blob_size,), dtype=mx.uint8)
                self._slots[slot] = array
                created.append(array)
        if created:
            mx.eval(*created)
            for slot in slots:
                array = self._slots[slot]
                if self._views[slot] is None and array is not None:
                    self._views[slot] = memoryview(array)

    def select_individual(self, slots: list[int]) -> tuple[ExpertWeights, ...]:
        arrays = []
        for slot in slots:
            array = self._slots[slot]
            if not self._loaded[slot] or array is None:
                raise RuntimeError("expert slot is empty")
            arrays.append(array)
        return self._layout.individual(arrays, self._array)

    def _array(self, packed: mx.array, name: str) -> mx.array:
        region = self._regions[name]
        value = packed[region.offset : region.offset + region.length]
        if region.dtype == "U32":
            return value.view(mx.uint32).reshape(region.shape)
        value = value.reshape(region.shape)
        if region.dtype == "I8":
            value = value.view(mx.int8)
        return value.view(mx.uint32) if name.endswith(".weight") else value

    def store(self, slots: list[int], blobs: list[bytes]) -> float:
        started = time.perf_counter()
        self.prepare(slots)
        for slot, blob in zip(slots, blobs):
            if len(blob) != self._model.expert_blob_size:
                raise ValueError("expert blob size does not match the manifest")
            source = memoryview(blob)
            for name, region in self._source_regions.items():
                self.writable_region(slot, name)[:] = source[
                    region.offset : region.offset + region.length
                ]
            self._loaded[slot] = 1
        return time.perf_counter() - started

    def writable_region(self, slot: int, name: str) -> memoryview:
        region = self._regions[name]
        view = self._views[slot]
        if view is None:
            raise RuntimeError("expert slot buffer is not prepared")
        return view[region.offset : region.offset + region.length]

    def write_views(self, buffer: memoryview, slot: int) -> list[memoryview]:
        base = slot * self._model.expert_blob_size
        return [
            buffer[
                base + self._regions[region.name].offset :
                base + self._regions[region.name].offset + region.length
            ]
            for region in self._model.expert_regions
        ]

    def mark_loaded(self, slot: int) -> None:
        self._loaded[slot] = 1

    def mark_empty(self, slot: int) -> None:
        self._loaded[slot] = 0

    def batched(self, packed: mx.array) -> BatchedExperts:
        """View a slot-major arena (paired with write_views), including research callers."""
        return self._layout.batched(
            packed, lambda buffer, name: self._batched_array(buffer, name, layer_major=False))

    def batched_layer(self, packed: mx.array) -> BatchedExperts:
        """View a region-major layer buffer (paired with layer_write_views)."""
        return self._layout.batched(packed, self._batched_array)

    def layer_write_views(self, buffer: memoryview, expert: int) -> list[memoryview]:
        views = []
        for region in self._model.expert_regions:
            base, stride, inner = self._batched_layout[region.name]
            start = base + expert * stride + inner
            views.append(buffer[start:start + region.length])
        return views

    def _batched_array(self, packed: mx.array, name: str, *, layer_major: bool = True) -> mx.array:
        region = self._regions[name]
        base, expert_stride, inner = (
            self._batched_layout[name] if layer_major
            else (region.offset, self._model.expert_blob_size, 0)
        )
        offset = base + inner
        if offset % 4 or expert_stride % 4:
            raise ValueError("batched expert layer regions must be 4-byte aligned")
        if inner == 0 and expert_stride == region.length:
            # Slice/reshape preserves contiguous batched projections for gather_qmm.
            shape = (self._model.expert_count, *region.shape)
            if region.dtype != "U32":
                if region.shape[-1] % 4:
                    raise ValueError("batched expert regions must be 4-byte aligned")
                shape = (*shape[:-1], shape[-1] // 4)
            end = offset + self._model.expert_count * expert_stride
            value = packed[offset // 4:end // 4].reshape(shape)
            return value if region.dtype == "U32" or name.endswith(".weight") else value.view(mx.uint8)
        region = Tensor(region.name, region.dtype, region.shape, offset, region.length)
        if region.dtype == "U32":
            shape = (self._model.expert_count, *region.shape)
            row_strides = []
            stride = 1
            for size in reversed(region.shape):
                row_strides.append(stride)
                stride *= size
            return mx.as_strided(
                packed,
                shape=shape,
                strides=(
                    expert_stride // 4,
                    *reversed(row_strides),
                ),
                offset=region.offset // 4,
            )
        if (
            expert_stride % 4
            or region.offset % 4
            or region.shape[-1] % 4
        ):
            raise ValueError("batched expert regions must be 4-byte aligned")
        packed_shape = (*region.shape[:-1], region.shape[-1] // 4)
        shape = (self._model.expert_count, *packed_shape)
        row_strides = []
        stride = 1
        for size in reversed(packed_shape):
            row_strides.append(stride)
            stride *= size
        value = mx.as_strided(
            packed,
            shape=shape,
            strides=(expert_stride // 4, *reversed(row_strides)),
            offset=region.offset // 4,
        )
        return value if name.endswith(".weight") else value.view(mx.uint8)


class _StagedSlotPool(_SlotPool):
    """Split each direct slot into fixed w13 and w2 Metal-visible arrays."""

    def __init__(self, model: InstalledModel, slots: int) -> None:
        self._model = model
        self._source_regions = {
            region.name: region for region in model.expert_regions
        }
        self._layout = support_for_installed(model).expert_layout
        self._regions = self._layout.regions(model)
        self._w13_regions, self._w2_regions = _staged_slot_regions(model)
        self._batched_layout = batched_layer_layout(
            self._regions, self._layout.batched_regions,
            model.expert_blob_size, model.expert_count,
        )
        self._w13_size = sum(
            region.length
            for name, region in self._w13_regions.items()
            if not name.startswith("w13")
        )
        self._w2_size = sum(region.length for region in self._w2_regions.values())
        self._w13_slots: list[mx.array | None] = [None] * slots
        self._w2_slots: list[mx.array | None] = [None] * slots
        self._w13_views: list[memoryview | None] = [None] * slots
        self._w2_views: list[memoryview | None] = [None] * slots
        self._loaded = bytearray(slots)

    def prepare(self, slots: list[int]) -> None:
        created = []
        for slot in slots:
            if self._w13_slots[slot] is None:
                self._w13_slots[slot] = mx.empty(
                    (self._w13_size,),
                    dtype=mx.uint8,
                )
                self._w2_slots[slot] = mx.empty(
                    (self._w2_size,),
                    dtype=mx.uint8,
                )
                created.extend(
                    [self._w13_slots[slot], self._w2_slots[slot]]
                )
        if created:
            mx.eval(*created)
            for slot in slots:
                w13 = self._w13_slots[slot]
                w2 = self._w2_slots[slot]
                if self._w13_views[slot] is None and w13 is not None:
                    self._w13_views[slot] = memoryview(w13)
                if self._w2_views[slot] is None and w2 is not None:
                    self._w2_views[slot] = memoryview(w2)

    @staticmethod
    def _array_from(
        packed: mx.array,
        region: Tensor,
    ) -> mx.array:
        value = packed[region.offset : region.offset + region.length]
        value = value.reshape(region.shape)
        if region.dtype == "I8":
            value = value.view(mx.int8)
        return value.view(mx.uint32) if region.name.endswith(".weight") else value

    def _weights(self, slot: int, require_loaded: bool) -> ExpertWeights:
        w13 = self._w13_slots[slot]
        w2 = self._w2_slots[slot]
        if w13 is None or w2 is None or (require_loaded and not self._loaded[slot]):
            raise RuntimeError("staged expert slot is empty")
        return ExpertWeights(
            w1=self._array_from(w13, self._w13_regions["w1.weight"]),
            w1_scales=self._array_from(w13, self._w13_regions["w1.scale"]),
            w2=self._array_from(w2, self._w2_regions["w2.weight"]),
            w2_scales=self._array_from(w2, self._w2_regions["w2.scale"]),
            w3=self._array_from(w13, self._w13_regions["w3.weight"]),
            w3_scales=self._array_from(w13, self._w13_regions["w3.scale"]),
            w13=self._array_from(w13, self._w13_regions["w13.weight"]),
            w13_scales=self._array_from(w13, self._w13_regions["w13.scale"]),
        )

    def select_individual(self, slots: list[int]) -> tuple[ExpertWeights, ...]:
        return tuple(self._weights(slot, require_loaded=True) for slot in slots)

    def select_staged_individual(
        self,
        slots: list[int],
    ) -> tuple[ExpertWeights, ...]:
        return tuple(self._weights(slot, require_loaded=False) for slot in slots)

    def store(self, slots: list[int], blobs: list[bytes]) -> float:
        started = time.perf_counter()
        self.prepare(slots)
        for slot, blob in zip(slots, blobs):
            if len(blob) != self._model.expert_blob_size:
                raise ValueError("expert blob size does not match the manifest")
            source = memoryview(blob)
            for name, region in self._source_regions.items():
                self.writable_region(slot, name)[:] = source[
                    region.offset : region.offset + region.length
                ]
            self._loaded[slot] = 1
        return time.perf_counter() - started

    def writable_region(self, slot: int, name: str) -> memoryview:
        if name in self._w13_regions:
            region = self._w13_regions[name]
            view = self._w13_views[slot]
        else:
            region = self._w2_regions[name]
            view = self._w2_views[slot]
        if view is None:
            raise RuntimeError("staged expert slot buffer is not prepared")
        return view[region.offset : region.offset + region.length]

    def mark_loaded(self, slot: int) -> None:
        self._loaded[slot] = 1

    def mark_empty(self, slot: int) -> None:
        self._loaded[slot] = 0


class ExpertCache:
    """A fixed-size, layer-aware LFU/LRU slot pool for routed expert weights."""

    def __init__(
        self,
        installed_model: InstalledModel,
        slots: int = 1_152,
        read_workers: int = 4,
        prefetch_read_workers: int = 2,
        layer_count: int | None = None,
        expert_directory: Path | None = None,
        route_trace_path: str | Path | None = None,
        ready_expert_decode: bool = False,
        read_limiter: _ReadLimiter | None = None,
        page_cache_probe: bool = False,
        file_cache_policy: str = "cached",
        staged_expert_streaming: bool = False,
        eviction_policy: str = "lfu",
        separate_prefill_io: bool = False,
        prefill_read_experts: int = 1,
        prefill_seed_experts: int = 0,
    ) -> None:
        if slots < installed_model.selected_expert_count:
            raise ValueError("slot count must hold at least one token's routed experts")
        if read_workers < 1:
            raise ValueError("read worker count must be greater than zero")
        if prefetch_read_workers < 1:
            raise ValueError("prefetch read worker count must be greater than zero")
        if file_cache_policy not in EXPERT_FILE_CACHE_POLICIES:
            raise ValueError(
                f"unknown expert-file cache policy: {file_cache_policy}"
            )
        if eviction_policy not in ("lfu", "lru", "route"):
            raise ValueError(f"unknown expert eviction policy: {eviction_policy}")
        if type(prefill_read_experts) is not int or not 1 <= prefill_read_experts <= 32:
            raise ValueError("prefill read batch must be an integer from 1 through 32")
        if type(prefill_seed_experts) is not int or not 0 <= prefill_seed_experts <= 128:
            raise ValueError("prefill seed count must be an integer from 0 through 128")
        if ((prefill_read_experts != 1 or prefill_seed_experts)
                and (installed_model.model_kind != "qwen3.8-flash-next" or staged_expert_streaming)):
            raise ValueError("prefill read/seed experiments require direct Qwen slots")
        self.prefill_read_experts = prefill_read_experts
        self.prefill_seed_experts = prefill_seed_experts
        self._prefill_seed_routes = None
        self._seeded_keys: set[tuple[int, int]] = set()
        self.eviction_policy = eviction_policy
        self.model = installed_model
        self.slots = slots
        self._decode_slots = slots
        self._prefill_slots = slots
        self._phase_memory_active = False
        self._memory_phase = None
        self._phase_resize_count = 0
        self.read_workers = read_workers
        self.prefetch_read_workers = min(read_workers, prefetch_read_workers)
        self.layer_count = layer_count or installed_model.layer_count
        self.expert_directory = expert_directory or installed_model.root / "experts"
        self.ready_expert_decode = ready_expert_decode
        self._read_limiter = read_limiter
        self.page_cache_probe = page_cache_probe
        self.file_cache_policy = file_cache_policy
        self.staged_expert_streaming = staged_expert_streaming
        filesystem = os.statvfs(self.expert_directory)
        self.direct_io_alignment = (
            int(filesystem.f_frsize or filesystem.f_bsize)
            if file_cache_policy == "bypass"
            else 0
        )
        if self.direct_io_alignment < 0:
            raise ValueError("direct-I/O alignment cannot be negative")
        if self.layer_count < 1:
            raise ValueError("expert cache layer count must be greater than zero")
        from .route_cache import RouteCachePolicy
        self._route_policy = (RouteCachePolicy(self.layer_count, installed_model.expert_count,
                                               slots, installed_model.selected_expert_count)
                              if eviction_policy == "route" else None)
        self._route_phase = "decode"
        self.metrics = CacheMetrics()
        self._pool = (
            _StagedSlotPool(installed_model, slots)
            if staged_expert_streaming
            else _SlotPool(installed_model, slots)
        )
        self._entries: dict[tuple[int, int], _Entry] = {}
        self._free_slots = list(reversed(range(slots)))
        self._heap: list[tuple[int, int, int, int, int]] = []
        self._route_heaps = [[] for _ in range(self.layer_count)] if self._route_policy is not None else []
        self._layer_counts = [0] * self.layer_count
        base = slots // self.layer_count
        self._layer_reserve = base // 2
        self._clock = 0
        self._last_decay = 0
        self._pinned_layers: set[int] = set()
        self._pinned_expert_keys: set[tuple[int, int]] = set()
        self._expert_union_profiles: list[ExpertUnionProfile] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=read_workers)
        self._staged_w2_executor = (
            ThreadPoolExecutor(max_workers=read_workers)
            if staged_expert_streaming
            else None
        )
        self._prefetched_layers: dict[int, _LayerRead] = {}
        self._layer_buffers: list[mx.array] | None = None
        self._batched_layer: tuple[int, BatchedExperts] | None = None
        self._active_prefetch_trace: _ActivePrefetchTrace | None = None
        self._route_trace_path = route_trace_path
        if route_trace_path is not None:
            from .route_trace import RouteTraceRecorder

            self._route_trace = RouteTraceRecorder(
                self.layer_count,
                installed_model.expert_count,
                installed_model.selected_expert_count,
                installed_model.expert_blob_size,
            )
        else:
            self._route_trace = None
        self._descriptors: list[int] = []
        self._prefill_reader = None
        try:
            for layer in range(self.layer_count):
                descriptor = os.open(
                    self.expert_directory / f"layer_{layer:02d}.bin",
                    os.O_RDONLY,
                )
                try:
                    configure_expert_file_cache_policy(
                        descriptor,
                        self.file_cache_policy,
                    )
                except Exception:
                    os.close(descriptor)
                    raise
                self._descriptors.append(descriptor)
            if separate_prefill_io:
                from .prefill_io import PrefillReader
                self._prefill_reader = PrefillReader(
                    self.expert_directory, self.layer_count,
                    self.model.expert_blob_size, read_limiter, batch_experts=prefill_read_experts,
                )
        except Exception:
            for descriptor in self._descriptors:
                os.close(descriptor)
            self._executor.shutdown(wait=False, cancel_futures=True)
            if self._staged_w2_executor is not None:
                self._staged_w2_executor.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
            raise

    def close(self) -> None:
        for job in self._prefetched_layers.values():
            for future in job.futures:
                future.cancel()
        self._prefetched_layers.clear()
        self._executor.shutdown(wait=True)
        if self._staged_w2_executor is not None:
            self._staged_w2_executor.shutdown(wait=True)
        if self._prefill_reader is not None:
            self._prefill_reader.close()
        for descriptor in self._descriptors:
            os.close(descriptor)
        self._descriptors.clear()
        if self._route_trace is not None and self._route_trace_path is not None:
            self._route_trace.write(self._route_trace_path)

    def __enter__(self) -> ExpertCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def resident_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def release_prefill_slots(self) -> None:
        """Release decode storage before batched prefill; caller holds the request lock."""
        with self._lock:
            if (
                self._pinned_layers
                or self._pinned_expert_keys
                or self._batched_layer is not None
            ):
                raise RuntimeError("cannot release expert slots while they are in use")
        self.discard_prefetched_layers()
        mx.synchronize()
        with self._lock:
            # Preserve the direct or staged storage layout.
            self._pool = type(self._pool)(self.model, self.slots)
            self._entries.clear()
            self._seeded_keys.clear()
            self._prefill_seed_routes = None
            self._free_slots = list(reversed(range(self.slots)))
            self._heap.clear()
            if self._route_policy is not None:
                self._route_heaps = [[] for _ in range(self.layer_count)]
            self._layer_counts = [0] * self.layer_count
            self._clock = 0
            self._last_decay = 0
        mx.clear_cache()

    def enable_phase_memory(self) -> None:
        """Opt-in: half capacity during prefill, a full layer when possible.

        Does not allocate workspace or increase the configured expert ceiling.
        Called at model load, before a request can own expert weights.
        """
        if type(self._pool) is not _SlotPool:
            raise ValueError("phase memory requires direct expert storage")
        from .qwen_phase_budget import QwenPhaseBudget
        prefill = min(self._decode_slots, max(self.model.expert_count, self._decode_slots // 2))
        blob = self.model.expert_blob_size
        plan = QwenPhaseBudget(self._decode_slots * blob, blob, self.model.selected_expert_count,
                               (self._decode_slots - prefill) * blob)
        self._prefill_slots = plan.prefill_slots

    @contextmanager
    def phase_memory_request(self):
        # The request lock and generation MLX stream enclose this scope.
        if self._prefill_slots == self._decode_slots:
            yield
            return
        if self._phase_memory_active:
            raise RuntimeError("phase memory request is already active")
        self._phase_memory_active = True
        self._memory_phase = None
        try:
            yield
        finally:
            try:
                self.set_memory_phase("decode")
            finally:
                self._phase_memory_active = False
                self._memory_phase = None

    def set_memory_phase(self, phase: str) -> None:
        if not self._phase_memory_active or self._memory_phase == phase:
            return
        if phase not in ("prefill", "decode"):
            raise ValueError("unknown expert memory phase")
        capacity = self._prefill_slots if phase == "prefill" else self._decode_slots
        if capacity != self.slots:
            self._resize_phase_slots(capacity)
        self._memory_phase = phase

    def _resize_phase_slots(self, capacity: int) -> None:
        if not self.model.selected_expert_count <= capacity <= self._decode_slots:
            raise ValueError("phase capacity exceeds the configured expert budget")
        with self._lock:
            if self._pinned_layers or self._pinned_expert_keys or self._batched_layer is not None:
                raise RuntimeError("cannot resize expert slots while they are in use")
        # Demand reads finish before the generation iterator yields. Prefetches
        # can outlive it: cancel/drain those before dropping their buffers.
        self.discard_prefetched_layers()
        mx.synchronize()
        previous = self.slots
        with self._lock:
            removed = [key for key, entry in self._entries.items() if entry.slot >= capacity]
            for key in removed:
                del self._entries[key]
                self._seeded_keys.discard(key)
                self._layer_counts[key[0]] -= 1
            self.metrics.evictions += len(removed)
            self._pool.resize(capacity)
            self.slots = capacity
            occupied = {entry.slot for entry in self._entries.values()}
            self._free_slots = [slot for slot in reversed(range(capacity)) if slot not in occupied]
            self._layer_reserve = (capacity // self.layer_count) // 2
            if self._route_policy is not None:
                self._route_policy.capacity = min(capacity, self.layer_count * self.model.expert_count)
                self._route_policy.rebalance()
                self._rebuild_route_heap_locked()
            else:
                self._heap = [(self._eviction_rank(entry), entry.last_access, entry.version, layer, expert)
                              for (layer, expert), entry in self._entries.items()]
                heapq.heapify(self._heap)
            self._phase_resize_count += 1
        if capacity < previous:
            mx.clear_cache()

    def phase_memory_snapshot(self) -> dict:
        with self._lock:
            return dict(prefill_slots=self._prefill_slots, decode_slots=self._decode_slots,
                        active_slots=self.slots, resize_count=self._phase_resize_count)

    def discard_prefetched_layers(self) -> None:
        """Abandon unused layer reads; caller holds the request lock."""
        with self._lock:
            pending = list(self._prefetched_layers.values())
            self._prefetched_layers.clear()
        cancel_and_drain(future for job in pending for future in job.futures)

    @contextmanager
    def reuse_layer_buffers(self):
        """Request-local storage; no pooled layer survives prefill or cancellation."""
        if self._layer_buffers is not None or self._batched_layer is not None:
            raise RuntimeError("batched prefill buffers are already in use")
        self._layer_buffers = []
        try:
            yield
        finally:
            self.discard_prefetched_layers()
            mx.synchronize()
            self._layer_buffers = None
            mx.clear_cache()

    def resident_expert_keys(
        self,
        experts_by_layer: dict[int, list[int] | tuple[int, ...]],
    ) -> frozenset[tuple[int, int]]:
        """Snapshot which requested experts are currently in the main LFU cache."""
        requested = {
            (layer, expert)
            for layer, experts in experts_by_layer.items()
            for expert in experts
        }
        with self._lock:
            return frozenset(requested.intersection(self._entries))

    def metrics_snapshot(self) -> CacheMetrics:
        with self._lock:
            return replace(self.metrics)

    def prefill_io_snapshot(self) -> dict | None:
        return self._prefill_reader.snapshot() if self._prefill_reader is not None else None

    def record_routing_sync(self, seconds: float) -> None:
        with self._lock:
            self.metrics.routing_sync_seconds += seconds

    def record_gather_qmm(self, calls: int = 3) -> None:
        with self._lock:
            self.metrics.gather_qmm_calls += calls

    def record_staged_first_stage_submit(self, seconds: float) -> None:
        with self._lock:
            self.metrics.staged_first_stage_submit_seconds += seconds

    @property
    def route_trace_enabled(self) -> bool:
        return self._route_trace is not None

    @contextmanager
    def trace_routes(self, phase: str):
        previous = self._route_phase
        self._route_phase = phase
        try:
            if self._route_trace is None:
                yield
            else:
                with self._route_trace.phase(phase):
                    yield
        finally:
            self._route_phase = previous
            if self._route_policy is not None and phase == "prefill" and previous != "prefill":
                with self._lock:
                    self._route_policy.rebalance()
                    self._rebuild_route_heap_locked()

    def begin_route_request(self) -> None:
        # One prompt may have several Prefill phases (layer-major + final token).
        if self._route_policy is not None:
            with self._lock:
                self._route_policy.begin_prefill()

    @property
    def route_cache_enabled(self) -> bool:
        return self._route_policy is not None

    def observe_batched_routes(self, layer: int, selected: np.ndarray) -> None:
        # Normal acquisitions observe at get_many/iter_ready; batched Prefill has no slots.
        if self._route_policy is not None:
            with self._lock:
                self._observe_access_locked(layer, selected)

    def route_cache_snapshot(self) -> dict | None:
        with self._lock:
            if self._route_policy is None:
                return None
            result = self._route_policy.snapshot()
            for row, resident in zip(result["layers"], self._layer_counts):
                row["resident_slots"] = resident
            return result

    def _observe_access_locked(self, layer, selected):
        if self._route_policy is not None and self._route_policy.observe(layer, selected, self._route_phase):
            self._rebuild_route_heap_locked()

    def _rebuild_route_heap_locked(self):
        self._route_heaps = [[] for _ in range(self.layer_count)]
        for (layer, expert), entry in self._entries.items():
            entry.version += 1
            self._route_heaps[layer].append((float(self._route_policy.scores[layer, expert]),
                                            entry.last_access, entry.version, layer, expert))
        for heap in self._route_heaps:
            heapq.heapify(heap)

    def _evict_route_locked(self, protected, incoming_layer):
        # First inspect over-budget layers. Lower tiers matter only if every
        # candidate in the preceding tier is protected; avoid touching their heaps.
        tiers = ([], [], [])
        for layer, (count, quota) in enumerate(zip(self._layer_counts, self._route_policy.quotas.tolist())):
            if not count:
                continue
            eligible = count > quota or (layer == incoming_layer and count >= quota)
            category = 2 if layer in self._pinned_layers else (0 if eligible else 1)
            tiers[category].append(layer)
        held = []
        candidates = []
        winner = None
        try:
            for layers in tiers:
                for layer in layers:
                    heap = self._route_heaps[layer]
                    while heap:
                        item = heapq.heappop(heap)
                        key = (layer, item[4])
                        entry = self._entries.get(key)
                        if entry is None or (self._eviction_rank(entry), entry.last_access, entry.version) != item[:3]:
                            continue
                        held.append(item)
                        if key in protected or key in self._pinned_expert_keys:
                            continue
                        candidates.append(item)
                        break
                if candidates:
                    break
            if not candidates:
                raise RuntimeError("no expert cache slot can be evicted")
            winner = min(candidates)
            key = (winner[3], winner[4])
            return key, self._entries[key]
        finally:
            for item in held:
                if item != winner:
                    heapq.heappush(self._route_heaps[item[3]], item)

    def record_routes(self, layer: int, selected: np.ndarray) -> None:
        active = self._active_prefetch_trace
        if active is not None and active.layer == layer:
            active.used_experts.update(
                int(expert) for expert in np.asarray(selected).reshape(-1)
            )
        if self._route_trace is not None:
            self._route_trace.record(layer, selected)

    @contextmanager
    def capture_expert_unions(self) -> Iterator[ExpertUnionProfile]:
        """Capture the unique expert set used by each layer in one forward."""
        profile = ExpertUnionProfile()
        with self._lock:
            self._expert_union_profiles.append(profile)
        try:
            yield profile
        finally:
            with self._lock:
                for index, active in enumerate(self._expert_union_profiles):
                    if active is profile:
                        del self._expert_union_profiles[index]
                        break

    def _record_expert_union_locked(
        self,
        layer: int,
        routed_expert_assignments: int,
        unique_experts: int,
        cache_misses: int,
    ) -> None:
        self.metrics.expert_union_calls += 1
        self.metrics.routed_expert_assignments += routed_expert_assignments
        self.metrics.expert_union_experts += unique_experts
        self.metrics.expert_union_misses += cache_misses
        if not self._expert_union_profiles:
            return
        metrics = ExpertUnionLayerMetrics(
            layer=layer,
            routed_expert_assignments=routed_expert_assignments,
            unique_experts=unique_experts,
            cache_misses=cache_misses,
        )
        for profile in self._expert_union_profiles:
            profile.layers.append(metrics)

    def _record_residency(
        self,
        layer: int,
        selected: list[int],
        missing: list[int],
    ) -> None:
        if self._route_trace is not None:
            self._route_trace.record_residency(layer, selected, missing)

    def current_batched(self, layer: int) -> BatchedExperts | None:
        current = self._batched_layer
        return current[1] if current is not None and current[0] == layer else None

    def record_compute_submit(self, layer: int) -> None:
        active = self._active_prefetch_trace
        if (
            active is not None
            and active.layer == layer
            and active.compute_submit is None
        ):
            active.compute_submit = time.perf_counter()

    def prefetch_layer(
        self,
        layer: int,
        experts: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        check_cancelled()
        if not 0 <= layer < self.layer_count:
            return
        selected = (
            tuple(range(self.model.expert_count))
            if experts is None
            else tuple(sorted(set(experts)))
        )
        if not selected:
            raise ValueError("batched expert prefetch cannot be empty")
        if any(not 0 <= expert < self.model.expert_count for expert in selected):
            raise ValueError("batched expert prefetch contains an invalid expert")
        with self._lock:
            existing = self._prefetched_layers.get(layer)
            if existing is not None:
                if existing.experts != selected:
                    raise RuntimeError(
                        "batched expert layer is already prefetched with a "
                        "different expert set"
                    )
                return
            length = self.model.expert_count * self.model.expert_blob_size
            if length % 4:
                raise ValueError("batched expert layer must be 4-byte aligned")
            if self._layer_buffers:
                packed = self._layer_buffers.pop()
            else:
                if (self._layer_buffers is not None
                        and len(self._prefetched_layers) + int(self._batched_layer is not None) >= 2):
                    raise RuntimeError("batched prefill allows only two layer buffers")
                packed = mx.empty((length // 4,), dtype=mx.uint32)
                mx.eval(packed)
            step = (
                len(selected) + self.prefetch_read_workers - 1
            ) // self.prefetch_read_workers
            batches = [
                selected[start : start + step]
                for start in range(0, len(selected), step)
            ]
            submitted = time.perf_counter()
            self._prefetched_layers[layer] = _LayerRead(
                packed,
                tuple(
                    self._executor.submit(
                        self._read_expert_ids,
                        layer,
                        packed,
                        batch,
                    )
                    for batch in batches
                ),
                selected,
                submitted,
            )

    @contextmanager
    def batched_layer(
        self,
        layer: int,
        enabled: bool = True,
        experts: list[int] | tuple[int, ...] | None = None,
    ):
        if not enabled:
            yield None
            return
        self.prefetch_layer(layer, experts)
        with self._lock:
            job = self._prefetched_layers.pop(layer)
        deadline = time.perf_counter()
        was_ready = all(future.done() for future in job.futures)
        wait_started = time.perf_counter()
        reads = wait_for_futures(job.futures)
        future_wait_seconds = time.perf_counter() - wait_started
        elapsed = max(
            (read.finished - read.started for read in reads),
            default=0.0,
        )
        packed = job.packed
        batched = self._pool.batched_layer(packed)
        with self._lock:
            self.metrics.bytes_read += len(job.experts) * self.model.expert_blob_size
            self.metrics.read_seconds += elapsed
            self.metrics.wait_seconds += future_wait_seconds
            self.metrics.batched_layers += 1
            if was_ready:
                self.metrics.prefetched_layer_hits += 1
        self._batched_layer = (layer, batched)
        self._active_prefetch_trace = _ActivePrefetchTrace(
            layer=layer,
            job=job,
            reads=reads,
            deadline=deadline,
            future_wait_seconds=future_wait_seconds,
        )
        self._prefill_seed_routes = None
        try:
            yield batched
            if self.prefill_seed_experts and self._prefill_seed_routes is not None:
                self._seed_from_batched_layer(layer, packed, job.experts)
        finally:
            self._prefill_seed_routes = None
            # Reads into this storage may start as soon as it returns to the
            # pool. Fence GPU consumers even when prefill raises or is cancelled.
            if self._layer_buffers is not None:
                mx.synchronize()
                self._layer_buffers.append(packed)
            active = self._active_prefetch_trace
            if active is not None and self._route_trace is not None:
                self._route_trace.record_prefetch_event(
                    layer=active.layer,
                    requested_experts=len(active.job.experts),
                    used_experts=len(active.used_experts),
                    read_submit=active.job.submitted,
                    read_start=min(read.started for read in active.reads),
                    read_complete=max(read.finished for read in active.reads),
                    expert_deadline=active.deadline,
                    compute_submit=active.compute_submit,
                    future_wait_seconds=active.future_wait_seconds,
                )
            self._active_prefetch_trace = None
            self._batched_layer = None


    def capture_prefill_seed_routes(self, layer: int, routes) -> None:
        # Keep at most 128 token positions across chunks, without a host sync.
        # Layer-major prefill evaluates each chunk before the next call.
        if self.prefill_seed_experts and self._batched_layer is not None:
            if self._batched_layer[0] != layer:
                raise RuntimeError("prefill seed layer does not own the batched buffer")
            if routes.ndim != 3 or routes.shape[0] != 1:
                raise ValueError("prefill retention requires batch-one routing")
            tail = routes[:, -128:]
            parts = list(self._prefill_seed_routes or ())
            parts.append(tail)
            excess = sum(part.shape[1] for part in parts) - 128
            while excess > 0:
                if parts[0].shape[1] <= excess:
                    excess -= parts.pop(0).shape[1]
                else:
                    parts[0] = parts[0][:, excess:]
                    excess = 0
            self._prefill_seed_routes = parts

    def _seed_from_batched_layer(self, layer, packed, loaded_experts) -> None:
        from .qwen_streaming_policy import hot_tail_experts
        check_cancelled()
        started = time.perf_counter()
        # Drain submitted GPU work. Layer-major callers evaluate each chunk output.
        mx.synchronize()
        parts = self._prefill_seed_routes
        routes = np.asarray(parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1))
        allowed = set(loaded_experts)
        quota = min(self.prefill_seed_experts, self.slots // self.layer_count)
        selected = [e for e in hot_tail_experts(routes, self.model.expert_count, quota)
                    if e in allowed]
        with self._lock:
            selected = [e for e in selected if (layer, e) not in self._entries]
            # Do not evict demand residents to make room for a prediction.
            selected = selected[-len(self._free_slots):] if self._free_slots else []
            if not selected:
                return
            try:
                assigned = self._reserve_slots(layer, selected, Counter({e: 1 for e in selected}), set())
            except BaseException:
                partial = {e: self._entries[(layer, e)].slot for e in selected if (layer, e) in self._entries}
                self._release_slots(layer, partial)
                raise
        source = None
        try:
            source = memoryview(packed).cast("B")
            for expert, slot in assigned.items():
                check_cancelled()
                regions = self._pool.layer_write_views(source, expert)
                try:
                    for descriptor, region in zip(self.model.expert_regions, regions):
                        destination = self._pool.writable_region(slot, descriptor.name)
                        try:
                            destination[:] = region
                        finally:
                            destination.release()
                finally:
                    for region in regions:
                        region.release()
            with self._lock:
                for expert, slot in assigned.items():
                    self._pool.mark_loaded(slot)
                    self._seeded_keys.add((layer, expert))
                self.metrics.prefill_seeded_experts += len(assigned)
                self.metrics.prefill_seed_bytes += len(assigned) * self.model.expert_blob_size
                self.metrics.prefill_seed_seconds += time.perf_counter() - started
        except BaseException:
            with self._lock:
                self._release_slots(layer, assigned)
            raise
        finally:
            if source is not None:
                source.release()

    def record_shared_overlap(self) -> None:
        with self._lock:
            self.metrics.shared_overlap_submissions += 1

    @contextmanager
    def pin_layer(self, layer: int):
        with self._lock:
            self._pinned_layers.add(layer)
        try:
            yield
        finally:
            with self._lock:
                self._pinned_layers.discard(layer)


    def get_many(self, layer: int, expert_ids: list[int]) -> ResidentExperts:
        check_cancelled()
        if not 0 <= layer < self.layer_count:
            raise ValueError(f"invalid layer {layer}")
        frequencies = Counter(expert_ids)
        unique = sorted(frequencies)
        if len(unique) > self.slots:
            raise ValueError(
                f"prefill selected {len(unique)} experts, but the cache has {self.slots} slots"
            )
        for expert in unique:
            if not 0 <= expert < self.model.expert_count:
                raise ValueError(f"invalid expert {expert}")

        missing: list[int] = []
        resident: list[int] = []
        protected = {(layer, expert) for expert in unique}
        with self._lock:
            self._observe_access_locked(layer, expert_ids)
            for expert in unique:
                entry = self._entries.get((layer, expert))
                if entry is None:
                    self.metrics.misses += 1
                    missing.append(expert)
                else:
                    self.metrics.hits += 1
                    self._touch(layer, expert, entry, frequencies[expert])
                    resident.append(expert)
            self._record_expert_union_locked(
                layer,
                len(expert_ids),
                len(unique),
                len(missing),
            )
            assigned = self._reserve_slots(layer, missing, frequencies, protected)
        self._record_residency(layer, expert_ids, missing)

        started = time.perf_counter()
        futures = {
            expert: self._executor.submit(
                self._read_expert_into_slot,
                layer,
                expert,
                assigned[expert],
            )
            for expert in missing
        }
        wait_started = time.perf_counter()
        try:
            wait_for_futures(futures.values())
        except Exception:
            with self._lock:
                self._release_slots(layer, assigned)
            raise
        waited = time.perf_counter() - wait_started if missing else 0.0
        elapsed = time.perf_counter() - started if missing else 0.0
        with self._lock:
            self.metrics.wait_seconds += waited
            self.metrics.bytes_read += len(missing) * self.model.expert_blob_size
            self.metrics.read_seconds += elapsed
            for slot in assigned.values():
                self._pool.mark_loaded(slot)
            self._decay_if_needed()
            pack_started = time.perf_counter()
            main_experts = [*resident, *missing]
            main_weights = self._pool.select_individual(
                [self._entries[(layer, expert)].slot for expert in main_experts]
            )
            self.metrics.pack_seconds += time.perf_counter() - pack_started
            weights_by_expert = dict(zip(main_experts, main_weights))
            individual_weights = tuple(weights_by_expert[expert] for expert in unique)
            return ResidentExperts(
                individual_weights,
                {expert: slot for slot, expert in enumerate(unique)},
            )

    def iter_ready(
        self,
        layer: int,
        expert_ids: list[int],
    ) -> Iterator[tuple[int, ExpertWeights]]:
        """Yield resident and newly read experts without waiting for the slowest read."""
        if not 0 <= layer < self.layer_count:
            raise ValueError(f"invalid layer {layer}")
        frequencies = Counter(expert_ids)
        unique = sorted(frequencies)
        if len(unique) > self.slots:
            raise ValueError("selected experts exceed the expert slot count")
        if any(not 0 <= expert < self.model.expert_count for expert in unique):
            raise ValueError("selected experts contain an invalid expert ID")

        resident: list[tuple[int, int]] = []
        missing: list[int] = []
        protected = {(layer, expert) for expert in unique}
        with self._lock:
            self._observe_access_locked(layer, expert_ids)
            for expert in unique:
                entry = self._entries.get((layer, expert))
                if entry is None:
                    self.metrics.misses += 1
                    missing.append(expert)
                else:
                    self.metrics.hits += 1
                    self._touch(layer, expert, entry, frequencies[expert])
                    resident.append((expert, entry.slot))
            self._record_expert_union_locked(
                layer,
                len(expert_ids),
                len(unique),
                len(missing),
            )
            assigned = self._reserve_slots(layer, missing, frequencies, protected)
        self._record_residency(layer, expert_ids, missing)

        started = time.perf_counter()
        futures = {
            self._executor.submit(
                self._read_expert_into_slot,
                layer,
                expert,
                assigned[expert],
            ): expert
            for expert in missing
        }
        ready_at = started
        completed: set[int] = set()
        try:
            # Closing while yielding a cache hit must also drain missing reads.
            for expert, slot in resident:
                pack_started = time.perf_counter()
                weights = self._pool.select_individual([slot])[0]
                with self._lock:
                    self.metrics.pack_seconds += time.perf_counter() - pack_started
                yield expert, weights
            pending = iter(as_completed(futures))
            for _ in range(len(futures)):
                wait_started = time.perf_counter()
                future = next(pending)
                with self._lock:
                    self.metrics.wait_seconds += time.perf_counter() - wait_started
                expert = futures[future]
                finished = future.result()
                ready_at = max(ready_at, finished)
                completed.add(expert)
                with self._lock:
                    slot = assigned[expert]
                    self._pool.mark_loaded(slot)
                    pack_started = time.perf_counter()
                    weights = self._pool.select_individual([slot])[0]
                    self.metrics.pack_seconds += time.perf_counter() - pack_started
                yield expert, weights
        finally:
            incomplete = {
                expert: assigned[expert]
                for expert in missing
                if expert not in completed
            }
            if incomplete:
                for future, expert in futures.items():
                    if expert in incomplete:
                        future.cancel()
                for future, expert in futures.items():
                    if expert in incomplete:
                        try:
                            future.result()
                        except Exception:
                            pass
                with self._lock:
                    self._release_slots(layer, incomplete)

        with self._lock:
            self.metrics.bytes_read += len(completed) * self.model.expert_blob_size
            self.metrics.read_seconds += ready_at - started if missing else 0.0
            self._decay_if_needed()

    def iter_staged_ready(
        self,
        layer: int,
        expert_ids: list[int],
    ) -> Iterator[StagedReadyExpert]:
        """Yield w13-ready split slots and admit each only after w2 succeeds."""
        pool = self._pool
        w2_executor = self._staged_w2_executor
        if (
            not self.staged_expert_streaming
            or not isinstance(pool, _StagedSlotPool)
            or w2_executor is None
        ):
            raise RuntimeError("staged expert streaming is not configured")
        if not 0 <= layer < self.layer_count:
            raise ValueError(f"invalid layer {layer}")
        frequencies = Counter(expert_ids)
        unique = sorted(frequencies)
        if len(unique) > self.slots:
            raise ValueError("selected experts exceed the expert slot count")
        if any(not 0 <= expert < self.model.expert_count for expert in unique):
            raise ValueError("selected experts contain an invalid expert ID")

        resident: list[tuple[int, int]] = []
        missing: list[int] = []
        protected = {(layer, expert) for expert in unique}
        with self._lock:
            self._observe_access_locked(layer, expert_ids)
            for expert in unique:
                entry = self._entries.get((layer, expert))
                if entry is None:
                    self.metrics.misses += 1
                    missing.append(expert)
                else:
                    self.metrics.hits += 1
                    self._touch(layer, expert, entry, frequencies[expert])
                    resident.append((expert, entry.slot))
            self._record_expert_union_locked(
                layer,
                len(expert_ids),
                len(unique),
                len(missing),
            )
            assigned = self._reserve_slots(layer, missing, frequencies, protected)
        self._record_residency(layer, expert_ids, missing)

        started = time.perf_counter()
        begin_futures = {
            self._executor.submit(
                self._begin_staged_expert_read,
                layer,
                expert,
                assigned[expert],
            ): expert
            for expert in missing
        }
        for expert, slot in resident:
            pack_started = time.perf_counter()
            weights = pool.select_individual([slot])[0]
            with self._lock:
                self.metrics.pack_seconds += time.perf_counter() - pack_started
            yield StagedReadyExpert(self, expert, weights, slot)

        ready_at = started
        completed: set[int] = set()
        handles: dict[int, StagedReadyExpert] = {}
        try:
            for future in as_completed(begin_futures):
                expert = begin_futures[future]
                w13_read, w2_future = future.result()
                slot = assigned[expert]
                pack_started = time.perf_counter()
                weights = pool.select_staged_individual([slot])[0]
                with self._lock:
                    self.metrics.pack_seconds += (
                        time.perf_counter() - pack_started
                    )
                handle = StagedReadyExpert(
                    self,
                    expert,
                    weights,
                    slot,
                    w13_read,
                    w2_future,
                )
                handles[expert] = handle
                yield handle
                if not handle.finished:
                    handle.finish_w2()
                if handle.w2_read is None:
                    raise RuntimeError("staged expert completed without w2 read")
                ready_at = max(ready_at, handle.w2_read.finished)
                if self._route_policy is not None:
                    with self._lock:
                        self._route_policy.record_read(layer,
                            w13_read.finished - w13_read.started +
                            handle.w2_read.finished - handle.w2_read.started)
                completed.add(expert)
        finally:
            for future, expert in begin_futures.items():
                if expert not in completed:
                    future.cancel()
            for future, expert in begin_futures.items():
                if expert in completed:
                    continue
                try:
                    _, pending_w2 = future.result()
                except Exception:
                    continue
                pending_w2.cancel()
                try:
                    pending_w2.result()
                except Exception:
                    pass
            for expert, handle in handles.items():
                if expert in completed or handle._w2_future is None:
                    continue
                handle._w2_future.cancel()
                try:
                    handle._w2_future.result()
                except Exception:
                    pass
            incomplete = {
                expert: assigned[expert]
                for expert in missing
                if expert not in completed
            }
            if incomplete:
                with self._lock:
                    self._release_slots(layer, incomplete)

        with self._lock:
            completed_count = len(completed)
            self.metrics.bytes_read += completed_count * self.model.expert_blob_size
            self.metrics.read_seconds += ready_at - started if missing else 0.0
            self.metrics.staged_expert_reads += completed_count
            self.metrics.staged_w13_bytes_read += completed_count * pool._w13_size
            self.metrics.staged_w2_bytes_read += completed_count * pool._w2_size
            self.metrics.staged_read_seconds += (
                ready_at - started if missing else 0.0
            )
            self._decay_if_needed()

    def _reserve_slots(
        self,
        layer: int,
        missing: list[int],
        frequencies: Counter,
        protected: set[tuple[int, int]],
    ) -> dict[int, int]:
        assigned = {}
        for expert in missing:
            if self._free_slots:
                slot = self._free_slots.pop()
            else:
                eviction_started = time.perf_counter()
                victim_key, victim = self._evict(protected, layer)
                self.metrics.eviction_seconds += time.perf_counter() - eviction_started
                slot = victim.slot
                del self._entries[victim_key]
                self._seeded_keys.discard(victim_key)
                self._layer_counts[victim_key[0]] -= 1
                self.metrics.evictions += 1
                self._pool.mark_empty(slot)
            entry = _Entry(slot, frequencies[expert], 0, layer=layer, expert=expert)
            self._entries[(layer, expert)] = entry
            self._layer_counts[layer] += 1
            self._touch(layer, expert, entry, 0)
            assigned[expert] = slot
        self._pool.prepare(list(assigned.values()))
        return assigned

    def _release_slots(self, layer: int, assigned: dict[int, int]) -> None:
        for expert, slot in assigned.items():
            entry = self._entries.get((layer, expert))
            if entry is None or entry.slot != slot:
                continue
            del self._entries[(layer, expert)]
            self._seeded_keys.discard((layer, expert))
            self._layer_counts[layer] -= 1
            self._pool.mark_empty(slot)
            self._free_slots.append(slot)

    def _touch(self, layer: int, expert: int, entry: _Entry, count: int) -> None:
        if (layer, expert) in self._seeded_keys:
            self._seeded_keys.remove((layer, expert))
            self.metrics.prefill_seed_hits += 1
        self._clock += 1
        entry.frequency += count
        entry.last_access = self._clock
        entry.version += 1
        heapq.heappush(
            self._route_heaps[layer] if self._route_policy is not None else self._heap,
            (self._eviction_rank(entry), entry.last_access, entry.version, layer, expert),
        )

    def _eviction_rank(self, entry: _Entry) -> float:
        if self._route_policy is not None:
            # Slot owns one expert, so retain its identity independently of the heap.
            return float(self._route_policy.scores[entry.layer, entry.expert])
        # Keep actual assignment counts for profiling and decay. LRU changes
        # ranking only; reservation, pinning and in-flight protection are shared.
        return entry.frequency if self.eviction_policy == "lfu" else 0

    def _evict(self, protected: set[tuple[int, int]], incoming_layer: int | None = None) -> tuple[tuple[int, int], _Entry]:
        if self._route_policy is not None:
            return self._evict_route_locked(protected, incoming_layer)
        protected_items: list[tuple[int, int, int, int, int]] = []
        pinned_items: list[tuple[int, int, int, int, int]] = []
        reserved_items: list[tuple[int, int, int, int, int]] = []
        try:
            while self._heap:
                item = heapq.heappop(self._heap)
                frequency, last_access, version, layer, expert = item
                key = (layer, expert)
                entry = self._entries.get(key)
                if entry is None or (
                    self._eviction_rank(entry),
                    entry.last_access,
                    entry.version,
                ) != (frequency, last_access, version):
                    continue
                if key in protected or key in self._pinned_expert_keys:
                    protected_items.append(item)
                    continue
                if layer in self._pinned_layers:
                    pinned_items.append(item)
                    continue
                if self._layer_counts[layer] <= self._layer_reserve:
                    reserved_items.append(item)
                    continue
                return key, entry
            candidates = reserved_items or pinned_items
            if not candidates:
                raise RuntimeError("no expert cache slot can be evicted")
            item = candidates.pop(0)
            key = (item[3], item[4])
            return key, self._entries[key]
        finally:
            for item in (
                *protected_items,
                *pinned_items,
                *reserved_items,
            ):
                heapq.heappush(self._heap, item)

    def _decay_if_needed(self) -> None:
        if self._clock - self._last_decay < max(self.slots * 8, 64):
            return
        self._last_decay = self._clock
        if self._route_policy is not None:
            self._rebuild_route_heap_locked()
            return
        self._heap = []
        for (layer, expert), entry in self._entries.items():
            entry.frequency = max(1, entry.frequency // 2)
            entry.version += 1
            heapq.heappush(
                self._heap,
                (self._eviction_rank(entry), entry.last_access, entry.version, layer, expert),
            )

    def _read_expert_into_slot(self, layer: int, expert: int, slot: int) -> float:
        return self._read_expert_into_pool(self._pool, layer, expert, slot).finished

    def _begin_staged_expert_read(
        self,
        layer: int,
        expert: int,
        slot: int,
    ) -> tuple[_SpeculativeRead, Future[_SpeculativeRead]]:
        pool = self._pool
        executor = self._staged_w2_executor
        if not isinstance(pool, _StagedSlotPool) or executor is None:
            raise RuntimeError("staged expert streaming is not configured")
        w13_read = self._read_staged_w13_into_pool(
            pool,
            layer,
            expert,
            slot,
        )
        w2_future = executor.submit(
            self._read_staged_w2_into_pool,
            pool,
            layer,
            expert,
            slot,
        )
        return w13_read, w2_future

    def _read_staged_w13_into_pool(
        self,
        pool: _StagedSlotPool,
        layer: int,
        expert: int,
        slot: int,
    ) -> _SpeculativeRead:
        source = {region.name: region for region in self.model.expert_regions}
        base = expert * self.model.expert_blob_size
        started = time.perf_counter()
        page_cache = self._pread_views(
            layer,
            [
                pool.writable_region(slot, "w1.weight"),
                pool.writable_region(slot, "w1.scale"),
            ],
            base + source["w1.weight"].offset,
        )
        page_cache += self._pread_views(
            layer,
            [
                pool.writable_region(slot, "w3.weight"),
                pool.writable_region(slot, "w3.scale"),
            ],
            base + source["w3.weight"].offset,
        )
        return _SpeculativeRead(
            started,
            time.perf_counter(),
            page_cache,
        )

    def _read_staged_w2_into_pool(
        self,
        pool: _StagedSlotPool,
        layer: int,
        expert: int,
        slot: int,
    ) -> _SpeculativeRead:
        source = {region.name: region for region in self.model.expert_regions}
        base = expert * self.model.expert_blob_size
        started = time.perf_counter()
        page_cache = self._pread_views(
            layer,
            [
                pool.writable_region(slot, "w2.weight"),
                pool.writable_region(slot, "w2.scale"),
            ],
            base + source["w2.weight"].offset,
        )
        return _SpeculativeRead(
            started,
            time.perf_counter(),
            page_cache,
        )

    def _read_expert_into_pool(
        self,
        pool: _SlotPool,
        layer: int,
        expert: int,
        slot: int,
    ) -> _SpeculativeRead:
        started = time.perf_counter()
        views = [
            pool.writable_region(slot, region.name)
            for region in self.model.expert_regions
        ]
        page_cache = self._pread_views(
            layer,
            views,
            expert * self.model.expert_blob_size,
        )
        finished = time.perf_counter()
        if self._route_policy is not None:
            with self._lock:
                self._route_policy.record_read(layer, finished - started)
        return _SpeculativeRead(started, finished, page_cache)

    def _read_expert_ids(
        self,
        layer: int,
        packed: mx.array,
        experts: tuple[int, ...],
    ) -> _SpeculativeRead:
        started = time.perf_counter()
        view = memoryview(packed).cast("B")
        try:
            from .qwen_streaming_policy import contiguous_batches
            batches = ((e,) for e in experts) if self.prefill_read_experts == 1 else contiguous_batches(experts, self.prefill_read_experts)
            for batch in batches:
                check_cancelled()
                views = [part for expert in batch for part in self._pool.layer_write_views(view, expert)]
                try:
                    self._pread_views(layer, views,
                                      batch[0] * self.model.expert_blob_size,
                                      prefill=True)
                    with self._lock:
                        self.metrics.prefill_read_batches += 1
                finally:
                    for part in views:
                        part.release()
        finally:
            view.release()
        return _SpeculativeRead(started, time.perf_counter())

    def _pread_views(
        self,
        layer: int,
        views: list[memoryview],
        offset: int,
        *,
        prefill: bool = False,
    ) -> PageCacheReadClassification:
        reader = self._prefill_reader if (prefill or self._route_phase == "prefill") else None
        descriptor = reader.descriptors[layer] if reader is not None else self._descriptors[layer]
        pending = list(views)
        position = offset
        page_cache = PageCacheReadClassification()
        while pending:
            if reader is None and self.file_cache_policy == "bypass":
                self._validate_direct_read(position, pending)
            sample = None
            requested = sum(len(view) for view in pending)
            if self.page_cache_probe:
                sample = page_cache_residency_snapshot(
                    descriptor,
                    position,
                    requested,
                )
            if reader is not None:
                count = reader.read(layer, pending, position)
            elif self._read_limiter is None:
                count = os.preadv(descriptor, pending, position)
            else:
                count = self._read_limiter.preadv(
                    descriptor, pending, position
                )
            if count <= 0:
                raise EOFError(
                    f"expert file ended early at layer {layer}, offset {position}"
                )
            if self.page_cache_probe:
                page_cache += (
                    sample.classify(count)
                    if sample is not None
                    else PageCacheReadClassification.unavailable(count)
                )
            consumed = count
            while pending and consumed >= len(pending[0]):
                consumed -= len(pending[0])
                pending.pop(0)
            if pending and consumed:
                pending[0] = pending[0][consumed:]
            position += count
        if self.page_cache_probe:
            with self._lock:
                self.metrics.page_cache_probe_calls += page_cache.probe_calls
                self.metrics.page_cache_probe_failures += (
                    page_cache.probe_failures
                )
                self.metrics.page_cache_classified_bytes += (
                    page_cache.classified_bytes
                )
                self.metrics.page_cache_resident_bytes_before_read += (
                    page_cache.resident_bytes
                )
                self.metrics.page_cache_nonresident_bytes_before_read += (
                    page_cache.nonresident_bytes
                )
                self.metrics.page_cache_unclassified_bytes += (
                    page_cache.unclassified_bytes
                )
        return page_cache

    def _validate_direct_read(
        self,
        offset: int,
        views: list[memoryview],
    ) -> None:
        alignment = self.direct_io_alignment
        if alignment < 1:
            raise RuntimeError("cache-bypass read has no direct-I/O alignment")
        if offset % alignment:
            raise RuntimeError(
                f"cache-bypass file offset {offset} is not {alignment}-byte aligned"
            )
        for view in views:
            address = ctypes.addressof(ctypes.c_char.from_buffer(view))
            if address % alignment:
                raise RuntimeError(
                    "cache-bypass destination address is not "
                    f"{alignment}-byte aligned"
                )
            if len(view) % alignment:
                raise RuntimeError(
                    "cache-bypass iovec length is not "
                    f"{alignment}-byte aligned"
                )
