from __future__ import annotations

import ctypes
import fcntl
import os
import sys
from dataclasses import dataclass
from functools import cache

_RUSAGE_INFO_V2_FLAVOR = 2
_PROT_READ = 0x1
_MAP_SHARED = 0x1
_MAP_FAILED = ctypes.c_void_p(-1).value
EXPERT_FILE_CACHE_POLICIES = ("cached", "bypass")


@dataclass(frozen=True)
class ProcessDiskIO:
    bytes_read: int
    bytes_written: int

    def delta(self, before: ProcessDiskIO) -> ProcessDiskIO:
        return ProcessDiskIO(
            bytes_read=max(0, self.bytes_read - before.bytes_read),
            bytes_written=max(0, self.bytes_written - before.bytes_written),
        )


@dataclass(frozen=True)
class PageCacheReadClassification:
    """Byte accounting for a read range sampled before the read."""

    classified_bytes: int = 0
    resident_bytes: int = 0
    nonresident_bytes: int = 0
    unclassified_bytes: int = 0
    probe_calls: int = 0
    probe_failures: int = 0

    def __add__(
        self,
        other: PageCacheReadClassification,
    ) -> PageCacheReadClassification:
        return PageCacheReadClassification(
            classified_bytes=self.classified_bytes + other.classified_bytes,
            resident_bytes=self.resident_bytes + other.resident_bytes,
            nonresident_bytes=self.nonresident_bytes + other.nonresident_bytes,
            unclassified_bytes=(
                self.unclassified_bytes + other.unclassified_bytes
            ),
            probe_calls=self.probe_calls + other.probe_calls,
            probe_failures=self.probe_failures + other.probe_failures,
        )

    @classmethod
    def unavailable(cls, bytes_read: int) -> PageCacheReadClassification:
        return cls(
            unclassified_bytes=bytes_read,
            probe_calls=1,
            probe_failures=1,
        )


@dataclass(frozen=True)
class PageCacheResidency:
    """Page residency snapshot for one descriptor range before a read."""

    requested_offset: int
    requested_length: int
    mapped_offset: int
    page_size: int
    resident_pages: tuple[bool, ...]

    def classify(self, bytes_read: int) -> PageCacheReadClassification:
        if not 0 <= bytes_read <= self.requested_length:
            raise ValueError("classified bytes must fit the sampled range")
        start = self.requested_offset
        end = start + bytes_read
        resident_bytes = 0
        nonresident_bytes = 0
        for index, resident in enumerate(self.resident_pages):
            page_start = self.mapped_offset + index * self.page_size
            page_end = page_start + self.page_size
            overlap = max(0, min(end, page_end) - max(start, page_start))
            if resident:
                resident_bytes += overlap
            else:
                nonresident_bytes += overlap
        if resident_bytes + nonresident_bytes != bytes_read:
            raise RuntimeError("page-cache classification did not cover the read")
        return PageCacheReadClassification(
            classified_bytes=bytes_read,
            resident_bytes=resident_bytes,
            nonresident_bytes=nonresident_bytes,
            probe_calls=1,
        )


def configure_expert_file_cache_policy(
    descriptor: int,
    policy: str,
) -> None:
    """Apply the research expert-file cache policy to one descriptor."""
    if policy not in EXPERT_FILE_CACHE_POLICIES:
        raise ValueError(f"unknown expert-file cache policy: {policy}")
    if policy == "cached":
        return
    if sys.platform != "darwin":
        raise RuntimeError("expert-file cache bypass requires Darwin")
    fcntl.fcntl(descriptor, fcntl.F_NOCACHE, 1)
    # Darwin bsd/sys/fcntl.h defines F_RDAHEAD as 45. Some supported CPython
    # builds do not export the constant; this fallback remains Darwin-only.
    fcntl.fcntl(descriptor, getattr(fcntl, "F_RDAHEAD", 45), 0)


class _RUsageInfoV2(ctypes.Structure):
    """Darwin sys/resource.h rusage_info_v2 ABI."""

    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        ("user_time", ctypes.c_uint64),
        ("system_time", ctypes.c_uint64),
        ("package_idle_wakeups", ctypes.c_uint64),
        ("interrupt_wakeups", ctypes.c_uint64),
        ("pageins", ctypes.c_uint64),
        ("wired_size", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("physical_footprint", ctypes.c_uint64),
        ("process_start_absolute_time", ctypes.c_uint64),
        ("process_exit_absolute_time", ctypes.c_uint64),
        ("child_user_time", ctypes.c_uint64),
        ("child_system_time", ctypes.c_uint64),
        ("child_package_idle_wakeups", ctypes.c_uint64),
        ("child_interrupt_wakeups", ctypes.c_uint64),
        ("child_pageins", ctypes.c_uint64),
        ("child_elapsed_absolute_time", ctypes.c_uint64),
        ("disk_io_bytes_read", ctypes.c_uint64),
        ("disk_io_bytes_written", ctypes.c_uint64),
    ]


@cache
def _load_proc_pid_rusage():
    try:
        library = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        function = library.proc_pid_rusage
        function.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        function.restype = ctypes.c_int
        return function
    except (AttributeError, OSError):
        return None


def process_disk_io_snapshot() -> ProcessDiskIO | None:
    """Return Darwin's process-level disk I/O counters when available."""
    if sys.platform != "darwin":
        return None
    function = _load_proc_pid_rusage()
    if function is None:
        return None
    usage = _RUsageInfoV2()
    if function(os.getpid(), _RUSAGE_INFO_V2_FLAVOR, ctypes.byref(usage)) != 0:
        return None
    return ProcessDiskIO(
        bytes_read=int(usage.disk_io_bytes_read),
        bytes_written=int(usage.disk_io_bytes_written),
    )


@cache
def _load_page_cache_functions():
    if sys.platform != "darwin":
        return None
    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        mmap_function = library.mmap
        mmap_function.argtypes = (
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
        )
        mmap_function.restype = ctypes.c_void_p
        mincore_function = library.mincore
        mincore_function.argtypes = (
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte),
        )
        mincore_function.restype = ctypes.c_int
        munmap_function = library.munmap
        munmap_function.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        munmap_function.restype = ctypes.c_int
        return mmap_function, mincore_function, munmap_function
    except (AttributeError, OSError):
        return None


def page_cache_residency_snapshot(
    descriptor: int,
    offset: int,
    length: int,
) -> PageCacheResidency | None:
    """Sample pre-read file-page residency with Darwin mincore(2).

    This reports virtual-memory page residency, not physical SSD traffic.
    Mapping and mincore do not intentionally fault the requested pages in.
    """
    if offset < 0:
        raise ValueError("page-cache probe offset must be non-negative")
    if length < 1:
        raise ValueError("page-cache probe length must be positive")
    functions = _load_page_cache_functions()
    if functions is None:
        return None
    mmap_function, mincore_function, munmap_function = functions
    page_size = os.sysconf("SC_PAGE_SIZE")
    mapped_offset = offset - offset % page_size
    prefix = offset - mapped_offset
    mapped_length = ((prefix + length + page_size - 1) // page_size) * page_size
    address = mmap_function(
        None,
        mapped_length,
        _PROT_READ,
        _MAP_SHARED,
        descriptor,
        mapped_offset,
    )
    if address is None or address == _MAP_FAILED:
        return None
    try:
        page_count = mapped_length // page_size
        vector = (ctypes.c_ubyte * page_count)()
        if mincore_function(address, mapped_length, vector) != 0:
            return None
        return PageCacheResidency(
            requested_offset=offset,
            requested_length=length,
            mapped_offset=mapped_offset,
            page_size=page_size,
            resident_pages=tuple(bool(value & 0x1) for value in vector),
        )
    finally:
        munmap_function(address, mapped_length)
