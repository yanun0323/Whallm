"""Positioned N-gram row reads with deduplication and a bounded packed-row LRU.

pread is buffered POSIX I/O, NOT O_DIRECT and not a physical SSD-byte counter.
The cache limit covers retained packed payload, not Python metadata, returned
arrays, the OS page cache, or transient (at most max_read_bytes) read buffers.
"""
from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import Callable

import numpy as np


class NGramRowReader:
    def __init__(
        self, path: Path, row_count: int, row_bytes: int, *,
        cache_bytes: int = 0, max_read_bytes: int = 256 * 1024,
        check_cancelled: Callable[[], None] | None = None,
    ):
        for name, value, minimum in (
            ("row_count", row_count, 1), ("row_bytes", row_bytes, 1),
            ("cache_bytes", cache_bytes, 0), ("max_read_bytes", max_read_bytes, row_bytes),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        self.row_count = row_count
        self.row_bytes = row_bytes
        self._capacity = cache_bytes // row_bytes
        self._rows_per_read = max_read_bytes // row_bytes
        self._check_cancelled = check_cancelled or (lambda: None)
        self._lock = RLock()
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._stats = dict(lookups=0, requested_rows=0, unique_rows=0,
                           cache_hits=0, read_calls=0, bytes_read=0)
        self._fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            if os.fstat(self._fd).st_size != row_count * row_bytes:
                raise ValueError("N-gram file size does not match the manifest")
        except BaseException:
            os.close(self._fd)
            self._fd = -1
            raise

    def _read_exact(self, offset: int, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            self._check_cancelled()
            try:
                self._stats["read_calls"] += 1
                data = os.pread(self._fd, remaining, offset)
            except InterruptedError:
                continue
            if not data:
                raise EOFError("unexpected EOF in N-gram row read")
            chunks.append(data)
            self._stats["bytes_read"] += len(data)
            offset += len(data)
            remaining -= len(data)
        self._check_cancelled()
        return b"".join(chunks)

    def lookup(self, row_ids: np.ndarray) -> np.ndarray:
        """Return owned uint8 rows in the exact input shape/order, including repeats."""
        ids = np.asarray(row_ids)
        if ids.size and not np.issubdtype(ids.dtype, np.integer):
            raise ValueError("N-gram row IDs must be integers")
        if ids.size and (np.any(ids < 0) or np.any(ids >= self.row_count)):
            raise ValueError("N-gram row is outside ngram.bin")
        with self._lock:
            if self._fd < 0:
                raise ValueError("N-gram reader is closed")
            self._check_cancelled()
            unique, inverse = np.unique(ids.reshape(-1), return_inverse=True)
            rows = np.empty((unique.size, self.row_bytes), dtype=np.uint8)
            self._stats["lookups"] += 1
            self._stats["requested_rows"] += ids.size
            self._stats["unique_rows"] += unique.size
            misses = []
            for position, raw_id in enumerate(unique):
                row_id = int(raw_id)
                packed = self._cache.get(row_id)
                if packed is None:
                    misses.append((position, row_id))
                else:
                    rows[position] = np.frombuffer(packed, dtype=np.uint8)
                    self._cache.move_to_end(row_id)
                    self._stats["cache_hits"] += 1
            start = 0
            while start < len(misses):
                self._check_cancelled()
                end = start + 1
                # Coalesce only consecutive requested rows, never read gaps.
                while (end < len(misses) and end - start < self._rows_per_read
                       and misses[end][1] == misses[end - 1][1] + 1):
                    end += 1
                data = self._read_exact(misses[start][1] * self.row_bytes,
                                        (end - start) * self.row_bytes)
                for index in range(start, end):
                    position, row_id = misses[index]
                    offset = (index - start) * self.row_bytes
                    packed = data[offset:offset + self.row_bytes]
                    rows[position] = np.frombuffer(packed, dtype=np.uint8)
                    if self._capacity:
                        # Evict BEFORE insertion, including transient payload accounting.
                        if len(self._cache) >= self._capacity:
                            self._cache.popitem(last=False)
                        self._cache[row_id] = packed
                start = end
            return rows[inverse].reshape(*ids.shape, self.row_bytes)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {**self._stats, "cached_rows": len(self._cache),
                    "cached_payload_bytes": len(self._cache) * self.row_bytes,
                    "capacity_rows": self._capacity}

    def close(self) -> None:
        with self._lock:
            if self._fd >= 0:
                fd, self._fd = self._fd, -1
                self._cache.clear()
                os.close(fd)

    def __enter__(self) -> NGramRowReader:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
