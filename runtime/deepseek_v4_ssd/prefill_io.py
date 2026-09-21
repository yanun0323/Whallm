"""Aligned, cache-bypassing Prefill reads with a bounded buffer per worker."""
from __future__ import annotations

import ctypes
import mmap
import os
import threading

from .cancellation import check_cancelled
from .io_metrics import configure_expert_file_cache_policy


class PrefillReader:
    def __init__(self, directory, layers: int, blob_size: int, read_limiter=None, *, batch_experts: int = 1):
        if type(batch_experts) is not int or not 1 <= batch_experts <= 32:
            raise ValueError("batch_experts must be an integer from 1 through 32")
        self.descriptors: list[int] = []
        self._local = threading.local()
        self._buffers: list[mmap.mmap] = []
        self._lock = threading.Lock()
        self._capacity = blob_size * batch_experts + 2 * mmap.PAGESIZE
        self._read_limiter = read_limiter
        self.read_calls = 0
        self.direct_bytes = 0
        self.copied_bytes = 0
        try:
            for layer in range(layers):
                fd = os.open(directory / f"layer_{layer:02d}.bin", os.O_RDONLY)
                self.descriptors.append(fd)
                configure_expert_file_cache_policy(fd, "bypass")
        except BaseException:
            self.close()
            raise

    def close(self):
        # The owner drains its workers before closing their buffers and files.
        for fd in self.descriptors:
            os.close(fd)
        self.descriptors.clear()
        for buffer in self._buffers:
            buffer.close()
        self._buffers.clear()
        self._local = threading.local()

    def _preadv(self, fd, views, offset):
        check_cancelled()
        while True:
            check_cancelled()
            with self._lock:
                self.read_calls += 1
            try:
                if self._read_limiter is None:
                    return os.preadv(fd, views, offset)
                return self._read_limiter.preadv(fd, views, offset)
            except InterruptedError:
                continue

    def read(self, layer: int, views: list[memoryview], offset: int) -> int:
        alignment = mmap.PAGESIZE
        length = sum(map(len, views))
        if offset < 0 or length < 1:
            raise ValueError("Prefill read requires a nonempty range")
        fd = self.descriptors[layer]
        direct = offset % alignment == 0 and all(
            len(view) % alignment == 0
            and ctypes.addressof(ctypes.c_char.from_buffer(view)) % alignment == 0
            for view in views
        )
        if direct:
            count = self._preadv(fd, views, offset)
            if count <= 0:
                raise EOFError(f"Prefill file ended at layer {layer}, offset {offset}")
            with self._lock:
                self.direct_bytes += count
            return count

        begin = offset // alignment * alignment
        prefix = offset - begin
        size = (prefix + length + alignment - 1) // alignment * alignment
        if size > self._capacity:
            raise ValueError("Prefill read exceeds configured staging capacity")
        buffer = getattr(self._local, "buffer", None)
        if buffer is None:
            buffer = mmap.mmap(-1, self._capacity)
            self._local.buffer = buffer
            with self._lock:
                self._buffers.append(buffer)
        target = memoryview(buffer)[:size]
        try:
            done = 0
            while done < prefix + length:
                part = target[done:]
                try:
                    count = self._preadv(fd, [part], begin + done)
                finally:
                    part.release()
                if count <= 0:
                    raise EOFError(f"Prefill file ended at layer {layer}, offset {begin + done}")
                done += count
                if done < prefix + length and done % alignment:
                    # Never continue bypass I/O with an unaligned destination.
                    if begin + done >= os.fstat(fd).st_size:
                        raise EOFError("Prefill expert is truncated")
                    raise OSError("Unaligned short Prefill read")
            cursor = prefix
            for view in views:
                view[:] = target[cursor:cursor + len(view)]
                cursor += len(view)
        finally:
            target.release()
        with self._lock:
            self.copied_bytes += length
        return length

    def snapshot(self):
        with self._lock:
            return {"read_calls": self.read_calls, "direct_bytes": self.direct_bytes, "copied_bytes": self.copied_bytes,
                    "staging_buffers": len(self._buffers),
                    "staging_capacity_bytes": len(self._buffers) * self._capacity}
