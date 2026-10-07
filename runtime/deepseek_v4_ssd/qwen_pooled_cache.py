"""Opt-in derived QSA index-key cache; raw history remains the source of truth.

Derived rows are immutable, so speculative forks can share them. Serialization
saves raw history only and rebuilds derived rows on restore. No model weights or
RNG are captured by the cache.
"""
from __future__ import annotations

import mlx.core as mx
from mlx_lm.models.cache import KVCache
from .qwen_quantized_cache import QSAQuantizedCache


class _PooledKeys:
    def _reset_pooled(self):
        self.pooled_keys = None
        self._pooled_ratio = None

    def pooled(self, raw: mx.array, ratio: int, transform) -> mx.array:
        if type(ratio) is not int or ratio < 1:
            raise ValueError("QSA pooling ratio must be a positive integer")
        if raw.ndim != 3 or raw.shape[1] != self.offset:
            raise ValueError("QSA raw history does not match the index cache")
        blocks = raw.shape[1] // ratio
        if self._pooled_ratio != ratio:
            self._reset_pooled()
            self._pooled_ratio = ratio
        previous = 0 if self.pooled_keys is None else self.pooled_keys.shape[1]
        if previous > blocks:
            self.pooled_keys = self.pooled_keys[:, :blocks]
            previous = blocks
        if previous < blocks:
            fresh = transform(raw[:, previous * ratio:blocks * ratio], previous)
            if fresh.shape != (raw.shape[0], blocks - previous, raw.shape[2]):
                raise ValueError("QSA pooled rows have an invalid shape")
            self.pooled_keys = fresh if previous == 0 else mx.concatenate(
                [self.pooled_keys, fresh], axis=1)
        if blocks == 0:
            # Match the baseline's empty tensor without invoking a reduction.
            return raw[:, :0]
        return self.pooled_keys

    def trim(self, n):
        count = super().trim(n)
        # Complete blocks before the new offset are unchanged by later appends;
        # drop only blocks that include trimmed or future rows.
        if self.pooled_keys is not None and self._pooled_ratio is not None:
            self.pooled_keys = self.pooled_keys[:, : self.offset // self._pooled_ratio]
        return count

    @property
    def nbytes(self):
        return super().nbytes + (0 if self.pooled_keys is None else self.pooled_keys.nbytes)


class QSAPooledIndexCache(_PooledKeys, KVCache):
    def __init__(self):
        super().__init__()
        self._reset_pooled()

    @property
    def state(self):
        return KVCache.state.fget(self)

    @state.setter
    def state(self, value):
        KVCache.state.fset(self, value)
        self._reset_pooled()


class QSAPooledQuantizedIndexCache(_PooledKeys, QSAQuantizedCache):
    def __init__(self, bits, dimension):
        super().__init__(bits, dimension)
        self._reset_pooled()

    @property
    def state(self):
        return QSAQuantizedCache.state.fget(self)

    @state.setter
    def state(self, value):
        QSAQuantizedCache.state.fset(self, value)
        self._reset_pooled()

    def restore_persistence_state(self, saved):
        super().restore_persistence_state(saved)
        self._reset_pooled()
