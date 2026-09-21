"""Exact read scheduling and prompt-tail retention; never replace native routes."""
from __future__ import annotations

import numpy as np


def contiguous_batches(experts, maximum: int):
    """Partition sorted unique IDs without reading gaps or changing order."""
    if type(maximum) is not int or not 1 <= maximum <= 32:
        raise ValueError("read batch size must be an integer from 1 through 32")
    ids = tuple(experts)
    if any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("expert IDs must be nonnegative integers")
    if any(a >= b for a, b in zip(ids, ids[1:])):
        raise ValueError("expert IDs must be sorted and unique")
    start = 0
    for end in range(1, len(ids) + 1):
        if end == len(ids) or end - start == maximum or ids[end] != ids[end - 1] + 1:
            yield ids[start:end]
            start = end


def hot_tail_experts(routes, expert_count: int, limit: int):
    """Rank the last 128 token positions by frequency, then recency, then ID.

    Return selected IDs from least to most valuable, so LRU insertion leaves
    the hottest one most recent. Only observed IDs are admitted. Integer routes
    have already been evaluated by the prefill owner before this host function.
    """
    if type(limit) is not int or limit < 0 or type(expert_count) is not int or expert_count < 1:
        raise ValueError("invalid prefill retention dimensions")
    rows = np.asarray(routes)
    if rows.ndim not in (2, 3) or rows.dtype.kind not in "iu" or rows.shape[-1] < 1:
        raise ValueError("routes must be integer token-by-expert rows")
    if np.any(rows < 0) or np.any(rows >= expert_count):
        raise ValueError("routed expert outside model range")
    ids = rows.reshape(-1, rows.shape[-1])[-128:].reshape(-1).astype(np.int64)
    if not ids.size or not limit:
        return []
    count = np.bincount(ids, minlength=expert_count)
    last = np.full(expert_count, -1, dtype=np.int64)
    np.maximum.at(last, ids, np.arange(ids.size))
    observed = np.flatnonzero(count)
    order = np.lexsort((observed, -last[observed], -count[observed]))
    return observed[order[:limit]][::-1].tolist()
