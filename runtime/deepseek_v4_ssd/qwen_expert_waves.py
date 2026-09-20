"""Exact, capacity-bounded scheduling of Qwen (token, expert) pairs.

This module has no MLX dependency. The caller must evaluate each wave's GPU
outputs before loading the next wave into reusable expert slots.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class ExpertWave:
    groups: tuple[tuple[int, np.ndarray], ...]

    @property
    def expert_ids(self) -> tuple[int, ...]:
        return tuple(expert for expert, _ in self.groups)


def expert_waves(indices: np.ndarray, capacity: int, expert_count: int) -> Iterator[ExpertWave]:
    """Visit every flattened routing pair exactly once, in stable expert order.

    A whole expert's token group stays together, preserving the baseline matrix
    multiplication shape. Returned positions restore the original top-k order;
    they must not be used to reorder or renormalize routing probabilities.
    """
    if type(capacity) is not int or capacity < 1:
        raise ValueError("expert wave capacity must be a positive integer")
    if type(expert_count) is not int or expert_count < 1:
        raise ValueError("expert count must be a positive integer")
    selected = np.asarray(indices)
    if selected.size == 0:
        return
    if not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("expert IDs must be integers")
    flat = selected.reshape(-1)
    if np.any(flat < 0) or np.any(flat >= expert_count):
        raise ValueError("expert ID is outside the model")
    order = np.argsort(flat, kind="stable")
    boundaries = np.flatnonzero(np.diff(flat[order])) + 1
    groups = []
    for positions in np.split(order, boundaries):
        groups.append((int(flat[positions[0]]), positions))
        if len(groups) == capacity:
            yield ExpertWave(tuple(groups))
            groups = []
    if groups:
        yield ExpertWave(tuple(groups))
