"""Exact host-side Qwen N-gram hashing; no MLX import or mutable state."""
from __future__ import annotations

import numpy as np


def shift_right_ignore_eos(token_ids: np.ndarray, shift: int, eos_token_id: int) -> np.ndarray:
    """Shift within each EOS-delimited segment, preserving the original dtype."""
    if shift == 0:
        return token_ids
    if shift < 0:
        raise ValueError("N-gram history shift must be nonnegative")
    result = np.full_like(token_ids, eos_token_id)
    length = token_ids.shape[1]
    if shift >= length:
        return result
    count = length - shift
    # Only shifts 1 and 2 occur in Qwen. Loop over history distance, never tokens.
    blocked = token_ids[:, :count] == eos_token_id
    for distance in range(1, shift):
        blocked |= token_ids[:, distance:distance + count] == eos_token_id
    result[:, shift:] = np.where(blocked, eos_token_id, token_ids[:, :count])
    return result


def ngram_ids(token_ids, multipliers, descriptor, *, eos_token_id: int) -> np.ndarray:
    """Return the same signed-int64 overflow/remainder hashes as the scalar path."""
    tokens = np.asarray(token_ids, dtype=np.int64)
    shifted = [shift_right_ignore_eos(tokens, shift, eos_token_id) for shift in range(3)]
    with np.errstate(over="ignore"):
        pair_hash = np.bitwise_xor(shifted[0] * np.int64(multipliers[0]),
                                  shifted[1] * np.int64(multipliers[1]))
        triple_hash = np.bitwise_xor(pair_hash, shifted[2] * np.int64(multipliers[2]))
    blocks = []
    for start, mixed in ((0, pair_hash), (8, triple_hash)):
        sizes = np.asarray(descriptor.head_vocab_sizes[start:start + 8], dtype=np.int64)
        offsets = np.asarray(descriptor.head_offsets[start:start + 8], dtype=np.int64)
        blocks.append(np.remainder(mixed[..., None], sizes) + offsets)
    return np.concatenate(blocks, axis=-1)
