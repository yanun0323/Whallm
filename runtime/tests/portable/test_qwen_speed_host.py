"""Scalar-oracle checks for exact host-side Qwen hashing."""
import itertools
import unittest
from types import SimpleNamespace

import numpy as np

import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[2] / "deepseek_v4_ssd/qwen_ngram_hash.py"
spec = importlib.util.spec_from_file_location("_portable_qwen_hash", path)
hashing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hashing)
ngram_ids = hashing.ngram_ids
shift_right_ignore_eos = hashing.shift_right_ignore_eos


def scalar_shift(tokens, shift, eos):
    if shift == 0:
        return tokens
    result = np.full_like(tokens, eos)
    for batch in range(tokens.shape[0]):
        for position in range(tokens.shape[1]):
            source = position - shift
            if source < 0 or eos in tokens[batch, source:position]:
                continue
            result[batch, position] = tokens[batch, source]
    return result


def scalar_ids(tokens, multipliers, descriptor, eos):
    tokens = np.asarray(tokens, dtype=np.int64)
    shifted = [scalar_shift(tokens, i, eos) for i in range(3)]
    blocks = []
    for n in (2, 3):
        with np.errstate(over="ignore"):
            mixed = shifted[0] * np.int64(multipliers[0])
            for i in range(1, n):
                mixed = np.bitwise_xor(mixed, shifted[i] * np.int64(multipliers[i]))
        start = (n - 2) * 8
        blocks.append(mixed[..., None] % np.asarray(descriptor.head_vocab_sizes[start:start + 8])
                      + np.asarray(descriptor.head_offsets[start:start + 8]))
    return np.concatenate(blocks, axis=-1)


class QwenHashTests(unittest.TestCase):
    descriptor = SimpleNamespace(head_vocab_sizes=tuple(range(101, 117)),
                                 head_offsets=tuple(1000 * i for i in range(16)))

    def test_exhaustive_eos_boundaries(self):
        for length in range(7):
            tokens = np.array(list(itertools.product((0, 1, 2), repeat=length)), dtype=np.int64).reshape(-1, length) if length else np.empty((1, 0), dtype=np.int64)
            for shift in range(length + 2):
                np.testing.assert_array_equal(shift_right_ignore_eos(tokens, shift, 0), scalar_shift(tokens, shift, 0))

    def test_dtype_strides_empty_and_no_mutation(self):
        for dtype in (np.int32, np.int64):
            tokens = np.arange(50, dtype=dtype).reshape(2, 25)[:, ::-2]
            saved = tokens.copy()
            for shift in (0, 1, 2, 5, 13, 99):
                actual = shift_right_ignore_eos(tokens, shift, 6)
                self.assertEqual(actual.dtype, dtype)
                np.testing.assert_array_equal(actual, scalar_shift(tokens, shift, 6))
            np.testing.assert_array_equal(tokens, saved)
        self.assertEqual(shift_right_ignore_eos(np.empty((0, 10)), 2, 0).shape, (0, 10))
        with self.assertRaises(ValueError):
            shift_right_ignore_eos(tokens, -1, 0)

    def test_signed_overflow_hashes_match_scalar_oracle(self):
        rng = np.random.default_rng(391)
        for length in (0, 1, 2, 3, 17, 128, 2048):
            tokens = rng.integers(0, 248320, (3, length), dtype=np.int64)
            tokens[:, ::7] = 248044
            for multipliers in (np.array([1, 3, 9]), np.array([2**62 + 3, -2**62, 2**63 - 1])):
                expected = scalar_ids(tokens, multipliers, self.descriptor, 248044)
                np.testing.assert_array_equal(ngram_ids(tokens, multipliers, self.descriptor, eos_token_id=248044), expected)

    def test_chunked_history_hashes_are_identical(self):
        rng = np.random.default_rng(456)
        tokens = rng.integers(0, 20, (1, 307), dtype=np.int64)
        multipliers = np.array([2**62 + 7, -3, 7])
        expected = ngram_ids(tokens, multipliers, self.descriptor, eos_token_id=0)
        for step in (1, 2, 3, 17, 128):
            history = np.zeros((1, 2), dtype=np.int64)
            outputs = []
            for start in range(0, tokens.shape[1], step):
                current = tokens[:, start:start + step]
                joined = np.concatenate([history, current], axis=1)
                outputs.append(ngram_ids(joined, multipliers, self.descriptor, eos_token_id=0)[:, -current.shape[1]:])
                history = joined[:, -2:]
            np.testing.assert_array_equal(np.concatenate(outputs, axis=1), expected)


if __name__ == '__main__':
    unittest.main()
