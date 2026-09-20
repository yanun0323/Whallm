"""Host-only tests: real positioned I/O and exact pair scheduling, no MLX stubs."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

# Import standalone host modules without executing the package's MLX imports.
ROOT = Path(__file__).resolve().parents[2] / "deepseek_v4_ssd"

def load_host_module(name):
    spec = importlib.util.spec_from_file_location("_portable_" + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

waves = load_host_module("qwen_expert_waves")
io = load_host_module("qwen_flash_io")
settings = load_host_module("qwen_flash_config")


class ExpertWaveTests(unittest.TestCase):
    def test_random_pair_coverage_order_and_capacity(self):
        rng = np.random.default_rng(20260921)
        for tokens in (1, 2, 17, 128, 1024):
            selected = rng.integers(0, 512, size=(1, tokens, 10), dtype=np.int32)
            for capacity in (1, 2, 10, 32, 511, 512, 1024):
                with self.subTest(tokens=tokens, capacity=capacity):
                    visited = []
                    expert_ids = []
                    for wave in waves.expert_waves(selected, capacity, 512):
                        self.assertLessEqual(len(wave.groups), capacity)
                        self.assertEqual(len(set(wave.expert_ids)), len(wave.expert_ids))
                        for expert, positions in wave.groups:
                            np.testing.assert_array_equal(selected.reshape(-1)[positions], expert)
                            self.assertTrue(np.all(np.diff(positions) > 0))
                            visited.extend(positions)
                            expert_ids.append(expert)
                    np.testing.assert_array_equal(np.sort(visited), np.arange(selected.size))
                    self.assertEqual(expert_ids, sorted(set(selected.reshape(-1))))

    def test_weighted_moe_preserves_every_pair_and_reduction_order(self):
        rng = np.random.default_rng(7)
        source = rng.normal(size=(31, 8)).astype(np.float32)
        matrix = rng.normal(size=(13, 8, 8)).astype(np.float32)
        selected = rng.integers(0, 13, size=(31, 10))
        scores = rng.random(selected.shape).astype(np.float32)
        scores /= scores.sum(axis=-1, keepdims=True)
        baseline = np.empty((selected.size, 8), dtype=np.float32)
        # Match the baseline's per-expert GEMM shape, not singleton products.
        for expert in np.unique(selected):
            positions = np.flatnonzero(selected.reshape(-1) == expert)
            baseline[positions] = source[positions // 10] @ matrix[expert]
        for capacity in (1, 3, 13):
            outputs, order = [], []
            for wave in waves.expert_waves(selected, capacity, 13):
                for expert, positions in wave.groups:
                    outputs.append(source[positions // 10] @ matrix[expert])
                    order.extend(positions)
            restored = np.concatenate(outputs)[np.argsort(order)]
            np.testing.assert_array_equal(restored, baseline)
            np.testing.assert_array_equal(
                (restored.reshape(31, 10, 8) * scores[..., None]).sum(axis=1),
                (baseline.reshape(31, 10, 8) * scores[..., None]).sum(axis=1))

    def test_empty_and_repeated_routes(self):
        self.assertEqual(list(waves.expert_waves(np.empty((0, 10), dtype=int), 1, 512)), [])
        result = list(waves.expert_waves(np.full((9, 10), 5), 1, 512))
        self.assertEqual(len(result), 1)
        np.testing.assert_array_equal(result[0].groups[0][1], np.arange(90))

    def test_rejects_invalid_capacity_count_and_ids(self):
        for capacity in (0, -1, True, 2.0):
            with self.assertRaises(ValueError):
                list(waves.expert_waves(np.array([0]), capacity, 512))
        for count in (0, -1, True, 1.0):
            with self.assertRaises(ValueError):
                list(waves.expert_waves(np.array([0]), 1, count))
        for ids in ([-1], [512], [1.2], [True], np.array([2**64 - 1], dtype=np.uint64)):
            with self.assertRaises(ValueError):
                list(waves.expert_waves(np.asarray(ids), 2, 512))


class RowReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ngram.bin"
        self.rows = np.random.default_rng(42).integers(0, 256, (256, 160), dtype=np.uint8)
        self.rows.tofile(self.path)

    def reader(self, **kwargs):
        reader = io.NGramRowReader(self.path, 256, 160, **kwargs)
        self.addCleanup(reader.close)
        return reader

    def test_all_rows_are_byte_exact_and_owned(self):
        reader = self.reader()
        ids = np.arange(256)[::-1].reshape(16, 16)
        actual = reader.lookup(ids)
        np.testing.assert_array_equal(actual, self.rows[ids])
        actual[:] = 0
        np.testing.assert_array_equal(reader.lookup(ids), self.rows[ids])

    def test_scalar_empty_and_multidimensional_duplicates(self):
        reader = self.reader(cache_bytes=160 * 256)
        for ids in (np.array(7), np.empty((1, 0, 16), dtype=np.int64),
                    np.array([[[4, 4, 7], [2, 9, 4]]]), np.array([], dtype=float)):
            with self.subTest(shape=ids.shape):
                np.testing.assert_array_equal(reader.lookup(ids), self.rows[ids.astype(np.int64)])

    def test_deduplicates_sorts_and_coalesces_without_reading_gaps(self):
        reader = self.reader(max_read_bytes=160 * 2)
        actual = reader.lookup(np.array([7, 3, 2, 1, 2, 3]))
        np.testing.assert_array_equal(actual, self.rows[[7, 3, 2, 1, 2, 3]])
        stats = reader.snapshot()
        self.assertEqual(stats["read_calls"], 3)  # [1,2], [3], [7]
        self.assertEqual(stats["bytes_read"], 4 * 160)
        self.assertEqual(stats["requested_rows"], 6)
        self.assertEqual(stats["unique_rows"], 4)

    def test_warm_rows_do_not_issue_new_reads(self):
        reader = self.reader(cache_bytes=160 * 4)
        reader.lookup(np.array([7, 9, 7, 11]))
        before = reader.snapshot()
        np.testing.assert_array_equal(reader.lookup(np.array([11, 7, 9, 7])), self.rows[[11, 7, 9, 7]])
        after = reader.snapshot()
        self.assertEqual(after["read_calls"], before["read_calls"])
        self.assertEqual(after["bytes_read"], before["bytes_read"])
        self.assertEqual(after["cache_hits"], 3)

    def test_lru_and_floor_budget_are_bounded(self):
        reader = self.reader(cache_bytes=160 * 2 + 159)
        reader.lookup(np.array([1, 2]))
        reader.lookup(np.array([1]))  # 2 becomes the eviction candidate.
        reader.lookup(np.array([3]))
        self.assertEqual(set(reader._cache), {1, 3})
        self.assertEqual(reader.snapshot()["cached_payload_bytes"], 320)
        reader.lookup(np.arange(256))
        self.assertEqual(reader.snapshot()["cached_payload_bytes"], 320)
        np.testing.assert_array_equal(reader.lookup(np.arange(256)), self.rows)

    def test_zero_and_subrow_budget_disable_retention(self):
        for limit in (0, 159):
            reader = self.reader(cache_bytes=limit)
            reader.lookup(np.array([2, 2, 2]))
            reader.lookup(np.array([2]))
            self.assertEqual(reader.snapshot()["bytes_read"], 320)
            self.assertEqual(reader.snapshot()["cached_rows"], 0)

    def test_short_reads_and_interrupted_reads_are_completed(self):
        reader = self.reader()
        real = os.pread
        calls = 0
        def partial(fd, size, offset):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise InterruptedError()
            return real(fd, min(size, 13), offset)
        with patch.object(io.os, "pread", side_effect=partial):
            np.testing.assert_array_equal(reader.lookup(np.array([8, 9])), self.rows[[8, 9]])
        self.assertEqual(reader.snapshot()["bytes_read"], 320)
        self.assertGreater(reader.snapshot()["read_calls"], 2)

    def test_truncation_fails_and_does_not_cache_incomplete_rows(self):
        reader = self.reader(cache_bytes=320)
        os.truncate(self.path, 100)
        with self.assertRaises(EOFError):
            reader.lookup(np.array([0]))
        self.assertEqual(reader.snapshot()["cached_rows"], 0)
        self.rows.tofile(self.path)
        np.testing.assert_array_equal(reader.lookup(np.array([0])), self.rows[[0]])

    def test_invalid_ids_rejected_before_read(self):
        reader = self.reader()
        for ids in ([-1], [256], [1.5], [True], np.array([2**64-1], dtype=np.uint64)):
            with self.assertRaises(ValueError):
                reader.lookup(np.asarray(ids))
        self.assertEqual(reader.snapshot()["read_calls"], 0)

    def test_invalid_sizes_close_acquired_descriptor(self):
        os.truncate(self.path, 3)
        real_close = os.close
        with patch.object(io.os, "close", wraps=real_close) as close:
            with self.assertRaises(ValueError):
                self.reader()
            self.assertEqual(close.call_count, 1)
        for kwargs in ({"cache_bytes": -1}, {"cache_bytes": True}, {"max_read_bytes": 159}):
            with self.assertRaises(ValueError):
                self.reader(**kwargs)

    def test_cancelled_read_can_be_retried_without_partial_cache(self):
        class Cancelled(Exception):
            pass
        calls = 0
        def cancel():
            nonlocal calls
            calls += 1
            if calls == 3:
                raise Cancelled()
        reader = self.reader(cache_bytes=320, check_cancelled=cancel)
        with self.assertRaises(Cancelled):
            reader.lookup(np.array([2, 3]))
        self.assertEqual(reader.snapshot()["cached_rows"], 0)
        np.testing.assert_array_equal(reader.lookup(np.array([2, 3])), self.rows[[2, 3]])

    def test_threaded_reads_and_close_lifecycle(self):
        reader = self.reader(cache_bytes=160 * 8)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda i: reader.lookup(np.array([i, i + 1])), range(32)))
        for i, result in enumerate(results):
            np.testing.assert_array_equal(result, self.rows[[i, i + 1]])
        fd = reader._fd
        reader.close()
        reader.close()
        with self.assertRaises(OSError):
            os.fstat(fd)
        with self.assertRaises(ValueError):
            reader.lookup(np.array([0]))
        self.assertEqual(reader.snapshot()["cached_payload_bytes"], 0)


class FlashConfigTests(unittest.TestCase):
    def test_defaults_and_cli_roundtrip(self):
        parser = argparse.ArgumentParser()
        settings.add_flash_arguments(parser)
        self.assertEqual(settings.flash_arguments(parser.parse_args([])), settings.DEFAULTS)
        args = parser.parse_args(["--qwen-expert-wave-slots", "16", "--qwen-ngram-io", "pread",
                                  "--qwen-ngram-cache-bytes", "1048576", "--qwen-sparse-sdpa"])
        settings.validate_flash_config(args)
        self.assertEqual(args.qwen_expert_wave_slots, 16)
        self.assertTrue(args.qwen_sparse_sdpa)
        self.assertFalse(parser.parse_args(["--no-qwen-sparse-sdpa"]).qwen_sparse_sdpa)

    def test_config_validation(self):
        for name, values in {
            "qwen_expert_wave_slots": (-1, 513, True, 1.0),
            "qwen_ngram_cache_bytes": (-1, 512 * 1024**2 + 1, True, 1.0),
            "qwen_ngram_io": (None, [], "direct"),
            "qwen_sparse_sdpa": (None, 1, "false"),
        }.items():
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    settings.validate_flash_config(SimpleNamespace(**{**settings.DEFAULTS, name: value}))
        for changes in ({"qwen_ngram_cache_bytes": 1},
                        {"qwen_expert_wave_slots": 32, "qwen_next_layer_prefetch": True}):
            with self.assertRaises(ValueError):
                settings.validate_flash_config(SimpleNamespace(**{**settings.DEFAULTS, **changes}))
        settings.validate_flash_config(SimpleNamespace())
        settings.validate_flash_config(SimpleNamespace(qwen_expert_wave_slots=512,
            qwen_ngram_io="pread", qwen_ngram_cache_bytes=512 * 1024**2, qwen_sparse_sdpa=True))


if __name__ == "__main__":
    unittest.main()
