import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import mlx.core as mx

from deepseek_v4_ssd.qwen_profile import QwenProfile, arrays, process_snapshot


class QwenProfileTests(unittest.TestCase):
    def test_nested_accounting_and_failure_restore(self):
        profile = QwenProfile()
        with patch.object(profile, "snapshot", return_value={"expert_bytes_read": 0}), \
             patch("deepseek_v4_ssd.qwen_profile.time.perf_counter", side_effect=[0, 1, 3, 3, 5]):
            with profile.span("parent", 7):
                with profile.span("child"):
                    pass
        rows = {r["component"]: r for r in profile.report()["rows"]}
        self.assertEqual(rows["parent"]["inclusive_seconds"], 5)
        self.assertEqual(rows["parent"]["exclusive_seconds"], 3)
        self.assertEqual(rows["child"]["layer"], 7)
        with self.assertRaisesRegex(ValueError, "broken"):
            with profile.span("failure", 2):
                raise ValueError("broken")
        self.assertEqual(profile.layer, -1)
        self.assertFalse(profile.stack)
        self.assertEqual(next(r for r in profile.report()["rows"] if r["component"] == "failure")["errors"], 1)

    def test_hooks_preserve_results_restore_and_skip_other_threads(self):
        class Component:
            def __call__(self, value):
                return value * 2 + 1
        obj = Component()
        original = Component.__call__
        for mode in ("host", "sync"):
            profile = QwenProfile(mode)
            try:
                profile.hook(Component, "__call__", {id(obj): ("component", 4)})
                result = obj(mx.array([1., 2.]))
                self.assertEqual(result.tolist(), [3., 5.])
                worker = threading.Thread(target=lambda: obj(3))
                worker.start()
                worker.join()
                self.assertEqual(profile.report()["rows"][0]["calls"], 1)
                self.assertEqual(Component()(3), 7)  # Unregistered instance.
                self.assertEqual(profile.report()["rows"][0]["calls"], 1)
            finally:
                profile.close()
            self.assertIs(Component.__call__, original)

    def test_process_unavailable_and_memory_is_gauge(self):
        with patch("deepseek_v4_ssd.qwen_profile._load_proc_pid_rusage", return_value=None):
            self.assertEqual(process_snapshot(), {})
        profile = QwenProfile()
        samples = [{"physical_footprint": 50, "disk_io_bytes_read": 10},
                   {"physical_footprint": 40, "disk_io_bytes_read": 30}]
        with patch.object(profile, "snapshot", side_effect=samples):
            with profile.span("test"):
                pass
        row = profile.report()["rows"][0]
        self.assertEqual(row["boundary_max"]["physical_footprint"], 50)
        self.assertEqual(row["boundary_min"]["physical_footprint"], 40)
        self.assertEqual(row["counters"], {"disk_io_bytes_read": 20})

    def test_requests_are_separate_and_owner_cpu_is_available(self):
        profile = QwenProfile()
        self.assertGreaterEqual(profile.snapshot()['owner_cpu_seconds'], 0)
        for request in (0, 1):
            profile.begin_phase('decode', request)
            with profile.span('decoder', 0):
                sum(range(100))
            profile.end_phase()
        rows = profile.report()['rows']
        self.assertEqual([r['request'] for r in rows], [0, 1])
        self.assertTrue(all(r['calls'] == 1 and r['counters']['owner_cpu_seconds'] >= 0 for r in rows))

    def test_timeline_records_decoder_and_phase_and_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'timeline.jsonl'
            profile = QwenProfile('sync', timeline_path=path)
            with patch.object(profile, '_clock', side_effect=[10, 20, 30, 40]):
                profile.begin_phase('prefill', 2)
                with profile.span('decoder', 7):
                    with profile.span('child'):
                        pass
                profile.end_phase()
            profile.close()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows[0]['clock'], 'mach_absolute_time')
            self.assertEqual([(r['kind'], r['begin'], r['end']) for r in rows[1:]],
                             [('decoder', 20, 30), ('phase', 10, 40)])
            self.assertEqual(rows[1]['request'], 2)
            self.assertEqual(rows[1]['layer'], 7)
            self.assertFalse(rows[1]['failed'])
            self.assertTrue(profile._timeline.closed)
            profile.close()  # Idempotent cleanup.
            with self.assertRaises(FileExistsError):QwenProfile('sync', timeline_path=path)

    def test_failed_timeline_phase_is_not_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'timeline.jsonl'
            profile = QwenProfile(timeline_path=path)
            profile.begin_phase('decode', 0)
            with self.assertRaises(ValueError):
                with profile.span('decoder', 1):raise ValueError('injected')
            profile.close()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertTrue(rows[1]['failed'])
            self.assertTrue(rows[2]['failed'])
            self.assertFalse(profile.stack)

    def test_array_tree_and_invalid_mode(self):
        x = mx.array([1])
        self.assertEqual(len(list(arrays({"a": [x, (x, None)], "b": 1}))), 2)
        with self.assertRaises(ValueError):
            QwenProfile("gpu")


if __name__ == "__main__":
    unittest.main()
