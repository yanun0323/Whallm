"""Portable configuration and per-chunk workspace tests."""
import argparse
import unittest
from types import SimpleNamespace
import importlib.util
from pathlib import Path

def load_host_module(name):
    path = Path(__file__).resolve().parents[2] / "deepseek_v4_ssd" / (name + ".py")
    spec = importlib.util.spec_from_file_location("_portable_qsa_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

schedule = load_host_module("qwen_qsa_schedule")
config = load_host_module("qwen_flash_config")


class QSAScheduleTests(unittest.TestCase):
    def test_defaults_cli_and_dependencies(self):
        parser = argparse.ArgumentParser()
        config.add_flash_arguments(parser)
        defaults = config.flash_arguments(parser.parse_args([]))
        self.assertEqual(defaults["qwen_qsa_query_chunk"], 4)
        self.assertFalse(defaults["qwen_qsa_indexed"])
        config.validate_flash_config(SimpleNamespace(**defaults))
        args = parser.parse_args(["--qwen-qsa-query-chunk", "32", "--qwen-qsa-indexed", "--qwen-sparse-sdpa"])
        config.validate_flash_config(SimpleNamespace(**config.flash_arguments(args)))
        self.assertTrue(args.qwen_qsa_indexed)
        self.assertFalse(parser.parse_args(["--no-qwen-qsa-indexed"]).qwen_qsa_indexed)
        with self.assertRaises(ValueError):
            config.validate_flash_config(SimpleNamespace(qwen_qsa_indexed=True))

    def test_invalid_values(self):
        for value in (0, -1, 129, True, False, 4.0, None, "32"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                config.validate_flash_config(SimpleNamespace(qwen_qsa_query_chunk=value))
        for value in (1, "true", None):
            with self.assertRaises(ValueError):
                config.validate_flash_config(SimpleNamespace(qwen_qsa_indexed=value))

    def test_schedule_stays_bounded(self):
        for tokens in (2048, 8192, 32768, 262144):
            for itemsize in (2, 4):
                for request in (1, 4, 16, 32, 128):
                    kw = dict(selected_rows=2052, kv_heads=2, query_heads=24, head_dim=256,
                              element_bytes=itemsize, index_blocks=tokens//4, index_heads=4)
                    result = schedule.query_chunk_size(request, **kw)
                    cost = 2*2052*2*256*itemsize + 12*2052*24 + 4*(tokens//4)*7
                    self.assertGreaterEqual(result, 1)
                    self.assertLessEqual(result, request)
                    self.assertLessEqual(cost*result, schedule.QSA_WORKSPACE_BYTES)
        self.assertEqual(schedule.query_chunk_size(4, **kw), 4)
