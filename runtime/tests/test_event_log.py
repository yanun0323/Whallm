import io
import threading
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from deepseek_v4_ssd.event_log import logger, failure, request_scope
from deepseek_v4_ssd.generation import RuntimeMetrics
from runtime.tests.test_native_qwen import fake_runtime, FakeState


class EventLogTests(unittest.TestCase):
    def test_level_filter_and_concurrent_request_ids(self):
        output = io.StringIO()
        barrier = threading.Barrier(2)
        ids = {}
        def worker(level):
            with request_scope(level, "private-key") as context:
                ids[level] = context.id
                barrier.wait()
                logger.debug("detail %s private-key", level)
                logger.info("summary %s", level)
                logger.error("failure %s", level)
        with redirect_stderr(output):
            threads = [threading.Thread(target=worker, args=(level,)) for level in ("debug", "error")]
            for thread in threads: thread.start()
            for thread in threads: thread.join()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 4)
        self.assertNotIn("private-key", output.getvalue())
        self.assertIn("[redacted]", output.getvalue())
        self.assertNotEqual(ids["debug"], ids["error"])
        for line in lines:
            level = "error" if line.endswith("failure error") else "debug"
            self.assertIn("request=" + ids[level], line)
            self.assertRegex(line, r"^\d{4}-\d\d-\d\dT.* \[(DEBUG|INFO|ERROR)\] request=req-")

    def test_exception_values_are_not_logged_but_locations_and_numeric_notes_are(self):
        output = io.StringIO()
        with redirect_stderr(output), request_scope("error"):
            try:
                raise ValueError("private-prompt data:image/png;base64,SECRET")
            except ValueError as cause:
                try:
                    error = RuntimeError("Bearer secret")
                    error.add_note("private tool output")
                    error.add_note("generation: prompt_tokens=42 generated_tokens=0 prefill_step_size=8")
                    raise error from cause
                except RuntimeError as error:
                    failure("generation failed", error)
        log = output.getvalue()
        for secret in ("private-prompt", "SECRET", "Bearer secret", "private tool output"):
            self.assertNotIn(secret, log)
        self.assertIn("error=ValueError", log)
        self.assertIn("error=RuntimeError", log)
        self.assertIn("prompt_tokens=42", log)
        self.assertIn("test_exception_values", log)

    def test_native_cache_failures_are_logged_and_remain_nonfatal(self):
        runtime = fake_runtime()
        runtime.engine.contract = "test"
        runtime.installed = SimpleNamespace(model_id="model", revision="revision")
        output = io.StringIO()
        with redirect_stderr(output), request_scope("debug"):
            with patch.object(runtime, "_best_disk_prefix", return_value=Path("cache")), \
                 patch.object(runtime.engine, "load", create=True, side_effect=OSError("private path")):
                self.assertEqual(runtime._continue_from_cache([1, 2]), 0)
            runtime._directory = Path("/unused-cache")
            with patch.object(Path, "exists", return_value=False), \
                 patch.object(runtime.engine, "save", create=True, side_effect=OSError("private path")):
                runtime._persist(FakeState([1, 2]))
            runtime.config = replace(runtime.config, persistent_prompt_cache=True)
            with patch.object(Path, "mkdir", side_effect=PermissionError("private path")):
                self.assertIsNone(runtime._open_prompt_cache_directory())
        log = output.getvalue()
        for event in ("prompt cache read failed", "prompt cache write failed", "prompt cache directory unavailable"):
            self.assertIn(event, log)
        self.assertIn("source=miss reused_tokens=0", log)
        self.assertNotIn("private path", log)

    def test_fallback_events_are_info_and_mtp_is_reported_once(self):
        output = io.StringIO()
        metrics = RuntimeMetrics()
        with redirect_stderr(output), request_scope("info"):
            metrics.record_mtp_round(2, 0, 0, 0, 0, True)
            metrics.record_mtp_round(2, 0, 0, 0, 0, True)
            metrics.record_dspark_fallback()
        self.assertEqual(output.getvalue().count("MTP disabled"), 1)
        self.assertIn("DSpark disabled", output.getvalue())
        self.assertNotIn("[ERROR]", output.getvalue())

    def test_model_lifecycle_uses_canonical_id_and_cancellation_is_not_error(self):
        from deepseek_v4_ssd.cancellation import GenerationCancelled
        from deepseek_v4_ssd.model_manager import ModelManager, ModelSpec, ModelDefaults
        from deepseek_v4_ssd.model import RuntimeConfig
        from runtime.tests.test_server import FakeRuntime
        spec = ModelSpec("deepseek-v4-flash-0731", None, "/unused", "deepseek-v4",
                         RuntimeConfig(), ModelDefaults(64, 0, 1, 0))
        manager = ModelManager([spec], runtime_loader=lambda _: FakeRuntime(), clear_cache=lambda: None)
        output = io.StringIO()
        with redirect_stderr(output), request_scope("debug"):
            manager.load(spec.id)
            manager.unload(spec.id)
        for event in ("model load started", "model load completed", "model unload started", "model unload completed"):
            self.assertIn(event + " model=" + spec.id, output.getvalue())
        self.assertIn("runtime settings", output.getvalue())
        manager._runtime_loader = lambda _: (_ for _ in ()).throw(GenerationCancelled("disconnect"))
        output = io.StringIO()
        with redirect_stderr(output), request_scope("info") as context:
            with self.assertRaises(GenerationCancelled):
                manager.load(spec.id)
        self.assertNotIn("[ERROR]", output.getvalue())
        self.assertIsNone(manager._loading)

    def test_closed_stderr_does_not_fail_generation(self):
        with patch("sys.stderr.write", side_effect=BrokenPipeError), request_scope("info") as context:
            context.generating = True
            logger.error("diagnostic")
