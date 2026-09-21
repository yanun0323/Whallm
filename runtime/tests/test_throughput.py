from __future__ import annotations

import gzip
import hashlib
import json
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from test_server import FakeRuntime
from deepseek_v4_ssd.cancellation import check_cancelled
from deepseek_v4_ssd.generation import GeneratedPiece, GenerationOptions
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import ModelDefaults, ModelManager, ModelSpec
from deepseek_v4_ssd.server import OpenAIServer
from deepseek_v4_ssd.throughput import CONTEXT_LENGTHS, CONTEXT_TYPES, CORPUS_DIRECTORY, prompt_tokens, run_trial


class BenchmarkRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.prompt = []
        self.entered = threading.Event()
        self.config = RuntimeConfig(slots=2304)
        self.installed = SimpleNamespace(maximum_context=262144, is_qwen=False)
        self.metrics = SimpleNamespace(snapshot=lambda: {
            "time_to_first_token_seconds": 0.5,
            "decode_tokens_per_second": 4.0,
            "prefill_tokens_per_second": len(self.prompt) / 0.5,
            "prompt_cache_reused_tokens": 0,
        })
        self.finished = threading.Event()
        self.slow = False
        self.fail = False

    def _encode_prompt(self, text):
        return list(text.encode())

    def stream(self, prompt, options):
        self.prompt = prompt
        self.entered.set()
        self.options = options
        self.benchmark_config = self.config
        try:
            if self.fail:
                raise ValueError("test generation failure")
            while self.slow:
                check_cancelled()
                time.sleep(0.01)
            yield GeneratedPiece("a", 12, len(prompt), 1, None)
            yield GeneratedPiece("b", 13, len(prompt), 2, "stop")
        finally:
            self.finished.set()


class ThroughputTests(unittest.TestCase):
    def setUp(self):
        self.runtime = BenchmarkRuntime()
        self.original_config = self.runtime.config
        self.loads = 0
        self.loading_memory_states = []
        def load(_):
            self.loads += 1
            self.loading_memory_states.append(self.server.status_memory.snapshot())
            return self.runtime
        self.manager = ModelManager([
            ModelSpec("deepseek-v4-flash-0731", None, "/tmp/model", "deepseek-v4",
                      RuntimeConfig(), ModelDefaults(272000, 0.2, 0.98, 0))
        ], runtime_loader=load, clear_cache=lambda: None)
        self.server = OpenAIServer(("127.0.0.1", 0), self.manager, api_key="secret")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/api/benchmark/throughput"

    def tearDown(self):
        self.runtime.slow = False
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.manager.close()

    def request(self, **changes):
        payload = dict(model="deepseek-v4-flash-0731", context_length=1024, generation_length=128)
        payload.update(changes)
        return Request(self.url, data=json.dumps(payload).encode(), headers={
            "Authorization": "Bearer secret", "Content-Type": "application/json"})

    def test_benchmark_evidence_reports_effective_config_and_unknown_io(self):
        result = run_trial(self.runtime, GenerationOptions(max_tokens=128), 1024, iter, lambda _: None)
        diagnostics = result["diagnostics"]
        self.assertIsNone(diagnostics["request_total"])
        settings = json.loads(diagnostics["runtime_config_json"])
        self.assertEqual(settings["prompt_cache_entries"], 0)
        self.assertEqual(settings["qwen_prefill_read_experts"], 1)
        self.assertEqual(len(diagnostics["source_files_sha256"]), 64)
        self.assertEqual(diagnostics["config_sha256"],
                         hashlib.sha256(diagnostics["runtime_config_json"].encode()).hexdigest())
        self.assertIs(self.runtime.config, self.original_config)

    def test_evidence_failure_always_preserves_original_configuration(self):
        for target in ("source_files_hash", "make_report"):
            with patch("deepseek_v4_ssd.throughput." + target, side_effect=OSError("evidence failed")):
                with self.assertRaisesRegex(OSError, "evidence failed"):
                    run_trial(self.runtime, GenerationOptions(max_tokens=128), 1024, iter, lambda _: None)
            self.assertIs(self.runtime.config, self.original_config)

    def test_both_assets_use_original_prefixes_and_match_manifest(self):
        manifest = json.loads((CORPUS_DIRECTORY / "manifest.json").read_text())
        total_bytes = 0
        for context in CONTEXT_TYPES:
            path = CORPUS_DIRECTORY / f"{context}.txt.gz"
            total_bytes += path.stat().st_size
            data = gzip.decompress(path.read_bytes())
            expected_hash = hashlib.sha256(data).hexdigest()
            self.assertEqual(expected_hash, manifest["contexts"][context]["sha256"])
            self.assertEqual(len(data), manifest["contexts"][context]["utf8_bytes"])
            original = self.runtime._encode_prompt(data.decode())
            for length in CONTEXT_LENGTHS:
                tokens, corpus_hash = prompt_tokens(self.runtime, length, context)
                self.assertEqual(len(tokens), length)
                self.assertEqual(tokens, original[:length])
                self.assertEqual(corpus_hash, expected_hash)
        self.assertLess(total_bytes, 750_000)

    def test_short_corpus_is_rejected_instead_of_repeated(self):
        with patch.object(self.runtime, "_encode_prompt", return_value=[1, 2, 3]):
            for context in CONTEXT_TYPES:
                with self.assertRaisesRegex(ValueError, "only 3 tokens.*1024 were requested"):
                    prompt_tokens(self.runtime, 1024, context)
        with self.assertRaisesRegex(ValueError, "Choose Code or Novel"):
            prompt_tokens(self.runtime, 1024, "../../invalid")

    def test_512_output_tokens_reach_runtime_and_streamed_result(self):
        with urlopen(self.request(generation_length=512)) as response:
            self.assertEqual(response.status, 200)
            body = response.read().decode()
        events = [json.loads(line[6:]) for line in body.splitlines()
                  if line.startswith("data: {")]
        result = next(event["result"] for event in events if "result" in event)
        self.assertEqual(self.runtime.options.max_tokens, 512)
        self.assertEqual(result["generation_limit"], 512)
        self.assertEqual(result["generation_tokens"], 2)  # Mock runtime stops at EOS.
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    def test_novel_selection_reaches_generation_and_result(self):
        with urlopen(self.request(benchmark_context="novel")) as response:
            events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                      if line.startswith("data: {")]
        result = next(event["result"] for event in events if "result" in event)
        expected, expected_hash = prompt_tokens(self.runtime, 1024, "novel")
        self.assertEqual(self.runtime.prompt, expected)
        self.assertEqual(result["benchmark_context"], "novel")
        self.assertEqual(result["corpus_sha256"], expected_hash)

    def test_auto_load_and_request_scoped_results_restore_settings(self):
        with patch("deepseek_v4_ssd.app_memory.physical_footprint", return_value=123456), \
             patch("deepseek_v4_ssd.app_memory.app_process_ids", return_value=(11, 22)):
            with urlopen(self.request()) as response:
                body = response.read().decode()
        events = [json.loads(line[6:]) for line in body.splitlines()
                  if line.startswith("data: {")]
        result = next(event["result"] for event in events if "result" in event)
        self.assertEqual(self.loads, 1)
        self.assertEqual(len(self.runtime.prompt), 1024)
        self.assertIsInstance(self.runtime.prompt, list)
        self.assertEqual(self.runtime.options.max_tokens, 128)
        self.assertEqual(self.runtime.benchmark_config.prompt_cache_entries, 0)
        self.assertFalse(self.runtime.benchmark_config.persistent_prompt_cache)
        self.assertFalse(self.runtime.benchmark_config.dspark_prompt_cache)
        self.assertIs(self.runtime.config, self.original_config)
        self.assertEqual(result["generation_tokens"], 2)  # EOS is not reported as 128.
        self.assertEqual(result["context_tokens"], 1024)
        self.assertEqual(result["benchmark_context"], "code")
        self.assertEqual(result["slots"], self.original_config.slots)
        self.assertEqual(result["corpus_sha256"], prompt_tokens(self.runtime, 1024)[1])
        self.assertEqual(result["ttft_ms"], 500)
        self.assertEqual(result["tpot_ms"], 250)
        self.assertEqual(result["peak_app_memory_bytes"], 246912)
        self.assertEqual(result["memory_scope"], "app")
        self.assertNotIn("peak_memory_bytes", result)
        self.assertEqual(result["output_token_sha256"], hashlib.sha256(b"12\n13\n").hexdigest())
        self.assertTrue(body.endswith("data: [DONE]\n\n"))

    def test_http_sampling_is_fixed_despite_model_defaults_or_payload(self):
        # Qwen adaptive sampling normally supplies temperature 0.7, not zero.
        for qwen in (False, True):
            self.runtime.installed.is_qwen = qwen
            with self.subTest(qwen=qwen), urlopen(self.request(temperature=1.8, seed=9)) as response:
                events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                          if line.startswith("data: {")]
            result = next(event["result"] for event in events if "result" in event)
            self.assertEqual(self.runtime.options.temperature, 0)
            self.assertEqual(self.runtime.options.seed, 42)
            self.assertEqual(result["temperature"], 0)
            self.assertEqual(result["seed"], 42)
            self.assertEqual(result["top_p"], 0.8 if qwen else 0.98)
            self.assertEqual(result["top_k"], 20 if qwen else 0)
            self.assertEqual(result["presence_penalty"], 1.5 if qwen else 0)
            self.assertEqual(result["repetition_penalty"], 1)

    def test_direct_trial_pins_sampling_without_mutating_options_or_settings(self):
        options = GenerationOptions(max_tokens=128, temperature=1.7, seed=7,
            top_p=0.9, top_k=10, min_p=0.05, presence_penalty=0.4, repetition_penalty=1.1)
        result = run_trial(self.runtime, options, 1024, iter, lambda _: None)
        self.assertEqual(options.temperature, 1.7)
        self.assertEqual(options.seed, 7)
        self.assertEqual(self.runtime.options.temperature, 0)
        self.assertEqual(self.runtime.options.seed, 42)
        self.assertIs(self.runtime.config, self.original_config)
        for key in ("temperature", "seed", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
            self.assertEqual(result[key], getattr(self.runtime.options, key))
        for key in ("top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty"):
            self.assertEqual(getattr(self.runtime.options, key), getattr(options, key))

    def test_normal_request_sampling_defaults_are_unchanged(self):
        from deepseek_v4_ssd.server import _options
        from deepseek_v4_ssd.model_support import get_support
        defaults = ModelDefaults(128, 1.7, 0.9, 10)
        for kind, expected in (("deepseek-v4", 1.7), ("deepseek-v4.1", 1.7), ("qwen3.8-flash-next", 0.7)):
            support = get_support(kind)
            with self.subTest(kind=kind):
                normal = _options({"max_tokens": 128}, defaults, support=support)
                self.assertEqual(normal.temperature, expected)
                self.assertIsNone(normal.seed)
                fixed = _options({"max_tokens": 128, "temperature": 0.3, "seed": 99}, defaults, support=support)
                self.assertEqual(fixed.temperature, 0.3)
                self.assertEqual(fixed.seed, 99)

    def test_status_memory_covers_each_generation_route_and_returns_to_idle(self):
        original = self.runtime.stream
        observed = []
        def stream(*args, **kwargs):
            observed.append(self.server.status_memory.snapshot())
            yield from original(*args, **kwargs)
        with patch.object(self.runtime, 'stream', side_effect=stream):
            for route in ('/v1/completions', '/v1/chat/completions', '/v1/responses', '/api/benchmark/throughput'):
                body = dict(model='deepseek-v4-flash-0731', prompt='hi', input='hi',
                            messages=[dict(role='user', content='hi')], max_tokens=2,
                            context_length=1024, generation_length=128)
                request = Request(self.url.replace('/api/benchmark/throughput', route),
                    data=json.dumps(body).encode(), headers={'Authorization': 'Bearer secret', 'Content-Type': 'application/json'})
                with self.subTest(route=route), urlopen(request) as response:
                    response.read()
                self.assertEqual(observed[-1]['sample_interval_seconds'], 0.01)
                self.assertGreaterEqual(observed[-1]['active_requests'], 1)
                # HTTP completion may precede the handler's final context exit.
                deadline = time.monotonic() + 1
                while self.server.status_memory.snapshot()['active_requests'] and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertEqual(self.server.status_memory.snapshot()['sample_interval_seconds'], 1)
        self.assertEqual(self.loading_memory_states[0]['sample_interval_seconds'], 0.01)
        self.assertEqual(len(observed), 4)

    def test_model_controls_use_fast_sampling_without_generation(self):
        observed = []
        def operation(*args, **kwargs):
            observed.append(self.server.status_memory.snapshot())
        for action in ('load', 'configure', 'unload'):
            url = self.url.replace('/api/benchmark/throughput', '/api/models/' + action)
            request = Request(url, data=b'{"model":"deepseek-v4-flash-0731","configuration":{}}',
                headers={'Authorization': 'Bearer secret', 'Content-Type': 'application/json'})
            with patch.object(self.manager, action, side_effect=operation), urlopen(request) as response:
                result = json.loads(response.read())
            self.assertEqual(observed[-1]['sample_interval_seconds'], 0.01)
            self.assertEqual(result['app_memory']['sample_interval_seconds'], 1)
            self.assertEqual(result['app_memory']['active_requests'], 0)

    def test_status_memory_reset_is_authenticated_and_not_a_work_request(self):
        from deepseek_v4_ssd.status_memory import StatusMemorySampler
        self.server.status_memory.close()
        value = [100]
        self.server.status_memory = StatusMemorySampler(reader=lambda _: value[0], pids=(11, 22)).start()
        value[0] = 10
        with self.server.status_memory.activity():
            pass
        status_url = self.url.replace('/api/benchmark/throughput', '/api/status')
        with urlopen(Request(status_url, headers={'Authorization': 'Bearer secret'})) as response:
            memory = json.loads(response.read())['app_memory']
        self.assertEqual(memory['peak_app_memory_bytes'], 200)
        self.assertEqual(memory['current_app_memory_bytes'], 20)
        self.assertEqual(memory['memory_scope'], 'app')
        self.assertEqual(memory['sample_interval_seconds'], 1)
        reset_url = status_url + '/memory/reset'
        with self.assertRaises(HTTPError) as denied:
            urlopen(Request(reset_url, data=b'{}', headers={'Content-Type': 'application/json'}))
        self.assertEqual(denied.exception.code, 401)
        denied.exception.close()
        self.assertEqual(self.server.status_memory.snapshot()['epoch'], memory['epoch'])
        with urlopen(Request(reset_url, data=b'{}', headers={'Authorization': 'Bearer secret', 'Content-Type': 'application/json'})) as response:
            reset = json.loads(response.read())['app_memory']
        self.assertNotEqual(reset['epoch'], memory['epoch'])
        self.assertEqual(reset['peak_app_memory_bytes'], 20)
        self.assertEqual(reset['active_requests'], 0)
        self.assertEqual(reset['sample_interval_seconds'], 1)

    def test_failed_generation_restores_status_idle_sampling(self):
        self.runtime.fail = True
        with urlopen(self.request()) as response:
            response.read()
        deadline = time.monotonic() + 1
        while self.server.status_memory.snapshot()['active_requests'] and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(self.server.status_memory.snapshot()['active_requests'], 0)
        self.assertEqual(self.server.status_memory.snapshot()['sample_interval_seconds'], 1)

    def test_memory_sampling_covers_loading_and_stops_after_result(self):
        from deepseek_v4_ssd.app_memory import AppMemorySampler
        samplers = []
        def create_sampler():
            sampler = AppMemorySampler(reader=lambda _: 10, pids=(11, 22), interval=60)
            samplers.append(sampler)
            return sampler
        # The 'loading' SSE event is sent before the model manager is entered.
        original_sse = self.server.RequestHandlerClass._sse
        def observe_sse(handler, event):
            if event.get("phase") == "loading":
                sampler = samplers[-1]
                self.assertTrue(sampler._thread.is_alive())
                sampler._reader = lambda _: 100
                sampler.sample()
                sampler._reader = lambda _: 10
            if "result" in event:
                self.assertIsNone(samplers[-1]._thread)
            return original_sse(handler, event)
        with patch("deepseek_v4_ssd.server.AppMemorySampler", side_effect=create_sampler), \
             patch.object(self.server.RequestHandlerClass, "_sse", observe_sse):
            with urlopen(self.request()) as response:
                events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                          if line.startswith("data: {")]
        result = next(event["result"] for event in events if "result" in event)
        self.assertEqual(result["peak_app_memory_bytes"], 200)
        self.assertIsNone(samplers[-1]._thread)

    def test_unavailable_memory_does_not_fail_generation(self):
        with patch("deepseek_v4_ssd.app_memory.physical_footprint", return_value=None):
            with urlopen(self.request()) as response:
                events = [json.loads(line[6:]) for line in response.read().decode().splitlines()
                          if line.startswith("data: {")]
        result = next(event["result"] for event in events if "result" in event)
        self.assertIsNone(result["peak_app_memory_bytes"])
        self.assertNotIn("peak_memory_bytes", result)

    def test_rejects_invalid_lengths_and_auth_before_loading(self):
        for changes in [dict(context_length=True), dict(context_length=200000),
                        dict(generation_length=256), dict(generation_length="128"),
                        dict(benchmark_context="other"), dict(benchmark_context=None),
                        dict(benchmark_context=["code"])]:
            with self.assertRaises(HTTPError) as caught:
                urlopen(self.request(**changes))
            self.assertEqual(caught.exception.code, 400)
            caught.exception.close()
        request = self.request()
        request.remove_header("Authorization")
        with self.assertRaises(HTTPError) as caught:
            urlopen(request)
        self.assertEqual(caught.exception.code, 401)
        caught.exception.close()
        self.assertEqual(self.loads, 0)

    def test_generation_error_restores_config_and_returns_stream_error(self):
        self.runtime.fail = True
        with urlopen(self.request()) as response:
            body = response.read().decode()
        self.assertIn('"error"', body)
        self.assertIn("test generation failure", body)
        self.assertIs(self.runtime.config, self.original_config)
        self.assertFalse(self.server.metrics.snapshot()["generating"])

    def test_missing_asset_returns_a_stream_error(self):
        with patch("deepseek_v4_ssd.throughput.prompt_tokens", side_effect=FileNotFoundError("Missing benchmark context")):
            with urlopen(self.request()) as response:
                body = response.read().decode()
        self.assertIn('"error"', body)
        self.assertIn("Missing benchmark context", body)
        self.assertTrue(body.endswith("data: [DONE]\n\n"))
        self.assertIs(self.runtime.config, self.original_config)

    def test_context_limit_rejected_without_generation(self):
        self.runtime.installed.maximum_context = 1100
        with urlopen(self.request()) as response:
            body = response.read().decode()
        self.assertIn("context limit", body)
        self.assertFalse(self.runtime.finished.is_set())

    def test_disconnect_cancels_prefill_and_restores_config(self):
        self.runtime.slow = True
        connection = socket.create_connection(self.server.server_address)
        data = self.request().data
        connection.sendall((
            "POST /api/benchmark/throughput HTTP/1.1\r\nHost: localhost\r\n"
            "Authorization: Bearer secret\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(data)}\r\n\r\n").encode() + data)
        self.assertTrue(self.runtime.entered.wait(3))
        connection.shutdown(socket.SHUT_RDWR)
        connection.close()
        # The App immediately asks to unload after cancellation. The endpoint
        # must wait for generation to unwind before closing the runtime.
        unload = Request(
            self.url.replace("api/benchmark/throughput", "api/models/unload"),
            data=json.dumps({"model": "deepseek-v4-flash-0731"}).encode(),
            headers={"Authorization": "Bearer secret", "Content-Type": "application/json"},
        )
        with urlopen(unload, timeout=5) as response:
            self.assertEqual(response.status, 200)
        self.assertTrue(self.runtime.finished.wait(3))
        self.assertIs(self.runtime.config, self.original_config)
        self.assertTrue(self.runtime.closed)
        self.assertIsNone(self.manager.status_snapshot()["loaded_model"])
        self.assertFalse(self.server.metrics.snapshot()["generating"])
        deadline = time.monotonic() + 1
        while self.server.status_memory.snapshot()['active_requests'] and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(self.server.status_memory.snapshot()['active_requests'], 0)
        self.assertEqual(self.server.status_memory.snapshot()['sample_interval_seconds'], 1)
