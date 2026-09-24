"""Responses tool-result compatibility, using fake runtimes (no model weights)."""
from __future__ import annotations

import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.model_manager import ModelDefaults, ModelManager, ModelSpec
from deepseek_v4_ssd.model_support import DESCRIPTORS, get_support
from deepseek_v4_ssd.server import OpenAIServer
from runtime.tests.test_server import FakeRuntime


class ResponseToolOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtimes = {}

        def load(spec):
            runtime = FakeRuntime()
            runtime.support = get_support(spec.model_kind)
            cls.runtimes[spec.id] = runtime
            return runtime

        cls.manager = ModelManager(
            [
                ModelSpec(
                    id=descriptor.api_model_id,
                    alias=None,
                    path="/tmp/tool-output-fixture",
                    model_kind=descriptor.kind,
                    runtime=RuntimeConfig(),
                    defaults=ModelDefaults(64, 0, 1, 0),
                )
                for descriptor in DESCRIPTORS
            ],
            runtime_loader=load,
            clear_cache=lambda: None,
        )
        cls.server = OpenAIServer(("127.0.0.1", 0), cls.manager, log_level="error")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1/responses"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.manager.close()

    def payload(self, model, output, *, stream=False):
        # Issue #13's text parts and request options, with the matching call
        # restored: Whallm does not store conversation history server-side.
        return {
            "model": model,
            "instructions": "the instructions",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "example_tool",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": output,
                },
            ],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "reasoning": {"effort": "high", "summary": "auto"},
            "store": False,
            "stream": stream,
            "include": ["reasoning.encrypted_content"],
        }

    def request(self, payload):
        request = Request(
            self.url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=10) as response:
                return response.status, response.headers.get_content_type(), response.read()
        except HTTPError as error:
            try:
                return error.code, error.headers.get_content_type(), error.read()
            finally:
                error.close()

    def test_strings_and_text_parts_for_every_model(self):
        cases = (
            ("plain text", "plain text"),
            ("", ""),
            ([{"type": "input_text", "text": ""}], ""),
            (
                [
                    {"type": "input_text", "text": "Wall time: 0.0517 seconds\nOutput:"},
                    {"type": "input_text", "text": "the actual output"},
                ],
                "Wall time: 0.0517 seconds\nOutput:the actual output",
            ),
            (
                [
                    {"type": "input_text", "text": "檢查結果\n"},
                    {"type": "input_text", "text": ""},
                    {"type": "input_text", "text": "  OK\n"},
                ],
                "檢查結果\n  OK\n",
            ),
        )
        for descriptor in DESCRIPTORS:
            model = descriptor.api_model_id
            for stream in (False, True):
                for output, expected in cases:
                    with self.subTest(model=model, stream=stream, output=output):
                        status, mime, body = self.request(self.payload(model, output, stream=stream))
                        self.assertEqual(status, 200, body)
                        self.assertEqual(self.runtimes[model].last_messages[-1], {
                            "role": "tool", "tool_call_id": "call_1", "content": expected,
                        })
                        if stream:
                            self.assertEqual(mime, "text/event-stream")
                            events = [json.loads(line[6:]) for line in body.decode().splitlines()
                                      if line.startswith("data: {")]
                            self.assertEqual(events[-1]["type"], "response.completed")
                            self.assertEqual(events[-1]["response"]["status"], "completed")
                        else:
                            self.assertEqual(mime, "application/json")
                            self.assertEqual(json.loads(body)["status"], "completed")

    def assert_rejected_before_generation(self, payload, param):
        model = payload["model"]
        runtime = self.runtimes.get(model)
        before = runtime.stream_call_count if runtime else 0
        status, mime, body = self.request(payload)
        self.assertEqual(status, 400, body)
        self.assertEqual(mime, "application/json")
        error = json.loads(body)["error"]
        self.assertEqual(error["type"], "invalid_request_error")
        self.assertEqual(error["param"], param)
        current = self.runtimes[model]
        self.assertEqual(current.stream_call_count, before if current is runtime else 0)

    def test_invalid_outputs_for_every_model(self):
        for descriptor in DESCRIPTORS:
            model = descriptor.api_model_id
            for output in (None, True, 42, {"text": "not an array"}):
                with self.subTest(model=model, output=output):
                    self.assert_rejected_before_generation(
                        self.payload(model, output, stream=True), "input.1.output",
                    )
            for output in (
                ["not an object"], [None], [{}],
                [{"type": "input_text"}],
                [{"type": "input_text", "text": 42}],
                [{"type": "input_text", "text": None}],
                [{"type": "unknown", "text": "not text"}],
                [{"type": "input_text", "text": "valid"}, {"type": "input_text", "text": False}],
            ):
                with self.subTest(model=model, output=output):
                    self.assert_rejected_before_generation(
                        self.payload(model, output, stream=True), "messages.2.content",
                    )

    def test_non_text_outputs_stay_unsupported_for_text_models(self):
        for descriptor in DESCRIPTORS:
            if any(descriptor.supports(f) for f in ("imageInput", "documentInput", "audioInput")):
                continue
            for kind in ("input_image", "input_file", "input_audio"):
                with self.subTest(model=descriptor.api_model_id, kind=kind):
                    output = [{"type": "input_text", "text": "valid"}, {"type": kind}]
                    self.assert_rejected_before_generation(
                        self.payload(descriptor.api_model_id, output, stream=True), "messages.2.content",
                    )

    def test_text_parts_still_require_a_matching_call(self):
        output = [{"type": "input_text", "text": "result"}]
        for descriptor in DESCRIPTORS:
            model = descriptor.api_model_id
            for call_id in (None, "", "call_other"):
                with self.subTest(model=model, call_id=call_id):
                    payload = self.payload(model, output, stream=True)
                    payload["input"][-1]["call_id"] = call_id
                    self.assert_rejected_before_generation(
                        payload, "messages.2.tool_call_id" if call_id else "input.1.call_id",
                    )
            with self.subTest(model=model, history="missing"):
                payload = self.payload(model, output, stream=True)
                payload["input"].pop(0)
                self.assert_rejected_before_generation(payload, "messages.1.tool_call_id")


if __name__ == "__main__":
    unittest.main()
