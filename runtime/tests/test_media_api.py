import json
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from unittest.mock import patch
from deepseek_v4_ssd.media import ImagePart, expand_documents
from deepseek_v4_ssd.documents import DOCX
from runtime.tests.test_documents import docx
from deepseek_v4_ssd.model_support import get_support
from deepseek_v4_ssd.model_manager import ModelManager, ModelSpec, ModelDefaults
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.server import OpenAIServer, MAX_REQUEST_BYTES
from runtime.tests.test_server import FakeRuntime
from runtime.tests.test_media import png


class MediaAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = FakeRuntime()
        cls.runtime.support = get_support("mimo-v2.6-flash-rl")
        cls.manager = ModelManager([ModelSpec(id="mimo-v2.6-flash-rl", alias=None, path="/tmp/fixture",
            model_kind="mimo-v2.6-flash-rl", runtime=RuntimeConfig(), defaults=ModelDefaults(64, 0, 1, 0))],
            runtime_loader=lambda _: cls.runtime, clear_cache=lambda: None)
        cls.server = OpenAIServer(("127.0.0.1", 0), cls.manager, api_key="fixture-key", log_level="error")
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(); cls.manager.close()

    def request(self, path, *, data=None, method=None, mime="application/json", auth=True):
        headers = {"Content-Type": mime}
        if auth: headers["Authorization"] = "Bearer fixture-key"
        if isinstance(data, dict): data = json.dumps(data).encode()
        try:
            with urlopen(Request(self.base + path, data=data, headers=headers, method=method)) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, json.load(error)

    def test_upload_reference_chat_responses_delete(self):
        status, uploaded = self.request("/api/assets", data=png(), mime="image/png")
        self.assertEqual(status, 201)
        parts = [{"type": "text", "text": "before"}, {"type": "image", "file_id": uploaded["id"]},
                 {"type": "text", "text": "after"}]
        for endpoint, body in (("/v1/chat/completions", {"messages": [{"role": "user", "content": parts}]}),
                               ("/v1/responses", {"input": [{"role": "user", "content": parts}]})):
            status, result = self.request(endpoint, data={"model": "mimo-v2.6-flash-rl", **body})
            self.assertEqual(status, 200, result)
            content = self.runtime.last_messages[0]["content"]
            self.assertEqual(content[0], "before")
            self.assertIsInstance(content[1], ImagePart)
            self.assertEqual(content[2], "after")
        status, _ = self.request("/api/assets/" + uploaded["id"], method="DELETE")
        self.assertEqual(status, 200)
        status, _ = self.request("/v1/chat/completions", data={"model": "mimo-v2.6-flash-rl", "messages": [{"role": "user", "content": parts}]})
        self.assertEqual(status, 400)

    def test_responses_tool_result_preserves_supported_media(self):
        status, uploaded = self.request("/api/assets", data=png(), mime="image/png")
        self.assertEqual(status, 201)
        try:
            status, result = self.request("/v1/responses", data={
                "model": "mimo-v2.6-flash-rl",
                "input": [
                    {"type": "function_call", "call_id": "call_1", "name": "get_image", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "call_1", "output": [
                        {"type": "input_text", "text": "before"},
                        {"type": "input_image", "file_id": uploaded["id"]},
                        {"type": "input_text", "text": "after"},
                    ]},
                ],
            })
            self.assertEqual(status, 200, result)
            message = self.runtime.last_messages[-1]
            self.assertEqual(message["role"], "tool")
            self.assertEqual(message["tool_call_id"], "call_1")
            self.assertEqual(message["content"][0], "before")
            self.assertIsInstance(message["content"][1], ImagePart)
            self.assertEqual(message["content"][2], "after")
        finally:
            self.request("/api/assets/" + uploaded["id"], method="DELETE")

    def test_authentication_and_capabilities(self):
        for path, data, method in (("/api/assets", png(), "POST"), ("/api/assets/file-missing", None, "DELETE"),
                                    ("/api/capabilities", None, "GET")):
            status, _ = self.request(path, data=data, method=method, mime="image/png", auth=False)
            self.assertEqual(status, 401)
        status, result = self.request("/api/capabilities")
        self.assertEqual(status, 200)
        self.assertEqual(result["models"]["mimo-v2.6-flash-rl"]["input"], ["text", "image", "document", "audio"])
        self.assertEqual(result["models"]["qwen3.8-flash-next-fp8"]["input"], ["text"])
        self.assertFalse(result["media"]["remote_urls"])
        self.assertEqual(MAX_REQUEST_BYTES, 1_048_576)

    def test_document_upload_both_adapters_and_malformed_document_error(self):
        original = self.runtime.encode_chat
        def encode(messages, *args, **kwargs):
            return original(expand_documents(messages, enabled=True), *args, **kwargs)
        status, uploaded = self.request('/api/assets', data=docx(), mime=DOCX)
        self.assertEqual(status, 201)
        with patch.object(self.runtime, 'encode_chat', side_effect=encode):
            for endpoint, field in (('/v1/chat/completions', 'messages'), ('/v1/responses', 'input')):
                status, result = self.request(endpoint, data={'model':'mimo-v2.6-flash-rl', field:[
                    {'role':'user','content':[{'type':'input_file','file_id':uploaded['id']}]}]})
                self.assertEqual(status, 200, result)
                self.assertIn('Invoice total: 42', self.runtime.last_messages[0]['content'])
            self.request('/api/assets/' + uploaded['id'], method='DELETE')
            status, bad = self.request('/api/assets', data=b'not a zip', mime=DOCX)
            self.assertEqual(status, 201)
            before = self.runtime.stream_call_count
            status, result = self.request('/v1/chat/completions', data={'model':'mimo-v2.6-flash-rl','messages':[
                {'role':'user','content':[{'type':'file','file_id':bad['id']}]}]})
            self.assertEqual(status, 400, result)
            self.assertEqual(self.runtime.stream_call_count, before)
            self.request('/api/assets/' + bad['id'], method='DELETE')

    def test_audio_upload_adapters_and_malformed_audio_before_stream(self):
        from runtime.tests.test_mimo_audio import wav
        from deepseek_v4_ssd.media import media_request
        from deepseek_v4_ssd.mimo.inputs import validate_media
        from deepseek_v4_ssd.mimo.audio_processor import AudioInput
        original = self.runtime.encode_chat
        def encode(messages, *args, **kwargs):
            request = media_request(messages, 'chat', [], None, 'low')
            return original(list(validate_media(request).messages), *args, **kwargs)
        status, asset = self.request('/api/assets', data=wav(), mime='audio/wav')
        self.assertEqual(status, 201)
        with patch.object(self.runtime, 'encode_chat', side_effect=encode):
            for endpoint, field in (('/v1/chat/completions','messages'),('/v1/responses','input')):
                status, result = self.request(endpoint, data={'model':'mimo-v2.6-flash-rl', field:[
                    {'role':'user','content':[{'type':'input_audio','file_id':asset['id']}]}]})
                self.assertEqual(status, 200, result)
                self.assertIsInstance(self.runtime.last_messages[0]['content'][0], AudioInput)
            self.request('/api/assets/' + asset['id'], method='DELETE')
            status, bad = self.request('/api/assets', data=wav(rate=16000), mime='audio/wav')
            self.assertEqual(status, 201)
            before = self.runtime.stream_call_count
            status, result = self.request('/v1/chat/completions', data={'model':'mimo-v2.6-flash-rl','stream':True,'messages':[
                {'role':'user','content':[{'type':'input_audio','file_id':bad['id']}]}]})
            self.assertEqual(status, 400, result)
            self.assertEqual(self.runtime.stream_call_count, before)
            self.request('/api/assets/' + bad['id'], method='DELETE')
        _, caps = self.request('/api/capabilities')
        self.assertEqual(caps['media']['audio_sample_rate'], 24000)
        self.assertEqual(caps['media']['max_audio_seconds'], 30)
        self.assertEqual(caps['media']['max_audios'], 2)

    def test_remote_url_rejected_without_generation(self):
        before = self.runtime.stream_call_count
        status, _ = self.request("/v1/chat/completions", data={"model": "mimo-v2.6-flash-rl", "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://127.0.0.1/private"}}]}]})
        self.assertEqual(status, 400)
        self.assertEqual(before, self.runtime.stream_call_count)
