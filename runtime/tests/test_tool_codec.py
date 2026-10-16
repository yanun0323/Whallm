from __future__ import annotations

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from types import SimpleNamespace

from deepseek_v4_ssd.tool_codec import (
    DeepSeekV41ToolCodec,
    DeepSeekV41ToolStreamParser,
    ToolCall,
    ToolChoice,
    ToolCodec,
    ToolStreamParser,
)


class ToolCodecTests(unittest.TestCase):
    def test_codec_loads_encoder_and_normalizes_tool_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoding = root / "encoding"
            encoding.mkdir()
            (encoding / "encoding_dsv4.py").write_text(
                '''
eos_token = "<eos>"

def encode_messages(messages, thinking_mode, reasoning_effort=None):
    return repr((messages, thinking_mode, reasoning_effort))

def parse_message_from_completion_text(text, thinking_mode):
    assert text.endswith(eos_token)
    return {
        "content": "",
        "reasoning_content": "plan" if thinking_mode == "thinking" else "",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Taipei"}'},
        }],
    }
''',
                encoding="utf-8",
            )
            digest = sha256((encoding / "encoding_dsv4.py").read_bytes()).hexdigest()
            with patch("deepseek_v4_ssd.tool_codec.ENCODER_SHA256", digest):
                codec = ToolCodec.open(root)
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ]

            prompt = codec.encode(
                [{"role": "user", "content": "Weather?"}],
                "thinking",
                tools,
                ToolChoice("required"),
                "max",
            )
            turn = codec.parse("raw", "thinking")

            self.assertIn("Call one or more tools", prompt)
            self.assertIn("get_weather", prompt)
            self.assertIn("'max'", prompt)
            self.assertEqual(turn.reasoning_content, "plan")
            self.assertEqual(turn.tool_calls[0].name, "get_weather")
            self.assertEqual(turn.tool_calls[0].arguments, '{"city":"Taipei"}')

    def test_v41_codec_sends_developer_messages_as_system(self):
        # The V4.1 encoder raises on an unknown role; Responses instructions arrive as developer.
        def encode_messages(messages, thinking_mode, reasoning_effort=None):
            for message in messages:
                if message["role"] not in {"system", "user", "assistant", "tool"}:
                    raise NotImplementedError(f"Unknown role: {message['role']}")
            return [(message["role"], message.get("content"), bool(message.get("tools"))) for message in messages]

        codec = DeepSeekV41ToolCodec(SimpleNamespace(encode_messages=encode_messages))
        messages = [
            {"role": "developer", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "developer", "content": "Now be formal."},
            {"role": "user", "content": "Bye"},
        ]
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
        self.assertEqual(
            codec.encode(messages, "chat", tools),
            [("system", "Be brief.", True), ("user", "Hi", False), ("assistant", "Hello", False),
             ("system", "Now be formal.", False), ("user", "Bye", False)],
        )
        self.assertEqual(messages[0]["role"], "developer")

    def test_codec_rejects_an_unverified_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            encoding = root / "encoding"
            encoding.mkdir()
            (encoding / "encoding_dsv4.py").write_text("unverified = True")

            with self.assertRaisesRegex(RuntimeError, "checksum"):
                ToolCodec.open(root)

    def test_tool_stream_parser_handles_single_character_fragments(self):
        raw = (
            'plan</think>Summary\n\n<｜DSML｜tool_calls>\n'
            '<｜DSML｜invoke name="get_weather">\n'
            '<｜DSML｜parameter name="city" string="true">台"北'
            '</｜DSML｜parameter>\n'
            '<｜DSML｜parameter name="days" string="false">2'
            '</｜DSML｜parameter>\n'
            '</｜DSML｜invoke>\n</｜DSML｜tool_calls>'
        )
        parser = ToolStreamParser("thinking")
        deltas = [delta for character in raw for delta in parser.feed(character)]
        deltas.extend(parser.finish())

        reasoning = "".join(delta.reasoning_content for delta in deltas)
        content = "".join(delta.content for delta in deltas)
        name = next(delta.tool_name for delta in deltas if delta.tool_name)
        arguments = "".join(
            delta.arguments for delta in deltas if delta.tool_index == 0
        )
        calls = (ToolCall(name, arguments),)

        self.assertEqual(reasoning, "plan")
        self.assertEqual(content, "Summary")
        self.assertEqual(name, "get_weather")
        self.assertEqual(arguments, '{"city": "台\\"北", "days": 2}')
        self.assertTrue(parser.matches(calls))

    def test_v41_tool_stream_parser_handles_single_character_fragments(self):
        raw = (
            'plan</think>Summary\n\n<｜DSML｜ calls>\n'
            '<｜DSML｜ invoke name="get_weather">\n'
            '<｜DSML｜ parameter name="city" string="true">Paris'
            '</｜DSML｜ parameter>\n'
            '<｜DSML｜ parameter name="days" string="false">2'
            '</｜DSML｜ parameter>\n'
            '</｜DSML｜ invoke>\n</｜DSML｜ calls>'
        )
        parser = DeepSeekV41ToolStreamParser("thinking")
        deltas = [delta for character in raw for delta in parser.feed(character)]
        deltas.extend(parser.finish())

        reasoning = "".join(delta.reasoning_content for delta in deltas)
        content = "".join(delta.content for delta in deltas)
        name = next(delta.tool_name for delta in deltas if delta.tool_name)
        arguments = "".join(
            delta.arguments for delta in deltas if delta.tool_index == 0
        )
        calls = (ToolCall(name, arguments),)

        self.assertEqual(reasoning, "plan")
        self.assertEqual(content, "Summary")
        self.assertEqual(name, "get_weather")
        self.assertEqual(arguments, '{"city": "Paris", "days": 2}')
        self.assertTrue(parser.matches(calls))


if __name__ == "__main__":
    unittest.main()
