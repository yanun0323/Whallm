from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

ENCODER_SHA256 = (
    "abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6"
)
V41_ENCODER_SHA256 = (
    "502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1"
)


@dataclass(frozen=True)
class ToolChoice:
    mode: str = "auto"
    name: str | None = None


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: str


@dataclass(frozen=True)
class AssistantTurn:
    content: str
    reasoning_content: str
    tool_calls: tuple[ToolCall, ...]


@dataclass(frozen=True)
class ToolStreamDelta:
    reasoning_content: str = ""
    content: str = ""
    tool_index: int | None = None
    tool_name: str | None = None
    arguments: str = ""


class ToolStreamParser:
    """Incrementally separate DeepSeek text and DSML tool calls."""

    _tool_start = "\n\n<｜DSML｜tool_calls>"
    _tools_end = "</｜DSML｜tool_calls>"
    _invoke_start = "<｜DSML｜invoke"
    _invoke_end = "</｜DSML｜invoke>"
    _parameter_start = "<｜DSML｜parameter"
    _parameter_end = "</｜DSML｜parameter>"

    def __init__(self, thinking_mode: str):
        self._state = "reasoning" if thinking_mode == "thinking" else "content"
        self._buffer = ""
        self._tool_index = -1
        self._parameter_is_string = False
        self._names: list[str] = []
        self._arguments: list[str] = []
        self.failed = False

    def feed(self, text: str) -> tuple[ToolStreamDelta, ...]:
        if self.failed:
            return ()
        self._buffer += text
        deltas: list[ToolStreamDelta] = []
        while not self.failed:
            before = (self._state, self._buffer)
            if self._state == "reasoning":
                self._read_text("</think>", "content", deltas, reasoning=True)
            elif self._state == "content":
                self._read_text(self._tool_start, "tool", deltas)
            elif self._state == "tool":
                self._read_tool(deltas)
            elif self._state == "parameter_or_end":
                self._read_parameter_or_end(deltas)
            elif self._state == "parameter_value":
                self._read_parameter_value(deltas)
            else:
                break
            if before == (self._state, self._buffer):
                break
        return tuple(deltas)

    def finish(self) -> tuple[ToolStreamDelta, ...]:
        if self.failed:
            return ()
        if self._state == "content" and self._buffer:
            delta = ToolStreamDelta(content=self._buffer)
            self._buffer = ""
            return (delta,)
        return ()

    def matches(self, calls: tuple[ToolCall, ...]) -> bool:
        if (
            self.failed
            or len(calls) != len(self._names)
            or (calls and self._state != "done")
        ):
            return False
        for index, call in enumerate(calls):
            if call.name != self._names[index]:
                return False
            try:
                if json.loads(call.arguments) != json.loads(self._arguments[index]):
                    return False
            except json.JSONDecodeError:
                return False
        return True

    @property
    def streamed_tool_count(self) -> int:
        return len(self._names)

    def _read_text(
        self,
        marker: str,
        next_state: str,
        deltas: list[ToolStreamDelta],
        *,
        reasoning: bool = False,
    ) -> None:
        position = self._buffer.find(marker)
        if position >= 0:
            self._emit_text(self._buffer[:position], deltas, reasoning)
            self._buffer = self._buffer[position + len(marker) :]
            self._state = next_state
            return
        keep = self._marker_suffix_length(self._buffer, marker)
        ready = self._buffer[:-keep] if keep else self._buffer
        self._buffer = self._buffer[-keep:] if keep else ""
        self._emit_text(ready, deltas, reasoning)

    @staticmethod
    def _emit_text(
        text: str,
        deltas: list[ToolStreamDelta],
        reasoning: bool,
    ) -> None:
        if text:
            deltas.append(
                ToolStreamDelta(
                    reasoning_content=text if reasoning else "",
                    content="" if reasoning else text,
                )
            )

    def _read_tool(self, deltas: list[ToolStreamDelta]) -> None:
        self._buffer = self._buffer.lstrip()
        if not self._buffer:
            return
        if self._wait_for_marker(self._tools_end):
            if self._buffer.startswith(self._tools_end):
                self._buffer = self._buffer[len(self._tools_end) :]
                self._state = "done"
            return
        if self._invoke_start.startswith(self._buffer):
            return
        if not self._buffer.startswith(self._invoke_start):
            self.failed = True
            return
        end = self._buffer.find(">\n")
        if end < 0:
            return
        header = self._buffer[: end + 1]
        match = re.fullmatch(
            re.escape(self._invoke_start) + r' name="([A-Za-z0-9_-]{1,64})">',
            header,
        )
        if match is None:
            self.failed = True
            return
        self._buffer = self._buffer[end + 2 :]
        self._tool_index += 1
        name = match.group(1)
        self._names.append(name)
        self._arguments.append("")
        self._state = "parameter_or_end"
        deltas.append(ToolStreamDelta(tool_index=self._tool_index, tool_name=name))

    def _read_parameter_or_end(self, deltas: list[ToolStreamDelta]) -> None:
        self._buffer = self._buffer.lstrip()
        if not self._buffer:
            return
        if self._wait_for_marker(self._invoke_end):
            if self._buffer.startswith(self._invoke_end):
                self._buffer = self._buffer[len(self._invoke_end) :]
                self._append_arguments("}", deltas, empty="{}")
                self._state = "tool"
            return
        if self._parameter_start.startswith(self._buffer):
            return
        if not self._buffer.startswith(self._parameter_start):
            self.failed = True
            return
        end = self._buffer.find(">")
        if end < 0:
            return
        header = self._buffer[: end + 1]
        match = re.fullmatch(
            re.escape(self._parameter_start)
            + r' name="([^"]+)" string="(true|false)">',
            header,
        )
        if match is None:
            self.failed = True
            return
        self._buffer = self._buffer[end + 1 :]
        self._parameter_is_string = match.group(2) == "true"
        prefix = (
            ("{" if not self._arguments[self._tool_index] else ", ")
            + json.dumps(match.group(1), ensure_ascii=False)
            + ": "
            + ('"' if self._parameter_is_string else "")
        )
        self._append_arguments(prefix, deltas)
        self._state = "parameter_value"

    def _read_parameter_value(self, deltas: list[ToolStreamDelta]) -> None:
        position = self._buffer.find(self._parameter_end)
        if position >= 0:
            self._append_value(self._buffer[:position], deltas)
            self._buffer = self._buffer[position + len(self._parameter_end) :]
            if self._parameter_is_string:
                self._append_arguments('"', deltas)
            self._state = "parameter_or_end"
            return
        keep = self._marker_suffix_length(self._buffer, self._parameter_end)
        ready = self._buffer[:-keep] if keep else self._buffer
        self._buffer = self._buffer[-keep:] if keep else ""
        self._append_value(ready, deltas)

    def _append_value(
        self,
        value: str,
        deltas: list[ToolStreamDelta],
    ) -> None:
        if not value:
            return
        fragment = (
            json.dumps(value, ensure_ascii=False)[1:-1]
            if self._parameter_is_string
            else value
        )
        self._append_arguments(fragment, deltas)

    def _append_arguments(
        self,
        fragment: str,
        deltas: list[ToolStreamDelta],
        *,
        empty: str | None = None,
    ) -> None:
        if empty is not None and not self._arguments[self._tool_index]:
            fragment = empty
        self._arguments[self._tool_index] += fragment
        deltas.append(
            ToolStreamDelta(tool_index=self._tool_index, arguments=fragment)
        )

    def _wait_for_marker(self, marker: str) -> bool:
        return self._buffer.startswith(marker) or marker.startswith(self._buffer)

    @staticmethod
    def _marker_suffix_length(text: str, marker: str) -> int:
        maximum = min(len(text), len(marker) - 1)
        for length in range(maximum, 0, -1):
            if text.endswith(marker[:length]):
                return length
        return 0


class DeepSeekV41ToolStreamParser(ToolStreamParser):
    _tool_start = "\n\n<｜DSML｜ calls>"
    _tools_end = "</｜DSML｜ calls>"
    _invoke_start = "<｜DSML｜ invoke"
    _invoke_end = "</｜DSML｜ invoke>"
    _parameter_start = "<｜DSML｜ parameter"
    _parameter_end = "</｜DSML｜ parameter>"


class ToolCodec:
    """Use the pinned DeepSeek-V4 encoder through one small interface."""

    def __init__(self, encoding: ModuleType):
        self._encoding = encoding

    @classmethod
    def open(cls, model_root: Path, tokenizer=None) -> ToolCodec:
        from .model_support import get_support
        manifest_path = model_root / "manifest.json"
        kind = "deepseek-v4"
        if manifest_path.is_file():
            with manifest_path.open("rb") as file:
                manifest = json.load(file)
            kind = manifest.get("modelKind", kind)
        return get_support(kind).open_codec(model_root, tokenizer)

    @classmethod
    def from_encoder(cls, path: Path, expected_digest: str) -> ToolCodec:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_digest:
            raise RuntimeError("DeepSeek encoder checksum does not match")
        spec = importlib.util.spec_from_file_location(
            f"_deepseek_v4_encoding_{abs(hash(path))}",
            path,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load DeepSeek-V4 encoder: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not hasattr(module, "encode_messages") or not hasattr(
            module,
            "parse_message_from_completion_text",
        ):
            raise RuntimeError("DeepSeek-V4 encoder is missing required functions")
        return cls(module)

    def encode(
        self,
        messages: list[dict[str, Any]],
        thinking_mode: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = ToolChoice(),
        reasoning_effort: str = "low",
    ) -> str:
        prepared = copy.deepcopy(messages)
        active_tools = [] if tool_choice.mode == "none" else list(tools or [])
        if active_tools:
            instruction = self._choice_instruction(tool_choice)
            target = next(
                (message for message in prepared if message["role"] == "system"),
                None,
            )
            if target is None:
                target = {"role": "system", "content": ""}
                prepared.insert(0, target)
            if instruction:
                content = target.get("content") or ""
                target["content"] = f"{content}\n\n{instruction}".strip()
            target["tools"] = active_tools
        return self._encoding.encode_messages(
            prepared,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )

    def parse(self, text: str, thinking_mode: str) -> AssistantTurn:
        eos = self._encoding.eos_token
        complete = text if text.endswith(eos) else text + eos
        parsed = self._encoding.parse_message_from_completion_text(
            complete,
            thinking_mode=thinking_mode,
        )
        calls = tuple(
            ToolCall(
                name=call["function"]["name"],
                arguments=call["function"]["arguments"],
            )
            for call in parsed.get("tool_calls") or []
        )
        return AssistantTurn(
            content=parsed.get("content") or "",
            reasoning_content=parsed.get("reasoning_content") or "",
            tool_calls=calls,
        )

    @staticmethod
    def _choice_instruction(choice: ToolChoice) -> str:
        if choice.mode == "required":
            return "Call one or more tools before you give a final answer."
        if choice.mode == "function":
            return f'Call the "{choice.name}" tool before you give a final answer.'
        return ""


def _qwen_parameter_value(value: str) -> Any:
    value = value.strip()
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoded = value
    return _unescape_qwen_value(decoded)


_QWEN_RESERVED_MARKERS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<think>",
    "</think>",
    "<tools>",
    "</tools>",
    "<tool_call>",
    "</tool_call>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
    "<tool_response>",
    "</tool_response>",
)
_QWEN_TOOL_ESCAPE_INSTRUCTION = (
    "Inside tool parameter values, use &lt; instead of the opening angle bracket "
    "of Qwen control markers. Use &amp;lt; when the intended value already contains "
    "&lt;."
)


def _escape_qwen_text(text: str) -> str:
    if not any(
        marker in text or f"&lt;{marker[1:]}" in text
        for marker in _QWEN_RESERVED_MARKERS
    ):
        return text
    text = text.replace("&", "&amp;")
    for marker in _QWEN_RESERVED_MARKERS:
        text = text.replace(marker, f"&lt;{marker[1:]}")
    return text


def _escape_qwen_value(value: Any) -> Any:
    if isinstance(value, str):
        return _escape_qwen_text(value)
    if isinstance(value, list):
        return [_escape_qwen_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _escape_qwen_value(item) for key, item in value.items()}
    return value


def _unescape_qwen_text(text: str) -> str:
    if not any(
        f"&lt;{marker[1:]}" in text or f"&amp;lt;{marker[1:]}" in text
        for marker in _QWEN_RESERVED_MARKERS
    ):
        return text
    for marker in _QWEN_RESERVED_MARKERS:
        text = text.replace(f"&lt;{marker[1:]}", marker)
    return text.replace("&amp;", "&")


def _unescape_qwen_value(value: Any) -> Any:
    if isinstance(value, str):
        return _unescape_qwen_text(value)
    if isinstance(value, list):
        return [_unescape_qwen_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _unescape_qwen_value(item) for key, item in value.items()}
    return value


def _parse_qwen_call(text: str) -> ToolCall:
    match = re.fullmatch(
        r"<tool_call>\s*<function=([A-Za-z0-9_-]{1,64})>\s*(.*?)\s*</function>\s*</tool_call>",
        text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError("invalid Qwen tool call XML")
    body = match.group(2)
    arguments: dict[str, Any] = {}
    position = 0
    pattern = re.compile(
        r"\s*<parameter=([^<>\r\n]*)>\s*\n?(.*?)\n?\s*</parameter>",
        re.DOTALL,
    )
    while position < len(body):
        parameter = pattern.match(body, position)
        if parameter is None:
            if body[position:].strip():
                raise ValueError("invalid Qwen tool parameter XML")
            break
        name = parameter.group(1)
        if name in arguments:
            raise ValueError(f"duplicate Qwen tool parameter {name}")
        arguments[name] = _qwen_parameter_value(parameter.group(2))
        position = parameter.end()
    return ToolCall(
        match.group(1),
        json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
    )


def _split_qwen_output(text: str, thinking_mode: str) -> AssistantTurn:
    reasoning = ""
    content_and_calls = text
    if thinking_mode == "thinking":
        if "</think>" not in text:
            raise ValueError("Qwen thinking output has no closing tag")
        reasoning, content_and_calls = text.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
        content_and_calls = content_and_calls.lstrip("\n")

    first_call = content_and_calls.find("<tool_call>")
    if first_call < 0:
        if "</tool_call>" in content_and_calls or "<function=" in content_and_calls:
            raise ValueError("invalid Qwen tool call XML")
        return AssistantTurn(content_and_calls, reasoning, ())

    content = content_and_calls[:first_call].rstrip()
    suffix = content_and_calls[first_call:]
    calls = []
    while suffix.strip():
        suffix = suffix.lstrip()
        end = suffix.find("</tool_call>")
        if end < 0:
            raise ValueError("Qwen tool call XML is incomplete")
        end += len("</tool_call>")
        calls.append(_parse_qwen_call(suffix[:end]))
        suffix = suffix[end:]
    return AssistantTurn(content, reasoning, tuple(calls))


class QwenToolCodec(ToolCodec):
    """Use the checkpoint chat template and Qwen XML tool format."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer

    def encode(
        self,
        messages: list[dict[str, Any]],
        thinking_mode: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: ToolChoice = ToolChoice(),
        reasoning_effort: str = "low",
    ) -> str:
        prepared = copy.deepcopy(messages)
        instruction_contents = []
        conversation = []
        for message in prepared:
            content = message.get("content")
            if isinstance(content, str):
                content = _escape_qwen_text(content)
                message["content"] = content
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str):
                message["reasoning_content"] = _escape_qwen_text(reasoning)
            if message.get("role") in {"system", "developer"}:
                if content:
                    instruction_contents.append(content)
                continue
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function", tool_call)
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                    if not isinstance(arguments, dict):
                        raise ValueError(
                            "Qwen tool call arguments must contain a JSON object"
                        )
                if isinstance(arguments, dict):
                    function["arguments"] = _escape_qwen_value(arguments)
            conversation.append(message)
        prepared = conversation
        if instruction_contents:
            prepared.insert(
                0,
                {"role": "system", "content": "\n\n".join(instruction_contents)},
            )
        if not any(message.get("role") == "user" for message in prepared):
            index = 1 if prepared and prepared[0].get("role") == "system" else 0
            prepared.insert(index, {"role": "user", "content": ""})
        active_tools = (
            []
            if tool_choice.mode == "none"
            else _escape_qwen_value(list(tools or []))
        )
        instruction = ""
        if active_tools:
            instruction = "\n\n".join(
                part
                for part in (
                    self._choice_instruction(tool_choice),
                    _QWEN_TOOL_ESCAPE_INSTRUCTION,
                )
                if part
            )
        if instruction:
            system = next(
                (message for message in prepared if message.get("role") == "system"),
                None,
            )
            if system is None:
                system = {"role": "system", "content": ""}
                prepared.insert(0, system)
            system["content"] = f"{system.get('content') or ''}\n\n{instruction}".strip()
        effort = {
            "low": "low",
            "medium": "medium",
            "high": "xhigh",
            "xhigh": "xhigh",
            "max": "xhigh",
        }.get(reasoning_effort, "low")
        return self._tokenizer.apply_chat_template(
            prepared,
            tools=active_tools or None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking_mode == "thinking",
            reasoning_effort=effort,
        )

    def parse(self, text: str, thinking_mode: str) -> AssistantTurn:
        return _split_qwen_output(text, thinking_mode)


class QwenToolStreamParser:
    """Incrementally separate Qwen text and XML tool calls."""

    def __init__(self, thinking_mode: str):
        self._state = "reasoning" if thinking_mode == "thinking" else "content"
        self._buffer = ""
        self._names: list[str] = []
        self._arguments: list[str] = []
        self.failed = False

    def feed(self, text: str) -> tuple[ToolStreamDelta, ...]:
        if self.failed:
            return ()
        self._buffer += text
        deltas: list[ToolStreamDelta] = []
        while not self.failed:
            before = (self._state, self._buffer)
            if self._state == "reasoning":
                self._read_text("</think>", "content", deltas, reasoning=True)
                if self._state == "content":
                    self._buffer = self._buffer.lstrip("\n")
            elif self._state == "content":
                self._read_text("<tool_call>", "tool", deltas)
                if self._state == "tool":
                    self._buffer = "<tool_call>" + self._buffer
            elif self._state == "tool":
                # Separators may arrive after the previous call's closing chunk.
                # Match the complete-output parser without touching parameter values.
                self._buffer = self._buffer.lstrip()
                end = self._buffer.find("</tool_call>")
                if end < 0:
                    break
                end += len("</tool_call>")
                try:
                    call = _parse_qwen_call(self._buffer[:end])
                except ValueError:
                    self.failed = True
                    break
                index = len(self._names)
                self._names.append(call.name)
                self._arguments.append(call.arguments)
                deltas.append(ToolStreamDelta(tool_index=index, tool_name=call.name))
                deltas.append(ToolStreamDelta(tool_index=index, arguments=call.arguments))
                self._buffer = self._buffer[end:].lstrip()
            if before == (self._state, self._buffer):
                break
        return tuple(deltas)

    def finish(self) -> tuple[ToolStreamDelta, ...]:
        if self.failed:
            return ()
        if self._state == "content" and self._buffer:
            value, self._buffer = self._buffer, ""
            return (ToolStreamDelta(content=value),)
        if self._state == "tool" and self._buffer.strip():
            self.failed = True
        return ()

    def matches(self, calls: tuple[ToolCall, ...]) -> bool:
        if self.failed or len(calls) != len(self._names):
            return False
        return all(
            call.name == self._names[index]
            and json.loads(call.arguments) == json.loads(self._arguments[index])
            for index, call in enumerate(calls)
        )

    @property
    def streamed_tool_count(self) -> int:
        return len(self._names)

    def _read_text(
        self,
        marker: str,
        next_state: str,
        deltas: list[ToolStreamDelta],
        *,
        reasoning: bool = False,
    ) -> None:
        position = self._buffer.find(marker)
        if position >= 0:
            self._emit(self._buffer[:position], deltas, reasoning)
            self._buffer = self._buffer[position + len(marker) :]
            self._state = next_state
            return
        keep = ToolStreamParser._marker_suffix_length(self._buffer, marker)
        ready = self._buffer[:-keep] if keep else self._buffer
        self._buffer = self._buffer[-keep:] if keep else ""
        self._emit(ready, deltas, reasoning)

    @staticmethod
    def _emit(
        value: str,
        deltas: list[ToolStreamDelta],
        reasoning: bool,
    ) -> None:
        if value:
            deltas.append(
                ToolStreamDelta(
                    reasoning_content=value if reasoning else "",
                    content="" if reasoning else value,
                )
            )
