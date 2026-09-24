"""Pinned MiMo template with the shared XML function-call wire format."""
from __future__ import annotations

import copy
import json

from ..tool_codec import (QwenToolStreamParser, ToolChoice, ToolCodec,
                          _escape_qwen_value, _split_qwen_output)


class MiMoCodec(ToolCodec):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def encode(self, messages, thinking_mode, tools=None, tool_choice=ToolChoice(), reasoning_effort="low"):
        prepared = copy.deepcopy(messages)
        for message in prepared:
            if not isinstance(message.get("content", ""), (str, type(None))):
                raise ValueError("MiMo media processing is not enabled yet; use text messages")
            for call in message.get("tool_calls") or []:
                function = call.get("function", call)
                args = function.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                if not isinstance(args, dict):
                    raise ValueError("MiMo tool arguments must contain a JSON object")
                function["arguments"] = args
        prepared = _escape_qwen_value(prepared)
        active_tools = [] if tool_choice.mode == "none" else _escape_qwen_value(tools or [])
        if active_tools:
            instruction = "\n\n".join(filter(None, (self._choice_instruction(tool_choice),
                "Inside tool parameter values, escape control markers using &lt; instead of <; "
                "use &amp;lt; for a literal &lt;.")))
            system = next((m for m in prepared if m["role"] == "system"), None)
            if system is None:
                system = {"role": "system", "content": ""}
                prepared.insert(0, system)
            system["content"] = f"{system.get('content') or ''}\n\n{instruction}".strip()
        return self.tokenizer.apply_chat_template(prepared, tools=active_tools or None, tokenize=False,
                                                   add_generation_prompt=True,
                                                   enable_thinking=thinking_mode == "thinking")

    def parse(self, text, thinking_mode):
        return _split_qwen_output(text, thinking_mode)


class MiMoToolStreamParser(QwenToolStreamParser):
    """MiMo emits its opening think tag; Qwen's template supplies it instead."""
    def __init__(self, thinking_mode):
        super().__init__(thinking_mode)
        self._opening = "" if thinking_mode == "thinking" else None

    def feed(self, text):
        if self._opening is not None:
            self._opening += text
            if "<think>".startswith(self._opening):
                if self._opening == "<think>":
                    self._opening = None
                return ()
            text, self._opening = self._opening.removeprefix("<think>"), None
        return super().feed(text)

    def finish(self):
        if self._opening:
            self.failed = True
            return ()
        return super().finish()
