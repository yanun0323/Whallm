"""Complete-checkpoint two-image smoke and media Prompt Cache isolation check."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
from unittest.mock import patch

from PIL import Image
from deepseek_v4_ssd.media import ImagePart
from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
from deepseek_v4_ssd.manifest import InstalledModel
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.mimo.install import atomic_json, file_sha
from deepseek_v4_ssd.tool_codec import ToolChoice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = RuntimeConfig(slots=512, memory_limit_gib=28, prefill_step_size=32,
                           fp8_kv_cache=False, layer_major_prefill=False, prompt_cache_entries=2)
    runtime = ModelRuntime(InstalledModel.open(args.model), config)
    captured = []
    original = runtime.support.prepare_input
    def capture(owner, request):
        prepared = original(owner, request)
        captured.append((prepared.token_ids, prepared.input_identity))
        return prepared
    report = {"scope": "full-model image smoke, not comprehensive modality acceptance",
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "platform": platform.platform(), "python": platform.python_version(),
              "manifestSHA256": file_sha(args.model / "manifest.json"),
              "config": asdict(config), "cacheState": "new process; media KV cache disabled; OS cache uncontrolled", "results": []}
    try:
        with patch.object(runtime.support, "prepare_input", side_effect=capture):
            for color in ("red", "blue"):
                buffer = io.BytesIO()
                Image.new("RGB", (128, 128), color).save(buffer, format="PNG")
                image = ImagePart.from_bytes(buffer.getvalue(), "image/png")
                prompt = runtime.encode_chat([{"role": "user", "content": (
                    "Answer with one word. What is the dominant color in this image?", image)}])
                pieces = list(runtime.stream(prompt, GenerationOptions(max_tokens=8, temperature=0, seed=0)))
                text = "".join(p.text for p in pieces)
                tokens = [p.token for p in pieces]
                print(color, repr(text), flush=True)
                if color not in text.lower():
                    raise ValueError(f"image smoke failed for {color}: {text!r}")
                report["results"].append({"image": color, "imageSHA256": image.digest, "output": text,
                    "tokens": tokens, "outputTokenHash": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    "inputIdentity": captured[-1][1]})
        if captured[0][0] != captured[1][0] or captured[0][1] == captured[1][1] or runtime._prompt_caches:
            raise ValueError("media identity/cache isolation contract failed")
        report["sameTokenIDsDifferentMediaIdentity"] = True
        report["mediaPromptCachesStored"] = len(runtime._prompt_caches)
        tools = [{"type": "function", "function": {"name": "lookup_weather", "description": "Look up the temperature of a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
        messages = [{"role": "user", "content": "What is the temperature in Taipei? Call lookup_weather first."}]
        prompt = runtime.encode_chat(messages, tools=tools, tool_choice=ToolChoice("function", "lookup_weather"))
        chunks = list(runtime.stream(prompt, GenerationOptions(max_tokens=96, temperature=0, seed=0)))
        raw = "".join(p.text for p in chunks)
        turn = runtime.parse_chat(raw, "chat")
        if len(turn.tool_calls) != 1 or turn.tool_calls[0].name != "lookup_weather" or json.loads(turn.tool_calls[0].arguments).get("city") != "Taipei":
            raise ValueError(f"native tool calling failed: {raw!r}")
        parser = runtime.make_tool_stream_parser("chat")
        deltas = [delta for chunk in chunks for delta in parser.feed(chunk.text)] + list(parser.finish())
        if parser.failed or [d.tool_name for d in deltas if d.tool_name] != ["lookup_weather"]:
            raise ValueError("streaming and non-streaming tool parsing disagree")
        messages += [{"role": "assistant", "content": turn.content, "tool_calls": [{"id": "fixture-weather", "type": "function",
            "function": {"name": "lookup_weather", "arguments": turn.tool_calls[0].arguments}}]},
            {"role": "tool", "tool_call_id": "fixture-weather", "content": '{"city":"Taipei","temperature_c":23}'}]
        followup = list(runtime.stream(runtime.encode_chat(messages, tools=tools), GenerationOptions(max_tokens=48, temperature=0, seed=0)))
        answer = "".join(p.text for p in followup)
        print("tool call", raw, "tool result answer", answer, flush=True)
        if "23" not in answer:
            raise ValueError("tool result was not used in the next turn")
        report["toolRoundTrip"] = {"rawCall": raw, "answer": answer,
            "callTokenHash": hashlib.sha256(json.dumps([p.token for p in chunks]).encode()).hexdigest(),
            "answerTokenHash": hashlib.sha256(json.dumps([p.token for p in followup]).encode()).hexdigest()}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.report, report)
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
