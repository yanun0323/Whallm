"""Opt-in complete-artifact text smoke test (not multimodal/reference parity)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from deepseek_v4_ssd.generation import GenerationOptions, ModelRuntime
from deepseek_v4_ssd.manifest import InstalledModel
from deepseek_v4_ssd.model import RuntimeConfig
from deepseek_v4_ssd.mimo.install import atomic_json, file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config = RuntimeConfig(slots=512, memory_limit_gib=28, prefill_step_size=32,
                           fp8_kv_cache=False, layer_major_prefill=False, prompt_cache_entries=0)
    report = {"kind": "complete-artifact-text-smoke-not-numerical-parity", "platform": platform.platform(),
              "python": platform.python_version(), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "config": asdict(config), "cacheState": "fresh process; OS file cache uncontrolled",
              "manifestSHA256": file_sha(args.model / "manifest.json"), "results": []}
    print("opening complete installed model", flush=True)
    runtime = ModelRuntime(InstalledModel.open(args.model), config)
    print("all target-backbone parameters loaded", flush=True)
    try:
        for question in ("Reply with exactly: Hello.", "請只回答：你好。"):
            prompt = runtime.encode_chat([{"role": "user", "content": question}], "chat")
            tokens, text = [], ""
            started = time.monotonic()
            for piece in runtime.stream(prompt, GenerationOptions(max_tokens=24, temperature=0, seed=0)):
                tokens.append(piece.token)
                text += piece.text
                print(piece.text, end="", flush=True)
            print(flush=True)
            report["results"].append({"question": question, "prompt": prompt, "output": text, "tokens": tokens,
                                      "outputTokenHash": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                                      "seconds": time.monotonic() - started})
            args.report.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.report, report)
            if not text.strip():
                raise ValueError("complete model produced no text")
        pending = runtime.stream(runtime.encode_chat([{"role": "user", "content": "Count from 1 to 10."}]),
                                 GenerationOptions(max_tokens=24, temperature=0, seed=0))
        next(pending)
        pending.close()
        recovered = list(runtime.stream(runtime.encode_chat([{"role": "user", "content": "Say OK."}]),
                                        GenerationOptions(max_tokens=4, temperature=0, seed=0)))
        if not recovered:
            raise ValueError("generation did not recover after generator close")
        report["generatorCloseRecovery"] = {"tokens": [p.token for p in recovered], "output": "".join(p.text for p in recovered)}
        atomic_json(args.report, report)
    finally:
        runtime.close()
    print("text smoke and generator-close recovery complete", flush=True)


if __name__ == "__main__":
    main()
