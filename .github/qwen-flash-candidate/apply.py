"""Apply the reviewed, digest-pinned candidate; temporary transport only."""
import base64
import gzip
import hashlib
from pathlib import Path
import subprocess
import tempfile

EXPECTED = "bf4ad96f1c3c73a4789b258a85bce268300baa41d37c224274461b5a83c8a057"
ALLOWED = set("""Makefile
Scripts/benchmark_qwen_flash_host.py
Scripts/research_qwen_optimizations.py
docs/qwen-flash-optimization.md
docs/validation/qwen-flash-host-20260921.json
runtime/deepseek_v4_ssd/cli.py
runtime/deepseek_v4_ssd/model.py
runtime/deepseek_v4_ssd/model_manager.py
runtime/deepseek_v4_ssd/model_support/base.py
runtime/deepseek_v4_ssd/model_support/qwen.py
runtime/deepseek_v4_ssd/qwen4_exp.py
runtime/deepseek_v4_ssd/qwen_expert_waves.py
runtime/deepseek_v4_ssd/qwen_flash_config.py
runtime/deepseek_v4_ssd/qwen_flash_io.py
runtime/deepseek_v4_ssd/server.py
runtime/tests/portable/__init__.py
runtime/tests/portable/test_qwen_flash_portable.py
runtime/tests/test_qwen_flash_mlx.py""".splitlines())
root = Path(__file__).resolve().parent
payload = base64.b64decode("".join((root / f"part{i}.b64").read_text() for i in range(4)), validate=True)
if hashlib.sha256(payload).hexdigest() != EXPECTED:
    raise SystemExit("Candidate digest mismatch")
patch = gzip.decompress(payload)
with tempfile.NamedTemporaryFile(suffix=".patch") as stream:
    stream.write(patch)
    stream.flush()
    stats = subprocess.check_output(["git", "apply", "--numstat", stream.name], text=True)
    paths = {line.split("\t", 2)[2] for line in stats.splitlines()}
    if paths != ALLOWED:
        raise SystemExit(f"Unexpected candidate paths: {paths.symmetric_difference(ALLOWED)}")
    subprocess.run(["git", "apply", "--check", "--index", stream.name], check=True)
    subprocess.run(["git", "apply", "--index", stream.name], check=True)
    subprocess.run(["git", "diff", "--cached", "--check"], check=True)
print(f"Applied {len(ALLOWED)} reviewed files; compressed SHA256={EXPECTED}", flush=True)
