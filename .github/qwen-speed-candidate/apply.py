"""Apply the checksum-pinned, reviewed Qwen speed experiment diff."""
import gzip
import hashlib
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
parts = Path(__file__).resolve().parent
packed = b''.join((parts / f'part{i}.gz').read_bytes() for i in range(1, 6))
if hashlib.sha256(packed).hexdigest() != 'e89ef5fb5e83d3ffb99cf5ceaaa6f5f753f21a9ba683d61c6b785ba0f7507816':
    raise RuntimeError('candidate checksum mismatch')
patch = gzip.decompress(packed)
if len(patch) != 48338:
    raise RuntimeError('candidate length mismatch')
allowed = {
    'Makefile', 'Scripts/benchmark_qwen_speed.py',
    'Sources/DeepSeekV4SSDApp/QwenOptimization.swift',
    *(f'Sources/DeepSeekV4SSDApp/Resources/{locale}.lproj/Localizable.strings' for locale in ('en', 'zh-Hans', 'zh-Hant')),
    'runtime/deepseek_v4_ssd/qwen4_exp.py',
    'runtime/deepseek_v4_ssd/qwen_ngram_hash.py',
    'runtime/deepseek_v4_ssd/qwen_tensor_ops.py',
    'runtime/tests/portable/test_qwen_speed_host.py',
    'runtime/tests/test_qwen_speed.py',
}
headers = [line.split() for line in patch.decode().splitlines() if line.startswith('diff --git ')]
if len(headers) != len(allowed) or {row[3][2:] for row in headers} != allowed or any(row[2][2:] != row[3][2:] for row in headers):
    raise RuntimeError('candidate paths mismatch')
subprocess.run(['git', 'apply', '--check', '-'], input=patch, cwd=root, check=True)
subprocess.run(['git', 'apply', '-'], input=patch, cwd=root, check=True)
print('Applied 11 reviewed source/test files; checksum verified.')
