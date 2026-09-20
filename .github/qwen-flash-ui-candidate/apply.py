"""Apply only the digest-pinned, reviewed Qwen App candidate and build fixes."""
import gzip
import hashlib
from pathlib import Path
import subprocess
import tempfile

root = Path(__file__).resolve().parent
payload = b''.join((root / f'part-{index:02}').read_bytes() for index in range(7))
expected = '6c1e7dff5a410135a0fa49e7af2f6099aa3d8b19af736f4fb0fdafc00b5bbca3'
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError('Candidate digest mismatch; refusing to apply')
patch = gzip.decompress(payload)
if len(patch) != 55886:
    raise RuntimeError('Unexpected candidate patch length')
fix = (root / 'build-fix.patch').read_bytes()
if hashlib.sha256(fix).hexdigest() != 'd226aae49c3737e29425b3a12927aa14e821d3bef2d86fbe75a449d04b106d67':
    raise RuntimeError('Build-fix digest mismatch; refusing to apply')
for content in (patch, fix):
    with tempfile.NamedTemporaryFile(suffix='.patch') as temporary:
        temporary.write(content)
        temporary.flush()
        subprocess.run(['git', 'apply', '--check', temporary.name], check=True)
        subprocess.run(['git', 'apply', temporary.name], check=True)
print(f'Applied reviewed Qwen App candidate and Swift type-checking fixes; compressed SHA256={expected}')
