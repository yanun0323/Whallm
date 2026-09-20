"""Apply only the digest-pinned, reviewed Qwen App candidate."""
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
with tempfile.NamedTemporaryFile(suffix='.patch') as temporary:
    temporary.write(patch)
    temporary.flush()
    subprocess.run(['git', 'apply', '--check', temporary.name], check=True)
    subprocess.run(['git', 'apply', temporary.name], check=True)
print(f'Applied reviewed Qwen App candidate; compressed SHA256={expected}')
