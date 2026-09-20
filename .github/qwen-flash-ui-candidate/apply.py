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
patches = [patch]
for filename, digest in [
    ('build-fix.patch', 'd226aae49c3737e29425b3a12927aa14e821d3bef2d86fbe75a449d04b106d67'),
    ('test-build-fix.patch', '854e3e98bbc7fcc19c905195cb4e20b71c9f9578f44d8b449fcd70465ceaa6fa'),
    ('test-maintenance.patch', '56d0fc310ef98c1e5fb247adc686c6d632370514a64798b9474116abf9f02d73'),
]:
    content = (root / filename).read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise RuntimeError(f'{filename} digest mismatch; refusing to apply')
    patches.append(content)
for content in patches:
    with tempfile.NamedTemporaryFile(suffix='.patch') as temporary:
        temporary.write(content)
        temporary.flush()
        subprocess.run(['git', 'apply', '--check', temporary.name], check=True)
        subprocess.run(['git', 'apply', temporary.name], check=True)
print(f'Applied reviewed Qwen App candidate and test maintenance; compressed SHA256={expected}')
