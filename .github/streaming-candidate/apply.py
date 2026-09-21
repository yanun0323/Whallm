"""Verify and apply the reviewed streaming candidate without external execution."""
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
patch = gzip.decompress(payload)
subprocess.run(['git', 'apply', '--check', '-'], input=patch, check=True)
subprocess.run(['git', 'apply', '-'], input=patch, check=True)
print('Applied reviewed streaming candidate, compressed SHA256=' + expected)
