"""Verify and apply a digest-pinned, reviewed QSA candidate."""
import base64
import gzip
import hashlib
import subprocess
from pathlib import Path

parts = Path('.github/qsa-next-candidate')
payload = base64.b64decode(''.join((parts / f'part{i}.b64').read_text().strip() for i in range(3)), validate=True)
expected = '42a117608d98e23bc5ff7f3ed0216ee9385b6f47f7b7ff10ea1a8e486515c0ab'
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError('Candidate digest mismatch')
patch = gzip.decompress(payload)
subprocess.run(['git', 'apply', '--check', '-'], input=patch, check=True)
subprocess.run(['git', 'apply', '-'], input=patch, check=True)
print('Applied reviewed QSA candidate, compressed SHA256=' + expected)
