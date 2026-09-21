"""Apply the reviewed, hash-pinned Qwen demand-prefill candidate."""
import base64
import gzip
import hashlib
import subprocess
from pathlib import Path

parts = Path('.github/qwen-demand-candidate')
payload = base64.b64decode(''.join((parts / f'part{i}.b64').read_text().strip() for i in range(11)), validate=True)
expected = '0dd1bc88af41c670cb918d6da49d09a6a7962afb931261af379d2c71a165cd1f'
if hashlib.sha256(payload).hexdigest() != expected:
    raise RuntimeError('Candidate digest mismatch')
patch = gzip.decompress(payload)
subprocess.run(['git', 'apply', '--check', '-'], input=patch, check=True)
subprocess.run(['git', 'apply', '-'], input=patch, check=True)
print('Applied Qwen demand-prefill candidate, compressed SHA256=' + expected)
