"""Apply the reviewed singleton guard after the measured multi-token regression."""
import hashlib
from pathlib import Path
import subprocess
root = Path(__file__).resolve().parents[2]
patch = Path(__file__).with_name('refinement.patch').read_bytes()
if hashlib.sha256(patch).hexdigest() != '4133def99b68fb371498d3fbfec2a952ace9afff1c67563164d529811dd7d3b0':
    raise RuntimeError('refinement checksum mismatch')
subprocess.run(['git', 'apply', '--check', '-'], input=patch, cwd=root, check=True)
subprocess.run(['git', 'apply', '-'], input=patch, cwd=root, check=True)
print('Applied measured multi-token fallback; checksum verified.')
