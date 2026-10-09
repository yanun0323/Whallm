"""Installed image weights are pinned independently of the text manifest."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from ..cancellation import check_cancelled
from ..manifest import Tensor, _validate_tensors
from ..media import MediaError


@lru_cache(maxsize=1)
def catalog():
    bundled = Path(__file__).with_name('QwenVision.json')
    path = bundled if bundled.is_file() else Path(__file__).resolve().parents[3] / 'Sources/DeepSeekRepack/Resources/QwenVision.json'
    data = json.loads(path.read_text())
    if data['version'] != 1 or len(data['models']) != 2:
        raise ValueError('Invalid Qwen vision catalog')
    return data


def definition(installed):
    item = next((m for m in catalog()['models'] if m['kind'] == installed.model_kind), None)
    if (item is None or item['checkpointModelID'] != installed.model_id
            or item['checkpointRevision'] != installed.revision):
        raise MediaError('Image weights do not match this Qwen model.')
    return item


def weight_path(installed, *, verify=False):
    item = definition(installed)['file']
    root = installed.root.resolve()
    path = root / item['path']
    if (path.resolve() != path or not path.is_file() or path.stat().st_size != item['size']):
        raise MediaError('Qwen image weights are missing or damaged. Use Verify and Repair on the Model page.')
    if verify:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(8 * 1024**2):
                check_cancelled()
                digest.update(block)
        if digest.hexdigest() != item['sha256']:
            raise MediaError('Qwen image weights failed verification. Use Verify and Repair on the Model page.')
    return path


def validate_runtime(runtime):
    # Reject before encoding or sending the HTTP streaming response.
    weight_path(runtime.installed)


def load_encoder(installed):
    from ..model import _load_tensor_file
    from .encoder import VisionEncoder
    data = catalog()
    config = json.loads((installed.root / 'config.json').read_text())
    if (any(config.get('vision_config', {}).get(k) != v for k, v in data['visionConfig'].items())
            or config.get('image_token_id') != 248056 or config.get('vision_start_token_id') != 248053
            or config.get('vision_end_token_id') != 248054):
        raise MediaError('The installed Qwen image configuration does not match its pinned weights.')
    path = weight_path(installed, verify=True)
    tensors = tuple(Tensor(t['name'], t['dtype'], tuple(t['shape']), t['offset'], t['length']) for t in data['tensors'])
    _validate_tensors(tensors, path.stat().st_size, 'Qwen vision')
    encoder = VisionEncoder(config['vision_config'])
    weights = _load_tensor_file(path, tensors)
    encoder.load_weights([(name.removeprefix('model.visual.'), value) for name, value in weights.items()], strict=True)
    return encoder
