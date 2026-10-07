"""Bounded PNG/JPEG/WebP decoding using the pinned Transformers PIL processor."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import io
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..cancellation import check_cancelled
from ..media import ImagePart, MediaError, MAX_IMAGE_PIXELS, MAX_MEDIA_TOKENS


@dataclass(frozen=True)
class ImageInput:
    patches: np.ndarray
    grid: tuple[int, int, int]
    digest: str


def image_input(part):
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(part.data)) as image:
                if image.format != {'image/png': 'PNG', 'image/jpeg': 'JPEG', 'image/webp': 'WEBP'}[part.mime]:
                    raise MediaError('Image bytes do not match the declared media type.')
                if getattr(image, 'n_frames', 1) != 1:
                    raise MediaError('Animated images are not supported; choose a still frame.')
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise MediaError('Image exceeds 1,048,576 decoded pixels; resize it before uploading.')
                image = ImageOps.exif_transpose(image).convert('RGB')
                processor = Qwen2VLImageProcessorPil(size={'shortest_edge': 65536, 'longest_edge': MAX_IMAGE_PIXELS},
                    patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[.5]*3, image_std=[.5]*3)
                result = processor(images=image, return_tensors='np')
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as error:
        if isinstance(error, MediaError):
            raise
        raise MediaError('Cannot decode this image; use a PNG, JPEG or WebP with a less extreme aspect ratio.') from error
    grid = tuple(int(x) for x in result['image_grid_thw'][0])
    if grid[0] != 1 or grid[1] * grid[2] * 256 > MAX_IMAGE_PIXELS:
        raise MediaError('Resized image exceeds the processor pixel budget.')
    return ImageInput(np.ascontiguousarray(result['pixel_values']), grid, part.digest)


def validate_media(request):
    messages = copy.deepcopy(list(request.messages))
    tokens = 0
    for message in messages:
        if not isinstance(message.get('content'), tuple):
            continue
        parts = []
        for part in message['content']:
            check_cancelled()
            if isinstance(part, ImagePart):
                part = image_input(part)
                tokens += part.grid[1] * part.grid[2] // 4
            elif not isinstance(part, str):
                raise MediaError('Qwen accepts text and still images only.')
            if tokens > MAX_MEDIA_TOKENS:
                raise MediaError('Inputs exceed the 2048 media-token request limit.')
            parts.append(part)
        message['content'] = tuple(parts)
    return replace(request, messages=tuple(messages))
