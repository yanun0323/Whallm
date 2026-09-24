"""MiMo image preprocessing: pinned SGLang bilinear/patch equations, not PIL resize."""
from __future__ import annotations

from dataclasses import dataclass
import io
import math

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from ..media import ImagePart, MediaError, MAX_IMAGE_PIXELS


@dataclass(frozen=True)
class ImageInput:
    patches: np.ndarray
    grid: tuple[int, int, int]
    digest: str


def smart_resize(height, width, factor=32, min_pixels=8192, max_pixels=8388608):
    if min(height, width) <= 0 or max(height, width) / min(height, width) > 200:
        raise MediaError("Invalid image dimensions or aspect ratio.")
    if min(height, width) < factor:
        scale = factor / min(height, width)
        height, width = round(height * scale), round(width * scale)
    h, w = round(height / factor) * factor, round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h, w = math.floor(height / beta / factor) * factor, math.floor(width / beta / factor) * factor
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h, w = math.ceil(height * beta / factor) * factor, math.ceil(width * beta / factor) * factor
    if h <= 0 or w <= 0:
        raise MediaError("Image aspect ratio cannot fit the processor budget.")
    return h, w


def bilinear(image, height, width):
    """Equivalent to F.interpolate(..., bilinear, align_corners=False), no antialias."""
    h, w, _ = image.shape
    # Round the scale to F32, but round the multiply/add only once, matching
    # PyTorch's fused source-coordinate calculation rather than two NumPy ops.
    y = np.maximum((np.arange(height, dtype=np.float64) + .5) * float(np.float32(h / height)) - .5, 0).astype(np.float32)
    x = np.maximum((np.arange(width, dtype=np.float64) + .5) * float(np.float32(w / width)) - .5, 0).astype(np.float32)
    y0, x0 = y.astype(np.int32), x.astype(np.int32)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (y - y0).astype(np.float32)[:, None, None], (x - x0).astype(np.float32)[None, :, None]
    top = image[y0[:, None], x0[None, :]] * (1 - fx) + image[y0[:, None], x1[None, :]] * fx
    bottom = image[y1[:, None], x0[None, :]] * (1 - fx) + image[y1[:, None], x1[None, :]] * fx
    return top * (1 - fy) + bottom * fy


def image_input(part: ImagePart):
    try:
        with Image.open(io.BytesIO(part.data)) as image:
            expected = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}[part.mime]
            if image.format != expected:
                raise MediaError("Image bytes do not match the declared media type.")
            if getattr(image, "n_frames", 1) != 1:
                raise MediaError("Animated images are not supported; choose a still frame.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise MediaError("Image exceeds 1,048,576 decoded pixels; resize it before uploading.")
            image = ImageOps.exif_transpose(image)
            # Match the reference's RGB conversion. Do not invent alpha compositing.
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError) as error:
        if isinstance(error, MediaError):
            raise
        raise MediaError("Cannot decode this image.") from error
    height, width = smart_resize(*rgb.shape[:2])
    if height * width > MAX_IMAGE_PIXELS:
        raise MediaError("Resized image exceeds the processor pixel budget.")
    pixels = bilinear(rgb, height, width)
    mean = np.array([123.675, 116.28, 103.53], np.float32)
    std = np.array([58.395, 57.12, 57.375], np.float32)
    pixels = ((pixels - mean) / std).transpose(2, 0, 1)
    frames = np.stack((pixels, pixels))
    h, w = height // 16, width // 16
    patches = frames.reshape(1, 2, 3, h // 2, 2, 16, w // 2, 2, 16)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(h * w, -1)
    return ImageInput(np.ascontiguousarray(patches), (1, h, w), part.digest)
