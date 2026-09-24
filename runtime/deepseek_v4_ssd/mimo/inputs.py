"""CPU validation before streaming; GPU preparation belongs to the generation owner."""
from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
import secrets

from ..media import (ImagePart, AudioPart, MediaError, MediaRequest, ModalSpan, PreparedPrompt,
                     MAX_MEDIA_TOKENS)
from .processor import ImageInput, image_input
from .audio_processor import AudioInput, audio_input


def validate_media(request: MediaRequest):
    messages = copy.deepcopy(list(request.messages))
    tokens = 0
    for message in messages:
        if not isinstance(message.get("content"), tuple):
            continue
        parts = []
        for part in message["content"]:
            if isinstance(part, ImagePart):
                part = image_input(part)
                tokens += part.grid[0] * part.grid[1] * part.grid[2] // 4
            elif isinstance(part, AudioPart):
                part = audio_input(part)
                tokens += part.token_count
            if tokens > MAX_MEDIA_TOKENS:
                raise MediaError("Inputs exceed the 2048 media-token request limit.")
            parts.append(part)
        message["content"] = tuple(parts)
    return replace(request, messages=tuple(messages))


def prepare(runtime, request: MediaRequest):
    import mlx.core as mx
    from ..manifest import Tensor
    from ..model import _load_tensor_file
    from .vision import VisionEncoder
    from .audio import load_audio
    from ..cancellation import check_cancelled
    root = runtime.installed.root
    config = json.loads((root / "config.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    # No unbounded media/encoder cache: request owns weights and embeddings.
    # Audio-only inputs must not load the ViT (and vice versa).
    vision, audio_tokenizer, audio_patch = None, None, None
    messages, media, markers = [], [], []
    for original in request.messages:
        message = dict(original)
        if isinstance(message.get("content"), tuple):
            content = []
            for part in message["content"]:
                if isinstance(part, (ImageInput, AudioInput)):
                    marker = "WHALLM_MEDIA_" + secrets.token_hex(24)
                    markers.append(marker)
                    media.append(part)
                    content.append(marker)
                elif isinstance(part, str):
                    content.append(part)
                else:
                    raise MediaError("Unprepared media content.")
            message["content"] = "".join(content)
        messages.append(message)
    prompt = runtime._codec.encode(messages, request.thinking_mode, list(request.tools),
                                   request.tool_choice, request.reasoning_effort)
    tokens, spans, features = [], [], []
    for marker, part in zip(markers, media):
        check_cancelled()
        if prompt.count(marker) != 1:
            raise MediaError("Media placement was not preserved by the chat template.")
        before, prompt = prompt.split(marker, 1)
        tokens.extend(runtime.tokenizer.encode(before, add_special_tokens=False))
        pc = config["processor_config"]
        if isinstance(part, ImageInput):
            if vision is None:
                vision = VisionEncoder(config["vision_config"])
                table = manifest["components"]["vision"]
                tensors = tuple(Tensor(t["name"], t["dtype"], tuple(t["shape"]), t["offset"], t["length"]) for t in table["tensors"])
                weights = _load_tensor_file(root / table["file"], tensors)
                vision.load_weights([(name.removeprefix("visual."), value) for name, value in weights.items()], strict=True)
                del weights
            feature = vision(mx.array(part.patches), part.grid)
            count = part.grid[0] * part.grid[1] * part.grid[2] // 4
            grid, kind = part.grid, "image"
            start, placeholder, end = pc["vision_start_token_id"], pc["image_token_id"], pc["vision_end_token_id"]
        else:
            if audio_tokenizer is None:
                audio_tokenizer, audio_patch = load_audio(root, manifest, config)
            codes = audio_tokenizer(mx.array(part.mel))
            feature = audio_patch(codes)
            count = part.token_count
            grid, kind = (part.samples, part.mel.shape[0], 20), "audio"
            start, placeholder, end = pc["audio_start_token_id"], pc["audio_token_id"], pc["audio_end_token_id"]
        mx.eval(feature)
        check_cancelled()
        if feature.shape != (count, runtime.model.args.hidden_size) or not mx.all(mx.isfinite(feature)).item():
            raise MediaError("Media encoder produced invalid embeddings.")
        tokens.append(start)
        spans.append(ModalSpan(len(tokens), count, part.digest, grid, kind))
        features.append(feature)
        tokens.extend([placeholder] * count)
        tokens.append(end)
    tokens.extend(runtime.tokenizer.encode(prompt, add_special_tokens=False))
    if len(tokens) >= runtime.installed.maximum_context:
        raise MediaError("Expanded media prompt exceeds the model context.")
    text_embeddings = runtime.model.model.embed_tokens(mx.array(tokens))
    segments, end = [], 0
    for span, feature in zip(spans, features):
        segments.extend((text_embeddings[end:span.start], feature.astype(text_embeddings.dtype)))
        end = span.start + span.length
    segments.append(text_embeddings[end:])
    embeddings = mx.concatenate(segments)
    mx.eval(embeddings)
    identity = hashlib.sha256(json.dumps({"version": 2, "revision": runtime.installed.revision,
        "processor": config["processor_config"], "tokens": tokens,
        "spans": [(s.start, s.length, s.digest, s.grid, s.kind) for s in spans]}, sort_keys=True).encode()).hexdigest()
    result = PreparedPrompt(tuple(tokens), embeddings, tuple(spans), identity)
    result.validate(runtime.model.args.hidden_size)
    del vision, audio_tokenizer, audio_patch
    mx.clear_cache()
    return result
