"""Prepare request-owned image embeddings and positions on the generation thread."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import secrets

import mlx.core as mx
import numpy as np

from ..cancellation import check_cancelled
from ..media import MediaError, ModalSpan, PreparedPrompt
from .artifact import load_encoder, validate_runtime
from .positions import ImageGenerationModel, image_positions
from .processor import ImageInput


@dataclass(frozen=True)
class ImagePrompt(PreparedPrompt):
    positions: np.ndarray

    def validate(self, hidden_size):
        super().validate(hidden_size)
        if (self.positions.shape != (3, len(self.token_ids))
                or not np.array_equal(self.positions, image_positions(len(self.token_ids), self.spans))):
            raise MediaError('Qwen image positions do not match the prompt.')


def generation_model(model, prepared):
    if not isinstance(prepared, ImagePrompt):
        raise MediaError('Qwen image input requires prepared spatial positions.')
    return ImageGenerationModel(model, prepared.positions)


def prepare(runtime, request):
    validate_runtime(runtime)
    messages, images, markers = [], [], []
    for original in request.messages:
        message = dict(original)
        if isinstance(message.get('content'), tuple):
            parts = []
            for part in message['content']:
                if isinstance(part, ImageInput):
                    marker = 'WHALLM_IMAGE_' + secrets.token_hex(24)
                    markers.append(marker)
                    images.append(part)
                    parts.append(marker)
                elif isinstance(part, str):
                    parts.append(part)
                else:
                    raise MediaError('Unprepared Qwen image input.')
            message['content'] = ''.join(parts)
        messages.append(message)
    prompt = runtime._codec.encode(messages, request.thinking_mode, list(request.tools),
                                   request.tool_choice, request.reasoning_effort)
    tokens, spans = [], []
    for marker, part in zip(markers, images):
        if prompt.count(marker) != 1:
            raise MediaError('Image placement was not preserved by the chat template.')
        before, prompt = prompt.split(marker, 1)
        tokens.extend(runtime.tokenizer.encode(before, add_special_tokens=False))
        count = part.grid[1] * part.grid[2] // 4
        tokens.append(248053)
        spans.append(ModalSpan(len(tokens), count, part.digest, part.grid))
        tokens.extend([248056] * count)
        tokens.append(248054)
    tokens.extend(runtime.tokenizer.encode(prompt, add_special_tokens=False))
    if not spans or len(tokens) >= runtime.installed.maximum_context:
        raise MediaError('Expanded image prompt exceeds the model context or has no images.')
    # Validate geometry and context before loading any image weights.
    positions = image_positions(len(tokens), spans)
    encoder = None
    try:
        check_cancelled()
        encoder = load_encoder(runtime.installed)
        text = runtime.model.model.embed_tokens(mx.array(tokens))
        segments, end = [], 0
        for span, part in zip(spans, images):
            check_cancelled()
            feature = encoder(mx.array(part.patches), part.grid)
            mx.eval(feature)
            check_cancelled()
            if feature.shape != (span.length, runtime.model.args.hidden_size) or not mx.all(mx.isfinite(feature)).item():
                raise MediaError('Qwen image encoder produced invalid embeddings.')
            segments.extend([text[end:span.start], feature.astype(text.dtype)])
            end = span.start + span.length
        segments.append(text[end:])
        embeddings = mx.concatenate(segments)
        mx.eval(embeddings)
        identity = hashlib.sha256(json.dumps({'version': 1, 'kind': runtime.installed.model_kind,
            'revision': runtime.installed.revision, 'tokens': tokens,
            'spans': [(s.start, s.length, s.digest, s.grid) for s in spans]}, sort_keys=True).encode()).hexdigest()
        result = ImagePrompt(tuple(tokens), embeddings, tuple(spans), identity, positions)
        result.validate(runtime.model.args.hidden_size)
        return result
    finally:
        # Encoder and weights are never cached on the text model. Drain failures
        # too, before a later request reuses shared GPU/expert resources.
        mx.synchronize()
        del encoder
        mx.clear_cache()
