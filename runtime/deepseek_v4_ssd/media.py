"""Bounded media ownership and ordered inputs; no network or filesystem URLs."""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import secrets
import tempfile
import threading
import time

from .cancellation import check_cancelled
from .documents import DOCUMENT_TYPES, MAX_DOCUMENTS
from .mimo.audio_processor import AUDIO_TYPES, MAX_AUDIOS

IMAGE_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
MAX_ASSET_BYTES = 8 * 1024**2
MAX_REQUEST_MEDIA_BYTES = 32 * 1024**2
MAX_IMAGES = 8
MAX_IMAGE_PIXELS = 1_048_576
MAX_MEDIA_TOKENS = 2048


class MediaError(ValueError):
    pass


@dataclass(frozen=True)
class ImagePart:
    data: bytes
    mime: str
    digest: str

    @classmethod
    def from_bytes(cls, data, mime):
        if mime not in IMAGE_TYPES or not data or len(data) > MAX_ASSET_BYTES:
            raise MediaError("Use a PNG, JPEG or WebP image of at most 8 MiB.")
        return cls(bytes(data), mime, hashlib.sha256(data).hexdigest())


@dataclass(frozen=True)
class DocumentPart:
    data: bytes
    mime: str
    digest: str

    @classmethod
    def from_bytes(cls, data, mime):
        if mime not in DOCUMENT_TYPES or not data or len(data) > MAX_ASSET_BYTES:
            raise MediaError("Use a PDF, DOCX, PPTX or XLSX file of at most 8 MiB.")
        return cls(bytes(data), mime, hashlib.sha256(data).hexdigest())


@dataclass(frozen=True)
class AudioPart:
    data: bytes
    mime: str
    digest: str

    @classmethod
    def from_bytes(cls, data, mime):
        if mime not in AUDIO_TYPES or not data or len(data) > MAX_ASSET_BYTES:
            raise MediaError("Use a PCM WAV file of at most 8 MiB.")
        return cls(bytes(data), mime, hashlib.sha256(data).hexdigest())


def asset_part(data, mime):
    kind = AudioPart if mime in AUDIO_TYPES else DocumentPart if mime in DOCUMENT_TYPES else ImagePart
    return kind.from_bytes(data, mime)


@dataclass(frozen=True)
class MediaRequest:
    messages: tuple[dict, ...]
    thinking_mode: str
    tools: tuple[dict, ...]
    tool_choice: object
    reasoning_effort: str


@dataclass(frozen=True)
class ModalSpan:
    start: int
    length: int
    digest: str
    grid: tuple[int, int, int]
    kind: str = "image"


@dataclass(frozen=True)
class PreparedPrompt:
    token_ids: tuple[int, ...]
    input_embeddings: object
    spans: tuple[ModalSpan, ...]
    input_identity: str

    def validate(self, hidden_size):
        if not self.token_ids or any(type(t) is not int or t < 0 for t in self.token_ids):
            raise MediaError("Invalid prepared prompt tokens.")
        if getattr(self.input_embeddings, "shape", None) != (len(self.token_ids), hidden_size):
            raise MediaError("Media embeddings do not align with prompt tokens.")
        end = 0
        for span in self.spans:
            if span.start < end or span.length < 1 or span.start + span.length > len(self.token_ids):
                raise MediaError("Invalid or overlapping media span.")
            end = span.start + span.length
        if not self.spans or sum(s.length for s in self.spans) > MAX_MEDIA_TOKENS:
            raise MediaError("Media token budget exceeded.")


class AssetStore:
    """Ephemeral per-server files. The authenticated API principal owns each ID."""
    def __init__(self, *, ttl=900, quota=64 * 1024**2, max_files=64, clock=time.monotonic):
        self._directory = tempfile.TemporaryDirectory(prefix="whallm-assets-")
        self.root = Path(self._directory.name)
        self.ttl, self.quota, self.max_files, self.clock = ttl, quota, max_files, clock
        self._items = {}
        self._lock = threading.Lock()
        self._uploads = threading.BoundedSemaphore(4)

    def _expire(self):
        for key, item in list(self._items.items()):
            if item["expires"] <= self.clock():
                (self.root / key).unlink(missing_ok=True)
                del self._items[key]

    def put(self, stream, length, mime, owner):
        if type(length) is not int or not 0 < length <= MAX_ASSET_BYTES or mime not in IMAGE_TYPES | DOCUMENT_TYPES | AUDIO_TYPES:
            raise MediaError("Upload a supported image, audio or document of at most 8 MiB.")
        if not self._uploads.acquire(blocking=False):
            raise MediaError("Too many concurrent uploads; try again later.")
        try:
            data = bytearray()
            deadline = time.monotonic() + 30
            while len(data) < length:
                check_cancelled()
                if time.monotonic() > deadline:
                    raise MediaError("Media upload timed out.")
                chunk = stream.read(min(64 * 1024, length - len(data)))
                if not chunk:
                    raise MediaError("Media upload was truncated.")
                data.extend(chunk)
            part = asset_part(data, mime)
            with self._lock:
                self._expire()
                if len(self._items) >= self.max_files or sum(i["size"] for i in self._items.values()) + length > self.quota:
                    raise MediaError("Media storage quota exceeded; delete unused assets.")
                key = "file-" + secrets.token_hex(24)
                path = self.root / key
                with path.open("xb") as file:
                    path.chmod(0o600)
                    file.write(data)
                self._items[key] = {"owner": owner, "size": length, "mime": mime,
                                    "digest": part.digest, "expires": self.clock() + self.ttl}
                return {"id": key, "object": "file", "bytes": length, "mime_type": mime,
                        "expires_in": self.ttl, "sha256": part.digest}
        finally:
            self._uploads.release()

    def get(self, key, owner):
        with self._lock:
            self._expire()
            item = self._items.get(key) if isinstance(key, str) else None
            if item is None or item["owner"] != owner:
                raise MediaError("Media file does not exist or has expired.")
            part = asset_part((self.root / key).read_bytes(), item["mime"])
            if part.digest != item["digest"]:
                raise MediaError("Stored media failed integrity validation.")
            return part

    def delete(self, key, owner):
        with self._lock:
            self._expire()
            item = self._items.get(key)
            if item is None or item["owner"] != owner:
                raise MediaError("Media file does not exist or has expired.")
            (self.root / key).unlink(missing_ok=True)
            del self._items[key]

    def close(self):
        with self._lock:
            self._items.clear()
            self._directory.cleanup()


def ordered_content(value, *, asset_store=None, owner=""):
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise MediaError("Message content must be text or a non-empty array of parts.")
    parts = []
    for item in value:
        if not isinstance(item, dict):
            raise MediaError("Each content part must be an object.")
        kind = item.get("type")
        if kind in ("text", "input_text", "output_text") and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif kind in ("image", "image_url", "input_image"):
            if item.get("file_id") is not None:
                if asset_store is None:
                    raise MediaError("File IDs require the asset endpoint.")
                part = asset_store.get(item["file_id"], owner)
                if not isinstance(part, ImagePart):
                    raise MediaError("An image part must reference an image asset.")
                parts.append(part)
                continue
            url = item.get("image_url", item.get("url"))
            if isinstance(url, dict):
                if url.get("detail", "auto") != "auto":
                    raise MediaError("Only automatic image detail is supported.")
                url = url.get("url")
            if not isinstance(url, str) or not url.startswith("data:") or ";base64," not in url:
                raise MediaError("Use an uploaded file_id or inline image data URL; remote and filesystem URLs are disabled.")
            header, encoded = url.split(";base64,", 1)
            if len(encoded) > (MAX_ASSET_BYTES + 2) // 3 * 4:
                raise MediaError("Inline media exceeds the byte limit.")
            try:
                data = base64.b64decode(encoded, validate=True)
            except ValueError as error:
                raise MediaError("Invalid base64 image.") from error
            parts.append(ImagePart.from_bytes(data, header[5:]))
        elif kind in ("audio", "input_audio"):
            if item.get("file_id") is not None:
                if asset_store is None:
                    raise MediaError("File IDs require the asset endpoint.")
                part = asset_store.get(item["file_id"], owner)
                if not isinstance(part, AudioPart):
                    raise MediaError("An audio part must reference an audio asset.")
            else:
                audio = item.get("input_audio")
                if not isinstance(audio, dict) or audio.get("format") != "wav" or not isinstance(audio.get("data"), str):
                    raise MediaError("Use an uploaded audio file_id or base64 input_audio with format wav; URLs are disabled.")
                if len(audio["data"]) > (MAX_ASSET_BYTES + 2) // 3 * 4:
                    raise MediaError("Inline audio exceeds the byte limit.")
                try:
                    part = AudioPart.from_bytes(base64.b64decode(audio["data"], validate=True), "audio/wav")
                except ValueError as error:
                    raise MediaError("Invalid base64 WAV audio.") from error
            parts.append(part)
        elif kind in ("file", "input_file"):
            reference = item.get("file", item)
            if not isinstance(reference, dict) or not isinstance(reference.get("file_id"), str) or asset_store is None:
                raise MediaError("Documents require an uploaded file_id; URLs and inline files are disabled.")
            part = asset_store.get(reference["file_id"], owner)
            if not isinstance(part, DocumentPart):
                raise MediaError("A document part must reference a document asset.")
            parts.append(part)
        else:
            raise MediaError("Unsupported content part; use text, a supported image or audio, or an uploaded document.")
    assets = [p for p in parts if isinstance(p, (ImagePart, DocumentPart, AudioPart))]
    if (sum(isinstance(p, ImagePart) for p in assets) > MAX_IMAGES
            or sum(isinstance(p, DocumentPart) for p in assets) > MAX_DOCUMENTS
            or sum(isinstance(p, AudioPart) for p in assets) > MAX_AUDIOS
            or sum(len(p.data) for p in assets) > MAX_REQUEST_MEDIA_BYTES):
        raise MediaError("Message media budget exceeded.")
    return tuple(parts) if assets else "".join(parts)


def expand_documents(messages, *, enabled):
    from .documents import extract_document, MAX_TEXT_BYTES
    assets = [p for m in messages if isinstance(m.get("content"), tuple) for p in m["content"]
              if isinstance(p, (ImagePart, DocumentPart, AudioPart))]
    documents = [p for p in assets if isinstance(p, DocumentPart)]
    if not documents:
        return messages
    if not enabled:
        raise MediaError("This model does not support document inputs.")
    if len(documents) > MAX_DOCUMENTS or sum(len(p.data) for p in assets) > MAX_REQUEST_MEDIA_BYTES:
        raise MediaError("Request document or asset budget exceeded.")
    for message in messages:
        if isinstance(message.get("content"), tuple) and message.get("role") not in ("user", "tool"):
            raise MediaError("Documents are supported only in user messages and tool results.")
    result, text_bytes, image_count, image_bytes = [], 0, 0, 0
    for original in messages:
        message = dict(original)
        if isinstance(message.get("content"), tuple):
            parts = []
            for part in message["content"]:
                check_cancelled()
                if isinstance(part, DocumentPart):
                    expanded = extract_document(part.data, part.mime)
                    text_bytes += sum(len(p.encode()) for p in expanded if isinstance(p, str))
                    if text_bytes > MAX_TEXT_BYTES:
                        raise MediaError("Request document text exceeds 64 KiB.")
                    parts.extend(expanded)
                else:
                    parts.append(part)
            images = [p for p in parts if isinstance(p, ImagePart)]
            media = [p for p in parts if isinstance(p, (ImagePart, AudioPart))]
            image_count += len(images)
            image_bytes += sum(len(p.data) for p in media)
            if image_count > MAX_IMAGES or image_bytes > MAX_REQUEST_MEDIA_BYTES:
                raise MediaError("Expanded document media exceed the request budget.")
            message["content"] = tuple(parts) if media else "".join(parts)
        result.append(message)
    return result


def media_request(messages, thinking_mode, tools, tool_choice, reasoning_effort):
    assets = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, tuple):
            if message.get("role") not in ("user", "tool"):
                raise MediaError("Media is supported only in user messages and tool results.")
            assets.extend(p for p in content if isinstance(p, (ImagePart, AudioPart)))
    if not assets:
        return None
    if (sum(isinstance(p, ImagePart) for p in assets) > MAX_IMAGES
            or sum(isinstance(p, AudioPart) for p in assets) > MAX_AUDIOS
            or sum(len(p.data) for p in assets) > MAX_REQUEST_MEDIA_BYTES):
        raise MediaError("Request media budget exceeded.")
    return MediaRequest(tuple(messages), thinking_mode, tuple(tools or ()), tool_choice, reasoning_effort)
