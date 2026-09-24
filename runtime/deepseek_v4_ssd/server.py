from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import select
import socket
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import urlsplit

from . import throughput
from .app_memory import AppMemorySampler
from .status_memory import StatusMemorySampler
from .model import _apply_prompt_cache_mode

from .cancellation import GenerationCancelled, cancellation_scope, check_cancelled

from .generation import (
    APPROXIMATION_MODES,
    EXACT_APPROXIMATION_MODE,
    LEARNED_ROUTE_DROP_LOWEST_1,
    GenerationOptions,
    GeneratedPiece,
    THINK_END,
)
from .io_metrics import EXPERT_FILE_CACHE_POLICIES
from .manifest import InstalledModel
from .media import (AssetStore, MediaError, ordered_content, IMAGE_TYPES, MAX_ASSET_BYTES,
                    MAX_REQUEST_MEDIA_BYTES, MAX_IMAGES, MAX_IMAGE_PIXELS, MAX_MEDIA_TOKENS)
from .model_support import get_support, support_for_installed, support_for_runtime
from .model_manager import (
    MODEL_IDS,
    ModelCatalogError,
    ModelDefaults,
    ModelLoadFailed,
    ModelManager,
    ModelNotFound,
    ModelRequest,
    load_model_catalog,
    parse_model_catalog,
    validate_runtime_config,
)
from .model import RuntimeConfig, _POWER_SAVING_LIMITS_GBPS
from .tool_codec import ToolChoice, ToolStreamDelta, ToolStreamParser

RESPONSE_HEARTBEAT_SECONDS = 10.0
MAX_REQUEST_BYTES = 1_048_576
MAX_GENERATION_TOKENS = 272_000


class APIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status: int = 400,
        param: str | None = None,
        code: str | None = None,
        error_type: str = "invalid_request_error",
    ):
        super().__init__(message)
        self.status = status
        self.param = param
        self.code = code
        self.error_type = error_type

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": str(self),
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


@dataclass(frozen=True)
class ServerDefaults:
    max_tokens: int = 272_000
    temperature: float = 0.2
    top_p: float = 0.98
    top_k: int = 0


class GenerationMetrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generating = False
        self._tokens = 0
        self._tokens_per_second = 0.0
        self._first_token_at = 0.0

    def start(self) -> None:
        with self._lock:
            self._generating = True
            self._tokens = 0
            self._tokens_per_second = 0.0
            self._first_token_at = 0.0

    def record(self, tokens: int) -> None:
        now = time.perf_counter()
        with self._lock:
            self._tokens = tokens
            if tokens == 1 or self._first_token_at == 0:
                self._first_token_at = now
                return
            elapsed = now - self._first_token_at
            if elapsed > 0:
                self._tokens_per_second = (tokens - 1) / elapsed

    def finish(self) -> None:
        with self._lock:
            self._generating = False

    def snapshot(self) -> dict[str, bool | int | float]:
        with self._lock:
            return {
                "generating": self._generating,
                "generation_tokens": self._tokens,
                "tokens_per_second": self._tokens_per_second,
            }


class OpenAIServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 32

    def __init__(
        self,
        address: tuple[str, int],
        model_manager: ModelManager,
        *,
        api_key: str | None = None,
        log_level: str = "info",
    ):
        self.model_manager = model_manager
        self.api_key = api_key
        self.log_level = log_level
        self.metrics = GenerationMetrics()
        self.status_memory = None
        self.assets = None
        super().__init__(address, OpenAIHandler)
        try:
            self.assets = AssetStore()
            self.status_memory = StatusMemorySampler()
            self.status_memory.start()
        except BaseException:
            self.server_close()
            raise

    def server_close(self):
        if self.assets is not None:
            self.assets.close()
        if self.status_memory is not None:
            self.status_memory.close()
        super().server_close()

    def track(self, pieces: Iterator[GeneratedPiece]) -> Iterator[GeneratedPiece]:
        self.metrics.start()
        try:
            for piece in pieces:
                check_cancelled()
                self.metrics.record(piece.generation_tokens)
                yield piece
        finally:
            try:
                close = getattr(pieces, "close", None)
                if close is not None:
                    close()
            finally:
                self.metrics.finish()


class ReasoningParser:
    def __init__(self, enabled: bool):
        self.reasoning = enabled
        self.pending = ""

    def feed(self, text: str) -> tuple[str, str]:
        if not self.reasoning:
            return "", text
        combined = self.pending + text
        if THINK_END in combined:
            reasoning, content = combined.split(THINK_END, 1)
            self.pending = ""
            self.reasoning = False
            return reasoning, content
        keep = min(len(THINK_END) - 1, len(combined))
        if keep == len(combined):
            self.pending = combined
            return "", ""
        self.pending = combined[-keep:] if keep else ""
        return combined[:-keep] if keep else combined, ""

    def finish(self) -> tuple[str, str]:
        pending, self.pending = self.pending, ""
        return (pending, "") if self.reasoning else ("", pending)


class ClientConnection:
    """Observe EOF/reset independently of SSE output, including queued requests."""

    def __init__(self, connection):
        self.connection = connection
        self.cancelled = threading.Event()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._observe, daemon=True,
                                       name="whallm-client-disconnect")

    def _observe(self):
        while not self.stopped.wait(0.05):
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                if readable and not self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                    self.cancelled.set()
                    return
            except BlockingIOError:
                continue
            except OSError:
                self.cancelled.set()
                return

    def __enter__(self):
        self.scope = cancellation_scope(self.cancelled)
        self.scope.__enter__()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stopped.set()
        self.thread.join()
        self.scope.__exit__(*args)


class ResponseEvents:
    """Keep Responses SSE alive without moving MLX work to another thread."""

    def __init__(self, handler, response: dict[str, Any]):
        self.handler = handler
        self.response = response
        self.sequence = 0
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.error: OSError | None = None
        self.last_sent = time.monotonic()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True,
                                       name="whallm-response-heartbeat")

    def check_connection(self) -> None:
        if self.error is not None:
            raise self.error

    def send(self, event_type: str, **values: Any) -> None:
        with self.lock:
            self.check_connection()
            if self.stopped.is_set():
                return
            try:
                self.handler._sse({"type": event_type, "sequence_number": self.sequence, **values})
            except OSError as error:
                self.error = error
                self.stopped.set()
                raise
            self.sequence += 1
            self.last_sent = time.monotonic()
            if event_type in {"response.completed", "response.failed", "response.incomplete"}:
                self.stopped.set()

    def _heartbeat(self) -> None:
        while not self.stopped.wait(RESPONSE_HEARTBEAT_SECONDS):
            try:
                with self.lock:
                    if time.monotonic() - self.last_sent >= RESPONSE_HEARTBEAT_SECONDS:
                        # An SSE comment does not reach every client's event idle timer.
                        self.send("response.in_progress", response=self.response)
            except OSError:
                return

    def __enter__(self):
        self.send("response.created", response=self.response)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        self.thread.join()


class OpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "Whallm/1"

    @property
    def app(self) -> OpenAIServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:
        self._run(self._get)

    def do_POST(self) -> None:
        self._run(self._post)

    def do_PUT(self) -> None:
        self._run(self._put)

    def do_DELETE(self) -> None:
        self._run(self._delete)

    def _run(self, action) -> None:
        try:
            action()
        except APIError as error:
            self._json(error.status, error.body())
        except MediaError as error:
            self._json(400, APIError(str(error), param="content", code="invalid_media").body())
        except ModelCatalogError as error:
            self._json(
                400,
                APIError(str(error), param="configuration").body(),
            )
        except ModelNotFound as error:
            self._json(
                400,
                APIError(
                    f"The model '{error}' does not exist.",
                    param="model",
                    code="model_not_found",
                ).body(),
            )
        except ModelLoadFailed as error:
            sys.stderr.write(f"model load failed for {error.model}: {error.cause}\n")
            self._json(
                500,
                APIError(
                    f"Unable to load model '{error.model}'. Check the server log and try again.",
                    status=500,
                    param="model",
                    code="model_load_failed",
                    error_type="server_error",
                ).body(),
            )
        except (GenerationCancelled, BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as error:
            sys.stderr.write(f"request failed: {error}\n")
            if self.close_connection:
                return
            self._json(
                500,
                APIError(
                    "The server could not complete the request.",
                    status=500,
                    code="internal_error",
                    error_type="server_error",
                ).body(),
            )

    def _get(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self._json(
                200,
                {
                    "name": "Whallm",
                    "status": "ok",
                    "api_base": "/v1",
                },
            )
        elif path == "/favicon.ico":
            self._bytes(204, b"", "image/x-icon")
        elif path == "/healthz":
            self._json(200, {"status": "ok"})
        elif path == "/api/status":
            self._authorize()
            self._json(200, self._status())
        elif path == "/api/capabilities":
            self._authorize()
            from .model_support.catalog import DESCRIPTORS
            from .documents import DOCUMENT_TYPES, MAX_DOCUMENTS, MAX_PAGES, MAX_TEXT_BYTES
            from .mimo.audio_processor import AUDIO_TYPES, MAX_AUDIOS, MAX_AUDIO_SECONDS, SAMPLE_RATE
            self._json(200, {"version": 1, "models": {
                d.api_model_id: {"input": ["text"] + (["image"] if d.supports("imageInput") else [])
                    + (["document"] if d.supports("documentInput") else [])
                    + (["audio"] if d.supports("audioInput") else [])}
                for d in DESCRIPTORS}, "media": {"image_mime_types": sorted(IMAGE_TYPES),
                "document_mime_types": sorted(DOCUMENT_TYPES), "max_documents": MAX_DOCUMENTS,
                "audio_mime_types": sorted(AUDIO_TYPES), "max_audios": MAX_AUDIOS,
                "max_audio_seconds": MAX_AUDIO_SECONDS, "audio_sample_rate": SAMPLE_RATE,
                "audio_pcm_bits": 16, "audio_channels": [1, 2],
                "max_document_pages_slides_sheets": MAX_PAGES, "max_document_text_bytes": MAX_TEXT_BYTES,
                "max_asset_bytes": MAX_ASSET_BYTES, "max_request_bytes": MAX_REQUEST_MEDIA_BYTES,
                "max_images": MAX_IMAGES, "max_decoded_pixels": MAX_IMAGE_PIXELS,
                "max_media_tokens": MAX_MEDIA_TOKENS, "asset_ttl_seconds": 900,
                "remote_urls": False, "upload_path": "/api/assets"}})
        elif path == "/v1/models":
            self._authorize()
            self._json(
                200,
                {
                    "object": "list",
                    "data": self.app.model_manager.models(),
                    "models": self.app.model_manager.codex_models(),
                },
            )
        else:
            raise APIError("Route not found.", status=404, code="not_found")

    def _post(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/assets":
            self._upload_asset()
        elif path == "/v1/chat/completions":
            self._authorize()
            payload = self._request_json()
            with ClientConnection(self.connection), self.app.status_memory.activity(), self.app.model_manager.request(payload.get("model")) as model:
                self._chat(payload, model)
        elif path == "/v1/responses":
            self._authorize()
            payload = self._request_json()
            with ClientConnection(self.connection), self.app.status_memory.activity(), self.app.model_manager.request(payload.get("model")) as model:
                self._responses(payload, model)
        elif path == "/v1/completions":
            self._authorize()
            payload = self._request_json()
            with ClientConnection(self.connection), self.app.status_memory.activity(), self.app.model_manager.request(payload.get("model")) as model:
                self._completion(payload, model)
        elif path == "/api/benchmark/throughput":
            self._authorize()
            payload = self._request_json()
            self._throughput(payload)
        elif path == "/api/models/load":
            self._authorize()
            payload = self._request_json()
            with self.app.status_memory.activity():
                self.app.model_manager.load(
                    payload.get("model"), payload.get("configuration")
                )
            self._json(200, self._status())
        elif path == "/api/models/configure":
            self._authorize()
            payload = self._request_json()
            with self.app.status_memory.activity():
                self.app.model_manager.configure(payload.get("configuration"))
            self._json(200, self._status())
        elif path == "/api/models/unload":
            self._authorize()
            payload = self._request_json()
            with self.app.status_memory.activity():
                self.app.model_manager.unload(payload.get("model"))
            self._json(200, self._status())
        elif path == "/api/status/memory/reset":
            self._authorize()
            self._request_json()
            self.app.status_memory.reset()
            self._json(200, self._status())
        else:
            raise APIError("Route not found.", status=404, code="not_found")

    def _throughput(self, payload: dict[str, Any]) -> None:
        context = payload.get("context_length")
        generation = payload.get("generation_length")
        benchmark_context = payload.get("benchmark_context", "code")
        if benchmark_context not in throughput.CONTEXT_TYPES:
            raise APIError("Choose Code or Novel as the benchmark context.", param="benchmark_context")
        if type(context) is not int or context not in throughput.CONTEXT_LENGTHS:
            raise APIError("Choose a supported context length.", param="context_length")
        if type(generation) is not int or generation not in throughput.GENERATION_LENGTHS:
            raise APIError("Choose 128, 512, 1024, or 4096 output tokens.", param="generation_length")
        self._start_sse()
        try:
            with ClientConnection(self.connection), self.app.status_memory.activity(), AppMemorySampler() as memory:
                self._sse({"phase": "loading"})
                with self.app.model_manager.request(payload.get("model")) as model:
                    runtime = model.runtime
                    support = support_for_runtime(runtime)
                    options, _ = self._common(
                        {"max_tokens": generation, "temperature": throughput.TEMPERATURE,
                         "seed": throughput.SEED}, model.defaults, support=support,
                        dspark=bool(getattr(runtime.config, "dspark_enabled", False)),
                    )
                    _validate_approximation_runtime(options, runtime)
                    self._sse({"phase": "running", "generated": 0})
                    result = throughput.run_trial(
                        runtime, options, context, self.app.track,
                        lambda count: self._sse({"phase": "running", "generated": count}),
                        benchmark_context=benchmark_context,
                    )
                    self._sse({"result": {
                        **result, "model": model.model_id,
                        "peak_app_memory_bytes": memory.finish(),
                        "memory_scope": memory.scope,
                    }})
        except (GenerationCancelled, BrokenPipeError, ConnectionResetError):
            self.close_connection = True
            return
        except Exception as error:
            self._sse({"error": {"message": str(error)}})
        self._sse_done()

    def _put(self) -> None:
        raise APIError("Route not found.", status=404, code="not_found")

    def _asset_owner(self):
        return hashlib.sha256((self.app.api_key or "anonymous").encode()).hexdigest()

    def _content(self, value, param):
        try:
            return ordered_content(value, asset_store=self.app.assets, owner=self._asset_owner())
        except MediaError as error:
            raise APIError(str(error), param=param, code="invalid_media") from error

    def _upload_asset(self):
        self._authorize()
        self.close_connection = True
        lengths = self.headers.get_all("Content-Length", [])
        if (len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdigit()
                or self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding")):
            raise APIError("Upload requires one Content-Length and an uncompressed body.", param="file")
        self.connection.settimeout(30)
        result = self.app.assets.put(self.rfile, int(lengths[0]), self.headers.get_content_type(), self._asset_owner())
        self._json(201, result)

    def _delete(self):
        self._authorize()
        path = urlsplit(self.path).path
        if not path.startswith("/api/assets/"):
            raise APIError("Route not found.", status=404, code="not_found")
        key = path.removeprefix("/api/assets/")
        self.app.assets.delete(key, self._asset_owner())
        self._json(200, {"id": key, "deleted": True})

    def _chat(self, payload: dict[str, Any], model: ModelRequest) -> None:
        runtime = model.runtime
        tools, tool_choice = _tool_request(payload)
        support = support_for_runtime(runtime)
        messages = _messages(payload.get("messages"), content_parser=(
            self._content if any(support.descriptor.supports(f) for f in ("imageInput", "documentInput", "audioInput")) else None))
        thinking_mode, reasoning_effort = _reasoning_settings(
            payload,
            support=support,
        )
        options, stream = self._common(
            payload,
            model.defaults,
            support=support,
            dspark=bool(getattr(runtime.config, "dspark_enabled", False)),
            thinking_mode=thinking_mode,
        )
        _validate_approximation_runtime(options, runtime)
        prompt = runtime.encode_chat(
            messages,
            thinking_mode,
            tools,
            tool_choice,
            reasoning_effort,
        )
        request_id = "chatcmpl-" + uuid.uuid4().hex
        pieces = self.app.track(runtime.stream(prompt, options))
        tool_calling = bool(tools) and tool_choice.mode != "none"
        if stream:
            stream_options = payload.get("stream_options") or {}
            if tool_calling:
                self._stream_tool_chat(
                    pieces,
                    request_id,
                    thinking_mode,
                    bool(stream_options.get("include_usage", False)),
                    model.name,
                    runtime,
                    options.approximation_mode,
                )
                return
            self._stream_chat(
                pieces,
                request_id,
                thinking_mode == "thinking",
                bool(stream_options.get("include_usage", False)),
                model.name,
                options.approximation_mode,
            )
            return

        if tool_calling:
            raw, prompt_tokens, generated, finish = _collect_raw(pieces)
            try:
                turn = runtime.parse_chat(raw, thinking_mode)
            except Exception as error:
                raise APIError(
                    "The model returned an invalid tool call.",
                    status=500,
                    code="invalid_tool_call",
                    error_type="server_error",
                ) from error
            tool_calls = _response_tool_calls(turn.tool_calls)
            message: dict[str, Any] = {
                "role": "assistant",
                "content": turn.content or None,
            }
            if turn.reasoning_content:
                message["reasoning_content"] = turn.reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
            self._json(
                200,
                {
                    "id": request_id,
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model.name,
                    "approximation": {"mode": options.approximation_mode},
                    "choices": [
                        {
                            "index": 0,
                            "message": message,
                            "finish_reason": "tool_calls" if tool_calls else finish,
                        }
                    ],
                    "usage": _usage(prompt_tokens, generated),
                },
            )
            return

        text, reasoning, prompt_tokens, generated, finish = _collect(
            pieces,
            thinking_mode == "thinking",
        )
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if reasoning:
            message["reasoning_content"] = reasoning
        self._json(
            200,
            {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model.name,
                "approximation": {"mode": options.approximation_mode},
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": finish,
                    }
                ],
                "usage": _usage(prompt_tokens, generated),
            },
        )

    def _responses(self, payload: dict[str, Any], model: ModelRequest) -> None:
        runtime = model.runtime
        request = dict(payload)
        if "max_output_tokens" in request:
            request["max_tokens"] = request["max_output_tokens"]
        support = support_for_runtime(runtime)
        messages = _response_messages(payload, content_parser=(
            self._content if any(support.descriptor.supports(f) for f in ("imageInput", "documentInput", "audioInput")) else None))
        tools, tool_choice, response_tools = _response_tool_request(payload)
        if support.prefer_tool_first(messages, tool_choice, response_tools):
            tool_choice = ToolChoice("required")
        thinking_mode, reasoning_effort = _reasoning_settings(
            payload,
            responses_api=True,
            support=support,
        )
        options, stream = self._common(
            request,
            model.defaults,
            support=support,
            dspark=bool(getattr(runtime.config, "dspark_enabled", False)),
            thinking_mode=thinking_mode,
        )
        _validate_approximation_runtime(options, runtime)
        _validate_response_request(payload)
        prompt = runtime.encode_chat(
            messages,
            thinking_mode,
            tools,
            tool_choice,
            reasoning_effort,
        )
        request_id = "resp_" + uuid.uuid4().hex
        if tool_choice.mode in {"required", "function"}:
            pieces = self._validated_response_pieces(
                runtime,
                prompt,
                messages,
                tools,
                tool_choice,
                reasoning_effort,
                thinking_mode,
                options,
            )
        else:
            pieces = self._response_pieces(runtime.stream(prompt, options))
        tool_calling = bool(tools) and tool_choice.mode != "none"
        if stream:
            self._stream_response(
                pieces,
                request_id,
                payload,
                options,
                thinking_mode,
                tool_calling,
                tool_choice,
                response_tools,
                model.name,
                runtime,
            )
            return

        if tool_calling:
            raw, prompt_tokens, generated, _ = _collect_raw(pieces)
            try:
                turn = runtime.parse_chat(raw, thinking_mode)
            except Exception as error:
                raise APIError(
                    "The model returned an invalid tool call.",
                    status=500,
                    code="invalid_tool_call",
                    error_type="server_error",
                ) from error
            output = _response_output(
                turn.content,
                turn.reasoning_content,
                include_empty=False,
            )
            output.extend(_response_function_calls(turn.tool_calls, response_tools))
        else:
            text, reasoning, prompt_tokens, generated, _ = _collect(
                pieces,
                thinking_mode == "thinking",
            )
            output = _response_output(text, reasoning)
        response = _response_object(
            request_id,
            model.name,
            payload,
            options,
            output,
            _response_usage(prompt_tokens, generated),
        )
        self._json(200, response)

    def _validated_response_pieces(
        self,
        runtime: Any,
        prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: ToolChoice,
        reasoning_effort: str,
        thinking_mode: str,
        options: GenerationOptions,
    ) -> Iterator[GeneratedPiece]:
        current_prompt = prompt
        parse_failed = False
        for attempt in range(2):
            pieces = list(self._response_pieces(runtime.stream(current_prompt, options)))
            raw = "".join(piece.text for piece in pieces)
            try:
                turn = runtime.parse_chat(raw, thinking_mode)
            except Exception:
                parse_failed = True
            else:
                parse_failed = False
                if _tool_choice_satisfied(turn.tool_calls, tool_choice):
                    yield from pieces
                    return
            if attempt == 0:
                retry_messages = [
                    {
                        "role": "developer",
                        "content": _tool_retry_instruction(tool_choice),
                    },
                    *messages,
                ]
                current_prompt = runtime.encode_chat(
                    retry_messages,
                    thinking_mode,
                    tools,
                    tool_choice,
                    reasoning_effort,
                )
        if parse_failed:
            raise APIError(
                "The model returned an invalid required tool call after one retry.",
                status=500,
                code="invalid_tool_call",
                error_type="server_error",
            )
        raise APIError(
            "The model did not return the required tool call after one retry.",
            status=500,
            code="tool_choice_not_satisfied",
            error_type="server_error",
        )

    def _stream_response(
        self,
        pieces: Iterator[GeneratedPiece],
        request_id: str,
        payload: dict[str, Any],
        options: GenerationOptions,
        thinking_mode: str,
        tool_calling: bool,
        tool_choice: ToolChoice,
        response_tools: dict[str, dict[str, str]],
        model_name: str,
        runtime: Any,
    ) -> None:
        self._start_sse()
        created = _response_object(
            request_id, model_name, payload, options, [], None, status="in_progress",
        )
        with ResponseEvents(self, created) as events:
            self._active_response_events = events
            try:
                self._write_response(
                    pieces, request_id, payload, options, thinking_mode, tool_calling,
                    tool_choice, response_tools, model_name, runtime, events.send,
                )
            except (GenerationCancelled, BrokenPipeError, ConnectionResetError):
                raise
            except Exception as error:
                sys.stderr.write(f"response {request_id} failed: {error}\n")
                failed = {**created, "status": "failed", "error": {
                    "code": error.code if isinstance(error, APIError) and error.code else "server_error",
                    "message": str(error) if isinstance(error, APIError) else "Generation failed. Check the server log.",
                }}
                events.send("response.failed", response=failed)
            finally:
                close = getattr(pieces, "close", None)
                if close is not None:
                    close()
                self._active_response_events = None

    def _response_pieces(self, pieces: Iterator[GeneratedPiece]) -> Iterator[GeneratedPiece]:
        tracked = self.app.track(pieces)
        try:
            for piece in tracked:
                events = getattr(self, "_active_response_events", None)
                if events is not None:
                    events.check_connection()
                yield piece
        finally:
            tracked.close()
            close = getattr(pieces, "close", None)
            if close is not None:
                close()

    def _write_response(
        self,
        pieces: Iterator[GeneratedPiece],
        request_id: str,
        payload: dict[str, Any],
        options: GenerationOptions,
        thinking_mode: str,
        tool_calling: bool,
        tool_choice: ToolChoice,
        response_tools: dict[str, dict[str, str]],
        model_name: str,
        runtime: Any,
        send,
    ) -> None:
        output: list[dict[str, Any]] = []
        prompt_tokens = generated = 0
        reasoning_item: dict[str, Any] | None = None
        reasoning_index = -1
        reasoning_text = ""
        message_item: dict[str, Any] | None = None
        message_index = -1
        message_text = ""
        calls: dict[int, tuple[int, dict[str, Any]]] = {}

        def start_reasoning() -> None:
            nonlocal reasoning_item, reasoning_index
            if reasoning_item is not None:
                return
            reasoning_index = len(output)
            reasoning_item = {
                "id": "rs_" + uuid.uuid4().hex,
                "type": "reasoning",
                "summary": [],
                "status": "in_progress",
            }
            output.append(reasoning_item)
            send(
                "response.output_item.added",
                output_index=reasoning_index,
                item=dict(reasoning_item),
            )
            send(
                "response.reasoning_summary_part.added",
                item_id=reasoning_item["id"],
                output_index=reasoning_index,
                summary_index=0,
                part={"type": "summary_text", "text": ""},
            )

        def finish_reasoning() -> None:
            if reasoning_item is None or reasoning_item["status"] == "completed":
                return
            part = {"type": "summary_text", "text": reasoning_text}
            reasoning_item["summary"] = [part]
            reasoning_item["status"] = "completed"
            send(
                "response.reasoning_summary_text.done",
                item_id=reasoning_item["id"],
                output_index=reasoning_index,
                summary_index=0,
                text=reasoning_text,
            )
            send(
                "response.reasoning_summary_part.done",
                item_id=reasoning_item["id"],
                output_index=reasoning_index,
                summary_index=0,
                part=part,
            )
            send(
                "response.output_item.done",
                output_index=reasoning_index,
                item=dict(reasoning_item),
            )

        def start_message() -> None:
            nonlocal message_item, message_index
            if message_item is not None:
                return
            finish_reasoning()
            message_index = len(output)
            message_item = _response_message("", status="in_progress")
            output.append(message_item)
            send(
                "response.output_item.added",
                output_index=message_index,
                item=dict(message_item),
            )
            send(
                "response.content_part.added",
                item_id=message_item["id"],
                output_index=message_index,
                content_index=0,
                part=_response_text(""),
            )

        def finish_message() -> None:
            if message_item is None or message_item["status"] == "completed":
                return
            text_part = _response_text(message_text)
            message_item["content"] = [text_part]
            message_item["status"] = "completed"
            send(
                "response.output_text.done",
                item_id=message_item["id"],
                output_index=message_index,
                content_index=0,
                text=message_text,
                logprobs=[],
            )
            send(
                "response.content_part.done",
                item_id=message_item["id"],
                output_index=message_index,
                content_index=0,
                part=text_part,
            )
            send(
                "response.output_item.done",
                output_index=message_index,
                item=dict(message_item),
            )

        def send_delta(delta: ToolStreamDelta) -> None:
            nonlocal reasoning_text, message_text
            if delta.reasoning_content:
                start_reasoning()
                reasoning_text += delta.reasoning_content
                send(
                    "response.reasoning_summary_text.delta",
                    item_id=reasoning_item["id"],
                    output_index=reasoning_index,
                    summary_index=0,
                    delta=delta.reasoning_content,
                )
            if delta.content:
                start_message()
                message_text += delta.content
                send(
                    "response.output_text.delta",
                    item_id=message_item["id"],
                    output_index=message_index,
                    content_index=0,
                    delta=delta.content,
                    logprobs=[],
                )
            if delta.tool_index is None:
                return
            if delta.tool_name is not None and delta.tool_index not in calls:
                finish_reasoning()
                finish_message()
                item = _response_tool_call(
                    delta.tool_name,
                    "",
                    response_tools,
                    status="in_progress",
                )
                output_index = len(output)
                output.append(item)
                calls[delta.tool_index] = (output_index, item)
                send(
                    "response.output_item.added",
                    output_index=output_index,
                    item=dict(item),
                )
            if delta.arguments and delta.tool_index in calls:
                output_index, item = calls[delta.tool_index]
                item["arguments"] += delta.arguments
                send(
                    "response.function_call_arguments.delta",
                    item_id=item["id"],
                    output_index=output_index,
                    delta=delta.arguments,
                )

        def fail(message: str, code: str = "invalid_tool_call") -> None:
            nonlocal message_item, message_index, message_text
            notice = f"Whallm could not complete the request: {message}"
            if message_text:
                notice = "\n\n" + notice
            if message_item is not None and message_item["status"] == "completed":
                message_item = None
                message_index = -1
                message_text = ""
            send_delta(ToolStreamDelta(content=notice))
            finish_reasoning()
            finish_message()
            error = {"code": code, "message": message}
            send("error", **error, param=None)
            failed = _response_object(
                request_id,
                model_name,
                payload,
                options,
                output,
                _response_usage(prompt_tokens, generated),
                status="failed",
            )
            failed["error"] = error
            send("response.completed", response=failed)

        if tool_calling:
            try:
                raw, prompt_tokens, generated, _ = _collect_raw(pieces)
            except APIError as error:
                fail(str(error), error.code or "invalid_tool_call")
                return
            try:
                turn = runtime.parse_chat(raw, thinking_mode)
            except Exception:
                fail("The model returned an invalid tool call.")
                return
            if not _tool_choice_satisfied(turn.tool_calls, tool_choice):
                fail(
                    "The model did not return the required tool call.",
                    "tool_choice_not_satisfied",
                )
                return
            send_delta(
                ToolStreamDelta(
                    reasoning_content=turn.reasoning_content,
                    content=turn.content,
                )
            )
            for index, call in enumerate(turn.tool_calls):
                send_delta(ToolStreamDelta(tool_index=index, tool_name=call.name))
                send_delta(ToolStreamDelta(tool_index=index, arguments=call.arguments))
        else:
            parser = ReasoningParser(thinking_mode == "thinking")
            for piece in pieces:
                prompt_tokens = piece.prompt_tokens
                generated = piece.generation_tokens
                reasoning, content = parser.feed(piece.text)
                send_delta(ToolStreamDelta(reasoning_content=reasoning, content=content))
            reasoning, content = parser.finish()
            send_delta(ToolStreamDelta(reasoning_content=reasoning, content=content))

        finish_reasoning()
        if message_item is None and not calls:
            start_message()
        finish_message()
        for output_index, item in calls.values():
            item["status"] = "completed"
            send(
                "response.function_call_arguments.done",
                item_id=item["id"],
                output_index=output_index,
                arguments=item["arguments"],
            )
            send(
                "response.output_item.done",
                output_index=output_index,
                item=dict(item),
            )
        completed = _response_object(
            request_id,
            model_name,
            payload,
            options,
            output,
            _response_usage(prompt_tokens, generated),
        )
        send("response.completed", response=completed)

    def _completion(self, payload: dict[str, Any], model: ModelRequest) -> None:
        runtime = model.runtime
        support = support_for_runtime(runtime)
        options, stream = self._common(
            payload,
            model.defaults,
            support=support,
            dspark=bool(getattr(runtime.config, "dspark_enabled", False)),
            thinking_mode="chat",
        )
        _validate_approximation_runtime(options, runtime)
        if payload.get("tools") not in (None, []):
            raise APIError("tools are only supported for chat completions.", param="tools")
        if payload.get("tool_choice") not in (None, "none"):
            raise APIError(
                "tool_choice is only supported for chat completions.",
                param="tool_choice",
            )
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            raise APIError("prompt must be a string.", param="prompt")
        request_id = "cmpl-" + uuid.uuid4().hex
        pieces = self.app.track(runtime.stream(prompt, options))
        if stream:
            self._stream_completion(
                pieces,
                request_id,
                model.name,
                options.approximation_mode,
            )
            return
        text, _, prompt_tokens, generated, finish = _collect(pieces, False)
        self._json(
            200,
            {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": model.name,
                "approximation": {"mode": options.approximation_mode},
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "logprobs": None,
                        "finish_reason": finish,
                    }
                ],
                "usage": _usage(prompt_tokens, generated),
            },
        )

    def _common(
        self,
        payload: dict[str, Any],
        defaults: ServerDefaults | ModelDefaults,
        *,
        qwen: bool = False,
        support=None,
        dspark: bool = False,
        thinking_mode: str = "chat",
    ) -> tuple[GenerationOptions, bool]:
        unsupported = {
            "response_format": payload.get("response_format"),
            "stop": payload.get("stop"),
        }
        for name, value in unsupported.items():
            if value is not None and value != []:
                raise APIError(f"{name} is not supported.", param=name)
        for name in ("frequency_penalty", "presence_penalty"):
            if payload.get(name) not in (None, 0, 0.0):
                raise APIError(f"{name} is not supported.", param=name)
        if payload.get("logprobs") not in (None, False):
            raise APIError("logprobs is not supported.", param="logprobs")
        if payload.get("echo") not in (None, False):
            raise APIError("echo is not supported.", param="echo")
        n = payload.get("n", 1)
        if type(n) is not int or n != 1:
            raise APIError("n must be 1.", param="n")
        stream = payload.get("stream", False)
        if type(stream) is not bool:
            raise APIError("stream must be a boolean.", param="stream")
        stream_options = payload.get("stream_options")
        if stream_options is not None and not isinstance(stream_options, dict):
            raise APIError("stream_options must be an object.", param="stream_options")
        if isinstance(stream_options, dict):
            include_usage = stream_options.get("include_usage", False)
            if type(include_usage) is not bool:
                raise APIError(
                    "stream_options.include_usage must be a boolean.",
                    param="stream_options.include_usage",
                )
        support = support or get_support("qwen3.8-flash-next" if qwen else "deepseek-v4")
        return _options(
            payload, defaults, support=support,
            approximation_default=support.default_approximation(
                dspark, getattr(defaults, "approximation_mode", "exact"),
            ),
            thinking_mode=thinking_mode,
        ), stream

    def _stream_chat(
        self,
        pieces: Iterator[GeneratedPiece],
        request_id: str,
        thinking: bool,
        include_usage: bool,
        model_name: str,
        approximation_mode: str,
    ) -> None:
        self._start_sse()
        created = int(time.time())
        base = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "approximation": {"mode": approximation_mode},
        }
        self._sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        parser = ReasoningParser(thinking)
        prompt_tokens = generated = 0
        finish = None
        for piece in pieces:
            prompt_tokens = piece.prompt_tokens
            generated = piece.generation_tokens
            finish = piece.finish_reason or finish
            reasoning, content = parser.feed(piece.text)
            if reasoning:
                self._sse({**base, "choices": [{"index": 0, "delta": {"reasoning_content": reasoning}, "finish_reason": None}]})
            if content:
                self._sse({**base, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
        reasoning, content = parser.finish()
        if reasoning:
            self._sse({**base, "choices": [{"index": 0, "delta": {"reasoning_content": reasoning}, "finish_reason": None}]})
        if content:
            self._sse({**base, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]})
        self._sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish or "stop"}]})
        if include_usage:
            self._sse({**base, "choices": [], "usage": _usage(prompt_tokens, generated)})
        self._sse_done()

    def _stream_tool_chat(
        self,
        pieces: Iterator[GeneratedPiece],
        request_id: str,
        thinking_mode: str,
        include_usage: bool,
        model_name: str,
        runtime: Any,
        approximation_mode: str,
    ) -> None:
        self._start_sse()
        created = int(time.time())
        base = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "approximation": {"mode": approximation_mode},
        }
        self._sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
        )
        factory = getattr(runtime, "make_tool_stream_parser", None)
        parser = (
            factory(thinking_mode)
            if callable(factory)
            else ToolStreamParser(thinking_mode)
        )
        raw_parts = []
        prompt_tokens = generated = 0
        finish = "stop"

        def send(delta: ToolStreamDelta) -> None:
            value: dict[str, Any]
            if delta.reasoning_content:
                value = {"reasoning_content": delta.reasoning_content}
            elif delta.content:
                value = {"content": delta.content}
            elif delta.tool_index is not None:
                function: dict[str, str] = {"arguments": delta.arguments}
                call: dict[str, Any] = {
                    "index": delta.tool_index,
                    "function": function,
                }
                if delta.tool_name is not None:
                    call_id = "call_" + uuid.uuid4().hex
                    call.update({"id": call_id, "type": "function"})
                    function["name"] = delta.tool_name
                value = {"tool_calls": [call]}
            else:
                return
            self._sse(
                {
                    **base,
                    "choices": [
                        {"index": 0, "delta": value, "finish_reason": None}
                    ],
                }
            )

        for piece in pieces:
            raw_parts.append(piece.text)
            prompt_tokens = piece.prompt_tokens
            generated = piece.generation_tokens
            finish = piece.finish_reason or finish
            for delta in parser.feed(piece.text):
                send(delta)
        for delta in parser.finish():
            send(delta)

        raw = "".join(raw_parts)
        try:
            turn = runtime.parse_chat(raw, thinking_mode)
        except Exception:
            self._sse(
                {
                    "error": {
                        "message": "The model returned an invalid tool call.",
                        "type": "server_error",
                        "param": None,
                        "code": "invalid_tool_call",
                    }
                }
            )
            self._sse_done()
            return
        if not parser.matches(turn.tool_calls):
            if parser.streamed_tool_count:
                self._sse(
                    {
                        "error": {
                            "message": "The streamed tool call failed validation.",
                            "type": "server_error",
                            "param": None,
                            "code": "invalid_tool_call",
                        }
                    }
                )
                self._sse_done()
                return
            for index, call in enumerate(_response_tool_calls(turn.tool_calls)):
                self._sse(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": [{"index": index, **call}]},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
        self._sse(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if turn.tool_calls else finish,
                    }
                ],
            }
        )
        if include_usage:
            self._sse(
                {
                    **base,
                    "choices": [],
                    "usage": _usage(prompt_tokens, generated),
                }
            )
        self._sse_done()

    def _stream_completion(
        self,
        pieces: Iterator[GeneratedPiece],
        request_id: str,
        model_name: str,
        approximation_mode: str,
    ) -> None:
        self._start_sse()
        created = int(time.time())
        finish = None
        for piece in pieces:
            finish = piece.finish_reason or finish
            self._sse(
                {
                    "id": request_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_name,
                    "approximation": {"mode": approximation_mode},
                    "choices": [
                        {
                            "index": 0,
                            "text": piece.text,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            )
        self._sse(
            {
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": model_name,
                "approximation": {"mode": approximation_mode},
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "logprobs": None,
                        "finish_reason": finish or "stop",
                    }
                ],
            }
        )
        self._sse_done()

    def _status(self) -> dict[str, Any]:
        snapshot = self.app.model_manager.status_snapshot()
        if snapshot["loaded_model"] is not None:
            snapshot["performance"] = {
                **snapshot["performance"],
                **self.app.metrics.snapshot(),
            }
        return {
            "status": "ready",
            "requires_api_key": self.app.api_key is not None,
            **snapshot,
            "app_memory": self.app.status_memory.snapshot(),
        }

    def _authorize(self) -> None:
        expected = self.app.api_key
        if expected is None:
            return
        value = self.headers.get("Authorization", "")
        supplied = value[7:] if value.startswith("Bearer ") else ""
        if not secrets.compare_digest(supplied, expected):
            raise APIError(
                "Provide a valid Bearer API key.",
                status=401,
                code="invalid_api_key",
                error_type="authentication_error",
            )

    def _request_json(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise APIError("Content-Length is required.", status=411)
        try:
            length = int(raw_length)
        except ValueError as error:
            raise APIError("Content-Length is invalid.", status=400) from error
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise APIError("Request body is too large.", status=413)
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise APIError("Request body must be valid JSON.") from error
        if not isinstance(value, dict):
            raise APIError("Request body must be a JSON object.")
        if self.app.log_level == "debug":
            body = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            sys.stderr.write(f"[DEBUG] request {self.command} {self.path}: {body}\n")
            sys.stderr.flush()
        return value

    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, value: dict[str, Any]) -> None:
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        self.wfile.write(f"data: {data}\n\n".encode())
        self.wfile.flush()

    def _sse_done(self) -> None:
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
        self._bytes(status, body, "application/json; charset=utf-8")

    def _bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if body:
            self.wfile.write(body)
        self.close_connection = True

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if urlsplit(self.path).path == "/api/status":
            return
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = 500
        if self.app.log_level == "error" and status < 400:
            return
        self.log_message('"%s" %s %s', self.requestline, str(code), str(size))

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"{self.client_address[0]} - {format % args}\n")
        sys.stderr.flush()


def _validate_response_request(payload: dict[str, Any]) -> None:
    for name in ("previous_response_id", "conversation"):
        if payload.get(name) is not None:
            raise APIError(f"{name} is not supported.", param=name)
    for name in ("store", "background"):
        if payload.get(name) not in (None, False):
            raise APIError(f"{name} is not supported.", param=name)
    parallel = payload.get("parallel_tool_calls")
    if parallel is not None and type(parallel) is not bool:
        raise APIError(
            "parallel_tool_calls must be a boolean.",
            param="parallel_tool_calls",
        )
    if payload.get("truncation") not in (None, "disabled"):
        raise APIError("truncation is not supported.", param="truncation")
    text = payload.get("text")
    if text not in (None, {}):
        if not isinstance(text, dict) or text.get("format", {"type": "text"}) != {
            "type": "text"
        }:
            raise APIError("Only plain text output is supported.", param="text.format")


_REASONING_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "low",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}


def _reasoning_settings(
    payload: dict[str, Any],
    *,
    responses_api: bool = False,
    qwen: bool = False,
    support=None,
) -> tuple[str, str]:
    thinking_mode = payload.get("thinking_mode")
    if thinking_mode is not None:
        if thinking_mode not in {"chat", "thinking"}:
            raise APIError(
                "thinking_mode must be 'chat' or 'thinking'.",
                param="thinking_mode",
            )
    if responses_api:
        reasoning = payload.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, dict):
            raise APIError("reasoning must be an object.", param="reasoning")
        effort = reasoning.get("effort") if isinstance(reasoning, dict) else None
        param = "reasoning.effort"
    else:
        effort = payload.get("reasoning_effort")
        param = "reasoning_effort"
    if effort is not None and not isinstance(effort, str):
        raise APIError(f"{param} must be a string.", param=param)
    if effort not in {None, "none", *_REASONING_EFFORT_MAP}:
        raise APIError(f"{param} is invalid.", param=param)
    support = support or get_support("qwen3.8-flash-next" if qwen else "deepseek-v4")
    return support.reasoning_settings(thinking_mode, effort)


def _response_messages(payload: dict[str, Any], *, content_parser=None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    instructions = payload.get("instructions")
    if instructions is not None:
        if not isinstance(instructions, str):
            raise APIError("instructions must be a string.", param="instructions")
        messages.append({"role": "developer", "content": instructions})

    value = payload.get("input")
    if isinstance(value, str):
        messages.append({"role": "user", "content": value})
        return _messages(messages, allow_assistant_final=True, content_parser=content_parser)
    if not isinstance(value, list) or not value:
        raise APIError("input must be a string or a non-empty array.", param="input")

    pending_calls: list[dict[str, Any]] = []

    def flush_calls() -> None:
        if pending_calls:
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": pending_calls[:]}
            )
            pending_calls.clear()

    for index, raw in enumerate(value):
        param = f"input.{index}"
        if not isinstance(raw, dict):
            raise APIError("Each input item must be an object.", param=param)
        item_type = raw.get("type")
        if item_type == "function_call":
            call_id = raw.get("call_id")
            name = raw.get("name")
            namespace = raw.get("namespace")
            arguments = raw.get("arguments")
            if not isinstance(call_id, str) or not call_id:
                raise APIError("call_id must be a non-empty string.", param=f"{param}.call_id")
            _validate_function_name(name, f"{param}.name")
            if namespace is not None:
                _validate_function_name(namespace, f"{param}.namespace")
            name = _response_internal_tool_name(name, namespace)
            if not isinstance(arguments, str):
                raise APIError("arguments must be a JSON string.", param=f"{param}.arguments")
            pending_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
            continue
        if item_type == "reasoning":
            continue
        flush_calls()
        if item_type == "function_call_output":
            call_id = raw.get("call_id")
            output = raw.get("output")
            if not isinstance(call_id, str) or not call_id:
                raise APIError("call_id must be a non-empty string.", param=f"{param}.call_id")
            if not isinstance(output, str) and not (content_parser is not None and isinstance(output, list)):
                raise APIError("output must be text or supported content parts.", param=f"{param}.output")
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": output}
            )
            continue
        if item_type not in (None, "message"):
            raise APIError("Only text messages and function calls are supported.", param=param)
        messages.append({"role": raw.get("role"), "content": raw.get("content")})
    flush_calls()
    return _messages(messages, allow_assistant_final=True, content_parser=content_parser)


def _response_tool_request(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], ToolChoice, dict[str, dict[str, str]]]:
    raw_tools = payload.get("tools")
    response_tools: dict[str, dict[str, str]] = {}
    if raw_tools is None:
        tools = None
    elif not isinstance(raw_tools, list):
        raise APIError("tools must be an array.", param="tools")
    else:
        tools = []

        def add_tool(raw: dict[str, Any], param: str, namespace: str | None = None) -> None:
            kind = raw.get("type")
            if kind != "function":
                raise APIError("Namespace tools must be functions.", param=param)
            name = raw.get("name")
            _validate_function_name(name, f"{param}.name")
            internal_name = _response_internal_tool_name(name, namespace)
            description = raw.get("description", "")
            if not isinstance(description, str):
                raise APIError("Tool description must be a string.", param=f"{param}.description")
            function = {
                field: raw[field]
                for field in ("description", "parameters", "strict")
                if field in raw
            }
            function["name"] = internal_name
            tools.append({"type": "function", "function": function})
            response_tools[internal_name] = {
                "type": kind,
                "name": name,
                **({"namespace": namespace} if namespace else {}),
            }

        for index, raw in enumerate(raw_tools):
            param = f"tools.{index}"
            if not isinstance(raw, dict):
                raise APIError("Each tool must be an object.", param=param)
            kind = raw.get("type")
            if kind == "function":
                add_tool(raw, param)
                continue
            if kind == "namespace":
                namespace = raw.get("name")
                _validate_function_name(namespace, f"{param}.name")
                namespace_tools = raw.get("tools")
                if not isinstance(namespace_tools, list):
                    raise APIError("Namespace tools must be an array.", param=f"{param}.tools")
                for child_index, child in enumerate(namespace_tools):
                    child_param = f"{param}.tools.{child_index}"
                    if not isinstance(child, dict):
                        raise APIError("Each namespace tool must be an object.", param=child_param)
                    add_tool(child, child_param, namespace)
                continue
            if kind == "web_search":
                continue
            raise APIError(
                "This tool type is not supported.",
                param=param,
            )
    choice = payload.get("tool_choice", "auto")
    if isinstance(choice, dict) and choice.get("type") == "function":
        selected = next(
            (
                internal_name
                for internal_name, tool in response_tools.items()
                if tool["type"] == choice.get("type")
                and tool["name"] == choice.get("name")
                and tool.get("namespace") == choice.get("namespace")
            ),
            None,
        )
        choice = {"type": "function", "function": {"name": selected}}
    request = {"tools": tools, "tool_choice": choice}
    parsed_tools, parsed_choice = _tool_request(request)
    return parsed_tools, parsed_choice, response_tools


def _use_qwen_codex_tool_first(
    qwen: bool,
    messages: list[dict[str, Any]],
    tool_choice: ToolChoice,
    response_tools: dict[str, dict[str, str]],
) -> bool:
    return qwen and get_support("qwen3.8-flash-next").prefer_tool_first(
        messages, tool_choice, response_tools,
    )


def _tool_choice_satisfied(calls: tuple[Any, ...], choice: ToolChoice) -> bool:
    if choice.mode == "required":
        return bool(calls)
    if choice.mode == "function":
        return any(call.name == choice.name for call in calls)
    return True


def _tool_retry_instruction(choice: ToolChoice) -> str:
    if choice.mode == "function":
        target = f'Call the required "{choice.name}" tool now.'
    else:
        target = "Call one of the provided tools now."
    return (
        "The previous attempt did not produce the required tool call. "
        f"{target} Return a tool call, not a status update or final answer."
    )


def _response_text(text: str) -> dict[str, Any]:
    return {"type": "output_text", "annotations": [], "logprobs": [], "text": text}


def _response_message(text: str, *, status: str = "completed") -> dict[str, Any]:
    return {
        "id": "msg_" + uuid.uuid4().hex,
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": [_response_text(text)],
    }


def _response_output(
    text: str,
    reasoning: str,
    *,
    include_empty: bool = True,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if reasoning:
        output.append(
            {
                "id": "rs_" + uuid.uuid4().hex,
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": reasoning}],
                "status": "completed",
            }
        )
    if text or (include_empty and not reasoning):
        output.append(_response_message(text))
    return output


def _response_internal_tool_name(name: str, namespace: str | None) -> str:
    if namespace is None:
        return name
    qualified = f"{namespace}__{name}"
    if len(qualified) <= 64:
        return qualified
    digest = hashlib.sha256(qualified.encode()).hexdigest()[:32]
    return f"{qualified[:30]}__{digest}"


def _response_tool_call(
    internal_name: str,
    arguments: str,
    response_tools: dict[str, dict[str, str]],
    *,
    status: str,
) -> dict[str, Any]:
    tool = response_tools.get(
        internal_name,
        {"type": "function", "name": internal_name},
    )
    item: dict[str, Any] = {
        "id": "fc_" + uuid.uuid4().hex,
        "call_id": "call_" + uuid.uuid4().hex,
        "type": "function_call",
        "name": tool["name"],
        "arguments": arguments,
        "status": status,
    }
    if "namespace" in tool:
        item["namespace"] = tool["namespace"]
    return item


def _response_function_calls(
    calls,
    response_tools: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    tools = response_tools or {}
    return [
        _response_tool_call(call.name, call.arguments, tools, status="completed")
        for call in calls
    ]


def _response_usage(input_tokens: int, output_tokens: int) -> dict[str, Any]:
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + output_tokens,
    }


def _response_object(
    request_id: str,
    model: str,
    payload: dict[str, Any],
    options: GenerationOptions,
    output: list[dict[str, Any]],
    usage: dict[str, Any] | None,
    *,
    status: str = "completed",
) -> dict[str, Any]:
    now = time.time()
    return {
        "id": request_id,
        "object": "response",
        "created_at": now,
        "status": status,
        "completed_at": now if status == "completed" else None,
        "error": None,
        "incomplete_details": None,
        "instructions": payload.get("instructions"),
        "max_output_tokens": options.max_tokens,
        "model": model,
        "approximation": {"mode": options.approximation_mode},
        "output": output,
        "parallel_tool_calls": payload.get("parallel_tool_calls", True),
        "previous_response_id": None,
        "reasoning": payload.get("reasoning"),
        "store": False,
        "temperature": options.temperature,
        "text": {"format": {"type": "text"}},
        "tool_choice": payload.get("tool_choice", "auto"),
        "tools": payload.get("tools") or [],
        "top_p": options.top_p,
        "truncation": "disabled",
        "usage": usage,
        "metadata": payload.get("metadata") or {},
    }


def _number(
    value: Any,
    name: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise APIError(f"{name} must be a number.", param=name)
    result = float(value)
    if not minimum <= result <= maximum:
        raise APIError(
            f"{name} must be between {minimum:g} and {maximum:g}.",
            param=name,
        )
    return result


def _options(
    payload: dict[str, Any],
    defaults: ServerDefaults | ModelDefaults,
    *,
    qwen: bool = False,
    support=None,
    approximation_default: str = EXACT_APPROXIMATION_MODE,
    thinking_mode: str = "chat",
) -> GenerationOptions:
    support = support or get_support("qwen3.8-flash-next" if qwen else "deepseek-v4")
    if "approximation" not in payload:
        approximation_mode = approximation_default
    else:
        approximation = payload["approximation"]
        if not isinstance(approximation, dict):
            raise APIError("approximation must be an object.", param="approximation")
        if set(approximation) != {"mode"}:
            raise APIError(
                "approximation must contain only mode.",
                param="approximation",
            )
        approximation_mode = approximation.get("mode")
        if not isinstance(approximation_mode, str):
            raise APIError(
                "approximation.mode must be a string.",
                param="approximation.mode",
            )
        if approximation_mode not in APPROXIMATION_MODES:
            raise APIError(
                "approximation.mode is not supported.",
                param="approximation.mode",
            )
        if not support.descriptor.supports("approximation") and approximation_mode != EXACT_APPROXIMATION_MODE:
            raise APIError(
                f"approximation.mode is not supported for {support.option_error_name}.",
                param="approximation.mode",
            )
    max_tokens = payload.get(
        "max_completion_tokens",
        payload.get("max_output_tokens", payload.get("max_tokens", defaults.max_tokens)),
    )
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
        raise APIError("max_tokens must be an integer.", param="max_tokens")
    if not 1 <= max_tokens <= MAX_GENERATION_TOKENS:
        raise APIError(
            f"max_tokens must be between 1 and {MAX_GENERATION_TOKENS}.",
            param="max_tokens",
        )
    sampling_defaults = support.sampling_defaults(defaults, thinking_mode)
    seed = payload.get("seed")
    if seed is not None and (type(seed) is not int or not 0 <= seed <= 2**32 - 1):
        raise APIError("seed must be an integer between 0 and 4294967295.", param="seed")
    temperature = _number(
        payload.get("temperature", sampling_defaults["temperature"]),
        "temperature",
        minimum=0,
        maximum=2,
    )
    top_p = _number(
        payload.get("top_p", sampling_defaults["top_p"]),
        "top_p",
        minimum=0.000001,
        maximum=1,
    )
    top_k = payload.get("top_k", sampling_defaults["top_k"])
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 0 <= top_k <= 248_320:
        raise APIError("top_k must be between 0 and 248320.", param="top_k")
    return GenerationOptions(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=sampling_defaults["min_p"],
        presence_penalty=sampling_defaults["presence_penalty"],
        repetition_penalty=sampling_defaults["repetition_penalty"],
        approximation_mode=approximation_mode,
        seed=seed,
    )


def _validate_approximation_runtime(options: GenerationOptions, runtime: Any) -> None:
    if options.approximation_mode != LEARNED_ROUTE_DROP_LOWEST_1:
        return
    if any(bool(getattr(getattr(runtime, "config", None), name, False))
           for name in ("dspark_enabled", "mtp_enabled")):
        raise APIError(
            "approximation.mode is not supported with DSpark or MTP.",
            param="approximation.mode",
        )
    if not support_for_runtime(runtime).descriptor.supports("approximation"):
        raise APIError(
            f"approximation.mode is not supported for {support_for_runtime(runtime).option_error_name}.",
            param="approximation.mode",
        )


def _tool_request(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], ToolChoice]:
    value = payload.get("tools")
    if value is None:
        tools: list[dict[str, Any]] = []
    elif not isinstance(value, list):
        raise APIError("tools must be an array.", param="tools")
    else:
        tools = []
        names = set()
        for index, raw in enumerate(value):
            param = f"tools.{index}"
            if not isinstance(raw, dict) or raw.get("type") != "function":
                raise APIError("Each tool must be a function.", param=param)
            function = raw.get("function")
            if not isinstance(function, dict):
                raise APIError("Tool function must be an object.", param=f"{param}.function")
            name = function.get("name")
            _validate_function_name(name, f"{param}.function.name")
            if name in names:
                raise APIError("Tool function names must be unique.", param=f"{param}.function.name")
            names.add(name)
            description = function.get("description")
            if description is not None and not isinstance(description, str):
                raise APIError(
                    "Tool description must be a string.",
                    param=f"{param}.function.description",
                )
            parameters = function.get("parameters")
            if parameters is not None and not isinstance(parameters, dict):
                raise APIError(
                    "Tool parameters must be an object.",
                    param=f"{param}.function.parameters",
                )
            tools.append(raw)

    raw_choice = payload.get("tool_choice", "auto")
    if isinstance(raw_choice, str) and raw_choice in {"auto", "none", "required"}:
        choice = ToolChoice(raw_choice)
    elif isinstance(raw_choice, dict):
        function = raw_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if raw_choice.get("type") != "function":
            raise APIError("tool_choice must select a function.", param="tool_choice")
        _validate_function_name(name, "tool_choice.function.name")
        if name not in {tool["function"]["name"] for tool in tools}:
            raise APIError(
                "tool_choice must name a provided tool.",
                param="tool_choice.function.name",
            )
        choice = ToolChoice("function", name)
    else:
        raise APIError(
            "tool_choice must be auto, none, required, or a function.",
            param="tool_choice",
        )
    if not tools and choice.mode in {"required", "function"}:
        raise APIError("tool_choice requires at least one tool.", param="tool_choice")
    return tools, choice


def _validate_function_name(value: Any, param: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value) is None:
        raise APIError(
            "Function name must use 1 to 64 letters, numbers, underscores, or hyphens.",
            param=param,
        )


def _assistant_tool_calls(value: Any, param: str) -> list[dict[str, Any]]:
    if value in (None, []):
        return []
    if not isinstance(value, list):
        raise APIError("tool_calls must be an array.", param=param)
    result = []
    ids = set()
    for index, raw in enumerate(value):
        item_param = f"{param}.{index}"
        if not isinstance(raw, dict) or raw.get("type") != "function":
            raise APIError("Each tool call must be a function.", param=item_param)
        call_id = raw.get("id")
        if not isinstance(call_id, str) or not call_id or call_id in ids:
            raise APIError("Tool call IDs must be unique strings.", param=f"{item_param}.id")
        function = raw.get("function")
        if not isinstance(function, dict):
            raise APIError("Tool call function must be an object.", param=f"{item_param}.function")
        name = function.get("name")
        _validate_function_name(name, f"{item_param}.function.name")
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            raise APIError(
                "Tool call arguments must be a JSON string.",
                param=f"{item_param}.function.arguments",
            )
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise APIError(
                "Tool call arguments must contain valid JSON.",
                param=f"{item_param}.function.arguments",
            ) from error
        if not isinstance(decoded, dict):
            raise APIError(
                "Tool call arguments must contain a JSON object.",
                param=f"{item_param}.function.arguments",
            )
        ids.add(call_id)
        result.append(raw)
    return result


def _messages(
    value: Any,
    *,
    allow_assistant_final: bool = False,
    content_parser=None,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise APIError("messages must be a non-empty array.", param="messages")
    result = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise APIError("Each message must be an object.", param=f"messages.{index}")
        role = raw.get("role")
        if role not in {"system", "developer", "user", "assistant", "tool"}:
            raise APIError(
                "Message role must be system, developer, user, assistant, or tool.",
                param=f"messages.{index}.role",
            )
        tool_calls = (
            _assistant_tool_calls(raw.get("tool_calls"), f"messages.{index}.tool_calls")
            if role == "assistant"
            else []
        )
        content_value = raw.get("content")
        content = (
            ""
            if role == "assistant" and content_value is None and tool_calls
            else (content_parser or _message_text)(content_value, f"messages.{index}.content")
        )
        message: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if role == "tool":
            call_id = raw.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                raise APIError(
                    "tool_call_id must be a non-empty string.",
                    param=f"messages.{index}.tool_call_id",
                )
            message["tool_call_id"] = call_id
        reasoning = raw.get("reasoning_content")
        if reasoning is not None:
            if role != "assistant" or not isinstance(reasoning, str):
                raise APIError(
                    "reasoning_content must be a string on an assistant message.",
                    param=f"messages.{index}.reasoning_content",
                )
            message["reasoning_content"] = reasoning
        result.append(message)

    pending_tool_ids: set[str] = set()
    for index, message in enumerate(result):
        role = message["role"]
        if role == "assistant":
            if pending_tool_ids:
                raise APIError(
                    "Provide every tool result before the next assistant message.",
                    param=f"messages.{index}",
                )
            pending_tool_ids = {
                call["id"] for call in message.get("tool_calls", [])
            }
        elif role == "tool":
            call_id = message["tool_call_id"]
            if call_id not in pending_tool_ids:
                raise APIError(
                    "tool_call_id must match the previous assistant tool call.",
                    param=f"messages.{index}.tool_call_id",
                )
            pending_tool_ids.remove(call_id)
        elif pending_tool_ids:
            raise APIError(
                "Provide every tool result before the next message.",
                param=f"messages.{index}",
            )
    if pending_tool_ids:
        raise APIError(
            "Provide every tool result before requesting another response.",
            param="messages",
        )
    allowed_final_roles = {"user", "developer", "tool"}
    if allow_assistant_final:
        allowed_final_roles.add("assistant")
    if result[-1]["role"] not in allowed_final_roles:
        raise APIError(
            "The final message must have the user, developer, or tool role.",
            param=f"messages.{len(result) - 1}.role",
        )
    return result


def _message_text(value: Any, param: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if not isinstance(item, dict) or item.get("type") not in {
                "text",
                "input_text",
                "output_text",
            }:
                raise APIError("Only text message content is supported.", param=param)
            text = item.get("text")
            if not isinstance(text, str):
                raise APIError("Message text must be a string.", param=param)
            parts.append(text)
        return "".join(parts)
    raise APIError("Message content must be text.", param=param)


def _collect(
    pieces: Iterator[GeneratedPiece],
    thinking: bool,
) -> tuple[str, str, int, int, str]:
    parser = ReasoningParser(thinking)
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    prompt_tokens = generated = 0
    finish = None
    for piece in pieces:
        prompt_tokens = piece.prompt_tokens
        generated = piece.generation_tokens
        finish = piece.finish_reason or finish
        reasoning, content = parser.feed(piece.text)
        reasoning_parts.append(reasoning)
        content_parts.append(content)
    reasoning, content = parser.finish()
    reasoning_parts.append(reasoning)
    content_parts.append(content)
    return (
        "".join(content_parts),
        "".join(reasoning_parts),
        prompt_tokens,
        generated,
        finish or "stop",
    )


def _collect_raw(
    pieces: Iterator[GeneratedPiece],
) -> tuple[str, int, int, str]:
    parts = []
    prompt_tokens = generated = 0
    finish = None
    for piece in pieces:
        parts.append(piece.text)
        prompt_tokens = piece.prompt_tokens
        generated = piece.generation_tokens
        finish = piece.finish_reason or finish
    return "".join(parts), prompt_tokens, generated, finish or "stop"


def _response_tool_calls(calls) -> list[dict[str, Any]]:
    return [
        {
            "id": "call_" + uuid.uuid4().hex,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.arguments,
            },
        }
        for call in calls
    ]


def _usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the installed model with an OpenAI-compatible API")
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--model")
    models.add_argument("--model-catalog")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11434)
    parser.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "error"),
        default="info",
    )
    parser.add_argument("--public-model")
    parser.add_argument("--slots", type=int, default=1_152)
    parser.add_argument("--read-workers", type=int, default=4)
    parser.add_argument("--prefetch-read-workers", type=int, default=2)
    parser.add_argument(
        "--power-saving-limit-gbps",
        type=float,
        choices=_POWER_SAVING_LIMITS_GBPS,
    )
    parser.add_argument(
        "--memory-limit-gib",
        type=int,
        default=0,
        help="MLX memory limit in GiB; 0 selects the model-safe automatic limit",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=0,
        help="prompt chunk size; 0 selects 128, 256, or 1,024 automatically",
    )
    parser.add_argument("--moe-prefill-step-size", type=int, default=0)
    parser.add_argument("--no-layer-major-prefill", action="store_true")
    parser.add_argument(
        "--layer-major-prefill-threshold",
        type=int,
        default=1_024,
        help="minimum uncached prompt tokens for DeepSeek layer-major prefill",
    )
    parser.add_argument("--no-batched-expert-prefill", action="store_true")
    parser.add_argument(
        "--qwen-grouped-experts", action=argparse.BooleanOptionalAction, default=True,
        help="group Qwen Prefill rows by expert (default: on when MTP is off)",
    )
    parser.add_argument(
        "--no-ane-prefill",
        action="store_true",
        help="use the original GPU Prefill path",
    )
    parser.add_argument("--ane-prefill-ratio", type=float, default=0.0)
    parser.add_argument("--prompt-cache-entries", type=int, default=2)
    parser.add_argument("--prompt-cache-memory-gib", type=int, default=8)
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument("--prompt-cache", choices=("off", "memory", "disk"),
                            help="reuse prompts: off, memory (default), or disk")
    cache_mode.add_argument("--persistent-prompt-cache", action=argparse.BooleanOptionalAction,
                            default=False, help="save prompt cache to disk (default: off)")
    parser.add_argument("--prompt-cache-directory")
    parser.add_argument("--warmup-prompt-file")
    parser.add_argument("--bf16-kv-cache", action="store_true")
    parser.add_argument("--no-fp4-index-cache", action="store_true")
    parser.add_argument("--no-ready-expert-decode", action="store_true")
    parser.add_argument(
        "--mtp",
        action="store_true",
        help="enable experimental Qwen MTP speculative decoding",
    )
    parser.add_argument("--mtp-slots", type=int, default=32)
    from .qwen_flash_config import add_flash_arguments
    add_flash_arguments(parser)
    parser.add_argument(
        "--expert-page-cache-probe",
        action="store_true",
        help=(
            "research-only pre-read mincore classification for expert-file "
            "page residency; not a physical SSD byte counter"
        ),
    )
    parser.add_argument(
        "--separate-prefill-io", action=argparse.BooleanOptionalAction, default=True,
        help="Use separate cache-bypassing Prefill reads (enabled by default).",
    )
    parser.add_argument(
        "--expert-file-cache-policy",
        choices=EXPERT_FILE_CACHE_POLICIES,
        default="cached",
        help=(
            "research-only expert descriptor policy; bypass uses Darwin "
            "F_NOCACHE with read-ahead disabled"
        ),
    )
    parser.add_argument(
        "--expert-eviction-policy", choices=("lfu", "lru", "route"), default="lfu",
        help="expert eviction ranking (default LFU)",
    )
    for name in ('qwen_quantized_kv', 'qwen_quantized_index', 'v41_packed_kv', 'v41_packed_index', 'v41_candidate_index', 'v41_ced_prefill', 'v41_next_layer_prefetch', 'deepseek_ane_prefill', 'v41_layer_major_prefill'):
        parser.add_argument("--" + name.replace("_", "-"), action=argparse.BooleanOptionalAction,
                            default=getattr(RuntimeConfig(), name))
    parser.add_argument("--dspark", action="store_true")
    parser.add_argument(
        "--dspark-prompt-cache",
        action="store_true",
        help="experimentally reuse an atomic target and DSpark prompt snapshot",
    )
    parser.add_argument(
        "--no-dspark-fallback",
        action="store_true",
        help="research only: continue DSpark after its wall-time stop gate",
    )
    parser.add_argument(
        "--dspark-sequential-verification",
        action="store_true",
        help="research oracle: verify each DSpark target position sequentially",
    )
    parser.add_argument("--dspark-slots", type=int, default=768)
    parser.add_argument("--dspark-confidence-threshold", type=float, default=0.6)
    parser.add_argument("--default-max-tokens", type=int, default=272_000)
    parser.add_argument("--default-temperature", type=float)
    parser.add_argument("--default-top-p", type=float)
    parser.add_argument("--default-top-k", type=int)
    return parser


def _explicit_arguments(argv: list[str]) -> set[str]:
    parser = _parser()
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    return set(vars(parser.parse_args(argv)))


def _catalog_overrides(specs, arguments, config, explicit):
    renamed = {
        "bf16_kv_cache": "fp8_kv_cache",
        "no_layer_major_prefill": "layer_major_prefill",
        "no_batched_expert_prefill": "batched_expert_prefill",
        "no_ane_prefill": "ane_prefill",
        "no_fp4_index_cache": "fp4_index_cache",
        "no_ready_expert_decode": "ready_expert_decode",
        "mtp": "mtp_enabled",
        "dspark": "dspark_enabled",
        "no_dspark_fallback": "dspark_fallback_enabled",
    }
    runtime = asdict(config)
    overrides = {
        renamed.get(name, name): runtime[renamed.get(name, name)]
        for name in explicit
        if renamed.get(name, name) in runtime
    }
    # An explicit legacy slot flag must override a catalog's memory budget too.
    for slot, budget in (("slots", "expert_cache_bytes"),
                         ("mtp_slots", "mtp_cache_bytes"),
                         ("dspark_slots", "dspark_cache_bytes")):
        if slot in explicit:
            overrides[budget] = None
    models = []
    for spec in specs:
        model = spec.to_json()
        merged = replace(spec.runtime, **overrides)
        model["runtime"] = asdict(_apply_prompt_cache_mode(merged, arguments.prompt_cache))
        for name in ("max_tokens", "temperature", "top_p", "top_k"):
            option = f"default_{name}"
            if option in explicit:
                model["defaults"][name] = getattr(arguments, option)
        models.append(model)
    return parse_model_catalog({"version": 1, "models": models})


def main() -> None:
    parser = _parser()
    arguments = parser.parse_args()
    if arguments.host not in {"127.0.0.1", "::1", "localhost"} and not arguments.api_key:
        parser.error("--api-key is required when --host is not local")
    if not 1 <= arguments.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    from .qwen_flash_config import flash_arguments
    config = RuntimeConfig(
        **flash_arguments(arguments),
        qwen_quantized_kv=arguments.qwen_quantized_kv,
        qwen_quantized_index=arguments.qwen_quantized_index,
        v41_packed_kv=arguments.v41_packed_kv,
        v41_packed_index=arguments.v41_packed_index,
        v41_candidate_index=arguments.v41_candidate_index,
        v41_ced_prefill=arguments.v41_ced_prefill,
        v41_next_layer_prefetch=arguments.v41_next_layer_prefetch,
        deepseek_ane_prefill=arguments.deepseek_ane_prefill,
        v41_layer_major_prefill=arguments.v41_layer_major_prefill,
        slots=arguments.slots,
        read_workers=arguments.read_workers,
        prefetch_read_workers=arguments.prefetch_read_workers,
        memory_limit_gib=arguments.memory_limit_gib,
        prefill_step_size=arguments.prefill_step_size,
        moe_prefill_step_size=arguments.moe_prefill_step_size,
        fp8_kv_cache=not arguments.bf16_kv_cache,
        layer_major_prefill=not arguments.no_layer_major_prefill,
        layer_major_prefill_threshold=arguments.layer_major_prefill_threshold,
        batched_expert_prefill=not arguments.no_batched_expert_prefill,
        qwen_grouped_experts=arguments.qwen_grouped_experts,
        ane_prefill=not arguments.no_ane_prefill,
        ane_prefill_ratio=arguments.ane_prefill_ratio,
        prompt_cache_entries=arguments.prompt_cache_entries,
        prompt_cache_memory_gib=arguments.prompt_cache_memory_gib,
        persistent_prompt_cache=arguments.persistent_prompt_cache,
        prompt_cache_directory=arguments.prompt_cache_directory,
        fp4_index_cache=not arguments.no_fp4_index_cache,
        expert_page_cache_probe=arguments.expert_page_cache_probe,
        expert_file_cache_policy=arguments.expert_file_cache_policy,
        separate_prefill_io=arguments.separate_prefill_io,
        expert_eviction_policy=arguments.expert_eviction_policy,
        ready_expert_decode=not arguments.no_ready_expert_decode,
        mtp_enabled=arguments.mtp,
        mtp_slots=arguments.mtp_slots,
        dspark_enabled=arguments.dspark,
        dspark_prompt_cache=arguments.dspark_prompt_cache,


        dspark_fallback_enabled=not arguments.no_dspark_fallback,
        dspark_sequential_verification=(
            arguments.dspark_sequential_verification
        ),

        dspark_slots=arguments.dspark_slots,
        dspark_confidence_threshold=arguments.dspark_confidence_threshold,
        power_saving_limit_gbps=arguments.power_saving_limit_gbps,
    )
    config = _apply_prompt_cache_mode(config, arguments.prompt_cache)
    if not arguments.model_catalog:
        try:
            validate_runtime_config(config)
        except ValueError as error:
            parser.error(str(error))

    if arguments.model_catalog and (
        arguments.public_model is not None or arguments.warmup_prompt_file is not None
    ):
        parser.error("--public-model and --warmup-prompt-file require --model")
    if arguments.public_model is not None and arguments.model is None:
        parser.error("--public-model requires --model")

    if arguments.model_catalog:
        try:
            model_specs = _catalog_overrides(
                load_model_catalog(arguments.model_catalog), arguments, config,
                _explicit_arguments(sys.argv[1:]),
            )
        except ModelCatalogError as error:
            parser.error(str(error))
    elif arguments.model:
        try:
            installed = InstalledModel.open(arguments.model)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))
        if installed.model_kind not in MODEL_IDS:
            parser.error(f"unsupported model kind: {installed.model_kind}")
        support = support_for_installed(installed)
        if not support.descriptor.supports("dspark") and arguments.dspark:
            parser.error(f"{support.option_error_name} does not support --dspark")
        if arguments.mtp and not support.descriptor.supports("mtp"):
            parser.error("--mtp is supported only by Qwen3.8-Flash-Next")
        if arguments.mtp and not installed.has_mtp:
            parser.error("--mtp requires an installed MTP sidecar")

        default_temperature = arguments.default_temperature
        default_top_p = arguments.default_top_p
        default_top_k = arguments.default_top_k
        if default_temperature is None:
            default_temperature = support.descriptor.defaults["temperature"]
        if default_top_p is None:
            default_top_p = support.descriptor.defaults["topP"]
        if default_top_k is None:
            default_top_k = support.descriptor.defaults["topK"]
        try:
            options = _options(
                {
                    "max_tokens": arguments.default_max_tokens,
                    "temperature": default_temperature,
                    "top_p": default_top_p,
                    "top_k": default_top_k,
                },
                ServerDefaults(),
                approximation_default=support.default_approximation(arguments.dspark),
                support=support,
            )
            model_specs = parse_model_catalog(
                {
                    "version": 1,
                    "models": [
                        {
                            "id": MODEL_IDS[installed.model_kind],
                            "alias": arguments.public_model,
                            "path": arguments.model,
                            "model_kind": installed.model_kind,
                            "runtime": asdict(config),
                            "defaults": {
                                "max_tokens": options.max_tokens,
                                "temperature": options.temperature,
                                "top_p": options.top_p,
                                "top_k": options.top_k,
                            },
                            "warmup_prompt_path": arguments.warmup_prompt_file,
                        }
                    ],
                }
            )
        except (APIError, ModelCatalogError) as error:
            parser.error(str(error))
    else:
        model_specs = ()

    model_manager = ModelManager(model_specs)
    server = None
    try:
        server = OpenAIServer(
            (arguments.host, arguments.port),
            model_manager,
            api_key=arguments.api_key,
            log_level=arguments.log_level,
        )
        print(f"Ready: http://{arguments.host}:{server.server_port}", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        if server is not None:
            server.server_close()
        model_manager.close()


if __name__ == "__main__":
    main()
