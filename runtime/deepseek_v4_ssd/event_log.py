"""Shared server/runtime events. Request state stays on the request's own context."""
from __future__ import annotations

import logging
import re
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class RequestLog:
    id: str
    level: str
    secret: str = ""
    model: str = "-"
    prompt_tokens: int = 0
    output_tokens: int = 0
    finish: str = "unknown"
    outcome: str = "completed"
    generating: bool = False


request_context: ContextVar[RequestLog | None] = ContextVar("request_log", default=None)


class EventHandler(logging.Handler):
    def emit(self, record):
        context = request_context.get()
        minimum = getattr(logging, (context.level if context else default_level).upper())
        if record.levelno < minimum:
            return
        message = record.getMessage()
        if context and context.secret:
            message = message.replace(context.secret, "[redacted]")
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        level = "ERROR" if record.levelno >= logging.ERROR else "INFO" if record.levelno >= logging.INFO else "DEBUG"
        prefix = f"{timestamp} [{level}] request={context.id if context else '-'} "
        # Prefix every line, including stack frames; no locals or request bodies.
        try:
            for line in message.splitlines() or [""]:
                sys.stderr.write(prefix + line + "\n")
            sys.stderr.flush()
        except OSError:
            pass  # A closed diagnostic pipe must not fail a generation request.


default_level = "info"
logger = logging.getLogger("whallm.events")
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.addHandler(EventHandler())


def configure(level: str) -> None:
    global default_level
    default_level = level


@contextmanager
def request_scope(level: str, secret: str = ""):
    context = RequestLog("req-" + uuid.uuid4().hex, level, secret)
    token = request_context.set(context)
    started = time.monotonic()
    try:
        yield context
    except Exception:
        context.outcome = "failed"
        context.finish = "error"
        raise
    finally:
        if context.generating:
            logger.log(logging.ERROR if context.outcome == "failed" else logging.INFO,
                       "generation %s model=%s seconds=%.3f input_tokens=%d output_tokens=%d finish=%s",
                       context.outcome, context.model, time.monotonic() - started,
                       context.prompt_tokens, context.output_tokens, context.finish)
        request_context.reset(token)


def failure(event: str, error: BaseException) -> None:
    # Exception messages may embed a prompt, URL credentials or arbitrary tool output.
    # Keep type, errno and stack locations, not exception values or local variables.
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        logger.error("%s error=%s errno=%s", event, type(error).__name__, getattr(error, "errno", "-"))
        logger.error("Traceback (most recent call last):")
        tb = error.__traceback__
        while tb is not None:
            code = tb.tb_frame.f_code
            logger.error("  at %s:%d in %s", code.co_filename, tb.tb_lineno, code.co_name)
            tb = tb.tb_next
        # Preserve our numeric diagnostic notes, never arbitrary exception text.
        for note in getattr(error, "__notes__", ()):
            if re.fullmatch(r"generation: prompt_tokens=\d+ generated_tokens=\d+ prefill_step_size=\d+", note):
                logger.error("%s", note)
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
        event = "caused by"
