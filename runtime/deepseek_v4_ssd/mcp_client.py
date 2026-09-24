"""Explicitly configured MCP clients; ordinary inference APIs never execute tools.

Use in a dedicated owner task/process. No shell, implicit credentials, auto approval
or retry of side-effecting calls. App broker/Keychain/OAuth integration is separate.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import os
import secrets
import sys
import time
from urllib.parse import urlsplit

MAX_RESULT_BYTES = 1_048_576


@dataclass(frozen=True)
class MCPServer:
    name: str
    command: tuple[str, ...] = ()
    url: str | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not 1 <= len(self.name) <= 64 or not isinstance(self.command, tuple) or bool(self.command) == bool(self.url):
            raise ValueError("Configure exactly one MCP transport")
        if self.url:
            url = urlsplit(self.url)
            if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.fragment:
                raise ValueError("Use an explicit HTTP(S) MCP endpoint without embedded credentials")
            if url.scheme == "http" and url.hostname not in ("localhost", "127.0.0.1", "::1"):
                raise ValueError("Remote MCP endpoints require HTTPS")
        if self.command and any(not isinstance(arg, str) or not arg or "\0" in arg for arg in self.command):
            raise ValueError("MCP command must be a non-empty argv array")


@dataclass(frozen=True)
class ToolProposal:
    id: str
    server: str
    tool: str
    display_name: str
    arguments_json: str
    declaration_digest: str
    expires_at: float


def _json(value, limit):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(text.encode()) > limit:
        raise ValueError("MCP data exceeds its byte limit")
    return text


def _schema(schema):
    text = _json(schema, 128 * 1024)
    def visit(value, depth=0):
        if depth > 32:
            raise ValueError("MCP schema nesting is too deep")
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("$ref", "$dynamicRef", "$recursiveRef") and (not isinstance(item, str) or not item.startswith("#")):
                    raise ValueError("External JSON Schema references are disabled")
                visit(item, depth + 1)
        elif isinstance(value, list):
            for item in value: visit(item, depth + 1)
    visit(schema)
    return text


async def _validate_arguments(schema, arguments):
    # Isolate schema regex/recursion CPU work; never let a tool's schema block
    # the generation owner or resolve network/file references.
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "TMPDIR", "PYTHONPATH") if k in os.environ}
    process = await asyncio.create_subprocess_exec(sys.executable, "-m", __name__, "validate", env=env,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
    try:
        payload = _json({"schema": schema, "arguments": arguments}, 256 * 1024).encode()
        async with asyncio.timeout(5):
            stdout, _ = await process.communicate(payload)
        if process.returncode != 0 or stdout.strip() != b"valid":
            raise ValueError("Tool arguments do not satisfy the declared schema")
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


class MCPClient:
    def __init__(self, server: MCPServer, *, timeout=30, approval_ttl=120):
        self.server, self.timeout, self.approval_ttl = server, timeout, approval_ttl
        self._stack = AsyncExitStack()
        self._pending = {}
        self._tools = {}
        self._lock = asyncio.Lock()
        self._validation_slots = asyncio.Semaphore(2)
        self._closed = False
        self._entered = False

    async def __aenter__(self):
        if self._entered or self._closed:
            raise ValueError("MCP client cannot be entered twice")
        self._entered = True
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client
        try:
            if self.server.command:
                env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LANG", "TMPDIR") if k in os.environ}
                transport = stdio_client(StdioServerParameters(command=self.server.command[0],
                    args=list(self.server.command[1:]), env=env))
            else:
                import httpx
                http = await self._stack.enter_async_context(httpx.AsyncClient(
                    timeout=self.timeout, follow_redirects=False, trust_env=False))
                transport = streamable_http_client(self.server.url, http_client=http)
            streams = await self._stack.enter_async_context(transport)
            self.session = await self._stack.enter_async_context(ClientSession(
                streams[0], streams[1], read_timeout_seconds=timedelta(seconds=self.timeout)))
            self.initialization = await self.session.initialize()
            await self.refresh_tools()
            return self
        except BaseException:
            await self._stack.aclose()
            raise

    async def __aexit__(self, *exc):
        self._closed = True
        self._pending.clear()
        await self._stack.aclose()

    async def refresh_tools(self):
        if self._closed or not hasattr(self, "initialization"):
            raise ValueError("MCP client is closed or not initialized")
        tools, cursor, seen = {}, None, set()
        if self.initialization.capabilities.tools is None:
            self._tools = {}
            return []
        for _ in range(16):
            result = await self.session.list_tools(cursor=cursor)
            for tool in result.tools:
                if len(tools) >= 128:
                    raise ValueError("MCP tool inventory exceeds 128 tools")
                schema = _schema(tool.inputSchema)
                name = "mcp_" + hashlib.sha256((self.server.name + "\0" + tool.name).encode()).hexdigest()[:40]
                if name in tools:
                    raise ValueError("MCP tool namespace collision")
                description = tool.description or ""
                if len(description.encode()) > 8192:
                    raise ValueError("MCP tool description exceeds its byte limit")
                tools[name] = {"name": tool.name, "description": description, "schema": schema,
                               "digest": hashlib.sha256(_json(tool.model_dump(mode="json", exclude_none=True), 256 * 1024).encode()).hexdigest()}
            cursor = result.nextCursor
            if cursor is None:
                self._tools = tools
                return self.tools()
            if cursor in seen:
                raise ValueError("MCP pagination repeated a cursor")
            seen.add(cursor)
        raise ValueError("MCP pagination exceeds its limit")

    def tools(self):
        return [{"type": "function", "function": {"name": name, "description": t["description"],
                 "parameters": json.loads(t["schema"])}} for name, t in self._tools.items()]

    async def propose(self, name, arguments):
        if self._closed or name not in self._tools or not isinstance(arguments, dict):
            raise ValueError("Unknown or unavailable MCP tool")
        now = time.monotonic()
        self._pending = {k: v for k, v in self._pending.items() if v.expires_at > now}
        if len(self._pending) >= 16:
            raise ValueError("Too many pending tool approvals")
        tool = self._tools[name]
        text = _json(arguments, 64 * 1024)
        async with self._validation_slots:
            if self._closed or len(self._pending) >= 16:
                raise ValueError("MCP client unavailable or approval queue full")
            await _validate_arguments(json.loads(tool["schema"]), json.loads(text))
        if self._closed or len(self._pending) >= 16:
            raise ValueError("MCP client unavailable or approval queue full")
        proposal = ToolProposal(secrets.token_urlsafe(24), self.server.name, name, tool["name"], text, tool["digest"], now + self.approval_ttl)
        self._pending[proposal.id] = proposal
        return proposal

    async def resolve(self, proposal_id, *, approve: bool):
        if type(approve) is not bool:
            raise ValueError("Approval must be an explicit boolean")
        proposal = self._pending.pop(proposal_id, None)
        if self._closed or proposal is None or proposal.expires_at <= time.monotonic():
            raise ValueError("Tool approval is missing, consumed or expired")
        if not approve:
            return {"isError": True, "content": [{"type": "text", "text": "The user declined this tool call."}]}
        # Consume before the first await: cancelling or timing out cannot make
        # this approval replayable. A remote side effect cannot be rolled back.
        async with self._lock:
            await self.refresh_tools()
            tool = self._tools.get(proposal.tool)
            if proposal.expires_at <= time.monotonic() or tool is None or tool["digest"] != proposal.declaration_digest:
                raise ValueError("Tool declaration changed or approval expired; request a new approval")
            result = await self.session.call_tool(tool["name"], json.loads(proposal.arguments_json),
                read_timeout_seconds=timedelta(seconds=self.timeout))
            data = result.model_dump(mode="json", by_alias=True, exclude_none=True)
            _json(data, MAX_RESULT_BYTES)
            # Keep typed text/image/audio/resource content, never flatten media
            # into a fabricated textual observation. Callers must apply their
            # own media capabilities and reject unsupported results explicitly.
            return data


def _validate_worker():
    import jsonschema
    from referencing import Registry
    raw = sys.stdin.buffer.read(256 * 1024 + 1)
    if len(raw) > 256 * 1024:
        raise ValueError("oversized schema validation request")
    request = json.loads(raw)
    _schema(request["schema"])
    validator = jsonschema.validators.validator_for(request["schema"])
    validator.check_schema(request["schema"])
    validator(request["schema"], registry=Registry()).validate(request["arguments"])
    print("valid")


if __name__ == "__main__":
    if sys.argv[1:] != ["validate"]:
        raise SystemExit("This module is an MCP client, not an unauthenticated tool-execution server")
    _validate_worker()
