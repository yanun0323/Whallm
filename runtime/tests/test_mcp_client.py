import asyncio
import json
import os
from pathlib import Path
import sys
import socket
import tempfile
import unittest
from unittest.mock import patch

from deepseek_v4_ssd.mcp_client import MCPClient, MCPServer, _schema, _validate_arguments


class MCPClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_stdio_requires_approval_and_never_replays_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "calls"
            server = MCPServer("fixture", command=(sys.executable, str(Path(__file__).parent / "fixtures/mcp_server.py"), str(log)))
            with patch.dict(os.environ, {"WHALLM_TEST_SECRET": "must-not-be-forwarded"}):
                async with MCPClient(server, timeout=5) as client:
                    tools = client.tools()
                    self.assertEqual(len(tools), 2)
                    name = next(t["function"]["name"] for t in tools if "value" in t["function"]["parameters"]["properties"])
                    self.assertRegex(name, r"^mcp_[a-f0-9]{40}$")
                    proposal = await client.propose(name, {"value": "hello"})
                    self.assertEqual(proposal.display_name, "echo")
                    self.assertFalse(log.exists())
                    declined = await client.resolve(proposal.id, approve=False)
                    self.assertTrue(declined["isError"])
                    self.assertFalse(log.exists())
                    proposal = await client.propose(name, {"value": "hello"})
                    result = await client.resolve(proposal.id, approve=True)
                    self.assertEqual(log.read_text(), "called\n")
                    payload = json.loads(result["content"][0]["text"])
                    self.assertEqual(payload["value"], "hello")
                    self.assertFalse(payload["credential_forwarded"])
                    with self.assertRaises(ValueError): await client.resolve(proposal.id, approve=True)
                    with self.assertRaises(ValueError): await client.propose(name, {"value": 123})
                    proposal = await client.propose(name, {"value": "changed"})
                    listing = await client.session.list_tools()
                    next(t for t in listing.tools if t.name == "echo").description = "Changed behavior"
                    with patch.object(client.session, "list_tools", return_value=listing):
                        with self.assertRaisesRegex(ValueError, "declaration changed"):
                            await client.resolve(proposal.id, approve=True)
                    self.assertEqual(log.read_text(), "called\n")

    async def test_cancellation_consumes_approval_and_connection_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            server = MCPServer("fixture", command=(sys.executable, str(Path(__file__).parent / "fixtures/mcp_server.py"), str(Path(directory)/"calls")))
            async with MCPClient(server, timeout=5) as client:
                name = next(t["function"]["name"] for t in client.tools() if "seconds" in t["function"]["parameters"]["properties"])
                proposal = await client.propose(name, {"seconds": 3})
                task = asyncio.create_task(client.resolve(proposal.id, approve=True))
                await asyncio.sleep(.1)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError): await task
                with self.assertRaises(ValueError): await client.resolve(proposal.id, approve=True)
            with self.assertRaises(ValueError): await client.refresh_tools()

    async def test_streamable_http_uses_the_same_explicit_approval_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            log = Path(directory) / "calls"
            process = await asyncio.create_subprocess_exec(sys.executable,
                str(Path(__file__).parent / "fixtures/mcp_server.py"), str(log), str(port),
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            try:
                for _ in range(100):
                    try:
                        reader, writer = await asyncio.open_connection("127.0.0.1", port)
                        writer.close(); await writer.wait_closed()
                        break
                    except OSError:
                        await asyncio.sleep(.05)
                async with MCPClient(MCPServer("http-fixture", url=f"http://127.0.0.1:{port}/mcp"), timeout=5) as client:
                    name = next(t["function"]["name"] for t in client.tools() if "value" in t["function"]["parameters"]["properties"])
                    proposal = await client.propose(name, {"value": "http"})
                    self.assertFalse(log.exists())
                    result = await client.resolve(proposal.id, approve=True)
                    self.assertEqual(json.loads(result["content"][0]["text"])["value"], "http")
                    self.assertEqual(log.read_text(), "called\n")
            finally:
                if process.returncode is None: process.terminate()
                await asyncio.wait_for(process.wait(), timeout=10)

    async def test_configuration_and_schema_fail_closed(self):
        for configuration in ({"name": "x"}, {"name": "x", "command": "sh"},
                              {"name": "x", "url": "http://example.com/mcp"},
                              {"name": "x", "url": "https://user:password@example.com/mcp"}):
            with self.subTest(configuration=configuration), self.assertRaises(ValueError): MCPServer(**configuration)
        for schema in ({"$ref": "https://example.com/schema"}, {"$dynamicRef": "file:///private/file"}):
            with self.assertRaises(ValueError): _schema(schema)
        self.assertIsInstance(_schema({"type": "object"}), str)
        await _validate_arguments({"$defs": {"value": {"type": "integer"}}, "type": "object",
            "properties": {"value": {"$ref": "#/$defs/value"}}}, {"value": 3})
