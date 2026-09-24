"""Local official-SDK fixture. No production services or external credentials."""
import asyncio
import os
from pathlib import Path
import sys
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

server = FastMCP("Whallm MCP test", log_level="ERROR", port=int(sys.argv[2]) if len(sys.argv) > 2 else 8000)
log = Path(sys.argv[1])


@server.tool(annotations=ToolAnnotations(readOnlyHint=True))
def echo(value: str) -> dict:
    # Deliberately side-effecting despite its annotation: approval is mandatory.
    with log.open("a") as file:
        file.write("called\n")
    return {"value": value, "credential_forwarded": "WHALLM_TEST_SECRET" in os.environ}


@server.tool()
async def slow(seconds: float) -> str:
    await asyncio.sleep(seconds)
    return "finished"


server.run(transport="streamable-http" if len(sys.argv) > 2 else "stdio")
