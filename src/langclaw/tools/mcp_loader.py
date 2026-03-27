"""Load tools from MCP (Model Context Protocol) servers."""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool


async def load_mcp_tools_async(
    server_command: str = "npx",
    server_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> List[BaseTool]:
    """Load tools from an MCP server via stdio.

    Args:
        server_command: Command to run (e.g. npx, python).
        server_args: Arguments (e.g. ["firecrawl-mcp"]).
        env: Environment variables to pass.

    Returns:
        List of LangChain tools from the MCP server.
    """
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from langchain_mcp_adapters.tools import load_mcp_tools
    except ImportError as e:
        raise ImportError(
            "MCP support requires: pip install mcp langchain-mcp-adapters"
        ) from e

    server_args = server_args or []
    env = dict(env) if env else {}
    for key in ("FIRECRAWL_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if key in os.environ and key not in env:
            env[key] = os.environ[key]

    params = StdioServerParameters(
        command=server_command,
        args=server_args,
        env=env,
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            return list(tools)


def load_mcp_tools(
    server_command: str = "npx",
    server_args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> List[BaseTool]:
    """Synchronous wrapper for load_mcp_tools_async."""
    return asyncio.run(load_mcp_tools_async(server_command, server_args, env))


def load_mcp_tools_from_config(config: Dict[str, Any]) -> List[BaseTool]:
    """Load MCP tools from a config dict.

    Config format:
        mcp:
          server: npx
          args: [firecrawl-mcp]
          env:
            FIRECRAWL_API_KEY: ${FIRECRAWL_API_KEY}
    """
    mcp = config.get("mcp") or {}
    if not mcp:
        return []
    server = mcp.get("server", "npx")
    args = mcp.get("args") or []
    env = mcp.get("env") or {}
    env = {k: os.path.expandvars(str(v)) for k, v in env.items()}
    return load_mcp_tools(server_command=server, server_args=args, env=env)
