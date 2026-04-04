"""Tool registry for loading and resolving tools by name."""

from pathlib import Path
from typing import Any, Callable, List, Optional

from langchain_core.tools import BaseTool

from langclaw.config.loader import A2APeerConfig
from langclaw.tools.builtin import BUILTIN_TOOLS
from langclaw.tools.mcp_loader import load_mcp_tools_from_config

# Map tool name -> tool instance
_BUILTIN_MAP = {t.name: t for t in BUILTIN_TOOLS}

# Populated by get_tools() from config ``a2a_peers`` (named A2A agent tools).
_PEER_TOOLS: dict[str, BaseTool] = {}


def _set_a2a_peer_tools(peer_tools: List[BaseTool]) -> None:
    """Replace the peer-tool map so :meth:`ToolRegistry.get` resolves configured agents."""
    _PEER_TOOLS.clear()
    for t in peer_tools:
        _PEER_TOOLS[t.name] = t


def get_tools(
    tool_names: Optional[List[str]] = None,
    mcp_config: Optional[dict] = None,
    skills_root: Optional[Path] = None,
    skill_names: Optional[List[str]] = None,
    a2a_peers: Optional[List[A2APeerConfig]] = None,
) -> List[BaseTool]:
    """Get tools: built-in by name + registered A2A peers + optional MCP tools.

    Args:
        tool_names: Built-in tool names. If None, use all built-in.
        mcp_config: Optional MCP config dict to load additional tools.
        skills_root: Unused here; skills affect system prompt, not tools.
        skill_names: Unused here.
        a2a_peers: Optional A2A peer configs; each becomes a named tool (also listed in tool_names).

    Returns:
        Combined list of tools.
    """
    from langclaw.tools.a2a_peers import load_a2a_peer_tools

    peer_list = a2a_peers or []
    peer_objs = load_a2a_peer_tools(peer_list)
    _set_a2a_peer_tools(peer_objs)

    if tool_names:
        tools = get_builtin_tools(tool_names)
    else:
        tools = list(BUILTIN_TOOLS) + peer_objs
    if mcp_config:
        try:
            mcp_tools = load_mcp_tools_from_config({"mcp": mcp_config})
            tools.extend(mcp_tools)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load MCP tools: {e}")
    return tools


class ToolRegistry:
    """Registry for tools. Resolves by name and supports custom tools."""

    def __init__(self):
        self._custom: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a custom tool."""
        self._custom[tool.name] = tool

    def register_a2a_peer_tool(self, tool: BaseTool) -> None:
        """Register an A2A peer tool on the global peer map (same as config ``a2a_peers``)."""
        _PEER_TOOLS[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """Get tool by name."""
        return (
            self._custom.get(name)
            or _PEER_TOOLS.get(name)
            or _BUILTIN_MAP.get(name)
        )

    def get_many(self, names: List[str]) -> List[BaseTool]:
        """Get multiple tools by name. Skips unknown names."""
        tools = []
        for name in names:
            t = self.get(name)
            if t:
                tools.append(t)
        return tools

    def list_builtin(self) -> List[str]:
        """List all built-in tool names."""
        return list(_BUILTIN_MAP.keys())


def get_builtin_tools(names: Optional[List[str]] = None) -> List[BaseTool]:
    """Get built-in tools. If names is None, return all."""
    if names is None:
        return BUILTIN_TOOLS.copy()
    registry = ToolRegistry()
    return registry.get_many(names)
