"""Tool registry, built-in tools, skills, and MCP."""

from langclaw.tools.a2a_peers import load_a2a_peer_tools, make_a2a_peer_tool
from langclaw.tools.builtin import BUILTIN_TOOLS
from langclaw.tools.registry import ToolRegistry, get_builtin_tools, get_tools
from langclaw.tools.skills import Skill, load_skills, build_skills_prompt
from langclaw.tools.mcp_loader import load_mcp_tools, load_mcp_tools_from_config

__all__ = [
    "BUILTIN_TOOLS",
    "load_a2a_peer_tools",
    "make_a2a_peer_tool",
    "ToolRegistry",
    "get_builtin_tools",
    "get_tools",
    "Skill",
    "load_skills",
    "build_skills_prompt",
    "load_mcp_tools",
    "load_mcp_tools_from_config",
]
