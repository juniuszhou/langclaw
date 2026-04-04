"""Register configured A2A peers as named LangChain tools."""

from __future__ import annotations

from typing import List

from langchain_core.tools import BaseTool, StructuredTool

from langclaw.config.loader import A2APeerConfig
from langclaw.tools.a2a_client import send_a2a_message
from langclaw.tools.builtin import BUILTIN_TOOLS


def fetch_peer_card_description(base_url: str) -> str | None:
    """Best-effort GET of ``/.well-known/agent-card.json`` for tool description text."""
    try:
        import httpx
        from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH
    except ImportError:
        return None
    root = base_url.rstrip("/")
    path = AGENT_CARD_WELL_KNOWN_PATH
    if not path.startswith("/"):
        path = "/" + path
    url = f"{root}{path}"
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(url)
            r.raise_for_status()
            data = r.json()
    except Exception:
        return None
    name = data.get("name") or "Agent"
    desc = (data.get("description") or "").strip()
    lines: list[str] = []
    if desc:
        lines.append(f"{name}: {desc}")
    else:
        lines.append(name)
    skills = data.get("skills") or []
    if skills:
        lines.append("Capabilities (from agent card):")
        for s in skills:
            sid = s.get("id") or ""
            sn = s.get("name") or sid
            sd = (s.get("description") or "").strip()
            lines.append(f"- {sn}: {sd}" if sd else f"- {sn}")
    return "\n".join(lines).strip() or None


def make_a2a_peer_tool(peer: A2APeerConfig) -> BaseTool:
    """Build a LangChain tool that calls one A2A peer by fixed ``base_url``."""
    base = peer.base_url.rstrip("/")
    description = (
        peer.description.strip()
        if peer.description
        else fetch_peer_card_description(base)
    )
    if not description:
        description = (
            f"Remote A2A agent at {base}. "
            "Send a natural-language message; returns the agent's final text reply. "
            "Optional context_id continues a remote session when supported."
        )

    def _call(message: str, context_id: str = "") -> str:
        return send_a2a_message(base, message, context_id)

    return StructuredTool.from_function(
        name=peer.tool_name,
        description=description,
        func=_call,
    )


def load_a2a_peer_tools(peers: List[A2APeerConfig]) -> List[BaseTool]:
    """Create one StructuredTool per configured peer."""
    builtin_names = {t.name for t in BUILTIN_TOOLS}
    tools: List[BaseTool] = []
    seen: set[str] = set()
    for p in peers:
        if p.tool_name in builtin_names:
            raise ValueError(
                f"a2a_peers.tool_name {p.tool_name!r} conflicts with a built-in tool; "
                "choose a different name."
            )
        if p.tool_name in seen:
            raise ValueError(
                f"Duplicate a2a_peers.tool_name {p.tool_name!r}; names must be unique."
            )
        seen.add(p.tool_name)
        tools.append(make_a2a_peer_tool(p))
    return tools
