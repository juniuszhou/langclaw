"""LangChain tool: call a remote Agent2Agent (A2A) peer via JSON-RPC ``message/send``."""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

from langchain_core.tools import tool


def _format_a2a_client_result(result: Any) -> str:
    from a2a.types import Message, Role, Task
    from a2a.utils.message import get_message_text

    if isinstance(result, Message):
        return get_message_text(result)
    if isinstance(result, tuple) and result:
        first = result[0]
        if isinstance(first, Task):
            if first.history:
                for m in reversed(first.history):
                    if m.role == Role.agent:
                        return get_message_text(m)
            sm = first.status.message
            if sm is not None:
                return get_message_text(sm)
            return str(first.status.state)
        return str(first)
    if result is None:
        return "(no response)"
    return str(result)


async def _a2a_send_async(
    agent_base_url: str, message: str, context_id: str
) -> str:
    import httpx
    from a2a.client.card_resolver import A2ACardResolver
    from a2a.client.client import ClientConfig
    from a2a.client.client_factory import ClientFactory
    from a2a.types import Message as A2AMessage
    from a2a.types import MessageSendConfiguration, Part, Role, TextPart

    base = agent_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=120.0) as client:
        resolver = A2ACardResolver(client, base)
        card = await resolver.get_agent_card()
        factory = ClientFactory(
            ClientConfig(streaming=False, polling=False, httpx_client=client)
        )
        a2a_client = factory.create(card)
        ctx = context_id.strip() or None
        payload = A2AMessage(
            role=Role.user,
            parts=[Part(root=TextPart(text=message))],
            message_id=str(uuid.uuid4()),
            context_id=ctx,
        )
        final: Any = None
        async for event in a2a_client.send_message(
            payload, configuration=MessageSendConfiguration(blocking=True)
        ):
            final = event
        return _format_a2a_client_result(final)


def send_a2a_message(agent_base_url: str, message: str, context_id: str = "") -> str:
    """Call a remote A2A agent and return the final text reply (sync)."""
    try:
        import a2a  # noqa: F401
    except ImportError:
        return (
            "Error: A2A client needs optional dependencies. "
            "Install with: pip install -e '.[a2a]'"
        )
    try:
        return asyncio.run(_a2a_send_async(agent_base_url, message, context_id))
    except Exception as e:
        return f"Error: {e}"


@tool
def a2a_send(agent_base_url: str, message: str, context_id: str = "") -> str:
    """Send a user message to another A2A agent and return its final text reply.

    ``agent_base_url`` is the peer's HTTP origin (e.g. ``http://127.0.0.1:9999``).
    The client fetches ``/.well-known/agent-card.json``, then calls ``message/send``
    on the advertised JSON-RPC URL.

    Reuse the same ``context_id`` string across calls to continue a remote conversation
    when the peer supports task history (optional; leave empty for a fresh context).

    Requires: ``pip install -e ".[a2a]"``

    Args:
        agent_base_url: Base URL of the remote A2A server (no trailing slash).
        message: Text instruction or question for the remote agent.
        context_id: Optional opaque id to correlate turns with the same remote session.

    Returns:
        The remote agent's reply text, or an error string.
    """
    return send_a2a_message(agent_base_url, message, context_id)
