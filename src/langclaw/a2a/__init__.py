"""Agent2Agent (A2A) protocol integration for LangClaw.

Install server/client pieces:

    pip install -e ".[a2a]"

Serve this agent to peers::

    uv run python run_a2a.py

Environment:

- ``LANGCLAW_A2A_PUBLIC_URL``: Agent card ``url`` (default: ``http://{host}:{port}``).

Call a remote A2A agent from the LangClaw tool loop via :func:`langclaw.tools.a2a_client.a2a_send`.
"""

from langclaw.a2a.app import create_langclaw_a2a_app
from langclaw.a2a.card import build_langclaw_agent_card
from langclaw.a2a.executor import LangclawA2AExecutor

__all__ = [
    "LangclawA2AExecutor",
    "build_langclaw_agent_card",
    "create_langclaw_a2a_app",
]
