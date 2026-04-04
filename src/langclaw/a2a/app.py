"""Assemble Starlette app + A2A JSON-RPC handler for LangClaw."""

from __future__ import annotations

from typing import TYPE_CHECKING

from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCard
from starlette.applications import Starlette

from langclaw.a2a.executor import LangclawA2AExecutor

if TYPE_CHECKING:
    from langclaw.memory.sqlite import SqliteMemory
    from langclaw.runtime import AgentRuntime


def create_langclaw_a2a_app(
    *,
    agent_card: AgentCard,
    runtime: AgentRuntime,
    memory: SqliteMemory | None = None,
) -> Starlette:
    """Build a Starlette app with ``/.well-known/agent-card.json`` and JSON-RPC ``POST /``."""

    executor = LangclawA2AExecutor(runtime, memory=memory)
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )
    builder = A2AStarletteApplication(agent_card=agent_card, http_handler=handler)
    return builder.build()
