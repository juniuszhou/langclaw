"""A2A (Agent2Agent) integration tests."""

import asyncio
import uuid
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from langclaw.a2a.card import build_langclaw_agent_card
from langclaw.a2a.executor import LangclawA2AExecutor
from langclaw.runtime import AgentRuntime


def test_build_langclaw_agent_card():
    card = build_langclaw_agent_card(
        public_url="http://127.0.0.1:9999",
        name="Test Agent",
        description="Does testing.",
        version="1.2.3",
    )
    assert card.name == "Test Agent"
    assert card.version == "1.2.3"
    assert card.url == "http://127.0.0.1:9999"


def test_langclaw_a2a_executor_invokes_runtime():
    pytest.importorskip("a2a.types")
    from a2a.server.agent_execution import RequestContext
    from a2a.types import Message as A2AMessage
    from a2a.types import MessageSendParams, Part, Role, TextPart

    runtime = MagicMock(spec=AgentRuntime)
    runtime.invoke.return_value = {"messages": [AIMessage(content="hello from agent")]}

    task_id = str(uuid.uuid4())
    context_id = str(uuid.uuid4())
    user_msg = A2AMessage(
        role=Role.user,
        parts=[Part(root=TextPart(text="ping"))],
        message_id=str(uuid.uuid4()),
        task_id=task_id,
        context_id=context_id,
    )
    mparams = MessageSendParams(message=user_msg)
    ctx = RequestContext(request=mparams, task_id=task_id, context_id=context_id)

    class FakeQueue:
        def __init__(self):
            self.events = []

        async def enqueue_event(self, e):
            self.events.append(e)

    executor = LangclawA2AExecutor(runtime, memory=None)
    q = FakeQueue()
    asyncio.run(executor.execute(ctx, q))  # type: ignore[arg-type]

    runtime.invoke.assert_called_once()
    assert q.events
