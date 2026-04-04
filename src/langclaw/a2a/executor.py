"""Bridge LangClaw :class:`AgentRuntime` to the A2A :class:`AgentExecutor` interface."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.utils.message import new_agent_text_message
from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from langclaw.memory.sqlite import SqliteMemory
    from langclaw.runtime import AgentRuntime

logger = logging.getLogger(__name__)


class LangclawA2AExecutor(AgentExecutor):
    """Runs one A2A task turn via :meth:`AgentRuntime.invoke`."""

    def __init__(
        self,
        runtime: AgentRuntime,
        memory: SqliteMemory | None = None,
        thread_prefix: str = "a2a:",
    ) -> None:
        self._runtime = runtime
        self._memory = memory
        self._thread_prefix = thread_prefix

    def _thread_id(self, context_id: str) -> str:
        return f"{self._thread_prefix}{context_id}"

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id
        context_id = context.context_id
        if not task_id or not context_id:
            logger.error("A2A execute missing task_id or context_id")
            return

        user_text = context.get_user_input()
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.start_work()

        thread = self._thread_id(context_id)
        cfg: dict = {"configurable": {"thread_id": thread}}

        try:
            if self._memory:
                history = self._memory.load_messages(thread, limit=50)
                input_messages = history + [HumanMessage(content=user_text)]
            else:
                input_messages = [HumanMessage(content=user_text)]

            result = await asyncio.to_thread(
                self._runtime.invoke,
                {"messages": input_messages},
                cfg,
            )
            last = result["messages"][-1]
            content = getattr(last, "content", str(last)) or "(no response)"

            if self._memory:
                self._memory.append_messages(
                    thread,
                    [HumanMessage(content=user_text), last],
                )

            reply = new_agent_text_message(
                str(content), context_id=context_id, task_id=task_id
            )
            await updater.complete(message=reply)
        except Exception as e:
            logger.exception("A2A execute failed")
            err = new_agent_text_message(
                f"Error: {e}", context_id=context_id, task_id=task_id
            )
            await updater.failed(message=err)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        tid = context.task_id or ""
        cid = context.context_id or ""
        updater = TaskUpdater(event_queue, tid, cid)
        await updater.cancel()
