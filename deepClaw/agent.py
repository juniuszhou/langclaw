"""DeepClaw: multi-subagent Deep Agent (LangGraph deepagents) with orchestrator + specialists."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, cast

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from .subagents import build_specialists
from .tools import fetch_url

# Shared LangClaw builtins (same stack as the rest of the project)
from langclaw.tools.builtin import telegram_send_message, web_search


MANAGER_SYSTEM_PROMPT = """## DeepClaw orchestrator

You are the top-level manager for the DeepClaw multi-agent system. Your role is to understand
the user's goal, break work into clear assignments, and delegate using the `task` tool to the
right specialist:

- **local-files** — workspace file reads/writes/search (code, configs, notes).
- **telegram-channel** — outbound Telegram messages via the project bot.
- **web-search** — broad web research (Tavily when configured).
- **web-crawler** — retrieve full page text from specific URLs (use after you have links).
- **security-review** — assess risk in plans, URLs, paths, or summarized outputs; use before risky steps.
- **quality-review** — critique another specialist's finished output; use when correctness matters.

You also have **general-purpose** for isolated complex tasks that mirror full capabilities.

**How to operate**

1. For non-trivial work, assign specialists rather than doing everything inline (saves context).
2. Run **security-review** early when dealing with untrusted URLs, shell, or sensitive files.
3. Use **quality-review** after substantive specialist output when verification is needed.
4. Synthesize specialist results into a clear, user-facing answer; the user does not see raw subagent logs.

You still have direct access to web_search, telegram_send_message, and fetch_url when a quick
single-step action is faster than delegating."""


def create_deepclaw_agent(
    model: str | BaseChatModel | None = None,
    *,
    workspace_root: str | Path | None = None,
    virtual_mode: bool = True,
    system_prompt: str | None = None,
    checkpointer: Checkpointer | None = None,
    name: str = "deepclaw",
    **kwargs: Any,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Build a Deep Agent with manager prompt, filesystem backend, and specialist subagents.

    Args:
        model: Same as `deepagents.create_deep_agent` (string id or chat model instance).
        workspace_root: Root directory for `FilesystemBackend` (default: current working directory).
        virtual_mode: If True, constrain relative paths to `workspace_root` (see deepagents docs).
        system_prompt: Prepended before the manager prompt; default is orchestration-only.
        checkpointer: Optional LangGraph checkpointer for durable threads.
        name: Agent name in metadata.
        **kwargs: Passed through to `create_deep_agent` (e.g. `interrupt_on`, `permissions`).

    Returns:
        Compiled LangGraph agent.
    """
    root = Path(workspace_root).resolve() if workspace_root else Path.cwd()
    backend = FilesystemBackend(root_dir=root, virtual_mode=virtual_mode)

    extra_tools = [web_search, telegram_send_message, fetch_url]
    subagents = build_specialists(extra_tools)

    merged_prompt = (
        f"{system_prompt.strip()}\n\n{MANAGER_SYSTEM_PROMPT}"
        if system_prompt
        else MANAGER_SYSTEM_PROMPT
    )

    return create_deep_agent(
        model=model,
        tools=extra_tools,
        subagents=subagents,
        system_prompt=merged_prompt,
        backend=backend,
        checkpointer=checkpointer,
        name=name,
        **kwargs,
    )


def _default_model_spec() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return "openai:gpt-4o-mini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic:claude-sonnet-4-20250514"
    return "ollama:llama3"


def _message_text(msg: BaseMessage) -> str:
    content = getattr(msg, "content", str(msg))
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def main() -> None:
    """Run DeepClaw in the terminal: read user lines and print agent replies."""
    parser = argparse.ArgumentParser(description="DeepClaw REPL (Deep Agent orchestrator)")
    parser.add_argument(
        "--model",
        default=os.getenv("DEEPCLAW_MODEL"),
        help="Chat model id (e.g. openai:gpt-4o-mini). Else env DEEPCLAW_MODEL or inferred from API keys.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Filesystem backend root (default: current directory)",
    )
    parser.add_argument(
        "--no-virtual-root",
        action="store_true",
        help="Disable virtual_mode on FilesystemBackend (less path isolation)",
    )
    parser.add_argument(
        "--thread-id",
        default="deepclaw-repl",
        help="LangGraph thread_id for this session (conversation memory)",
    )
    args = parser.parse_args()
    model = args.model or _default_model_spec()
    workspace = args.workspace.resolve() if args.workspace else Path.cwd()
    checkpointer = MemorySaver()
    graph = create_deepclaw_agent(
        model=model,
        workspace_root=workspace,
        virtual_mode=not args.no_virtual_root,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": args.thread_id}}

    print("DeepClaw — type a message and press Enter. Empty line or Ctrl+D to exit.")
    print(f"(model={model!r}, workspace={workspace}, thread_id={args.thread_id!r})\n")

    try:
        while True:
            try:
                line = input("You: ").strip()
            except EOFError:
                print()
                break
            if not line:
                break
            result = graph.invoke(
                {"messages": [HumanMessage(content=line)]},
                config=cast(RunnableConfig, config),
            )
            messages = result.get("messages") or []
            if messages:
                last = messages[-1]
                print(f"Agent: {_message_text(last)}\n")
            else:
                print("Agent: (no messages in result)\n")
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
