"""Run LangClaw agent from config."""

import asyncio
import uuid
import sys
import os
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage


def _build_runtime(agent_name: str, config_path: Path):
    """Build runtime + memory + config-derived settings.
    Returns:
        runtime, mem, agent_cfg, tools, memory_db, rag_info
    """
    from langclaw.config import load_config
    from langclaw.memory import SqliteMemory
    from langclaw.models import get_model
    from langclaw.tools import get_tools, load_skills, build_skills_prompt
    from langclaw.runtime import AgentRuntime

    cfg = load_config(config_path)
    app_dir = config_path.parent
    skills_root = app_dir / "skills"
    memory_db = app_dir / "langclaw.sqlite"

    if agent_name not in cfg.agents:
        print(
            f"Agent '{agent_name}' not found in config. Available: {list(cfg.agents)}"
        )
        sys.exit(1)

    agent_cfg = cfg.agents[agent_name]
    model = get_model(
        agent_cfg.model,
        temperature=agent_cfg.temperature,
    )

    mcp_config = agent_cfg.mcp.model_dump() if agent_cfg.mcp else None
    tools = get_tools(
        tool_names=agent_cfg.tools if agent_cfg.tools else None,
        mcp_config=mcp_config,
    )

    system_prompt = agent_cfg.system_prompt
    if agent_cfg.skills and skills_root.exists():
        skills = load_skills(skills_root, agent_cfg.skills)
        if skills:
            system_prompt = (
                system_prompt.rstrip() + "\n\n" + build_skills_prompt(skills)
            )

    rag_retriever = None
    rag_info: str | None = None
    if agent_cfg.rag and agent_cfg.rag.enabled and agent_cfg.rag.sources:
        from langclaw.rag import build_rag_retriever

        rag_retriever = build_rag_retriever(agent_cfg.rag, app_dir)
        rag_info = (
            f"on, sources={agent_cfg.rag.sources}, k={agent_cfg.rag.k}, "
            f"embed={agent_cfg.rag.embedding_model}"
        )

    runtime = AgentRuntime(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        rag_retriever=rag_retriever,
    )

    mem = SqliteMemory(memory_db)
    mem.setup()

    return runtime, mem, agent_cfg, tools, memory_db, rag_info


def _terminal_loop(runtime, mem, thread_id: str) -> None:
    """Blocking terminal loop. Shares the same SQLite memory."""
    print("=" * 60)
    print("LangClaw Agent - Terminal Mode")
    print("=" * 60)
    # Model/tools are printed by caller if desired.
    print("-" * 60)
    print("Enter 'quit' to exit.")
    print("=" * 60)

    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            # Load recent history from DB and feed it as new input.
            history = mem.load_messages(thread_id, limit=50)
            input_messages = history + [HumanMessage(content=user_input)]

            result = runtime.invoke(
                {"messages": input_messages},
                config=config,
            )
            messages = result["messages"]
            last = messages[-1]
            content = getattr(last, "content", str(last)) or "(no response)"
            print("\nAgent:", content)

            # Append the new turn (user + assistant) to the DB history.
            mem.append_messages(thread_id, [HumanMessage(content=user_input), last])
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback

            traceback.print_exc()


async def main_async(
    agent_name: str = "default", config_path: Optional[Path] = None
) -> None:
    """Run LangClaw based on configured channels (terminal, telegram, ...)."""
    config_path = config_path or Path(__file__).parent.parent / "config.yaml"
    runtime, mem, agent_cfg, tools, memory_db, rag_info = _build_runtime(
        agent_name, config_path
    )

    print("=" * 60)
    print("LangClaw Agent")
    print("=" * 60)
    print(f"Agent: {agent_name}")
    print(f"Model: {agent_cfg.model}")
    print(f"Channels: {agent_cfg.channels}")
    print(f"RAG: {rag_info or 'off'}")
    print(f"Tools: {[t.name for t in tools]}")
    print(f"Memory DB: {memory_db}")
    print("=" * 60)

    tasks: list[asyncio.Task] = []

    # Telegram channel
    if "telegram" in (agent_cfg.channels or []):
        from langclaw.channels.telegram import TelegramAdapter

        adapter = TelegramAdapter(token="", memory_db=memory_db)
        tasks.append(asyncio.create_task(adapter.run(runtime)))

    # Terminal channel (blocking) runs in a thread so it can coexist with async channels.
    if "terminal" in (agent_cfg.channels or []):
        thread_id = os.getenv("LANGCLAW_THREAD_ID") or str(uuid.uuid4())
        tasks.append(
            asyncio.create_task(
                asyncio.to_thread(_terminal_loop, runtime, mem, thread_id)
            )
        )

    if not tasks:
        raise SystemExit(
            "No channels enabled. Set agent.channels in config.yaml (e.g. [terminal, telegram])."
        )

    # Run until first task exits (terminal quit or telegram loop ends)
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()


def main(agent_name: str = "default", config_path: Optional[Path] = None) -> None:
    """Entry point used by applications/langclaw/run.py."""
    asyncio.run(main_async(agent_name=agent_name, config_path=config_path))


if __name__ == "__main__":
    agent = sys.argv[1] if len(sys.argv) > 1 else "default"
    main(agent_name=agent)
