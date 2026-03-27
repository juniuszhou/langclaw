#!/usr/bin/env python3
"""Demo: use history stored in SQLite DB as new input.

This shows how short-term memory works without relying on LangGraph checkpointer backends.

Run:
  source ../../.venv/bin/activate
  python demo_db_history.py
"""

from pathlib import Path

from langchain_core.messages import HumanMessage

from langclaw.memory import SqliteMemory
from langclaw.models import get_model
from langclaw.runtime import AgentRuntime
from langclaw.tools import get_tools


def main() -> None:
    app_dir = Path(__file__).parent
    db_path = app_dir / "langclaw.sqlite"
    mem = SqliteMemory(db_path)
    mem.setup()

    thread_id = "demo-thread-1"
    model = get_model("ollama/llama3.2:3b", temperature=0)
    tools = get_tools(["calculator", "get_current_time"])
    runtime = AgentRuntime(model=model, tools=tools, system_prompt="You are a helpful assistant.")

    # Turn 1
    history = mem.load_messages(thread_id, limit=50)
    turn1_user = HumanMessage(content="Remember that my favorite number is 7.")
    result1 = runtime.invoke({"messages": history + [turn1_user]}, config={"configurable": {"thread_id": thread_id}})
    turn1_ai = result1["messages"][-1]
    mem.append_messages(thread_id, [turn1_user, turn1_ai])
    print("Turn1 AI:", turn1_ai.content)

    # Turn 2: load history from DB and continue
    history2 = mem.load_messages(thread_id, limit=50)
    turn2_user = HumanMessage(content="What is my favorite number? Also compute 7*6.")
    result2 = runtime.invoke({"messages": history2 + [turn2_user]}, config={"configurable": {"thread_id": thread_id}})
    turn2_ai = result2["messages"][-1]
    mem.append_messages(thread_id, [turn2_user, turn2_ai])
    print("Turn2 AI:", turn2_ai.content)

    print(f"\nDB path: {db_path}")
    print(f"thread_id: {thread_id}")


if __name__ == "__main__":
    main()

