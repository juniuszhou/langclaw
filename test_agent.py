#!/usr/bin/env python3
"""Quick test of LangClaw agent (no interactive input)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from langchain_core.messages import HumanMessage
from langclaw.run import _build_runtime


def main():
    config_path = Path(__file__).parent / "config.yaml"
    runtime, _mem, agent_cfg, tools, _db, rag_info = _build_runtime(
        "default", config_path
    )
    if rag_info:
        print("RAG:", rag_info)

    config = {"configurable": {"thread_id": "test-session-1"}}
    result = runtime.invoke(
        {"messages": [HumanMessage(content="What is 15 * 7? Use the calculator.")]},
        config=config,
    )
    last = result["messages"][-1]
    print("Response:", getattr(last, "content", str(last)))
    print("Tools loaded:", [t.name for t in tools])


if __name__ == "__main__":
    main()
