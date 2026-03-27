#!/usr/bin/env python3
"""Run LangClaw agent connected to a Telegram bot (long polling).

Usage:
  export TELEGRAM_BOT_TOKEN="..."
  source ../../.venv/bin/activate
  python run_telegram.py [agent_name]
"""

import asyncio
import sys
from pathlib import Path

# Allow running without installation
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def main() -> None:
    from langclaw.channels.telegram import TelegramAdapter
    from langclaw.config import load_config
    from langclaw.run import _build_runtime

    agent_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    app_dir = Path(__file__).parent
    cfg = load_config(app_dir / "config.yaml")

    if agent_name not in cfg.agents:
        raise SystemExit(f"Agent '{agent_name}' not found. Available: {list(cfg.agents)}")

    runtime, _mem, agent_cfg, _tools, memory_db, rag_info = _build_runtime(
        agent_name, app_dir / "config.yaml"
    )
    print(f"Agent: {agent_name} | Model: {agent_cfg.model} | RAG: {rag_info or 'off'}")

    adapter = TelegramAdapter(
        token="",
        memory_db=memory_db,
    )
    await adapter.run(runtime)


if __name__ == "__main__":
    asyncio.run(main())
