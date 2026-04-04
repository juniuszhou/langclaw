#!/usr/bin/env python3
"""Run LangClaw as an A2A (Agent2Agent) JSON-RPC server.

Usage:
  pip install -e ".[a2a]"
  export LANGCLAW_A2A_PUBLIC_URL=http://127.0.0.1:9999  # optional; defaults from bind
  python run_a2a.py [agent_name] [--host 127.0.0.1] [--port 9999]

Discovery:
  GET  {public_url}/.well-known/agent-card.json
  POST {public_url}/           JSON-RPC 2.0 (e.g. message/send)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="LangClaw A2A server")
    parser.add_argument(
        "agent_name",
        nargs="?",
        default="default",
        help="Agent name in config.yaml (default: default)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError as e:
        raise SystemExit(
            "uvicorn is required. Install A2A extras: pip install -e '.[a2a]'"
        ) from e

    try:
        from langclaw.a2a import build_langclaw_agent_card, create_langclaw_a2a_app
        from langclaw.run import _build_runtime
    except ImportError as e:
        raise SystemExit(
            "A2A integration failed to import. Install: pip install -e '.[a2a]'"
        ) from e

    config_path = Path(__file__).resolve().parent / "config.yaml"
    runtime, mem, agent_cfg, _tools, _memory_db, _rag = _build_runtime(
        args.agent_name, config_path
    )

    public = (
        os.getenv("LANGCLAW_A2A_PUBLIC_URL", "").strip().rstrip("/")
        or f"http://{args.host}:{args.port}"
    )
    card = build_langclaw_agent_card(
        public_url=public,
        name=f"LangClaw ({args.agent_name})",
        description=agent_cfg.system_prompt[:500]
        if agent_cfg.system_prompt
        else "LangClaw agent",
    )
    app = create_langclaw_a2a_app(agent_card=card, runtime=runtime, memory=mem)

    print("A2A server starting")
    print(f"  Agent card: {public}/.well-known/agent-card.json")
    print(f"  JSON-RPC:   POST {public}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
