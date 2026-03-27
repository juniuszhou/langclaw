#!/usr/bin/env python3
"""Run LangClaw agent. Usage: python run.py [agent_name]"""

import sys
from pathlib import Path

# Add src to path when running from applications/langclaw
sys.path.insert(0, str(Path(__file__).parent / "src"))

from langclaw.run import main

if __name__ == "__main__":
    agent = sys.argv[1] if len(sys.argv) > 1 else "default"
    main(agent_name=agent, config_path=Path(__file__).parent / "config.yaml")
