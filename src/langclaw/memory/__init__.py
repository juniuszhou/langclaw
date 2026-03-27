"""Memory backends for LangClaw.

Phase 2:
- Short-term memory: persist chat history by thread_id (SQLite).
- Long-term memory: key/value notes with simple keyword search (SQLite).
"""

from langclaw.memory.sqlite import SqliteMemory

__all__ = ["SqliteMemory"]

