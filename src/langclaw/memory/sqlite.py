"""SQLite-backed memory for LangClaw.

This is intentionally simple and dependency-free (sqlite3 from stdlib).

Short-term memory:
- Persists chat history (messages) keyed by thread_id

Long-term memory:
- Persists "notes" keyed by (namespace, key), plus basic LIKE search
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _serialize_message(msg: BaseMessage) -> dict[str, Any]:
    # Keep only what we need; preserve tool_calls in additional_kwargs/response_metadata
    msg_type = msg.__class__.__name__
    return {
        "type": msg_type,
        "content": msg.content,
        "additional_kwargs": getattr(msg, "additional_kwargs", {}) or {},
        "response_metadata": getattr(msg, "response_metadata", {}) or {},
        # ToolMessage fields (best-effort)
        "tool_call_id": getattr(msg, "tool_call_id", None),
        "name": getattr(msg, "name", None),
    }


def _deserialize_message(payload: dict[str, Any]) -> BaseMessage:
    t = payload.get("type")
    content = payload.get("content", "")
    additional_kwargs = payload.get("additional_kwargs") or {}
    response_metadata = payload.get("response_metadata") or {}

    if t == "HumanMessage":
        return HumanMessage(content=content, additional_kwargs=additional_kwargs, response_metadata=response_metadata)
    if t == "AIMessage":
        return AIMessage(content=content, additional_kwargs=additional_kwargs, response_metadata=response_metadata)
    if t == "SystemMessage":
        return SystemMessage(content=content, additional_kwargs=additional_kwargs, response_metadata=response_metadata)
    if t == "ToolMessage":
        return ToolMessage(
            content=content,
            tool_call_id=payload.get("tool_call_id") or "",
            additional_kwargs=additional_kwargs,
            response_metadata=response_metadata,
            name=payload.get("name"),
        )
    # Fallback to AI message
    return AIMessage(content=content, additional_kwargs=additional_kwargs, response_metadata=response_metadata)


@dataclass(frozen=True)
class SqliteMemory:
    """SQLite memory wrapper."""

    db_path: Path

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def setup(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_ts ON chat_messages(thread_id, ts_ms);"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS long_term_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    namespace TEXT NOT NULL,
                    key TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    value_text TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_notes_ns_key ON long_term_notes(namespace, key);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_notes_ns_ts ON long_term_notes(namespace, ts_ms);"
            )

    # ---------------------------
    # Short-term chat history
    # ---------------------------
    def append_messages(self, thread_id: str, messages: Sequence[BaseMessage]) -> None:
        if not messages:
            return
        ts = _now_ms()
        rows = [(thread_id, ts, json.dumps(_serialize_message(m), ensure_ascii=False)) for m in messages]
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO chat_messages(thread_id, ts_ms, payload_json) VALUES (?, ?, ?);",
                rows,
            )

    def load_messages(self, thread_id: str, limit: int = 50) -> List[BaseMessage]:
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT payload_json
                FROM chat_messages
                WHERE thread_id = ?
                ORDER BY ts_ms DESC, id DESC
                LIMIT ?;
                """,
                (thread_id, limit),
            )
            rows = cur.fetchall()
        # Reverse to chronological order
        payloads = [json.loads(r[0]) for r in reversed(rows)]
        return [_deserialize_message(p) for p in payloads]

    # ---------------------------
    # Long-term memory (notes)
    # ---------------------------
    def note_put(self, namespace: str, key: str, value_text: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO long_term_notes(namespace, key, ts_ms, value_text)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(namespace, key)
                DO UPDATE SET ts_ms=excluded.ts_ms, value_text=excluded.value_text;
                """,
                (namespace, key, _now_ms(), value_text),
            )

    def note_get(self, namespace: str, key: str) -> Optional[str]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT value_text FROM long_term_notes WHERE namespace=? AND key=?;",
                (namespace, key),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def note_search(self, namespace: str, query: str, limit: int = 10) -> List[dict[str, Any]]:
        like = f"%{query}%"
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT key, ts_ms, value_text
                FROM long_term_notes
                WHERE namespace=? AND value_text LIKE ?
                ORDER BY ts_ms DESC
                LIMIT ?;
                """,
                (namespace, like, limit),
            )
            rows = cur.fetchall()
        return [{"key": k, "ts_ms": ts, "value": v} for (k, ts, v) in rows]

