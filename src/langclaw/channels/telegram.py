"""Telegram channel adapter (bot) using python-telegram-bot (async).

Long polling adapter:
- Each Telegram chat_id becomes a LangClaw thread_id
- Uses SQLite history so the agent has continuity per chat

Env:
  TELEGRAM_BOT_TOKEN=...   (required)
"""

from __future__ import annotations

import os
import asyncio
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage

from langclaw.memory import SqliteMemory
from langclaw.runtime import AgentRuntime


@dataclass
class TelegramAdapter:
    channel_name: str = "telegram"
    token: str = ""
    memory_db: Path = Path("langclaw.sqlite")

    def __post_init__(self) -> None:
        if not self.token:
            self.token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required for TelegramAdapter.")
        mem = SqliteMemory(self.memory_db)
        mem.setup()

    async def run(self, runtime: AgentRuntime) -> None:
        """Run telegram long-polling loop."""
        try:
            from telegram import Update
            from telegram.ext import (
                Application,
                CommandHandler,
                MessageHandler,
                ContextTypes,
                filters,
            )
        except ImportError as e:
            raise ImportError(
                "Telegram support requires: pip install 'python-telegram-bot>=21.0'"
            ) from e

        mem = SqliteMemory(self.memory_db)
        mem.setup()

        async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await update.message.reply_text(
                "LangClaw connected. Send a message and I will reply."
            )

        async def on_message(
            update: Update, context: ContextTypes.DEFAULT_TYPE
        ) -> None:
            if not update.message or update.message.text is None:
                return

            print("telegram message: ", update.message)

            chat_id = (
                str(update.effective_chat.id) if update.effective_chat else "unknown"
            )
            user_id = (
                str(update.effective_user.id) if update.effective_user else "unknown"
            )
            text = update.message.text.strip()
            if not text:
                return

            # Per-chat thread_id => persistent memory continuity
            thread_id = f"telegram:{chat_id}"

            history = mem.load_messages(thread_id, limit=50)
            input_messages = history + [HumanMessage(content=text)]

            try:
                result = runtime.invoke(
                    {"messages": input_messages},
                    config={"configurable": {"thread_id": thread_id}},
                )
                last = result["messages"][-1]
                reply = getattr(last, "content", str(last)) or "(no response)"
                await update.message.reply_text(reply[:4000])

                mem.append_messages(thread_id, [HumanMessage(content=text), last])
            except Exception as e:
                await update.message.reply_text(f"Error: {e}")

        app = Application.builder().token(self.token).build()
        app.add_handler(CommandHandler("start", start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))

        await app.initialize()
        await app.updater.start_polling()
        await app.start()

        # Run forever (until cancelled / interrupted). We keep the task alive with an Event.
        stop_event = asyncio.Event()
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            # Graceful shutdown when the parent task is cancelled
            pass
        finally:
            with contextlib.suppress(Exception):
                await app.updater.stop()
            with contextlib.suppress(Exception):
                await app.stop()
            with contextlib.suppress(Exception):
                await app.shutdown()
