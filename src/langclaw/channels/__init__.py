"""Channel adapters for LangClaw."""

from langclaw.channels.base import ChannelAdapter, NormalizedMessage
from langclaw.channels.telegram import TelegramAdapter

__all__ = ["ChannelAdapter", "NormalizedMessage", "TelegramAdapter"]

