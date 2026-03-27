"""Channel adapter base types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class NormalizedMessage:
    """A unified inbound message event from any channel."""

    channel: str
    channel_id: str
    sender_id: str
    thread_id: str
    text: str
    raw: Any | None = None


class ChannelAdapter(Protocol):
    """Channel adapter interface."""

    channel_name: str

    async def run(self) -> None:
        """Start receiving messages and emitting responses."""

