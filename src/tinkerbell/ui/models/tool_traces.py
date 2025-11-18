"""Models for tracking streaming tool traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...chat.message_model import ToolTrace


@dataclass(slots=True)
class PendingToolTrace:
    """Bookkeeping for streaming tool call data before it is displayed."""

    name: str
    arguments_chunks: list[str] = field(default_factory=list)
    raw_input: str | None = None
    pending_output: str | None = None
    pending_parsed: Any | None = None
    trace: ToolTrace | None = None
    started_at: float | None = None
    tool_call_id: str | None = None


__all__ = ["PendingToolTrace"]
