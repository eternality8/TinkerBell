"""Chat message and tool trace data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""

    return datetime.now(timezone.utc)

ChatRole = Literal["user", "assistant", "system", "tool"]


@dataclass(slots=True)
class ToolTrace:
    """Trace output for a single tool invocation."""

    name: str
    input_summary: str
    output_summary: str
    duration_ms: int = 0


@dataclass(slots=True)
class ChatMessage:
    """Represents a row inside the chat history list."""

    role: ChatRole
    content: str
    created_at: datetime = field(default_factory=_utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_traces: list[ToolTrace] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the message for persistence."""

        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "tool_traces": [trace.__dict__ for trace in self.tool_traces],
        }


@dataclass(slots=True)
class EditDirective:
    """Structured command emitted by the agent to modify the document."""

    action: str
    target_range: tuple[int, int]
    content: str
    rationale: Optional[str] = None
