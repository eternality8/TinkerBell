"""Chat message and tool trace data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from ..documents.ranges import TextRange


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""

    return datetime.now(timezone.utc)

ChatRole = Literal["user", "assistant", "system", "tool"]


@dataclass(slots=True)
class ToolPointerMessage:
    """Lightweight pointer descriptor used to compact oversized tool output."""

    pointer_id: str
    kind: str
    display_text: str
    rehydrate_instructions: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_chat_content(self) -> str:
        """Render the pointer as plain-text content for chat history."""

        header = f"[pointer:{self.pointer_id} kind={self.kind}]"
        body = self.display_text.strip() or "(no summary available)"
        footer = self.rehydrate_instructions.strip()
        blocks = [header, body]
        if footer:
            blocks.append(f"Rehydrate instructions: {footer}")
        return "\n".join(blocks)

    def as_dict(self) -> Dict[str, Any]:
        """Serialize the pointer for telemetry/UI traces."""

        payload: Dict[str, Any] = {
            "pointer_id": self.pointer_id,
            "kind": self.kind,
            "display_text": self.display_text,
            "rehydrate_instructions": self.rehydrate_instructions,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class ToolTrace:
    """Trace output for a single tool invocation."""

    name: str
    input_summary: str
    output_summary: str
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    step_index: Optional[int] = None


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
    target_range: TextRange
    content: str
    rationale: Optional[str] = None
    diff: Optional[str] = None
    match_text: Optional[str] = None
    expected_text: Optional[str] = None
    replace_all: Optional[bool] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_range", TextRange.from_value(self.target_range))
