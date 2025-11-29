"""Internal data classes for the AI controller's model turn handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(slots=True)
class ToolCallRequest:
    """Internal representation of tool call directives emitted by the model."""

    call_id: str
    name: str
    index: int
    arguments: str | None
    parsed: Any | None


@dataclass(slots=True)
class ModelTurnResult:
    """Aggregate of a single model turn (stream) including tool metadata."""

    assistant_message: Dict[str, Any]
    response_text: str
    tool_calls: list[ToolCallRequest]


@dataclass(slots=True)
class MessagePlan:
    """Normalized plan for the prompt and completion budgeting."""

    messages: list[dict[str, str]]
    completion_budget: int | None
    prompt_tokens: int


__all__ = [
    "ToolCallRequest",
    "ModelTurnResult",
    "MessagePlan",
]
