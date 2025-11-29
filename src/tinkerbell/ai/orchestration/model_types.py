"""Internal data classes for the AI orchestration turn handling."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Mapping, cast

from openai.types.chat import ChatCompletionToolParam


@dataclass(slots=True)
class OpenAIToolSpec:
    """Tool metadata formatted for OpenAI function calling API.
    
    This class handles runtime API formatting for tools registered
    with the orchestrator.
    
    Attributes:
        name: Tool identifier used in API calls.
        impl: The actual tool implementation (callable or object).
        description: Human-readable description (extracted from impl if not provided).
        parameters: JSON Schema for tool parameters.
        strict: Whether to use strict mode for function calling.
        summarizable: Whether tool output can be summarized for context.
    """

    name: str
    impl: Any
    description: str | None = None
    parameters: Mapping[str, Any] | None = None
    strict: bool = True
    summarizable: bool = True

    def as_openai_tool(self) -> ChatCompletionToolParam:
        """Return an OpenAI-compatible tool spec for the AI client."""
        parameters = self.parameters or getattr(self.impl, "args_schema", None)
        if callable(parameters):
            try:  # pragma: no cover - defensive, args_schema can be pydantic.BaseModel
                parameters = parameters()
            except TypeError:
                parameters = None
        if parameters is None:
            parameters = {"type": "object", "properties": {}}

        description = self.description or getattr(self.impl, "description", None) or (
            inspect.getdoc(self.impl) or f"Tool {self.name}"
        )

        return cast(
            ChatCompletionToolParam,
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": description,
                    "parameters": parameters,
                    "strict": bool(self.strict),
                },
            },
        )


# Backwards compatibility alias
ToolRegistration = OpenAIToolSpec


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
    "OpenAIToolSpec",
    "ToolRegistration",
    "ToolCallRequest",
    "ModelTurnResult",
    "MessagePlan",
]
