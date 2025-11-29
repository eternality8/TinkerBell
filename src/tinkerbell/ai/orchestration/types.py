"""Core type definitions for the orchestration pipeline.

This module defines the immutable dataclasses that flow through the turn pipeline.
All types are frozen to ensure immutability and enable safe sharing across stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Mapping, Sequence

from openai.types.chat import ChatCompletionMessageParam

# Re-export DocumentSnapshot from transaction module for convenience
from .transaction import DocumentSnapshot

__all__ = [
    # Core turn types
    "TurnInput",
    "TurnOutput",
    "TurnConfig",
    # Pipeline intermediate types
    "PreparedTurn",
    "AnalyzedTurn",
    "BudgetEstimate",
    # Model interaction types
    "Message",
    "ModelResponse",
    # Metrics and records
    "TurnMetrics",
    "ToolCallRecord",
    # Re-exports
    "DocumentSnapshot",
]


# -----------------------------------------------------------------------------
# Helper
# -----------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


# -----------------------------------------------------------------------------
# Message Type
# -----------------------------------------------------------------------------

MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(slots=True, frozen=True)
class Message:
    """Immutable chat message for pipeline processing.

    This is the canonical message type used throughout the pipeline.
    Can be converted to/from OpenAI's ChatCompletionMessageParam format.

    Attributes:
        role: The role of the message sender.
        content: The text content of the message.
        name: Optional name for function/tool messages.
        tool_call_id: ID linking tool result to its call.
        tool_calls: Tool calls made by the assistant.
        metadata: Additional metadata (not sent to model).
    """

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[Mapping[str, Any], ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_chat_param(self) -> ChatCompletionMessageParam:
        """Convert to OpenAI's ChatCompletionMessageParam format."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            payload["tool_calls"] = list(self.tool_calls)
        return payload  # type: ignore[return-value]

    @classmethod
    def from_chat_param(cls, param: Mapping[str, Any]) -> Message:
        """Create a Message from OpenAI's ChatCompletionMessageParam format."""
        tool_calls = param.get("tool_calls")
        if tool_calls is not None:
            tool_calls = tuple(tool_calls)
        return cls(
            role=param.get("role", "user"),  # type: ignore[arg-type]
            content=str(param.get("content", "")),
            name=param.get("name"),
            tool_call_id=param.get("tool_call_id"),
            tool_calls=tool_calls,
        )

    @classmethod
    def system(cls, content: str, **metadata: Any) -> Message:
        """Create a system message."""
        return cls(role="system", content=content, metadata=metadata)

    @classmethod
    def user(cls, content: str, **metadata: Any) -> Message:
        """Create a user message."""
        return cls(role="user", content=content, metadata=metadata)

    @classmethod
    def assistant(
        cls,
        content: str,
        tool_calls: Sequence[Mapping[str, Any]] | None = None,
        **metadata: Any,
    ) -> Message:
        """Create an assistant message."""
        return cls(
            role="assistant",
            content=content,
            tool_calls=tuple(tool_calls) if tool_calls else None,
            metadata=metadata,
        )

    @classmethod
    def tool(
        cls,
        content: str,
        tool_call_id: str,
        name: str | None = None,
        **metadata: Any,
    ) -> Message:
        """Create a tool result message."""
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
            metadata=metadata,
        )


# -----------------------------------------------------------------------------
# Budget Estimate
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class BudgetEstimate:
    """Token budget estimate for a prepared turn.

    Attributes:
        prompt_tokens: Estimated tokens in the prompt.
        completion_budget: Maximum tokens available for completion.
        total_budget: Total context window size.
        headroom: Tokens remaining after prompt and completion reserve.
        verdict: Budget evaluation result ("ok", "needs_summary", "reject").
        reason: Human-readable explanation of the verdict.
    """

    prompt_tokens: int
    completion_budget: int
    total_budget: int
    headroom: int = 0
    verdict: Literal["ok", "needs_summary", "reject"] = "ok"
    reason: str = "within-budget"

    @property
    def is_ok(self) -> bool:
        """Check if the budget verdict allows proceeding."""
        return self.verdict == "ok"

    @property
    def needs_trimming(self) -> bool:
        """Check if context needs to be reduced."""
        return self.verdict in ("needs_summary", "reject")


# -----------------------------------------------------------------------------
# Turn Configuration
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class TurnConfig:
    """Configuration for a single turn execution.

    Immutable configuration that controls turn behavior.

    Attributes:
        max_iterations: Maximum tool loop iterations.
        analysis_enabled: Whether to run preflight analysis.
        max_context_tokens: Maximum tokens in context window.
        response_reserve: Tokens reserved for model response.
        model_name: Name of the model to use.
        temperature: Sampling temperature.
        tool_timeout_seconds: Timeout for individual tool execution.
        streaming_enabled: Whether to use streaming responses.
    """

    max_iterations: int = 8
    analysis_enabled: bool = True
    max_context_tokens: int = 128_000
    response_reserve: int = 4_096
    model_name: str | None = None
    temperature: float = 0.7
    tool_timeout_seconds: float = 30.0
    streaming_enabled: bool = True

    def with_updates(self, **kwargs: Any) -> TurnConfig:
        """Return a new TurnConfig with updated values."""
        current = {
            "max_iterations": self.max_iterations,
            "analysis_enabled": self.analysis_enabled,
            "max_context_tokens": self.max_context_tokens,
            "response_reserve": self.response_reserve,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "tool_timeout_seconds": self.tool_timeout_seconds,
            "streaming_enabled": self.streaming_enabled,
        }
        current.update(kwargs)
        return TurnConfig(**current)


# -----------------------------------------------------------------------------
# Core Turn Types
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class TurnInput:
    """Immutable input to a chat turn.

    Contains everything needed to execute a turn through the pipeline.

    Attributes:
        prompt: The user's message/request.
        snapshot: Current document state snapshot.
        history: Previous conversation messages.
        config: Turn execution configuration.
        run_id: Unique identifier for this turn.
        document_id: Identifier for the active document.
    """

    prompt: str
    snapshot: DocumentSnapshot
    history: tuple[Message, ...] = ()
    config: TurnConfig = field(default_factory=TurnConfig)
    run_id: str = ""
    document_id: str = ""

    def __post_init__(self) -> None:
        # Ensure history is a tuple for immutability
        if not isinstance(self.history, tuple):
            object.__setattr__(self, "history", tuple(self.history))


@dataclass(slots=True, frozen=True)
class ToolCallRecord:
    """Record of a tool call execution.

    Immutable record capturing the details of a tool invocation.

    Attributes:
        call_id: Unique identifier for this call.
        name: Name of the tool that was called.
        arguments: Arguments passed to the tool (as JSON string or dict).
        result: The tool's output/result.
        success: Whether the tool executed successfully.
        duration_ms: Execution time in milliseconds.
        error: Error message if the call failed.
        metadata: Additional execution metadata.
    """

    call_id: str
    name: str
    arguments: str | Mapping[str, Any]
    result: str = ""
    success: bool = True
    duration_ms: float = 0.0
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for telemetry/logging."""
        return {
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments if isinstance(self.arguments, str) else dict(self.arguments),
            "result": self.result,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class TurnMetrics:
    """Metrics collected during turn execution.

    Captures timing, token usage, and other metrics for observability.

    Attributes:
        prompt_tokens: Tokens in the prompt.
        completion_tokens: Tokens in the completion.
        total_tokens: Total tokens used.
        duration_ms: Total turn execution time in milliseconds.
        tool_call_count: Number of tool calls made.
        iteration_count: Number of tool loop iterations.
        model_name: Model used for the turn.
        analysis_ran: Whether preflight analysis was executed.
        started_at: When the turn started.
        finished_at: When the turn finished.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    duration_ms: float = 0.0
    tool_call_count: int = 0
    iteration_count: int = 1
    model_name: str | None = None
    analysis_ran: bool = False
    started_at: datetime = field(default_factory=_utcnow)
    finished_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for telemetry/logging."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "tool_call_count": self.tool_call_count,
            "iteration_count": self.iteration_count,
            "model_name": self.model_name,
            "analysis_ran": self.analysis_ran,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
        }


@dataclass(slots=True, frozen=True)
class TurnOutput:
    """Immutable output from a chat turn.

    Contains everything produced by a turn execution.

    Attributes:
        response: The assistant's text response.
        tool_calls: Records of all tool calls made during the turn.
        metrics: Execution metrics for the turn.
        success: Whether the turn completed successfully.
        error: Error message if the turn failed.
        metadata: Additional output metadata.
    """

    response: str
    tool_calls: tuple[ToolCallRecord, ...] = ()
    metrics: TurnMetrics = field(default_factory=TurnMetrics)
    success: bool = True
    error: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Ensure tool_calls is a tuple for immutability
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))

    @classmethod
    def from_error(cls, error: str, metrics: TurnMetrics | None = None) -> TurnOutput:
        """Create a TurnOutput representing a failed turn."""
        return cls(
            response="",
            tool_calls=(),
            metrics=metrics or TurnMetrics(),
            success=False,
            error=error,
        )


# -----------------------------------------------------------------------------
# Pipeline Intermediate Types
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class PreparedTurn:
    """Output from the prepare stage.

    Contains the built messages and budget estimate ready for analysis/execution.

    Attributes:
        messages: The prepared message list for the model.
        budget: Token budget estimate.
        system_prompt: The system prompt used.
        document_context: Document content included in context.
    """

    messages: tuple[Message, ...]
    budget: BudgetEstimate
    system_prompt: str = ""
    document_context: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple):
            object.__setattr__(self, "messages", tuple(self.messages))


@dataclass(slots=True, frozen=True)
class AnalyzedTurn:
    """Output from the analyze stage.

    Contains the prepared turn plus any analysis hints/advice.

    Attributes:
        prepared: The prepared turn from the previous stage.
        hints: Analysis hints to inject into the prompt.
        advice: Raw analysis advice (if analysis was run).
        analysis_ran: Whether analysis was executed.
    """

    prepared: PreparedTurn
    hints: tuple[str, ...] = ()
    advice: Mapping[str, Any] | None = None
    analysis_ran: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.hints, tuple):
            object.__setattr__(self, "hints", tuple(self.hints))

    def messages_with_hints(self) -> tuple[Message, ...]:
        """Return messages with analysis hints injected."""
        if not self.hints:
            return self.prepared.messages

        messages = list(self.prepared.messages)
        hint_text = "\n".join(self.hints)

        # Find the last system message and append hints, or add a new one
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "system":
                original = messages[i]
                updated = Message(
                    role="system",
                    content=f"{original.content}\n\n{hint_text}",
                    name=original.name,
                    metadata=original.metadata,
                )
                messages[i] = updated
                return tuple(messages)

        # No system message found, prepend hints as system message
        messages.insert(0, Message.system(hint_text))
        return tuple(messages)


# -----------------------------------------------------------------------------
# Model Response Types
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ParsedToolCall:
    """A parsed tool call from the model response.

    Attributes:
        call_id: Unique identifier for this call.
        name: Name of the tool to call.
        arguments: Arguments as a JSON string.
        index: Position in the tool_calls array.
    """

    call_id: str
    name: str
    arguments: str
    index: int = 0


@dataclass(slots=True, frozen=True)
class ModelResponse:
    """Parsed response from the model.

    Contains the model's text response and any tool calls.

    Attributes:
        text: The text content of the response.
        tool_calls: Parsed tool calls from the response.
        finish_reason: Why the model stopped generating.
        prompt_tokens: Tokens used in the prompt.
        completion_tokens: Tokens used in the completion.
        model: Model that generated the response.
    """

    text: str
    tool_calls: tuple[ParsedToolCall, ...] = ()
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))

    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return len(self.tool_calls) > 0

    def to_message(self) -> Message:
        """Convert response to an assistant Message."""
        tool_calls_data = None
        if self.tool_calls:
            tool_calls_data = tuple(
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            )
        return Message.assistant(self.text, tool_calls=tool_calls_data)
