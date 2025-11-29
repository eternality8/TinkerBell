"""Pipeline stage: Execute Model.

This module provides the execute stage of the turn pipeline, responsible for
calling the model with prepared messages and parsing the response.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..types import AnalyzedTurn, Message, ModelResponse, ParsedToolCall
from ..tool_call_parser import parse_embedded_tool_calls

__all__ = [
    "ModelClient",
    "StreamEvent",
    "execute_model",
    "parse_response",
    "aggregate_streaming_events",
    "extract_tool_calls_from_events",
    "merge_tool_calls",
]


# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------


@runtime_checkable
class StreamEvent(Protocol):
    """Protocol for streaming events from the model client.

    Implementations should provide these attributes. The AIStreamEvent from
    client.py conforms to this protocol.

    Attributes:
        type: Event type (e.g., "content.delta", "tool_calls.function.arguments.done").
        content: Text content for content events.
        tool_name: Name of the tool for tool call events.
        tool_index: Index of the tool call in the array.
        tool_arguments: Complete tool arguments (for .done events).
        arguments_delta: Incremental arguments (for .delta events).
        tool_call_id: ID of the tool call.
    """

    type: str
    content: str | None
    tool_name: str | None
    tool_index: int | None
    tool_arguments: str | None
    arguments_delta: str | None
    tool_call_id: str | None


@runtime_checkable
class ModelClient(Protocol):
    """Protocol for model clients that can stream chat completions.

    Implementations must provide a stream_chat method that yields streaming events.
    The AIClient class conforms to this protocol.
    """

    def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Stream chat completions for the provided messages.

        Args:
            messages: The conversation messages.
            tools: Tool definitions for the model.
            tool_choice: Tool choice specification.
            temperature: Sampling temperature.
            max_completion_tokens: Maximum tokens in the completion.
            **kwargs: Additional model parameters.

        Returns:
            An async iterator of streaming events.
        """
        ...


# -----------------------------------------------------------------------------
# Response Parsing
# -----------------------------------------------------------------------------


def parse_response(
    text: str,
    tool_calls: Sequence[ParsedToolCall] = (),
    *,
    finish_reason: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model: str | None = None,
    parse_embedded_markers: bool = True,
) -> ModelResponse:
    """Parse model output into a structured ModelResponse.

    Combines native tool calls with any embedded tool call markers found in
    the text response.

    Args:
        text: The text content from the model.
        tool_calls: Native tool calls from the API response.
        finish_reason: Why the model stopped generating.
        prompt_tokens: Tokens used in the prompt.
        completion_tokens: Tokens used in the completion.
        model: Model that generated the response.
        parse_embedded_markers: Whether to parse embedded tool call markers.

    Returns:
        A structured ModelResponse containing text and all tool calls.
    """
    all_tool_calls: list[ParsedToolCall] = list(tool_calls)

    # Parse embedded tool calls from text if enabled
    if parse_embedded_markers and text:
        embedded = _parse_embedded_calls(text, start_index=len(all_tool_calls))
        all_tool_calls.extend(embedded)

    return ModelResponse(
        text=text,
        tool_calls=tuple(all_tool_calls),
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
    )


def _parse_embedded_calls(text: str, start_index: int = 0) -> list[ParsedToolCall]:
    """Extract tool calls from embedded markers in text."""
    parsed_dicts = parse_embedded_tool_calls(text, start_index=start_index)
    return [
        ParsedToolCall(
            call_id=d.get("id", f"embedded_{start_index + i}_{uuid.uuid4().hex[:8]}"),
            name=d.get("name", "unknown"),
            arguments=d.get("arguments", "{}"),
            index=d.get("index", start_index + i),
        )
        for i, d in enumerate(parsed_dicts)
    ]


# -----------------------------------------------------------------------------
# Streaming Aggregation
# -----------------------------------------------------------------------------


def aggregate_streaming_events(events: Sequence[StreamEvent]) -> tuple[str, list[dict[str, Any]]]:
    """Aggregate streaming events into text content and tool calls.

    Processes a sequence of streaming events to reconstruct the full
    response text and tool calls.

    Args:
        events: Sequence of streaming events from the model.

    Returns:
        A tuple of (content_text, tool_call_dicts).
    """
    content_parts: list[str] = []
    tool_calls_by_index: dict[int, dict[str, Any]] = {}

    for event in events:
        event_type = event.type

        # Handle content events
        if event_type == "content.delta":
            if event.content:
                content_parts.append(event.content)
        elif event_type == "content.done":
            # Final content is usually included, but deltas should cover it
            pass

        # Handle tool call events
        elif event_type == "tool_calls.function.arguments.delta":
            index = event.tool_index if event.tool_index is not None else 0
            if index not in tool_calls_by_index:
                tool_calls_by_index[index] = {
                    "index": index,
                    "id": event.tool_call_id or "",
                    "name": event.tool_name or "",
                    "arguments_parts": [],
                }
            if event.arguments_delta:
                tool_calls_by_index[index]["arguments_parts"].append(event.arguments_delta)
            # Capture name and ID if present
            if event.tool_name:
                tool_calls_by_index[index]["name"] = event.tool_name
            if event.tool_call_id:
                tool_calls_by_index[index]["id"] = event.tool_call_id

        elif event_type == "tool_calls.function.arguments.done":
            index = event.tool_index if event.tool_index is not None else 0
            if index not in tool_calls_by_index:
                tool_calls_by_index[index] = {
                    "index": index,
                    "id": event.tool_call_id or "",
                    "name": event.tool_name or "",
                    "arguments_parts": [],
                }
            # Use complete arguments from done event
            if event.tool_arguments:
                tool_calls_by_index[index]["arguments"] = event.tool_arguments
            if event.tool_name:
                tool_calls_by_index[index]["name"] = event.tool_name
            if event.tool_call_id:
                tool_calls_by_index[index]["id"] = event.tool_call_id

    # Build final text
    content_text = "".join(content_parts)

    # Build final tool calls list
    tool_call_list: list[dict[str, Any]] = []
    for index in sorted(tool_calls_by_index.keys()):
        tc = tool_calls_by_index[index]
        # Prefer complete arguments from .done event, fall back to assembled parts
        if "arguments" not in tc:
            tc["arguments"] = "".join(tc.get("arguments_parts", []))
        # Remove temporary parts list
        tc.pop("arguments_parts", None)
        tool_call_list.append(tc)

    return content_text, tool_call_list


def extract_tool_calls_from_events(
    events: Sequence[StreamEvent],
) -> tuple[ParsedToolCall, ...]:
    """Extract parsed tool calls from streaming events.

    Args:
        events: Sequence of streaming events from the model.

    Returns:
        Tuple of ParsedToolCall objects.
    """
    _, tool_call_dicts = aggregate_streaming_events(events)
    return _tool_dicts_to_parsed(tool_call_dicts)


def _tool_dicts_to_parsed(tool_dicts: Sequence[dict[str, Any]]) -> tuple[ParsedToolCall, ...]:
    """Convert tool call dictionaries to ParsedToolCall objects."""
    results: list[ParsedToolCall] = []
    for tc in tool_dicts:
        call_id = tc.get("id", "")
        if not call_id:
            call_id = f"call_{tc.get('index', 0)}_{uuid.uuid4().hex[:8]}"
        results.append(
            ParsedToolCall(
                call_id=call_id,
                name=tc.get("name", "unknown"),
                arguments=tc.get("arguments", "{}"),
                index=tc.get("index", 0),
            )
        )
    return tuple(results)


def merge_tool_calls(
    native_calls: Sequence[ParsedToolCall],
    embedded_calls: Sequence[ParsedToolCall],
) -> tuple[ParsedToolCall, ...]:
    """Merge native API tool calls with embedded marker tool calls.

    Deduplicates by tool name and arguments to avoid calling the same tool twice.

    Args:
        native_calls: Tool calls from the native API response.
        embedded_calls: Tool calls parsed from embedded markers.

    Returns:
        Merged and deduplicated tuple of tool calls.
    """
    seen: set[tuple[str, str]] = set()
    merged: list[ParsedToolCall] = []

    for tc in native_calls:
        key = (tc.name, tc.arguments)
        if key not in seen:
            seen.add(key)
            merged.append(tc)

    for tc in embedded_calls:
        key = (tc.name, tc.arguments)
        if key not in seen:
            seen.add(key)
            # Re-index embedded calls after native ones
            merged.append(
                ParsedToolCall(
                    call_id=tc.call_id,
                    name=tc.name,
                    arguments=tc.arguments,
                    index=len(merged),
                )
            )

    return tuple(merged)


# -----------------------------------------------------------------------------
# Model Execution
# -----------------------------------------------------------------------------


async def execute_model(
    turn: AnalyzedTurn,
    client: ModelClient,
    *,
    tools: Sequence[Mapping[str, Any]] | None = None,
    tool_choice: str | Mapping[str, Any] | None = None,
    temperature: float | None = None,
    max_completion_tokens: int | None = None,
    model: str | None = None,
    parse_embedded_markers: bool = True,
    on_content_delta: Any | None = None,
) -> ModelResponse:
    """Execute a model call with the prepared turn and return the response.

    This is the main entry point for the execute stage. It streams the model
    response, aggregates events, and parses the result into a ModelResponse.

    Args:
        turn: The analyzed turn containing prepared messages with hints.
        client: The model client to use for streaming.
        tools: Tool definitions for the model.
        tool_choice: Tool choice specification.
        temperature: Sampling temperature (overrides turn config).
        max_completion_tokens: Maximum completion tokens.
        model: Model name (for metadata).
        parse_embedded_markers: Whether to parse embedded tool call markers.
        on_content_delta: Optional callback for content deltas (for streaming UI).

    Returns:
        A ModelResponse containing the model's text and tool calls.
    """
    # Get messages with hints injected
    messages = turn.messages_with_hints()
    message_dicts = [m.to_chat_param() for m in messages]

    # Collect streaming events
    collected_events: list[StreamEvent] = []

    async for event in client.stream_chat(
        message_dicts,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    ):
        collected_events.append(event)

        # Callback for real-time content streaming
        if on_content_delta is not None and event.type == "content.delta" and event.content:
            try:
                on_content_delta(event.content)
            except Exception:
                pass  # Don't let callback errors break streaming

    # Aggregate events into text and tool calls
    content_text, tool_call_dicts = aggregate_streaming_events(collected_events)
    native_tool_calls = _tool_dicts_to_parsed(tool_call_dicts)

    # Parse response with embedded markers
    return parse_response(
        text=content_text,
        tool_calls=native_tool_calls,
        model=model,
        parse_embedded_markers=parse_embedded_markers,
    )
