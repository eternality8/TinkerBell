"""Tests for orchestration/pipeline/execute.py."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Sequence
from unittest.mock import MagicMock

import pytest

from tinkerbell.ai.orchestration.pipeline.execute import (
    ModelClient,
    StreamEvent,
    aggregate_streaming_events,
    execute_model,
    extract_tool_calls_from_events,
    merge_tool_calls,
    parse_response,
)
from tinkerbell.ai.orchestration.types import (
    AnalyzedTurn,
    BudgetEstimate,
    Message,
    ModelResponse,
    ParsedToolCall,
    PreparedTurn,
)


# -----------------------------------------------------------------------------
# Test Fixtures and Helpers
# -----------------------------------------------------------------------------


@dataclass
class MockStreamEvent:
    """Mock stream event for testing."""

    type: str
    content: str | None = None
    tool_name: str | None = None
    tool_index: int | None = None
    tool_arguments: str | None = None
    arguments_delta: str | None = None
    tool_call_id: str | None = None


class MockModelClient:
    """Mock model client for testing."""

    def __init__(
        self,
        events: list[MockStreamEvent] | None = None,
        raise_exception: Exception | None = None,
    ) -> None:
        self.events = events or []
        self.raise_exception = raise_exception
        self.last_call_args: dict[str, Any] = {}

    async def stream_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        tool_choice: str | Mapping[str, Any] | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[MockStreamEvent]:
        self.last_call_args = {
            "messages": list(messages),
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "kwargs": kwargs,
        }

        if self.raise_exception:
            raise self.raise_exception

        for event in self.events:
            yield event


def make_prepared_turn(
    messages: Sequence[Message] | None = None,
    prompt_tokens: int = 100,
) -> PreparedTurn:
    """Helper to create a PreparedTurn for testing."""
    if messages is None:
        messages = [
            Message.system("You are a helpful assistant."),
            Message.user("Hello!"),
        ]
    return PreparedTurn(
        messages=tuple(messages),
        budget=BudgetEstimate(
            prompt_tokens=prompt_tokens,
            completion_budget=4000,
            total_budget=128000,
        ),
        system_prompt="You are a helpful assistant.",
    )


def make_analyzed_turn(
    messages: Sequence[Message] | None = None,
    hints: Sequence[str] = (),
) -> AnalyzedTurn:
    """Helper to create an AnalyzedTurn for testing."""
    return AnalyzedTurn(
        prepared=make_prepared_turn(messages),
        hints=tuple(hints),
        analysis_ran=bool(hints),
    )


# -----------------------------------------------------------------------------
# Tests: parse_response
# -----------------------------------------------------------------------------


class TestParseResponse:
    """Tests for parse_response function."""

    def test_simple_text_response(self) -> None:
        """Test parsing a simple text response."""
        result = parse_response("Hello, world!")
        assert result.text == "Hello, world!"
        assert result.tool_calls == ()
        assert result.has_tool_calls is False

    def test_text_with_native_tool_calls(self) -> None:
        """Test parsing response with native tool calls."""
        tool_calls = [
            ParsedToolCall(call_id="call_1", name="read_file", arguments='{"path": "/a.txt"}', index=0),
            ParsedToolCall(call_id="call_2", name="write_file", arguments='{"path": "/b.txt"}', index=1),
        ]
        result = parse_response("Processing files...", tool_calls)
        assert result.text == "Processing files..."
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "read_file"
        assert result.tool_calls[1].name == "write_file"
        assert result.has_tool_calls is True

    def test_text_with_embedded_tool_markers(self) -> None:
        """Test parsing response with embedded tool call markers."""
        text = """Let me help you.
<|tool_calls_begin|>
<|tool_call_begin|>read_file<|tool_sep|>{"path": "/test.txt"}<|tool_call_end|>
<|tool_calls_end|>"""
        result = parse_response(text)
        assert "Let me help you" in result.text
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"

    def test_embedded_markers_disabled(self) -> None:
        """Test that embedded markers are not parsed when disabled."""
        text = """<|tool_calls_begin|>
<|tool_call_begin|>read_file<|tool_sep|>{"path": "/test.txt"}<|tool_call_end|>
<|tool_calls_end|>"""
        result = parse_response(text, parse_embedded_markers=False)
        assert result.tool_calls == ()

    def test_combined_native_and_embedded_calls(self) -> None:
        """Test combining native and embedded tool calls."""
        native = [ParsedToolCall(call_id="native_1", name="tool_a", arguments="{}", index=0)]
        text = """Response text
<|tool_calls_begin|>
<|tool_call_begin|>tool_b<|tool_sep|>{"x": 1}<|tool_call_end|>
<|tool_calls_end|>"""
        result = parse_response(text, native)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "tool_a"
        assert result.tool_calls[1].name == "tool_b"

    def test_metadata_preserved(self) -> None:
        """Test that metadata fields are preserved."""
        result = parse_response(
            "Test",
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=50,
            model="gpt-4o",
        )
        assert result.finish_reason == "stop"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.model == "gpt-4o"

    def test_empty_text(self) -> None:
        """Test parsing empty text."""
        result = parse_response("")
        assert result.text == ""
        assert result.tool_calls == ()

    def test_to_message_conversion(self) -> None:
        """Test converting ModelResponse to Message."""
        tool_calls = [
            ParsedToolCall(call_id="call_1", name="test_tool", arguments='{"a": 1}', index=0),
        ]
        result = parse_response("Response text", tool_calls)
        message = result.to_message()
        assert message.role == "assistant"
        assert message.content == "Response text"
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1


# -----------------------------------------------------------------------------
# Tests: aggregate_streaming_events
# -----------------------------------------------------------------------------


class TestAggregateStreamingEvents:
    """Tests for aggregate_streaming_events function."""

    def test_content_delta_events(self) -> None:
        """Test aggregating content delta events."""
        events = [
            MockStreamEvent(type="content.delta", content="Hello"),
            MockStreamEvent(type="content.delta", content=" "),
            MockStreamEvent(type="content.delta", content="world!"),
        ]
        text, tool_calls = aggregate_streaming_events(events)
        assert text == "Hello world!"
        assert tool_calls == []

    def test_content_done_event(self) -> None:
        """Test that content.done events don't duplicate content."""
        events = [
            MockStreamEvent(type="content.delta", content="Hello"),
            MockStreamEvent(type="content.done", content="Hello"),
        ]
        text, _ = aggregate_streaming_events(events)
        # Only deltas contribute to text
        assert text == "Hello"

    def test_tool_call_delta_events(self) -> None:
        """Test aggregating tool call delta events."""
        events = [
            MockStreamEvent(
                type="tool_calls.function.arguments.delta",
                tool_name="read_file",
                tool_index=0,
                tool_call_id="call_1",
                arguments_delta='{"pa',
            ),
            MockStreamEvent(
                type="tool_calls.function.arguments.delta",
                tool_index=0,
                arguments_delta='th": "/test.txt"}',
            ),
        ]
        _, tool_calls = aggregate_streaming_events(events)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "read_file"
        assert tool_calls[0]["arguments"] == '{"path": "/test.txt"}'

    def test_tool_call_done_event(self) -> None:
        """Test tool_calls.function.arguments.done event."""
        events = [
            MockStreamEvent(
                type="tool_calls.function.arguments.done",
                tool_name="write_file",
                tool_index=0,
                tool_call_id="call_1",
                tool_arguments='{"path": "/out.txt", "content": "data"}',
            ),
        ]
        _, tool_calls = aggregate_streaming_events(events)
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "write_file"
        assert tool_calls[0]["id"] == "call_1"
        assert tool_calls[0]["arguments"] == '{"path": "/out.txt", "content": "data"}'

    def test_multiple_tool_calls(self) -> None:
        """Test aggregating multiple tool calls."""
        events = [
            MockStreamEvent(
                type="tool_calls.function.arguments.done",
                tool_name="tool_a",
                tool_index=0,
                tool_call_id="call_a",
                tool_arguments='{"x": 1}',
            ),
            MockStreamEvent(
                type="tool_calls.function.arguments.done",
                tool_name="tool_b",
                tool_index=1,
                tool_call_id="call_b",
                tool_arguments='{"y": 2}',
            ),
        ]
        _, tool_calls = aggregate_streaming_events(events)
        assert len(tool_calls) == 2
        assert tool_calls[0]["name"] == "tool_a"
        assert tool_calls[1]["name"] == "tool_b"

    def test_mixed_content_and_tool_calls(self) -> None:
        """Test aggregating mixed content and tool call events."""
        events = [
            MockStreamEvent(type="content.delta", content="Processing..."),
            MockStreamEvent(
                type="tool_calls.function.arguments.done",
                tool_name="process",
                tool_index=0,
                tool_call_id="call_p",
                tool_arguments="{}",
            ),
        ]
        text, tool_calls = aggregate_streaming_events(events)
        assert text == "Processing..."
        assert len(tool_calls) == 1

    def test_empty_events(self) -> None:
        """Test with empty event list."""
        text, tool_calls = aggregate_streaming_events([])
        assert text == ""
        assert tool_calls == []

    def test_none_content_in_delta(self) -> None:
        """Test that None content in deltas is handled."""
        events = [
            MockStreamEvent(type="content.delta", content=None),
            MockStreamEvent(type="content.delta", content="text"),
        ]
        text, _ = aggregate_streaming_events(events)
        assert text == "text"

    def test_tool_call_index_ordering(self) -> None:
        """Test that tool calls are ordered by index."""
        events = [
            MockStreamEvent(
                type="tool_calls.function.arguments.done",
                tool_name="second",
                tool_index=1,
                tool_call_id="c2",
                tool_arguments="{}",
            ),
            MockStreamEvent(
                type="tool_calls.function.arguments.done",
                tool_name="first",
                tool_index=0,
                tool_call_id="c1",
                tool_arguments="{}",
            ),
        ]
        _, tool_calls = aggregate_streaming_events(events)
        assert tool_calls[0]["name"] == "first"
        assert tool_calls[1]["name"] == "second"


# -----------------------------------------------------------------------------
# Tests: extract_tool_calls_from_events
# -----------------------------------------------------------------------------


class TestExtractToolCallsFromEvents:
    """Tests for extract_tool_calls_from_events function."""

    def test_extract_tool_calls(self) -> None:
        """Test extracting tool calls from events."""
        events = [
            MockStreamEvent(
                type="tool_calls.function.arguments.done",
                tool_name="test_tool",
                tool_index=0,
                tool_call_id="call_1",
                tool_arguments='{"arg": "value"}',
            ),
        ]
        result = extract_tool_calls_from_events(events)
        assert len(result) == 1
        assert isinstance(result[0], ParsedToolCall)
        assert result[0].name == "test_tool"
        assert result[0].call_id == "call_1"

    def test_empty_events_returns_empty_tuple(self) -> None:
        """Test that empty events return empty tuple."""
        result = extract_tool_calls_from_events([])
        assert result == ()

    def test_generates_call_id_if_missing(self) -> None:
        """Test that a call ID is generated if missing."""
        events = [
            MockStreamEvent(
                type="tool_calls.function.arguments.done",
                tool_name="tool",
                tool_index=0,
                tool_call_id="",  # Empty
                tool_arguments="{}",
            ),
        ]
        result = extract_tool_calls_from_events(events)
        assert len(result) == 1
        assert result[0].call_id.startswith("call_0_")


# -----------------------------------------------------------------------------
# Tests: merge_tool_calls
# -----------------------------------------------------------------------------


class TestMergeToolCalls:
    """Tests for merge_tool_calls function."""

    def test_no_duplicates_same_call(self) -> None:
        """Test that identical calls are not duplicated."""
        native = [
            ParsedToolCall(call_id="n1", name="read_file", arguments='{"path": "/a.txt"}', index=0),
        ]
        embedded = [
            ParsedToolCall(call_id="e1", name="read_file", arguments='{"path": "/a.txt"}', index=0),
        ]
        result = merge_tool_calls(native, embedded)
        assert len(result) == 1
        assert result[0].call_id == "n1"  # Native takes precedence

    def test_different_calls_merged(self) -> None:
        """Test that different calls are merged."""
        native = [
            ParsedToolCall(call_id="n1", name="tool_a", arguments='{"x": 1}', index=0),
        ]
        embedded = [
            ParsedToolCall(call_id="e1", name="tool_b", arguments='{"y": 2}', index=0),
        ]
        result = merge_tool_calls(native, embedded)
        assert len(result) == 2
        assert result[0].name == "tool_a"
        assert result[1].name == "tool_b"

    def test_same_name_different_args(self) -> None:
        """Test that same tool with different args is not deduplicated."""
        native = [
            ParsedToolCall(call_id="n1", name="read_file", arguments='{"path": "/a.txt"}', index=0),
        ]
        embedded = [
            ParsedToolCall(call_id="e1", name="read_file", arguments='{"path": "/b.txt"}', index=0),
        ]
        result = merge_tool_calls(native, embedded)
        assert len(result) == 2

    def test_empty_inputs(self) -> None:
        """Test with empty inputs."""
        result = merge_tool_calls([], [])
        assert result == ()

    def test_embedded_reindexed(self) -> None:
        """Test that embedded calls are reindexed after native."""
        native = [
            ParsedToolCall(call_id="n1", name="tool_a", arguments="{}", index=0),
            ParsedToolCall(call_id="n2", name="tool_b", arguments="{}", index=1),
        ]
        embedded = [
            ParsedToolCall(call_id="e1", name="tool_c", arguments="{}", index=0),
        ]
        result = merge_tool_calls(native, embedded)
        assert result[2].index == 2  # Reindexed from 0 to 2


# -----------------------------------------------------------------------------
# Tests: execute_model
# -----------------------------------------------------------------------------


class TestExecuteModel:
    """Tests for execute_model async function."""

    @pytest.mark.asyncio
    async def test_basic_text_response(self) -> None:
        """Test executing model with simple text response."""
        client = MockModelClient(
            events=[
                MockStreamEvent(type="content.delta", content="Hello, "),
                MockStreamEvent(type="content.delta", content="world!"),
            ]
        )
        turn = make_analyzed_turn()
        result = await execute_model(turn, client)
        assert result.text == "Hello, world!"
        assert result.tool_calls == ()

    @pytest.mark.asyncio
    async def test_response_with_tool_calls(self) -> None:
        """Test executing model with tool calls in response."""
        client = MockModelClient(
            events=[
                MockStreamEvent(type="content.delta", content="Let me check."),
                MockStreamEvent(
                    type="tool_calls.function.arguments.done",
                    tool_name="read_file",
                    tool_index=0,
                    tool_call_id="call_123",
                    tool_arguments='{"path": "/test.txt"}',
                ),
            ]
        )
        turn = make_analyzed_turn()
        result = await execute_model(turn, client)
        assert "Let me check" in result.text
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "read_file"

    @pytest.mark.asyncio
    async def test_messages_passed_to_client(self) -> None:
        """Test that messages are correctly passed to client."""
        client = MockModelClient(events=[])
        messages = [
            Message.system("System prompt"),
            Message.user("User message"),
        ]
        turn = make_analyzed_turn(messages=messages)
        await execute_model(turn, client)

        # Verify messages were passed
        assert len(client.last_call_args["messages"]) == 2
        assert client.last_call_args["messages"][0]["role"] == "system"
        assert client.last_call_args["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_hints_injected_into_messages(self) -> None:
        """Test that analysis hints are injected into messages."""
        client = MockModelClient(events=[])
        messages = [
            Message.system("Base system prompt"),
            Message.user("User message"),
        ]
        hints = ["Hint 1: Do X", "Hint 2: Avoid Y"]
        turn = make_analyzed_turn(messages=messages, hints=hints)
        await execute_model(turn, client)

        # Verify hints were injected into system message
        system_content = client.last_call_args["messages"][0]["content"]
        assert "Hint 1: Do X" in system_content
        assert "Hint 2: Avoid Y" in system_content

    @pytest.mark.asyncio
    async def test_tools_passed_to_client(self) -> None:
        """Test that tools parameter is passed to client."""
        client = MockModelClient(events=[])
        turn = make_analyzed_turn()
        tools = [{"type": "function", "function": {"name": "test"}}]
        await execute_model(turn, client, tools=tools)
        assert client.last_call_args["tools"] == tools

    @pytest.mark.asyncio
    async def test_temperature_passed_to_client(self) -> None:
        """Test that temperature parameter is passed to client."""
        client = MockModelClient(events=[])
        turn = make_analyzed_turn()
        await execute_model(turn, client, temperature=0.5)
        assert client.last_call_args["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_max_completion_tokens_passed(self) -> None:
        """Test that max_completion_tokens is passed to client."""
        client = MockModelClient(events=[])
        turn = make_analyzed_turn()
        await execute_model(turn, client, max_completion_tokens=1000)
        assert client.last_call_args["max_completion_tokens"] == 1000

    @pytest.mark.asyncio
    async def test_tool_choice_passed(self) -> None:
        """Test that tool_choice parameter is passed to client."""
        client = MockModelClient(events=[])
        turn = make_analyzed_turn()
        await execute_model(turn, client, tool_choice="auto")
        assert client.last_call_args["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_model_metadata_in_response(self) -> None:
        """Test that model name appears in response."""
        client = MockModelClient(
            events=[MockStreamEvent(type="content.delta", content="Response")]
        )
        turn = make_analyzed_turn()
        result = await execute_model(turn, client, model="gpt-4o")
        assert result.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_on_content_delta_callback(self) -> None:
        """Test that content delta callback is called."""
        client = MockModelClient(
            events=[
                MockStreamEvent(type="content.delta", content="Hello"),
                MockStreamEvent(type="content.delta", content=" World"),
            ]
        )
        turn = make_analyzed_turn()

        deltas: list[str] = []

        def capture_delta(delta: str) -> None:
            deltas.append(delta)

        await execute_model(turn, client, on_content_delta=capture_delta)
        assert deltas == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_callback_error_does_not_break_streaming(self) -> None:
        """Test that callback errors don't break streaming."""
        client = MockModelClient(
            events=[
                MockStreamEvent(type="content.delta", content="Part1"),
                MockStreamEvent(type="content.delta", content="Part2"),
            ]
        )
        turn = make_analyzed_turn()

        def bad_callback(delta: str) -> None:
            raise ValueError("Callback error!")

        # Should not raise
        result = await execute_model(turn, client, on_content_delta=bad_callback)
        assert result.text == "Part1Part2"

    @pytest.mark.asyncio
    async def test_embedded_markers_parsed(self) -> None:
        """Test that embedded tool markers are parsed."""
        text_with_markers = """Response text
<|tool_calls_begin|>
<|tool_call_begin|>embedded_tool<|tool_sep|>{"key": "value"}<|tool_call_end|>
<|tool_calls_end|>"""
        client = MockModelClient(
            events=[MockStreamEvent(type="content.delta", content=text_with_markers)]
        )
        turn = make_analyzed_turn()
        result = await execute_model(turn, client)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "embedded_tool"

    @pytest.mark.asyncio
    async def test_embedded_markers_disabled(self) -> None:
        """Test disabling embedded marker parsing."""
        text_with_markers = """<|tool_calls_begin|>
<|tool_call_begin|>tool<|tool_sep|>{}<|tool_call_end|>
<|tool_calls_end|>"""
        client = MockModelClient(
            events=[MockStreamEvent(type="content.delta", content=text_with_markers)]
        )
        turn = make_analyzed_turn()
        result = await execute_model(turn, client, parse_embedded_markers=False)
        assert result.tool_calls == ()


# -----------------------------------------------------------------------------
# Tests: Protocol Compliance
# -----------------------------------------------------------------------------


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_mock_stream_event_is_stream_event(self) -> None:
        """Test that MockStreamEvent conforms to StreamEvent protocol."""
        event = MockStreamEvent(type="content.delta", content="test")
        # Check attributes exist
        assert hasattr(event, "type")
        assert hasattr(event, "content")
        assert hasattr(event, "tool_name")
        assert hasattr(event, "tool_index")
        assert hasattr(event, "tool_arguments")
        assert hasattr(event, "arguments_delta")
        assert hasattr(event, "tool_call_id")

    def test_mock_model_client_is_model_client(self) -> None:
        """Test that MockModelClient has stream_chat method."""
        client = MockModelClient()
        assert hasattr(client, "stream_chat")
        assert callable(client.stream_chat)


# -----------------------------------------------------------------------------
# Tests: Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_empty_response(self) -> None:
        """Test handling empty response from model."""
        client = MockModelClient(events=[])
        turn = make_analyzed_turn()
        result = await execute_model(turn, client)
        assert result.text == ""
        assert result.tool_calls == ()

    @pytest.mark.asyncio
    async def test_only_tool_calls_no_content(self) -> None:
        """Test response with only tool calls and no content."""
        client = MockModelClient(
            events=[
                MockStreamEvent(
                    type="tool_calls.function.arguments.done",
                    tool_name="tool",
                    tool_index=0,
                    tool_call_id="c1",
                    tool_arguments="{}",
                ),
            ]
        )
        turn = make_analyzed_turn()
        result = await execute_model(turn, client)
        assert result.text == ""
        assert len(result.tool_calls) == 1

    def test_parse_response_with_none_tool_calls(self) -> None:
        """Test parse_response handles None-like empty sequences."""
        result = parse_response("text", tool_calls=[])
        assert result.tool_calls == ()

    def test_aggregate_unknown_event_type(self) -> None:
        """Test that unknown event types are ignored."""
        events = [
            MockStreamEvent(type="unknown.event", content="ignored"),
            MockStreamEvent(type="content.delta", content="kept"),
        ]
        text, _ = aggregate_streaming_events(events)
        assert text == "kept"

    def test_tool_call_with_none_index(self) -> None:
        """Test tool call event with None index defaults to 0."""
        events = [
            MockStreamEvent(
                type="tool_calls.function.arguments.done",
                tool_name="tool",
                tool_index=None,  # None index
                tool_call_id="c1",
                tool_arguments="{}",
            ),
        ]
        _, tool_calls = aggregate_streaming_events(events)
        assert len(tool_calls) == 1
        assert tool_calls[0]["index"] == 0
