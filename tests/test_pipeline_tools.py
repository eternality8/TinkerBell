"""Tests for orchestration/pipeline/tools.py."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping
from unittest.mock import MagicMock

import pytest

from tinkerbell.ai.orchestration.pipeline.tools import (
    ToolExecutionResult,
    ToolExecutor,
    ToolResults,
    append_tool_results,
    build_tool_result_message,
    execute_tool_call,
    execute_tools,
    format_tool_result_content,
    parse_tool_arguments,
    tool_call_to_record,
)
from tinkerbell.ai.orchestration.types import (
    Message,
    ModelResponse,
    ParsedToolCall,
    ToolCallRecord,
)


# -----------------------------------------------------------------------------
# Test Fixtures and Helpers
# -----------------------------------------------------------------------------


class MockToolExecutor:
    """Mock tool executor for testing."""

    def __init__(
        self,
        results: dict[str, Any] | None = None,
        errors: dict[str, Exception] | None = None,
        delay_seconds: float = 0.0,
    ) -> None:
        self.results = results or {}
        self.errors = errors or {}
        self.delay_seconds = delay_seconds
        self.calls: list[tuple[str, dict[str, Any], str]] = []

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> Any:
        self.calls.append((name, dict(arguments), call_id))

        if self.delay_seconds > 0:
            await asyncio.sleep(self.delay_seconds)

        if name in self.errors:
            raise self.errors[name]

        return self.results.get(name, f"Result for {name}")


def make_tool_call(
    name: str = "test_tool",
    arguments: str = "{}",
    call_id: str = "call_1",
    index: int = 0,
) -> ParsedToolCall:
    """Helper to create a ParsedToolCall."""
    return ParsedToolCall(
        call_id=call_id,
        name=name,
        arguments=arguments,
        index=index,
    )


def make_model_response(
    text: str = "",
    tool_calls: list[ParsedToolCall] | None = None,
) -> ModelResponse:
    """Helper to create a ModelResponse."""
    return ModelResponse(
        text=text,
        tool_calls=tuple(tool_calls) if tool_calls else (),
    )


# -----------------------------------------------------------------------------
# Tests: format_tool_result_content
# -----------------------------------------------------------------------------


class TestFormatToolResultContent:
    """Tests for format_tool_result_content function."""

    def test_none_returns_null(self) -> None:
        """Test None is formatted as 'null'."""
        assert format_tool_result_content(None) == "null"

    def test_string_returned_as_is(self) -> None:
        """Test strings are returned unchanged."""
        assert format_tool_result_content("hello") == "hello"

    def test_bool_true(self) -> None:
        """Test True is formatted as 'true'."""
        assert format_tool_result_content(True) == "true"

    def test_bool_false(self) -> None:
        """Test False is formatted as 'false'."""
        assert format_tool_result_content(False) == "false"

    def test_int_formatted(self) -> None:
        """Test integers are formatted as strings."""
        assert format_tool_result_content(42) == "42"

    def test_float_formatted(self) -> None:
        """Test floats are formatted as strings."""
        assert format_tool_result_content(3.14) == "3.14"

    def test_dict_formatted_as_json(self) -> None:
        """Test dicts are formatted as JSON."""
        result = format_tool_result_content({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_list_formatted_as_json(self) -> None:
        """Test lists are formatted as JSON."""
        result = format_tool_result_content([1, 2, 3])
        assert "[" in result
        assert "1" in result

    def test_object_with_to_dict(self) -> None:
        """Test objects with to_dict method are serialized."""
        obj = MagicMock()
        obj.to_dict.return_value = {"data": "test"}
        result = format_tool_result_content(obj)
        assert '"data"' in result
        assert '"test"' in result

    def test_fallback_to_str(self) -> None:
        """Test fallback to str() for unknown types."""

        class CustomClass:
            def __str__(self) -> str:
                return "custom_repr"

        result = format_tool_result_content(CustomClass())
        assert result == "custom_repr"


# -----------------------------------------------------------------------------
# Tests: parse_tool_arguments
# -----------------------------------------------------------------------------


class TestParseToolArguments:
    """Tests for parse_tool_arguments function."""

    def test_empty_string_returns_empty_dict(self) -> None:
        """Test empty string returns empty dict."""
        assert parse_tool_arguments("") == {}

    def test_empty_object_string(self) -> None:
        """Test '{}' returns empty dict."""
        assert parse_tool_arguments("{}") == {}

    def test_valid_json_parsed(self) -> None:
        """Test valid JSON is parsed correctly."""
        result = parse_tool_arguments('{"key": "value", "num": 42}')
        assert result == {"key": "value", "num": 42}

    def test_invalid_json_raises(self) -> None:
        """Test invalid JSON raises ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_tool_arguments("{not valid json}")

    def test_non_object_json_raises(self) -> None:
        """Test non-object JSON raises ValueError."""
        with pytest.raises(ValueError, match="must be a JSON object"):
            parse_tool_arguments("[1, 2, 3]")

    def test_whitespace_only_returns_empty(self) -> None:
        """Test whitespace-only string returns empty dict."""
        assert parse_tool_arguments("   ") == {}


# -----------------------------------------------------------------------------
# Tests: ToolExecutionResult
# -----------------------------------------------------------------------------


class TestToolExecutionResult:
    """Tests for ToolExecutionResult dataclass."""

    def test_from_success(self) -> None:
        """Test creating success result."""
        result = ToolExecutionResult.from_success(
            call_id="c1",
            name="test",
            result={"data": "value"},
            duration_ms=100.0,
        )
        assert result.success is True
        assert result.call_id == "c1"
        assert result.name == "test"
        assert result.error is None
        assert '"data"' in result.result
        assert result.duration_ms == 100.0

    def test_from_error(self) -> None:
        """Test creating error result."""
        result = ToolExecutionResult.from_error(
            call_id="c1",
            name="test",
            error="Something went wrong",
            duration_ms=50.0,
        )
        assert result.success is False
        assert result.error == "Something went wrong"
        assert "Error:" in result.result
        assert result.duration_ms == 50.0

    def test_to_record(self) -> None:
        """Test converting to ToolCallRecord."""
        result = ToolExecutionResult.from_success(
            call_id="c1",
            name="test",
            result="ok",
        )
        record = result.to_record(arguments='{"x": 1}')
        assert isinstance(record, ToolCallRecord)
        assert record.call_id == "c1"
        assert record.name == "test"
        assert record.arguments == '{"x": 1}'
        assert record.success is True


# -----------------------------------------------------------------------------
# Tests: ToolResults
# -----------------------------------------------------------------------------


class TestToolResults:
    """Tests for ToolResults dataclass."""

    def test_empty_results(self) -> None:
        """Test empty results."""
        results = ToolResults()
        assert results.results == ()
        assert results.success_count == 0
        assert results.error_count == 0
        assert results.all_succeeded is True
        assert results.has_errors is False

    def test_success_count(self) -> None:
        """Test counting successful results."""
        results = ToolResults(
            results=(
                ToolExecutionResult.from_success("c1", "t1", "r1"),
                ToolExecutionResult.from_success("c2", "t2", "r2"),
                ToolExecutionResult.from_error("c3", "t3", "err"),
            )
        )
        assert results.success_count == 2
        assert results.error_count == 1
        assert results.all_succeeded is False
        assert results.has_errors is True

    def test_all_succeeded(self) -> None:
        """Test all_succeeded property."""
        results = ToolResults(
            results=(
                ToolExecutionResult.from_success("c1", "t1", "r1"),
                ToolExecutionResult.from_success("c2", "t2", "r2"),
            )
        )
        assert results.all_succeeded is True
        assert results.has_errors is False

    def test_to_records(self) -> None:
        """Test converting to ToolCallRecords."""
        tool_calls = [
            make_tool_call(name="tool_a", call_id="c1", arguments='{"a": 1}'),
            make_tool_call(name="tool_b", call_id="c2", arguments='{"b": 2}'),
        ]
        results = ToolResults(
            results=(
                ToolExecutionResult.from_success("c1", "tool_a", "result_a"),
                ToolExecutionResult.from_success("c2", "tool_b", "result_b"),
            )
        )
        records = results.to_records(tool_calls)
        assert len(records) == 2
        assert records[0].arguments == '{"a": 1}'
        assert records[1].arguments == '{"b": 2}'

    def test_results_coerced_to_tuple(self) -> None:
        """Test that list results are coerced to tuple."""
        results = ToolResults(
            results=[ToolExecutionResult.from_success("c1", "t1", "r1")]  # type: ignore
        )
        assert isinstance(results.results, tuple)


# -----------------------------------------------------------------------------
# Tests: execute_tool_call
# -----------------------------------------------------------------------------


class TestExecuteToolCall:
    """Tests for execute_tool_call function."""

    @pytest.mark.asyncio
    async def test_successful_execution(self) -> None:
        """Test successful tool execution."""
        executor = MockToolExecutor(results={"read_file": {"content": "hello"}})
        call = make_tool_call(
            name="read_file",
            arguments='{"path": "/test.txt"}',
            call_id="c1",
        )
        result = await execute_tool_call(call, executor)
        assert result.success is True
        assert result.call_id == "c1"
        assert result.name == "read_file"
        assert '"content"' in result.result

    @pytest.mark.asyncio
    async def test_execution_error(self) -> None:
        """Test handling execution error."""
        executor = MockToolExecutor(errors={"bad_tool": ValueError("Tool failed")})
        call = make_tool_call(name="bad_tool", call_id="c1")
        result = await execute_tool_call(call, executor)
        assert result.success is False
        assert "Tool failed" in result.error

    @pytest.mark.asyncio
    async def test_invalid_arguments(self) -> None:
        """Test handling invalid JSON arguments."""
        executor = MockToolExecutor()
        call = make_tool_call(name="tool", arguments="{invalid json}")
        result = await execute_tool_call(call, executor)
        assert result.success is False
        assert "Invalid" in result.error

    @pytest.mark.asyncio
    async def test_timeout(self) -> None:
        """Test handling timeout."""
        executor = MockToolExecutor(delay_seconds=1.0)
        call = make_tool_call(name="slow_tool")
        result = await execute_tool_call(call, executor, timeout_seconds=0.01)
        assert result.success is False
        assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_arguments_passed_to_executor(self) -> None:
        """Test that parsed arguments are passed to executor."""
        executor = MockToolExecutor()
        call = make_tool_call(
            name="tool",
            arguments='{"key": "value", "num": 42}',
            call_id="test_call",
        )
        await execute_tool_call(call, executor)
        assert len(executor.calls) == 1
        name, args, call_id = executor.calls[0]
        assert name == "tool"
        assert args == {"key": "value", "num": 42}
        assert call_id == "test_call"

    @pytest.mark.asyncio
    async def test_duration_recorded(self) -> None:
        """Test that execution duration is recorded."""
        executor = MockToolExecutor(delay_seconds=0.01)
        call = make_tool_call(name="tool")
        result = await execute_tool_call(call, executor)
        assert result.duration_ms > 0

    @pytest.mark.asyncio
    async def test_empty_error_message_uses_type_name(self) -> None:
        """Test that empty exception message falls back to type name."""

        class CustomError(Exception):
            pass

        executor = MockToolExecutor(errors={"tool": CustomError()})
        call = make_tool_call(name="tool")
        result = await execute_tool_call(call, executor)
        assert result.success is False
        assert "CustomError" in result.error


# -----------------------------------------------------------------------------
# Tests: execute_tools
# -----------------------------------------------------------------------------


class TestExecuteTools:
    """Tests for execute_tools function."""

    @pytest.mark.asyncio
    async def test_no_tool_calls(self) -> None:
        """Test with response containing no tool calls."""
        executor = MockToolExecutor()
        response = make_model_response(text="Just text")
        results = await execute_tools(response, executor)
        assert results.results == ()
        assert results.success_count == 0

    @pytest.mark.asyncio
    async def test_single_tool_call(self) -> None:
        """Test executing single tool call."""
        executor = MockToolExecutor(results={"tool_a": "result_a"})
        response = make_model_response(
            tool_calls=[make_tool_call(name="tool_a", call_id="c1")]
        )
        results = await execute_tools(response, executor)
        assert len(results.results) == 1
        assert results.results[0].name == "tool_a"
        assert results.success_count == 1

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_sequential(self) -> None:
        """Test executing multiple tool calls sequentially."""
        executor = MockToolExecutor(
            results={"tool_a": "r1", "tool_b": "r2", "tool_c": "r3"}
        )
        response = make_model_response(
            tool_calls=[
                make_tool_call(name="tool_a", call_id="c1"),
                make_tool_call(name="tool_b", call_id="c2"),
                make_tool_call(name="tool_c", call_id="c3"),
            ]
        )
        results = await execute_tools(response, executor, parallel=False)
        assert len(results.results) == 3
        assert results.all_succeeded is True

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_parallel(self) -> None:
        """Test executing multiple tool calls in parallel."""
        executor = MockToolExecutor(
            results={"tool_a": "r1", "tool_b": "r2"}
        )
        response = make_model_response(
            tool_calls=[
                make_tool_call(name="tool_a", call_id="c1"),
                make_tool_call(name="tool_b", call_id="c2"),
            ]
        )
        results = await execute_tools(response, executor, parallel=True)
        assert len(results.results) == 2
        assert results.all_succeeded is True

    @pytest.mark.asyncio
    async def test_mixed_success_and_error(self) -> None:
        """Test with mix of successful and failed executions."""
        executor = MockToolExecutor(
            results={"good_tool": "ok"},
            errors={"bad_tool": RuntimeError("Failed")},
        )
        response = make_model_response(
            tool_calls=[
                make_tool_call(name="good_tool", call_id="c1"),
                make_tool_call(name="bad_tool", call_id="c2"),
            ]
        )
        results = await execute_tools(response, executor)
        assert results.success_count == 1
        assert results.error_count == 1
        assert results.has_errors is True

    @pytest.mark.asyncio
    async def test_total_duration_recorded(self) -> None:
        """Test that total duration is recorded."""
        executor = MockToolExecutor(delay_seconds=0.01)
        response = make_model_response(
            tool_calls=[make_tool_call(name="tool", call_id="c1")]
        )
        results = await execute_tools(response, executor)
        assert results.total_duration_ms > 0

    @pytest.mark.asyncio
    async def test_timeout_passed_to_calls(self) -> None:
        """Test that timeout is passed to individual calls."""
        executor = MockToolExecutor(delay_seconds=1.0)
        response = make_model_response(
            tool_calls=[make_tool_call(name="slow", call_id="c1")]
        )
        results = await execute_tools(response, executor, timeout_seconds=0.01)
        assert results.error_count == 1
        assert "timed out" in results.results[0].error


# -----------------------------------------------------------------------------
# Tests: append_tool_results
# -----------------------------------------------------------------------------


class TestAppendToolResults:
    """Tests for append_tool_results function."""

    def test_no_tool_calls_with_text(self) -> None:
        """Test response with no tool calls but with text."""
        messages = [Message.user("Hello")]
        response = make_model_response(text="Hi there!")
        tool_results = ToolResults()
        result = append_tool_results(messages, response, tool_results)
        assert len(result) == 2
        assert result[1].role == "assistant"
        assert result[1].content == "Hi there!"

    def test_no_tool_calls_no_text(self) -> None:
        """Test response with no tool calls and no text."""
        messages = [Message.user("Hello")]
        response = make_model_response(text="")
        tool_results = ToolResults()
        result = append_tool_results(messages, response, tool_results)
        assert len(result) == 1  # Original message only

    def test_with_tool_calls(self) -> None:
        """Test response with tool calls."""
        messages = [Message.user("Read file")]
        tool_calls = [
            make_tool_call(name="read_file", call_id="c1"),
        ]
        response = make_model_response(text="Reading...", tool_calls=tool_calls)
        tool_results = ToolResults(
            results=(ToolExecutionResult.from_success("c1", "read_file", "file content"),)
        )
        result = append_tool_results(messages, response, tool_results)

        # Should have: user + assistant + tool
        assert len(result) == 3
        assert result[0].role == "user"
        assert result[1].role == "assistant"
        assert result[1].tool_calls is not None
        assert result[2].role == "tool"
        assert result[2].tool_call_id == "c1"

    def test_multiple_tool_calls(self) -> None:
        """Test response with multiple tool calls."""
        messages = []
        tool_calls = [
            make_tool_call(name="tool_a", call_id="c1"),
            make_tool_call(name="tool_b", call_id="c2"),
        ]
        response = make_model_response(text="", tool_calls=tool_calls)
        tool_results = ToolResults(
            results=(
                ToolExecutionResult.from_success("c1", "tool_a", "result_a"),
                ToolExecutionResult.from_success("c2", "tool_b", "result_b"),
            )
        )
        result = append_tool_results(messages, response, tool_results)

        # Should have: assistant + tool + tool
        assert len(result) == 3
        assert result[0].role == "assistant"
        assert result[1].role == "tool"
        assert result[1].tool_call_id == "c1"
        assert result[2].role == "tool"
        assert result[2].tool_call_id == "c2"

    def test_preserves_original_messages(self) -> None:
        """Test that original messages are preserved."""
        messages = [
            Message.system("System prompt"),
            Message.user("User message"),
            Message.assistant("Assistant response"),
        ]
        response = make_model_response(text="New response")
        tool_results = ToolResults()
        result = append_tool_results(messages, response, tool_results)
        assert len(result) == 4
        assert result[0].content == "System prompt"
        assert result[1].content == "User message"
        assert result[2].content == "Assistant response"

    def test_returns_tuple(self) -> None:
        """Test that result is a tuple."""
        messages = [Message.user("Test")]
        response = make_model_response(text="Response")
        tool_results = ToolResults()
        result = append_tool_results(messages, response, tool_results)
        assert isinstance(result, tuple)


# -----------------------------------------------------------------------------
# Tests: build_tool_result_message
# -----------------------------------------------------------------------------


class TestBuildToolResultMessage:
    """Tests for build_tool_result_message function."""

    def test_basic_message(self) -> None:
        """Test building basic tool result message."""
        call = make_tool_call(name="test_tool", call_id="c1")
        result = ToolExecutionResult.from_success("c1", "test_tool", "success data")
        message = build_tool_result_message(call, result)
        assert message.role == "tool"
        assert message.tool_call_id == "c1"
        assert message.name == "test_tool"
        assert "success data" in message.content

    def test_error_message(self) -> None:
        """Test building error tool result message."""
        call = make_tool_call(name="failing_tool", call_id="c2")
        result = ToolExecutionResult.from_error("c2", "failing_tool", "Something broke")
        message = build_tool_result_message(call, result)
        assert message.role == "tool"
        assert "Error:" in message.content


# -----------------------------------------------------------------------------
# Tests: tool_call_to_record
# -----------------------------------------------------------------------------


class TestToolCallToRecord:
    """Tests for tool_call_to_record function."""

    def test_success_record(self) -> None:
        """Test creating record from successful execution."""
        call = make_tool_call(
            name="tool",
            arguments='{"key": "value"}',
            call_id="c1",
        )
        result = ToolExecutionResult.from_success(
            "c1", "tool", "result data", duration_ms=50.0
        )
        record = tool_call_to_record(call, result)
        assert record.call_id == "c1"
        assert record.name == "tool"
        assert record.arguments == '{"key": "value"}'
        assert record.success is True
        assert record.duration_ms == 50.0

    def test_error_record(self) -> None:
        """Test creating record from failed execution."""
        call = make_tool_call(name="tool", call_id="c1")
        result = ToolExecutionResult.from_error(
            "c1", "tool", "Failed!", duration_ms=10.0
        )
        record = tool_call_to_record(call, result)
        assert record.success is False
        assert record.error == "Failed!"


# -----------------------------------------------------------------------------
# Tests: Protocol Compliance
# -----------------------------------------------------------------------------


class TestProtocolCompliance:
    """Tests for protocol compliance."""

    def test_mock_executor_is_tool_executor(self) -> None:
        """Test that MockToolExecutor has execute method."""
        executor = MockToolExecutor()
        assert hasattr(executor, "execute")
        assert callable(executor.execute)

    @pytest.mark.asyncio
    async def test_executor_signature(self) -> None:
        """Test that executor has correct signature."""
        executor = MockToolExecutor()
        # Should accept name, arguments, and optional call_id
        result = await executor.execute("test", {"arg": 1}, call_id="c1")
        assert result is not None


# -----------------------------------------------------------------------------
# Tests: Edge Cases
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_arguments_string(self) -> None:
        """Test tool call with empty arguments."""
        executor = MockToolExecutor(results={"tool": "ok"})
        call = make_tool_call(name="tool", arguments="")
        result = await execute_tool_call(call, executor)
        assert result.success is True
        # Empty args should be passed as empty dict
        assert executor.calls[0][1] == {}

    @pytest.mark.asyncio
    async def test_nested_json_arguments(self) -> None:
        """Test tool call with nested JSON arguments."""
        executor = MockToolExecutor(results={"tool": "ok"})
        call = make_tool_call(
            name="tool",
            arguments='{"nested": {"key": [1, 2, 3]}}',
        )
        result = await execute_tool_call(call, executor)
        assert result.success is True
        assert executor.calls[0][1] == {"nested": {"key": [1, 2, 3]}}

    def test_format_unicode_content(self) -> None:
        """Test formatting unicode content."""
        result = format_tool_result_content({"emoji": "🎉", "chinese": "你好"})
        assert "🎉" in result
        assert "你好" in result

    @pytest.mark.asyncio
    async def test_tool_returns_none(self) -> None:
        """Test tool that returns None."""
        executor = MockToolExecutor(results={"tool": None})
        call = make_tool_call(name="tool")
        result = await execute_tool_call(call, executor)
        assert result.success is True
        assert result.result == "null"
