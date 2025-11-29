"""Pipeline stage: Tools.

This module provides the tools stage of the turn pipeline, responsible for
executing tool calls from the model response and collecting results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from ..types import Message, ModelResponse, ParsedToolCall, ToolCallRecord

__all__ = [
    "ToolExecutor",
    "ToolExecutionResult",
    "ToolResults",
    "execute_tools",
    "execute_tool_call",
    "append_tool_results",
    "tool_call_to_record",
    "format_tool_result_content",
]

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Result Types
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ToolExecutionResult:
    """Result from executing a single tool call.

    Attributes:
        call_id: The ID of the tool call.
        name: Name of the tool that was called.
        success: Whether execution succeeded.
        result: The result data (stringified for message).
        error: Error message if failed.
        duration_ms: Execution time in milliseconds.
        raw_result: The raw result before stringification.
    """

    call_id: str
    name: str
    success: bool
    result: str
    error: str | None = None
    duration_ms: float = 0.0
    raw_result: Any = None

    @classmethod
    def from_success(
        cls,
        call_id: str,
        name: str,
        result: Any,
        duration_ms: float = 0.0,
    ) -> ToolExecutionResult:
        """Create a successful result."""
        return cls(
            call_id=call_id,
            name=name,
            success=True,
            result=format_tool_result_content(result),
            error=None,
            duration_ms=duration_ms,
            raw_result=result,
        )

    @classmethod
    def from_error(
        cls,
        call_id: str,
        name: str,
        error: str,
        duration_ms: float = 0.0,
    ) -> ToolExecutionResult:
        """Create a failed result."""
        return cls(
            call_id=call_id,
            name=name,
            success=False,
            result=f"Error: {error}",
            error=error,
            duration_ms=duration_ms,
            raw_result=None,
        )

    def to_record(self, arguments: str | Mapping[str, Any] = "") -> ToolCallRecord:
        """Convert to a ToolCallRecord for metrics/logging."""
        return ToolCallRecord(
            call_id=self.call_id,
            name=self.name,
            arguments=arguments,
            result=self.result,
            success=self.success,
            duration_ms=self.duration_ms,
            error=self.error,
        )


@dataclass(slots=True, frozen=True)
class ToolResults:
    """Collection of results from executing tool calls.

    Attributes:
        results: Individual tool execution results.
        total_duration_ms: Total time for all tool executions.
        success_count: Number of successful executions.
        error_count: Number of failed executions.
    """

    results: tuple[ToolExecutionResult, ...] = ()
    total_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple):
            object.__setattr__(self, "results", tuple(self.results))

    @property
    def success_count(self) -> int:
        """Count of successful tool executions."""
        return sum(1 for r in self.results if r.success)

    @property
    def error_count(self) -> int:
        """Count of failed tool executions."""
        return sum(1 for r in self.results if not r.success)

    @property
    def all_succeeded(self) -> bool:
        """Check if all tool calls succeeded."""
        return all(r.success for r in self.results)

    @property
    def has_errors(self) -> bool:
        """Check if any tool calls failed."""
        return any(not r.success for r in self.results)

    def to_records(
        self,
        tool_calls: Sequence[ParsedToolCall] = (),
    ) -> tuple[ToolCallRecord, ...]:
        """Convert results to ToolCallRecords with arguments.

        Args:
            tool_calls: Original tool calls to get arguments from.

        Returns:
            Tuple of ToolCallRecord objects.
        """
        # Build a lookup for arguments by call_id
        args_by_id: dict[str, str] = {tc.call_id: tc.arguments for tc in tool_calls}

        return tuple(
            r.to_record(args_by_id.get(r.call_id, ""))
            for r in self.results
        )


# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------


@runtime_checkable
class ToolExecutor(Protocol):
    """Protocol for tool execution implementations.

    Implementations must provide an execute method that runs a tool by name
    with the given arguments and returns the result.
    """

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> Any:
        """Execute a tool by name with arguments.

        Args:
            name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            call_id: Optional call ID for tracing.

        Returns:
            The tool result (any type, will be stringified).

        Raises:
            Exception: If tool execution fails.
        """
        ...


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def format_tool_result_content(result: Any) -> str:
    """Format a tool result for inclusion in a message.

    Converts various result types to a string suitable for the model.

    Args:
        result: The raw tool result.

    Returns:
        String representation of the result.
    """
    if result is None:
        return "null"

    if isinstance(result, str):
        return result

    if isinstance(result, bool):
        return "true" if result else "false"

    if isinstance(result, (int, float)):
        return str(result)

    if isinstance(result, (dict, list)):
        try:
            return json.dumps(result, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            return str(result)

    # Check for to_dict method
    if hasattr(result, "to_dict") and callable(result.to_dict):
        try:
            return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            pass

    return str(result)


def parse_tool_arguments(arguments: str) -> dict[str, Any]:
    """Parse tool arguments from JSON string.

    Args:
        arguments: JSON string of arguments.

    Returns:
        Parsed arguments dictionary.

    Raises:
        ValueError: If arguments cannot be parsed.
    """
    if not arguments or arguments.strip() in ("", "{}"):
        return {}

    try:
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError(f"Arguments must be a JSON object, got {type(parsed).__name__}")
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in tool arguments: {e}") from e


def tool_call_to_record(
    call: ParsedToolCall,
    result: ToolExecutionResult,
) -> ToolCallRecord:
    """Convert a tool call and its result to a ToolCallRecord.

    Args:
        call: The original tool call.
        result: The execution result.

    Returns:
        A ToolCallRecord for metrics/logging.
    """
    return ToolCallRecord(
        call_id=call.call_id,
        name=call.name,
        arguments=call.arguments,
        result=result.result,
        success=result.success,
        duration_ms=result.duration_ms,
        error=result.error,
    )


# -----------------------------------------------------------------------------
# Tool Execution
# -----------------------------------------------------------------------------


async def execute_tool_call(
    call: ParsedToolCall,
    executor: ToolExecutor,
    *,
    timeout_seconds: float | None = None,
) -> ToolExecutionResult:
    """Execute a single tool call.

    Args:
        call: The parsed tool call from the model.
        executor: The tool executor to use.
        timeout_seconds: Optional timeout for execution.

    Returns:
        The execution result (success or error).
    """
    start_time = time.perf_counter()

    try:
        # Parse arguments
        arguments = parse_tool_arguments(call.arguments)
    except ValueError as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        LOGGER.warning("Failed to parse arguments for tool %s: %s", call.name, e)
        return ToolExecutionResult.from_error(
            call_id=call.call_id,
            name=call.name,
            error=f"Invalid arguments: {e}",
            duration_ms=duration_ms,
        )

    try:
        # Execute with optional timeout
        if timeout_seconds is not None and timeout_seconds > 0:
            raw_result = await asyncio.wait_for(
                executor.execute(call.name, arguments, call_id=call.call_id),
                timeout=timeout_seconds,
            )
        else:
            raw_result = await executor.execute(
                call.name, arguments, call_id=call.call_id
            )

        duration_ms = (time.perf_counter() - start_time) * 1000
        return ToolExecutionResult.from_success(
            call_id=call.call_id,
            name=call.name,
            result=raw_result,
            duration_ms=duration_ms,
        )

    except asyncio.TimeoutError:
        duration_ms = (time.perf_counter() - start_time) * 1000
        LOGGER.warning("Tool %s timed out after %.1fs", call.name, timeout_seconds)
        return ToolExecutionResult.from_error(
            call_id=call.call_id,
            name=call.name,
            error=f"Tool execution timed out after {timeout_seconds}s",
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = (time.perf_counter() - start_time) * 1000
        error_msg = str(e) if str(e) else type(e).__name__
        LOGGER.warning("Tool %s failed: %s", call.name, error_msg)
        return ToolExecutionResult.from_error(
            call_id=call.call_id,
            name=call.name,
            error=error_msg,
            duration_ms=duration_ms,
        )


async def execute_tools(
    response: ModelResponse,
    executor: ToolExecutor,
    *,
    timeout_seconds: float | None = None,
    parallel: bool = False,
) -> ToolResults:
    """Execute all tool calls from a model response.

    Args:
        response: The model response containing tool calls.
        executor: The tool executor to use.
        timeout_seconds: Optional timeout per tool execution.
        parallel: Whether to execute tools in parallel (default: sequential).

    Returns:
        Collection of all tool execution results.
    """
    if not response.has_tool_calls:
        return ToolResults(results=(), total_duration_ms=0.0)

    start_time = time.perf_counter()
    results: list[ToolExecutionResult] = []

    if parallel:
        # Execute all tools concurrently
        tasks = [
            execute_tool_call(call, executor, timeout_seconds=timeout_seconds)
            for call in response.tool_calls
        ]
        results = list(await asyncio.gather(*tasks))
    else:
        # Execute tools sequentially (default - safer for tools with side effects)
        for call in response.tool_calls:
            result = await execute_tool_call(
                call, executor, timeout_seconds=timeout_seconds
            )
            results.append(result)

    total_duration_ms = (time.perf_counter() - start_time) * 1000
    return ToolResults(
        results=tuple(results),
        total_duration_ms=total_duration_ms,
    )


# -----------------------------------------------------------------------------
# Message Building
# -----------------------------------------------------------------------------


def append_tool_results(
    messages: Sequence[Message],
    response: ModelResponse,
    tool_results: ToolResults,
) -> tuple[Message, ...]:
    """Append tool results to the message list for the next model turn.

    Creates an assistant message with the tool calls, followed by tool result
    messages for each execution result.

    Args:
        messages: The current message list.
        response: The model response that triggered the tool calls.
        tool_results: The results from executing the tools.

    Returns:
        Updated message tuple with assistant and tool messages appended.
    """
    if not response.has_tool_calls:
        # No tool calls - just append assistant message if there's text
        if response.text:
            return tuple(messages) + (response.to_message(),)
        return tuple(messages)

    # Start with existing messages
    result_messages: list[Message] = list(messages)

    # Add assistant message with tool calls
    result_messages.append(response.to_message())

    # Add tool result messages
    for call, exec_result in zip(response.tool_calls, tool_results.results):
        result_messages.append(
            Message.tool(
                content=exec_result.result,
                tool_call_id=call.call_id,
                name=call.name,
            )
        )

    return tuple(result_messages)


def build_tool_result_message(
    call: ParsedToolCall,
    result: ToolExecutionResult,
) -> Message:
    """Build a single tool result message.

    Args:
        call: The original tool call.
        result: The execution result.

    Returns:
        A tool message for the conversation.
    """
    return Message.tool(
        content=result.result,
        tool_call_id=call.call_id,
        name=call.name,
    )
