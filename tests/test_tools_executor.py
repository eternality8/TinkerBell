"""Tests for orchestration/tools/executor.py."""

from __future__ import annotations

import asyncio
from typing import Any, Mapping

import pytest

from tinkerbell.ai.orchestration.tools import (
    ExecutorConfig,
    SimpleTool,
    ToolExecutionError,
    ToolExecutor,
    ToolNotFoundError,
    ToolRegistry,
    ToolSpec,
)


# -----------------------------------------------------------------------------
# Test Fixtures and Helpers
# -----------------------------------------------------------------------------


def make_spec(name: str = "test_tool", description: str = "A test tool") -> ToolSpec:
    """Helper to create a ToolSpec."""
    return ToolSpec(name=name, description=description)


def make_registry_with_tools() -> ToolRegistry:
    """Create a registry with some test tools."""
    registry = ToolRegistry()

    # Simple sync tool
    registry.register_function(
        spec=make_spec("echo", "Echo the input"),
        handler=lambda args: {"echo": args.get("message", "")},
    )

    # Async tool
    async def async_greet(args: dict) -> dict:
        return {"greeting": f"Hello, {args.get('name', 'World')}!"}

    registry.register_function(
        spec=make_spec("greet", "Greet someone"),
        handler=async_greet,
    )

    # Tool that raises
    def failing_tool(args: dict) -> None:
        raise ValueError("Intentional failure")

    registry.register_function(
        spec=make_spec("fail", "Always fails"),
        handler=failing_tool,
    )

    # Slow tool for timeout testing
    async def slow_tool(args: dict) -> dict:
        await asyncio.sleep(args.get("delay", 1.0))
        return {"status": "done"}

    registry.register_function(
        spec=make_spec("slow", "Slow tool"),
        handler=slow_tool,
    )

    return registry


# -----------------------------------------------------------------------------
# Tests: ExecutorConfig
# -----------------------------------------------------------------------------


class TestExecutorConfig:
    """Tests for ExecutorConfig."""

    def test_defaults(self) -> None:
        """Test default configuration."""
        config = ExecutorConfig()
        assert config.default_timeout == 30.0
        assert config.log_arguments is False
        assert config.log_results is False
        assert config.strict_mode is False

    def test_custom_values(self) -> None:
        """Test custom configuration."""
        config = ExecutorConfig(
            default_timeout=60.0,
            log_arguments=True,
            log_results=True,
            strict_mode=True,
        )
        assert config.default_timeout == 60.0
        assert config.log_arguments is True
        assert config.strict_mode is True

    def test_frozen(self) -> None:
        """Test that config is frozen."""
        config = ExecutorConfig()
        with pytest.raises(AttributeError):
            config.default_timeout = 100.0  # type: ignore


# -----------------------------------------------------------------------------
# Tests: ToolExecutor - Basic Execution
# -----------------------------------------------------------------------------


class TestToolExecutorBasic:
    """Tests for basic ToolExecutor operations."""

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self) -> None:
        """Test executing a synchronous tool."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        result = await executor.execute("echo", {"message": "test"})
        assert result["echo"] == "test"

    @pytest.mark.asyncio
    async def test_execute_async_tool(self) -> None:
        """Test executing an asynchronous tool."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        result = await executor.execute("greet", {"name": "Alice"})
        assert result["greeting"] == "Hello, Alice!"

    @pytest.mark.asyncio
    async def test_execute_with_call_id(self) -> None:
        """Test executing with call ID for tracing."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        result = await executor.execute(
            "echo",
            {"message": "traced"},
            call_id="trace_123",
        )
        assert result["echo"] == "traced"

    @pytest.mark.asyncio
    async def test_execute_empty_arguments(self) -> None:
        """Test executing with empty arguments."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        result = await executor.execute("echo", {})
        assert result["echo"] == ""


# -----------------------------------------------------------------------------
# Tests: ToolExecutor - Tool Not Found
# -----------------------------------------------------------------------------


class TestToolExecutorNotFound:
    """Tests for tool not found handling."""

    @pytest.mark.asyncio
    async def test_not_found_non_strict(self) -> None:
        """Test missing tool in non-strict mode returns error dict."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry, config=ExecutorConfig(strict_mode=False))
        result = await executor.execute("missing", {})
        assert result["error"] == "tool_not_found"

    @pytest.mark.asyncio
    async def test_not_found_strict_raises(self) -> None:
        """Test missing tool in strict mode raises."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry, config=ExecutorConfig(strict_mode=True))
        with pytest.raises(ToolNotFoundError):
            await executor.execute("missing", {})

    @pytest.mark.asyncio
    async def test_disabled_tool_not_found(self) -> None:
        """Test disabled tool is treated as not found."""
        registry = make_registry_with_tools()
        registry.disable("echo")
        executor = ToolExecutor(registry)
        result = await executor.execute("echo", {})
        assert result["error"] == "tool_not_found"


# -----------------------------------------------------------------------------
# Tests: ToolExecutor - Error Handling
# -----------------------------------------------------------------------------


class TestToolExecutorErrors:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_tool_exception_wrapped(self) -> None:
        """Test that tool exceptions are wrapped."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        with pytest.raises(ToolExecutionError) as exc_info:
            await executor.execute("fail", {})
        assert exc_info.value.tool_name == "fail"
        assert exc_info.value.cause is not None

    @pytest.mark.asyncio
    async def test_execution_error_message(self) -> None:
        """Test execution error contains message."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        with pytest.raises(ToolExecutionError) as exc_info:
            await executor.execute("fail", {})
        assert "Intentional failure" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Tests: ToolExecutor - Timeout
# -----------------------------------------------------------------------------


class TestToolExecutorTimeout:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_default(self) -> None:
        """Test using default timeout."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(
            registry,
            config=ExecutorConfig(default_timeout=0.05),
        )
        with pytest.raises(asyncio.TimeoutError):
            await executor.execute("slow", {"delay": 1.0})

    @pytest.mark.asyncio
    async def test_timeout_override(self) -> None:
        """Test timeout override per call."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(
            registry,
            config=ExecutorConfig(default_timeout=10.0),
        )
        with pytest.raises(asyncio.TimeoutError):
            await executor.execute("slow", {"delay": 1.0}, timeout=0.05)

    @pytest.mark.asyncio
    async def test_no_timeout(self) -> None:
        """Test execution without timeout."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(
            registry,
            config=ExecutorConfig(default_timeout=None),
        )
        # Should complete quickly without timeout
        result = await executor.execute("echo", {"message": "fast"})
        assert result["echo"] == "fast"


# -----------------------------------------------------------------------------
# Tests: ToolExecutor - execute_many
# -----------------------------------------------------------------------------


class TestToolExecutorMany:
    """Tests for execute_many."""

    @pytest.mark.asyncio
    async def test_execute_many_sequential(self) -> None:
        """Test executing multiple tools sequentially."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        calls = [
            ("echo", {"message": "one"}),
            ("echo", {"message": "two"}),
            ("greet", {"name": "Bob"}),
        ]
        results = await executor.execute_many(calls, parallel=False)
        assert len(results) == 3
        assert results[0]["echo"] == "one"
        assert results[1]["echo"] == "two"
        assert "Bob" in results[2]["greeting"]

    @pytest.mark.asyncio
    async def test_execute_many_parallel(self) -> None:
        """Test executing multiple tools in parallel."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        calls = [
            ("echo", {"message": "a"}),
            ("echo", {"message": "b"}),
        ]
        results = await executor.execute_many(calls, parallel=True)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_execute_many_with_failures(self) -> None:
        """Test execute_many captures failures."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        calls = [
            ("echo", {"message": "ok"}),
            ("fail", {}),
            ("echo", {"message": "also ok"}),
        ]
        results = await executor.execute_many(calls, parallel=False)
        assert results[0]["echo"] == "ok"
        assert isinstance(results[1], Exception)
        assert results[2]["echo"] == "also ok"


# -----------------------------------------------------------------------------
# Tests: ToolExecutor - Utility Methods
# -----------------------------------------------------------------------------


class TestToolExecutorUtility:
    """Tests for utility methods."""

    def test_has_tool_existing(self) -> None:
        """Test has_tool with existing tool."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        assert executor.has_tool("echo") is True

    def test_has_tool_missing(self) -> None:
        """Test has_tool with missing tool."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        assert executor.has_tool("nonexistent") is False

    def test_list_tools(self) -> None:
        """Test listing available tools."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        tools = executor.list_tools()
        assert "echo" in tools
        assert "greet" in tools

    def test_registry_property(self) -> None:
        """Test registry property access."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)
        assert executor.registry is registry

    def test_config_property(self) -> None:
        """Test config property access."""
        config = ExecutorConfig(default_timeout=60.0)
        executor = ToolExecutor(ToolRegistry(), config=config)
        assert executor.config.default_timeout == 60.0


# -----------------------------------------------------------------------------
# Tests: Integration with Pipeline Protocol
# -----------------------------------------------------------------------------


class TestToolExecutorProtocol:
    """Tests for pipeline ToolExecutor protocol compliance."""

    @pytest.mark.asyncio
    async def test_protocol_signature(self) -> None:
        """Test that execute() matches the protocol signature."""
        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)

        # Protocol requires: execute(name, arguments, *, call_id) -> Any
        result = await executor.execute(
            "echo",
            {"message": "protocol test"},
            call_id="test_call",
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_used_with_pipeline_tools(self) -> None:
        """Test executor can be used with pipeline.tools module."""
        from tinkerbell.ai.orchestration.pipeline.tools import execute_tool_call
        from tinkerbell.ai.orchestration.types import ParsedToolCall

        registry = make_registry_with_tools()
        executor = ToolExecutor(registry)

        call = ParsedToolCall(
            call_id="c1",
            name="echo",
            arguments='{"message": "integration test"}',
            index=0,
        )

        # execute_tool_call expects ToolExecutor protocol
        result = await execute_tool_call(call, executor)
        assert result.success is True
        assert "integration test" in result.result
