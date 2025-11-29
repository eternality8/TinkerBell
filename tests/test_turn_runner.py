"""Tests for the turn runner module.

This module tests the TurnRunner class that orchestrates the complete
turn pipeline with support for the tool loop.
"""

from __future__ import annotations

import asyncio
import json
import pytest
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
from collections.abc import AsyncIterator

from tinkerbell.ai.orchestration.runner import (
    TurnRunner,
    RunnerConfig,
    ContentCallback,
    ToolCallback,
    create_runner,
)
from tinkerbell.ai.orchestration.services import Services, create_services
from tinkerbell.ai.orchestration.types import (
    DocumentSnapshot,
    Message,
    ModelResponse,
    ParsedToolCall,
    PreparedTurn,
    ToolCallRecord,
    TurnConfig,
    TurnInput,
    TurnMetrics,
    TurnOutput,
)


# =============================================================================
# Test Fixtures and Mocks
# =============================================================================


@dataclass
class MockStreamEvent:
    """Mock stream event for testing."""

    type: str = "content.delta"
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
        responses: Sequence[Sequence[MockStreamEvent]] | None = None,
    ):
        """Initialize with a sequence of response event sequences."""
        self.responses = list(responses or [])
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    def set_responses(self, responses: Sequence[Sequence[MockStreamEvent]]):
        """Set the sequence of responses to return."""
        self.responses = list(responses)
        self.call_count = 0

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
        """Mock streaming chat completion."""
        self.calls.append({
            "messages": list(messages),
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            **kwargs,
        })

        if self.call_count < len(self.responses):
            events = self.responses[self.call_count]
            self.call_count += 1
            for event in events:
                yield event
        else:
            # Default response - simple content
            yield MockStreamEvent(type="content.delta", content="Default response")


class MockToolExecutor:
    """Mock tool executor for testing."""

    def __init__(
        self,
        results: Mapping[str, Any] | None = None,
        raise_error: bool = False,
    ):
        """Initialize with tool results."""
        self.results = results or {}
        self.raise_error = raise_error
        self.calls: list[tuple[str, Any, str]] = []

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        call_id: str = "",
    ) -> Any:
        """Mock tool execution."""
        self.calls.append((name, arguments, call_id))

        if self.raise_error:
            raise ValueError(f"Tool {name} failed")

        if name in self.results:
            return self.results[name]
        return f"Result for {name}"


@pytest.fixture
def snapshot() -> DocumentSnapshot:
    """Create a document snapshot for testing."""
    return DocumentSnapshot(
        tab_id="test-tab-123",
        content="Test document content",
        version_token="v1",
    )


@pytest.fixture
def turn_input(snapshot: DocumentSnapshot) -> TurnInput:
    """Create a basic turn input for testing."""
    return TurnInput(
        prompt="Hello, assistant!",
        snapshot=snapshot,
        run_id="test-run-001",
    )


@pytest.fixture
def mock_client() -> MockModelClient:
    """Create a mock model client."""
    return MockModelClient()


@pytest.fixture
def mock_executor() -> MockToolExecutor:
    """Create a mock tool executor."""
    return MockToolExecutor()


def make_content_events(content: str) -> list[MockStreamEvent]:
    """Create content streaming events."""
    return [MockStreamEvent(type="content.delta", content=content)]


def make_tool_call_events(
    tool_name: str,
    arguments: dict[str, Any],
    call_id: str = "call_123",
    index: int = 0,
) -> list[MockStreamEvent]:
    """Create tool call streaming events."""
    return [
        MockStreamEvent(
            type="tool_calls.function.arguments.done",
            tool_name=tool_name,
            tool_arguments=json.dumps(arguments),
            tool_index=index,
            tool_call_id=call_id,
        ),
    ]


# =============================================================================
# TurnRunner Construction Tests
# =============================================================================


class TestTurnRunnerConstruction:
    """Tests for TurnRunner initialization."""

    def test_minimal_construction(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """Should construct with minimal parameters."""
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        assert runner.client is mock_client
        assert runner.tool_executor is mock_executor
        assert runner.config == RunnerConfig()
        assert runner.tools == ()

    def test_construction_with_config(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """Should construct with custom config."""
        config = RunnerConfig(
            max_iterations=5,
            streaming_enabled=True,
            tool_timeout=60.0,
        )

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            config=config,
        )

        assert runner.config.max_iterations == 5
        assert runner.config.streaming_enabled is True
        assert runner.config.tool_timeout == 60.0

    def test_construction_with_tools(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """Should construct with tool definitions."""
        tools = [
            {"type": "function", "function": {"name": "test_tool"}},
        ]

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            tools=tools,
        )

        assert len(runner.tools) == 1
        assert runner.tools[0]["function"]["name"] == "test_tool"

    def test_tools_are_immutable(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """Tools should be stored as immutable tuple."""
        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            tools=tools,
        )

        assert isinstance(runner.tools, tuple)


# =============================================================================
# Runner Configuration Tests
# =============================================================================


class TestRunnerConfig:
    """Tests for RunnerConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = RunnerConfig()

        assert config.max_iterations is None
        assert config.streaming_enabled is None
        assert config.tool_timeout == 30.0
        assert config.log_pipeline_stages is True
        assert config.raise_on_tool_error is False
        assert config.allow_empty_prompt is False

    def test_custom_values(self):
        """Should accept custom values."""
        config = RunnerConfig(
            max_iterations=10,
            streaming_enabled=False,
            tool_timeout=120.0,
            log_pipeline_stages=False,
            raise_on_tool_error=True,
            allow_empty_prompt=True,
        )

        assert config.max_iterations == 10
        assert config.streaming_enabled is False
        assert config.tool_timeout == 120.0
        assert config.log_pipeline_stages is False
        assert config.raise_on_tool_error is True
        assert config.allow_empty_prompt is True

    def test_frozen_dataclass(self):
        """Config should be frozen."""
        config = RunnerConfig()

        with pytest.raises(AttributeError):
            config.max_iterations = 5  # type: ignore


# =============================================================================
# with_* Method Tests
# =============================================================================


class TestRunnerWithMethods:
    """Tests for with_* methods that create new runners."""

    def test_with_tools_returns_new_runner(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """with_tools should return a new runner with updated tools."""
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        tools = [{"type": "function", "function": {"name": "new_tool"}}]
        new_runner = runner.with_tools(tools)

        assert new_runner is not runner
        assert len(new_runner.tools) == 1
        assert runner.tools == ()

    def test_with_tools_preserves_other_settings(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """with_tools should preserve other runner settings."""
        config = RunnerConfig(max_iterations=10)
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            config=config,
        )

        tools = [{"type": "function", "function": {"name": "new_tool"}}]
        new_runner = runner.with_tools(tools)

        assert new_runner.client is mock_client
        assert new_runner.tool_executor is mock_executor
        assert new_runner.config.max_iterations == 10

    def test_with_config_returns_new_runner(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """with_config should return a new runner with updated config."""
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        new_config = RunnerConfig(max_iterations=20)
        new_runner = runner.with_config(new_config)

        assert new_runner is not runner
        assert new_runner.config.max_iterations == 20
        assert runner.config.max_iterations is None

    def test_with_config_preserves_tools(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """with_config should preserve tools."""
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            tools=tools,
        )

        new_config = RunnerConfig(max_iterations=20)
        new_runner = runner.with_config(new_config)

        assert len(new_runner.tools) == 1

    def test_services_property(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """services property should return the services container."""
        services = create_services()
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            services=services,
        )

        assert runner.services is services

    def test_services_default_none(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """services should be None by default."""
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        assert runner.services is None

    def test_with_services_returns_new_runner(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """with_services should return a new runner with updated services."""
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        services = create_services()
        new_runner = runner.with_services(services)

        assert new_runner is not runner
        assert new_runner.services is services
        assert runner.services is None

    def test_with_services_preserves_other_settings(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """with_services should preserve other runner settings."""
        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        config = RunnerConfig(max_iterations=10)
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            config=config,
            tools=tools,
        )

        services = create_services()
        new_runner = runner.with_services(services)

        assert new_runner.client is mock_client
        assert new_runner.tool_executor is mock_executor
        assert new_runner.config.max_iterations == 10
        assert len(new_runner.tools) == 1

    def test_with_tools_preserves_services(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """with_tools should preserve services."""
        services = create_services()
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            services=services,
        )

        tools = [{"type": "function", "function": {"name": "new_tool"}}]
        new_runner = runner.with_tools(tools)

        assert new_runner.services is services

    def test_with_config_preserves_services(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """with_config should preserve services."""
        services = create_services()
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            services=services,
        )

        new_config = RunnerConfig(max_iterations=20)
        new_runner = runner.with_config(new_config)

        assert new_runner.services is services


# =============================================================================
# Basic Run Tests
# =============================================================================


class TestBasicRun:
    """Tests for basic turn execution without tools."""

    @pytest.mark.asyncio
    async def test_simple_run_returns_output(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Simple run should return TurnOutput."""
        mock_client.set_responses([
            make_content_events("Hello! How can I help?"),
        ])

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        output = await runner.run(turn_input)

        assert isinstance(output, TurnOutput)
        assert output.success is True
        assert output.response == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_run_with_empty_response(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Run should handle empty response."""
        mock_client.set_responses([
            [MockStreamEvent(type="content.delta", content="")],
        ])

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        output = await runner.run(turn_input)

        assert output.success is True
        assert output.response == ""

    @pytest.mark.asyncio
    async def test_run_collects_metrics(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Run should collect execution metrics."""
        mock_client.set_responses([
            make_content_events("Response text"),
        ])

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        output = await runner.run(turn_input)

        assert output.metrics is not None
        assert output.metrics.iteration_count == 1
        assert output.metrics.duration_ms > 0

    @pytest.mark.asyncio
    async def test_empty_prompt_raises_by_default(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        snapshot: DocumentSnapshot,
    ):
        """Empty prompt should raise error by default."""
        input = TurnInput(
            prompt="",
            snapshot=snapshot,
        )

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        output = await runner.run(input)

        # Should return error output, not raise
        assert output.success is False
        assert "Empty prompt" in (output.error or "")

    @pytest.mark.asyncio
    async def test_empty_prompt_allowed_with_config(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        snapshot: DocumentSnapshot,
    ):
        """Empty prompt should be allowed when configured."""
        mock_client.set_responses([
            make_content_events("Response to empty prompt"),
        ])

        input = TurnInput(
            prompt="",
            snapshot=snapshot,
        )

        config = RunnerConfig(allow_empty_prompt=True)
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            config=config,
        )

        output = await runner.run(input)

        assert output.success is True


# =============================================================================
# Tool Loop Tests
# =============================================================================


class TestToolLoop:
    """Tests for the tool execution loop."""

    @pytest.mark.asyncio
    async def test_single_tool_call_and_response(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Should execute tool and return final response."""
        # First call: tool call, Second call: final response
        mock_client.set_responses([
            make_tool_call_events("test_tool", {"arg": "value"}, "call_001"),
            make_content_events("Final response after tool"),
        ])

        mock_executor.results = {"test_tool": "Tool result"}

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        output = await runner.run(turn_input)

        assert output.success is True
        assert output.response == "Final response after tool"
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0].name == "test_tool"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_sequence(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Should handle multiple sequential tool calls."""
        mock_client.set_responses([
            make_tool_call_events("tool_1", {"x": 1}, "call_001"),
            make_tool_call_events("tool_2", {"y": 2}, "call_002"),
            make_content_events("Done with all tools"),
        ])

        mock_executor.results = {
            "tool_1": "Result 1",
            "tool_2": "Result 2",
        }

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        output = await runner.run(turn_input)

        assert output.success is True
        assert output.response == "Done with all tools"
        assert len(output.tool_calls) == 2
        assert output.tool_calls[0].name == "tool_1"
        assert output.tool_calls[1].name == "tool_2"
        assert output.metrics.iteration_count == 3

    @pytest.mark.asyncio
    async def test_max_iterations_limit(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Should stop at max iterations."""
        # Keep returning tool calls forever
        mock_client.set_responses([
            make_tool_call_events("tool", {"i": 1}, f"call_{i}")
            for i in range(20)
        ])

        config = RunnerConfig(max_iterations=3)
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            config=config,
        )

        output = await runner.run(turn_input)

        # Should have run 3 iterations
        assert output.metrics.iteration_count == 3
        assert len(output.tool_calls) == 3
        assert output.metadata.get("max_iterations_reached") is True

    @pytest.mark.asyncio
    async def test_tool_results_appended_to_messages(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Tool results should be appended to messages."""
        mock_client.set_responses([
            make_tool_call_events("test_tool", {"arg": "val"}, "call_001"),
            make_content_events("Final response"),
        ])

        mock_executor.results = {"test_tool": "Tool output"}

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        await runner.run(turn_input)

        # Check second call has tool result message
        assert len(mock_client.calls) == 2
        second_call_messages = mock_client.calls[1]["messages"]
        # Should have tool result message appended
        assert any(
            msg.get("role") == "tool" for msg in second_call_messages
        )


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Tests for content and tool callbacks."""

    @pytest.mark.asyncio
    async def test_content_callback_invoked(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Content callback should be invoked with streamed content."""
        mock_client.set_responses([
            make_content_events("Streamed content"),
        ])

        received_content: list[str] = []

        def on_content(content: str) -> None:
            received_content.append(content)

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        await runner.run(turn_input, content_callback=on_content)

        assert len(received_content) > 0
        assert "Streamed content" in "".join(received_content)

    @pytest.mark.asyncio
    async def test_tool_callback_invoked(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Tool callback should be invoked before tool execution."""
        mock_client.set_responses([
            make_tool_call_events("test_tool", {"arg": "value"}, "call_001"),
            make_content_events("Done"),
        ])

        tool_calls_received: list[tuple[str, str, dict]] = []

        def on_tool(call_id: str, name: str, args: Mapping[str, Any]) -> None:
            tool_calls_received.append((call_id, name, dict(args)))

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        await runner.run(turn_input, tool_callback=on_tool)

        assert len(tool_calls_received) == 1
        assert tool_calls_received[0][1] == "test_tool"
        assert tool_calls_received[0][2] == {"arg": "value"}

    @pytest.mark.asyncio
    async def test_callback_exception_handled(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Callback exceptions should be handled gracefully."""
        mock_client.set_responses([
            make_tool_call_events("test_tool", {"arg": "value"}, "call_001"),
            make_content_events("Done"),
        ])

        def bad_callback(call_id: str, name: str, args: Mapping[str, Any]) -> None:
            raise ValueError("Callback error")

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        # Should not raise
        output = await runner.run(turn_input, tool_callback=bad_callback)

        assert output.success is True


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in the runner."""

    @pytest.mark.asyncio
    async def test_model_error_returns_error_output(
        self,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Model error should return error output."""
        class FailingClient:
            async def stream_chat(self, *args, **kwargs):
                raise ValueError("Model API error")
                yield  # Make it a generator

        runner = TurnRunner(
            client=FailingClient(),
            tool_executor=mock_executor,
        )

        output = await runner.run(turn_input)

        assert output.success is False
        assert "Model API error" in (output.error or "")

    @pytest.mark.asyncio
    async def test_tool_executor_error_continues(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Tool executor error should be handled."""
        mock_client.set_responses([
            make_tool_call_events("failing_tool", {}, "call_001"),
            make_content_events("Handled the error"),
        ])

        mock_executor.raise_error = True

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        output = await runner.run(turn_input)

        # Should continue despite tool error
        assert output.success is True
        assert len(output.tool_calls) == 1


# =============================================================================
# Configuration Resolution Tests
# =============================================================================


class TestConfigResolution:
    """Tests for configuration resolution."""

    @pytest.mark.asyncio
    async def test_runner_config_overrides_turn_config(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        snapshot: DocumentSnapshot,
    ):
        """Runner config should override turn config where specified."""
        turn_config = TurnConfig(max_iterations=10)
        input = TurnInput(
            prompt="Test",
            snapshot=snapshot,
            config=turn_config,
        )

        # Keep returning tool calls
        mock_client.set_responses([
            make_tool_call_events("tool", {}, f"call_{i}")
            for i in range(20)
        ])

        runner_config = RunnerConfig(max_iterations=2)
        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            config=runner_config,
        )

        output = await runner.run(input)

        # Runner config (2) should override turn config (10)
        assert output.metrics.iteration_count == 2

    @pytest.mark.asyncio
    async def test_turn_config_used_when_runner_not_specified(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        snapshot: DocumentSnapshot,
    ):
        """Turn config should be used when runner config not specified."""
        turn_config = TurnConfig(max_iterations=2)
        input = TurnInput(
            prompt="Test",
            snapshot=snapshot,
            config=turn_config,
        )

        mock_client.set_responses([
            make_tool_call_events("tool", {}, f"call_{i}")
            for i in range(20)
        ])

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            # No runner config max_iterations
        )

        output = await runner.run(input)

        assert output.metrics.iteration_count == 2


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateRunner:
    """Tests for the create_runner factory function."""

    def test_create_runner_minimal(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """create_runner should work with minimal arguments."""
        runner = create_runner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        assert isinstance(runner, TurnRunner)
        assert runner.client is mock_client
        assert runner.tool_executor is mock_executor

    def test_create_runner_with_tools(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """create_runner should accept tools."""
        tools = [{"type": "function", "function": {"name": "test"}}]

        runner = create_runner(
            client=mock_client,
            tool_executor=mock_executor,
            tools=tools,
        )

        assert len(runner.tools) == 1

    def test_create_runner_with_config_options(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
    ):
        """create_runner should accept config options."""
        runner = create_runner(
            client=mock_client,
            tool_executor=mock_executor,
            max_iterations=15,
            streaming_enabled=False,
        )

        assert runner.config.max_iterations == 15
        assert runner.config.streaming_enabled is False

    @pytest.mark.asyncio
    async def test_created_runner_executes(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Created runner should execute successfully."""
        mock_client.set_responses([
            make_content_events("Hello from factory runner!"),
        ])

        runner = create_runner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        output = await runner.run(turn_input)

        assert output.success is True
        assert output.response == "Hello from factory runner!"


# =============================================================================
# Integration-style Tests
# =============================================================================


class TestIntegration:
    """Integration-style tests with more realistic scenarios."""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_flow(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        snapshot: DocumentSnapshot,
    ):
        """Should handle multi-turn conversation with history."""
        history = (
            Message(role="user", content="Previous question"),
            Message(role="assistant", content="Previous answer"),
        )

        input = TurnInput(
            prompt="Follow-up question",
            snapshot=snapshot,
            history=history,
        )

        mock_client.set_responses([
            make_content_events("Follow-up answer"),
        ])

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        output = await runner.run(input)

        assert output.success is True
        # Check that history was passed to model
        messages = mock_client.calls[0]["messages"]
        assert len(messages) >= 3  # system + history + user

    @pytest.mark.asyncio
    async def test_complex_tool_loop_scenario(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Should handle complex tool loop with multiple tools per iteration."""
        # Iteration 1: Two tool calls
        iter1_events = [
            MockStreamEvent(
                type="tool_calls.function.arguments.done",
                tool_name="search",
                tool_arguments='{"query": "test"}',
                tool_index=0,
                tool_call_id="call_search",
            ),
            MockStreamEvent(
                type="tool_calls.function.arguments.done",
                tool_name="analyze",
                tool_arguments='{"data": "input"}',
                tool_index=1,
                tool_call_id="call_analyze",
            ),
        ]

        mock_client.set_responses([
            iter1_events,
            make_content_events("Final result after tools"),
        ])

        mock_executor.results = {
            "search": ["result1", "result2"],
            "analyze": {"status": "complete"},
        }

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        output = await runner.run(turn_input)

        assert output.success is True
        assert output.response == "Final result after tools"
        assert len(output.tool_calls) == 2
        assert {tc.name for tc in output.tool_calls} == {"search", "analyze"}

    @pytest.mark.asyncio
    async def test_run_id_generated_if_missing(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        snapshot: DocumentSnapshot,
    ):
        """Should generate run_id if not provided."""
        input = TurnInput(
            prompt="Test",
            snapshot=snapshot,
            # No run_id
        )

        mock_client.set_responses([
            make_content_events("Response"),
        ])

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
        )

        output = await runner.run(input)

        assert output.success is True
        # Internal run_id should have been generated

    @pytest.mark.asyncio
    async def test_analysis_disabled_by_default(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Analysis should be disabled when no provider is set."""
        mock_client.set_responses([
            make_content_events("Response"),
        ])

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            # No analysis_provider
        )

        output = await runner.run(turn_input)

        assert output.success is True
        assert output.metrics.analysis_ran is False


# =============================================================================
# Token Counting Tests
# =============================================================================


class TestTokenCounting:
    """Tests for token counting integration."""

    @pytest.mark.asyncio
    async def test_token_counter_passed_to_prepare(
        self,
        mock_client: MockModelClient,
        mock_executor: MockToolExecutor,
        turn_input: TurnInput,
    ):
        """Token counter should be used in preparation."""
        token_count_calls: list[str] = []

        def mock_counter(text: str) -> int:
            token_count_calls.append(text)
            return len(text.split())

        mock_client.set_responses([
            make_content_events("Response"),
        ])

        runner = TurnRunner(
            client=mock_client,
            tool_executor=mock_executor,
            token_counter=mock_counter,
        )

        await runner.run(turn_input)

        # Token counter should have been called
        assert len(token_count_calls) > 0
