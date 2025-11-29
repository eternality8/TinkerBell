"""Unit tests for orchestration pipeline types."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tinkerbell.ai.orchestration.types import (
    AnalyzedTurn,
    BudgetEstimate,
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
from tinkerbell.ai.orchestration.transaction import DocumentSnapshot


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_snapshot() -> DocumentSnapshot:
    """Create a sample document snapshot."""
    return DocumentSnapshot(
        tab_id="test-tab",
        content="Hello, world!",
        version_token="v1",
    )


@pytest.fixture
def sample_config() -> TurnConfig:
    """Create a sample turn configuration."""
    return TurnConfig(
        max_iterations=5,
        analysis_enabled=True,
        max_context_tokens=100_000,
    )


@pytest.fixture
def sample_messages() -> tuple[Message, ...]:
    """Create sample messages."""
    return (
        Message.system("You are a helpful assistant."),
        Message.user("Hello!"),
        Message.assistant("Hi there!"),
    )


# -----------------------------------------------------------------------------
# Message Tests
# -----------------------------------------------------------------------------


class TestMessage:
    """Tests for the Message dataclass."""

    def test_message_is_frozen(self) -> None:
        """Message should be immutable."""
        msg = Message(role="user", content="test")
        with pytest.raises(AttributeError):
            msg.content = "modified"  # type: ignore[misc]

    def test_system_factory(self) -> None:
        """system() should create a system message."""
        msg = Message.system("System prompt")
        assert msg.role == "system"
        assert msg.content == "System prompt"

    def test_user_factory(self) -> None:
        """user() should create a user message."""
        msg = Message.user("User input")
        assert msg.role == "user"
        assert msg.content == "User input"

    def test_assistant_factory(self) -> None:
        """assistant() should create an assistant message."""
        msg = Message.assistant("Response")
        assert msg.role == "assistant"
        assert msg.content == "Response"

    def test_assistant_with_tool_calls(self) -> None:
        """assistant() should handle tool calls."""
        tool_calls = [{"id": "call_1", "name": "test_tool"}]
        msg = Message.assistant("Response", tool_calls=tool_calls)
        assert msg.role == "assistant"
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["id"] == "call_1"

    def test_tool_factory(self) -> None:
        """tool() should create a tool result message."""
        msg = Message.tool("Result", tool_call_id="call_1", name="test_tool")
        assert msg.role == "tool"
        assert msg.content == "Result"
        assert msg.tool_call_id == "call_1"
        assert msg.name == "test_tool"

    def test_to_chat_param(self) -> None:
        """to_chat_param() should convert to OpenAI format."""
        msg = Message.user("Hello")
        param = msg.to_chat_param()
        assert param["role"] == "user"
        assert param["content"] == "Hello"

    def test_to_chat_param_with_tool_call_id(self) -> None:
        """to_chat_param() should include tool_call_id."""
        msg = Message.tool("Result", tool_call_id="call_1")
        param = msg.to_chat_param()
        assert param["tool_call_id"] == "call_1"

    def test_from_chat_param(self) -> None:
        """from_chat_param() should create Message from OpenAI format."""
        param = {"role": "assistant", "content": "Hello"}
        msg = Message.from_chat_param(param)
        assert msg.role == "assistant"
        assert msg.content == "Hello"

    def test_from_chat_param_with_tool_calls(self) -> None:
        """from_chat_param() should handle tool_calls."""
        param = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_1", "function": {"name": "test"}}],
        }
        msg = Message.from_chat_param(param)
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_metadata_factory_kwargs(self) -> None:
        """Factory methods should accept metadata kwargs."""
        msg = Message.user("test", source="test_suite", priority=1)
        assert msg.metadata["source"] == "test_suite"
        assert msg.metadata["priority"] == 1


# -----------------------------------------------------------------------------
# BudgetEstimate Tests
# -----------------------------------------------------------------------------


class TestBudgetEstimate:
    """Tests for the BudgetEstimate dataclass."""

    def test_is_ok_when_ok(self) -> None:
        """is_ok should return True when verdict is ok."""
        budget = BudgetEstimate(
            prompt_tokens=1000,
            completion_budget=4000,
            total_budget=128000,
            verdict="ok",
        )
        assert budget.is_ok is True
        assert budget.needs_trimming is False

    def test_is_ok_when_needs_summary(self) -> None:
        """is_ok should return False when verdict is needs_summary."""
        budget = BudgetEstimate(
            prompt_tokens=120000,
            completion_budget=4000,
            total_budget=128000,
            verdict="needs_summary",
        )
        assert budget.is_ok is False
        assert budget.needs_trimming is True

    def test_is_ok_when_reject(self) -> None:
        """is_ok should return False when verdict is reject."""
        budget = BudgetEstimate(
            prompt_tokens=200000,
            completion_budget=4000,
            total_budget=128000,
            verdict="reject",
        )
        assert budget.is_ok is False
        assert budget.needs_trimming is True


# -----------------------------------------------------------------------------
# TurnConfig Tests
# -----------------------------------------------------------------------------


class TestTurnConfig:
    """Tests for the TurnConfig dataclass."""

    def test_default_values(self) -> None:
        """TurnConfig should have sensible defaults."""
        config = TurnConfig()
        assert config.max_iterations == 8
        assert config.analysis_enabled is True
        assert config.streaming_enabled is True

    def test_is_frozen(self) -> None:
        """TurnConfig should be immutable."""
        config = TurnConfig()
        with pytest.raises(AttributeError):
            config.max_iterations = 10  # type: ignore[misc]

    def test_with_updates(self) -> None:
        """with_updates() should return a new config with changes."""
        config = TurnConfig(max_iterations=5)
        updated = config.with_updates(max_iterations=10, temperature=0.5)
        
        # Original unchanged
        assert config.max_iterations == 5
        assert config.temperature == 0.7
        
        # New config has updates
        assert updated.max_iterations == 10
        assert updated.temperature == 0.5


# -----------------------------------------------------------------------------
# TurnInput Tests
# -----------------------------------------------------------------------------


class TestTurnInput:
    """Tests for the TurnInput dataclass."""

    def test_basic_creation(self, sample_snapshot: DocumentSnapshot) -> None:
        """TurnInput should be created with required fields."""
        turn_input = TurnInput(
            prompt="Test prompt",
            snapshot=sample_snapshot,
        )
        assert turn_input.prompt == "Test prompt"
        assert turn_input.snapshot == sample_snapshot
        assert turn_input.history == ()

    def test_history_converted_to_tuple(self, sample_snapshot: DocumentSnapshot) -> None:
        """History list should be converted to tuple."""
        history = [Message.user("Hello"), Message.assistant("Hi")]
        turn_input = TurnInput(
            prompt="Test",
            snapshot=sample_snapshot,
            history=history,  # type: ignore[arg-type]
        )
        assert isinstance(turn_input.history, tuple)
        assert len(turn_input.history) == 2

    def test_is_frozen(self, sample_snapshot: DocumentSnapshot) -> None:
        """TurnInput should be immutable."""
        turn_input = TurnInput(prompt="Test", snapshot=sample_snapshot)
        with pytest.raises(AttributeError):
            turn_input.prompt = "Modified"  # type: ignore[misc]


# -----------------------------------------------------------------------------
# ToolCallRecord Tests
# -----------------------------------------------------------------------------


class TestToolCallRecord:
    """Tests for the ToolCallRecord dataclass."""

    def test_basic_creation(self) -> None:
        """ToolCallRecord should be created with required fields."""
        record = ToolCallRecord(
            call_id="call_1",
            name="test_tool",
            arguments='{"key": "value"}',
            result="success",
        )
        assert record.call_id == "call_1"
        assert record.name == "test_tool"
        assert record.success is True

    def test_to_dict(self) -> None:
        """to_dict() should serialize the record."""
        record = ToolCallRecord(
            call_id="call_1",
            name="test_tool",
            arguments={"key": "value"},
            result="done",
            duration_ms=123.45,
        )
        data = record.to_dict()
        assert data["call_id"] == "call_1"
        assert data["name"] == "test_tool"
        assert data["arguments"] == {"key": "value"}
        assert data["duration_ms"] == 123.45


# -----------------------------------------------------------------------------
# TurnMetrics Tests
# -----------------------------------------------------------------------------


class TestTurnMetrics:
    """Tests for the TurnMetrics dataclass."""

    def test_default_values(self) -> None:
        """TurnMetrics should have sensible defaults."""
        metrics = TurnMetrics()
        assert metrics.prompt_tokens == 0
        assert metrics.completion_tokens == 0
        assert metrics.iteration_count == 1

    def test_to_dict(self) -> None:
        """to_dict() should serialize the metrics."""
        metrics = TurnMetrics(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500,
            duration_ms=1234.5,
        )
        data = metrics.to_dict()
        assert data["prompt_tokens"] == 1000
        assert data["completion_tokens"] == 500
        assert data["total_tokens"] == 1500
        assert data["duration_ms"] == 1234.5

    def test_timestamps_in_to_dict(self) -> None:
        """to_dict() should serialize timestamps as ISO format."""
        now = datetime.now(timezone.utc)
        metrics = TurnMetrics(started_at=now, finished_at=now)
        data = metrics.to_dict()
        assert data["started_at"] == now.isoformat()
        assert data["finished_at"] == now.isoformat()


# -----------------------------------------------------------------------------
# TurnOutput Tests
# -----------------------------------------------------------------------------


class TestTurnOutput:
    """Tests for the TurnOutput dataclass."""

    def test_basic_creation(self) -> None:
        """TurnOutput should be created with required fields."""
        output = TurnOutput(response="Hello!")
        assert output.response == "Hello!"
        assert output.success is True
        assert output.tool_calls == ()

    def test_tool_calls_converted_to_tuple(self) -> None:
        """Tool calls list should be converted to tuple."""
        records = [
            ToolCallRecord(call_id="1", name="tool1", arguments="{}"),
            ToolCallRecord(call_id="2", name="tool2", arguments="{}"),
        ]
        output = TurnOutput(response="Done", tool_calls=records)  # type: ignore[arg-type]
        assert isinstance(output.tool_calls, tuple)
        assert len(output.tool_calls) == 2

    def test_from_error(self) -> None:
        """from_error() should create a failed output."""
        output = TurnOutput.from_error("Something went wrong")
        assert output.success is False
        assert output.error == "Something went wrong"
        assert output.response == ""


# -----------------------------------------------------------------------------
# PreparedTurn Tests
# -----------------------------------------------------------------------------


class TestPreparedTurn:
    """Tests for the PreparedTurn dataclass."""

    def test_basic_creation(self, sample_messages: tuple[Message, ...]) -> None:
        """PreparedTurn should be created with required fields."""
        budget = BudgetEstimate(
            prompt_tokens=100,
            completion_budget=4000,
            total_budget=128000,
        )
        prepared = PreparedTurn(messages=sample_messages, budget=budget)
        assert len(prepared.messages) == 3
        assert prepared.budget.prompt_tokens == 100

    def test_messages_converted_to_tuple(self) -> None:
        """Messages list should be converted to tuple."""
        messages = [Message.user("Hello")]
        budget = BudgetEstimate(prompt_tokens=10, completion_budget=100, total_budget=1000)
        prepared = PreparedTurn(messages=messages, budget=budget)  # type: ignore[arg-type]
        assert isinstance(prepared.messages, tuple)


# -----------------------------------------------------------------------------
# AnalyzedTurn Tests
# -----------------------------------------------------------------------------


class TestAnalyzedTurn:
    """Tests for the AnalyzedTurn dataclass."""

    def test_messages_with_hints_no_hints(self, sample_messages: tuple[Message, ...]) -> None:
        """messages_with_hints() should return original messages when no hints."""
        budget = BudgetEstimate(prompt_tokens=100, completion_budget=4000, total_budget=128000)
        prepared = PreparedTurn(messages=sample_messages, budget=budget)
        analyzed = AnalyzedTurn(prepared=prepared)
        
        result = analyzed.messages_with_hints()
        assert result == sample_messages

    def test_messages_with_hints_appends_to_system(self, sample_messages: tuple[Message, ...]) -> None:
        """messages_with_hints() should append hints to last system message."""
        budget = BudgetEstimate(prompt_tokens=100, completion_budget=4000, total_budget=128000)
        prepared = PreparedTurn(messages=sample_messages, budget=budget)
        analyzed = AnalyzedTurn(
            prepared=prepared,
            hints=("Hint 1", "Hint 2"),
        )
        
        result = analyzed.messages_with_hints()
        # Find the system message
        system_msg = next(m for m in result if m.role == "system")
        assert "Hint 1" in system_msg.content
        assert "Hint 2" in system_msg.content

    def test_messages_with_hints_no_system_message(self) -> None:
        """messages_with_hints() should prepend system message if none exists."""
        messages = (Message.user("Hello"),)
        budget = BudgetEstimate(prompt_tokens=100, completion_budget=4000, total_budget=128000)
        prepared = PreparedTurn(messages=messages, budget=budget)
        analyzed = AnalyzedTurn(
            prepared=prepared,
            hints=("Important hint",),
        )
        
        result = analyzed.messages_with_hints()
        assert len(result) == 2
        assert result[0].role == "system"
        assert "Important hint" in result[0].content


# -----------------------------------------------------------------------------
# ModelResponse Tests
# -----------------------------------------------------------------------------


class TestModelResponse:
    """Tests for the ModelResponse dataclass."""

    def test_has_tool_calls_false(self) -> None:
        """has_tool_calls should return False when no tool calls."""
        response = ModelResponse(text="Hello")
        assert response.has_tool_calls is False

    def test_has_tool_calls_true(self) -> None:
        """has_tool_calls should return True when tool calls exist."""
        tool_call = ParsedToolCall(
            call_id="call_1",
            name="test_tool",
            arguments="{}",
        )
        response = ModelResponse(text="", tool_calls=(tool_call,))
        assert response.has_tool_calls is True

    def test_to_message(self) -> None:
        """to_message() should convert to assistant Message."""
        response = ModelResponse(text="Hello there!")
        msg = response.to_message()
        assert msg.role == "assistant"
        assert msg.content == "Hello there!"

    def test_to_message_with_tool_calls(self) -> None:
        """to_message() should include tool calls in Message."""
        tool_call = ParsedToolCall(
            call_id="call_1",
            name="test_tool",
            arguments='{"arg": "value"}',
        )
        response = ModelResponse(text="", tool_calls=(tool_call,))
        msg = response.to_message()
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["id"] == "call_1"
        assert msg.tool_calls[0]["function"]["name"] == "test_tool"


# -----------------------------------------------------------------------------
# ParsedToolCall Tests
# -----------------------------------------------------------------------------


class TestParsedToolCall:
    """Tests for the ParsedToolCall dataclass."""

    def test_basic_creation(self) -> None:
        """ParsedToolCall should be created with required fields."""
        call = ParsedToolCall(
            call_id="call_abc123",
            name="search_document",
            arguments='{"query": "test"}',
        )
        assert call.call_id == "call_abc123"
        assert call.name == "search_document"
        assert call.arguments == '{"query": "test"}'
        assert call.index == 0

    def test_is_frozen(self) -> None:
        """ParsedToolCall should be immutable."""
        call = ParsedToolCall(call_id="1", name="test", arguments="{}")
        with pytest.raises(AttributeError):
            call.name = "modified"  # type: ignore[misc]
