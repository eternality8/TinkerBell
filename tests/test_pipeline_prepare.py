"""Unit tests for the prepare pipeline stage."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from tinkerbell.ai.orchestration.pipeline.prepare import (
    build_messages,
    estimate_budget,
    estimate_message_tokens,
    estimate_text_tokens,
    prepare_turn,
    sanitize_history,
)
from tinkerbell.ai.orchestration.transaction import DocumentSnapshot
from tinkerbell.ai.orchestration.types import (
    Message,
    TurnConfig,
    TurnInput,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_snapshot() -> DocumentSnapshot:
    """Create a sample document snapshot."""
    return DocumentSnapshot(
        tab_id="test-doc",
        content="Hello, world!\nThis is a test document.\nWith multiple lines.",
        version_token="v123",
    )


@pytest.fixture
def sample_config() -> TurnConfig:
    """Create a sample turn configuration."""
    return TurnConfig(
        max_iterations=5,
        analysis_enabled=True,
        max_context_tokens=8_000,
        response_reserve=2_000,
    )


@pytest.fixture
def mock_token_counter():
    """Create a mock token counter that counts words."""
    def counter(text: str) -> int:
        return len(text.split())
    return counter


@pytest.fixture
def sample_history() -> list[Message]:
    """Create sample conversation history."""
    return [
        Message.user("First question"),
        Message.assistant("First answer"),
        Message.user("Second question"),
        Message.assistant("Second answer"),
    ]


# -----------------------------------------------------------------------------
# estimate_text_tokens Tests
# -----------------------------------------------------------------------------


class TestEstimateTextTokens:
    """Tests for estimate_text_tokens function."""

    def test_empty_text(self) -> None:
        """Empty text should return 0 tokens."""
        assert estimate_text_tokens("") == 0

    def test_simple_text_no_counter(self) -> None:
        """Without counter, should use byte-based estimation."""
        # "Hello" = 5 bytes, ceil(5/4) = 2 tokens
        result = estimate_text_tokens("Hello")
        assert result >= 1

    def test_with_token_counter(self, mock_token_counter) -> None:
        """Should use provided token counter."""
        result = estimate_text_tokens("hello world test", token_counter=mock_token_counter)
        assert result == 3  # 3 words

    def test_counter_failure_fallback(self) -> None:
        """Should fallback to heuristic if counter fails."""
        def failing_counter(text: str) -> int:
            raise ValueError("Counter failed")

        result = estimate_text_tokens("Hello world", token_counter=failing_counter)
        assert result >= 1


# -----------------------------------------------------------------------------
# estimate_message_tokens Tests
# -----------------------------------------------------------------------------


class TestEstimateMessageTokens:
    """Tests for estimate_message_tokens function."""

    def test_message_object(self, mock_token_counter) -> None:
        """Should handle Message dataclass."""
        msg = Message.user("hello world")
        result = estimate_message_tokens(msg, token_counter=mock_token_counter)
        # 2 words + 4 for role overhead
        assert result == 6

    def test_message_dict(self, mock_token_counter) -> None:
        """Should handle dict-style message."""
        msg = {"role": "user", "content": "hello world test"}
        result = estimate_message_tokens(msg, token_counter=mock_token_counter)
        # 3 words + 4 for role overhead
        assert result == 7

    def test_message_dict_no_role(self, mock_token_counter) -> None:
        """Dict without role should not add role overhead."""
        msg = {"content": "hello world"}
        result = estimate_message_tokens(msg, token_counter=mock_token_counter)
        # 2 words, no role overhead
        assert result == 2

    def test_empty_content(self, mock_token_counter) -> None:
        """Empty content should return overhead only."""
        msg = Message.user("")
        result = estimate_message_tokens(msg, token_counter=mock_token_counter)
        assert result == 4  # Just role overhead


# -----------------------------------------------------------------------------
# sanitize_history Tests
# -----------------------------------------------------------------------------


class TestSanitizeHistory:
    """Tests for sanitize_history function."""

    def test_empty_history(self) -> None:
        """Empty history should return empty tuple."""
        result = sanitize_history([])
        assert result == ()

    def test_converts_to_messages(self) -> None:
        """Should convert dicts to Message objects."""
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = sanitize_history(history)
        assert len(result) == 2
        assert all(isinstance(m, Message) for m in result)
        assert result[0].role == "user"
        assert result[1].role == "assistant"

    def test_preserves_message_objects(self, sample_history: list[Message]) -> None:
        """Should preserve existing Message objects."""
        result = sanitize_history(sample_history)
        assert len(result) == 4
        assert result[0].content == "First question"

    def test_filters_invalid_roles(self) -> None:
        """Should filter messages with invalid roles."""
        history = [
            {"role": "user", "content": "Valid"},
            {"role": "invalid_role", "content": "Invalid"},
            {"role": "assistant", "content": "Also valid"},
        ]
        result = sanitize_history(history)
        assert len(result) == 2

    def test_filters_empty_content(self) -> None:
        """Should filter messages with empty content."""
        history = [
            {"role": "user", "content": "Has content"},
            {"role": "user", "content": ""},
            {"role": "user", "content": "   "},
        ]
        result = sanitize_history(history)
        assert len(result) == 1

    def test_respects_limit(self) -> None:
        """Should limit to most recent messages."""
        history = [Message.user(f"Message {i}") for i in range(10)]
        result = sanitize_history(history, limit=5)
        assert len(result) == 5
        # Should keep most recent
        assert result[-1].content == "Message 9"

    def test_respects_token_budget(self, mock_token_counter) -> None:
        """Should trim to fit token budget."""
        history = [
            Message.user("short"),  # 1 word + 4 overhead = 5
            Message.user("another short one"),  # 3 words + 4 overhead = 7
            Message.user("last message here"),  # 3 words + 4 overhead = 7
        ]
        # Budget of 15 should fit last 2 messages (7 + 7 = 14)
        result = sanitize_history(history, token_budget=15, token_counter=mock_token_counter)
        assert len(result) == 2
        assert result[-1].content == "last message here"

    def test_zero_budget_returns_empty(self) -> None:
        """Zero budget should return empty."""
        history = [Message.user("Hello")]
        result = sanitize_history(history, token_budget=0)
        assert result == ()

    def test_includes_at_least_one_message(self, mock_token_counter) -> None:
        """Should include at least one message even if over budget."""
        history = [Message.user("this is a very long message")]
        result = sanitize_history(history, token_budget=1, token_counter=mock_token_counter)
        assert len(result) == 1


# -----------------------------------------------------------------------------
# build_messages Tests
# -----------------------------------------------------------------------------


class TestBuildMessages:
    """Tests for build_messages function."""

    def test_basic_structure(self, sample_snapshot: DocumentSnapshot, sample_config: TurnConfig) -> None:
        """Should return system, history, and user messages."""
        result = build_messages(
            prompt="Test prompt",
            snapshot=sample_snapshot,
            history=[],
            config=sample_config,
        )
        assert len(result) >= 2
        # First should be system
        assert result[0].role == "system"
        # Last should be user
        assert result[-1].role == "user"

    def test_includes_history(
        self,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
        sample_history: list[Message],
    ) -> None:
        """Should include sanitized history."""
        result = build_messages(
            prompt="Test prompt",
            snapshot=sample_snapshot,
            history=sample_history,
            config=sample_config,
        )
        # system + 4 history + user = 6
        assert len(result) == 6
        # History should be between system and user
        assert result[1].role == "user"
        assert result[1].content == "First question"

    def test_trims_history_for_budget(
        self,
        sample_snapshot: DocumentSnapshot,
        mock_token_counter,
    ) -> None:
        """Should trim history to fit budget."""
        # Very small context window
        config = TurnConfig(
            max_context_tokens=500,
            response_reserve=100,
        )
        long_history = [Message.user(f"Message {i} " * 20) for i in range(50)]

        result = build_messages(
            prompt="Test",
            snapshot=sample_snapshot,
            history=long_history,
            config=config,
            token_counter=mock_token_counter,
        )

        # Should have fewer than all history messages
        assert len(result) < 52  # system + 50 history + user

    def test_user_prompt_contains_document_info(
        self,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """User message should contain document context."""
        result = build_messages(
            prompt="Edit the document",
            snapshot=sample_snapshot,
            config=sample_config,
        )
        user_msg = result[-1]
        assert "Edit the document" in user_msg.content
        # Should include document info from snapshot
        assert "Document:" in user_msg.content or "test-doc" in user_msg.content

    def test_returns_tuple(
        self,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """Should return immutable tuple."""
        result = build_messages(
            prompt="Test",
            snapshot=sample_snapshot,
            config=sample_config,
        )
        assert isinstance(result, tuple)


# -----------------------------------------------------------------------------
# estimate_budget Tests
# -----------------------------------------------------------------------------


class TestEstimateBudget:
    """Tests for estimate_budget function."""

    def test_within_budget(self, mock_token_counter) -> None:
        """Should return 'ok' when within budget."""
        messages = (
            Message.system("short system"),
            Message.user("short user"),
        )
        config = TurnConfig(
            max_context_tokens=10_000,
            response_reserve=2_000,
        )

        result = estimate_budget(messages, config, token_counter=mock_token_counter)

        assert result.verdict == "ok"
        assert result.is_ok is True
        assert result.headroom > 0

    def test_exceeds_budget(self, mock_token_counter) -> None:
        """Should return 'needs_summary' when exceeding budget."""
        # Create messages that will exceed budget
        messages = tuple(Message.user("word " * 100) for _ in range(100))
        config = TurnConfig(
            max_context_tokens=1_000,
            response_reserve=500,
        )

        result = estimate_budget(messages, config, token_counter=mock_token_counter)

        assert result.verdict in ("needs_summary", "reject")
        assert result.is_ok is False

    def test_calculates_prompt_tokens(self, mock_token_counter) -> None:
        """Should calculate total prompt tokens."""
        messages = (
            Message.system("one two"),  # 2 words + 4 = 6
            Message.user("three four five"),  # 3 words + 4 = 7
        )
        config = TurnConfig(max_context_tokens=10_000, response_reserve=2_000)

        result = estimate_budget(messages, config, token_counter=mock_token_counter)

        assert result.prompt_tokens == 13  # 6 + 7

    def test_completion_budget(self, mock_token_counter) -> None:
        """Should set completion_budget based on response_reserve."""
        messages = (Message.user("test"),)
        config = TurnConfig(
            max_context_tokens=10_000,
            response_reserve=3_000,
        )

        result = estimate_budget(messages, config, token_counter=mock_token_counter)

        assert result.completion_budget == 3_000

    def test_total_budget(self, mock_token_counter) -> None:
        """Should set total_budget from config."""
        messages = (Message.user("test"),)
        config = TurnConfig(max_context_tokens=50_000)

        result = estimate_budget(messages, config, token_counter=mock_token_counter)

        assert result.total_budget == 50_000


# -----------------------------------------------------------------------------
# prepare_turn Tests
# -----------------------------------------------------------------------------


class TestPrepareTurn:
    """Tests for prepare_turn function."""

    def test_returns_prepared_turn(
        self,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """Should return a PreparedTurn object."""
        turn_input = TurnInput(
            prompt="Test prompt",
            snapshot=sample_snapshot,
            config=sample_config,
        )

        result = prepare_turn(turn_input)

        from tinkerbell.ai.orchestration.types import PreparedTurn
        assert isinstance(result, PreparedTurn)

    def test_prepared_turn_has_messages(
        self,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """PreparedTurn should contain messages."""
        turn_input = TurnInput(
            prompt="Test",
            snapshot=sample_snapshot,
            config=sample_config,
        )

        result = prepare_turn(turn_input)

        assert len(result.messages) >= 2
        assert result.messages[0].role == "system"
        assert result.messages[-1].role == "user"

    def test_prepared_turn_has_budget(
        self,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """PreparedTurn should contain budget estimate."""
        turn_input = TurnInput(
            prompt="Test",
            snapshot=sample_snapshot,
            config=sample_config,
        )

        result = prepare_turn(turn_input)

        from tinkerbell.ai.orchestration.types import BudgetEstimate
        assert isinstance(result.budget, BudgetEstimate)
        assert result.budget.prompt_tokens > 0

    def test_prepared_turn_has_system_prompt(
        self,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """PreparedTurn should extract system prompt."""
        turn_input = TurnInput(
            prompt="Test",
            snapshot=sample_snapshot,
            config=sample_config,
        )

        result = prepare_turn(turn_input)

        assert result.system_prompt != ""
        # System prompt should match first message
        assert result.system_prompt == result.messages[0].content

    def test_prepared_turn_has_document_context(
        self,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """PreparedTurn should include document content."""
        turn_input = TurnInput(
            prompt="Test",
            snapshot=sample_snapshot,
            config=sample_config,
        )

        result = prepare_turn(turn_input)

        assert result.document_context == sample_snapshot.content

    def test_with_history(
        self,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
        sample_history: list[Message],
    ) -> None:
        """Should include history in messages."""
        turn_input = TurnInput(
            prompt="Test",
            snapshot=sample_snapshot,
            history=tuple(sample_history),
            config=sample_config,
        )

        result = prepare_turn(turn_input)

        # system + history + user
        assert len(result.messages) > 2

    def test_with_custom_token_counter(
        self,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
        mock_token_counter,
    ) -> None:
        """Should use provided token counter."""
        turn_input = TurnInput(
            prompt="Test prompt here",
            snapshot=sample_snapshot,
            config=sample_config,
        )

        result = prepare_turn(turn_input, token_counter=mock_token_counter)

        # Budget should be calculated using word-based counter
        assert result.budget.prompt_tokens > 0


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestPrepareIntegration:
    """Integration tests for the prepare stage."""

    def test_full_pipeline_flow(
        self,
        sample_snapshot: DocumentSnapshot,
        sample_history: list[Message],
    ) -> None:
        """Test complete prepare flow."""
        config = TurnConfig(
            max_context_tokens=100_000,
            response_reserve=4_000,
            model_name="gpt-4",
        )

        turn_input = TurnInput(
            prompt="Please edit the document",
            snapshot=sample_snapshot,
            history=tuple(sample_history),
            config=config,
            run_id="test-run-123",
            document_id="test-doc",
        )

        result = prepare_turn(turn_input)

        # Verify structure
        assert len(result.messages) >= 2
        assert result.budget.verdict == "ok"
        assert result.budget.total_budget == 100_000
        assert result.system_prompt != ""
        assert result.document_context == sample_snapshot.content

    def test_prepared_messages_are_immutable(
        self,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """PreparedTurn messages should be immutable."""
        turn_input = TurnInput(
            prompt="Test",
            snapshot=sample_snapshot,
            config=sample_config,
        )

        result = prepare_turn(turn_input)

        # Should be a tuple
        assert isinstance(result.messages, tuple)
        # Individual messages should be frozen
        with pytest.raises(AttributeError):
            result.messages[0].content = "modified"  # type: ignore[misc]
