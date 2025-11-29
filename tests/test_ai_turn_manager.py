"""Tests for AITurnManager domain manager."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from tinkerbell.ui.domain.ai_turn_manager import AITurnManager
from tinkerbell.ui.events import (
    AITurnCanceled,
    AITurnCompleted,
    AITurnFailed,
    AITurnStarted,
    AITurnStreamChunk,
    Event,
    EventBus,
)
from tinkerbell.ui.models.ai_models import AITurnState, AITurnStatus


# =============================================================================
# Fixtures
# =============================================================================


class MockOrchestrator:
    """Mock AIOrchestrator for testing."""

    def __init__(self, response: Any = None, error: Exception | None = None, use_default: bool = True) -> None:
        if response is None and use_default:
            self.response = {"response": "Test response"}
        else:
            self.response = response
        self.error = error
        self.run_chat_calls: list[tuple] = []
        self.canceled = False

    async def run_chat(
        self,
        prompt: str,
        snapshot: Any,
        *,
        metadata: Any = None,
        history: Any = None,
        on_event: Any = None,
    ) -> Any:
        self.run_chat_calls.append((prompt, snapshot, metadata, history))
        if self.error:
            raise self.error
        return self.response

    def cancel(self) -> None:
        self.canceled = True


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def mock_orchestrator() -> MockOrchestrator:
    return MockOrchestrator()


@pytest.fixture
def ai_turn_manager(
    mock_orchestrator: MockOrchestrator, event_bus: EventBus
) -> AITurnManager:
    return AITurnManager(
        orchestrator_provider=lambda: mock_orchestrator,
        event_bus=event_bus,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestAITurnManagerInit:
    """Tests for AITurnManager initialization."""

    def test_initial_state(self, ai_turn_manager: AITurnManager) -> None:
        """Manager starts with no current turn."""
        assert ai_turn_manager.current_turn is None
        assert ai_turn_manager.is_running() is False


# =============================================================================
# Start Turn Tests
# =============================================================================


class TestAITurnManagerStartTurn:
    """Tests for AITurnManager.start_turn()."""

    @pytest.mark.asyncio
    async def test_creates_turn_state(self, ai_turn_manager: AITurnManager) -> None:
        """start_turn creates AITurnState."""
        await ai_turn_manager.start_turn(
            prompt="Hello",
            snapshot={"text": "content"},
            metadata={"key": "value"},
            history=None,
        )

        assert ai_turn_manager.current_turn is not None
        assert ai_turn_manager.current_turn.prompt == "Hello"
        assert ai_turn_manager.current_turn.turn_id.startswith("turn-")

    @pytest.mark.asyncio
    async def test_generates_unique_turn_id(
        self, mock_orchestrator: MockOrchestrator, event_bus: EventBus
    ) -> None:
        """Each turn gets a unique ID."""
        manager = AITurnManager(
            orchestrator_provider=lambda: mock_orchestrator,
            event_bus=event_bus,
        )

        await manager.start_turn("prompt1", {}, {}, None)
        turn1_id = manager.current_turn.turn_id

        # Reset for second turn
        manager._current_turn = None

        await manager.start_turn("prompt2", {}, {}, None)
        turn2_id = manager.current_turn.turn_id

        assert turn1_id != turn2_id

    @pytest.mark.asyncio
    async def test_emits_started_event(
        self, ai_turn_manager: AITurnManager, event_bus: EventBus
    ) -> None:
        """start_turn emits AITurnStarted event."""
        events: list[Event] = []
        event_bus.subscribe(AITurnStarted, events.append)

        await ai_turn_manager.start_turn("Test prompt", {}, {}, None)

        assert len(events) == 1
        assert events[0].prompt == "Test prompt"
        assert events[0].turn_id == ai_turn_manager.current_turn.turn_id

    @pytest.mark.asyncio
    async def test_emits_completed_event(
        self, ai_turn_manager: AITurnManager, event_bus: EventBus
    ) -> None:
        """start_turn emits AITurnCompleted on success."""
        events: list[Event] = []
        event_bus.subscribe(AITurnCompleted, events.append)

        await ai_turn_manager.start_turn("Test", {}, {}, None)

        assert len(events) == 1
        assert events[0].success is True
        assert events[0].response_text == "Test response"

    @pytest.mark.asyncio
    async def test_marks_turn_completed(
        self, ai_turn_manager: AITurnManager
    ) -> None:
        """Successful turn is marked completed."""
        await ai_turn_manager.start_turn("Test", {}, {}, None)

        assert ai_turn_manager.current_turn.status == AITurnStatus.COMPLETED
        assert ai_turn_manager.current_turn.is_successful is True

    @pytest.mark.asyncio
    async def test_raises_if_already_running(
        self, ai_turn_manager: AITurnManager
    ) -> None:
        """Cannot start turn while one is running."""
        # Manually set running state
        ai_turn_manager._current_turn = AITurnState(
            turn_id="test",
            prompt="test",
            status=AITurnStatus.RUNNING,
        )

        with pytest.raises(RuntimeError, match="already in progress"):
            await ai_turn_manager.start_turn("Another", {}, {}, None)

    @pytest.mark.asyncio
    async def test_raises_if_no_orchestrator(self, event_bus: EventBus) -> None:
        """Raises if orchestrator unavailable."""
        manager = AITurnManager(
            orchestrator_provider=lambda: None,
            event_bus=event_bus,
        )

        with pytest.raises(RuntimeError, match="unavailable"):
            await manager.start_turn("Test", {}, {}, None)

    @pytest.mark.asyncio
    async def test_passes_parameters_to_orchestrator(
        self,
        ai_turn_manager: AITurnManager,
        mock_orchestrator: MockOrchestrator,
    ) -> None:
        """Parameters are passed to orchestrator.run_chat."""
        snapshot = {"text": "doc content"}
        metadata = {"key": "value"}
        history = [{"role": "user", "content": "hi"}]

        await ai_turn_manager.start_turn("prompt", snapshot, metadata, history)

        assert len(mock_orchestrator.run_chat_calls) == 1
        call_args = mock_orchestrator.run_chat_calls[0]
        assert call_args[0] == "prompt"
        assert call_args[1] == snapshot
        assert call_args[2] == metadata
        assert call_args[3] == history


# =============================================================================
# Failure Tests
# =============================================================================


class TestAITurnManagerFailure:
    """Tests for turn failure handling."""

    @pytest.mark.asyncio
    async def test_emits_failed_event_on_error(
        self, event_bus: EventBus
    ) -> None:
        """Emits AITurnFailed when orchestrator raises."""
        orchestrator = MockOrchestrator(error=ValueError("Test error"))
        manager = AITurnManager(
            orchestrator_provider=lambda: orchestrator,
            event_bus=event_bus,
        )

        events: list[Event] = []
        event_bus.subscribe(AITurnFailed, events.append)

        with pytest.raises(ValueError):
            await manager.start_turn("Test", {}, {}, None)

        assert len(events) == 1
        assert events[0].error == "Test error"

    @pytest.mark.asyncio
    async def test_marks_turn_failed(self, event_bus: EventBus) -> None:
        """Turn is marked failed on error."""
        orchestrator = MockOrchestrator(error=RuntimeError("fail"))
        manager = AITurnManager(
            orchestrator_provider=lambda: orchestrator,
            event_bus=event_bus,
        )

        with pytest.raises(RuntimeError):
            await manager.start_turn("Test", {}, {}, None)

        assert manager.current_turn.status == AITurnStatus.FAILED
        assert manager.current_turn.error == "fail"


# =============================================================================
# Cancel Tests
# =============================================================================


class TestAITurnManagerCancel:
    """Tests for turn cancellation."""

    def test_cancel_no_running_turn(
        self, ai_turn_manager: AITurnManager, event_bus: EventBus
    ) -> None:
        """Cancel does nothing if no turn running."""
        events: list[Event] = []
        event_bus.subscribe(AITurnCanceled, events.append)

        ai_turn_manager.cancel()

        assert len(events) == 0

    def test_cancel_calls_orchestrator_cancel(
        self,
        ai_turn_manager: AITurnManager,
        mock_orchestrator: MockOrchestrator,
    ) -> None:
        """Cancel calls orchestrator.cancel()."""
        # Set up running turn
        ai_turn_manager._current_turn = AITurnState(
            turn_id="test",
            prompt="test",
            status=AITurnStatus.RUNNING,
        )

        ai_turn_manager.cancel()

        assert mock_orchestrator.canceled is True

    def test_cancel_emits_canceled_event(
        self, ai_turn_manager: AITurnManager, event_bus: EventBus
    ) -> None:
        """Cancel emits AITurnCanceled event."""
        ai_turn_manager._current_turn = AITurnState(
            turn_id="test-123",
            prompt="test",
            status=AITurnStatus.RUNNING,
        )

        events: list[Event] = []
        event_bus.subscribe(AITurnCanceled, events.append)

        ai_turn_manager.cancel()

        assert len(events) == 1
        assert events[0].turn_id == "test-123"

    def test_cancel_marks_turn_canceled(
        self, ai_turn_manager: AITurnManager
    ) -> None:
        """Cancel marks turn as canceled."""
        ai_turn_manager._current_turn = AITurnState(
            turn_id="test",
            prompt="test",
            status=AITurnStatus.RUNNING,
        )

        ai_turn_manager.cancel()

        assert ai_turn_manager.current_turn.status == AITurnStatus.CANCELED


# =============================================================================
# Streaming Tests
# =============================================================================


class TestAITurnManagerStreaming:
    """Tests for streaming event handling."""

    @pytest.mark.asyncio
    async def test_emits_stream_chunk_events(
        self, event_bus: EventBus
    ) -> None:
        """Stream chunks emit AITurnStreamChunk events."""

        class StreamingOrchestrator:
            async def run_chat(
                self,
                prompt: str,
                snapshot: Any,
                *,
                metadata: Any = None,
                history: Any = None,
                on_event: Any = None,
            ) -> dict:
                # Simulate streaming events
                if on_event:
                    event1 = MagicMock()
                    event1.type = "content.delta"
                    event1.content = "Hello"
                    on_event(event1)

                    event2 = MagicMock()
                    event2.type = "content.delta"
                    event2.content = " world"
                    on_event(event2)

                return {"response": "Hello world"}

        manager = AITurnManager(
            orchestrator_provider=lambda: StreamingOrchestrator(),
            event_bus=event_bus,
        )

        events: list[AITurnStreamChunk] = []
        event_bus.subscribe(AITurnStreamChunk, events.append)

        await manager.start_turn("Test", {}, {}, None)

        assert len(events) == 2
        assert events[0].content == "Hello"
        assert events[1].content == " world"

    @pytest.mark.asyncio
    async def test_forwards_events_to_external_handler(
        self, event_bus: EventBus
    ) -> None:
        """External handler receives stream events."""
        received_events: list[Any] = []

        class StreamingOrchestrator:
            async def run_chat(
                self,
                prompt: str,
                snapshot: Any,
                *,
                metadata: Any = None,
                history: Any = None,
                on_event: Any = None,
            ) -> dict:
                if on_event:
                    event = MagicMock()
                    event.type = "content.delta"
                    event.content = "chunk"
                    on_event(event)
                return {"response": "done"}

        manager = AITurnManager(
            orchestrator_provider=lambda: StreamingOrchestrator(),
            event_bus=event_bus,
        )

        await manager.start_turn(
            "Test",
            {},
            {},
            None,
            on_stream_event=received_events.append,
        )

        assert len(received_events) == 1
        assert received_events[0].content == "chunk"


# =============================================================================
# Edit Count Tests
# =============================================================================


class TestAITurnManagerEditCount:
    """Tests for edit count tracking."""

    def test_increment_edit_count(self, ai_turn_manager: AITurnManager) -> None:
        """increment_edit_count increases count."""
        ai_turn_manager._current_turn = AITurnState(
            turn_id="test",
            prompt="test",
            status=AITurnStatus.RUNNING,
        )

        assert ai_turn_manager.current_turn.edit_count == 0
        ai_turn_manager.increment_edit_count()
        assert ai_turn_manager.current_turn.edit_count == 1
        ai_turn_manager.increment_edit_count()
        assert ai_turn_manager.current_turn.edit_count == 2

    def test_increment_no_current_turn(
        self, ai_turn_manager: AITurnManager
    ) -> None:
        """increment_edit_count is safe with no turn."""
        ai_turn_manager.increment_edit_count()  # Should not raise

    @pytest.mark.asyncio
    async def test_edit_count_in_completed_event(
        self, ai_turn_manager: AITurnManager, event_bus: EventBus
    ) -> None:
        """Edit count is included in completion event."""
        # Manually increment during the "turn"
        original_run_chat = ai_turn_manager._get_orchestrator().run_chat

        async def run_with_edits(*args: Any, **kwargs: Any) -> dict:
            ai_turn_manager.increment_edit_count()
            ai_turn_manager.increment_edit_count()
            return {"response": "done"}

        ai_turn_manager._get_orchestrator().run_chat = run_with_edits

        events: list[AITurnCompleted] = []
        event_bus.subscribe(AITurnCompleted, events.append)

        await ai_turn_manager.start_turn("Test", {}, {}, None)

        assert len(events) == 1
        assert events[0].edit_count == 2


# =============================================================================
# Response Extraction Tests
# =============================================================================


class TestAITurnManagerResponseExtraction:
    """Tests for response text extraction."""

    @pytest.mark.asyncio
    async def test_extracts_response_from_dict(
        self, event_bus: EventBus
    ) -> None:
        """Extracts response from dict result."""
        orchestrator = MockOrchestrator(response={"response": "Dict response"})
        manager = AITurnManager(
            orchestrator_provider=lambda: orchestrator,
            event_bus=event_bus,
        )

        events: list[AITurnCompleted] = []
        event_bus.subscribe(AITurnCompleted, events.append)

        await manager.start_turn("Test", {}, {}, None)

        assert events[0].response_text == "Dict response"

    @pytest.mark.asyncio
    async def test_extracts_response_from_object(
        self, event_bus: EventBus
    ) -> None:
        """Extracts response from ChatResult-like object."""
        result = MagicMock()
        result.response = "Object response"

        orchestrator = MockOrchestrator(response=result)
        manager = AITurnManager(
            orchestrator_provider=lambda: orchestrator,
            event_bus=event_bus,
        )

        events: list[AITurnCompleted] = []
        event_bus.subscribe(AITurnCompleted, events.append)

        await manager.start_turn("Test", {}, {}, None)

        assert events[0].response_text == "Object response"

    @pytest.mark.asyncio
    async def test_handles_none_response(self, event_bus: EventBus) -> None:
        """Handles None result gracefully."""
        orchestrator = MockOrchestrator(response=None, use_default=False)
        manager = AITurnManager(
            orchestrator_provider=lambda: orchestrator,
            event_bus=event_bus,
        )

        events: list[AITurnCompleted] = []
        event_bus.subscribe(AITurnCompleted, events.append)

        await manager.start_turn("Test", {}, {}, None)

        assert events[0].response_text == ""


# =============================================================================
# is_running Tests
# =============================================================================


class TestAITurnManagerIsRunning:
    """Tests for is_running() method."""

    def test_not_running_initially(self, ai_turn_manager: AITurnManager) -> None:
        """Not running with no turn."""
        assert ai_turn_manager.is_running() is False

    def test_running_when_turn_active(
        self, ai_turn_manager: AITurnManager
    ) -> None:
        """Running when turn status is RUNNING."""
        ai_turn_manager._current_turn = AITurnState(
            turn_id="t",
            prompt="p",
            status=AITurnStatus.RUNNING,
        )
        assert ai_turn_manager.is_running() is True

    def test_not_running_when_completed(
        self, ai_turn_manager: AITurnManager
    ) -> None:
        """Not running after completion."""
        ai_turn_manager._current_turn = AITurnState(
            turn_id="t",
            prompt="p",
            status=AITurnStatus.COMPLETED,
        )
        assert ai_turn_manager.is_running() is False


# =============================================================================
# Module Export Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_exported_from_domain(self) -> None:
        """AITurnManager is exported from domain package."""
        from tinkerbell.ui.domain import AITurnManager as ATM

        assert ATM is AITurnManager
