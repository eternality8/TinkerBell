"""Unit tests for :mod:`tinkerbell.ui.ai_turn_coordinator`."""

from __future__ import annotations

from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, cast

import pytest

from tinkerbell.ai.orchestration.editor_lock import EditorLockManager, LockReason, LockState
from tinkerbell.chat.message_model import ChatMessage
from tinkerbell.ui.ai_turn_coordinator import AITurnCoordinator
from tinkerbell.ui.tool_trace_presenter import ToolTracePresenter


class _RecordingController:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def run_chat(
        self,
        prompt: str,
        snapshot: dict[str, Any],
        *,
        metadata: dict[str, Any],
        history: list[dict[str, Any]] | None,
        on_event: Callable[[Any], Any],
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "prompt": prompt,
                "snapshot": snapshot,
                "metadata": metadata,
                "history": history,
            }
        )
        # on_event is a sync callback, don't await it
        on_event(SimpleNamespace(type="content.delta", content="partial"))
        return {
            "response": " done ",
            "tool_calls": [{"id": "call-123"}],
            "trace_compaction": {"tokens_saved": 42},
        }


class _FailingController:
    async def run_chat(self, *args: Any, **kwargs: Any) -> Any:  # noqa: D401
        raise RuntimeError("boom")


class _FakeChatPanel:
    def __init__(self) -> None:
        self.messages: list[tuple[ChatMessage, bool]] = []

    def append_ai_message(self, message: ChatMessage, *, streaming: bool) -> None:
        self.messages.append((message, streaming))


class _FakePresenter:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.annotated_records: Any = None
        self.argument_events: list[Any] = []
        self.finalized_events: list[Any] = []
        self.result_events: list[Any] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def annotate_compaction(self, records: Any) -> None:
        self.annotated_records = records

    def record_arguments_delta(self, event: Any) -> None:
        self.argument_events.append(event)

    def finalize_arguments(self, event: Any) -> None:
        self.finalized_events.append(event)

    def record_result(self, event: Any) -> None:
        self.result_events.append(event)


class _FakeTelemetry:
    def __init__(self) -> None:
        self.compaction_stats: Any = None
        self.refresh_calls = 0

    def set_compaction_stats(self, stats: Any) -> None:
        self.compaction_stats = stats

    def refresh_context_usage_status(self) -> None:
        self.refresh_calls += 1


class _FakeReviewController:
    def __init__(self) -> None:
        self.finalizations: list[bool] = []

    def finalize_pending_turn_review(self, *, success: bool) -> None:
        self.finalizations.append(success)


def _build_coordinator(controller: Any | None) -> tuple[AITurnCoordinator, SimpleNamespace]:
    tracker = SimpleNamespace(statuses=[], responses=[], failures=[], stream_states=[])
    chat_panel = _FakeChatPanel()
    presenter = _FakePresenter()
    telemetry = _FakeTelemetry()
    review = _FakeReviewController()

    coordinator = AITurnCoordinator(
        controller_resolver=lambda: controller,
        chat_panel=chat_panel,
        review_controller=review,
        telemetry_controller=telemetry,
        status_updater=lambda message: tracker.statuses.append(message),
        failure_handler=lambda exc: tracker.failures.append(exc),
        response_finalizer=lambda text: tracker.responses.append(text),
        tool_trace_presenter=cast(ToolTracePresenter, presenter),
        stream_state_setter=lambda active: tracker.stream_states.append(active),
        stream_text_coercer=lambda payload: str(payload or "").strip(),
    )

    tracker.chat_panel = chat_panel
    tracker.presenter = presenter
    tracker.telemetry = telemetry
    tracker.review = review
    return coordinator, tracker


@pytest.mark.asyncio
async def test_run_ai_turn_invokes_controller_and_updates_components() -> None:
    controller = _RecordingController()
    coordinator, tracker = _build_coordinator(controller)

    metadata = MappingProxyType({"selection": "doc"})
    history = [{"role": "user", "content": "Hello"}]
    snapshot = {"document": "text"}

    await coordinator.run_ai_turn(prompt="Summarize", snapshot=snapshot, metadata=metadata, history=history)

    assert controller.calls and controller.calls[0]["prompt"] == "Summarize"
    recorded_metadata = controller.calls[0]["metadata"]
    assert isinstance(recorded_metadata, dict)
    assert recorded_metadata == {"selection": "doc"}
    assert recorded_metadata is not metadata

    assert tracker.presenter.reset_calls == 1
    assert tracker.presenter.annotated_records == [{"id": "call-123"}]
    assert tracker.telemetry.compaction_stats == {"tokens_saved": 42}
    assert tracker.telemetry.refresh_calls == 1
    assert tracker.review.finalizations == [True]
    assert tracker.responses == ["done"]
    assert tracker.statuses == ["AI thinking…", "AI response ready"]
    assert tracker.stream_states == [False, True]

    assert len(tracker.chat_panel.messages) == 1
    message, streaming = tracker.chat_panel.messages[0]
    assert streaming is True
    assert isinstance(message, ChatMessage)
    assert message.content == "partial"


@pytest.mark.asyncio
async def test_run_ai_turn_returns_early_when_controller_missing() -> None:
    coordinator, tracker = _build_coordinator(controller=None)

    await coordinator.run_ai_turn(prompt="Hi", snapshot={}, metadata={}, history=None)

    assert tracker.presenter.reset_calls == 0
    assert tracker.statuses == []
    assert tracker.stream_states == []


@pytest.mark.asyncio
async def test_run_ai_turn_handles_controller_failures() -> None:
    coordinator, tracker = _build_coordinator(_FailingController())

    await coordinator.run_ai_turn(prompt="Hi", snapshot={}, metadata={}, history=None)

    assert len(tracker.failures) == 1
    assert isinstance(tracker.failures[0], RuntimeError)
    assert tracker.review.finalizations == [False]
    assert tracker.responses == []
    assert tracker.statuses == ["AI thinking…"]
    assert tracker.telemetry.refresh_calls == 0
    assert tracker.presenter.annotated_records is None


def test_process_stream_event_routes_tool_and_content_events() -> None:
    coordinator, tracker = _build_coordinator(controller=None)

    coordinator._process_stream_event(SimpleNamespace(type="content.delta", content=" first "))
    assert len(tracker.chat_panel.messages) == 1
    assert tracker.chat_panel.messages[0][0].content == "first"
    assert tracker.stream_states == [True]

    coordinator._process_stream_event(SimpleNamespace(type="content.done", content="ignored"))
    assert len(tracker.chat_panel.messages) == 1

    coordinator._ai_stream_active = False
    coordinator._process_stream_event(SimpleNamespace(type="content.done", content="final"))
    assert len(tracker.chat_panel.messages) == 2
    assert tracker.chat_panel.messages[-1][0].content == "final"

    arg_delta = SimpleNamespace(type="tool_calls.function.arguments.delta", tool_call_id="t-1")
    coordinator._process_stream_event(arg_delta)
    assert tracker.presenter.argument_events == [arg_delta]

    arg_done = SimpleNamespace(type="tool_calls.function.arguments.done", tool_call_id="t-1")
    coordinator._process_stream_event(arg_done)
    assert tracker.presenter.finalized_events == [arg_done]

    result_event = SimpleNamespace(type="tool_calls.result", tool_call_id="t-1")
    coordinator._process_stream_event(result_event)
    assert tracker.presenter.result_events == [result_event]


# =============================================================================
# Editor Lock Integration Tests
# =============================================================================


def _build_coordinator_with_lock(controller: Any | None) -> tuple[AITurnCoordinator, SimpleNamespace, EditorLockManager]:
    """Build coordinator with an editor lock manager."""
    tracker = SimpleNamespace(statuses=[], responses=[], failures=[], stream_states=[])
    chat_panel = _FakeChatPanel()
    presenter = _FakePresenter()
    telemetry = _FakeTelemetry()
    review = _FakeReviewController()
    lock_manager = EditorLockManager()

    coordinator = AITurnCoordinator(
        controller_resolver=lambda: controller,
        chat_panel=chat_panel,
        review_controller=review,
        telemetry_controller=telemetry,
        status_updater=lambda message: tracker.statuses.append(message),
        failure_handler=lambda exc: tracker.failures.append(exc),
        response_finalizer=lambda text: tracker.responses.append(text),
        tool_trace_presenter=cast(ToolTracePresenter, presenter),
        stream_state_setter=lambda active: tracker.stream_states.append(active),
        stream_text_coercer=lambda payload: str(payload or "").strip(),
        lock_manager=lock_manager,
    )

    tracker.chat_panel = chat_panel
    tracker.presenter = presenter
    tracker.telemetry = telemetry
    tracker.review = review
    return coordinator, tracker, lock_manager


@pytest.mark.asyncio
async def test_run_ai_turn_acquires_and_releases_lock() -> None:
    """Editor lock is acquired at start and released after successful completion."""
    controller = _RecordingController()
    coordinator, tracker, lock_manager = _build_coordinator_with_lock(controller)

    # Lock starts unlocked
    assert lock_manager.state == LockState.UNLOCKED
    assert not lock_manager.is_locked

    await coordinator.run_ai_turn(prompt="Test", snapshot={}, metadata={}, history=None)

    # Lock should be released after turn completes
    assert lock_manager.state == LockState.UNLOCKED
    assert not lock_manager.is_locked
    assert coordinator._active_lock_session is None


@pytest.mark.asyncio
async def test_run_ai_turn_releases_lock_on_failure() -> None:
    """Editor lock is released even when the controller fails."""
    coordinator, tracker, lock_manager = _build_coordinator_with_lock(_FailingController())

    await coordinator.run_ai_turn(prompt="Test", snapshot={}, metadata={}, history=None)

    # Lock should be released despite the failure
    assert lock_manager.state == LockState.UNLOCKED
    assert not lock_manager.is_locked
    assert len(tracker.failures) == 1


def test_force_release_lock_unlocks_editor() -> None:
    """force_release_lock releases the lock immediately."""
    coordinator, tracker, lock_manager = _build_coordinator_with_lock(None)

    # Manually acquire a lock
    session = lock_manager.acquire(LockReason.AI_TURN)
    assert lock_manager.is_locked
    coordinator._active_lock_session = session

    # Force release
    coordinator.force_release_lock()

    assert not lock_manager.is_locked
    assert coordinator._active_lock_session is None


def test_coordinator_without_lock_manager_works() -> None:
    """Coordinator works fine without a lock manager (lock_manager=None)."""
    coordinator, tracker = _build_coordinator(controller=None)

    # Should not raise even without lock manager
    coordinator._acquire_editor_lock()
    coordinator._release_editor_lock()
    coordinator.force_release_lock()
