"""Telemetry controller chunk-flow integration tests."""

from __future__ import annotations

from tinkerbell.services import telemetry as telemetry_service
from tinkerbell.services.unsaved_cache import UnsavedCache
from tinkerbell.ui.models.window_state import WindowContext
from tinkerbell.ui.telemetry_controller import TelemetryController
from tinkerbell.widgets.status_bar import StatusBar


class _ChatProbe:
    def __init__(self) -> None:
        self.state: tuple[str, str] = ("", "")

    def set_guardrail_state(
        self,
        status: str | None,
        *,
        detail: str | None = None,
        category: str | None = None,
    ) -> None:
        normalized = (status or "").strip()
        detail_text = (detail or "").strip()
        self.state = (normalized, detail_text)


def _clear_chunk_flow_listeners() -> None:
    for name in (
        "chunk_flow.requested",
        "chunk_flow.escaped_full_snapshot",
        "chunk_flow.retry_success",
    ):
        telemetry_service._EVENT_LISTENERS.pop(name, None)  # type: ignore[attr-defined]


def _clear_subagent_listeners() -> None:
    for name in (
        "subagent.job_started",
        "subagent.job_completed",
        "subagent.job_failed",
        "subagent.job_skipped",
        "subagent.jobs_queued",
    ):
        telemetry_service._EVENT_LISTENERS.pop(name, None)  # type: ignore[attr-defined]


def test_chunk_flow_events_update_indicators() -> None:
    _clear_chunk_flow_listeners()
    status_bar = StatusBar()
    chat_probe = _ChatProbe()
    controller = TelemetryController(
        status_bar=status_bar,
        context=WindowContext(unsaved_cache=UnsavedCache()),
        chat_panel=chat_probe,
    )

    controller.register_chunk_flow_listeners()

    telemetry_service.emit(
        "chunk_flow.escaped_full_snapshot",
        {"document_id": "doc-123", "document_length": 120000},
    )

    warning_state = status_bar.chunk_flow_state
    assert warning_state[0] == "Chunk Flow Warning"
    assert "doc-123" in warning_state[1]
    assert chat_probe.state[0] == "Chunk Flow Warning"

    telemetry_service.emit(
        "chunk_flow.retry_success",
        {"document_id": "doc-123", "recovered_via": "document_chunk"},
    )

    recovered_state = status_bar.chunk_flow_state
    assert recovered_state[0] == "Chunk Flow Recovered"
    assert "document_chunk" in recovered_state[1]

    controller.reset_chunk_flow_state()
    assert status_bar.chunk_flow_state == ("", "")
    assert chat_probe.state == ("", "")


def test_subagent_queue_events_drive_indicator() -> None:
    _clear_subagent_listeners()
    status_bar = StatusBar()
    controller = TelemetryController(
        status_bar=status_bar,
        context=WindowContext(unsaved_cache=UnsavedCache()),
    )

    controller.register_subagent_listeners()

    try:
        telemetry_service.emit(
            "subagent.jobs_queued",
            {
                "job_ids": ["job-1", "job-2"],
                "chunk_ids": ["chunk-a", "chunk-b", "chunk-c", "chunk-d"],
                "reasons": ["dirty_chunks", "long_selection"],
            },
        )

        status, detail = status_bar.subagent_state
        assert status == "Queued (2)"
        assert "2 queued jobs" in detail
        assert "chunks chunk-a, chunk-b, chunk-c, +1 more" in detail
        assert "reasons: dirty_chunks, long_selection" in detail

        telemetry_service.emit("subagent.job_started", {"job_id": "job-1"})
        status, detail = status_bar.subagent_state
        assert status == "Running (1)"
        assert "1 active job" in detail
        assert "1 queued job" in detail

        telemetry_service.emit("subagent.job_completed", {"job_id": "job-1", "tokens_used": 120})
        status, detail = status_bar.subagent_state
        assert status == "Queued (1)"
        assert "1 queued job" in detail

        telemetry_service.emit("subagent.job_started", {"job_id": "job-2"})
        telemetry_service.emit("subagent.job_skipped", {"job_id": "job-2", "reason": "cache_hit"})

        status, detail = status_bar.subagent_state
        assert status == "Idle"
        assert "queued" not in detail
    finally:
        _clear_subagent_listeners()
