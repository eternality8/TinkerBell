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


def test_chunk_flow_events_update_indicators() -> None:
    _clear_chunk_flow_listeners()
    status_bar = StatusBar()
    chat_probe = _ChatProbe()
    controller = TelemetryController(
        status_bar=status_bar,
        context=WindowContext(unsaved_cache=UnsavedCache()),
        initial_subagent_enabled=False,
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
