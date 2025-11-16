"""Tests for the services.telemetry helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinkerbell.ai.services.telemetry import ContextUsageEvent
from tinkerbell.services import telemetry as telemetry_service


def _make_event(**overrides: object) -> ContextUsageEvent:
    base: dict[str, object] = {
        "document_id": "doc-123",
        "model": "gpt-test",
        "prompt_tokens": 120,
        "tool_tokens": 15,
        "response_reserve": 60,
        "timestamp": 1.0,
        "conversation_length": 2,
        "tool_names": ("search", "apply_patch"),
        "run_id": "run-1",
    }
    base.update(overrides)
    return ContextUsageEvent(**base)  # type: ignore[arg-type]


def test_summarize_usage_event_clamps_invalid_values() -> None:
    event = _make_event(prompt_tokens=-5, tool_tokens=-10, tool_names=("diff",))
    summary = telemetry_service.summarize_usage_event(event)
    assert summary is not None
    assert summary.prompt_tokens == 0
    assert summary.tool_tokens == 0
    assert summary.last_tool == "diff"


def test_summarize_usage_events_supports_generators() -> None:
    event_a = _make_event(prompt_tokens=10)
    event_b = _make_event(prompt_tokens=45, tool_names=("search", "validate"))
    summary = telemetry_service.summarize_usage_events(event for event in (event_a, event_b))
    assert summary is not None
    assert summary.prompt_tokens == 45
    assert summary.tool_names == ("search", "validate")


def test_format_usage_summary_highlights_last_tool_and_reserve() -> None:
    event = _make_event(prompt_tokens=500, tool_tokens=80, response_reserve=32, tool_names=("list_tabs", "patch"))
    status = telemetry_service.format_usage_summary([event])
    assert status is not None
    assert "Prompt 500" in status
    assert "Tools 80" in status
    assert "Reserve 32" in status
    assert "Last tool patch" in status


def test_count_text_tokens_prefers_precise_and_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Counter:
        def __init__(self) -> None:
            self.count_calls: list[str] = []
            self.estimate_calls: list[str] = []

        def count(self, text: str) -> int:
            self.count_calls.append(text)
            if "boom" in text:
                raise RuntimeError("count failed")
            return len(text)

        def estimate(self, text: str) -> int:
            self.estimate_calls.append(text)
            return 99

    counter = _Counter()
    monkeypatch.setattr(telemetry_service, "get_token_counter", lambda model=None: counter)

    assert telemetry_service.count_text_tokens("abc") == 3
    assert counter.count_calls == ["abc"]

    assert telemetry_service.count_text_tokens("boom text") == 99
    assert counter.estimate_calls == ["boom text"]

    assert telemetry_service.count_text_tokens("estimate only", estimate_only=True) == 99
    assert counter.estimate_calls[-1] == "estimate only"


def test_token_counter_status_reflects_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry_service.ai_client, "tiktoken", object())
    status = telemetry_service.token_counter_status()
    assert status.precise is True
    assert "tiktoken" in status.source.lower()


def test_token_counter_status_reports_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(telemetry_service.ai_client, "tiktoken", None)
    status = telemetry_service.token_counter_status()
    assert status.precise is False
    assert "approximate byte counter" in status.source


def test_build_usage_dashboard_combines_summary_and_totals() -> None:
    events = [
        _make_event(prompt_tokens=50, tool_tokens=10, tool_names=("diff",)),
        _make_event(prompt_tokens=25, tool_tokens=5, tool_names=("patch",)),
    ]
    dashboard = telemetry_service.build_usage_dashboard(events)
    assert dashboard is not None
    assert dashboard.summary_text.startswith("Prompt 25")
    assert "Σ Prompt 75" in dashboard.totals_text
    assert "Events 2" in dashboard.totals_text
    assert dashboard.summary.last_tool == "patch"


def test_persistent_sink_persists_and_limits(tmp_path: Path) -> None:
    path = tmp_path / "telemetry" / "context_usage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    sink = telemetry_service.PersistentTelemetrySink(path=path, capacity=10)
    for idx in range(12):
        sink.record(_make_event(run_id=f"run-{idx}", prompt_tokens=idx, tool_tokens=idx))

    assert path.exists()
    events = telemetry_service.load_persistent_events(path)
    assert len(events) == 10
    assert [event.run_id for event in events][:2] == ["run-2", "run-3"]


def test_load_persistent_events_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "telemetry2" / "context_usage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    sink = telemetry_service.PersistentTelemetrySink(path=path, capacity=5)
    for idx in range(4):
        sink.record(_make_event(run_id=f"run-{idx}", prompt_tokens=idx + 1, tool_tokens=idx))

    events = telemetry_service.load_persistent_events(path, limit=2)
    assert [event.run_id for event in events] == ["run-2", "run-3"]


def test_register_event_listener_receives_payload() -> None:
    received: list[dict[str, object]] = []
    event_name = "test-budget-event"

    def _listener(payload: dict[str, object]) -> None:
        received.append(payload)

    telemetry_service.register_event_listener(event_name, _listener)
    telemetry_service.emit(event_name, {"verdict": "ok", "deficit": 0})

    assert received
    assert received[-1]["verdict"] == "ok"


def test_emit_handles_missing_listeners() -> None:
    # Should not raise even when no listeners are registered.
    telemetry_service.emit("nonexistent-event", {"value": 1})
