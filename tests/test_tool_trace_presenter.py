"""Unit tests for :mod:`tinkerbell.ui.tool_trace_presenter`."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from tinkerbell.chat.message_model import ToolTrace
from tinkerbell.ui.models.tool_traces import PendingToolTrace
from tinkerbell.ui.tool_trace_presenter import ToolTracePresenter


class _FakeChatPanel:
    def __init__(self) -> None:
        self.shown: list[ToolTrace] = []
        self.updated: list[ToolTrace] = []

    def show_tool_trace(self, trace: ToolTrace) -> None:
        self.shown.append(trace)

    def update_tool_trace(self, trace: ToolTrace) -> None:
        self.updated.append(trace)


class _FakeClock:
    def __init__(self, *, start: float = 10.0, delta: float = 0.25) -> None:
        self.current = start
        self.delta = delta

    def __call__(self) -> float:
        value = self.current
        self.current += self.delta
        return value


def _make_presenter(**overrides: Any) -> tuple[ToolTracePresenter, _FakeChatPanel, dict[str, ToolTrace]]:
    chat_panel = overrides.get("chat_panel", _FakeChatPanel())
    summarize = overrides.get("summarize")
    clock = overrides.get("clock", _FakeClock())
    index: dict[str, ToolTrace] = overrides.get("index", {})
    presenter = ToolTracePresenter(
        chat_panel=chat_panel,
        tool_trace_index=index,
        summarize_io=summarize,
        clock=clock,
    )
    return presenter, chat_panel, index


def test_presenter_records_arguments_and_results_with_duration() -> None:
    clock = _FakeClock(delta=0.5)
    presenter, chat_panel, index = _make_presenter(
        summarize=lambda payload: f"summary:{payload}",
        clock=clock,
    )

    presenter.record_arguments_delta(SimpleNamespace(tool_call_id="call-1", tool_name="search", arguments_delta="{\"q\":"))
    presenter.record_arguments_delta(SimpleNamespace(tool_call_id="call-1", arguments_delta="term\"}"))
    presenter.record_result(
        SimpleNamespace(tool_call_id="call-1", content="Full output", parsed={"total": 1})
    )
    presenter.finalize_arguments(
        SimpleNamespace(tool_call_id="call-1", tool_arguments='{"q":"term"}', tool_name="search")
    )

    assert len(chat_panel.shown) == 1
    assert len(chat_panel.updated) == 1
    trace = chat_panel.updated[0]
    assert trace.input_summary == 'summary:{"q":"term"}'
    assert trace.output_summary == "summary:Full output"
    assert trace.metadata["raw_input"] == '{"q":"term"}'
    assert trace.metadata["raw_output"] == "Full output"
    assert trace.metadata["parsed_output"] == {"total": 1}
    assert trace.metadata["tool_call_id"] == "call-1"
    assert index["call-1"] is trace
    assert trace.duration_ms == 500
    assert presenter.pending_tool_traces == {}


def test_presenter_buffers_results_until_arguments_known() -> None:
    presenter, chat_panel, index = _make_presenter()

    presenter.record_result(SimpleNamespace(name="math", tool_index=2, content="42"))
    assert chat_panel.shown == []
    assert chat_panel.updated == []

    presenter.record_arguments_delta(SimpleNamespace(name="math", tool_index=2, arguments_delta="{"))
    presenter.record_arguments_delta(SimpleNamespace(name="math", tool_index=2, arguments_delta="}"))
    presenter.finalize_arguments(SimpleNamespace(name="math", tool_index=2, tool_name="math"))

    assert len(chat_panel.shown) == 1
    assert len(chat_panel.updated) == 1
    trace = chat_panel.updated[-1]
    assert trace.output_summary == "42"
    assert trace.metadata["raw_output"] == "42"
    assert trace.metadata["raw_input"] == "{}"
    assert "parsed_output" not in trace.metadata
    assert index["math:2"] is trace


def test_annotate_compaction_updates_trace_metadata() -> None:
    presenter, chat_panel, index = _make_presenter()
    trace = ToolTrace(name="search", input_summary="in", output_summary="out", metadata={"tool_call_id": "call-9"})
    index["call-9"] = trace

    pointer = {
        "pointer_id": "ptr-1",
        "display_text": "short summary",
        "rehydrate_instructions": "rehydrate ops",
    }

    presenter.annotate_compaction([
        {"id": "call-9", "pointer": pointer},
    ])

    assert trace.metadata["compacted"] is True
    assert trace.metadata["pointer"] == pointer
    assert trace.metadata["pointer_instructions"] == "rehydrate ops"
    assert trace.metadata["pointer_summary"] == "short summary"
    assert trace.output_summary == "short summary"
    assert chat_panel.updated[-1] is trace


def test_reset_clears_pending_and_index() -> None:
    presenter, _, index = _make_presenter()
    presenter.pending_tool_traces["temp"] = PendingToolTrace(name="temp")
    index["call"] = ToolTrace(name="x", input_summary="in", output_summary="out")

    presenter.reset()

    assert presenter.pending_tool_traces == {}
    assert index == {}