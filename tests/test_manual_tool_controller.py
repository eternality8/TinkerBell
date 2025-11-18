from __future__ import annotations

from typing import Any, Mapping

from tinkerbell.chat.message_model import ToolTrace
from tinkerbell.ui.manual_tool_controller import ManualToolController


class _ChatPanelStub:
    def __init__(self) -> None:
        self.traces: list[ToolTrace] = []

    def show_tool_trace(self, trace: ToolTrace) -> None:
        self.traces.append(trace)


def _make_controller() -> tuple[ManualToolController, _ChatPanelStub]:
    panel = _ChatPanelStub()
    controller = ManualToolController(
        chat_panel=panel,
        document_label_resolver=lambda doc_id, fallback=None: doc_id or fallback or "Untitled",
        whitespace_normalizer=lambda text: " ".join(text.split()),
    )
    return controller, panel


def test_render_outline_response_lists_headings_and_notes() -> None:
    controller, _ = _make_controller()
    response: Mapping[str, Any] = {
        "status": "ok",
        "document_id": "Doc A",
        "nodes": [{"text": "Top", "children": [{"text": "Child"}]}],
        "trimmed": True,
        "trimmed_reason": "max_nodes",
        "generated_at": "2025-11-01T00:00:00Z",
        "outline_digest": "abc123",
    }

    message = controller.render_outline_response(response, "Requested")

    assert "Document outline (ok) for Doc A." in message
    assert "- Top" in message and "- Child" in message
    assert "trimmed=max_nodes" in message
    assert "Digest: abc123." in message


def test_render_retrieval_response_formats_details_and_matches() -> None:
    controller, _ = _make_controller()
    response = {
        "status": "ok",
        "document_id": "Doc B",
        "query": "foo",
        "pointers": [
            {
                "pointer_id": "p1",
                "outline_context": {"heading": "Intro"},
                "score": 0.98,
                "preview": "First   line\nSecond line",
            },
            {"pointer_id": "p2", "preview": " extra "},
            {"pointer_id": "p3"},
            {"pointer_id": "p4"},
            {"pointer_id": "p5"},
            {"pointer_id": "p6"},
        ],
        "latency_ms": 12.3,
    }

    message = controller.render_retrieval_response(response, None, None)

    assert "Find sections (ok) for Doc B" in message
    assert "Matches:" in message
    assert "1. p1 · Intro" in message
    assert "… 1 additional match(es)." in message


def test_record_manual_tool_trace_attaches_metadata() -> None:
    controller, panel = _make_controller()

    controller.record_manual_tool_trace(
        name="manual:test",
        input_summary="summary",
        output_summary="ok",
        args={"a": 1},
        response={"status": "ok"},
    )

    assert len(panel.traces) == 1
    trace = panel.traces[0]
    assert trace.name == "manual:test"
    assert trace.metadata["manual_command"] is True
    assert "raw_input" in trace.metadata and "raw_output" in trace.metadata


def test_summarize_manual_input_prefers_known_fields() -> None:
    summary = ManualToolController.summarize_manual_input(
        "document_find_sections",
        {"query": "foo", "document_id": "doc", "extra": 1},
    )

    assert "document_id=doc" in summary
    assert "query=foo" in summary
