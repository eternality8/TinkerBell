"""Regression coverage for the document status window headless logic."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import tinkerbell.ui.presentation.dialogs.document_status_window as document_status_window_module
from tinkerbell.ui.document_status import DocumentDescriptor
from tinkerbell.ui.presentation.dialogs.document_status_window import DocumentStatusWindow


def _ensure_qapp() -> None:
    qt_widgets = pytest.importorskip("PySide6.QtWidgets")
    app = qt_widgets.QApplication.instance()
    if app is None:  # pragma: no cover - depends on PySide6 availability
        qt_widgets.QApplication([])


def _build_window(payload: dict[str, object], *, enable_qt: bool = False) -> DocumentStatusWindow:
    document_payload = dict(payload.get("document", {}))

    def _loader(document_id: str | None) -> dict[str, object]:
        resolved_id = document_id or document_payload.get("document_id") or "doc-1"
        merged_document = {**document_payload, "document_id": resolved_id}
        return {**payload, "document": merged_document}

    return DocumentStatusWindow(
        documents=[DocumentDescriptor(document_id="doc-1", label="Doc", tab_id="tab-1")],
        status_loader=_loader,
        enable_qt=enable_qt,
    )


def test_document_status_window_show_returns_payload(tmp_path: Path) -> None:
    payload: dict[str, object] = {
        "chunks": {"chunk_profile": "auto"},
        "summary": "Doc status",
    }
    window = _build_window(payload)

    result = window.show()

    assert result is not None

    assert result["document"]["document_id"] == "doc-1"
    output_path = tmp_path / "status.json"
    body = window.export_payload(output_path)
    assert json.loads(body)["summary"] == "Doc status"
    assert output_path.read_text(encoding="utf-8").strip() == body.strip()


def test_document_status_window_update_documents_replaces_entries() -> None:
    payload: dict[str, object] = {"summary": "ready"}
    window = _build_window(payload)

    new_docs = [
        DocumentDescriptor(document_id="doc-2", label="Doc 2", tab_id="tab-2"),
        DocumentDescriptor(document_id="doc-3", label="Doc 3", tab_id="tab-3"),
    ]

    window.update_documents(new_docs)

    assert len(window._documents) == 2  # noqa: SLF001 - intentional white-box check
    assert window._documents[0].document_id == "doc-2"


def test_document_status_window_refresh_reloads_current_document() -> None:
    calls: list[str | None] = []

    def _loader(document_id: str | None) -> dict[str, object]:
        calls.append(document_id)
        return {
            "chunks": {},
            "summary": "ok",
            "document": {"document_id": document_id or "doc-1"},
        }

    window = DocumentStatusWindow(
        documents=[
            DocumentDescriptor(document_id="doc-1", label="Doc 1", tab_id="tab-1"),
        ],
        status_loader=_loader,
        enable_qt=False,
    )

    window.show(document_id="doc-1")
    window.refresh()

    assert len(calls) == 2
    assert calls[-1] == "doc-1"


def test_format_chunk_lines_returns_human_summary() -> None:
    chunk_payload: dict[str, Any] = {
        "chunk_profile": "precise",
        "chunk_manifest": {
            "generated_at": "2024-05-01T12:00:00Z",
            "chunks": [
                {"id": "a"},
                {"id": "b"},
                {"id": "c"},
            ],
        },
        "window": {
            "start": 0,
            "end": 512,
        },
        "document_version": {
            "version": "v7",
            "content_hash": "abc123",
        },
    }

    lines = DocumentStatusWindow._format_chunk_lines(chunk_payload)

    assert "Chunks: 3 (profile precise)" in lines[0]
    assert any(line.startswith("Window:") for line in lines)
    assert any(line.startswith("Manifest generated at") for line in lines)
    assert any(line.startswith("Document version:") for line in lines)


def test_format_outline_lines_summarizes_status() -> None:
    outline_payload: dict[str, Any] = {
        "status": "ok",
        "node_count": 8,
        "updated_at": "2025-11-19T19:45:00Z",
        "version_id": "outline-v3",
        "outline_hash": "deadbeef",
        "summary": "Outline is synced through chapter 8.",
        "highlights": [
            "Chapter 5 still needs revision",
            {"text": "Add continuity check for Act 3"},
        ],
    }
    document_payload = {"version_id": "doc-v4"}

    lines = DocumentStatusWindow._format_outline_lines(outline_payload, document_payload)

    assert any(line.startswith("Outline status:") for line in lines)
    assert any("Version mismatch" in line for line in lines)
    assert "Summary:" in lines
    assert any(line.startswith("- ") for line in lines if "Highlights:" not in line)


def test_format_plot_lines_lists_arcs_and_entities() -> None:
    plot_payload: dict[str, Any] = {
        "entity_count": 4,
        "arc_count": 2,
        "version_id": "plot-v1",
        "generated_at": "2025-11-18T10:00:00Z",
        "arcs": [
            {
                "name": "Primary arc",
                "summary": "Hero confronts rival",
                "beats": [{"summary": "Setup"}, {"summary": "Conflict"}],
            }
        ],
        "entities": [
            {"name": "Aerin", "summary": "Protagonist", "salience": 0.87},
            {"name": "Boros"},
        ],
        "metadata": {"stats": {"ingested_chunks": 12}},
        "overrides": [
            {"override_id": "ov1", "summary": "Force alternate ending", "author": "qa"},
        ],
    }

    lines = DocumentStatusWindow._format_plot_lines(plot_payload)

    assert any("Plot state" in line for line in lines)
    assert any(line.startswith("Arcs:") for line in lines)
    assert any(line.startswith("Entities:") for line in lines)
    assert any("Chunks ingested" in line for line in lines)
    assert any("Overrides:" in line for line in lines)
    assert any("ov1" in line for line in lines)


def test_document_status_window_qt_updates_badge_and_tabs() -> None:
    _ensure_qapp()
    payload: dict[str, Any] = {
        "document": {
            "document_id": "doc-1",
            "label": "Novel Draft",
            "language": "en",
            "length": 1_024,
            "version_id": "v7",
            "content_hash": "hash",
        },
        "badge": {"status": "Chunk Flow Warning", "detail": "Full snapshot fallback", "severity": "warning"},
        "summary": "Document is ready with chunk warnings.",
        "chunks": {
            "chunk_profile": "precise",
            "stats": {"chunk_count": 3, "generated_at": "2025-11-19T10:00:00Z"},
            "document_version": {"version": "v7"},
        },
        "outline": {
            "status": "ok",
            "node_count": 8,
            "updated_at": "2025-11-19T19:45:00Z",
            "version_id": "outline-v3",
            "summary": "Outline synced",
        },
        "plot": {
            "entity_count": 2,
            "arc_count": 1,
            "arcs": [{"name": "Arc", "beats": [{"summary": "Beat"}]}],
            "overrides": [{"override_id": "ov1", "summary": "Force ending", "author": "qa"}],
        },
        "concordance": {
            "entity_count": 1,
            "entities": [{"name": "Hero", "mention_count": 4}],
        },
        "planner": {
            "pending": 1,
            "completed": 2,
            "tasks": [{"task_id": "t1", "status": "pending", "note": "Review"}],
        },
        "telemetry": {
            "chunk_flow": {"status": "Chunk Flow Warning", "detail": "Full snapshot fallback"},
            "analysis": {
                "status": "Outline refresh required",
                "badge": "Preflight: Outline",
                "detail": "Chunk profile precise",
            },
        },
    }
    window = _build_window(payload, enable_qt=True)

    window.show(document_id="doc-1")

    assert window._badge_label.text() == "Chunk Flow Warning"  # noqa: SLF001
    assert window._badge_label.toolTip() == "Full snapshot fallback"  # noqa: SLF001
    assert "Chunk flow" in window._telemetry_view.toPlainText()  # noqa: SLF001
    assert window._telemetry_view.property("data-severity") == "warning"  # noqa: SLF001
    assert "Plot state" in window._plot_view.toPlainText()  # noqa: SLF001
    assert window._metadata_labels["document"].text() == "Novel Draft"  # noqa: SLF001


def test_document_status_window_save_action_exports_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_qapp()
    payload: dict[str, Any] = {
        "document": {"document_id": "doc-1", "label": "Doc", "language": "en"},
        "badge": {"status": "Doc Ready", "detail": ""},
        "summary": "Doc ready",
    }
    window = _build_window(payload, enable_qt=True)
    window.show()

    target = tmp_path / "status.json"

    class _FakeFileDialog:
        @staticmethod
        def getSaveFileName(*_: Any) -> tuple[str, str]:
            return str(target), "json"

    messages: list[str] = []

    class _FakeMessageBox:
        @staticmethod
        def information(_parent: Any, _title: str, text: str) -> None:
            messages.append(text)

    fake_dialog = SimpleNamespace(getSaveFileName=_FakeFileDialog.getSaveFileName)
    monkeypatch.setattr(document_status_window_module, "QFileDialog", fake_dialog)
    monkeypatch.setattr(document_status_window_module, "QMessageBox", _FakeMessageBox)

    window._handle_save_clicked()  # noqa: SLF001

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == window._last_payload  # noqa: SLF001
    assert any("Saved to" in message for message in messages)


def test_format_concordance_lines_reports_entities() -> None:
    concordance_payload: dict[str, Any] = {
        "entity_count": 3,
        "generated_at": "2025-11-19T11:05:00Z",
        "version_id": "doc-v5",
        "stats": {"ingested_chunks": 5},
        "entities": [
            {
                "name": "Celine",
                "pronouns": ["she", "her"],
                "mention_count": 7,
            },
            {
                "name": "Darius",
                "pronouns": ["he"],
                "mention_count": 3,
            },
        ],
    }

    lines = DocumentStatusWindow._format_concordance_lines(concordance_payload)

    assert any("Concordance:" in line for line in lines)
    assert any("Celine" in line for line in lines)
    assert any("Darius" in line for line in lines)


def test_format_planner_lines_highlights_tasks() -> None:
    planner_payload: dict[str, Any] = {
        "pending": 2,
        "completed": 1,
        "tasks": [
            {"task_id": "t1", "status": "pending", "note": "Outline Act 2"},
            {"task_id": "t2", "status": "completed", "note": "Finalize cast"},
        ],
    }

    lines = DocumentStatusWindow._format_planner_lines(planner_payload)

    assert lines[0].startswith("Planner:")
    assert any("Outline Act 2" in line for line in lines)


def test_format_telemetry_lines_shows_chunk_flow_and_analysis() -> None:
    telemetry_payload: dict[str, Any] = {
        "chunk_flow": {"status": "warning", "detail": "Full snapshot fallback"},
        "analysis": {"badge": "Preflight: Outline refresh", "detail": "Chunk profile precise"},
    }

    lines = DocumentStatusWindow._format_telemetry_lines(telemetry_payload)

    assert any("Chunk flow" in line for line in lines)
    assert any("Preflight" in line for line in lines)


def test_determine_telemetry_severity_prioritizes_chunk_flow() -> None:
    telemetry_payload: dict[str, Any] = {
        "chunk_flow": {"status": "Chunk Flow Warning", "detail": "Fallback"},
        "analysis": {"badge": "Ready", "detail": ""},
    }

    severity = DocumentStatusWindow._determine_telemetry_severity(telemetry_payload)

    assert severity == "warning"


def test_build_document_meta_fields_formats_values() -> None:
    payload: dict[str, Any] = {
        "label": "My Doc",
        "path": "C:/tmp/doc.txt",
        "length": 2048,
        "version_id": "v2",
        "content_hash": "abcd",
        "language": "en",
    }

    fields = DocumentStatusWindow._build_document_meta_fields(payload)

    assert fields["document"] == "My Doc"
    assert fields["path"].endswith("doc.txt")
    assert fields["length"] == "2,048 chars"
    assert fields["version"] == "v2 / abcd"
    assert fields["language"] == "en"
