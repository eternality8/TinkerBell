"""Tests for the DocumentStatusService aggregator."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast

from tinkerbell.ai.memory.buffers import DocumentSummaryMemory, SummaryRecord
from tinkerbell.ai.memory.character_map import CharacterMapStore
from tinkerbell.ai.memory.plot_state import DocumentPlotStateStore
from tinkerbell.editor.workspace import DocumentWorkspace
from tinkerbell.ui.document_status_service import DocumentStatusService
from tinkerbell.ui.telemetry_controller import TelemetryController


class _StubDocument:
    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self.metadata = SimpleNamespace(path="C:/repo/doc.txt", language="en")
        self.dirty = False
        self.text = "alpha beta gamma"

    def version_info(self) -> SimpleNamespace:
        return SimpleNamespace(version_id="v1", content_hash="hash")


class _StubTab:
    def __init__(self, tab_id: str, title: str, document: _StubDocument) -> None:
        self.id = tab_id
        self.title = title
        self._document = document
        self.untitled_index: int | None = None

    def document(self) -> _StubDocument:
        return self._document


class _StubWorkspace:
    def __init__(self, tabs: list[_StubTab]) -> None:
        self._tabs = tabs
        self.active_tab = tabs[0]
        self.active_tab_id = tabs[0].id

    def iter_tabs(self):
        for tab in self._tabs:
            yield tab

    def find_document_by_id(self, document_id: str) -> _StubDocument | None:
        for tab in self._tabs:
            doc = tab.document()
            if doc.document_id == document_id:
                return doc
        return None

    def get_tab(self, tab_id: str) -> _StubTab:
        for tab in self._tabs:
            if tab.id == tab_id:
                return tab
        raise KeyError(tab_id)


class _StubOutlineMemory:
    def __init__(self, record: SummaryRecord) -> None:
        self._record = record

    def get(self, document_id: str) -> SummaryRecord | None:
        if document_id == self._record.document_id:
            return self._record
        return None


class _StubPlotStore:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def snapshot_enriched(self, document_id: str, **_: Any) -> dict[str, Any] | None:
        if document_id == self._payload.get("document_id"):
            return dict(self._payload)
        return None

    def snapshot(self, document_id: str, **kwargs: Any) -> dict[str, Any] | None:
        return self.snapshot_enriched(document_id, **kwargs)


class _StubCharacterMap:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def snapshot(self, document_id: str) -> dict[str, Any] | None:
        if document_id == self._payload.get("document_id"):
            return dict(self._payload)
        return None


class _StubTelemetry:
    def __init__(self, chunk_flow: dict[str, Any] | None) -> None:
        self._chunk_flow = chunk_flow

    def chunk_flow_snapshot(self) -> dict[str, Any] | None:
        return self._chunk_flow

    def describe_analysis_indicator(self, advice: Any, *, document_label: str) -> dict[str, str]:
        detail = f"Chunk profile {advice.chunk_profile}"
        return {
            "status": "Ready",
            "badge": f"Preflight: {document_label}",
            "detail": detail,
        }


class _StubAdvice:
    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self.chunk_profile = "precise"
        self.required_tools = ("snapshot",)
        self.optional_tools: tuple[str, ...] = ()
        self.cache_state = "fresh"
        self.must_refresh_outline = False
        self.warnings: list[Any] = []


class _StubController:
    def __init__(self, advice: _StubAdvice) -> None:
        self._advice = advice

    def get_latest_analysis_advice(self, document_id: str) -> _StubAdvice | None:
        if document_id == self._advice.document_id:
            return self._advice
        return None


def _build_service(*, chunk_flow: dict[str, Any] | None, planner_pending: int) -> DocumentStatusService:
    document = _StubDocument("doc-1")
    tab = _StubTab("tab-1", "Document 1", document)
    workspace = _StubWorkspace([tab])
    outline_record = SummaryRecord(
        document_id=document.document_id,
        summary="Outline synced",
        highlights=["Note"],
        updated_at=datetime(2025, 11, 19, 12, 0, 0),
        version_id=3,
        outline_hash="deadbeef",
        nodes=[],
        content_hash="hash",
    )
    outline_memory = _StubOutlineMemory(outline_record)

    plot_payload = {
        "document_id": document.document_id,
        "entity_count": 2,
        "arc_count": 1,
        "arcs": [{"beats": [{"summary": "Beat"}]}],
        "overrides": [],
    }
    plot_store = _StubPlotStore(plot_payload)

    concordance_payload = {
        "document_id": document.document_id,
        "entity_count": 3,
        "planner_progress": {
            "pending": planner_pending,
            "completed": 2,
            "tasks": [{"task_id": "t1", "status": "pending", "note": "Outline Act 2"}],
        },
    }
    character_map = _StubCharacterMap(concordance_payload)

    telemetry = _StubTelemetry(chunk_flow)
    advice = _StubAdvice(document.document_id)
    controller = _StubController(advice)

    snapshot_payload = {
        "chunk_manifest": {
            "chunk_profile": "precise",
            "generated_at": "2025-11-19T12:00:00Z",
            "chunks": [{"id": "a"}, {"id": "b"}],
        },
        "window": {"start": 0, "end": 256},
        "text_range": {"start": 0, "end": 256},
        "version": "v1",
        "version_id": "v1",
        "content_hash": "hash",
    }

    def _snapshot_provider(tab_obj: Any | None, doc_obj: Any | None) -> dict[str, Any]:
        assert doc_obj is document
        return dict(snapshot_payload)

    service = DocumentStatusService(
        workspace=cast(DocumentWorkspace, workspace),
        bridge=None,
        telemetry=cast(TelemetryController | None, telemetry),
        controller_resolver=lambda: controller,
        outline_memory_resolver=lambda: cast(DocumentSummaryMemory | None, outline_memory),
        plot_state_resolver=lambda: cast(DocumentPlotStateStore | None, plot_store),
        character_map_resolver=lambda: cast(CharacterMapStore | None, character_map),
        snapshot_provider=_snapshot_provider,
    )
    return service


def test_document_status_service_builds_payload_with_sections() -> None:
    service = _build_service(chunk_flow={"status": "Chunk Flow Warning", "detail": "Fallback"}, planner_pending=1)

    payload = service.build_status_payload("doc-1")

    assert payload["badge"]["severity"] == "warning"
    assert payload["chunks"]["stats"]["chunk_count"] == 2
    assert payload["outline"]["summary"] == "Outline synced"
    assert payload["planner"]["pending"] == 1
    assert payload["telemetry"]["analysis"]["badge"].startswith("Preflight")


def test_document_status_service_badge_falls_back_to_planner_when_no_chunk_flow() -> None:
    service = _build_service(chunk_flow=None, planner_pending=2)

    payload = service.build_status_payload("doc-1")

    assert payload["badge"]["severity"] == "info"
    assert payload["badge"]["status"].startswith("Planner pending")
