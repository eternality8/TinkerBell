"""Aggregates payloads for the document status console."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..ai.analysis import AnalysisAdvice
from ..ai.memory.buffers import DocumentSummaryMemory, SummaryRecord
from ..ai.memory.character_map import CharacterMapStore
from ..ai.memory.plot_state import DocumentPlotStateStore
from ..editor.document_model import DocumentState
from ..editor.workspace import DocumentTab, DocumentWorkspace
from ..services.bridge_router import WorkspaceBridgeRouter
from .document_status import DocumentDescriptor, format_document_status_summary
from .telemetry_controller import TelemetryController

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _DocumentTargets:
    document: DocumentState | None
    tab: DocumentTab | None


class DocumentStatusService:
    """Collects metadata needed by :class:`DocumentStatusWindow`."""

    def __init__(
        self,
        *,
        workspace: DocumentWorkspace,
        bridge: WorkspaceBridgeRouter | None,
        telemetry: TelemetryController | None,
        controller_resolver: Callable[[], Any],
        outline_memory_resolver: Callable[[], DocumentSummaryMemory | None],
        plot_state_resolver: Callable[[], DocumentPlotStateStore | None],
        character_map_resolver: Callable[[], CharacterMapStore | None],
        snapshot_provider: Callable[[DocumentTab | None, DocumentState | None], Mapping[str, Any] | None] | None = None,
    ) -> None:
        self._workspace = workspace
        self._bridge = bridge
        self._telemetry = telemetry
        self._controller_resolver = controller_resolver
        self._outline_memory_resolver = outline_memory_resolver
        self._plot_state_resolver = plot_state_resolver
        self._character_map_resolver = character_map_resolver
        self._snapshot_provider = snapshot_provider or self._default_snapshot_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def list_document_descriptors(self) -> list[DocumentDescriptor]:
        descriptors: list[DocumentDescriptor] = []
        for tab in self._workspace.iter_tabs():
            document = tab.document()
            label = self._label_for(tab, document)
            descriptors.append(
                DocumentDescriptor(
                    document_id=document.document_id,
                    label=label,
                    tab_id=tab.id,
                )
            )
        return descriptors

    def build_status_payload(self, document_id: str | None = None) -> dict[str, Any]:
        targets = self._resolve_document_targets(document_id)
        if targets.document is None:
            raise ValueError("Document status requires an open document")
        document = targets.document
        descriptor_label = self._label_for(targets.tab, document)
        snapshot = self._snapshot_provider(targets.tab, document)
        chunk_payload = self._build_chunk_payload(snapshot)
        outline_payload = self._build_outline_payload(document.document_id)
        plot_payload = self._build_plot_payload(document.document_id)
        concordance_payload = self._build_concordance_payload(document.document_id)
        planner_payload = self._build_planner_payload(concordance_payload)
        telemetry_payload = self._build_telemetry_payload(document.document_id, descriptor_label)
        badge_payload = self._build_badge(chunk_payload, outline_payload, planner_payload, telemetry_payload)
        document_info = self._serialize_document_info(document, targets.tab, descriptor_label)

        payload = {
            "document": document_info,
            "chunks": chunk_payload,
            "outline": outline_payload,
            "plot": plot_payload,
            "concordance": concordance_payload,
            "planner": planner_payload,
            "telemetry": telemetry_payload,
            "badge": badge_payload,
        }
        payload["summary"] = format_document_status_summary(payload)
        return payload

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_document_targets(self, document_id: str | None) -> _DocumentTargets:
        if document_id:
            for tab in self._workspace.iter_tabs():
                document = tab.document()
                if document.document_id == document_id:
                    return _DocumentTargets(document=document, tab=tab)
            document = self._workspace.find_document_by_id(document_id)
            return _DocumentTargets(document=document, tab=None)
        tab = self._workspace.active_tab
        document = tab.document() if tab is not None else None
        return _DocumentTargets(document=document, tab=tab)

    def _default_snapshot_provider(
        self,
        tab: DocumentTab | None,
        document: DocumentState | None,
    ) -> Mapping[str, Any] | None:
        if document is None:
            return None
        controller = self._controller_resolver()
        getter = getattr(controller, "get_latest_snapshot", None)
        if callable(getter):
            cached = getter(document.document_id)
            if isinstance(cached, Mapping):
                return dict(cached)
        if tab is None or self._bridge is None:
            return None
        try:
            return self._bridge.generate_snapshot(
                tab_id=tab.id,
                include_open_documents=False,
                include_text=False,
                window="selection",
            )
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Document status snapshot generation failed", exc_info=True)
            return None

    def _serialize_document_info(
        self,
        document: DocumentState,
        tab: DocumentTab | None,
        label: str,
    ) -> dict[str, Any]:
        selection = document.selection.as_tuple()
        path = document.metadata.path
        selection_payload = {"start": selection[0], "end": selection[1]}
        version = document.version_info()
        info = {
            "document_id": document.document_id,
            "label": label,
            "language": document.metadata.language,
            "dirty": document.dirty,
            "length": len(document.text),
            "selection": selection_payload,
            "version_id": version.version_id,
            "content_hash": version.content_hash,
        }
        if path is not None:
            info["path"] = str(Path(path))
        if tab is not None:
            info["tab_id"] = tab.id
            info["tab_title"] = tab.title
        return info

    def _build_chunk_payload(self, snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if not isinstance(snapshot, Mapping):
            return payload
        manifest = snapshot.get("chunk_manifest") if isinstance(snapshot.get("chunk_manifest"), Mapping) else None
        window_payload = snapshot.get("window") if isinstance(snapshot.get("window"), Mapping) else None
        selection_payload = snapshot.get("selection") if isinstance(snapshot.get("selection"), Mapping) else None
        payload["chunk_manifest"] = manifest
        payload["window"] = window_payload
        payload["selection"] = selection_payload
        if manifest and manifest.get("chunk_profile"):
            payload["chunk_profile"] = manifest.get("chunk_profile")
        elif snapshot.get("chunk_profile"):
            payload["chunk_profile"] = snapshot.get("chunk_profile")
        payload["document_version"] = {
            "version": snapshot.get("version"),
            "version_id": snapshot.get("version_id"),
            "content_hash": snapshot.get("content_hash"),
        }
        if manifest:
            chunks = manifest.get("chunks")
            if isinstance(chunks, Sequence):
                payload.setdefault("stats", {})["chunk_count"] = len(chunks)
            if manifest.get("generated_at"):
                payload.setdefault("stats", {})["generated_at"] = manifest.get("generated_at")
        return payload

    def _build_outline_payload(self, document_id: str | None) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        memory = self._outline_memory_resolver()
        if not document_id or memory is None:
            return payload
        record = memory.get(document_id)
        if record is None:
            payload["status"] = "pending"
            return payload
        payload.update(self._serialize_outline_record(record))
        return payload

    def _serialize_outline_record(self, record: SummaryRecord) -> dict[str, Any]:
        node_count = len(record.nodes) if record.nodes else 0
        updated_at = record.updated_at.isoformat()
        status = "ok" if node_count else "pending"
        return {
            "status": status,
            "node_count": node_count,
            "outline_hash": record.outline_hash,
            "version_id": record.version_id,
            "updated_at": updated_at,
            "summary": record.summary,
            "highlights": list(record.highlights),
        }

    def _build_plot_payload(self, document_id: str | None) -> dict[str, Any] | None:
        if not document_id:
            return None
        store = self._plot_state_resolver()
        if store is None:
            return None
        snapshot = self._build_plot_snapshot(store, document_id)
        return dict(snapshot) if isinstance(snapshot, Mapping) else None

    @staticmethod
    def _build_plot_snapshot(store: Any, document_id: str) -> Mapping[str, Any] | None:
        max_entities = 12
        max_beats = 24
        snapshot = None
        enriched = getattr(store, "snapshot_enriched", None)
        if callable(enriched):
            try:
                snapshot = enriched(
                    document_id,
                    max_entities=max_entities,
                    max_beats=max_beats,
                    max_overrides=10,
                    max_dependencies=12,
                )
            except TypeError:
                snapshot = None
        if isinstance(snapshot, Mapping):
            return snapshot
        basic = getattr(store, "snapshot", None)
        if callable(basic):
            candidate = basic(document_id, max_entities=max_entities, max_beats=max_beats)
            if isinstance(candidate, Mapping):
                return candidate
        return None

    def _build_concordance_payload(self, document_id: str | None) -> dict[str, Any] | None:
        if not document_id:
            return None
        store = self._character_map_resolver()
        if store is None:
            return None
        snapshot = store.snapshot(document_id)
        return snapshot if isinstance(snapshot, Mapping) else None

    def _build_planner_payload(self, concordance_payload: Mapping[str, Any] | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"pending": 0, "completed": 0, "tasks": []}
        if not isinstance(concordance_payload, Mapping):
            return payload
        planner = concordance_payload.get("planner_progress")
        if not isinstance(planner, Mapping):
            return payload
        pending = int(planner.get("pending", 0) or 0)
        completed = int(planner.get("completed", 0) or 0)
        tasks = planner.get("tasks")
        payload.update(
            {
                "pending": max(0, pending),
                "completed": max(0, completed),
                "tasks": list(tasks) if isinstance(tasks, Sequence) else [],
            }
        )
        return payload

    def _build_telemetry_payload(
        self,
        document_id: str | None,
        document_label: str,
    ) -> dict[str, Any]:
        telemetry: dict[str, Any] = {}
        controller = self._controller_resolver()
        if self._telemetry is not None:
            chunk_flow = self._telemetry.chunk_flow_snapshot()
            if chunk_flow:
                telemetry["chunk_flow"] = chunk_flow
        if controller is None or not document_id:
            return telemetry
        advice = controller.get_latest_analysis_advice(document_id)
        if advice is None:
            return telemetry
        overview = self._describe_analysis(advice, document_label)
        if overview:
            telemetry["analysis"] = overview
        return telemetry

    def _describe_analysis(self, advice: AnalysisAdvice, document_label: str) -> Mapping[str, str] | None:
        if self._telemetry is not None:
            return self._telemetry.describe_analysis_indicator(advice, document_label=document_label)
        status = "Ready"
        if advice.must_refresh_outline:
            status = "Outline refresh required"
        detail_lines = [
            f"Chunk profile: {advice.chunk_profile}",
            f"Required tools: {', '.join(advice.required_tools) if advice.required_tools else 'none'}",
        ]
        if advice.warnings:
            detail_lines.append(f"Warnings: {len(advice.warnings)}")
        return {
            "status": status,
            "badge": f"Preflight: {status}",
            "detail": "\n".join(detail_lines),
        }

    def _build_badge(
        self,
        chunks: Mapping[str, Any],
        outline: Mapping[str, Any],
        planner: Mapping[str, Any],
        telemetry: Mapping[str, Any],
    ) -> dict[str, Any]:
        chunk_flow = telemetry.get("chunk_flow") if isinstance(telemetry, Mapping) else None
        if isinstance(chunk_flow, Mapping) and chunk_flow.get("status"):
            return {
                "status": str(chunk_flow.get("status")),
                "detail": str(chunk_flow.get("detail") or ""),
                "severity": "warning",
            }
        pending_tasks = int(planner.get("pending", 0)) if isinstance(planner, Mapping) else 0
        if pending_tasks:
            completed = int(planner.get("completed", 0)) if isinstance(planner, Mapping) else 0
            detail = f"Completed {completed}" if completed else ""
            return {
                "status": f"Planner pending ({pending_tasks})",
                "detail": detail,
                "severity": "info",
            }
        outline_status = (outline or {}).get("status") if isinstance(outline, Mapping) else None
        if outline_status and outline_status != "ok":
            return {
                "status": f"Outline: {outline_status}",
                "detail": str(outline.get("updated_at") or ""),
                "severity": "info",
            }
        chunk_profile = chunks.get("chunk_profile") if isinstance(chunks, Mapping) else None
        detail = f"Chunk profile {chunk_profile}" if chunk_profile else ""
        return {
            "status": "Doc Ready",
            "detail": detail,
            "severity": "normal",
        }

    def _label_for(self, tab: DocumentTab | None, document: DocumentState | None) -> str:
        if tab is not None and tab.title:
            return tab.title.strip()
        if document is not None:
            path = document.metadata.path
            if isinstance(path, Path):
                name = path.name or str(path)
                if name:
                    return name
            if document.metadata.path:
                return str(document.metadata.path)
            if document.document_id:
                return document.document_id
        return "document"


__all__ = ["DocumentStatusService"]
