"""Tool that applies manual plot-state updates from the manager agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Sequence

from ...editor.document_model import DocumentState
from ..memory.plot_memory import PlotStateMemory
from ..memory.plot_state import DocumentPlotStateStore
from ..services import telemetry as telemetry_service

StoreResolver = Callable[[], DocumentPlotStateStore | None]
DocumentResolver = Callable[[str], DocumentState | None]
FlagResolver = Callable[[], bool]


@dataclass(slots=True)
class PlotStateUpdateTool:
    """Persist agent-authored plot arc, beat, override, and dependency updates."""

    plot_state_resolver: StoreResolver
    active_document_provider: Callable[[], DocumentState | None] | None = None
    document_lookup: DocumentResolver | None = None
    feature_enabled: FlagResolver | None = None

    summarizable: ClassVar[bool] = False

    def run(
        self,
        *,
        document_id: str | None = None,
        version_id: str | None = None,
        arcs: Sequence[Mapping[str, Any]] | None = None,
        overrides: Sequence[Mapping[str, Any]] | None = None,
        dependencies: Sequence[Mapping[str, Any]] | None = None,
        remove_override_ids: Sequence[str] | None = None,
        note: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self._feature_enabled():
            return {
                "status": "plot_state_disabled",
                "reason": "feature_flag_disabled",
            }

        store = self._resolve_store()
        if not isinstance(store, PlotStateMemory):
            return {
                "status": "plot_state_unavailable",
                "reason": "plot_memory_upgrade_missing",
            }

        target_id = self._resolve_document_id(document_id)
        if not target_id:
            return {
                "status": "no_document",
                "reason": "document_id_unavailable",
            }

        arc_updates = self._coerce_sequence(arcs)
        override_updates = self._coerce_sequence(overrides)
        dependency_updates = self._coerce_sequence(dependencies)
        note_payload = {"last_update_note": note.strip()} if isinstance(note, str) and note.strip() else {}
        metadata_payload = dict(metadata or {})
        metadata_payload.update(note_payload)
        update_summary = store.apply_manual_update(
            target_id,
            version_id=version_id,
            arcs=arc_updates,
            overrides=override_updates,
            dependencies=dependency_updates,
            metadata=metadata_payload,
        )

        removed = 0
        if remove_override_ids:
            for override_id in remove_override_ids:
                if override_id and store.remove_override(target_id, override_id):
                    removed += 1

        telemetry_service.emit(
            "plot_state.write",
            {
                "document_id": target_id,
                "arc_updates": update_summary.get("arc_updates", 0),
                "override_updates": update_summary.get("override_updates", 0),
                "dependency_updates": update_summary.get("dependency_updates", 0),
                "removed_overrides": removed,
            },
        )

        return {
            "status": "ok",
            "document_id": target_id,
            "version_id": update_summary.get("version_id"),
            "arc_updates": update_summary.get("arc_updates", 0),
            "override_updates": update_summary.get("override_updates", 0),
            "dependency_updates": update_summary.get("dependency_updates", 0),
            "removed_overrides": removed,
            "note": note_payload.get("last_update_note"),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _feature_enabled(self) -> bool:
        resolver = self.feature_enabled
        if resolver is None:
            return True
        try:
            return bool(resolver())
        except Exception:
            return False

    def _resolve_store(self) -> DocumentPlotStateStore | None:
        try:
            return self.plot_state_resolver()
        except Exception:
            return None

    def _resolve_document_id(self, explicit: str | None) -> str | None:
        if explicit:
            return explicit
        if callable(self.active_document_provider):
            try:
                document = self.active_document_provider()
            except Exception:
                document = None
            if isinstance(document, DocumentState):
                return document.document_id
        return None

    @staticmethod
    def _coerce_sequence(items: Sequence[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
        if not items:
            return []
        sanitized: list[Mapping[str, Any]] = []
        for entry in items:
            if isinstance(entry, Mapping):
                sanitized.append(entry)
        return sanitized


__all__ = ["PlotStateUpdateTool"]
