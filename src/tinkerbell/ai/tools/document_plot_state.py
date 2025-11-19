"""Plot-state tools exposing cached arc/entity scaffolding to the agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Sequence

from ...editor.document_model import DocumentState
from ..memory.plot_memory import PlotStateMemory
from ..memory.plot_state import DocumentPlotStateStore
from ..services import telemetry as telemetry_service

DocumentResolver = Callable[[str], DocumentState | None]
StoreResolver = Callable[[], DocumentPlotStateStore | None]
FlagResolver = Callable[[], bool]


@dataclass(slots=True)
class PlotOutlineTool:
    """Return structured plot scaffolding (entities, arcs, overrides, dependencies)."""

    plot_state_resolver: StoreResolver
    active_document_provider: Callable[[], DocumentState | None] | None = None
    document_lookup: DocumentResolver | None = None
    feature_enabled: FlagResolver | None = None
    default_max_entities: int = 10
    default_max_beats: int = 12
    default_max_overrides: int = 8
    default_max_dependencies: int = 8

    summarizable: ClassVar[bool] = False

    def run(
        self,
        *,
        document_id: str | None = None,
        include_entities: bool = True,
        include_arcs: bool = True,
        include_overrides: bool = True,
        include_dependencies: bool = True,
        max_entities: int | None = None,
        max_beats: int | None = None,
        max_overrides: int | None = None,
        max_dependencies: int | None = None,
    ) -> dict[str, Any]:
        if not self._feature_enabled():
            return {
                "status": "plot_state_disabled",
                "reason": "feature_flag_disabled",
            }

        store = self._resolve_store()
        if store is None:
            return {
                "status": "plot_state_unavailable",
                "reason": "plot_state_store_uninitialized",
            }

        target_id = self._resolve_document_id(document_id)
        if not target_id:
            return {
                "status": "no_document",
                "reason": "document_id_unavailable",
            }

        snapshot = self._snapshot(
            store,
            target_id,
            max_entities or self.default_max_entities,
            max_beats or self.default_max_beats,
            include_overrides,
            include_dependencies,
            max_overrides or self.default_max_overrides,
            max_dependencies or self.default_max_dependencies,
        )
        if snapshot is None:
            return {
                "status": "no_plot_state",
                "document_id": target_id,
                "plot_state_available": False,
            }

        payload = {
            "status": "ok",
            "document_id": target_id,
            "plot_state_available": True,
            "entity_count": snapshot.get("entity_count", 0),
            "arc_count": snapshot.get("arc_count", 0),
            "generated_at": snapshot.get("generated_at"),
            "version_id": snapshot.get("version_id"),
            "metadata": snapshot.get("metadata", {}),
            "version_metadata": snapshot.get("version_metadata", {}),
        }
        payload["entities"] = snapshot.get("entities", []) if include_entities else []
        payload["arcs"] = snapshot.get("arcs", []) if include_arcs else []
        payload["overrides"] = snapshot.get("overrides", []) if include_overrides else []
        payload["dependencies"] = snapshot.get("dependencies", []) if include_dependencies else []

        telemetry_service.emit(
            "plot_state.read",
            {
                "document_id": target_id,
                "entities": payload.get("entity_count", 0),
                "arcs": payload.get("arc_count", 0),
                "overrides": len(payload.get("overrides", ())),
                "dependencies": len(payload.get("dependencies", ())),
            },
        )
        return payload

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
        document = self._resolve_active_document()
        if document is not None:
            return document.document_id
        return None

    def _resolve_active_document(self) -> DocumentState | None:
        if callable(self.active_document_provider):
            try:
                return self.active_document_provider()
            except Exception:
                return None
        return None

    def _snapshot(
        self,
        store: DocumentPlotStateStore,
        document_id: str,
        max_entities: int,
        max_beats: int,
        include_overrides: bool,
        include_dependencies: bool,
        max_overrides: int,
        max_dependencies: int,
    ) -> dict[str, Any] | None:
        if isinstance(store, PlotStateMemory):
            return store.snapshot_enriched(
                document_id,
                max_entities=max_entities,
                max_beats=max_beats,
                max_overrides=max_overrides if include_overrides else 0,
                max_dependencies=max_dependencies if include_dependencies else 0,
            )
        snapshot = store.snapshot(document_id, max_entities=max_entities, max_beats=max_beats)
        if snapshot is None:
            return None
        if include_overrides:
            snapshot.setdefault("overrides", [])
        if include_dependencies:
            snapshot.setdefault("dependencies", [])
        snapshot.setdefault("version_metadata", {})
        return snapshot


DocumentPlotStateTool = PlotOutlineTool


__all__ = ["PlotOutlineTool", "DocumentPlotStateTool"]
