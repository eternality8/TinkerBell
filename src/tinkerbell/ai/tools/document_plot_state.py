"""Tool exposing cached plot/entity scaffolding to the agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from ...editor.document_model import DocumentState
from ..memory.plot_state import DocumentPlotStateStore

DocumentResolver = Callable[[str], DocumentState | None]
StoreResolver = Callable[[], DocumentPlotStateStore | None]
FlagResolver = Callable[[], bool]


@dataclass(slots=True)
class DocumentPlotStateTool:
    """Return lightweight entity + arc scaffolding assembled from subagent runs."""

    plot_state_resolver: StoreResolver
    active_document_provider: Callable[[], DocumentState | None] | None = None
    document_lookup: DocumentResolver | None = None
    feature_enabled: FlagResolver | None = None
    default_max_entities: int = 10
    default_max_beats: int = 12

    summarizable: ClassVar[bool] = False

    def run(
        self,
        *,
        document_id: str | None = None,
        include_entities: bool = True,
        include_arcs: bool = True,
        max_entities: int | None = None,
        max_beats: int | None = None,
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

        snapshot = store.snapshot(
            target_id,
            max_entities=max_entities or self.default_max_entities,
            max_beats=max_beats or self.default_max_beats,
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
        }
        if include_entities:
            payload["entities"] = snapshot.get("entities", [])
        else:
            payload["entities"] = []
        if include_arcs:
            payload["arcs"] = snapshot.get("arcs", [])
        else:
            payload["arcs"] = []
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
        if document is None:
            return None
        return document.document_id

    def _resolve_active_document(self) -> DocumentState | None:
        if callable(self.active_document_provider):
            try:
                return self.active_document_provider()
            except Exception:
                return None
        return None


__all__ = ["DocumentPlotStateTool"]
