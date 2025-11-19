"""Tool exposing cached character/entity concordance data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

from ...editor.document_model import DocumentState
from ..memory.character_map import CharacterMapStore

DocumentResolver = Callable[[str], DocumentState | None]
StoreResolver = Callable[[], CharacterMapStore | None]
FlagResolver = Callable[[], bool]


@dataclass(slots=True)
class CharacterMapTool:
    """Return cached entity mentions gathered from subagent chunk summaries."""

    character_map_resolver: StoreResolver
    active_document_provider: Callable[[], DocumentState | None] | None = None
    document_lookup: DocumentResolver | None = None
    feature_enabled: FlagResolver | None = None
    default_max_entities: int = 12
    default_max_mentions: int = 5

    summarizable: ClassVar[bool] = False

    def run(
        self,
        *,
        document_id: str | None = None,
        include_mentions: bool = True,
        include_stats: bool = False,
        max_entities: int | None = None,
        max_mentions: int | None = None,
    ) -> dict[str, Any]:
        if not self._feature_enabled():
            return {
                "status": "character_map_disabled",
                "reason": "feature_flag_disabled",
            }

        store = self._resolve_store()
        if store is None:
            return {
                "status": "character_map_unavailable",
                "reason": "character_map_store_uninitialized",
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
            max_mentions=max_mentions or self.default_max_mentions,
        )
        if snapshot is None:
            return {
                "status": "no_character_map",
                "document_id": target_id,
                "character_map_available": False,
            }

        payload: dict[str, Any] = {
            "status": "ok",
            "document_id": target_id,
            "character_map_available": True,
            "entity_count": snapshot.get("entity_count", 0),
            "generated_at": snapshot.get("generated_at"),
            "version_id": snapshot.get("version_id"),
        }
        if include_stats:
            payload["stats"] = snapshot.get("stats", {})
        payload["entities"] = self._maybe_strip_mentions(snapshot.get("entities", []), include_mentions)
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

    def _resolve_store(self) -> CharacterMapStore | None:
        try:
            return self.character_map_resolver()
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

    @staticmethod
    def _maybe_strip_mentions(entities: list[dict[str, Any]], include_mentions: bool) -> list[dict[str, Any]]:
        if include_mentions:
            return entities
        trimmed: list[dict[str, Any]] = []
        for entity in entities:
            cloned = dict(entity)
            cloned.pop("mentions", None)
            trimmed.append(cloned)
        return trimmed


__all__ = ["CharacterMapTool"]
