"""Tool that builds and updates character-wide edit plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Sequence

from ...editor.document_model import DocumentState
from ..memory.character_map import CharacterMapStore
from ..services import telemetry as telemetry_service

StoreResolver = Callable[[], CharacterMapStore | None]
FlagResolver = Callable[[], bool]


@dataclass(slots=True)
class CharacterEditPlannerTool:
    """Return scoped edit tasks for characters and record completion progress."""

    character_map_resolver: StoreResolver
    active_document_provider: Callable[[], DocumentState | None] | None = None
    feature_enabled: FlagResolver | None = None
    default_max_entities: int = 8
    default_max_mentions: int = 4
    default_max_tasks: int = 40

    summarizable: ClassVar[bool] = False

    def run(
        self,
        *,
        document_id: str | None = None,
        targets: Sequence[str] | None = None,
        completed_task_ids: Sequence[str] | None = None,
        notes: Mapping[str, str] | None = None,
        include_completed: bool = True,
        max_entities: int | None = None,
        max_mentions: int | None = None,
        max_tasks: int | None = None,
    ) -> dict[str, Any]:
        if not self._feature_enabled():
            return {
                "status": "character_planner_disabled",
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

        applied = 0
        if completed_task_ids:
            updates = self._build_updates(completed_task_ids, notes)
            applied = store.update_planner_progress(target_id, updates)

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

        tasks = self._build_tasks(snapshot, targets, include_completed, max_tasks=max_tasks)
        pending = sum(1 for task in tasks if task.get("status") != "completed")
        entity_scope = len({task.get("entity_id") for task in tasks if task.get("entity_id")})
        payload = {
            "status": "ok",
            "document_id": target_id,
            "character_map_available": True,
            "entity_scope": entity_scope,
            "tasks": tasks,
            "task_count": len(tasks),
            "pending_tasks": pending,
            "completed_tasks": len(tasks) - pending,
            "applied_updates": applied,
            "filters": {
                "targets": list(targets or []),
                "max_entities": max_entities or self.default_max_entities,
                "max_mentions": max_mentions or self.default_max_mentions,
                "max_tasks": max_tasks or self.default_max_tasks,
                "include_completed": include_completed,
            },
        }
        payload["next_task"] = self._next_task(tasks)
        telemetry_service.emit(
            "concordance.planner_run",
            {
                "document_id": target_id,
                "tasks_returned": len(tasks),
                "pending": pending,
                "updates_applied": applied,
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

    def _build_updates(
        self,
        completed_task_ids: Sequence[str],
        notes: Mapping[str, str] | None,
    ) -> list[dict[str, Any]]:
        note_lookup = {str(key): str(value) for key, value in (notes or {}).items()}
        updates: list[dict[str, Any]] = []
        for task_id in completed_task_ids:
            slug = str(task_id).strip()
            if not slug:
                continue
            updates.append({
                "task_id": slug,
                "status": "completed",
                "note": note_lookup.get(slug),
            })
        return updates

    def _build_tasks(
        self,
        snapshot: Mapping[str, Any],
        targets: Sequence[str] | None,
        include_completed: bool,
        *,
        max_tasks: int | None = None,
    ) -> list[dict[str, Any]]:
        max_allowed = max_tasks or self.default_max_tasks
        planner_progress = self._planner_progress_map(snapshot)
        filters = {self._normalize_target(value) for value in (targets or []) if value}
        raw_entities = snapshot.get("entities")
        if not isinstance(raw_entities, list):
            return []
        tasks: list[dict[str, Any]] = []
        for entity in raw_entities:
            entity_id = str(entity.get("entity_id") or "")
            canonical = str(entity.get("name") or "")
            if filters and not self._entity_matches(entity, filters, entity_id, canonical):
                continue
            mentions = entity.get("mentions")
            if not isinstance(mentions, list):
                continue
            for mention in mentions:
                task_id = str(mention.get("mention_id") or "")
                if not task_id:
                    continue
                status_entry = planner_progress.get(task_id)
                status = status_entry.get("status") if status_entry else "pending"
                if not include_completed and status == "completed":
                    continue
                task = {
                    "task_id": task_id,
                    "entity_id": entity_id,
                    "entity_name": canonical,
                    "chunk_id": mention.get("chunk_id"),
                    "pointer_id": mention.get("pointer_id"),
                    "char_range": mention.get("char_range"),
                    "snippet": mention.get("snippet"),
                    "status": status,
                    "note": status_entry.get("note") if status_entry else None,
                    "last_updated": status_entry.get("updated_at") if status_entry else None,
                    "aliases": entity.get("aliases", []),
                    "pronouns": entity.get("pronouns", []),
                }
                tasks.append(task)
                if len(tasks) >= max_allowed:
                    return tasks
        return tasks

    def _planner_progress_map(self, snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        planner = snapshot.get("planner_progress")
        if not isinstance(planner, Mapping):
            return {}
        raw_tasks = planner.get("tasks") if isinstance(planner.get("tasks"), list) else []
        if not isinstance(raw_tasks, list):
            return {}
        progress: dict[str, Mapping[str, Any]] = {}
        for entry in raw_tasks:
            if isinstance(entry, Mapping):
                task_id = str(entry.get("task_id") or "")
                if task_id:
                    progress[task_id] = entry
        return progress

    def _entity_matches(
        self,
        entity: Mapping[str, Any],
        filters: set[str],
        entity_id: str,
        canonical: str,
    ) -> bool:
        normalized_id = self._normalize_target(entity_id)
        if normalized_id and normalized_id in filters:
            return True
        normalized_name = self._normalize_target(canonical)
        if normalized_name and normalized_name in filters:
            return True
        alias_value = entity.get("aliases")
        aliases = alias_value if isinstance(alias_value, list) else []
        for alias in aliases:
            if self._normalize_target(alias) in filters:
                return True
        return False

    @staticmethod
    def _normalize_target(value: str | None) -> str:
        if not value:
            return ""
        return value.strip().lower()

    @staticmethod
    def _next_task(tasks: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        for task in tasks:
            if task.get("status") != "completed":
                return task
        return tasks[0] if tasks else None


__all__ = ["CharacterEditPlannerTool"]
