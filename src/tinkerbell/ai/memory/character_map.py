"""Character/entity concordance cache built from chunk summaries."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .cache_bus import (
    DocumentCacheBus,
    DocumentCacheEvent,
    DocumentChangedEvent,
    DocumentClosedEvent,
    get_document_cache_bus,
)
from .plot_state import _extract_entity_names as _extract_names, _slugify as _slugify_name
from ..services import telemetry as telemetry_service


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


_PRONOUN_PATTERN = re.compile(r"\b(she|her|hers|he|him|his|they|them|theirs)\b", re.IGNORECASE)


@dataclass(slots=True)
class CharacterMention:
    """Single mention/reference of an entity inside a chunk or summary."""

    mention_id: str
    snippet: str
    chunk_id: str | None
    pointer_id: str | None
    chunk_hash: str | None
    char_range: tuple[int, int] | None
    version_id: str | None
    seen_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "mention_id": self.mention_id,
            "snippet": self.snippet,
            "chunk_id": self.chunk_id,
            "pointer_id": self.pointer_id,
            "chunk_hash": self.chunk_hash,
            "char_range": self.char_range,
            "version_id": self.version_id,
            "seen_at": self.seen_at.isoformat(),
        }
        return {key: value for key, value in payload.items() if value not in (None, "")}


@dataclass(slots=True)
class CharacterRecord:
    """Canonical record describing an entity with alias + mention tracking."""

    entity_id: str
    canonical_name: str
    aliases: set[str] = field(default_factory=set)
    pronouns: set[str] = field(default_factory=set)
    mentions: list[CharacterMention] = field(default_factory=list)
    first_seen_at: datetime = field(default_factory=_utcnow)
    last_seen_at: datetime = field(default_factory=_utcnow)

    def register_mention(
        self,
        snippet: str,
        *,
        chunk_id: str | None,
        pointer_id: str | None,
        chunk_hash: str | None,
        char_range: tuple[int, int] | None,
        version_id: str | None,
        pronoun_hints: Sequence[str] | None,
    ) -> CharacterMention:
        mention = CharacterMention(
            mention_id=uuid.uuid4().hex,
            snippet=snippet,
            chunk_id=chunk_id,
            pointer_id=pointer_id,
            chunk_hash=chunk_hash,
            char_range=char_range,
            version_id=version_id,
        )
        self.mentions.append(mention)
        self.last_seen_at = mention.seen_at
        if pronoun_hints:
            for pronoun in pronoun_hints:
                normalized = pronoun.strip().lower()
                if normalized:
                    self.pronouns.add(normalized)
        return mention

    def add_alias(self, alias: str) -> None:
        text = alias.strip()
        if text and text.lower() != self.canonical_name.lower():
            self.aliases.add(text)

    def to_payload(self, *, max_mentions: int | None = None) -> dict[str, Any]:
        mentions = self.mentions[-max_mentions:] if max_mentions else list(self.mentions)
        return {
            "entity_id": self.entity_id,
            "name": self.canonical_name,
            "aliases": sorted(self.aliases),
            "pronouns": sorted(self.pronouns),
            "mention_count": len(self.mentions),
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
            "mentions": [mention.to_dict() for mention in mentions],
        }


@dataclass(slots=True)
class PlannerTaskProgress:
    """Tracks completion metadata for planner tasks keyed by mention id."""

    task_id: str
    status: str = "pending"
    note: str | None = None
    updated_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "status": self.status,
            "note": self.note,
            "updated_at": self.updated_at.isoformat(),
        }
        return {key: value for key, value in payload.items() if value not in (None, "")}


@dataclass(slots=True)
class CharacterMapDocument:
    """Document-scoped concordance cache."""

    document_id: str
    version_id: str | None = None
    updated_at: datetime = field(default_factory=_utcnow)
    entities: dict[str, CharacterRecord] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    planner_progress: dict[str, PlannerTaskProgress] = field(default_factory=dict)

    def touch(self, version_id: str | None = None) -> None:
        self.updated_at = _utcnow()
        if version_id is not None:
            self.version_id = version_id


class CharacterMapStore:
    """In-memory concordance index keyed per document identifier."""

    def __init__(
        self,
        *,
        max_documents: int = 12,
        max_entities: int = 32,
        max_mentions_per_entity: int = 12,
        bus: DocumentCacheBus | None = None,
    ) -> None:
        self._documents: dict[str, CharacterMapDocument] = {}
        self._max_documents = max(1, max_documents)
        self._max_entities = max(1, max_entities)
        self._max_mentions = max(1, max_mentions_per_entity)
        self._bus = bus or get_document_cache_bus()
        self._bus.subscribe(DocumentChangedEvent, self._handle_event, weak=True)
        self._bus.subscribe(DocumentClosedEvent, self._handle_event, weak=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_summary(
        self,
        document_id: str,
        summary: str,
        *,
        version_id: str | None,
        chunk_id: str | None,
        pointer_id: str | None,
        chunk_hash: str | None,
        char_range: tuple[int, int] | None = None,
    ) -> CharacterMapDocument:
        if not document_id:
            raise ValueError("document_id is required")
        state = self._documents.get(document_id)
        if state is None:
            state = CharacterMapDocument(document_id=document_id)
            self._documents[document_id] = state
        state.touch(version_id=version_id)
        snippet = (summary or "").strip()
        if not snippet:
            return state

        names = _extract_names(snippet)
        pronouns = self._extract_pronouns(snippet)
        ingested_entities = 0
        for name in names:
            entity = self._upsert_entity(state, name)
            entity.register_mention(
                snippet[:360],
                chunk_id=chunk_id,
                pointer_id=pointer_id,
                chunk_hash=chunk_hash,
                char_range=char_range,
                version_id=version_id,
                pronoun_hints=pronouns,
            )
            ingested_entities += 1
        if ingested_entities:
            stats = state.stats.setdefault("ingested_chunks", 0)
            state.stats["ingested_chunks"] = stats + 1
            self._enforce_limits(state)
            telemetry_service.emit(
                "concordance.ingested",
                {
                    "document_id": document_id,
                    "entities": len(state.entities),
                    "mentions": sum(len(record.mentions) for record in state.entities.values()),
                    "chunk_hash": chunk_hash,
                },
            )
        return state

    def snapshot(
        self,
        document_id: str,
        *,
        max_entities: int | None = None,
        max_mentions: int | None = None,
    ) -> dict[str, Any] | None:
        state = self._documents.get(document_id)
        if state is None:
            return None
        entity_records = list(state.entities.values())
        slice_size = max_entities or len(entity_records)
        entities = [
            record.to_payload(max_mentions=max_mentions or self._max_mentions)
            for record in entity_records[:slice_size]
        ]
        planner_entries = [progress.to_dict() for progress in state.planner_progress.values()]
        completed = sum(1 for entry in planner_entries if entry.get("status") == "completed")
        return {
            "document_id": state.document_id,
            "version_id": state.version_id,
            "generated_at": state.updated_at.isoformat(),
            "entity_count": len(state.entities),
            "entities": entities,
            "stats": dict(state.stats),
            "planner_progress": {
                "tasks": planner_entries,
                "completed": completed,
                "pending": max(0, len(planner_entries) - completed),
            },
        }

    def clear(self, document_id: str | None = None) -> None:
        if document_id is None:
            self._documents.clear()
            return
        self._documents.pop(document_id, None)

    def update_planner_progress(
        self,
        document_id: str,
        updates: Sequence[Mapping[str, Any]],
    ) -> int:
        if not updates:
            return 0
        state = self._documents.get(document_id)
        if state is None:
            return 0
        applied = 0
        for entry in updates:
            task_id = str(entry.get("task_id") or "").strip()
            if not task_id:
                continue
            status = str(entry.get("status") or "completed").strip().lower() or "completed"
            note_value = entry.get("note")
            note = str(note_value) if note_value not in (None, "") else None
            progress = state.planner_progress.get(task_id)
            if progress is None:
                progress = PlannerTaskProgress(task_id=task_id)
            progress.status = status
            progress.note = note
            progress.updated_at = _utcnow()
            state.planner_progress[task_id] = progress
            applied += 1
        if applied:
            telemetry_service.emit(
                "concordance.planner_updated",
                {
                    "document_id": document_id,
                    "tasks_updated": applied,
                    "total_tracked": len(state.planner_progress),
                },
            )
        return applied

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _upsert_entity(self, state: CharacterMapDocument, name: str) -> CharacterRecord:
        normalized = name.strip()
        slug = _slugify_name(normalized)
        record = state.entities.get(slug)
        if record is None:
            record = CharacterRecord(entity_id=slug, canonical_name=normalized)
            state.entities[slug] = record
        elif normalized.lower() != record.canonical_name.lower():
            record.add_alias(normalized)
        return record

    def _enforce_limits(self, state: CharacterMapDocument) -> None:
        # Trim entity count based on recency.
        entities = list(state.entities.values())
        entities.sort(key=lambda record: record.last_seen_at, reverse=True)
        trimmed = entities[: self._max_entities]
        state.entities = {record.entity_id: record for record in trimmed}
        # Trim per-entity mentions to the configured max.
        for record in state.entities.values():
            if len(record.mentions) > self._max_mentions:
                record.mentions = record.mentions[-self._max_mentions :]
        # Enforce global document limit by removing oldest entries.
        if len(self._documents) > self._max_documents:
            oldest = sorted(self._documents.values(), key=lambda doc: doc.updated_at)[0]
            self.clear(oldest.document_id)

    def _extract_pronouns(self, snippet: str) -> list[str]:
        matches = _PRONOUN_PATTERN.findall(snippet)
        return [match.lower() for match in matches]

    def _handle_event(self, event: DocumentCacheEvent) -> None:
        document_id = getattr(event, "document_id", None)
        if document_id:
            self.clear(document_id)


__all__ = [
    "CharacterMapStore",
    "CharacterMapDocument",
    "CharacterRecord",
    "CharacterMention",
    "PlannerTaskProgress",
]
