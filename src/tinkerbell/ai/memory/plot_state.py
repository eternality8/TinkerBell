"""Lightweight character and plot scaffolding stored per document."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .cache_bus import (
    DocumentCacheBus,
    DocumentCacheEvent,
    DocumentChangedEvent,
    DocumentClosedEvent,
    get_document_cache_bus,
)
from ..services import telemetry as telemetry_service


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slugify(value: str) -> str:
    text = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", text)
    normalized = normalized.strip("-")
    return normalized or uuid.uuid4().hex[:8]


@dataclass(slots=True)
class PlotEntity:
    """Represents a recurring character, place, or concept."""

    entity_id: str
    name: str
    kind: str = "character"
    summary: str = ""
    salience: float = 0.0
    supporting_pointers: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    first_seen_at: datetime = field(default_factory=_utcnow)
    last_seen_at: datetime = field(default_factory=_utcnow)

    def touch(self, *, pointer_id: str | None, summary: str | None) -> None:
        self.last_seen_at = _utcnow()
        if summary:
            self.summary = summary[:360]
        if pointer_id and pointer_id not in self.supporting_pointers:
            self.supporting_pointers = (*self.supporting_pointers, pointer_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "kind": self.kind,
            "summary": self.summary,
            "salience": round(self.salience, 3),
            "supporting_pointers": list(self.supporting_pointers),
            "attributes": dict(self.attributes),
            "first_seen_at": self.first_seen_at.isoformat(),
            "last_seen_at": self.last_seen_at.isoformat(),
        }


@dataclass(slots=True)
class PlotBeat:
    """Single beat inside a plot arc."""

    beat_id: str
    arc_id: str
    summary: str
    pointer_id: str | None = None
    chunk_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "beat_id": self.beat_id,
            "summary": self.summary,
            "arc_id": self.arc_id,
            "pointer_id": self.pointer_id,
            "chunk_hash": self.chunk_hash,
            "created_at": self.created_at.isoformat(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class PlotArc:
    """A collection of beats describing a narrative arc."""

    arc_id: str
    name: str
    summary: str = ""
    beats: list[PlotBeat] = field(default_factory=list)

    def to_dict(self, *, max_beats: int | None = None) -> dict[str, Any]:
        beats = self.beats[: max_beats or len(self.beats)]
        return {
            "arc_id": self.arc_id,
            "name": self.name,
            "summary": self.summary or self.name,
            "beats": [beat.to_dict() for beat in beats],
        }


@dataclass(slots=True)
class DocumentPlotState:
    """Aggregated plot scaffolding for a single document/version."""

    document_id: str
    version_id: str | None = None
    updated_at: datetime = field(default_factory=_utcnow)
    entities: list[PlotEntity] = field(default_factory=list)
    arcs: list[PlotArc] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self, *, version_id: str | None = None) -> None:
        self.updated_at = _utcnow()
        if version_id is not None:
            self.version_id = version_id

    def upsert_entity(self, name: str, *, pointer_id: str | None, summary: str | None) -> PlotEntity:
        normalized = name.strip()
        slug = _slugify(normalized)
        for entity in self.entities:
            if entity.entity_id == slug:
                entity.touch(pointer_id=pointer_id, summary=summary)
                return entity
        entity = PlotEntity(entity_id=slug, name=normalized)
        entity.touch(pointer_id=pointer_id, summary=summary)
        self.entities.append(entity)
        return entity

    def arc(self, arc_id: str, *, name: str | None = None) -> PlotArc:
        for arc in self.arcs:
            if arc.arc_id == arc_id:
                return arc
        arc = PlotArc(arc_id=arc_id, name=name or arc_id.title())
        self.arcs.append(arc)
        return arc

    def add_beat(
        self,
        arc_id: str,
        *,
        summary: str,
        pointer_id: str | None,
        chunk_hash: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PlotBeat:
        arc = self.arc(arc_id, name="Primary arc")
        beat = PlotBeat(
            beat_id=uuid.uuid4().hex,
            arc_id=arc.arc_id,
            summary=summary,
            pointer_id=pointer_id,
            chunk_hash=chunk_hash,
            metadata=dict(metadata or {}),
        )
        arc.beats.append(beat)
        return beat

    def to_payload(
        self,
        *,
        max_entities: int | None = None,
        max_beats: int | None = None,
    ) -> dict[str, Any]:
        entity_slice = self.entities[: max_entities or len(self.entities)]
        arcs_payload = [arc.to_dict(max_beats=max_beats) for arc in self.arcs]
        return {
            "document_id": self.document_id,
            "version_id": self.version_id,
            "generated_at": self.updated_at.isoformat(),
            "entity_count": len(self.entities),
            "arc_count": len(self.arcs),
            "entities": [entity.to_dict() for entity in entity_slice],
            "arcs": arcs_payload,
            "metadata": dict(self.metadata),
        }


_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")
_ENTITY_STOPWORDS = {
    "The",
    "This",
    "That",
    "There",
    "Document",
    "Section",
    "Chapter",
    "Subagent",
    "Editor",
}


def _extract_entity_names(summary: str, *, limit: int = 4) -> list[str]:
    candidates: list[str] = []
    for match in _ENTITY_PATTERN.finditer(summary or ""):
        name = match.group(1).strip()
        if not name or len(name) < 3:
            continue
        if name in candidates:
            continue
        if name.split()[0] in _ENTITY_STOPWORDS:
            continue
        candidates.append(name)
        if len(candidates) >= limit:
            break
    return candidates


class DocumentPlotStateStore:
    """In-memory plot/entity cache keyed by document identifier."""

    def __init__(
        self,
        *,
        max_documents: int = 12,
        max_entities: int = 24,
        max_beats_per_arc: int = 24,
        bus: DocumentCacheBus | None = None,
    ) -> None:
        self._records: dict[str, DocumentPlotState] = {}
        self._chunk_index: dict[str, set[str]] = {}
        self._max_documents = max(1, max_documents)
        self._max_entities = max(4, max_entities)
        self._max_beats_per_arc = max(4, max_beats_per_arc)
        self._bus = bus or get_document_cache_bus()
        self._bus.subscribe(DocumentChangedEvent, self._handle_document_event, weak=True)
        self._bus.subscribe(DocumentClosedEvent, self._handle_document_event, weak=True)

    def ingest_chunk_summary(
        self,
        document_id: str,
        summary: str,
        *,
        version_id: str | None,
        chunk_hash: str | None,
        pointer_id: str | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> DocumentPlotState:
        state = self._records.get(document_id)
        if state is None:
            state = DocumentPlotState(document_id=document_id)
            self._records[document_id] = state
        state.touch(version_id=version_id)
        snippet = (summary or "").strip()
        if not snippet:
            return state

        chunk_registry = self._chunk_index.setdefault(document_id, set())
        if chunk_hash:
            chunk_registry.add(chunk_hash)

        arc_id = "primary"
        state.add_beat(
            arc_id,
            summary=snippet[:360],
            pointer_id=pointer_id,
            chunk_hash=chunk_hash,
            metadata=metadata,
        )
        for name in _extract_entity_names(snippet):
            state.upsert_entity(name, pointer_id=pointer_id, summary=snippet)

        stats = state.metadata.setdefault("stats", {})
        stats["ingested_chunks"] = int(stats.get("ingested_chunks", 0)) + 1
        if pointer_id:
            state.metadata["last_pointer_id"] = pointer_id

        self._enforce_limits(state)
        telemetry_service.emit(
            "plot_state.ingested",
            {
                "document_id": document_id,
                "entities": len(state.entities),
                "arcs": len(state.arcs),
                "chunk_hash": chunk_hash,
            },
        )
        return state

    def get(self, document_id: str) -> DocumentPlotState | None:
        return self._records.get(document_id)

    def clear(self, document_id: str | None = None) -> None:
        if document_id is None:
            self._records.clear()
            self._chunk_index.clear()
            return
        self._records.pop(document_id, None)
        self._chunk_index.pop(document_id, None)

    def _handle_document_event(self, event: DocumentCacheEvent) -> None:
        document_id = getattr(event, "document_id", None)
        if not document_id:
            return
        self.clear(document_id)

    def _enforce_limits(self, state: DocumentPlotState) -> None:
        state.entities.sort(key=lambda entity: entity.last_seen_at, reverse=True)
        state.entities[:] = state.entities[: self._max_entities]
        for arc in state.arcs:
            arc.beats = arc.beats[-self._max_beats_per_arc :]
        if len(self._records) > self._max_documents:
            oldest = sorted(self._records.values(), key=lambda item: item.updated_at)[0]
            self.clear(oldest.document_id)

    def snapshot(
        self,
        document_id: str,
        *,
        max_entities: int | None = None,
        max_beats: int | None = None,
    ) -> dict[str, Any] | None:
        state = self.get(document_id)
        if state is None:
            return None
        return state.to_payload(max_entities=max_entities, max_beats=max_beats)


__all__ = [
    "DocumentPlotState",
    "DocumentPlotStateStore",
    "PlotArc",
    "PlotBeat",
    "PlotEntity",
]
