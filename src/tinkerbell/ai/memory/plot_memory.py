"""Enhanced plot state memory with overrides and dependencies."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .plot_state import DocumentPlotState, DocumentPlotStateStore, PlotArc, PlotBeat, PlotEntity

LOGGER = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slugify(value: str) -> str:
    filtered = ''.join(ch.lower() if ch.isalnum() else '-' for ch in (value or '').strip())
    collapsed = '-'.join(part for part in filtered.split('-') if part)
    return collapsed or uuid.uuid4().hex[:8]


@dataclass(slots=True)
class PlotDependency:
    """Represents a dependency between two arcs (or beats)."""

    dependency_id: str
    source_arc_id: str
    target_arc_id: str
    kind: str = "depends_on"
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "dependency_id": self.dependency_id,
            "source_arc_id": self.source_arc_id,
            "target_arc_id": self.target_arc_id,
            "kind": self.kind,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class PlotOverride:
    """Represents a human-authored override for arcs or beats."""

    override_id: str
    summary: str
    arc_id: str | None = None
    beat_id: str | None = None
    author: str = "operator"
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def touch(self, *, summary: str | None = None, metadata: Mapping[str, Any] | None = None) -> None:
        if summary:
            self.summary = summary
        if metadata:
            self.metadata.update(metadata)
        self.updated_at = _utcnow()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "override_id": self.override_id,
            "summary": self.summary,
            "arc_id": self.arc_id,
            "beat_id": self.beat_id,
            "author": self.author,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": dict(self.metadata),
        }
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PlotOverride":
        return cls(
            override_id=str(payload.get("override_id") or uuid.uuid4().hex),
            summary=str(payload.get("summary") or ""),
            arc_id=payload.get("arc_id"),
            beat_id=payload.get("beat_id"),
            author=str(payload.get("author") or "operator"),
            metadata=dict(payload.get("metadata") or {}),
            created_at=_parse_timestamp(payload.get("created_at")),
            updated_at=_parse_timestamp(payload.get("updated_at")),
        )


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return _utcnow()


class PlotOverrideStore:
    """Persists plot overrides to disk so human edits survive restarts."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (Path.home() / ".tinkerbell" / "plot_overrides.json")
        self._lock = threading.Lock()

    def load_all(self) -> dict[str, list[dict[str, Any]]]:
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Unable to read plot overrides: %s", exc)
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            LOGGER.warning("Plot override file is invalid JSON: %s", exc)
            return {}
        if not isinstance(payload, Mapping):
            return {}
        result: dict[str, list[dict[str, Any]]] = {}
        for key, value in payload.items():
            if not isinstance(value, list):
                continue
            result[str(key)] = [item for item in value if isinstance(item, Mapping)]  # type: ignore[arg-type]
        return result

    def save_all(self, payload: Mapping[str, list[Mapping[str, Any]]]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True)
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._path.with_suffix(".tmp")
            tmp_path.write_text(body, encoding="utf-8")
            tmp_path.replace(self._path)


class PlotStateMemory(DocumentPlotStateStore):
    """Augments :class:`DocumentPlotStateStore` with dependencies and overrides."""

    def __init__(
        self,
        *,
        override_store: PlotOverrideStore | None = None,
        max_dependencies: int = 32,
        max_overrides: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._dependencies: dict[str, list[PlotDependency]] = {}
        self._overrides: dict[str, dict[str, PlotOverride]] = {}
        self._version_metadata: dict[str, dict[str, Any]] = {}
        self._max_dependencies = max(1, max_dependencies)
        self._max_overrides = max(1, max_overrides)
        self._override_store = override_store or PlotOverrideStore()
        self._load_persisted_overrides()

    # ------------------------------------------------------------------
    # Overrides & dependencies
    # ------------------------------------------------------------------
    def record_version_metadata(self, document_id: str, metadata: Mapping[str, Any]) -> None:
        entry = self._version_metadata.setdefault(document_id, {})
        entry.update({k: v for k, v in metadata.items() if v is not None})

    def list_overrides(self, document_id: str) -> list[PlotOverride]:
        return list(self._overrides.get(document_id, {}).values())

    def upsert_override(
        self,
        document_id: str,
        *,
        summary: str,
        override_id: str | None = None,
        arc_id: str | None = None,
        beat_id: str | None = None,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PlotOverride:
        override_map = self._overrides.setdefault(document_id, {})
        normalized_id = override_id or uuid.uuid4().hex
        override = override_map.get(normalized_id)
        if override is None:
            override = PlotOverride(
                override_id=normalized_id,
                summary=summary,
                arc_id=arc_id,
                beat_id=beat_id,
                author=author or "operator",
                metadata=dict(metadata or {}),
            )
            override_map[normalized_id] = override
        else:
            override.touch(summary=summary, metadata=metadata)
            if arc_id is not None:
                override.arc_id = arc_id
            if beat_id is not None:
                override.beat_id = beat_id
            if author:
                override.author = author
        self._enforce_override_limits(document_id)
        self._persist_overrides()
        return override

    def remove_override(self, document_id: str, override_id: str) -> bool:
        override_map = self._overrides.get(document_id)
        if not override_map or override_id not in override_map:
            return False
        override_map.pop(override_id, None)
        if not override_map:
            self._overrides.pop(document_id, None)
        self._persist_overrides()
        return True

    def list_dependencies(self, document_id: str) -> list[PlotDependency]:
        return list(self._dependencies.get(document_id, ()))

    def add_dependency(
        self,
        document_id: str,
        *,
        source_arc_id: str,
        target_arc_id: str,
        kind: str = "depends_on",
        summary: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> PlotDependency:
        dependency = PlotDependency(
            dependency_id=uuid.uuid4().hex,
            source_arc_id=source_arc_id,
            target_arc_id=target_arc_id,
            kind=kind,
            summary=(summary or "").strip(),
            metadata=dict(metadata or {}),
        )
        entries = self._dependencies.setdefault(document_id, [])
        entries.append(dependency)
        self._enforce_dependency_limits(document_id)
        return dependency

    def apply_manual_update(
        self,
        document_id: str,
        *,
        version_id: str | None = None,
        entities: Iterable[Mapping[str, Any]] | None = None,
        arcs: Iterable[Mapping[str, Any]] | None = None,
        overrides: Iterable[Mapping[str, Any]] | None = None,
        dependencies: Iterable[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self._records.get(document_id)
        if state is None:
            state = DocumentPlotState(document_id=document_id)
            self._records[document_id] = state
        state.touch(version_id=version_id)
        entity_updates = self._apply_entity_updates(state, entities or ())
        arc_updates = self._apply_arc_updates(state, arcs or ())
        override_updates = self._apply_override_updates(document_id, overrides or ())
        dependency_updates = self._apply_dependency_updates(document_id, dependencies or ())
        if metadata:
            self.record_version_metadata(document_id, metadata)
        self._enforce_limits(state)
        return {
            "document_id": document_id,
            "entity_updates": entity_updates,
            "arc_updates": arc_updates,
            "override_updates": override_updates,
            "dependency_updates": dependency_updates,
            "version_id": state.version_id,
        }

    def snapshot_enriched(
        self,
        document_id: str,
        *,
        max_entities: int | None = None,
        max_beats: int | None = None,
        max_overrides: int | None = None,
        max_dependencies: int | None = None,
    ) -> dict[str, Any] | None:
        snapshot = self.snapshot(document_id, max_entities=max_entities, max_beats=max_beats)
        if snapshot is None:
            return None
        overrides = self.list_overrides(document_id)
        dependencies = self.list_dependencies(document_id)
        if max_overrides is not None:
            overrides = overrides[: max(0, max_overrides)]
        if max_dependencies is not None:
            dependencies = dependencies[: max(0, max_dependencies)]
        snapshot["overrides"] = [item.to_dict() for item in overrides]
        snapshot["dependencies"] = [item.to_dict() for item in dependencies]
        snapshot["version_metadata"] = dict(self._version_metadata.get(document_id, {}))
        return snapshot

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def clear(self, document_id: str | None = None) -> None:  # type: ignore[override]
        super().clear(document_id)
        if document_id is None:
            self._dependencies.clear()
            self._overrides.clear()
            self._version_metadata.clear()
        else:
            self._dependencies.pop(document_id, None)
            self._overrides.pop(document_id, None)
            self._version_metadata.pop(document_id, None)
        self._persist_overrides()

    def _apply_entity_updates(self, state: DocumentPlotState, updates: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        for entry in updates:
            if not isinstance(entry, Mapping):
                continue
            raw_name = entry.get("name") or entry.get("entity_id")
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            entity_id = str(entry.get("entity_id") or _slugify(raw_name))
            entity = self._find_entity(state, entity_id)
            normalized_name = raw_name.strip()
            if entity is None:
                entity = PlotEntity(entity_id=entity_id, name=normalized_name)
                state.entities.append(entity)
            else:
                entity.name = normalized_name

            pointer_value = entry.get("pointer_id")
            pointer_id = pointer_value.strip() if isinstance(pointer_value, str) and pointer_value.strip() else None
            summary_value = entry.get("summary")
            summary_text = summary_value.strip() if isinstance(summary_value, str) else None
            entity.touch(pointer_id=pointer_id, summary=summary_text)

            kind_value = entry.get("kind") or entry.get("type")
            if isinstance(kind_value, str) and kind_value.strip():
                entity.kind = kind_value.strip()

            salience_value = entry.get("salience")
            if isinstance(salience_value, (int, float)):
                entity.salience = float(salience_value)

            supporting = entry.get("supporting_pointers")
            if isinstance(supporting, Iterable) and not isinstance(supporting, (str, bytes)):
                for pointer in supporting:
                    if isinstance(pointer, str) and pointer.strip():
                        entity.touch(pointer_id=pointer.strip(), summary=None)

            attribute_payload: dict[str, Any] = {}
            for key in ("attributes", "metadata"):
                value = entry.get(key)
                if isinstance(value, Mapping):
                    attribute_payload.update({str(k): v for k, v in value.items() if isinstance(k, str)})

            reserved = {
                "entity_id",
                "name",
                "summary",
                "kind",
                "type",
                "pointer_id",
                "supporting_pointers",
                "salience",
                "attributes",
                "metadata",
            }
            for key, value in entry.items():
                if not isinstance(key, str) or key in reserved:
                    continue
                attribute_payload.setdefault(key, value)

            if attribute_payload:
                entity.attributes.update(attribute_payload)

            count += 1
        return count

    def _apply_arc_updates(self, state: DocumentPlotState, updates: Iterable[Mapping[str, Any]]) -> int:
        applied = 0
        for entry in updates:
            name = str(entry.get("name") or entry.get("arc_id") or "Arc")
            arc_id = str(entry.get("arc_id") or _slugify(name))
            arc = state.arc(arc_id, name=name)
            summary = entry.get("summary")
            if isinstance(summary, str):
                arc.summary = summary.strip()
            beats = entry.get("beats")
            if isinstance(beats, Iterable):
                applied += self._apply_beat_updates(arc, beats)
        return applied

    def _apply_beat_updates(self, arc: PlotArc, beats: Iterable[Mapping[str, Any]]) -> int:
        updated = 0
        for entry in beats:
            summary = str(entry.get("summary") or "").strip()
            if not summary:
                continue
            beat_id = str(entry.get("beat_id") or uuid.uuid4().hex)
            pointer_id = entry.get("pointer_id")
            chunk_hash = entry.get("chunk_hash")
            metadata = dict(entry.get("metadata") or {})
            beat = self._find_beat(arc, beat_id)
            if beat is None:
                beat = PlotBeat(
                    beat_id=beat_id,
                    arc_id=arc.arc_id,
                    summary=summary,
                    pointer_id=pointer_id,
                    chunk_hash=chunk_hash,
                    metadata=metadata,
                )
                arc.beats.append(beat)
            else:
                beat.summary = summary
                beat.pointer_id = pointer_id
                beat.chunk_hash = chunk_hash
                if metadata:
                    beat.metadata.update(metadata)
            updated += 1
        return updated

    @staticmethod
    def _find_beat(arc: PlotArc, beat_id: str) -> PlotBeat | None:
        for beat in arc.beats:
            if beat.beat_id == beat_id:
                return beat
        return None

    @staticmethod
    def _find_entity(state: DocumentPlotState, entity_id: str) -> PlotEntity | None:
        for entity in state.entities:
            if entity.entity_id == entity_id:
                return entity
        return None

    def _apply_override_updates(self, document_id: str, overrides: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        for entry in overrides:
            summary = entry.get("summary")
            if not isinstance(summary, str) or not summary.strip():
                continue
            self.upsert_override(
                document_id,
                summary=summary.strip(),
                override_id=entry.get("override_id"),
                arc_id=entry.get("arc_id"),
                beat_id=entry.get("beat_id"),
                author=entry.get("author"),
                metadata=entry.get("metadata"),
            )
            count += 1
        return count

    def _apply_dependency_updates(self, document_id: str, dependencies: Iterable[Mapping[str, Any]]) -> int:
        count = 0
        if document_id not in self._dependencies:
            self._dependencies[document_id] = []
        for entry in dependencies:
            source = entry.get("source_arc_id")
            target = entry.get("target_arc_id")
            if not source or not target:
                continue
            self.add_dependency(
                document_id,
                source_arc_id=str(source),
                target_arc_id=str(target),
                kind=str(entry.get("kind") or "depends_on"),
                summary=(entry.get("summary") or "").strip(),
                metadata=entry.get("metadata"),
            )
            count += 1
        self._enforce_dependency_limits(document_id)
        return count

    def _enforce_dependency_limits(self, document_id: str) -> None:
        entries = self._dependencies.get(document_id)
        if not entries:
            return
        entries[:] = entries[-self._max_dependencies :]

    def _enforce_override_limits(self, document_id: str) -> None:
        entries = self._overrides.get(document_id)
        if not entries:
            return
        if len(entries) <= self._max_overrides:
            return
        sorted_overrides = sorted(entries.values(), key=lambda item: item.updated_at)
        for override in sorted_overrides[:-self._max_overrides]:
            entries.pop(override.override_id, None)

    def _persist_overrides(self) -> None:
        payload = {
            document_id: [override.to_dict() for override in overrides.values()]
            for document_id, overrides in self._overrides.items()
            if overrides
        }
        try:
            self._override_store.save_all(payload)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Unable to persist plot overrides: %s", exc)

    def _load_persisted_overrides(self) -> None:
        payload = self._override_store.load_all()
        for document_id, overrides in payload.items():
            override_map = self._overrides.setdefault(document_id, {})
            for entry in overrides:
                if not isinstance(entry, Mapping):
                    continue
                override = PlotOverride.from_payload(entry)
                override_map[override.override_id] = override


__all__ = [
    "PlotDependency",
    "PlotOverride",
    "PlotOverrideStore",
    "PlotStateMemory",
]
