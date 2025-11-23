"""Document chunk manifest index with cache-bus invalidation."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Mapping, Sequence

from .cache_bus import (
    DocumentCacheBus,
    DocumentChangedEvent,
    DocumentClosedEvent,
    get_document_cache_bus,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkIndexEntry:
    """Normalized record describing a chunk that can be rehydrated later."""

    document_id: str
    chunk_id: str
    start: int
    end: int
    length: int
    chunk_hash: str | None
    span_overlap: bool
    outline_pointer_id: str | None
    chunk_profile: str
    cache_key: str
    version: str | None
    content_hash: str | None
    chunk_chars: int | None
    chunk_overlap: int | None
    window: dict[str, Any]
    manifest_generated_at: float | None
    created_at: float = field(default_factory=time.time)
    position: int = 0


@dataclass(slots=True)
class ChunkManifestRecord:
    """Metadata persisted for each manifest so iterators can stay ordered."""

    document_id: str
    cache_key: str
    version: str | None
    content_hash: str | None
    chunk_profile: str
    chunk_ids: tuple[str, ...]
    generated_at: float | None
    window: dict[str, Any]


class ChunkIndex:
    """Index of chunk manifests keyed by document + cache key."""

    def __init__(
        self,
        *,
        bus: DocumentCacheBus | None = None,
        manifest_limit: int = 4,
    ) -> None:
        self._bus = bus or get_document_cache_bus()
        self._manifest_limit = max(1, int(manifest_limit or 1))
        self._lock = RLock()
        self._chunks: dict[str, dict[str, ChunkIndexEntry]] = {}
        self._manifests: dict[str, "OrderedDict[str, ChunkManifestRecord]"] = {}
        self._cache_index: dict[str, ChunkManifestRecord] = {}
        self._bus.subscribe(DocumentChangedEvent, self._handle_cache_event, weak=True)  # type: ignore[arg-type]
        self._bus.subscribe(DocumentClosedEvent, self._handle_cache_event, weak=True)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest_manifest(self, manifest: Mapping[str, Any] | None) -> list[ChunkIndexEntry]:
        """Store chunks from ``manifest`` and return the registered entries."""

        if not manifest:
            return []
        document_id = _string_or_none(manifest.get("document_id"))
        cache_key = _string_or_none(manifest.get("cache_key"))
        chunk_profile = (manifest.get("chunk_profile") or "auto").strip() or "auto"
        if not document_id:
            return []
        if not cache_key:
            cache_key = self._derive_cache_key(document_id, manifest)
        chunks = manifest.get("chunks")
        if not isinstance(chunks, Sequence):
            return []
        version = _string_or_none(manifest.get("version"))
        content_hash = _string_or_none(manifest.get("content_hash"))
        generated_at = _float_or_none(manifest.get("generated_at"))
        window = _window_dict(manifest.get("window"))
        chunk_chars = _int_or_none(manifest.get("chunk_chars"))
        chunk_overlap = _int_or_none(manifest.get("chunk_overlap"))
        entries: list[ChunkIndexEntry] = []
        with self._lock:
            doc_chunks = self._chunks.setdefault(document_id, {})
            for position, chunk in enumerate(chunks):
                if not isinstance(chunk, Mapping):
                    continue
                chunk_id = _string_or_none(chunk.get("id"))
                if not chunk_id:
                    continue
                start = max(0, int(chunk.get("start", 0)))
                end = max(start, int(chunk.get("end", start)))
                length = max(0, int(chunk.get("length", end - start)))
                entry = ChunkIndexEntry(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    start=start,
                    end=end,
                    length=length,
                    chunk_hash=_string_or_none(chunk.get("hash")),
                    span_overlap=bool(chunk.get("span_overlap")),
                    outline_pointer_id=_string_or_none(chunk.get("outline_pointer_id")),
                    chunk_profile=chunk_profile,
                    cache_key=cache_key,
                    version=version,
                    content_hash=content_hash,
                    chunk_chars=chunk_chars,
                    chunk_overlap=chunk_overlap,
                    window=window,
                    manifest_generated_at=generated_at,
                    position=position,
                )
                doc_chunks[chunk_id] = entry
                entries.append(entry)
            record = ChunkManifestRecord(
                document_id=document_id,
                cache_key=cache_key,
                version=version,
                content_hash=content_hash,
                chunk_profile=chunk_profile,
                chunk_ids=tuple(entry.chunk_id for entry in entries),
                generated_at=generated_at,
                window=window,
            )
            self._store_manifest_record(record)
        LOGGER.debug("ChunkIndex stored %s chunks for %s", len(entries), document_id)
        return entries

    def get_chunk(
        self,
        chunk_id: str,
        *,
        document_id: str | None = None,
        cache_key: str | None = None,
        version: str | None = None,
    ) -> ChunkIndexEntry | None:
        """Return the chunk entry if it is still valid."""

        if not chunk_id:
            return None
        target_doc = document_id or _document_id_from_chunk(chunk_id)
        if not target_doc:
            return None
        with self._lock:
            chunks = self._chunks.get(target_doc)
            if not chunks:
                return None
            entry = chunks.get(chunk_id)
            if entry is None:
                return None
            if cache_key and entry.cache_key != cache_key:
                return None
            if version and entry.version and entry.version != version:
                return None
            return entry

    def iter_chunks(
        self,
        cache_key: str,
        *,
        document_id: str | None = None,
        start_chunk_id: str | None = None,
        limit: int = 1,
    ) -> list[ChunkIndexEntry]:
        """Return up to ``limit`` chunks from a manifest (preserving order)."""

        if not cache_key:
            return []
        with self._lock:
            record = self._resolve_manifest(cache_key, document_id=document_id)
            if record is None or not record.chunk_ids:
                return []
            start_index = 0
            if start_chunk_id:
                try:
                    start_index = record.chunk_ids.index(start_chunk_id)
                except ValueError:
                    start_index = 0
            if start_index >= len(record.chunk_ids):
                return []
            available = len(record.chunk_ids) - start_index
            count = max(1, min(int(limit or 1), available))
            doc_chunks = self._chunks.get(record.document_id)
            if not doc_chunks:
                return []
            result: list[ChunkIndexEntry] = []
            for chunk_id in record.chunk_ids[start_index : start_index + count]:
                entry = doc_chunks.get(chunk_id)
                if entry is not None:
                    result.append(entry)
            return result

    def get_manifest(self, cache_key: str) -> ChunkManifestRecord | None:
        with self._lock:
            return self._cache_index.get(cache_key)

    def evict_document(self, document_id: str) -> None:
        """Remove all manifests and chunks for ``document_id``."""

        if not document_id:
            return
        with self._lock:
            self._chunks.pop(document_id, None)
            manifest_cache = self._manifests.pop(document_id, None)
            if manifest_cache:
                for cache_key in list(manifest_cache.keys()):
                    self._cache_index.pop(cache_key, None)
        LOGGER.debug("ChunkIndex evicted document %s", document_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _store_manifest_record(self, record: ChunkManifestRecord) -> None:
        manifest_cache = self._manifests.setdefault(record.document_id, OrderedDict())
        if record.cache_key in manifest_cache:
            manifest_cache.pop(record.cache_key, None)
        manifest_cache[record.cache_key] = record
        self._cache_index[record.cache_key] = record
        while len(manifest_cache) > self._manifest_limit:
            _, evicted = manifest_cache.popitem(last=False)
            self._cache_index.pop(evicted.cache_key, None)
            self._remove_manifest_chunks(evicted)

    def _remove_manifest_chunks(self, record: ChunkManifestRecord) -> None:
        doc_chunks = self._chunks.get(record.document_id)
        if not doc_chunks:
            return
        for chunk_id in record.chunk_ids:
            doc_chunks.pop(chunk_id, None)
        if not doc_chunks:
            self._chunks.pop(record.document_id, None)

    def _resolve_manifest(
        self,
        cache_key: str,
        *,
        document_id: str | None = None,
    ) -> ChunkManifestRecord | None:
        record = self._cache_index.get(cache_key)
        if record is None:
            return None
        if document_id and record.document_id != document_id:
            return None
        return record

    def _handle_cache_event(self, event: DocumentChangedEvent | DocumentClosedEvent) -> None:
        self.evict_document(event.document_id)

    @staticmethod
    def _derive_cache_key(document_id: str, manifest: Mapping[str, Any]) -> str:
        window = manifest.get("window") or {}
        start = getattr(window, "get", lambda *args, **kwargs: 0)("start", 0)
        end = getattr(window, "get", lambda *args, **kwargs: 0)("end", 0)
        profile = manifest.get("chunk_profile") or "auto"
        return f"{document_id}:{profile}:{start}:{end}"


def _document_id_from_chunk(chunk_id: str | None) -> str | None:
    if not chunk_id:
        return None
    parts = chunk_id.split(":", 3)
    if len(parts) >= 3 and parts[0] == "chunk":
        return parts[1]
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _window_dict(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, Mapping):
        window: dict[str, Any] = {
            "start": int(candidate.get("start", 0)),
            "end": int(candidate.get("end", 0)),
            "length": int(candidate.get("length", max(0, int(candidate.get("end", 0)) - int(candidate.get("start", 0))))),
        }
        return window
    return {"start": 0, "end": 0, "length": 0}


__all__ = ["ChunkIndex", "ChunkIndexEntry", "ChunkManifestRecord"]
