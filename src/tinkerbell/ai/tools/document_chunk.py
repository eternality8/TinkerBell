"""Tool that rehydrates chunk manifests into inline text payloads."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping

from ..memory.chunk_index import ChunkIndex, ChunkIndexEntry, ChunkManifestRecord
from ...services.telemetry import count_text_tokens, emit
from .document_snapshot import SnapshotProvider

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkRuntimeSettings:
    """Runtime chunking configuration shared with the AI controller."""

    default_profile: str = "auto"
    overlap_chars: int = 256
    max_inline_tokens: int = 1_800
    iterator_limit: int = 4


def _coerce_runtime_settings(payload: Any) -> ChunkRuntimeSettings:
    if isinstance(payload, ChunkRuntimeSettings):
        return payload
    if isinstance(payload, Mapping):
        return ChunkRuntimeSettings(
            default_profile=str(payload.get("default_profile", "auto") or "auto"),
            overlap_chars=int(payload.get("overlap_chars", 256) or 256),
            max_inline_tokens=int(payload.get("max_inline_tokens", 1_800) or 1_800),
            iterator_limit=int(payload.get("iterator_limit", 4) or 4),
        )
    default_profile = getattr(payload, "default_profile", "auto") or "auto"
    overlap_chars = getattr(payload, "overlap_chars", 256) or 256
    max_inline_tokens = getattr(payload, "max_inline_tokens", 1_800) or 1_800
    iterator_limit = getattr(payload, "iterator_limit", 4) or 4
    return ChunkRuntimeSettings(
        default_profile=str(default_profile),
        overlap_chars=int(overlap_chars),
        max_inline_tokens=int(max_inline_tokens),
        iterator_limit=int(iterator_limit),
    )


@dataclass(slots=True)
class DocumentChunkTool:
    """Return inline chunk text (or pointers) based on a cached manifest."""

    bridge: SnapshotProvider
    chunk_index: ChunkIndex
    chunk_config_resolver: Callable[[], Any] | None = None
    summarizable: ClassVar[bool] = False

    def run(
        self,
        *,
        chunk_id: str | None = None,
        document_id: str | None = None,
        cache_key: str | None = None,
        version: str | None = None,
        include_text: bool = True,
        max_tokens: int | None = None,
        iterator: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if iterator:
            return self._run_iterator(iterator, include_text=include_text, max_tokens=max_tokens)
        if not chunk_id:
            raise ValueError("chunk_id is required when iterator is not provided")
        entry = self.chunk_index.get_chunk(chunk_id, document_id=document_id, cache_key=cache_key, version=version)
        if entry is None:
            self._emit_cache_miss(chunk_id, document_id=document_id, cache_key=cache_key)
            return {
                "status": "not_found",
                "chunk_id": chunk_id,
                "document_id": document_id,
                "cache_key": cache_key,
            }
        payload = self._materialize_chunk(entry, include_text=include_text, max_tokens=max_tokens)
        status = "pointer_only" if payload.get("pointer") else "ok"
        return {
            "status": status,
            "chunk": payload,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run_iterator(self, iterator: Mapping[str, Any], *, include_text: bool, max_tokens: int | None) -> dict[str, Any]:
        cache_key = str(iterator.get("cache_key") or "").strip()
        if not cache_key:
            raise ValueError("iterator.cache_key is required")
        document_id = iterator.get("document_id")
        start_chunk_id = iterator.get("start_chunk_id")
        limit_override = iterator.get("limit")
        settings = self._chunk_settings()
        iterator_limit = max(1, settings.iterator_limit)
        if limit_override is not None:
            try:
                iterator_limit = max(1, min(int(limit_override), iterator_limit))
            except (TypeError, ValueError):
                iterator_limit = max(1, settings.iterator_limit)
        entries = self.chunk_index.iter_chunks(
            cache_key,
            document_id=document_id,
            start_chunk_id=str(start_chunk_id) if start_chunk_id else None,
            limit=iterator_limit,
        )
        if not entries:
            return {
                "status": "not_found",
                "iterator": {
                    "cache_key": cache_key,
                    "document_id": document_id,
                },
            }
        chunks = [self._materialize_chunk(entry, include_text=include_text, max_tokens=max_tokens) for entry in entries]
        manifest = self.chunk_index.get_manifest(cache_key)
        next_chunk_id = self._next_chunk_cursor(manifest, chunks[-1]["chunk_id"]) if manifest else None
        return {
            "status": "iterator",
            "iterator": {
                "cache_key": cache_key,
                "document_id": manifest.document_id if manifest else document_id,
                "count": len(chunks),
                "chunks": chunks,
                "next_chunk_id": next_chunk_id,
                "has_more": bool(next_chunk_id),
            },
        }

    def _materialize_chunk(self, entry: ChunkIndexEntry, *, include_text: bool, max_tokens: int | None) -> dict[str, Any]:
        settings = self._chunk_settings()
        inline_cap = max(256, settings.max_inline_tokens)
        if max_tokens:
            try:
                inline_cap = max(256, min(inline_cap, int(max_tokens)))
            except (TypeError, ValueError):
                pass
        inline_tokens: int | None = None
        text_payload = ""
        pointer: dict[str, Any] | None = None
        if include_text:
            text_payload = self._fetch_chunk_text(entry)
            if text_payload:
                inline_tokens = count_text_tokens(text_payload, estimate_only=True)
                if inline_tokens is not None and inline_tokens > inline_cap:
                    pointer = self._build_pointer(
                        entry,
                        reason="inline_cap_exceeded",
                        inline_tokens=inline_tokens,
                        inline_cap=inline_cap,
                    )
                    text_payload = ""
            else:
                pointer = self._build_pointer(entry, reason="empty_text", inline_cap=inline_cap)
        else:
            pointer = self._build_pointer(entry, reason="text_omitted", inline_cap=inline_cap)
        chunk_payload = {
            "document_id": entry.document_id,
            "chunk_id": entry.chunk_id,
            "chunk_profile": entry.chunk_profile,
            "cache_key": entry.cache_key,
            "start": entry.start,
            "end": entry.end,
            "length": entry.length,
            "span_overlap": entry.span_overlap,
            "chunk_hash": entry.chunk_hash,
            "outline_pointer_id": entry.outline_pointer_id,
            "version": entry.version,
            "content_hash": entry.content_hash,
            "window": entry.window,
            "chunk_chars": entry.chunk_chars,
            "chunk_overlap": entry.chunk_overlap,
            "text": text_payload,
            "inline_tokens": inline_tokens,
            "pointer": pointer,
            "max_inline_tokens": inline_cap,
        }
        self._emit_chunk_metrics(entry, inline_tokens=inline_tokens, pointer=pointer)
        return chunk_payload

    def _fetch_chunk_text(self, entry: ChunkIndexEntry) -> str:
        window = {
            "kind": "range",
            "start": entry.start,
            "end": entry.end,
            "padding": 0,
            "max_chars": max(entry.length, entry.end - entry.start, entry.chunk_chars or 0, 1),
        }
        generate = getattr(self.bridge, "generate_snapshot", None)
        if not callable(generate):
            LOGGER.debug("Chunk provider missing generate_snapshot")
            return ""
        try:
            snapshot = generate(
                delta_only=False,
                window=window,
                chunk_profile=entry.chunk_profile,
                include_text=True,
            )
        except TypeError:
            snapshot = generate(delta_only=False)
        except Exception:  # pragma: no cover - provider failures are surfaced as empty payloads
            LOGGER.debug("Chunk snapshot request failed", exc_info=True)
            return ""
        if not isinstance(snapshot, Mapping):
            return ""
        text = snapshot.get("text")
        if isinstance(text, str):
            return text
        return ""

    def _chunk_settings(self) -> ChunkRuntimeSettings:
        if self.chunk_config_resolver is None:
            return ChunkRuntimeSettings()
        try:
            payload = self.chunk_config_resolver()
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Chunk config resolver failed", exc_info=True)
            return ChunkRuntimeSettings()
        if payload is None:
            return ChunkRuntimeSettings()
        return _coerce_runtime_settings(payload)

    def _build_pointer(
        self,
        entry: ChunkIndexEntry,
        *,
        reason: str,
        inline_tokens: int | None = None,
        inline_cap: int | None = None,
    ) -> dict[str, Any]:
        pointer = {
            "reason": reason,
            "document_id": entry.document_id,
            "chunk_id": entry.chunk_id,
            "cache_key": entry.cache_key,
            "outline_pointer_id": entry.outline_pointer_id,
            "chunk_hash": entry.chunk_hash,
            "inline_tokens": inline_tokens,
            "max_inline_tokens": inline_cap,
        }
        return {key: value for key, value in pointer.items() if value is not None}

    @staticmethod
    def _next_chunk_cursor(manifest: ChunkManifestRecord | None, last_chunk_id: str | None) -> str | None:
        if manifest is None or not manifest.chunk_ids or not last_chunk_id:
            return None
        try:
            index = manifest.chunk_ids.index(last_chunk_id)
        except ValueError:
            return None
        if index + 1 >= len(manifest.chunk_ids):
            return None
        return manifest.chunk_ids[index + 1]

    def _emit_chunk_metrics(
        self,
        entry: ChunkIndexEntry,
        *,
        inline_tokens: int | None,
        pointer: Mapping[str, Any] | None,
    ) -> None:
        emit(
            "chunk_cache.hit",
            {
                "document_id": entry.document_id,
                "chunk_id": entry.chunk_id,
                "cache_key": entry.cache_key,
                "pointerized": bool(pointer),
            },
        )
        emit(
            "chunk_tool.window_tokens",
            {
                "document_id": entry.document_id,
                "chunk_id": entry.chunk_id,
                "cache_key": entry.cache_key,
                "window_length": entry.length,
                "inline_tokens": inline_tokens,
                "pointerized": bool(pointer),
            },
        )

    @staticmethod
    def _emit_cache_miss(chunk_id: str, *, document_id: str | None, cache_key: str | None) -> None:
        emit(
            "chunk_cache.miss",
            {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "cache_key": cache_key,
            },
        )


__all__ = ["DocumentChunkTool", "ChunkRuntimeSettings"]
