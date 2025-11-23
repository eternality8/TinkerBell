"""Caching helpers for the subagent sandbox."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from threading import RLock
from typing import Dict, MutableMapping, Sequence

from ..ai_types import SubagentJob, SubagentJobResult
from ..services import telemetry as telemetry_service
from .cache_bus import (
    DocumentCacheBus,
    DocumentCacheEvent,
    DocumentChangedEvent,
    DocumentClosedEvent,
    get_document_cache_bus,
)


@dataclass(slots=True)
class _CacheEntry:
    document_id: str
    chunk_hash: str
    signature: str
    version_id: str | None
    result: SubagentJobResult
    created_at: float


class SubagentResultCache:
    """In-memory cache of successful subagent job results."""

    def __init__(
        self,
        *,
        max_entries: int = 64,
        ttl_seconds: float | None = 900.0,
        bus: DocumentCacheBus | None = None,
    ) -> None:
        self._max_entries = max(1, int(max_entries))
        self._ttl_seconds = None if ttl_seconds is None else max(0.0, float(ttl_seconds))
        self._entries: Dict[str, _CacheEntry] = {}
        self._doc_index: MutableMapping[str, set[str]] = {}
        self._lock = RLock()
        self._bus = bus or get_document_cache_bus()
        self._bus.subscribe(DocumentChangedEvent, self._handle_cache_event, weak=True)  # type: ignore[arg-type]
        self._bus.subscribe(DocumentClosedEvent, self._handle_cache_event, weak=True)  # type: ignore[arg-type]

    def get(self, job: SubagentJob, *, emit_miss: bool = True) -> SubagentJobResult | None:
        signature = self._build_signature(job)
        if signature is None:
            return None
        now = time.time()
        with self._lock:
            entry = self._entries.get(signature)
            if entry is None:
                if emit_miss:
                    self._emit("subagent.cache_miss", job, reason="missing")
                return None
            if self._ttl_seconds and now - entry.created_at >= self._ttl_seconds:
                self._evict_locked(signature)
                if emit_miss:
                    self._emit("subagent.cache_miss", job, reason="expired")
                return None
            result = self._copy_result(entry.result)
        age_ms = max(0.0, (now - entry.created_at) * 1000.0)
        self._emit(
            "subagent.cache_hit",
            job,
            extra={
                "age_ms": round(age_ms, 3),
                "version_id": entry.version_id,
            },
        )
        return result

    def store(self, job: SubagentJob) -> None:
        signature = self._build_signature(job)
        if signature is None or job.result is None:
            return
        chunk_hash = job.chunk_hash or job.dedup_hash
        document_id = job.document_id
        if not chunk_hash or not document_id:
            return
        entry = _CacheEntry(
            document_id=document_id,
            chunk_hash=chunk_hash,
            signature=signature,
            version_id=job.chunk_ref.version_id,
            result=self._copy_result(job.result),
            created_at=time.time(),
        )
        with self._lock:
            self._purge_expired_locked()
            self._entries[signature] = entry
            self._doc_index.setdefault(document_id, set()).add(signature)
            self._enforce_capacity_locked()
        self._emit("subagent.cache_store", job)

    def clear(self, *, document_id: str | None = None) -> None:
        with self._lock:
            if document_id is None:
                self._entries.clear()
                self._doc_index.clear()
                return
            keys = self._doc_index.pop(document_id, set())
            for key in keys:
                self._entries.pop(key, None)

    def _handle_cache_event(self, event: DocumentCacheEvent) -> None:
        document_id = getattr(event, "document_id", None)
        if not document_id:
            return
        self.clear(document_id=document_id)

    def _purge_expired_locked(self) -> None:
        if not self._ttl_seconds:
            return
        now = time.time()
        stale_keys = [key for key, entry in self._entries.items() if now - entry.created_at >= self._ttl_seconds]
        for key in stale_keys:
            self._evict_locked(key)

    def _enforce_capacity_locked(self) -> None:
        if len(self._entries) <= self._max_entries:
            return
        surplus = len(self._entries) - self._max_entries
        order = sorted(self._entries.values(), key=lambda entry: entry.created_at)
        for entry in order:
            if surplus <= 0:
                break
            self._evict_locked(entry.signature)
            surplus -= 1

    def _evict_locked(self, signature: str) -> None:
        entry = self._entries.pop(signature, None)
        if entry is None:
            return
        doc_keys = self._doc_index.get(entry.document_id)
        if doc_keys is not None:
            doc_keys.discard(signature)
            if not doc_keys:
                self._doc_index.pop(entry.document_id, None)
        self._emit("subagent.cache_evicted", None, extra={"document_id": entry.document_id})

    def _build_signature(self, job: SubagentJob) -> str | None:
        document_id = job.document_id
        chunk_hash = job.chunk_hash or job.dedup_hash
        if not document_id or not chunk_hash:
            return None
        allowed_tools = job.allowed_tools or ()
        hasher = hashlib.sha1()
        hasher.update(document_id.encode("utf-8", errors="ignore"))
        hasher.update(b"|")
        hasher.update(chunk_hash.encode("utf-8", errors="ignore"))
        version = job.chunk_ref.version_id or ""
        hasher.update(b"|")
        hasher.update(version.encode("utf-8", errors="ignore"))
        instructions = (job.instructions or "").strip()
        hasher.update(b"|")
        hasher.update(instructions.encode("utf-8", errors="ignore"))
        for tool in allowed_tools:
            hasher.update(b"|tool=")
            hasher.update(tool.encode("utf-8", errors="ignore"))
        budget_signature = f"{job.budget.max_prompt_tokens}:{job.budget.max_completion_tokens}:{job.budget.max_runtime_seconds}".encode(
            "utf-8"
        )
        hasher.update(budget_signature)
        return hasher.hexdigest()

    def _copy_result(self, result: SubagentJobResult) -> SubagentJobResult:
        tool_calls: Sequence[dict[str, object]] | None = result.tool_calls
        if tool_calls:
            tool_calls = tuple(dict(call) for call in tool_calls)
        else:
            tool_calls = ()
        return replace(
            result,
            tool_calls=tool_calls,
        )

    def _emit(
        self,
        event_name: str,
        job: SubagentJob | None,
        *,
        reason: str | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        emitter = getattr(telemetry_service, "emit", None)
        if not callable(emitter):
            return
        payload: dict[str, object] = dict(extra or {})
        if job is not None:
            payload.setdefault("document_id", job.document_id)
            if job.chunk_hash:
                payload.setdefault("chunk_hash", job.chunk_hash)
            if job.chunk_ref.version_id:
                payload.setdefault("version_id", job.chunk_ref.version_id)
        if reason:
            payload["reason"] = reason
        try:
            emitter(event_name, payload)
        except Exception:  # pragma: no cover - telemetry failures shouldn't break cache
            pass


__all__ = ["SubagentResultCache"]
