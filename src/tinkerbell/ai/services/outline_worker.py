"""Asynchronous outline builder worker and parser helpers."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import heapq
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from ...editor.document_model import DocumentMetadata, DocumentState
from ..memory.buffers import (
    DocumentSummaryMemory,
    MemoryStore,
    OutlineCacheStore,
    OutlineNode,
    SummaryRecord,
)
from ..memory.embeddings import DocumentEmbeddingIndex
from ..memory.cache_bus import (
    DocumentCacheBus,
    DocumentCacheEvent,
    DocumentChangedEvent,
    DocumentClosedEvent,
    get_document_cache_bus,
)
from ..utils.document_checks import (
    document_size_bytes,
    is_huge_document,
    unsupported_format_reason,
)
from ...services.telemetry import count_text_tokens, emit

__all__ = ["OutlineBuilderWorker", "OutlineBuilderConfig", "build_outline_nodes"]

LOGGER = logging.getLogger(__name__)
_TOKEN_ESTIMATE_MIN = 1

TokenCounter = Callable[[str], int]
DocumentProvider = Callable[[str], DocumentState | None]


@dataclass(slots=True)
class OutlineBuilderConfig:
    """Tunable parameters for the outline worker."""

    max_blurb_chars: int = 240
    max_summary_chars: int = 1_200
    debounce_seconds: float = 0.75
    min_rebuild_interval: float = 5.0
    max_records: int = 32


@dataclass(slots=True)
class _OutlineJob:
    document_id: str
    version_id: int
    content_hash: str
    edited_ranges: tuple[tuple[int, int], ...]
    doc_length: int
    edit_size: int
    available_at: float
    priority: tuple[float, float, int] = field(default_factory=tuple)


@dataclass(slots=True)
class _Section:
    level: int
    title: str
    start: int
    content_start: int
    end: int = 0


@dataclass(slots=True)
class OutlineBuildStatus:
    document_id: str
    version_id: int | None
    latency_ms: float
    completed_at: float


class OutlineBuilderWorker:
    """Builds structured outlines in the background when documents change."""

    def __init__(
        self,
        *,
        document_provider: DocumentProvider,
        storage_dir: Path | str,
        cache_bus: DocumentCacheBus | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
        config: OutlineBuilderConfig | None = None,
        token_counter: TokenCounter | None = None,
        embedding_index: DocumentEmbeddingIndex | None = None,
        status_callback: Callable[[SummaryRecord, float], None] | None = None,
    ) -> None:
        if document_provider is None:
            raise ValueError("document_provider is required")
        self._document_provider = document_provider
        self._loop = loop or asyncio.get_event_loop()
        self._config = config or OutlineBuilderConfig()
        self._token_counter = token_counter or _estimate_tokens
        self._cache_bus = cache_bus or get_document_cache_bus()
        self._queue: list[tuple[tuple[float, float, int], _OutlineJob]] = []
        self._pending: dict[str, _OutlineJob] = {}
        self._last_build_at: dict[str, float] = {}
        self._queue_event = asyncio.Event()
        self._job_seq = 0
        self._closed = False
        self._embedding_index = embedding_index
        self._status_callback = status_callback
        self._build_stats: dict[str, OutlineBuildStatus] = {}
        self._active_document_id: str | None = None

        cache_root = Path(storage_dir).expanduser().resolve()
        memory_dir = cache_root / "memory"
        cache_dir = cache_root / "outline_cache"
        memory_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        self._memory = DocumentSummaryMemory(
            max_entries=max(1, self._config.max_records),
            max_summary_chars=max(0, self._config.max_summary_chars),
        )
        self._memory_store = MemoryStore(memory_dir)
        self._cache_store = OutlineCacheStore(cache_dir)

        # Load persisted summaries and cached outlines when available.
        try:
            self._memory_store.load_document_summaries(self._memory)
        except Exception:  # pragma: no cover - defensive path
            LOGGER.debug("Unable to load persisted document summaries", exc_info=True)
        cached_payload = self._cache_store.load_all()
        if cached_payload:
            valid_payload = self._validate_cached_payload(cached_payload)
            if valid_payload:
                self._memory.load_dict(valid_payload)

        self._worker_task = self._loop.create_task(self._drain_queue())
        self._cache_bus.subscribe(DocumentChangedEvent, self._handle_changed, weak=True)
        self._cache_bus.subscribe(DocumentClosedEvent, self._handle_closed, weak=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self._loop

    @property
    def memory(self) -> DocumentSummaryMemory:
        return self._memory

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue_event.set()
        if self._worker_task:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):  # type: ignore[name-defined]
                await self._worker_task

    # ------------------------------------------------------------------
    # Cache bus handlers
    # ------------------------------------------------------------------
    def _handle_changed(self, event: DocumentCacheEvent) -> None:
        if self._closed:
            return
        if not isinstance(event, DocumentChangedEvent):
            return
        self._loop.call_soon_threadsafe(self._enqueue_changed, event)

    def _handle_closed(self, event: DocumentCacheEvent) -> None:
        if self._closed:
            return
        if not isinstance(event, DocumentClosedEvent):
            return
        self._loop.call_soon_threadsafe(self._drop_document_jobs, event.document_id)

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------
    def _enqueue_changed(self, event: DocumentChangedEvent) -> None:
        document = self._document_provider(event.document_id)
        if document is None:
            return
        doc_length = len(document.text)
        edit_size = _total_edit_span(event.edited_ranges)
        available_at = _now() + max(0.0, self._config.debounce_seconds)
        priority = (-doc_length, -edit_size, self._job_seq)
        job = _OutlineJob(
            document_id=event.document_id,
            version_id=event.version_id,
            content_hash=event.content_hash,
            edited_ranges=tuple(event.edited_ranges),
            doc_length=doc_length,
            edit_size=edit_size,
            available_at=available_at,
            priority=priority,
        )
        self._pending[event.document_id] = job
        heapq.heappush(self._queue, (priority, job))
        self._job_seq += 1
        self._queue_event.set()

    def _drop_document_jobs(self, document_id: str) -> None:
        self._pending.pop(document_id, None)

    def is_rebuild_pending(self, document_id: str) -> bool:
        if self._closed:
            return False
        if document_id in self._pending:
            return True
        return self._active_document_id == document_id

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------
    async def _drain_queue(self) -> None:
        try:
            while not self._closed:
                await self._queue_event.wait()
                self._queue_event.clear()
                while self._queue and not self._closed:
                    _, job = heapq.heappop(self._queue)
                    if self._pending.get(job.document_id) is not job:
                        continue
                    now = _now()
                    if job.available_at > now:
                        await asyncio.sleep(job.available_at - now)
                    last_built = self._last_build_at.get(job.document_id)
                    if last_built is not None:
                        cooldown = self._config.min_rebuild_interval - (now - last_built)
                        if cooldown > 0:
                            await asyncio.sleep(cooldown)
                    await self._process_job(job)
                    self._pending.pop(job.document_id, None)
                    self._last_build_at[job.document_id] = _now()
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancellation
            return

    async def _process_job(self, job: _OutlineJob) -> None:
        document = self._document_provider(job.document_id)
        if document is None:
            return
        doc_text = document.text or ""
        doc_length = len(doc_text)
        document_tokens = count_text_tokens(doc_text, estimate_only=True) if doc_text else 0
        document_bytes = document_size_bytes(document)
        emit(
            "outline.build.start",
            {
                "document_id": document.document_id,
                "version_id": document.version_id,
                "document_length": doc_length,
                "document_tokens": document_tokens,
                "document_bytes": document_bytes,
                "edit_size": job.edit_size,
                "queue_depth": len(self._queue),
                "pending_jobs": len(self._pending),
            },
        )
        unsupported_reason = unsupported_format_reason(document)
        if unsupported_reason:
            record = self._record_unsupported_outline(document, unsupported_reason, document_bytes)
            emit(
                "outline.build.end",
                {
                    "document_id": document.document_id,
                    "version_id": document.version_id,
                    "status": "unsupported_format",
                    "latency_ms": 0.0,
                    "document_length": doc_length,
                    "document_tokens": document_tokens,
                    "document_bytes": document_bytes,
                    "node_count": 0,
                    "outline_tokens": 0,
                    "tokens_saved": None,
                    "outline_digest": record.outline_hash if record else None,
                    "reason": unsupported_reason,
                },
            )
            return
        started = time.perf_counter()
        self._active_document_id = document.document_id
        record: SummaryRecord | None = None
        error_text: str | None = None
        try:
            record = self._build_outline_for_document(document, document_bytes=document_bytes)
        except Exception as exc:  # pragma: no cover - defensive guard
            error_text = str(exc)
            LOGGER.exception("Outline builder failed for %s", job.document_id)
        finally:
            self._active_document_id = None
        latency_ms = (time.perf_counter() - started) * 1000.0
        node_count: int | None = None
        outline_tokens: int | None = None
        tokens_saved: int | None = None
        status = "ok"
        if record is None:
            status = "error" if error_text else "no_nodes"
        else:
            node_count = _count_outline_nodes(record.nodes)
            outline_tokens = _count_outline_tokens(record.nodes)
            tokens_saved = max(0, document_tokens - outline_tokens) if outline_tokens is not None else None
        payload = {
            "document_id": document.document_id,
            "version_id": document.version_id,
            "status": status,
            "latency_ms": round(latency_ms, 3),
            "document_length": doc_length,
            "document_tokens": document_tokens,
            "document_bytes": document_bytes,
            "node_count": node_count,
            "outline_tokens": outline_tokens,
            "tokens_saved": tokens_saved,
            "outline_digest": record.outline_hash if record else None,
        }
        if error_text:
            payload["error"] = error_text[:200]
        emit("outline.build.end", payload)
        if record is None:
            return
        self._record_build_stats(document.document_id, document.version_id, latency_ms)
        self._notify_status_listeners(record, latency_ms)
        await self._ingest_embeddings(document, record)
        try:
            self._memory_store.save_document_summaries(self._memory)
        except Exception:  # pragma: no cover - best effort persistence
            LOGGER.debug("Unable to persist document summaries", exc_info=True)

    # ------------------------------------------------------------------
    # Outline construction helpers
    # ------------------------------------------------------------------
    def _build_outline_for_document(self, document: DocumentState, *, document_bytes: int) -> SummaryRecord | None:
        nodes = build_outline_nodes(
            document.document_id,
            document.text,
            language=_resolve_language(document),
            max_blurb_chars=self._config.max_blurb_chars,
            token_counter=self._token_counter,
        )
        if not nodes:
            return None
        guardrail_applied = False
        if is_huge_document(document):
            guardrail_applied = _limit_outline_depth(nodes, max_level=1)
        outline_hash = _build_outline_hash(nodes, document.version_id, document.content_hash)
        summary_text = _format_outline_summary(nodes)
        highlights = _collect_highlights(nodes)
        metadata: dict[str, int | bool] = {"document_bytes": document_bytes}
        if guardrail_applied:
            metadata["huge_document_guardrail"] = True
        record = self._memory.update(
            document.document_id,
            summary=summary_text,
            highlights=highlights,
            version_id=document.version_id,
            outline_hash=outline_hash,
            nodes=nodes,
            content_hash=document.content_hash,
            metadata=metadata,
        )
        try:
            self._cache_store.save(record)
        except Exception:  # pragma: no cover - cache best-effort
            LOGGER.debug("Unable to save outline cache for %s", document.document_id, exc_info=True)
        return record

    async def _ingest_embeddings(self, document: DocumentState, record: SummaryRecord) -> None:
        if self._embedding_index is None:
            return
        nodes = record.nodes
        if not nodes:
            return
        try:
            await self._embedding_index.ingest_outline(
                document=document,
                nodes=nodes,
                outline_hash=record.outline_hash,
            )
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Embedding ingestion failed for %s", document.document_id, exc_info=True)

    def get_build_status(self, document_id: str) -> OutlineBuildStatus | None:
        return self._build_stats.get(document_id)

    def update_embedding_index(self, embedding_index: DocumentEmbeddingIndex | None) -> None:
        """Replace the embedding index used for ingestion at runtime."""

        self._embedding_index = embedding_index

    def _record_build_stats(self, document_id: str, version_id: int | None, latency_ms: float) -> None:
        self._build_stats[document_id] = OutlineBuildStatus(
            document_id=document_id,
            version_id=version_id,
            latency_ms=max(0.0, float(latency_ms)),
            completed_at=_now(),
        )

    def _notify_status_listeners(self, record: SummaryRecord, latency_ms: float) -> None:
        callback = self._status_callback
        if callback is None:
            return
        try:
            callback(record, float(latency_ms))
        except Exception:  # pragma: no cover - listeners must not break worker
            LOGGER.debug("Outline status callback failed", exc_info=True)

    def _record_unsupported_outline(
        self,
        document: DocumentState,
        reason: str,
        document_bytes: int,
    ) -> SummaryRecord | None:
        record = self._memory.update(
            document.document_id,
            summary="Unsupported document format",
            highlights=[],
            version_id=document.version_id,
            outline_hash=None,
            nodes=[],
            content_hash=document.content_hash,
            metadata={
                "unsupported_format": reason,
                "document_bytes": document_bytes,
            },
        )
        if record:
            try:
                self._cache_store.save(record)
            except Exception:  # pragma: no cover - cache best-effort
                LOGGER.debug("Unable to cache unsupported outline for %s", document.document_id, exc_info=True)
        return record

    def _validate_cached_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        valid: dict[str, Any] = {}
        for document_id, record_payload in payload.items():
            record_mapping = record_payload if isinstance(record_payload, Mapping) else None
            if record_mapping is None:
                continue
            try:
                record = SummaryRecord.from_dict(record_mapping)
            except Exception:  # pragma: no cover - defensive path
                continue
            outline_hash = record.outline_hash
            if outline_hash and record.version_id is not None and record.content_hash:
                expected = _build_outline_hash(record.nodes, record.version_id, record.content_hash)
                if expected != outline_hash:
                    LOGGER.debug("Outline cache mismatch detected for %s", document_id)
                    self._schedule_rebuild(document_id)
                    continue
            valid[document_id] = record_mapping
        return valid

    def _schedule_rebuild(self, document_id: str) -> None:
        document = self._document_provider(document_id)
        if document is None:
            return
        self._enqueue_changed(
            DocumentChangedEvent(
                document_id=document.document_id,
                version_id=document.version_id,
                content_hash=document.content_hash,
                edited_ranges=((0, len(document.text or "")),),
            )
        )


def build_outline_nodes(
    document_id: str,
    text: str,
    *,
    language: str | None = None,
    max_blurb_chars: int = 240,
    token_counter: TokenCounter | None = None,
) -> list[OutlineNode]:
    normalized = (language or "").strip().lower()
    counter = token_counter or _estimate_tokens
    sections: list[_Section]
    if normalized in {"md", "markdown"}:
        sections = _parse_markdown_sections(text)
    elif normalized in {"yaml", "yml"}:
        sections = _parse_yaml_sections(text)
    elif normalized == "json":
        sections = _parse_json_sections(text)
    else:
        sections = []
    if not sections:
        sections = _fallback_sections(text, chunk_chars=max(400, max_blurb_chars * 2))
    _assign_section_bounds(sections, len(text))
    nodes = _sections_to_nodes(document_id, text, sections, max_blurb_chars, counter)
    return nodes


def _resolve_language(document: DocumentState) -> str:
    language = (document.metadata.language if isinstance(document.metadata, DocumentMetadata) else None) or ""
    if language:
        return language
    path = getattr(document.metadata, "path", None)
    if path:
        suffix = Path(path).suffix.lower().lstrip(".")
        return suffix or "text"
    return "text"


def _parse_markdown_sections(text: str) -> list[_Section]:
    pattern = re.compile(r"^(?P<hashes>#{1,6})\s+(?P<title>[^\n]+)$", re.MULTILINE)
    sections: list[_Section] = []
    for match in pattern.finditer(text):
        level = len(match.group("hashes"))
        heading = match.group("title").strip(" #")
        start = match.start()
        content_start = match.end()
        sections.append(_Section(level=level, title=heading or f"Heading {len(sections)+1}", start=start, content_start=content_start))
    return sections


def _parse_yaml_sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            offset += len(line)
            continue
        indent = len(line) - len(line.lstrip(" \t"))
        if stripped.startswith("- "):
            title = stripped[2:].strip() or "List item"
            level = indent // 2 + 1
            sections.append(_Section(level=level, title=title, start=offset, content_start=offset + len(line)))
        elif ":" in stripped:
            key = stripped.split(":", 1)[0].strip().strip('"\'')
            if key:
                level = indent // 2 + 1
                sections.append(_Section(level=level, title=key, start=offset, content_start=offset + len(line)))
        offset += len(line)
    return sections


def _parse_json_sections(text: str) -> list[_Section]:
    sections: list[_Section] = []
    pattern = re.compile(r"^(?P<indent>\s*)\"(?P<key>[^\"]+)\"\s*:")
    offset = 0
    for line in text.splitlines(keepends=True):
        match = pattern.match(line)
        if match:
            indent = len(match.group("indent"))
            key = match.group("key").strip()
            level = indent // 2 + 1
            sections.append(_Section(level=level, title=key or "field", start=offset, content_start=offset + len(line)))
        offset += len(line)
    return sections


def _fallback_sections(text: str, *, chunk_chars: int) -> list[_Section]:
    sections: list[_Section] = []
    chunk_chars = max(1, chunk_chars)
    start = 0
    idx = 1
    length = len(text)
    while start < length:
        end = min(length, start + chunk_chars)
        excerpt = text[start:end].strip().splitlines()
        title = excerpt[0][:80] if excerpt else f"Chunk {idx}"
        sections.append(_Section(level=1, title=title or f"Chunk {idx}", start=start, content_start=start))
        sections[-1].end = end
        start = end
        idx += 1
    if not sections:
        sections.append(_Section(level=1, title="Document", start=0, content_start=0, end=length))
    return sections


def _assign_section_bounds(sections: Sequence[_Section], text_length: int) -> None:
    for index, section in enumerate(sections):
        next_start = sections[index + 1].start if index + 1 < len(sections) else text_length
        section.end = max(section.start, next_start)
        if section.content_start <= section.start:
            section.content_start = min(section.end, section.start)


def _sections_to_nodes(
    document_id: str,
    text: str,
    sections: Sequence[_Section],
    max_blurb_chars: int,
    token_counter: TokenCounter,
) -> list[OutlineNode]:
    nodes: list[OutlineNode] = []
    stack: list[OutlineNode] = []
    for index, section in enumerate(sections):
        start = max(0, min(len(text), section.start))
        end = max(start, min(len(text), section.end))
        content_start = max(start, min(end, section.content_start))
        excerpt = text[content_start:end].strip()
        blurb, truncated = _make_blurb(excerpt or section.title, max_blurb_chars)
        node = OutlineNode(
            id=f"node-{index}",
            parent_id=None,
            level=max(1, section.level),
            text=section.title or f"Section {index + 1}",
            char_range=(start, end),
            chunk_id=_build_chunk_id(document_id, start, end),
            blurb=blurb,
            token_estimate=token_counter(blurb),
            truncated=truncated,
        )
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if stack:
            node.parent_id = stack[-1].id
            stack[-1].children.append(node)
        else:
            nodes.append(node)
        stack.append(node)
    return nodes


def _make_blurb(text: str, max_chars: int) -> tuple[str, bool]:
    normalized = " ".join(text.split())
    if not normalized:
        return "", False
    if max_chars and len(normalized) > max_chars:
        return normalized[: max_chars].rstrip(), True
    return normalized, False


def _build_chunk_id(document_id: str, start: int, end: int) -> str:
    return f"{document_id}-chunk-{start}-{end}"


def _estimate_tokens(text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return max(_TOKEN_ESTIMATE_MIN, len(stripped.split()))


def _total_edit_span(spans: Iterable[tuple[int, int]]) -> int:
    total = 0
    for start, end in spans:
        total += max(0, end - start)
    return total


def _now() -> float:
    return time.monotonic()


def _build_outline_hash(nodes: Sequence[OutlineNode], version_id: int, content_hash: str) -> str:
    payload = {
        "version_id": version_id,
        "content_hash": content_hash,
        "nodes": [node.to_dict() for node in nodes],
    }
    digest = hashlib.sha1()
    digest.update(str(payload).encode("utf-8"))
    return digest.hexdigest()


def _format_outline_summary(nodes: Sequence[OutlineNode]) -> str:
    if not nodes:
        return ""
    root = nodes[0]
    if root.blurb:
        return f"{root.text}: {root.blurb}"
    return root.text


def _collect_highlights(nodes: Sequence[OutlineNode], *, limit: int = 4) -> list[str]:
    values: list[str] = []
    for node in _iter_nodes(nodes):
        if node.level == 1 and node.text:
            values.append(node.text)
        if len(values) >= limit:
            break
    return values


def _iter_nodes(nodes: Sequence[OutlineNode]) -> Iterable[OutlineNode]:
    for node in nodes:
        yield node
        if node.children:
            yield from _iter_nodes(node.children)


def _count_outline_nodes(nodes: Sequence[OutlineNode]) -> int:
    total = 0
    for node in nodes:
        total += 1
        if node.children:
            total += _count_outline_nodes(node.children)
    return total


def _count_outline_tokens(nodes: Sequence[OutlineNode]) -> int:
    total = 0
    for node in nodes:
        total += max(1, int(node.token_estimate or 0))
        if node.children:
            total += _count_outline_tokens(node.children)
    return total


def _limit_outline_depth(nodes: Sequence[OutlineNode], *, max_level: int) -> bool:
    if not nodes:
        return False
    changed = False
    stack: list[OutlineNode] = list(nodes)
    while stack:
        node = stack.pop()
        if node.level >= max_level and node.children:
            changed = True
            for child in node.children:
                _mark_subtree_truncated(child)
            node.children.clear()
            node.truncated = True
        else:
            stack.extend(node.children)
    return changed


def _mark_subtree_truncated(node: OutlineNode) -> None:
    node.truncated = True
    if node.children:
        for child in node.children:
            _mark_subtree_truncated(child)
        node.children.clear()
