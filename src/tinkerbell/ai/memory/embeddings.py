"""Document embedding index, persistence helpers, and provider adapters."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import sqlite3
import time
from concurrent.futures import Future
from threading import RLock
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Protocol, Sequence

from ...editor.document_model import DocumentState
from ...services.telemetry import emit
from .buffers import OutlineNode
from .cache_bus import (
    DocumentCacheBus,
    DocumentChangedEvent,
    DocumentClosedEvent,
    get_document_cache_bus,
)

try:  # pragma: no cover - optional dependency at runtime
    from openai import AsyncOpenAI
except Exception:  # pragma: no cover - fallback when openai missing
    AsyncOpenAI = None  # type: ignore[misc]

LOGGER = logging.getLogger(__name__)
Vector = tuple[float, ...]


class EmbeddingProvider(Protocol):
    """Protocol implemented by embedding backends."""

    name: str
    max_batch_size: int

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return embeddings for document chunks."""

    async def embed_query(self, text: str) -> Sequence[float]:
        """Return embedding vector for a query string."""


@dataclass(slots=True)
class ChunkEmbeddingRecord:
    """Persisted chunk embedding metadata."""

    document_id: str
    chunk_id: str
    version_id: int
    content_hash: str
    chunk_hash: str
    start_offset: int
    end_offset: int
    outline_node_id: str | None
    token_count: int
    outline_hash: str | None
    provider: str
    dims: int
    vector: Vector
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    dirty: bool = False

    def with_metadata(
        self,
        *,
        version_id: int,
        content_hash: str,
        chunk_hash: str,
        start_offset: int,
        end_offset: int,
        outline_node_id: str | None,
        token_count: int,
        outline_hash: str | None,
    ) -> "ChunkEmbeddingRecord":
        return ChunkEmbeddingRecord(
            document_id=self.document_id,
            chunk_id=self.chunk_id,
            version_id=version_id,
            content_hash=content_hash,
            chunk_hash=chunk_hash,
            start_offset=start_offset,
            end_offset=end_offset,
            outline_node_id=outline_node_id,
            token_count=token_count,
            outline_hash=outline_hash,
            provider=self.provider,
            dims=self.dims,
            vector=self.vector,
            created_at=self.created_at,
            updated_at=time.time(),
            dirty=False,
        )


@dataclass(slots=True)
class EmbeddingIngestResult:
    processed: int
    embedded: int
    reused: int
    status: str = "ok"


@dataclass(slots=True)
class EmbeddingMatch:
    record: ChunkEmbeddingRecord
    score: float


@dataclass(slots=True)
class _ChunkSpec:
    chunk_id: str
    outline_node_id: str | None
    start: int
    end: int
    text: str
    token_count: int
    chunk_hash: str


class AsyncRateLimiter:
    """Simple asynchronous rate limiter using a minimum interval."""

    def __init__(self, *, rate_per_minute: int | None = None) -> None:
        self._interval = 0.0
        self._lock = asyncio.Lock()
        self._last_acquire = 0.0
        if rate_per_minute and rate_per_minute > 0:
            self._interval = 60.0 / float(rate_per_minute)

    async def acquire(self, tokens: int = 1) -> None:
        if self._interval <= 0 or tokens <= 0:
            return
        wait_time = self._interval * max(1, tokens)
        async with self._lock:
            now = time.monotonic()
            delta = now - self._last_acquire
            remaining = wait_time - delta
            if remaining > 0:
                await asyncio.sleep(remaining)
            self._last_acquire = time.monotonic()


class EmbeddingStore:
    """SQLite-backed persistence for chunk embeddings."""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = RLock()
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunk_embeddings (
                        document_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        version_id INTEGER NOT NULL,
                        content_hash TEXT NOT NULL,
                        chunk_hash TEXT NOT NULL,
                        start_offset INTEGER NOT NULL,
                        end_offset INTEGER NOT NULL,
                        outline_node_id TEXT,
                        token_count INTEGER NOT NULL,
                        outline_hash TEXT,
                        provider TEXT NOT NULL,
                        dims INTEGER NOT NULL,
                        vector TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        dirty INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY (document_id, chunk_id)
                    )
                    """
                )
                self._conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_doc ON chunk_embeddings(document_id)"
                )

    def upsert_many(self, records: Sequence[ChunkEmbeddingRecord]) -> None:
        if not records:
            return
        payloads = [self._record_to_tuple(record) for record in records]
        with self._lock:
            with self._conn:
                self._conn.executemany(
                    """
                    INSERT INTO chunk_embeddings (
                        document_id, chunk_id, version_id, content_hash, chunk_hash,
                        start_offset, end_offset, outline_node_id, token_count, outline_hash,
                        provider, dims, vector, created_at, updated_at, dirty
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(document_id, chunk_id) DO UPDATE SET
                        version_id=excluded.version_id,
                        content_hash=excluded.content_hash,
                        chunk_hash=excluded.chunk_hash,
                        start_offset=excluded.start_offset,
                        end_offset=excluded.end_offset,
                        outline_node_id=excluded.outline_node_id,
                        token_count=excluded.token_count,
                        outline_hash=excluded.outline_hash,
                        provider=excluded.provider,
                        dims=excluded.dims,
                        vector=excluded.vector,
                        updated_at=excluded.updated_at,
                        dirty=excluded.dirty
                    """,
                    payloads,
                )

    def fetch_document(self, document_id: str) -> list[ChunkEmbeddingRecord]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM chunk_embeddings WHERE document_id = ? ORDER BY start_offset",
                (document_id,),
            )
            rows = cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    def prune_missing(self, document_id: str, keep_ids: set[str]) -> None:
        with self._lock:
            with self._conn:
                if not keep_ids:
                    self._conn.execute("DELETE FROM chunk_embeddings WHERE document_id = ?", (document_id,))
                    return
                placeholders = ",".join("?" for _ in keep_ids)
                params: list[Any] = [document_id, *keep_ids]
                self._conn.execute(
                    f"DELETE FROM chunk_embeddings WHERE document_id = ? AND chunk_id NOT IN ({placeholders})",
                    params,
                )

    def mark_dirty_all(self, document_id: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute(
                    "UPDATE chunk_embeddings SET dirty = 1 WHERE document_id = ?",
                    (document_id,),
                )

    def mark_dirty_ranges(self, document_id: str, ranges: Sequence[tuple[int, int]]) -> None:
        if not ranges:
            self.mark_dirty_all(document_id)
            return
        with self._lock:
            with self._conn:
                for start, end in ranges:
                    self._conn.execute(
                        """
                        UPDATE chunk_embeddings
                        SET dirty = 1
                        WHERE document_id = ?
                          AND NOT (end_offset <= ? OR start_offset >= ?)
                        """,
                        (document_id, start, end),
                    )

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM chunk_embeddings WHERE document_id = ?", (document_id,))

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover - defensive close
                LOGGER.debug("Failed to close embedding store", exc_info=True)

    def _record_to_tuple(self, record: ChunkEmbeddingRecord) -> tuple[Any, ...]:
        return (
            record.document_id,
            record.chunk_id,
            record.version_id,
            record.content_hash,
            record.chunk_hash,
            record.start_offset,
            record.end_offset,
            record.outline_node_id,
            record.token_count,
            record.outline_hash,
            record.provider,
            record.dims,
            json.dumps(record.vector),
            record.created_at,
            record.updated_at,
            1 if record.dirty else 0,
        )

    def _row_to_record(self, row: sqlite3.Row) -> ChunkEmbeddingRecord:
        vector_payload = json.loads(row["vector"])
        vector = tuple(float(value) for value in vector_payload)
        return ChunkEmbeddingRecord(
            document_id=row["document_id"],
            chunk_id=row["chunk_id"],
            version_id=row["version_id"],
            content_hash=row["content_hash"],
            chunk_hash=row["chunk_hash"],
            start_offset=row["start_offset"],
            end_offset=row["end_offset"],
            outline_node_id=row["outline_node_id"],
            token_count=row["token_count"],
            outline_hash=row["outline_hash"],
            provider=row["provider"],
            dims=row["dims"],
            vector=vector,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            dirty=bool(row["dirty"]),
        )


class DocumentEmbeddingIndex:
    """Manages chunk embeddings, caching, and retrieval per document."""

    def __init__(
        self,
        *,
        storage_dir: Path | str,
        provider: EmbeddingProvider | None,
        cache_bus: DocumentCacheBus | None = None,
        batch_size: int | None = None,
        requests_per_minute: int | None = 120,
        loop: asyncio.AbstractEventLoop | None = None,
        mode: str | None = None,
        provider_label: str | None = None,
        activity_callback: Callable[[bool, str | None], None] | None = None,
    ) -> None:
        self._provider = provider
        self._loop = loop or asyncio.get_event_loop()
        cache_root = Path(storage_dir).expanduser().resolve()
        db_path = cache_root / "embedding_cache" / "embeddings.sqlite"
        self._store = EmbeddingStore(db_path)
        self._cache_bus = cache_bus or get_document_cache_bus()
        self._rate_limiter = AsyncRateLimiter(rate_per_minute=requests_per_minute)
        provider_batch = provider.max_batch_size if provider else 8
        self._batch_size = max(1, batch_size or provider_batch)
        self._embedding_mode = mode
        self._provider_label = provider_label
        self._activity_callback = activity_callback
        self._activity_depth = 0
        self._cache_bus.subscribe(DocumentChangedEvent, self._handle_changed, weak=True)
        self._cache_bus.subscribe(DocumentClosedEvent, self._handle_closed, weak=True)

    @property
    def provider_name(self) -> str | None:
        provider = self._provider
        if provider is None:
            return None
        return getattr(provider, "name", None)

    def _telemetry_context(self) -> dict[str, Any]:
        context: dict[str, Any] = {"provider": self.provider_name}
        if self._embedding_mode:
            context["embedding_mode"] = self._embedding_mode
        if self._provider_label:
            context["provider_label"] = self._provider_label
        return context

    async def ingest_outline(
        self,
        *,
        document: DocumentState,
        nodes: Sequence[OutlineNode],
        outline_hash: str | None,
    ) -> EmbeddingIngestResult:
        if self._provider is None:
            return EmbeddingIngestResult(processed=0, embedded=0, reused=0, status="provider_unavailable")
        document_length = len(document.text or "")
        specs = self._build_chunk_specs(document, nodes)
        if not specs:
            return EmbeddingIngestResult(processed=0, embedded=0, reused=0, status="no_chunks")
        self._begin_activity(detail=f"Updating {document.document_id}")
        try:
            existing = await self._run_blocking(self._store.fetch_document, document.document_id)
            existing_map = {record.chunk_id: record for record in existing}
            dirty_specs: list[_ChunkSpec] = []
            reuse_specs: list[_ChunkSpec] = []
            for spec in specs:
                record = existing_map.get(spec.chunk_id)
                if record is None or record.chunk_hash != spec.chunk_hash or record.dirty:
                    dirty_specs.append(spec)
                else:
                    reuse_specs.append(spec)
            dirty_count = len(dirty_specs)
            reused_count = len(reuse_specs)
            reused_records = [
                existing_map[spec.chunk_id].with_metadata(
                    version_id=document.version_id,
                    content_hash=document.content_hash,
                    chunk_hash=spec.chunk_hash,
                    start_offset=spec.start,
                    end_offset=spec.end,
                    outline_node_id=spec.outline_node_id,
                    token_count=spec.token_count,
                    outline_hash=outline_hash,
                )
                for spec in reuse_specs
            ]
            if reused_records:
                await self._run_blocking(self._store.upsert_many, reused_records)
            if reused_count:
                emit(
                    "embedding.cache.hit",
                    {
                        "document_id": document.document_id,
                        "version_id": document.version_id,
                        "outline_hash": outline_hash,
                        "chunk_count": reused_count,
                        "processed_chunks": len(specs),
                        "document_length": document_length,
                        **self._telemetry_context(),
                    },
                )
            embedded_count = 0
            provider_name = self._provider.name
            now = time.time()
            for batch in _chunk_list(dirty_specs, self._batch_size):
                await self._rate_limiter.acquire(len(batch))
                vectors: Sequence[Sequence[float]]
                try:
                    vectors = await self._provider.embed_documents([spec.text for spec in batch])
                except Exception as exc:  # pragma: no cover - provider exceptions
                    LOGGER.exception("Embedding provider failed for %s: %s", document.document_id, exc)
                    emit(
                        "embedding.cache.miss",
                        {
                            "document_id": document.document_id,
                            "version_id": document.version_id,
                            "outline_hash": outline_hash,
                            "status": "provider_error",
                            "error": str(exc)[:200],
                            "embedded": embedded_count,
                            "dirty_chunks": dirty_count,
                            "document_length": document_length,
                            **self._telemetry_context(),
                        },
                    )
                    return EmbeddingIngestResult(
                        processed=len(specs),
                        embedded=embedded_count,
                        reused=len(reuse_specs),
                        status="provider_error",
                    )
                new_records = []
                for spec, vector in zip(batch, vectors):
                    normalized = tuple(float(value) for value in vector)
                    new_records.append(
                        ChunkEmbeddingRecord(
                            document_id=document.document_id,
                            chunk_id=spec.chunk_id,
                            version_id=document.version_id,
                            content_hash=document.content_hash,
                            chunk_hash=spec.chunk_hash,
                            start_offset=spec.start,
                            end_offset=spec.end,
                            outline_node_id=spec.outline_node_id,
                            token_count=spec.token_count,
                            outline_hash=outline_hash,
                            provider=provider_name,
                            dims=len(normalized),
                            vector=normalized,
                            created_at=now,
                            updated_at=time.time(),
                            dirty=False,
                        )
                    )
                await self._run_blocking(self._store.upsert_many, new_records)
                embedded_count += len(new_records)
            keep_ids = {spec.chunk_id for spec in specs}
            await self._run_blocking(self._store.prune_missing, document.document_id, keep_ids)
            if dirty_count:
                emit(
                    "embedding.cache.miss",
                    {
                        "document_id": document.document_id,
                        "version_id": document.version_id,
                        "outline_hash": outline_hash,
                        "embedded": embedded_count,
                        "dirty_chunks": dirty_count,
                        "status": "ok",
                        "document_length": document_length,
                        **self._telemetry_context(),
                    },
                )
            return EmbeddingIngestResult(
                processed=len(specs),
                embedded=embedded_count,
                reused=len(reuse_specs),
                status="ok",
            )
        finally:
            self._end_activity()

    async def similarity_search(
        self,
        document_id: str,
        *,
        query_text: str | None = None,
        query_vector: Sequence[float] | None = None,
        top_k: int = 6,
        min_score: float = 0.0,
    ) -> list[EmbeddingMatch]:
        if query_vector is None:
            if self._provider is None:
                return []
            if not query_text:
                raise ValueError("query_text is required when query_vector is not provided")
            query_vector = await self._provider.embed_query(query_text)
        candidates = await self._run_blocking(self._store.fetch_document, document_id)
        matches: list[EmbeddingMatch] = []
        query_tuple = tuple(float(value) for value in query_vector)
        for record in candidates:
            if not record.vector:
                continue
            score = _cosine_similarity(query_tuple, record.vector)
            if math.isnan(score) or score < min_score:
                continue
            matches.append(EmbeddingMatch(record=record, score=score))
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[: max(1, top_k)]

    async def get_document_embeddings(self, document_id: str) -> list[ChunkEmbeddingRecord]:
        return await self._run_blocking(self._store.fetch_document, document_id)

    async def aclose(self) -> None:
        await self._run_blocking(self._store.close)

    def _handle_changed(self, event: DocumentChangedEvent) -> None:
        ranges = tuple(event.edited_ranges)
        self._schedule_task(self._mark_dirty(event.document_id, ranges))

    def _handle_closed(self, event: DocumentClosedEvent) -> None:
        self._schedule_task(self._delete_document(event.document_id))

    async def _mark_dirty(self, document_id: str, ranges: Sequence[tuple[int, int]]) -> None:
        await self._run_blocking(self._store.mark_dirty_ranges, document_id, ranges)

    async def _delete_document(self, document_id: str) -> None:
        await self._run_blocking(self._store.delete_document, document_id)

    def _schedule_task(self, coro: Awaitable[Any]) -> None:
        if self._loop.is_closed():  # pragma: no cover - best effort guard
            return
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        future.add_done_callback(_log_future_exception)

    def _begin_activity(self, *, detail: str | None = None) -> None:
        self._activity_depth += 1
        if self._activity_depth == 1:
            self._emit_activity(True, detail)

    def _end_activity(self) -> None:
        if self._activity_depth == 0:
            return
        self._activity_depth -= 1
        if self._activity_depth == 0:
            self._emit_activity(False, None)

    def _emit_activity(self, active: bool, detail: str | None) -> None:
        callback = self._activity_callback
        if callback is None:
            return
        try:
            callback(active, detail)
        except Exception:  # pragma: no cover - activity reporting must never raise
            LOGGER.debug("Embedding activity callback failed", exc_info=True)

    def _build_chunk_specs(self, document: DocumentState, nodes: Sequence[OutlineNode]) -> list[_ChunkSpec]:
        text = document.text or ""
        length = len(text)
        specs: dict[str, _ChunkSpec] = {}
        for node in _iter_nodes(nodes):
            if node.chunk_id is None and not node.char_range:
                continue
            start, end = node.char_range
            start = max(0, min(length, start))
            end = max(start, min(length, end))
            chunk_text = text[start:end].strip()
            if not chunk_text:
                chunk_text = node.blurb or node.text or ""
            normalized = chunk_text.strip()
            if not normalized:
                continue
            chunk_id = node.chunk_id or _derive_chunk_id(document.document_id, start, end, normalized)
            chunk_hash = _chunk_hash(normalized)
            token_count = max(1, int(node.token_estimate or 0))
            if chunk_id in specs and specs[chunk_id].chunk_hash == chunk_hash:
                continue
            specs[chunk_id] = _ChunkSpec(
                chunk_id=chunk_id,
                outline_node_id=node.id,
                start=start,
                end=end,
                text=normalized,
                token_count=token_count,
                chunk_hash=chunk_hash,
            )
        return list(specs.values())

    async def _run_blocking(self, func: Callable[..., Any], *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args))


def _chunk_list(items: Sequence[_ChunkSpec], size: int) -> Iterable[list[_ChunkSpec]]:
    chunk: list[_ChunkSpec] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if not lhs or not rhs:
        return 0.0
    if len(lhs) != len(rhs):
        return 0.0
    dot = sum(a * b for a, b in zip(lhs, rhs))
    left = math.sqrt(sum(a * a for a in lhs))
    right = math.sqrt(sum(b * b for b in rhs))
    if left == 0 or right == 0:
        return 0.0
    return dot / (left * right)


def _iterate_children(node: OutlineNode) -> Iterable[OutlineNode]:
    for child in node.children:
        yield child
        if child.children:
            yield from _iterate_children(child)


def _iter_nodes(nodes: Sequence[OutlineNode]) -> Iterable[OutlineNode]:
    for node in nodes:
        yield node
        if node.children:
            yield from _iterate_children(node)


def _chunk_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _derive_chunk_id(document_id: str, start: int, end: int, text: str) -> str:
    digest = hashlib.sha1(f"{start}:{end}:{text[:32]}".encode("utf-8")).hexdigest()
    return f"{document_id}-chunk-{digest}"


def _log_future_exception(task: Future[Any]) -> None:  # pragma: no cover - debug helper
    try:
        task.result()
    except Exception as exc:
        LOGGER.debug("Embedding background task failed: %s", exc)


class LocalEmbeddingProvider:
    """Embedding provider backed by synchronous or async callables."""

    def __init__(
        self,
        *,
        embed_batch: Callable[[Sequence[str]], Sequence[Sequence[float]] | Awaitable[Sequence[Sequence[float]]]],
        embed_query: Callable[[str], Sequence[float] | Awaitable[Sequence[float]]] | None = None,
        name: str = "local",
        max_batch_size: int = 32,
    ) -> None:
        self._embed_batch = embed_batch
        self._embed_query = embed_query
        self.name = name
        self.max_batch_size = max(1, int(max_batch_size))

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return await _maybe_await(self._embed_batch(texts))

    async def embed_query(self, text: str) -> Sequence[float]:
        if self._embed_query is not None:
            return await _maybe_await(self._embed_query(text))
        vectors = await self.embed_documents([text])
        return vectors[0]


class OpenAIEmbeddingProvider:
    """Embedding provider that wraps :class:`openai.AsyncOpenAI`."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        name: str | None = None,
        max_batch_size: int = 16,
    ) -> None:
        if AsyncOpenAI is None:
            raise RuntimeError("openai package is not available")
        self._client = client
        self._model = model
        self.name = name or f"openai:{model}"
        self.max_batch_size = max(1, int(max_batch_size))

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        response = await self._client.embeddings.create(model=self._model, input=list(texts))
        data = sorted(response.data, key=lambda item: getattr(item, "index", 0))
        return [list(getattr(item, "embedding", [])) for item in data]

    async def embed_query(self, text: str) -> Sequence[float]:
        vectors = await self.embed_documents([text])
        return vectors[0]


class LangChainEmbeddingProvider:
    """Embedding provider that wraps LangChain embedding classes."""

    def __init__(
        self,
        *,
        embeddings: Any,
        name: str | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        if embeddings is None:
            raise ValueError("embeddings instance is required")
        self._embeddings = embeddings
        label = name or getattr(embeddings, "embedding_name", None) or embeddings.__class__.__name__
        self.name = f"langchain:{label}" if label and not label.startswith("langchain:") else (label or "langchain")
        inferred_batch = getattr(embeddings, "chunk_size", None)
        batch_candidate: int | None
        try:
            batch_candidate = int(inferred_batch)
        except (TypeError, ValueError):
            batch_candidate = None
        self.max_batch_size = max(1, int(max_batch_size or batch_candidate or 16))

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        inputs = list(texts)
        async_method = getattr(self._embeddings, "aembed_documents", None)
        if callable(async_method):
            raw = await async_method(inputs)
            return _normalize_vector_batch(raw)
        sync_method = getattr(self._embeddings, "embed_documents", None)
        if callable(sync_method):
            raw = await asyncio.to_thread(sync_method, inputs)
            return _normalize_vector_batch(raw)
        raise RuntimeError("LangChain embeddings implementation does not provide embed_documents().")

    async def embed_query(self, text: str) -> Sequence[float]:
        async_method = getattr(self._embeddings, "aembed_query", None)
        if callable(async_method):
            raw = await async_method(text)
            return _normalize_vector(raw)
        sync_method = getattr(self._embeddings, "embed_query", None)
        if callable(sync_method):
            raw = await asyncio.to_thread(sync_method, text)
            return _normalize_vector(raw)
        vectors = await self.embed_documents([text])
        return vectors[0]


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value) or isinstance(value, Awaitable):  # type: ignore[arg-type]
        return await value  # type: ignore[return-value]
    return value


def _normalize_vector_batch(value: Any) -> list[list[float]]:
    if not isinstance(value, Sequence):
        raise TypeError("Embedding batch must be a sequence")
    normalized: list[list[float]] = []
    for vector in value:
        normalized.append(_normalize_vector(vector))
    return normalized


def _normalize_vector(value: Any) -> list[float]:
    if isinstance(value, Sequence):
        return [float(component) for component in value]
    raise TypeError("Embedding vector must be a sequence")


__all__ = [
    "ChunkEmbeddingRecord",
    "DocumentEmbeddingIndex",
    "EmbeddingIngestResult",
    "EmbeddingMatch",
    "EmbeddingProvider",
    "EmbeddingStore",
    "LangChainEmbeddingProvider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
]
