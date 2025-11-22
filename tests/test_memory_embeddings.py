"""Tests for the embedding index service."""

from __future__ import annotations

import asyncio
from typing import Sequence

import pytest

from tinkerbell.ai.memory.buffers import OutlineNode
from tinkerbell.ai.memory.cache_bus import DocumentCacheBus, DocumentChangedEvent
from tinkerbell.ai.memory.embeddings import DocumentEmbeddingIndex, LangChainEmbeddingProvider
from tinkerbell.editor.document_model import DocumentState
from tinkerbell.services import telemetry as telemetry_service


@pytest.mark.asyncio
async def test_embedding_index_creates_embeddings(tmp_path) -> None:
    provider = _DeterministicProvider()
    index = DocumentEmbeddingIndex(
        storage_dir=tmp_path,
        provider=provider,
        loop=asyncio.get_running_loop(),
    )
    document = _build_document("doc-embed", _SOURCE_TEXT)
    nodes = _build_nodes(document)

    result = await index.ingest_outline(document=document, nodes=nodes, outline_hash="outline-v1")

    assert result.embedded == len(nodes)
    assert result.reused == 0
    assert provider.document_call_count == len(nodes)

    records = await index.get_document_embeddings(document.document_id)
    assert len(records) == len(nodes)
    assert all(record.outline_hash == "outline-v1" for record in records)
    assert all(not record.dirty for record in records)

    await index.aclose()


@pytest.mark.asyncio
async def test_embedding_index_emits_activity_events(tmp_path) -> None:
    provider = _DeterministicProvider()
    events: list[tuple[bool, str | None]] = []
    index = DocumentEmbeddingIndex(
        storage_dir=tmp_path,
        provider=provider,
        loop=asyncio.get_running_loop(),
        activity_callback=lambda active, detail: events.append((active, detail)),
    )
    document = _build_document("doc-activity", _SOURCE_TEXT)
    nodes = _build_nodes(document)

    await index.ingest_outline(document=document, nodes=nodes, outline_hash="outline-v1")

    assert events
    assert events[0][0] is True
    assert events[-1][0] is False
    assert any(detail and "doc-activity" in detail for _, detail in events if detail)

    await index.aclose()


@pytest.mark.asyncio
async def test_embedding_index_reuses_cached_chunks(tmp_path) -> None:
    provider = _DeterministicProvider()
    index = DocumentEmbeddingIndex(
        storage_dir=tmp_path,
        provider=provider,
        loop=asyncio.get_running_loop(),
    )
    document = _build_document("doc-cache", _SOURCE_TEXT)
    nodes = _build_nodes(document)
    await index.ingest_outline(document=document, nodes=nodes, outline_hash="outline-v1")

    document_b = _build_document(
        document.document_id,
        _SOURCE_TEXT,
        version_id=document.version_id + 1,
        content_hash=document.content_hash,
    )
    result = await index.ingest_outline(document=document_b, nodes=nodes, outline_hash="outline-v2")

    assert result.embedded == 0
    assert result.reused == len(nodes)
    assert provider.document_call_count == len(nodes)

    await index.aclose()


@pytest.mark.asyncio
async def test_embedding_index_reembeds_dirty_chunks(tmp_path) -> None:
    provider = _DeterministicProvider()
    index = DocumentEmbeddingIndex(
        storage_dir=tmp_path,
        provider=provider,
        loop=asyncio.get_running_loop(),
    )
    document = _build_document("doc-dirty", _SOURCE_TEXT)
    nodes = _build_nodes(document)
    await index.ingest_outline(document=document, nodes=nodes, outline_hash="outline-v1")

    updated_text = _SOURCE_TEXT.replace("Delta", "Delta updated")
    document_new = _build_document(
        document.document_id,
        updated_text,
        version_id=document.version_id + 1,
    )
    updated_nodes = _build_nodes(document_new)
    result = await index.ingest_outline(document=document_new, nodes=updated_nodes, outline_hash="outline-v2")

    assert result.embedded == 1
    assert result.reused == len(nodes) - 1
    assert provider.document_call_count == len(nodes) + 1

    await index.aclose()


@pytest.mark.asyncio
async def test_embedding_index_marks_dirty_from_cache_bus(tmp_path) -> None:
    provider = _DeterministicProvider()
    bus = DocumentCacheBus()
    index = DocumentEmbeddingIndex(
        storage_dir=tmp_path,
        provider=provider,
        cache_bus=bus,
        loop=asyncio.get_running_loop(),
    )
    document = _build_document("doc-events", _SOURCE_TEXT)
    nodes = _build_nodes(document)
    await index.ingest_outline(document=document, nodes=nodes, outline_hash="outline-v1")

    change_event = DocumentChangedEvent(
        document_id=document.document_id,
        version_id=document.version_id + 1,
        content_hash=document.content_hash,
        edited_ranges=[(0, 5)],
    )
    bus.publish(change_event)
    await asyncio.sleep(0)  # allow dirty task to run

    dirty_found = False
    for _ in range(10):
        records = await index.get_document_embeddings(document.document_id)
        if any(record.dirty for record in records):
            dirty_found = True
            break
        await asyncio.sleep(0.01)
    assert dirty_found

    updated_document = _build_document(
        document.document_id,
        document.text,
        version_id=document.version_id + 1,
        content_hash=document.content_hash,
    )
    result = await index.ingest_outline(document=updated_document, nodes=nodes, outline_hash="outline-v2")
    assert result.embedded == 1

    await index.aclose()


@pytest.mark.asyncio
async def test_embedding_index_similarity_search(tmp_path) -> None:
    provider = _DeterministicProvider()
    index = DocumentEmbeddingIndex(
        storage_dir=tmp_path,
        provider=provider,
        loop=asyncio.get_running_loop(),
    )
    document = _build_document("doc-search", _SOURCE_TEXT)
    nodes = _build_nodes(document)
    await index.ingest_outline(document=document, nodes=nodes, outline_hash="outline-v1")

    matches = await index.similarity_search(document.document_id, query_text="Delta content")
    assert matches
    top = matches[0]
    assert top.record.chunk_id.endswith("chunk-1")
    assert top.score > 0

    await index.aclose()


@pytest.mark.asyncio
async def test_embedding_index_emits_cache_events(tmp_path) -> None:
    provider = _DeterministicProvider()
    index = DocumentEmbeddingIndex(
        storage_dir=tmp_path,
        provider=provider,
        loop=asyncio.get_running_loop(),
    )
    hits: list[dict[str, object]] = []
    misses: list[dict[str, object]] = []
    telemetry_service.register_event_listener("embedding.cache.hit", lambda payload: hits.append(payload))
    telemetry_service.register_event_listener("embedding.cache.miss", lambda payload: misses.append(payload))

    document = _build_document("doc-telemetry", _SOURCE_TEXT)
    nodes = _build_nodes(document)

    await index.ingest_outline(document=document, nodes=nodes, outline_hash="outline-v1")
    assert misses
    miss_payload = misses[-1]
    assert miss_payload["document_id"] == document.document_id
    assert miss_payload["embedded"] == len(nodes)

    updated_document = _build_document(
        document.document_id,
        document.text,
        version_id=document.version_id + 1,
        content_hash=document.content_hash,
    )
    await index.ingest_outline(document=updated_document, nodes=nodes, outline_hash="outline-v2")
    assert hits
    hit_payload = hits[-1]
    assert hit_payload["chunk_count"] == len(nodes)

    await index.aclose()


@pytest.mark.asyncio
async def test_langchain_provider_wraps_sync_embeddings() -> None:
    class _FakeEmbeddings:
        def __init__(self) -> None:
            self.document_calls: list[list[str]] = []
            self.query_calls: list[str] = []

        def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
            self.document_calls.append(list(texts))
            return [[float(len(text))] for text in texts]

        def embed_query(self, text: str) -> Sequence[float]:
            self.query_calls.append(text)
            return [float(len(text))]

    fake = _FakeEmbeddings()
    provider = LangChainEmbeddingProvider(embeddings=fake, name="langchain:test", max_batch_size=4)

    vectors = await provider.embed_documents(["alpha", "beta"])
    assert vectors == [[5.0], [4.0]]
    assert fake.document_calls[-1] == ["alpha", "beta"]

    query_vector = await provider.embed_query("gamma")
    assert query_vector == [5.0]
    assert fake.query_calls[-1] == "gamma"
    assert provider.max_batch_size == 4


class _DeterministicProvider:
    """Deterministic embedding provider used for tests."""

    name = "deterministic"
    max_batch_size = 8

    def __init__(self) -> None:
        self.document_call_count = 0
        self.query_call_count = 0

    async def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.document_call_count += len(texts)
        return [self._vectorize(text) for text in texts]

    async def embed_query(self, text: str) -> Sequence[float]:
        self.query_call_count += 1
        return self._vectorize(text)

    def _vectorize(self, text: str) -> list[float]:
        normalized = text.lower()
        return [
            1.0 if "alpha" in normalized else 0.0,
            1.0 if "delta" in normalized else 0.0,
            float(len(normalized)),
        ]


def _build_document(
    document_id: str,
    text: str,
    *,
    version_id: int | None = None,
    content_hash: str | None = None,
) -> DocumentState:
    state = DocumentState(text=text, document_id=document_id)
    if version_id is not None:
        state.version_id = version_id
    if content_hash is not None:
        state.content_hash = content_hash
    return state


def _build_nodes(document: DocumentState) -> list[OutlineNode]:
    text = document.text
    second_heading = text.index("## Details")
    first_chunk = OutlineNode(
        id="node-0",
        parent_id=None,
        level=1,
        text="Intro",
        char_range=(0, second_heading),
        chunk_id=f"{document.document_id}-chunk-0",
        blurb="Intro section",
        token_estimate=12,
    )
    second_chunk = OutlineNode(
        id="node-1",
        parent_id=None,
        level=1,
        text="Details",
        char_range=(second_heading, len(text)),
        chunk_id=f"{document.document_id}-chunk-1",
        blurb="Delta section",
        token_estimate=14,
    )
    return [first_chunk, second_chunk]


_SOURCE_TEXT = """## Intro
Alpha bravo charlie
## Details
Delta echo foxtrot
"""
