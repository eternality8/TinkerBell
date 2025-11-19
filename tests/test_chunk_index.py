"""Unit tests for the ChunkIndex service."""

from __future__ import annotations

from tinkerbell.ai.memory.cache_bus import DocumentCacheBus, DocumentChangedEvent
from tinkerbell.ai.memory.chunk_index import ChunkIndex


def _sample_manifest(document_id: str, cache_key: str, ranges: list[tuple[int, int]]):
    chunks = []
    for idx, (start, end) in enumerate(ranges):
        chunks.append(
            {
                "id": f"chunk:{document_id}:{start}:{end}",
                "start": start,
                "end": end,
                "length": end - start,
                "hash": f"hash-{idx}",
                "selection_overlap": False,
                "outline_pointer_id": None,
            }
        )
    return {
        "document_id": document_id,
        "chunk_profile": "auto",
        "chunk_chars": 2_000,
        "chunk_overlap": 200,
        "window": {"start": 0, "end": ranges[-1][1], "length": ranges[-1][1]},
        "chunks": chunks,
        "cache_key": cache_key,
        "version": f"{document_id}:1:hash",
        "content_hash": "hash",
        "generated_at": 1.0,
    }


def test_chunk_index_ingest_and_lookup_returns_entries():
    bus = DocumentCacheBus()
    index = ChunkIndex(bus=bus)
    manifest = _sample_manifest("doc-1", "cache-a", [(0, 10), (10, 20)])

    entries = index.ingest_manifest(manifest)

    assert len(entries) == 2
    chunk = index.get_chunk(entries[0].chunk_id, document_id="doc-1", cache_key="cache-a")
    assert chunk is not None
    assert chunk.start == 0
    assert chunk.end == 10


def test_chunk_index_iter_chunks_preserves_order_and_limit():
    index = ChunkIndex()
    manifest = _sample_manifest("doc-iter", "cache-iter", [(0, 5), (5, 15), (15, 25)])
    index.ingest_manifest(manifest)

    selected = index.iter_chunks("cache-iter", limit=2)

    assert [entry.chunk_id for entry in selected] == [f"chunk:doc-iter:0:5", f"chunk:doc-iter:5:15"]


def test_chunk_index_evicted_when_document_changes():
    bus = DocumentCacheBus()
    index = ChunkIndex(bus=bus)
    manifest = _sample_manifest("doc-evict", "cache-z", [(0, 8)])
    entry = index.ingest_manifest(manifest)[0]

    bus.publish(
        DocumentChangedEvent(
            document_id="doc-evict",
            version_id=2,
            content_hash="new-hash",
            edited_ranges=[(0, 1)],
        )
    )

    assert index.get_chunk(entry.chunk_id, document_id="doc-evict") is None
