from __future__ import annotations

from tinkerbell.ai.memory.cache_bus import DocumentCacheBus, DocumentChangedEvent
from tinkerbell.ai.memory.plot_state import DocumentPlotStateStore


def test_plot_state_store_ingest_tracks_entities_and_arcs() -> None:
    bus = DocumentCacheBus()
    store = DocumentPlotStateStore(bus=bus, max_entities=8, max_beats_per_arc=4)

    store.ingest_chunk_summary(
        "doc-plot",
        "Alice confides in Bob while Carol spies from the gallery.",
        version_id="v1",
        chunk_hash="chunk-1",
        pointer_id="pointer-123",
        metadata={"source_job_id": "job-1", "latency_ms": 250},
    )

    snapshot = store.snapshot("doc-plot")
    assert snapshot is not None
    assert snapshot["entity_count"] >= 2
    assert snapshot["arc_count"] == 1
    assert snapshot["metadata"]["stats"]["ingested_chunks"] == 1
    assert snapshot["entities"][0]["supporting_pointers"]
    assert snapshot["arcs"][0]["beats"]


def test_plot_state_store_clears_on_document_change_event() -> None:
    bus = DocumentCacheBus()
    store = DocumentPlotStateStore(bus=bus)
    store.ingest_chunk_summary(
        "doc-stale",
        "Dorian meets Evelyn in London.",
        version_id="v2",
        chunk_hash="chunk-2",
        pointer_id="pointer-456",
        metadata={},
    )
    assert store.get("doc-stale") is not None

    bus.publish(
        DocumentChangedEvent(
            document_id="doc-stale",
            version_id=3,
            content_hash="hash-new",
            edited_ranges=[(0, 10)],
        )
    )

    assert store.get("doc-stale") is None
