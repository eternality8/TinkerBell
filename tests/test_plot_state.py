from __future__ import annotations

from pathlib import Path

from tinkerbell.ai.memory.cache_bus import DocumentCacheBus, DocumentChangedEvent
from tinkerbell.ai.memory.plot_memory import PlotOverrideStore, PlotStateMemory
from tinkerbell.ai.memory.plot_state import DocumentPlotStateStore


def _override_store(tmp_path: Path) -> PlotOverrideStore:
    return PlotOverrideStore(path=tmp_path / "overrides.json")


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


def test_plot_state_memory_includes_overrides_and_dependencies(tmp_path) -> None:
    store = PlotStateMemory(override_store=_override_store(tmp_path))
    store.ingest_chunk_summary(
        "doc-plot",
        "Alice confides in Bob while Carol spies from the gallery.",
        version_id="v1",
        chunk_hash="chunk-1",
        pointer_id="pointer-123",
        metadata={},
    )

    store.apply_manual_update(
        "doc-plot",
        arcs=[
            {
                "arc_id": "arc-rivalry",
                "summary": "Alice vs Carol arc",
                "beats": [
                    {"beat_id": "beat-1", "summary": "Alice senses the betrayal", "pointer_id": "pointer-123"}
                ],
            }
        ],
        overrides=[{"override_id": "keep-ending", "summary": "Keep original ending tone", "arc_id": "arc-rivalry"}],
        dependencies=[
            {
                "source_arc_id": "arc-rivalry",
                "target_arc_id": "arc-resolution",
                "kind": "blocks",
                "summary": "Resolve rivalry before epilogue",
            }
        ],
    )

    snapshot = store.snapshot_enriched("doc-plot")
    assert snapshot is not None
    assert snapshot["overrides"][0]["summary"].startswith("Keep original")
    assert snapshot["dependencies"][0]["source_arc_id"] == "arc-rivalry"


def test_plot_state_memory_persists_overrides(tmp_path) -> None:
    path = tmp_path / "persisted_overrides.json"
    first = PlotStateMemory(override_store=PlotOverrideStore(path=path))
    first.apply_manual_update(
        "doc-plot",
        overrides=[{"override_id": "manual", "summary": "Operator override"}],
    )

    second = PlotStateMemory(override_store=PlotOverrideStore(path=path))
    overrides = second.list_overrides("doc-plot")
    assert any(override.override_id == "manual" for override in overrides)
