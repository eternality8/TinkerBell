from __future__ import annotations

from tinkerbell.ai.memory.cache_bus import DocumentCacheBus, DocumentChangedEvent
from tinkerbell.ai.memory.character_map import CharacterMapStore


def test_character_map_store_ingest_tracks_entities_and_mentions() -> None:
    bus = DocumentCacheBus()
    store = CharacterMapStore(bus=bus, max_entities=8, max_mentions_per_entity=4)

    state = store.ingest_summary(
        "doc-1",
        "Alex meets Mira in the atrium where Alex promises her help.",
        version_id="v1",
        chunk_id="chunk-123",
        pointer_id="outline-1",
        chunk_hash="hash-1",
        char_range=(10, 42),
    )

    snapshot = store.snapshot("doc-1")
    assert snapshot is not None
    assert snapshot["entity_count"] >= 1
    entity = snapshot["entities"][0]
    assert entity["name"] in {"Alex", "Mira"}
    assert entity["mentions"], "mentions should be captured for the canonical entity"
    assert entity["mentions"][0]["chunk_id"] == "chunk-123"
    assert entity["mentions"][0]["pointer_id"] == "outline-1"
    assert "mention_count" in entity


def test_character_map_store_enforces_entity_limit() -> None:
    store = CharacterMapStore(max_entities=1)
    store.ingest_summary(
        "doc-2",
        "Alice greets Bob in the courtyard.",
        version_id="v1",
        chunk_id="chunk-A",
        pointer_id=None,
        chunk_hash="hash-A",
    )
    store.ingest_summary(
        "doc-2",
        "Charlie chastises Dana for missing rehearsal.",
        version_id="v1",
        chunk_id="chunk-B",
        pointer_id=None,
        chunk_hash="hash-B",
    )

    snapshot = store.snapshot("doc-2")
    assert snapshot is not None
    assert snapshot["entity_count"] == 1
    only_entity = snapshot["entities"][0]
    assert only_entity["name"] in {"Charlie", "Dana"}


def test_character_map_store_clears_on_document_change_event() -> None:
    bus = DocumentCacheBus()
    store = CharacterMapStore(bus=bus)
    store.ingest_summary(
        "doc-3",
        "Evelyn thanks Frank.",
        version_id="v1",
        chunk_id="chunk-1",
        pointer_id="outline-2",
        chunk_hash="hash-2",
    )

    bus.publish(
        DocumentChangedEvent(
            document_id="doc-3",
            version_id=2,
            content_hash="new-hash",
            edited_ranges=[(0, 5)],
        )
    )

    assert store.snapshot("doc-3") is None


def test_character_map_store_tracks_planner_progress() -> None:
    store = CharacterMapStore()
    state = store.ingest_summary(
        "doc-plan",
        "Gideon tells Hazel a secret while Iris listens nearby.",
        version_id="v5",
        chunk_id="chunk-plan",
        pointer_id="outline-plan",
        chunk_hash="hash-plan",
    )
    mention_id = next(iter(state.entities.values())).mentions[0].mention_id

    applied = store.update_planner_progress(
        "doc-plan",
        [
            {
                "task_id": mention_id,
                "status": "in_progress",
                "note": "drafted new banter",
            }
        ],
    )

    assert applied == 1
    snapshot = store.snapshot("doc-plan")
    assert snapshot is not None
    planner = snapshot["planner_progress"]
    assert planner["pending"] == 1
    task_entry = planner["tasks"][0]
    assert task_entry["task_id"] == mention_id
    assert task_entry["status"] == "in_progress"
    assert task_entry["note"] == "drafted new banter"
