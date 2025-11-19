from __future__ import annotations

from tinkerbell.ai.memory.cache_bus import DocumentCacheBus
from tinkerbell.ai.memory.character_map import CharacterMapStore
from tinkerbell.ai.tools.character_edit_planner import CharacterEditPlannerTool
from tinkerbell.editor.document_model import DocumentState


def _build_store_with_entities(document_id: str) -> tuple[CharacterMapStore, str]:
    bus = DocumentCacheBus()
    store = CharacterMapStore(bus=bus)
    state = store.ingest_summary(
        document_id,
        "Talia greets Mira and then reunites with Kieran in the archive.",
        version_id="v2",
        chunk_id="chunk-42",
        pointer_id="outline:characters:talia",
        chunk_hash="hash-42",
    )
    entity = next(iter(state.entities.values()))
    mention_id = entity.mentions[0].mention_id
    return store, mention_id


def test_character_edit_planner_tool_returns_tasks_and_updates_progress() -> None:
    doc_id = "doc-plan"
    store, mention_id = _build_store_with_entities(doc_id)
    tool = CharacterEditPlannerTool(
        character_map_resolver=lambda: store,
        active_document_provider=lambda: DocumentState(text="", document_id=doc_id),
        feature_enabled=lambda: True,
    )

    first_payload = tool.run(max_tasks=5)

    assert first_payload["status"] == "ok"
    assert first_payload["tasks"], "planner should return tasks"
    assert first_payload["next_task"]["task_id"] == mention_id

    updated_payload = tool.run(completed_task_ids=[mention_id], notes={mention_id: "updated scene"})

    assert updated_payload["applied_updates"] == 1
    assert updated_payload["tasks"][0]["status"] == "completed"
    assert updated_payload["tasks"][0]["note"] == "updated scene"


def test_character_edit_planner_tool_filters_targets() -> None:
    doc_id = "doc-plan-target"
    store, _ = _build_store_with_entities(doc_id)
    # Add a second entity so filters can exclude it.
    store.ingest_summary(
        doc_id,
        "Lena whispers to Mira about Gideon.",
        version_id="v2",
        chunk_id="chunk-43",
        pointer_id="outline:lena",
        chunk_hash="hash-43",
    )
    tool = CharacterEditPlannerTool(
        character_map_resolver=lambda: store,
        active_document_provider=lambda: DocumentState(text="", document_id=doc_id),
        feature_enabled=lambda: True,
    )

    payload = tool.run(targets=["lena"], include_completed=True)

    assert payload["status"] == "ok"
    assert all(task["entity_name"].lower() == "lena" for task in payload["tasks"])


def test_character_edit_planner_tool_handles_disabled_flag() -> None:
    tool = CharacterEditPlannerTool(
        character_map_resolver=lambda: None,
        feature_enabled=lambda: False,
    )

    result = tool.run()

    assert result["status"] == "character_planner_disabled"
