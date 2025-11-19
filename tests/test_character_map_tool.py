from __future__ import annotations

from tinkerbell.ai.memory.cache_bus import DocumentCacheBus
from tinkerbell.ai.memory.character_map import CharacterMapStore
from tinkerbell.ai.tools.character_map import CharacterMapTool
from tinkerbell.editor.document_model import DocumentState


def _build_store_with_doc(document_id: str) -> CharacterMapStore:
    bus = DocumentCacheBus()
    store = CharacterMapStore(bus=bus)
    store.ingest_summary(
        document_id,
        "Alice meets Bob in the archive while Dana listens for clues.",
        version_id="v1",
        chunk_id="chunk-1",
        pointer_id="outline-1",
        chunk_hash="hash-1",
        char_range=(12, 32),
    )
    return store


def test_character_map_tool_returns_payload_for_active_document() -> None:
    document_id = "doc-active"
    store = _build_store_with_doc(document_id)
    doc = DocumentState(text="Sample", document_id=document_id)
    tool = CharacterMapTool(
        character_map_resolver=lambda: store,
        active_document_provider=lambda: doc,
        feature_enabled=lambda: True,
    )

    payload = tool.run(max_entities=1, max_mentions=1)

    assert payload["status"] == "ok"
    assert payload["document_id"] == document_id
    assert payload["entities"]
    assert payload["entities"][0]["mentions"], "mentions should be present when include_mentions is True"


def test_character_map_tool_accepts_explicit_document_id_and_can_trim_mentions() -> None:
    store = _build_store_with_doc("doc-explicit")
    tool = CharacterMapTool(
        character_map_resolver=lambda: store,
        active_document_provider=lambda: None,
        feature_enabled=lambda: True,
    )

    payload = tool.run(document_id="doc-explicit", include_mentions=False, include_stats=True)

    assert payload["status"] == "ok"
    assert payload["entities"] and "mentions" not in payload["entities"][0]
    assert "stats" in payload and payload["character_map_available"] is True


def test_character_map_tool_handles_disabled_flag() -> None:
    tool = CharacterMapTool(
        character_map_resolver=lambda: None,
        feature_enabled=lambda: False,
    )

    result = tool.run()

    assert result["status"] == "character_map_disabled"


def test_character_map_tool_reports_missing_store() -> None:
    tool = CharacterMapTool(
        character_map_resolver=lambda: None,
        active_document_provider=lambda: DocumentState(text="", document_id="doc-missing"),
        feature_enabled=lambda: True,
    )

    result = tool.run()

    assert result["status"] == "character_map_unavailable"


def test_character_map_tool_reports_missing_document_id() -> None:
    store = _build_store_with_doc("doc-has-store")
    tool = CharacterMapTool(
        character_map_resolver=lambda: store,
        active_document_provider=lambda: None,
        feature_enabled=lambda: True,
    )

    result = tool.run()

    assert result["status"] == "no_document"


def test_character_map_tool_reports_when_no_snapshot_exists() -> None:
    store = CharacterMapStore(bus=DocumentCacheBus())
    tool = CharacterMapTool(
        character_map_resolver=lambda: store,
        active_document_provider=lambda: DocumentState(text="", document_id="doc-empty"),
        feature_enabled=lambda: True,
    )

    result = tool.run()

    assert result["status"] == "no_character_map"
    assert result["character_map_available"] is False
