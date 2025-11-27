"""Tests for the analysis memory adapter."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tinkerbell.ai.memory.analysis_adapter import (
    AnalysisMemoryAdapter,
    AnalysisResult,
    AnalysisResultCache,
)
from tinkerbell.ai.memory.character_map import CharacterMapStore
from tinkerbell.ai.memory.plot_state import DocumentPlotStateStore


# =============================================================================
# AnalysisResult Tests
# =============================================================================


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_from_tool_output_extracts_characters(self) -> None:
        output = {
            "characters": [
                {"name": "Alice", "role": "protagonist", "traits": ["brave", "curious"]},
                {"name": "Bob", "role": "antagonist", "traits": ["cunning"]},
            ],
            "version_id": "v1",
        }
        result = AnalysisResult.from_tool_output("doc-1", output)
        
        assert result.document_id == "doc-1"
        assert result.version_id == "v1"
        assert len(result.characters) == 2
        assert result.characters[0]["name"] == "Alice"

    def test_from_tool_output_extracts_plot_points(self) -> None:
        output = {
            "plot_points": [
                {"type": "event", "summary": "The beginning", "significance": "high"},
                {"type": "conflict", "summary": "The struggle", "significance": "medium"},
            ],
        }
        result = AnalysisResult.from_tool_output("doc-2", output)
        
        assert len(result.plot_points) == 2
        assert result.plot_points[0]["type"] == "event"

    def test_from_tool_output_extracts_themes(self) -> None:
        output = {
            "themes": ["love", "betrayal", "redemption"],
        }
        result = AnalysisResult.from_tool_output("doc-3", output)
        
        assert result.themes == ["love", "betrayal", "redemption"]

    def test_from_tool_output_extracts_summary(self) -> None:
        output = {
            "summary": "A tale of two cities.",
            "chunks_processed": 5,
        }
        result = AnalysisResult.from_tool_output("doc-4", output)
        
        assert result.summary == "A tale of two cities."
        assert result.chunks_processed == 5

    def test_from_tool_output_handles_empty_output(self) -> None:
        result = AnalysisResult.from_tool_output("doc-5", {})
        
        assert result.document_id == "doc-5"
        assert result.characters == []
        assert result.plot_points == []
        assert result.themes == []
        assert result.summary is None

    def test_generated_at_is_set(self) -> None:
        result = AnalysisResult.from_tool_output("doc-6", {})
        
        assert result.generated_at is not None
        assert isinstance(result.generated_at, datetime)


# =============================================================================
# AnalysisMemoryAdapter Tests
# =============================================================================


class TestAnalysisMemoryAdapter:
    """Tests for AnalysisMemoryAdapter."""

    def test_ingest_to_plot_store_creates_entities(self) -> None:
        plot_store = DocumentPlotStateStore()
        adapter = AnalysisMemoryAdapter(plot_store=plot_store)
        
        result = AnalysisResult(
            document_id="doc-1",
            version_id="v1",
            characters=[
                {"name": "Alice", "role": "protagonist", "traits": ["brave"]},
                {"name": "Bob", "role": "supporting"},
            ],
        )
        
        stats = adapter.ingest_analysis(result)
        
        assert stats["entities_ingested"] == 2
        state = plot_store.get("doc-1")
        assert state is not None
        assert len(state.entities) == 2
        names = {e.name for e in state.entities}
        assert "Alice" in names
        assert "Bob" in names

    def test_ingest_to_plot_store_creates_beats(self) -> None:
        plot_store = DocumentPlotStateStore()
        adapter = AnalysisMemoryAdapter(plot_store=plot_store)
        
        result = AnalysisResult(
            document_id="doc-2",
            plot_points=[
                {"type": "event", "summary": "Chapter begins"},
                {"type": "conflict", "summary": "Battle ensues"},
            ],
        )
        
        stats = adapter.ingest_analysis(result)
        
        assert stats["plot_points_ingested"] == 2
        state = plot_store.get("doc-2")
        assert state is not None
        assert len(state.arcs) == 2  # event and conflict arcs

    def test_ingest_to_plot_store_stores_themes(self) -> None:
        plot_store = DocumentPlotStateStore()
        adapter = AnalysisMemoryAdapter(plot_store=plot_store)
        
        result = AnalysisResult(
            document_id="doc-3",
            themes=["love", "loss"],
        )
        
        adapter.ingest_analysis(result)
        
        state = plot_store.get("doc-3")
        assert state is not None
        assert state.metadata.get("themes") == ["love", "loss"]

    def test_ingest_to_character_store(self) -> None:
        character_store = CharacterMapStore()
        adapter = AnalysisMemoryAdapter(character_store=character_store)
        
        result = AnalysisResult(
            document_id="doc-4",
            version_id="v1",
            characters=[
                {"name": "Charlie", "role": "protagonist", "aliases": ["Chuck"]},
            ],
        )
        
        stats = adapter.ingest_analysis(result)
        
        assert stats["characters_ingested"] == 1
        snapshot = character_store.snapshot("doc-4")
        assert snapshot is not None
        assert snapshot["entity_count"] >= 1

    def test_ingest_to_both_stores(self) -> None:
        plot_store = DocumentPlotStateStore()
        character_store = CharacterMapStore()
        adapter = AnalysisMemoryAdapter(
            plot_store=plot_store,
            character_store=character_store,
        )
        
        result = AnalysisResult(
            document_id="doc-5",
            characters=[{"name": "Dana", "role": "lead"}],
            plot_points=[{"summary": "Opening scene"}],
        )
        
        stats = adapter.ingest_analysis(result)
        
        assert stats["entities_ingested"] == 1
        assert stats["characters_ingested"] == 1
        assert stats["plot_points_ingested"] == 1

    def test_clear_clears_stores(self) -> None:
        plot_store = DocumentPlotStateStore()
        character_store = CharacterMapStore()
        adapter = AnalysisMemoryAdapter(
            plot_store=plot_store,
            character_store=character_store,
        )
        
        result = AnalysisResult(
            document_id="doc-6",
            characters=[{"name": "Eve"}],
        )
        adapter.ingest_analysis(result)
        
        adapter.clear("doc-6")
        
        assert plot_store.get("doc-6") is None
        assert character_store.snapshot("doc-6") is None

    def test_handles_empty_character_names(self) -> None:
        plot_store = DocumentPlotStateStore()
        adapter = AnalysisMemoryAdapter(plot_store=plot_store)
        
        result = AnalysisResult(
            document_id="doc-7",
            characters=[
                {"name": "", "role": "unknown"},  # Empty name
                {"name": "  ", "role": "unknown"},  # Whitespace name
                {"name": "Valid", "role": "lead"},
            ],
        )
        
        stats = adapter.ingest_analysis(result)
        
        assert stats["entities_ingested"] == 1  # Only Valid


# =============================================================================
# AnalysisResultCache Tests
# =============================================================================


class TestAnalysisResultCache:
    """Tests for AnalysisResultCache."""

    def test_store_and_get(self) -> None:
        cache = AnalysisResultCache()
        result = AnalysisResult(document_id="doc-1", summary="Test")
        
        cache.store(result)
        retrieved = cache.get("doc-1")
        
        assert retrieved is not None
        assert retrieved.summary == "Test"

    def test_get_returns_none_for_missing(self) -> None:
        cache = AnalysisResultCache()
        
        retrieved = cache.get("nonexistent")
        
        assert retrieved is None

    def test_clear_specific_document(self) -> None:
        cache = AnalysisResultCache()
        cache.store(AnalysisResult(document_id="doc-1"))
        cache.store(AnalysisResult(document_id="doc-2"))
        
        cache.clear("doc-1")
        
        assert cache.get("doc-1") is None
        assert cache.get("doc-2") is not None

    def test_clear_all(self) -> None:
        cache = AnalysisResultCache()
        cache.store(AnalysisResult(document_id="doc-1"))
        cache.store(AnalysisResult(document_id="doc-2"))
        
        cache.clear()
        
        assert cache.get("doc-1") is None
        assert cache.get("doc-2") is None

    def test_enforces_max_entries(self) -> None:
        cache = AnalysisResultCache(max_entries=2)
        
        cache.store(AnalysisResult(document_id="doc-1", generated_at=datetime(2025, 1, 1, tzinfo=timezone.utc)))
        cache.store(AnalysisResult(document_id="doc-2", generated_at=datetime(2025, 1, 2, tzinfo=timezone.utc)))
        cache.store(AnalysisResult(document_id="doc-3", generated_at=datetime(2025, 1, 3, tzinfo=timezone.utc)))
        
        # Oldest entry should be evicted
        assert cache.get("doc-1") is None
        assert cache.get("doc-2") is not None
        assert cache.get("doc-3") is not None

    def test_snapshot_returns_formatted_data(self) -> None:
        cache = AnalysisResultCache()
        result = AnalysisResult(
            document_id="doc-1",
            version_id="v1",
            characters=[{"name": "Alice"}],
            themes=["adventure"],
            summary="A story",
            chunks_processed=3,
        )
        cache.store(result)
        
        snapshot = cache.snapshot("doc-1")
        
        assert snapshot is not None
        assert snapshot["document_id"] == "doc-1"
        assert snapshot["version_id"] == "v1"
        assert snapshot["characters"] == [{"name": "Alice"}]
        assert snapshot["themes"] == ["adventure"]
        assert snapshot["summary"] == "A story"
        assert snapshot["chunks_processed"] == 3
        assert snapshot["source"] == "analyze_document"

    def test_snapshot_returns_none_for_missing(self) -> None:
        cache = AnalysisResultCache()
        
        snapshot = cache.snapshot("nonexistent")
        
        assert snapshot is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestAnalysisAdapterIntegration:
    """Integration tests for the analysis adapter system."""

    def test_full_workflow(self) -> None:
        """Test complete analysis ingestion and retrieval workflow."""
        # Create stores
        plot_store = DocumentPlotStateStore()
        character_store = CharacterMapStore()
        cache = AnalysisResultCache()
        adapter = AnalysisMemoryAdapter(
            plot_store=plot_store,
            character_store=character_store,
        )
        
        # Simulate analyze_document tool output
        tool_output = {
            "status": "complete",
            "characters": [
                {
                    "name": "Sherlock Holmes",
                    "role": "protagonist",
                    "aliases": ["Holmes"],
                    "traits": ["observant", "brilliant", "eccentric"],
                },
                {
                    "name": "Dr. Watson",
                    "role": "supporting",
                    "aliases": ["Watson", "John"],
                    "traits": ["loyal", "practical"],
                },
            ],
            "plot_points": [
                {
                    "type": "event",
                    "summary": "A mysterious client arrives",
                    "significance": "high",
                    "characters_involved": ["Sherlock Holmes", "Dr. Watson"],
                },
            ],
            "themes": ["mystery", "friendship", "deduction"],
            "summary": "A tale of mystery and deduction.",
            "chunks_processed": 10,
            "version_id": "v42",
        }
        
        # Create result and ingest
        result = AnalysisResult.from_tool_output("doc-mystery", tool_output)
        stats = adapter.ingest_analysis(result)
        cache.store(result)
        
        # Verify ingestion
        assert stats["entities_ingested"] == 2
        assert stats["plot_points_ingested"] == 1
        assert stats["characters_ingested"] == 2
        
        # Verify plot store
        plot_state = plot_store.get("doc-mystery")
        assert plot_state is not None
        assert len(plot_state.entities) == 2
        assert plot_state.metadata.get("themes") == ["mystery", "friendship", "deduction"]
        
        # Verify character store
        char_snapshot = character_store.snapshot("doc-mystery")
        assert char_snapshot is not None
        assert char_snapshot["entity_count"] >= 2
        
        # Verify cache
        cache_snapshot = cache.snapshot("doc-mystery")
        assert cache_snapshot is not None
        assert cache_snapshot["summary"] == "A tale of mystery and deduction."
        assert cache_snapshot["chunks_processed"] == 10
