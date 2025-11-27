"""Adapter to populate memory stores from analyze_document results.

This module bridges the gap between the new analyze_document tool output
and the legacy memory stores (DocumentPlotStateStore, CharacterMapStore).
It allows the Document Status panel to display analysis results from
either the legacy ingestion path or the new tool-based analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from .character_map import CharacterMapStore, CharacterMapDocument
from .plot_state import DocumentPlotStateStore, DocumentPlotState

LOGGER = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class AnalysisResult:
    """Normalized analysis result from analyze_document tool.
    
    This dataclass provides a consistent interface for analysis results
    regardless of whether they came from the subagent infrastructure
    or were provided directly.
    """

    document_id: str
    version_id: str | None = None
    characters: list[dict[str, Any]] | None = None
    plot_points: list[dict[str, Any]] | None = None
    themes: list[str] | None = None
    summary: str | None = None
    chunks_processed: int = 0
    generated_at: datetime | None = None

    @classmethod
    def from_tool_output(cls, document_id: str, output: Mapping[str, Any]) -> "AnalysisResult":
        """Create an AnalysisResult from analyze_document tool output."""
        return cls(
            document_id=document_id,
            version_id=output.get("version_id"),
            characters=list(output.get("characters") or []),
            plot_points=list(output.get("plot_points") or []),
            themes=list(output.get("themes") or []),
            summary=output.get("summary"),
            chunks_processed=int(output.get("chunks_processed", 0)),
            generated_at=_utcnow(),
        )


class AnalysisMemoryAdapter:
    """Adapts analyze_document results to populate memory stores.
    
    This adapter allows the Document Status panel to display analysis
    results from the new tool system while maintaining backward
    compatibility with the existing memory store interfaces.
    
    Usage:
        adapter = AnalysisMemoryAdapter(plot_store, character_store)
        
        # When analyze_document completes:
        result = AnalysisResult.from_tool_output(doc_id, tool_output)
        adapter.ingest_analysis(result)
        
        # Document Status panel can now query the stores as before
    """

    def __init__(
        self,
        plot_store: DocumentPlotStateStore | None = None,
        character_store: CharacterMapStore | None = None,
    ) -> None:
        self._plot_store = plot_store
        self._character_store = character_store

    def ingest_analysis(self, result: AnalysisResult) -> dict[str, int]:
        """Ingest analysis results into memory stores.
        
        Args:
            result: The analysis result to ingest.
            
        Returns:
            Dict with counts of ingested items per store.
        """
        stats = {
            "entities_ingested": 0,
            "plot_points_ingested": 0,
            "characters_ingested": 0,
        }

        if self._plot_store:
            stats.update(self._ingest_to_plot_store(result))

        if self._character_store:
            stats.update(self._ingest_to_character_store(result))

        LOGGER.debug(
            "Ingested analysis for document %s: %s",
            result.document_id,
            stats,
        )
        return stats

    def _ingest_to_plot_store(self, result: AnalysisResult) -> dict[str, int]:
        """Ingest analysis results to DocumentPlotStateStore."""
        if self._plot_store is None:
            return {}

        document_id = result.document_id
        stats = {"entities_ingested": 0, "plot_points_ingested": 0}

        # Get or create the plot state
        state = self._plot_store.get(document_id)
        if state is None:
            state = DocumentPlotState(document_id=document_id)
            self._plot_store._records[document_id] = state

        state.touch(version_id=result.version_id)

        # Ingest characters as entities
        for char_data in result.characters or []:
            name = char_data.get("name", "").strip()
            if not name:
                continue

            summary_parts = []
            if char_data.get("role"):
                summary_parts.append(f"Role: {char_data['role']}")
            if char_data.get("traits"):
                summary_parts.append(f"Traits: {', '.join(char_data['traits'][:3])}")

            summary = ". ".join(summary_parts) if summary_parts else None
            entity = state.upsert_entity(name, pointer_id=None, summary=summary)

            # Set entity kind based on role
            role = char_data.get("role", "").lower()
            if role in ("protagonist", "antagonist", "supporting", "minor"):
                entity.kind = "character"
            
            # Store aliases
            for alias in char_data.get("aliases", []):
                if alias and alias != name:
                    entity.attributes.setdefault("aliases", []).append(alias)

            stats["entities_ingested"] += 1

        # Ingest plot points as beats
        for point_data in result.plot_points or []:
            summary = point_data.get("summary", "").strip()
            if not summary:
                continue

            arc_id = "primary"
            point_type = point_data.get("type", "")
            if point_type:
                arc_id = point_type.lower().replace(" ", "_")

            state.add_beat(
                arc_id,
                summary=summary,
                pointer_id=None,
                chunk_hash=None,
                metadata={
                    "significance": point_data.get("significance"),
                    "characters_involved": point_data.get("characters_involved"),
                    "source": "analyze_document",
                },
            )
            stats["plot_points_ingested"] += 1

        # Store themes in metadata
        if result.themes:
            state.metadata["themes"] = result.themes

        # Store summary in metadata
        if result.summary:
            state.metadata["summary"] = result.summary

        # Mark analysis source
        state.metadata["analysis_source"] = "analyze_document"
        state.metadata["chunks_processed"] = result.chunks_processed

        # Enforce store limits
        self._plot_store._enforce_limits(state)

        return stats

    def _ingest_to_character_store(self, result: AnalysisResult) -> dict[str, int]:
        """Ingest analysis results to CharacterMapStore."""
        if self._character_store is None:
            return {}

        document_id = result.document_id
        stats = {"characters_ingested": 0}

        for char_data in result.characters or []:
            name = char_data.get("name", "").strip()
            if not name:
                continue

            # Build summary from character data
            summary_parts = [name]
            if char_data.get("role"):
                summary_parts.append(f"({char_data['role']})")
            if char_data.get("traits"):
                summary_parts.append(f"Traits: {', '.join(char_data['traits'][:5])}")

            summary = " - ".join(summary_parts)

            # Ingest via the standard interface
            self._character_store.ingest_summary(
                document_id=document_id,
                summary=summary,
                version_id=result.version_id,
                chunk_id=None,
                pointer_id=None,
                chunk_hash=None,
            )

            # Add aliases
            state = self._character_store._documents.get(document_id)
            if state:
                from .plot_state import _slugify
                entity_id = _slugify(name)
                record = state.entities.get(entity_id)
                if record:
                    for alias in char_data.get("aliases", []):
                        if alias:
                            record.add_alias(alias)

            stats["characters_ingested"] += 1

        return stats

    def clear(self, document_id: str | None = None) -> None:
        """Clear ingested analysis data from stores.
        
        Args:
            document_id: If provided, only clear this document. 
                         If None, clear all documents.
        """
        if self._plot_store:
            self._plot_store.clear(document_id)
        if self._character_store:
            self._character_store.clear(document_id)


class AnalysisResultCache:
    """Cache for analyze_document results.
    
    Provides a dedicated cache for analysis results that can be queried
    by document_status_service.py without going through the legacy stores.
    This enables a gradual migration path where both old and new data
    sources can coexist.
    """

    def __init__(self, max_entries: int = 24) -> None:
        self._entries: dict[str, AnalysisResult] = {}
        self._max_entries = max(1, max_entries)

    def store(self, result: AnalysisResult) -> None:
        """Store an analysis result."""
        self._entries[result.document_id] = result
        self._enforce_limits()

    def get(self, document_id: str) -> AnalysisResult | None:
        """Get cached analysis result for a document."""
        return self._entries.get(document_id)

    def clear(self, document_id: str | None = None) -> None:
        """Clear cached results."""
        if document_id is None:
            self._entries.clear()
        else:
            self._entries.pop(document_id, None)

    def _enforce_limits(self) -> None:
        if len(self._entries) <= self._max_entries:
            return
        # Remove oldest entries
        sorted_entries = sorted(
            self._entries.items(),
            key=lambda item: item[1].generated_at or _utcnow(),
        )
        for doc_id, _ in sorted_entries[:-self._max_entries]:
            self._entries.pop(doc_id, None)

    def snapshot(self, document_id: str) -> dict[str, Any] | None:
        """Get a snapshot of analysis results in the format expected by UI.
        
        Returns data in a format compatible with document_status_service.py.
        """
        result = self.get(document_id)
        if result is None:
            return None

        return {
            "document_id": result.document_id,
            "version_id": result.version_id,
            "generated_at": result.generated_at.isoformat() if result.generated_at else None,
            "characters": result.characters or [],
            "plot_points": result.plot_points or [],
            "themes": result.themes or [],
            "summary": result.summary,
            "chunks_processed": result.chunks_processed,
            "source": "analyze_document",
        }


__all__ = [
    "AnalysisResult",
    "AnalysisMemoryAdapter",
    "AnalysisResultCache",
]
