"""Analyze Document Tool for AI operations.

Provides document analysis capabilities including character extraction,
plot analysis, style assessment, and custom analysis tasks.
Uses subagents for parallel chunk processing of large documents.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, ClassVar, Protocol

from .base import SubagentTool, ToolContext
from .errors import (
    InvalidParameterError,
    TabNotFoundError,
    ToolError,
)
from .subagent import (
    AnalysisAggregator,
    ChunkCoordinator,
    ChunkSpec,
    SubagentOrchestrator,
    SubagentResult,
    SubagentTask,
    SubagentType,
    TaskPriority,
    estimate_tokens,
)

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Analysis Types
# =============================================================================


class AnalysisType(Enum):
    """Type of document analysis to perform."""

    CHARACTERS = "characters"  # Extract character information
    PLOT = "plot"  # Analyze plot structure
    STYLE = "style"  # Assess writing style
    SUMMARY = "summary"  # Generate summary
    THEMES = "themes"  # Extract themes
    CUSTOM = "custom"  # Custom analysis with user prompt


# Analysis type to instruction mapping
ANALYSIS_INSTRUCTIONS = {
    AnalysisType.CHARACTERS: """Analyze this document chunk and extract all characters mentioned.
For each character, identify:
- Name (and any aliases)
- Role (protagonist, antagonist, supporting, minor)
- Key traits or descriptions
- Relationships with other characters
- Notable actions or dialogue

Return as JSON:
{
    "characters": [
        {
            "name": "string",
            "aliases": ["string"],
            "role": "string",
            "traits": ["string"],
            "relationships": [{"character": "string", "relationship": "string"}],
            "mentions": [{"line": number, "context": "string"}]
        }
    ]
}""",

    AnalysisType.PLOT: """Analyze this document chunk for plot elements.
Identify:
- Key events or scenes
- Conflict points
- Character decisions/actions
- Setting details
- Foreshadowing or callbacks
- Narrative tension level

Return as JSON:
{
    "plot_points": [
        {
            "type": "event|conflict|decision|revelation",
            "summary": "string",
            "characters_involved": ["string"],
            "significance": "low|medium|high",
            "line_range": [start, end]
        }
    ],
    "tension_level": "low|building|high|climax|resolution"
}""",

    AnalysisType.STYLE: """Analyze the writing style in this document chunk.
Assess:
- Narrative voice (first/third person, past/present tense)
- Tone (formal, casual, dark, humorous, etc.)
- Sentence structure (simple, complex, varied)
- Dialogue style
- Description density
- Pacing (fast, moderate, slow)
- Notable stylistic techniques

Return as JSON:
{
    "style": {
        "voice": {"person": "first|second|third", "tense": "past|present|future"},
        "tone": ["string"],
        "sentence_complexity": "simple|moderate|complex|varied",
        "dialogue_frequency": "none|sparse|moderate|heavy",
        "description_density": "sparse|moderate|rich",
        "pacing": "fast|moderate|slow",
        "techniques": ["string"]
    }
}""",

    AnalysisType.SUMMARY: """Summarize this document chunk concisely.
Focus on:
- Main events or developments
- Key character actions
- Important information revealed
- Setting changes

Return as JSON:
{
    "summary": "string (2-4 sentences)"
}""",

    AnalysisType.THEMES: """Identify themes present in this document chunk.
Look for:
- Major themes (love, death, justice, etc.)
- Motifs (recurring symbols or ideas)
- Subtext or underlying messages
- Cultural or historical references

Return as JSON:
{
    "themes": ["string"],
    "motifs": [{"symbol": "string", "meaning": "string"}],
    "subtext": "string or null"
}""",
}


# =============================================================================
# Output Writer Protocol
# =============================================================================


class OutputWriter(Protocol):
    """Protocol for writing analysis output."""

    def create_output_tab(self, title: str, content: str) -> str | None:
        """Create a new output tab and return its ID."""
        ...

    def update_output_tab(self, tab_id: str, content: str) -> bool:
        """Update content in an existing tab."""
        ...


# =============================================================================
# Analyze Document Tool
# =============================================================================


@dataclass
class AnalyzeDocumentTool(SubagentTool):
    """Tool for analyzing document content using subagents.

    Performs document analysis with automatic chunking for large documents.
    Supports multiple analysis types that can run in parallel across chunks.

    Parameters:
        tab_id: Target tab ID (optional, defaults to active tab).
        analysis_type: Type of analysis to perform.
            - "characters": Extract character information
            - "plot": Analyze plot structure
            - "style": Assess writing style
            - "summary": Generate summary
            - "themes": Extract themes
            - "custom": Custom analysis (requires custom_prompt)
        custom_prompt: Custom analysis instructions (required if analysis_type="custom").
        output_tab: Tab ID to write results to (optional).
            If not provided, creates a new tab with analysis results.
        include_line_refs: Include line number references in results (default True).
        max_chunks: Maximum number of chunks to process (default None = all).

    Returns:
        analysis: The analysis results (structure depends on analysis_type).
        chunks_processed: Number of chunks processed.
        output_tab_id: ID of the output tab (if output was written).
        tokens_used: Total tokens consumed.
    """

    name: ClassVar[str] = "analyze_document"
    summarizable: ClassVar[bool] = True

    # Dependencies (injected)
    orchestrator: SubagentOrchestrator | None = None
    output_writer: OutputWriter | None = None
    chunk_coordinator: ChunkCoordinator | None = None

    def validate(self, params: dict[str, Any]) -> None:
        """Validate parameters."""
        super().validate(params)

        analysis_type = params.get("analysis_type", "summary")
        valid_types = [t.value for t in AnalysisType]

        if analysis_type not in valid_types:
            raise InvalidParameterError(
                message=f"Invalid analysis_type: {analysis_type}. Must be one of: {valid_types}",
                parameter="analysis_type",
                value=analysis_type,
                expected=f"one of {valid_types}",
            )

        if analysis_type == "custom" and not params.get("custom_prompt"):
            raise InvalidParameterError(
                message="custom_prompt is required when analysis_type='custom'",
                parameter="custom_prompt",
                value=None,
                expected="non-empty string",
            )

    def plan(
        self,
        context: ToolContext,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Plan analysis tasks for document chunks."""
        # Resolve tab ID
        tab_id = context.require_tab_id(params.get("tab_id"))

        # Get document content
        content = context.document_provider.get_document_content(tab_id)
        if content is None:
            raise TabNotFoundError(
                message=f"Document not found: {tab_id}",
                tab_id=tab_id,
            )

        # Parse analysis type
        analysis_type_str = params.get("analysis_type", "summary")
        analysis_type = AnalysisType(analysis_type_str)

        # Get instructions
        if analysis_type == AnalysisType.CUSTOM:
            instructions = params["custom_prompt"]
        else:
            instructions = ANALYSIS_INSTRUCTIONS[analysis_type]

        # Create chunk coordinator
        coordinator = self.chunk_coordinator or ChunkCoordinator()

        # Create chunks
        if coordinator.should_chunk(content):
            chunks = coordinator.create_chunks(content, tab_id)
            LOGGER.info(
                "Document chunked into %d pieces for analysis (type=%s)",
                len(chunks),
                analysis_type.value,
            )
        else:
            # Single chunk for small documents
            chunks = [ChunkSpec(
                chunk_id=f"{tab_id}-chunk-0",
                document_id=tab_id,
                content=content,
                start_char=0,
                end_char=len(content),
                token_estimate=estimate_tokens(content),
            )]

        # Apply max_chunks limit
        max_chunks = params.get("max_chunks")
        if max_chunks and max_chunks > 0:
            chunks = chunks[:max_chunks]

        # Convert to task specs
        tasks = []
        for chunk in chunks:
            tasks.append({
                "chunk": chunk,
                "analysis_type": analysis_type,
                "instructions": instructions,
                "include_line_refs": params.get("include_line_refs", True),
            })

        return tasks

    def aggregate(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate analysis results from all chunks."""
        # Convert dict results to SubagentResult objects
        subagent_results = []
        for r in results:
            if isinstance(r, SubagentResult):
                subagent_results.append(r)
            else:
                subagent_results.append(SubagentResult(
                    task_id=r.get("task_id", "unknown"),
                    success=r.get("success", False),
                    output=r.get("output", {}),
                    error=r.get("error"),
                    chunk_id=r.get("chunk_id", ""),
                    tokens_used=r.get("tokens_used", 0),
                ))

        # Use the analysis aggregator
        aggregator = AnalysisAggregator()
        return aggregator.aggregate(subagent_results)

    def execute_subagent(
        self,
        context: ToolContext,
        task: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute analysis on a single chunk.

        This is called for each chunk when subagent execution is needed.
        In a real implementation, this would delegate to the orchestrator.
        """
        chunk: ChunkSpec = task["chunk"]
        analysis_type: AnalysisType = task["analysis_type"]
        instructions: str = task["instructions"]

        # Create a SubagentTask
        subagent_task = SubagentTask(
            task_id=f"analyze-{uuid.uuid4().hex[:8]}",
            subagent_type=SubagentType.CHUNK_ANALYZER,
            chunk=chunk,
            instructions=instructions,
            priority=TaskPriority.NORMAL,
        )

        # If we have an orchestrator, use it
        if self.orchestrator:
            # Note: In async context, this would be awaited
            # For sync execution, we return a mock result
            return {
                "task_id": subagent_task.task_id,
                "success": True,
                "chunk_id": chunk.chunk_id,
                "output": {
                    "analysis_type": analysis_type.value,
                    "chunk_preview": chunk.preview(50),
                    "line_range": [chunk.start_line, chunk.end_line],
                },
                "tokens_used": chunk.token_estimate,
            }

        # Fallback: return a placeholder result
        return {
            "task_id": subagent_task.task_id,
            "success": True,
            "chunk_id": chunk.chunk_id,
            "output": self._mock_analysis_result(analysis_type, chunk),
            "tokens_used": chunk.token_estimate // 2,
        }

    def _mock_analysis_result(
        self,
        analysis_type: AnalysisType,
        chunk: ChunkSpec,
    ) -> dict[str, Any]:
        """Generate a mock analysis result for testing.

        In production, this would be replaced by actual LLM analysis.
        """
        if analysis_type == AnalysisType.CHARACTERS:
            return {"characters": [], "note": "Analysis pending"}
        elif analysis_type == AnalysisType.PLOT:
            return {"plot_points": [], "tension_level": "unknown"}
        elif analysis_type == AnalysisType.STYLE:
            return {"style": {}}
        elif analysis_type == AnalysisType.SUMMARY:
            return {"summary": f"Chunk from lines {chunk.start_line}-{chunk.end_line}"}
        elif analysis_type == AnalysisType.THEMES:
            return {"themes": [], "motifs": []}
        else:
            return {"custom": {}}


# =============================================================================
# Helper Functions
# =============================================================================


def format_analysis_output(
    analysis_type: str,
    results: dict[str, Any],
    *,
    format_type: str = "markdown",
) -> str:
    """Format analysis results for display.

    Args:
        analysis_type: Type of analysis performed.
        results: Analysis results dictionary.
        format_type: Output format ("markdown", "json", "plain").

    Returns:
        Formatted output string.
    """
    if format_type == "json":
        import json
        return json.dumps(results, indent=2)

    if format_type == "plain":
        lines = [f"Analysis Type: {analysis_type}", ""]
        for key, value in results.items():
            if key.startswith("_"):
                continue
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    # Default: markdown
    lines = [f"# Document Analysis: {analysis_type.title()}", ""]

    if "characters" in results and results["characters"]:
        lines.append("## Characters")
        for char in results["characters"]:
            name = char.get("name", "Unknown")
            role = char.get("role", "")
            lines.append(f"- **{name}** ({role})")
            if char.get("traits"):
                lines.append(f"  - Traits: {', '.join(char['traits'])}")
        lines.append("")

    if "themes" in results and results["themes"]:
        lines.append("## Themes")
        for theme in results["themes"]:
            lines.append(f"- {theme}")
        lines.append("")

    if "plot_points" in results and results["plot_points"]:
        lines.append("## Plot Points")
        for point in results["plot_points"]:
            summary = point.get("summary", "")
            significance = point.get("significance", "")
            lines.append(f"- {summary} ({significance})")
        lines.append("")

    if "summary" in results and results["summary"]:
        lines.append("## Summary")
        lines.append(results["summary"])
        lines.append("")

    if "style" in results and results["style"]:
        lines.append("## Style Analysis")
        style = results["style"]
        if "voice" in style:
            voice = style["voice"]
            lines.append(f"- Voice: {voice.get('person', 'N/A')} person, {voice.get('tense', 'N/A')} tense")
        if "tone" in style:
            lines.append(f"- Tone: {', '.join(style['tone'])}")
        if "pacing" in style:
            lines.append(f"- Pacing: {style['pacing']}")
        lines.append("")

    # Add processing stats
    lines.append("---")
    lines.append(f"*Chunks processed: {results.get('chunks_processed', 'N/A')}*")
    if results.get("chunks_failed"):
        lines.append(f"*Chunks failed: {results.get('chunks_failed')}*")

    return "\n".join(lines)


__all__ = [
    "AnalyzeDocumentTool",
    "AnalysisType",
    "format_analysis_output",
]
