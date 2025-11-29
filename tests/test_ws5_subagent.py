"""Tests for Workstream 5: Subagent Architecture.

Tests for:
- WS5.1: Subagent Infrastructure (ChunkCoordinator, SubagentOrchestrator, etc.)
- WS5.2: analyze_document Tool
- WS5.3: transform_document Tool
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Sequence
from unittest.mock import MagicMock, AsyncMock

import pytest

# WS5.1 imports
from tinkerbell.ai.tools.subagent import (
    SubagentType,
    TaskPriority,
    ChunkSpec,
    ChunkBoundary,
    SubagentTask,
    SubagentResult,
    ProgressTracker,
    ChunkCoordinator,
    SubagentOrchestrator,
    AnalysisAggregator,
    TransformAggregator,
    estimate_tokens,
    find_chunk_boundaries,
    split_into_chunks,
)

# WS5.2 imports
from tinkerbell.ai.tools.analyze_document import (
    AnalyzeDocumentTool,
    AnalysisType,
    format_analysis_output,
)

# WS5.3 imports
from tinkerbell.ai.tools.transform_document import (
    TransformDocumentTool,
    TransformationType,
    OutputMode,
    CharacterRenameSpec,
    ConsistencyIssue,
    find_character_mentions,
    apply_character_rename,
    check_transformation_consistency,
)

# Base imports for tool tests
from tinkerbell.ai.tools.base import ToolContext
from tinkerbell.ai.tools.version import VersionManager
from tinkerbell.ai.tools.errors import InvalidParameterError, ContentRequiredError


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockDocumentProvider:
    """Mock document provider for testing."""

    documents: dict[str, str] = field(default_factory=dict)
    active_tab: str | None = None
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_document_text(self, tab_id: str | None = None) -> str:
        tab = tab_id or self.active_tab
        return self.documents.get(tab, "")

    def get_active_tab_id(self) -> str | None:
        return self.active_tab

    def get_document_content(self, tab_id: str) -> str | None:
        return self.documents.get(tab_id)

    def get_document_metadata(self, tab_id: str) -> dict[str, Any] | None:
        return self.metadata.get(tab_id)


@dataclass
class MockSubagentExecutor:
    """Mock executor for subagent tasks."""

    delay: float = 0.01
    should_fail: bool = False
    fail_message: str = "Mock failure"

    async def execute(self, task: SubagentTask) -> SubagentResult:
        await asyncio.sleep(self.delay)

        if self.should_fail:
            return SubagentResult(
                task_id=task.task_id,
                success=False,
                error=self.fail_message,
                chunk_id=task.chunk.chunk_id,
            )

        return SubagentResult(
            task_id=task.task_id,
            success=True,
            output={
                "summary": f"Analyzed chunk {task.chunk.chunk_id}",
                "characters": [],
                "themes": [],
            },
            chunk_id=task.chunk.chunk_id,
            tokens_used=task.chunk.token_estimate // 2,
        )


@pytest.fixture
def sample_document() -> str:
    """Sample document for testing."""
    return """Chapter 1: The Beginning

Alice walked through the garden, admiring the roses. She met the Cheshire Cat,
who smiled mysteriously at her.

"Where are you going?" asked the Cat.

"I don't know," Alice replied. "Can you help me?"

***

Chapter 2: The Tea Party

The Mad Hatter poured tea for everyone. Alice watched in amazement as the
March Hare dipped his watch in butter.

"Time has stopped," the Hatter explained sadly.

Alice felt confused but intrigued by these strange characters.
"""


@pytest.fixture
def long_document() -> str:
    """Long document that exceeds chunking threshold."""
    # Create a document > 20k chars
    paragraph = """
This is a paragraph of text that will be repeated many times to create a long
document for testing the chunking functionality. It contains various sentences
and some dialogue. "Hello there," said the character. The story continued with
more narrative and descriptions of the setting.

"""
    return paragraph * 200  # ~200 * 300 chars = 60k chars


@pytest.fixture
def version_manager() -> VersionManager:
    """Version manager for testing."""
    vm = VersionManager()
    vm.register_tab("tab1", "doc1", "hash1")
    return vm


@pytest.fixture
def tool_context(version_manager: VersionManager) -> ToolContext:
    """Tool context for testing."""
    provider = MockDocumentProvider(
        documents={"tab1": "Hello world"},
        active_tab="tab1",
    )
    return ToolContext(
        document_provider=provider,
        version_manager=version_manager,
        tab_id="tab1",
    )


# =============================================================================
# WS5.1: Subagent Infrastructure Tests
# =============================================================================


class TestEstimateTokens:
    """Tests for token estimation."""

    def test_empty_string(self):
        """Empty string has 0 tokens."""
        assert estimate_tokens("") == 0

    def test_simple_text(self):
        """Simple text estimates tokens correctly."""
        # ~4 chars per token
        text = "Hello world"  # 11 chars
        tokens = estimate_tokens(text)
        assert 2 <= tokens <= 4

    def test_longer_text(self):
        """Longer text scales proportionally."""
        short = "Hello"
        long = "Hello " * 100
        assert estimate_tokens(long) > estimate_tokens(short) * 50


class TestFindChunkBoundaries:
    """Tests for chunk boundary detection."""

    def test_chapter_markers(self):
        """Detects chapter markers."""
        text = "Some text\n\nChapter 1\n\nMore text\n\nChapter 2"
        boundaries = find_chunk_boundaries(text)

        chapter_bounds = [b for b in boundaries if b.boundary_type == "chapter"]
        assert len(chapter_bounds) == 2

    def test_scene_breaks(self):
        """Detects scene breaks."""
        text = "Scene one\n\n***\n\nScene two"
        boundaries = find_chunk_boundaries(text)

        scene_bounds = [b for b in boundaries if b.boundary_type == "scene"]
        assert len(scene_bounds) == 1

    def test_paragraph_breaks(self):
        """Detects paragraph breaks."""
        text = "Para one.\n\nPara two.\n\nPara three."
        boundaries = find_chunk_boundaries(text)

        para_bounds = [b for b in boundaries if b.boundary_type == "paragraph"]
        assert len(para_bounds) >= 2

    def test_boundary_scores(self):
        """Chapter boundaries score higher than paragraphs."""
        text = "Chapter 1\n\nSome paragraph."
        boundaries = find_chunk_boundaries(text)

        chapter = [b for b in boundaries if b.boundary_type == "chapter"]
        para = [b for b in boundaries if b.boundary_type == "paragraph"]

        if chapter and para:
            assert chapter[0].score > para[0].score


class TestSplitIntoChunks:
    """Tests for document chunking."""

    def test_small_document_single_chunk(self):
        """Small documents return single chunk."""
        text = "Small document"
        chunks = split_into_chunks(text, "doc1")

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert chunks[0].document_id == "doc1"

    def test_large_document_multiple_chunks(self, long_document: str):
        """Large documents are split into multiple chunks."""
        chunks = split_into_chunks(long_document, "doc1", target_tokens=1000)

        assert len(chunks) > 1
        # All content should be covered
        total_chars = sum(len(c.content) for c in chunks)
        assert total_chars >= len(long_document) * 0.9  # Allow for overlap

    def test_chunk_has_metadata(self):
        """Chunks have proper metadata."""
        text = "Line 1\nLine 2\nLine 3"
        chunks = split_into_chunks(text, "doc1")

        chunk = chunks[0]
        assert chunk.chunk_id.startswith("doc1-chunk-")
        assert chunk.document_id == "doc1"
        assert chunk.start_char == 0
        assert chunk.end_char == len(text)
        assert chunk.token_estimate > 0

    def test_empty_document(self):
        """Empty document returns no chunks."""
        chunks = split_into_chunks("", "doc1")
        assert len(chunks) == 0


class TestChunkSpec:
    """Tests for ChunkSpec class."""

    def test_char_count(self):
        """char_count returns correct count."""
        chunk = ChunkSpec(
            chunk_id="c1",
            document_id="d1",
            content="hello world",
        )
        assert chunk.char_count == 11

    def test_line_count(self):
        """line_count returns correct count."""
        chunk = ChunkSpec(
            chunk_id="c1",
            document_id="d1",
            content="line1\nline2\nline3",
        )
        assert chunk.line_count == 3

    def test_preview(self):
        """preview truncates long content."""
        chunk = ChunkSpec(
            chunk_id="c1",
            document_id="d1",
            content="a" * 200,
        )
        preview = chunk.preview(max_chars=50)
        assert len(preview) < 60
        assert preview.endswith("...")


class TestChunkCoordinator:
    """Tests for ChunkCoordinator class."""

    def test_should_chunk_small(self):
        """Small documents don't need chunking."""
        coord = ChunkCoordinator(auto_chunk_threshold=20000)
        assert not coord.should_chunk("Small text")

    def test_should_chunk_large(self, long_document: str):
        """Large documents need chunking."""
        coord = ChunkCoordinator(auto_chunk_threshold=20000)
        assert coord.should_chunk(long_document)

    def test_create_chunks(self, long_document: str):
        """create_chunks produces chunks."""
        coord = ChunkCoordinator()
        chunks = coord.create_chunks(long_document, "doc1")
        assert len(chunks) > 1

    def test_merge_transformed_chunks(self):
        """merge_transformed_chunks combines results."""
        coord = ChunkCoordinator()
        original = "Hello World"

        chunks = [ChunkSpec(
            chunk_id="c1",
            document_id="d1",
            content="Hello World",
            start_char=0,
            end_char=11,
        )]

        results = [SubagentResult(
            task_id="t1",
            success=True,
            output={"transformed_content": "Hi World"},
            chunk_id="c1",
        )]

        merged = coord.merge_transformed_chunks(original, chunks, results)
        assert merged == "Hi World"


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_initial_state(self):
        """Initial state is correct."""
        tracker = ProgressTracker()
        assert tracker.total_tasks == 0
        assert tracker.completed_tasks == 0
        assert tracker.progress_percent == 100.0  # 0/0 = 100%

    def test_start(self):
        """start sets up tracking."""
        tracker = ProgressTracker()
        tracker.start(10)

        assert tracker.total_tasks == 10
        assert tracker.completed_tasks == 0
        assert tracker.progress_percent == 0.0

    def test_task_completed(self):
        """task_completed updates counters."""
        tracker = ProgressTracker()
        tracker.start(2)

        task = SubagentTask(
            task_id="t1",
            subagent_type=SubagentType.CHUNK_ANALYZER,
            chunk=ChunkSpec("c1", "d1", "content"),
        )
        result = SubagentResult(task_id="t1", success=True, chunk_id="c1")

        tracker.task_completed(task, result)

        assert tracker.completed_tasks == 1
        assert tracker.successful_tasks == 1
        assert tracker.progress_percent == 50.0

    def test_failed_task_tracking(self):
        """Failed tasks are tracked separately."""
        tracker = ProgressTracker()
        tracker.start(2)

        task = SubagentTask(
            task_id="t1",
            subagent_type=SubagentType.CHUNK_ANALYZER,
            chunk=ChunkSpec("c1", "d1", "content"),
        )
        result = SubagentResult(task_id="t1", success=False, error="failed", chunk_id="c1")

        tracker.task_completed(task, result)

        assert tracker.completed_tasks == 1
        assert tracker.failed_tasks == 1
        assert tracker.successful_tasks == 0


class TestSubagentOrchestrator:
    """Tests for SubagentOrchestrator class."""

    @pytest.mark.asyncio
    async def test_run_tasks_empty(self):
        """Empty task list returns empty results."""
        orchestrator = SubagentOrchestrator()
        results = await orchestrator.run_tasks([])
        assert results == []

    @pytest.mark.asyncio
    async def test_run_tasks_no_executor(self):
        """Tasks fail gracefully without executor."""
        orchestrator = SubagentOrchestrator()
        tasks = [SubagentTask(
            task_id="t1",
            subagent_type=SubagentType.CHUNK_ANALYZER,
            chunk=ChunkSpec("c1", "d1", "content"),
        )]

        results = await orchestrator.run_tasks(tasks)

        assert len(results) == 1
        assert not results[0].success
        assert "No executor" in results[0].error

    @pytest.mark.asyncio
    async def test_run_tasks_with_executor(self):
        """Tasks execute with executor."""
        executor = MockSubagentExecutor()
        orchestrator = SubagentOrchestrator(executor=executor)
        tasks = [SubagentTask(
            task_id="t1",
            subagent_type=SubagentType.CHUNK_ANALYZER,
            chunk=ChunkSpec("c1", "d1", "content"),
        )]

        results = await orchestrator.run_tasks(tasks)

        assert len(results) == 1
        assert results[0].success

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Tasks can run in parallel."""
        executor = MockSubagentExecutor(delay=0.1)
        orchestrator = SubagentOrchestrator(executor=executor, max_parallel=4)

        tasks = [
            SubagentTask(
                task_id=f"t{i}",
                subagent_type=SubagentType.CHUNK_ANALYZER,
                chunk=ChunkSpec(f"c{i}", "d1", "content"),
            )
            for i in range(4)
        ]

        import time
        start = time.perf_counter()
        results = await orchestrator.run_tasks(tasks, parallel=True)
        duration = time.perf_counter() - start

        assert len(results) == 4
        # Parallel should be faster than sequential (4 * 0.1 = 0.4s)
        assert duration < 0.3  # Allow some overhead


class TestAnalysisAggregator:
    """Tests for AnalysisAggregator class."""

    def test_aggregate_empty(self):
        """Empty results produce empty output."""
        agg = AnalysisAggregator()
        result = agg.aggregate([])

        assert result["chunks_processed"] == 0
        assert result["characters"] == []

    def test_aggregate_characters(self):
        """Characters are aggregated and deduplicated."""
        agg = AnalysisAggregator()
        results = [
            SubagentResult(
                task_id="t1",
                success=True,
                output={
                    "characters": [
                        {"name": "Alice", "role": "protagonist"},
                    ]
                },
                chunk_id="c1",
            ),
            SubagentResult(
                task_id="t2",
                success=True,
                output={
                    "characters": [
                        {"name": "Alice", "role": "protagonist"},
                        {"name": "Bob", "role": "supporting"},
                    ]
                },
                chunk_id="c2",
            ),
        ]

        aggregated = agg.aggregate(results)

        assert len(aggregated["characters"]) == 2
        names = [c["name"] for c in aggregated["characters"]]
        assert "Alice" in names
        assert "Bob" in names

    def test_aggregate_with_failures(self):
        """Failures are reported in output."""
        agg = AnalysisAggregator()
        results = [
            SubagentResult(task_id="t1", success=True, output={}, chunk_id="c1"),
            SubagentResult(task_id="t2", success=False, error="failed", chunk_id="c2"),
        ]

        aggregated = agg.aggregate(results)

        assert aggregated["chunks_processed"] == 1
        assert aggregated["chunks_failed"] == 1
        assert aggregated["errors"] is not None


class TestTransformAggregator:
    """Tests for TransformAggregator class."""

    def test_aggregate_replacements(self):
        """Replacement counts are summed."""
        agg = TransformAggregator()
        results = [
            SubagentResult(
                task_id="t1",
                success=True,
                output={"replacements": 5},
                chunk_id="c1",
                tokens_used=100,
            ),
            SubagentResult(
                task_id="t2",
                success=True,
                output={"replacements": 3},
                chunk_id="c2",
                tokens_used=80,
            ),
        ]

        aggregated = agg.aggregate(results)

        assert aggregated["total_replacements"] == 8
        assert aggregated["tokens_used"] == 180


# =============================================================================
# WS5.2: analyze_document Tool Tests
# =============================================================================


class TestAnalysisType:
    """Tests for AnalysisType enum."""

    def test_all_types(self):
        """All analysis types are defined."""
        assert AnalysisType.CHARACTERS.value == "characters"
        assert AnalysisType.PLOT.value == "plot"
        assert AnalysisType.STYLE.value == "style"
        assert AnalysisType.SUMMARY.value == "summary"
        assert AnalysisType.THEMES.value == "themes"
        assert AnalysisType.CUSTOM.value == "custom"


class TestAnalyzeDocumentTool:
    """Tests for AnalyzeDocumentTool."""

    def test_validate_valid_type(self, tool_context: ToolContext):
        """Valid analysis type passes validation."""
        tool = AnalyzeDocumentTool()
        # Should not raise
        tool.validate({"analysis_type": "characters"})

    def test_validate_invalid_type(self):
        """Invalid analysis type fails validation."""
        tool = AnalyzeDocumentTool()
        with pytest.raises(InvalidParameterError):
            tool.validate({"analysis_type": "invalid"})

    def test_validate_custom_requires_prompt(self):
        """Custom analysis requires custom_prompt."""
        tool = AnalyzeDocumentTool()
        with pytest.raises(InvalidParameterError):
            tool.validate({"analysis_type": "custom"})

    def test_validate_custom_with_prompt(self):
        """Custom analysis with prompt passes."""
        tool = AnalyzeDocumentTool()
        tool.validate({"analysis_type": "custom", "custom_prompt": "Analyze themes"})

    def test_plan_small_document(self, tool_context: ToolContext):
        """Small document produces single task."""
        tool = AnalyzeDocumentTool()
        tasks = tool.plan(tool_context, {"analysis_type": "summary"})

        assert len(tasks) == 1
        assert tasks[0]["analysis_type"] == AnalysisType.SUMMARY

    def test_plan_large_document(self, version_manager: VersionManager, long_document: str):
        """Large document produces multiple tasks."""
        provider = MockDocumentProvider(
            documents={"tab1": long_document},
            active_tab="tab1",
        )
        context = ToolContext(
            document_provider=provider,
            version_manager=version_manager,
            tab_id="tab1",
        )

        tool = AnalyzeDocumentTool()
        tasks = tool.plan(context, {"analysis_type": "characters"})

        assert len(tasks) > 1

    def test_plan_max_chunks(self, version_manager: VersionManager, long_document: str):
        """max_chunks limits number of tasks."""
        provider = MockDocumentProvider(
            documents={"tab1": long_document},
            active_tab="tab1",
        )
        context = ToolContext(
            document_provider=provider,
            version_manager=version_manager,
            tab_id="tab1",
        )

        tool = AnalyzeDocumentTool()
        tasks = tool.plan(context, {"analysis_type": "summary", "max_chunks": 2})

        assert len(tasks) <= 2

    def test_execute_subagent(self, tool_context: ToolContext):
        """execute_subagent returns error without orchestrator."""
        tool = AnalyzeDocumentTool()
        task = {
            "chunk": ChunkSpec("c1", "d1", "Hello world"),
            "analysis_type": AnalysisType.SUMMARY,
            "instructions": "Summarize",
            "include_line_refs": True,
        }

        result = tool.execute_subagent(tool_context, task)

        # Without orchestrator, subagent execution returns error
        assert result["success"] is False
        assert result["chunk_id"] == "c1"
        assert "not yet configured" in result["error"]

    def test_aggregate(self):
        """aggregate combines results."""
        tool = AnalyzeDocumentTool()
        results = [
            {"task_id": "t1", "success": True, "output": {"summary": "Part 1"}, "chunk_id": "c1"},
            {"task_id": "t2", "success": True, "output": {"summary": "Part 2"}, "chunk_id": "c2"},
        ]

        aggregated = tool.aggregate(results)

        assert aggregated["status"] == "complete"
        assert aggregated["chunks_processed"] == 2


class TestFormatAnalysisOutput:
    """Tests for format_analysis_output function."""

    def test_markdown_format(self):
        """Markdown format produces proper output."""
        results = {
            "characters": [{"name": "Alice", "role": "protagonist"}],
            "themes": ["adventure", "identity"],
            "chunks_processed": 2,
        }

        output = format_analysis_output("characters", results, format_type="markdown")

        assert "# Document Analysis" in output
        assert "Alice" in output
        assert "adventure" in output

    def test_json_format(self):
        """JSON format produces valid JSON."""
        import json
        results = {"characters": [], "themes": []}

        output = format_analysis_output("characters", results, format_type="json")

        # Should be valid JSON
        parsed = json.loads(output)
        assert "characters" in parsed

    def test_plain_format(self):
        """Plain format produces simple output."""
        results = {"summary": "Test summary"}

        output = format_analysis_output("summary", results, format_type="plain")

        assert "Analysis Type: summary" in output


# =============================================================================
# WS5.3: transform_document Tool Tests
# =============================================================================


class TestTransformationType:
    """Tests for TransformationType enum."""

    def test_all_types(self):
        """All transformation types are defined."""
        assert TransformationType.CHARACTER_RENAME.value == "character_rename"
        assert TransformationType.SETTING_CHANGE.value == "setting_change"
        assert TransformationType.STYLE_REWRITE.value == "style_rewrite"
        assert TransformationType.TENSE_CHANGE.value == "tense_change"
        assert TransformationType.POV_CHANGE.value == "pov_change"
        assert TransformationType.CUSTOM.value == "custom"


class TestFindCharacterMentions:
    """Tests for find_character_mentions function."""

    def test_simple_mention(self):
        """Finds simple name mentions."""
        text = "Alice went to the store. Alice bought apples."
        mentions = find_character_mentions(text, "Alice")

        assert len(mentions) == 2
        assert all(m[2] == "Alice" for m in mentions)

    def test_possessive_mention(self):
        """Finds possessive forms."""
        text = "Alice's cat was sleeping."
        mentions = find_character_mentions(text, "Alice")

        assert len(mentions) == 1
        assert mentions[0][2] == "Alice's"

    def test_case_insensitive(self):
        """Finds case variations."""
        text = "alice and ALICE"
        mentions = find_character_mentions(text, "Alice")

        assert len(mentions) == 2

    def test_with_aliases(self):
        """Finds aliases too."""
        text = "Alice, also known as Ali, went home."
        mentions = find_character_mentions(text, "Alice", aliases=["Ali"])

        assert len(mentions) == 2


class TestApplyCharacterRename:
    """Tests for apply_character_rename function."""

    def test_simple_rename(self):
        """Renames all occurrences."""
        text = "Alice went home. Alice was tired."
        spec = CharacterRenameSpec(old_name="Alice", new_name="Carol")

        result, count = apply_character_rename(text, spec)

        assert "Carol" in result
        assert "Alice" not in result
        assert count == 2

    def test_possessive_rename(self):
        """Renames possessive forms."""
        text = "Alice's cat was Alice's favorite."
        spec = CharacterRenameSpec(old_name="Alice", new_name="Carol")

        result, count = apply_character_rename(text, spec)

        assert "Carol's" in result
        assert "Alice's" not in result

    def test_with_aliases(self):
        """Renames aliases too."""
        text = "Alice and Ali went home."
        spec = CharacterRenameSpec(old_name="Alice", new_name="Carol", aliases=["Ali"])

        result, count = apply_character_rename(text, spec)

        assert result.count("Carol") == 2


class TestTransformDocumentTool:
    """Tests for TransformDocumentTool."""

    def test_validate_character_rename_valid(self):
        """Valid character_rename passes validation."""
        tool = TransformDocumentTool()
        tool.validate({
            "transformation_type": "character_rename",
            "old_name": "Alice",
            "new_name": "Carol",
        })

    def test_validate_character_rename_missing_old_name(self):
        """character_rename requires old_name."""
        tool = TransformDocumentTool()
        with pytest.raises(ContentRequiredError):
            tool.validate({
                "transformation_type": "character_rename",
                "new_name": "Carol",
            })

    def test_validate_character_rename_missing_new_name(self):
        """character_rename requires new_name."""
        tool = TransformDocumentTool()
        with pytest.raises(ContentRequiredError):
            tool.validate({
                "transformation_type": "character_rename",
                "old_name": "Alice",
            })

    def test_validate_setting_change(self):
        """setting_change requires old and new setting."""
        tool = TransformDocumentTool()
        with pytest.raises(ContentRequiredError):
            tool.validate({
                "transformation_type": "setting_change",
                "new_setting": "Paris",
            })

    def test_validate_style_rewrite(self):
        """style_rewrite requires target_style."""
        tool = TransformDocumentTool()
        with pytest.raises(ContentRequiredError):
            tool.validate({"transformation_type": "style_rewrite"})

    def test_validate_tense_change(self):
        """tense_change requires target_tense or from/to tense."""
        tool = TransformDocumentTool()
        # Missing all tense params should fail
        with pytest.raises(ContentRequiredError):
            tool.validate({"transformation_type": "tense_change"})
        # target_tense alone is valid (simpler API)
        tool.validate({
            "transformation_type": "tense_change",
            "target_tense": "present",
        })
        # from_tense + to_tense is also valid (explicit API)
        tool.validate({
            "transformation_type": "tense_change",
            "from_tense": "past",
            "to_tense": "present",
        })

    def test_validate_pov_change(self):
        """pov_change requires target_pov or from/to POV."""
        tool = TransformDocumentTool()
        # Missing all POV params should fail
        with pytest.raises(ContentRequiredError):
            tool.validate({"transformation_type": "pov_change"})
        # target_pov alone is valid (simpler API)
        tool.validate({
            "transformation_type": "pov_change",
            "target_pov": "first",
        })
        # from_pov + to_pov is also valid (explicit API)
        tool.validate({
            "transformation_type": "pov_change",
            "from_pov": "third",
            "to_pov": "first",
        })

    def test_validate_custom_requires_prompt(self):
        """custom requires custom_prompt."""
        tool = TransformDocumentTool()
        with pytest.raises(ContentRequiredError):
            tool.validate({"transformation_type": "custom"})

    def test_validate_invalid_output_mode(self):
        """Invalid output_mode fails."""
        tool = TransformDocumentTool()
        with pytest.raises(InvalidParameterError):
            tool.validate({
                "transformation_type": "custom",
                "custom_prompt": "Do something",
                "output_mode": "invalid",
            })

    def test_validate_in_place_requires_version(self):
        """in_place output_mode requires version."""
        tool = TransformDocumentTool()
        with pytest.raises(ContentRequiredError):
            tool.validate({
                "transformation_type": "custom",
                "custom_prompt": "Do something",
                "output_mode": "in_place",
            })

    def test_plan_creates_tasks(self, tool_context: ToolContext):
        """plan creates transformation tasks."""
        tool = TransformDocumentTool()
        tasks = tool.plan(tool_context, {
            "transformation_type": "character_rename",
            "old_name": "Alice",
            "new_name": "Carol",
        })

        assert len(tasks) >= 1
        assert tasks[0]["transformation_type"] == TransformationType.CHARACTER_RENAME

    def test_execute_subagent_character_rename(self, tool_context: ToolContext):
        """Character rename executes locally."""
        tool = TransformDocumentTool()
        task = {
            "chunk": ChunkSpec("c1", "d1", "Alice went home."),
            "transformation_type": TransformationType.CHARACTER_RENAME,
            "instructions": "",
            "params": {"old_name": "Alice", "new_name": "Carol"},
        }

        result = tool.execute_subagent(tool_context, task)

        assert result["success"]
        assert "Carol" in result["output"]["transformed_content"]
        assert result["output"]["replacements"] == 1


class TestCheckTransformationConsistency:
    """Tests for check_transformation_consistency function."""

    def test_no_issues_similar_length(self):
        """Similar length documents have no issues."""
        original = "Hello world"
        transformed = "Hi there world"

        issues = check_transformation_consistency(
            original,
            transformed,
            TransformationType.STYLE_REWRITE,
        )

        length_issues = [i for i in issues if i.issue_type == "length_change"]
        assert len(length_issues) == 0

    def test_significant_length_change(self):
        """Significant length change is flagged."""
        original = "Hello world"
        transformed = "Hi"

        issues = check_transformation_consistency(
            original,
            transformed,
            TransformationType.STYLE_REWRITE,
        )

        length_issues = [i for i in issues if i.issue_type == "length_change"]
        assert len(length_issues) == 1

    def test_quote_imbalance(self):
        """Quote imbalance is flagged."""
        original = '"Hello," said Alice. "How are you?"'
        transformed = 'Hello said Alice How are you'

        issues = check_transformation_consistency(
            original,
            transformed,
            TransformationType.STYLE_REWRITE,
        )

        quote_issues = [i for i in issues if i.issue_type == "quote_imbalance"]
        assert len(quote_issues) == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestWS5Integration:
    """Integration tests for WS5 components."""

    def test_chunking_analysis_workflow(self, long_document: str, version_manager: VersionManager):
        """Full chunking and analysis workflow - without orchestrator, reports failures."""
        # Setup
        provider = MockDocumentProvider(
            documents={"tab1": long_document},
            active_tab="tab1",
        )
        context = ToolContext(
            document_provider=provider,
            version_manager=version_manager,
            tab_id="tab1",
        )

        # Create tool with coordinator
        tool = AnalyzeDocumentTool(
            chunk_coordinator=ChunkCoordinator(target_tokens=2000),
        )

        # Plan
        tasks = tool.plan(context, {"analysis_type": "summary"})
        assert len(tasks) > 1

        # Execute each task
        results = []
        for task in tasks:
            result = tool.execute_subagent(context, task)
            results.append(result)

        # Aggregate
        aggregated = tool.aggregate(results)

        # Without orchestrator, all chunks fail with informative error
        assert aggregated["status"] == "complete"
        assert aggregated["chunks_failed"] == len(tasks)
        assert aggregated["chunks_processed"] == 0
        assert aggregated["errors"] is not None
        assert "not yet configured" in aggregated["errors"][0]["error"]

    def test_transform_with_rename(self, version_manager: VersionManager, sample_document: str):
        """Full transformation workflow with character rename."""
        provider = MockDocumentProvider(
            documents={"tab1": sample_document},
            active_tab="tab1",
        )
        context = ToolContext(
            document_provider=provider,
            version_manager=version_manager,
            tab_id="tab1",
        )

        tool = TransformDocumentTool()

        # Plan
        tasks = tool.plan(context, {
            "transformation_type": "character_rename",
            "old_name": "Alice",
            "new_name": "Dorothy",
        })

        # Execute
        results = []
        for task in tasks:
            result = tool.execute_subagent(context, task)
            results.append(result)

        # Aggregate
        aggregated = tool.aggregate(results)

        # Check results
        assert aggregated["status"] == "complete"
        assert aggregated["total_replacements"] > 0

        # Check transformed content
        for result in results:
            if result["success"]:
                content = result["output"].get("transformed_content", "")
                assert "Alice" not in content or "Dorothy" in content
