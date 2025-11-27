"""Subagent Infrastructure for Document Processing.

This module provides infrastructure for spawning and coordinating subagents
that process documents in parallel chunks. Used by analyze_document and
transform_document tools.

Key components:
- SubagentType: Enum of subagent specializations
- ChunkSpec: Specification for a document chunk to process
- SubagentTask: A single subagent task definition
- SubagentResult: Result from a subagent execution
- ChunkCoordinator: Manages document chunking and parallel processing
- SubagentOrchestrator: Coordinates subagent execution and aggregation
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Subagent Types
# =============================================================================


class SubagentType(Enum):
    """Type of subagent specialization."""

    CHUNK_ANALYZER = auto()  # Analyzes a document chunk
    CONSISTENCY_CHECKER = auto()  # Checks consistency across chunks
    TRANSFORMER = auto()  # Transforms content in a chunk
    AGGREGATOR = auto()  # Aggregates results from other subagents
    CUSTOM = auto()  # Custom subagent with user-defined behavior


class TaskPriority(Enum):
    """Priority level for subagent tasks."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


# =============================================================================
# Chunk Specification
# =============================================================================


@dataclass(slots=True)
class ChunkSpec:
    """Specification for a document chunk to process.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        document_id: ID of the source document.
        content: The text content of the chunk.
        start_char: Starting character index in original document.
        end_char: Ending character index in original document.
        start_line: Starting line number (0-indexed).
        end_line: Ending line number (0-indexed).
        token_estimate: Estimated token count.
        metadata: Additional chunk metadata.
    """

    chunk_id: str
    document_id: str
    content: str
    start_char: int = 0
    end_char: int = 0
    start_line: int = 0
    end_line: int = 0
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        """Get character count of this chunk."""
        return len(self.content)

    @property
    def line_count(self) -> int:
        """Get line count of this chunk."""
        return self.content.count("\n") + 1

    def preview(self, max_chars: int = 100) -> str:
        """Get a preview of the chunk content."""
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars].rstrip() + "..."


@dataclass(slots=True)
class ChunkBoundary:
    """Represents a preferred boundary for chunk splitting."""

    char_index: int
    boundary_type: str  # "paragraph", "scene", "chapter", "sentence"
    score: float = 1.0  # Higher score = better boundary


# =============================================================================
# Subagent Task & Result
# =============================================================================


@dataclass(slots=True)
class SubagentTask:
    """A single subagent task to execute.

    Attributes:
        task_id: Unique identifier for this task.
        subagent_type: Type of subagent to use.
        chunk: The chunk to process.
        instructions: Specific instructions for this task.
        priority: Task priority level.
        timeout_seconds: Maximum execution time.
        metadata: Additional task metadata.
    """

    task_id: str
    subagent_type: SubagentType
    chunk: ChunkSpec
    instructions: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    timeout_seconds: float = 60.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubagentResult:
    """Result from a subagent execution.

    Attributes:
        task_id: ID of the task that produced this result.
        success: Whether the task completed successfully.
        output: The result data if successful.
        error: Error message if failed.
        latency_ms: Execution time in milliseconds.
        tokens_used: Tokens consumed by the subagent.
        chunk_id: ID of the processed chunk.
        metadata: Additional result metadata.
    """

    task_id: str
    success: bool
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    chunk_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Progress Tracking
# =============================================================================


class ProgressListener(Protocol):
    """Protocol for receiving progress updates."""

    def on_progress(
        self,
        completed: int,
        total: int,
        current_task: SubagentTask | None,
    ) -> None:
        """Called when progress is made."""
        ...

    def on_task_started(self, task: SubagentTask) -> None:
        """Called when a task starts."""
        ...

    def on_task_completed(self, task: SubagentTask, result: SubagentResult) -> None:
        """Called when a task completes."""
        ...


@dataclass
class ProgressTracker:
    """Tracks progress of subagent execution.

    Attributes:
        total_tasks: Total number of tasks to execute.
        completed_tasks: Number of completed tasks.
        successful_tasks: Number of successful tasks.
        failed_tasks: Number of failed tasks.
        current_task: Currently executing task (if any).
        start_time: When execution started.
    """

    total_tasks: int = 0
    completed_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    current_task: SubagentTask | None = None
    start_time: datetime | None = None
    listeners: list[ProgressListener] = field(default_factory=list)

    @property
    def progress_percent(self) -> float:
        """Get progress as a percentage."""
        if self.total_tasks == 0:
            return 100.0
        return (self.completed_tasks / self.total_tasks) * 100.0

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        delta = datetime.now(timezone.utc) - self.start_time
        return delta.total_seconds() * 1000.0

    def start(self, total: int) -> None:
        """Start tracking with the given number of tasks."""
        self.total_tasks = total
        self.completed_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.start_time = datetime.now(timezone.utc)

    def task_started(self, task: SubagentTask) -> None:
        """Record that a task has started."""
        self.current_task = task
        for listener in self.listeners:
            try:
                listener.on_task_started(task)
            except Exception:
                LOGGER.debug("Progress listener failed on task_started", exc_info=True)

    def task_completed(self, task: SubagentTask, result: SubagentResult) -> None:
        """Record that a task has completed."""
        self.completed_tasks += 1
        if result.success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
        self.current_task = None

        for listener in self.listeners:
            try:
                listener.on_task_completed(task, result)
                listener.on_progress(self.completed_tasks, self.total_tasks, None)
            except Exception:
                LOGGER.debug("Progress listener failed on task_completed", exc_info=True)


# =============================================================================
# Document Chunking
# =============================================================================


def estimate_tokens(text: str) -> int:
    """Estimate token count for text.

    Uses a simple heuristic of ~4 characters per token.
    """
    if not text:
        return 0
    return max(1, len(text.encode("utf-8")) // 4)


def find_chunk_boundaries(
    text: str,
    *,
    preferred_types: Sequence[str] = ("chapter", "scene", "paragraph"),
) -> list[ChunkBoundary]:
    """Find natural boundaries in text for chunk splitting.

    Args:
        text: The text to analyze.
        preferred_types: Boundary types to look for, in priority order.

    Returns:
        List of ChunkBoundary objects, sorted by char_index.
    """
    boundaries: list[ChunkBoundary] = []

    # Chapter markers (highest priority)
    chapter_pattern = re.compile(
        r"^\s*(Chapter|CHAPTER|Part|PART|Book|BOOK)\s+[\dIVXLCivxlc]+",
        re.MULTILINE,
    )
    for match in chapter_pattern.finditer(text):
        boundaries.append(ChunkBoundary(
            char_index=match.start(),
            boundary_type="chapter",
            score=1.0,
        ))

    # Scene breaks (*** or ### markers)
    scene_pattern = re.compile(r"^\s*(\*\*\*|###|---)\s*$", re.MULTILINE)
    for match in scene_pattern.finditer(text):
        boundaries.append(ChunkBoundary(
            char_index=match.start(),
            boundary_type="scene",
            score=0.9,
        ))

    # Double line breaks (paragraph boundaries)
    para_pattern = re.compile(r"\n\s*\n")
    for match in para_pattern.finditer(text):
        boundaries.append(ChunkBoundary(
            char_index=match.end(),
            boundary_type="paragraph",
            score=0.7,
        ))

    # Sort by position
    boundaries.sort(key=lambda b: b.char_index)

    return boundaries


def split_into_chunks(
    text: str,
    document_id: str,
    *,
    target_tokens: int = 4000,
    max_tokens: int = 6000,
    min_tokens: int = 500,
    overlap_tokens: int = 100,
) -> list[ChunkSpec]:
    """Split text into chunks for parallel processing.

    Args:
        text: The text to split.
        document_id: ID of the source document.
        target_tokens: Target token count per chunk.
        max_tokens: Maximum token count per chunk.
        min_tokens: Minimum token count per chunk.
        overlap_tokens: Token overlap between chunks for context.

    Returns:
        List of ChunkSpec objects.
    """
    if not text:
        return []

    total_tokens = estimate_tokens(text)

    # If small enough, return single chunk
    if total_tokens <= max_tokens:
        return [ChunkSpec(
            chunk_id=f"{document_id}-chunk-0",
            document_id=document_id,
            content=text,
            start_char=0,
            end_char=len(text),
            start_line=0,
            end_line=text.count("\n"),
            token_estimate=total_tokens,
        )]

    # Find natural boundaries
    boundaries = find_chunk_boundaries(text)

    # Convert target tokens to approximate character count
    chars_per_token = len(text) / max(1, total_tokens)
    target_chars = int(target_tokens * chars_per_token)
    overlap_chars = int(overlap_tokens * chars_per_token)

    chunks: list[ChunkSpec] = []
    current_start = 0
    chunk_index = 0

    while current_start < len(text):
        # Find ideal end position
        ideal_end = min(current_start + target_chars, len(text))

        # Look for a nearby boundary
        best_boundary = None
        search_start = max(current_start, ideal_end - target_chars // 4)
        search_end = min(len(text), ideal_end + target_chars // 4)

        for boundary in boundaries:
            if search_start <= boundary.char_index <= search_end:
                if best_boundary is None or boundary.score > best_boundary.score:
                    best_boundary = boundary

        # Use boundary if found, otherwise use ideal_end
        chunk_end = best_boundary.char_index if best_boundary else ideal_end

        # Ensure we don't create tiny chunks at the end
        remaining = len(text) - chunk_end
        if remaining > 0 and remaining < target_chars // 2:
            chunk_end = len(text)

        # Extract chunk content
        chunk_content = text[current_start:chunk_end]
        if not chunk_content.strip():
            current_start = chunk_end
            continue

        # Calculate line numbers
        start_line = text[:current_start].count("\n")
        end_line = start_line + chunk_content.count("\n")

        chunks.append(ChunkSpec(
            chunk_id=f"{document_id}-chunk-{chunk_index}",
            document_id=document_id,
            content=chunk_content,
            start_char=current_start,
            end_char=chunk_end,
            start_line=start_line,
            end_line=end_line,
            token_estimate=estimate_tokens(chunk_content),
        ))

        chunk_index += 1

        # Move to next position with overlap
        current_start = max(current_start + 1, chunk_end - overlap_chars)

    return chunks


# =============================================================================
# Chunk Coordinator
# =============================================================================


class ChunkCoordinator:
    """Coordinates document chunking and chunk processing.

    Handles:
    - Automatic document chunking based on size
    - Chunk boundary optimization
    - Result merging across chunks
    """

    def __init__(
        self,
        *,
        target_tokens: int = 4000,
        max_tokens: int = 6000,
        min_tokens: int = 500,
        overlap_tokens: int = 100,
        auto_chunk_threshold: int = 20000,  # chars
    ) -> None:
        """Initialize the coordinator.

        Args:
            target_tokens: Target tokens per chunk.
            max_tokens: Maximum tokens per chunk.
            min_tokens: Minimum tokens per chunk.
            overlap_tokens: Token overlap between chunks.
            auto_chunk_threshold: Character count above which to auto-chunk.
        """
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_tokens = overlap_tokens
        self.auto_chunk_threshold = auto_chunk_threshold

    def should_chunk(self, text: str) -> bool:
        """Determine if a document should be chunked."""
        return len(text) > self.auto_chunk_threshold

    def create_chunks(self, text: str, document_id: str) -> list[ChunkSpec]:
        """Create chunks from document text."""
        return split_into_chunks(
            text,
            document_id,
            target_tokens=self.target_tokens,
            max_tokens=self.max_tokens,
            min_tokens=self.min_tokens,
            overlap_tokens=self.overlap_tokens,
        )

    def merge_transformed_chunks(
        self,
        original_text: str,
        chunks: list[ChunkSpec],
        results: list[SubagentResult],
    ) -> str:
        """Merge transformed chunks back into a single document.

        Args:
            original_text: The original document text.
            chunks: The chunks that were processed.
            results: Results containing transformed content.

        Returns:
            Merged document text.
        """
        if not results:
            return original_text

        # Create a mapping of chunk_id to result
        result_map = {r.chunk_id: r for r in results if r.success}

        # Sort chunks by start position
        sorted_chunks = sorted(chunks, key=lambda c: c.start_char)

        # Build merged text
        merged_parts: list[str] = []
        last_end = 0

        for chunk in sorted_chunks:
            result = result_map.get(chunk.chunk_id)

            if result and "transformed_content" in result.output:
                # Use transformed content
                if chunk.start_char > last_end:
                    # Add any gap between chunks
                    merged_parts.append(original_text[last_end:chunk.start_char])
                merged_parts.append(result.output["transformed_content"])
            else:
                # Keep original content
                if chunk.start_char > last_end:
                    merged_parts.append(original_text[last_end:chunk.start_char])
                merged_parts.append(chunk.content)

            last_end = chunk.end_char

        # Add any remaining text after last chunk
        if last_end < len(original_text):
            merged_parts.append(original_text[last_end:])

        return "".join(merged_parts)


# =============================================================================
# Subagent Orchestrator
# =============================================================================


class SubagentExecutorProtocol(Protocol):
    """Protocol for executing subagent tasks."""

    async def execute(self, task: SubagentTask) -> SubagentResult:
        """Execute a single subagent task."""
        ...


class SubagentOrchestrator:
    """Orchestrates subagent execution with progress tracking.

    Handles:
    - Parallel execution of subagent tasks
    - Progress tracking and status updates
    - Error handling and partial failure recovery
    - Result aggregation
    """

    def __init__(
        self,
        *,
        executor: SubagentExecutorProtocol | None = None,
        max_parallel: int = 4,
        progress_tracker: ProgressTracker | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            executor: The executor for running subagent tasks.
            max_parallel: Maximum concurrent subagents.
            progress_tracker: Optional progress tracker.
        """
        self._executor = executor
        self.max_parallel = max_parallel
        self.progress_tracker = progress_tracker or ProgressTracker()

    def set_executor(self, executor: SubagentExecutorProtocol) -> None:
        """Set the subagent executor."""
        self._executor = executor

    async def run_tasks(
        self,
        tasks: Sequence[SubagentTask],
        *,
        parallel: bool = True,
    ) -> list[SubagentResult]:
        """Execute a list of subagent tasks.

        Args:
            tasks: Tasks to execute.
            parallel: If True, run tasks in parallel (up to max_parallel).

        Returns:
            List of results in task order.
        """
        if not tasks:
            return []

        if not self._executor:
            return [
                SubagentResult(
                    task_id=task.task_id,
                    success=False,
                    error="No executor configured",
                    chunk_id=task.chunk.chunk_id,
                )
                for task in tasks
            ]

        self.progress_tracker.start(len(tasks))

        if parallel and len(tasks) > 1:
            return await self._run_parallel(list(tasks))
        else:
            return await self._run_sequential(list(tasks))

    async def _run_parallel(self, tasks: list[SubagentTask]) -> list[SubagentResult]:
        """Run tasks in parallel with bounded concurrency."""
        results: list[SubagentResult | None] = [None] * len(tasks)
        semaphore = asyncio.Semaphore(self.max_parallel)

        async def run_with_semaphore(idx: int, task: SubagentTask) -> None:
            async with semaphore:
                result = await self._execute_task(task)
                results[idx] = result

        # Create tasks
        coros = [run_with_semaphore(i, task) for i, task in enumerate(tasks)]

        # Run all tasks
        await asyncio.gather(*coros, return_exceptions=True)

        # Replace None with error results
        return [
            r if r is not None else SubagentResult(
                task_id=tasks[i].task_id,
                success=False,
                error="Task execution failed",
                chunk_id=tasks[i].chunk.chunk_id,
            )
            for i, r in enumerate(results)
        ]

    async def _run_sequential(self, tasks: list[SubagentTask]) -> list[SubagentResult]:
        """Run tasks sequentially."""
        results: list[SubagentResult] = []

        for task in tasks:
            result = await self._execute_task(task)
            results.append(result)

        return results

    async def _execute_task(self, task: SubagentTask) -> SubagentResult:
        """Execute a single task with tracking."""
        self.progress_tracker.task_started(task)
        start_time = time.perf_counter()

        try:
            if self._executor is None:
                result = SubagentResult(
                    task_id=task.task_id,
                    success=False,
                    error="No executor configured",
                    chunk_id=task.chunk.chunk_id,
                )
            else:
                result = await asyncio.wait_for(
                    self._executor.execute(task),
                    timeout=task.timeout_seconds,
                )
                result.latency_ms = (time.perf_counter() - start_time) * 1000.0

        except asyncio.TimeoutError:
            result = SubagentResult(
                task_id=task.task_id,
                success=False,
                error=f"Task timed out after {task.timeout_seconds}s",
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                chunk_id=task.chunk.chunk_id,
            )
        except Exception as exc:
            result = SubagentResult(
                task_id=task.task_id,
                success=False,
                error=str(exc),
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                chunk_id=task.chunk.chunk_id,
            )
            LOGGER.warning("Subagent task %s failed: %s", task.task_id, exc)

        self.progress_tracker.task_completed(task, result)
        return result


# =============================================================================
# Result Aggregation
# =============================================================================


class ResultAggregator(ABC):
    """Base class for aggregating subagent results."""

    @abstractmethod
    def aggregate(self, results: list[SubagentResult]) -> dict[str, Any]:
        """Aggregate results into a single output."""
        ...


class AnalysisAggregator(ResultAggregator):
    """Aggregates analysis results from chunk analyzers.

    Combines findings from multiple chunks into a unified analysis.
    """

    def aggregate(self, results: list[SubagentResult]) -> dict[str, Any]:
        """Aggregate analysis results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        # Collect all findings
        all_characters: list[dict[str, Any]] = []
        all_themes: list[str] = []
        all_plot_points: list[dict[str, Any]] = []
        all_summaries: list[str] = []
        all_custom: list[dict[str, Any]] = []

        for result in successful:
            output = result.output
            if "characters" in output:
                all_characters.extend(output["characters"])
            if "themes" in output:
                all_themes.extend(output["themes"])
            if "plot_points" in output:
                all_plot_points.extend(output["plot_points"])
            if "summary" in output:
                all_summaries.append(output["summary"])
            if "custom" in output:
                all_custom.append(output["custom"])

        # Deduplicate characters by name
        unique_characters = {}
        for char in all_characters:
            name = char.get("name", "")
            if name:
                if name not in unique_characters:
                    unique_characters[name] = char
                else:
                    # Merge mentions
                    existing = unique_characters[name]
                    existing_mentions = existing.get("mentions", [])
                    new_mentions = char.get("mentions", [])
                    existing["mentions"] = existing_mentions + new_mentions

        # Deduplicate themes
        unique_themes = list(dict.fromkeys(all_themes))

        return {
            "status": "complete",
            "chunks_processed": len(successful),
            "chunks_failed": len(failed),
            "characters": list(unique_characters.values()),
            "themes": unique_themes,
            "plot_points": all_plot_points,
            "summary": " ".join(all_summaries) if all_summaries else None,
            "custom": all_custom if all_custom else None,
            "errors": [{"chunk_id": r.chunk_id, "error": r.error} for r in failed] if failed else None,
        }


class TransformAggregator(ResultAggregator):
    """Aggregates transformation results.

    Collects transformation outputs and statistics.
    """

    def aggregate(self, results: list[SubagentResult]) -> dict[str, Any]:
        """Aggregate transformation results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        total_replacements = 0
        total_tokens_used = 0

        for result in successful:
            output = result.output
            total_replacements += output.get("replacements", 0)
            total_tokens_used += result.tokens_used

        return {
            "status": "complete" if not failed else "partial",
            "chunks_processed": len(successful),
            "chunks_failed": len(failed),
            "total_replacements": total_replacements,
            "tokens_used": total_tokens_used,
            "errors": [{"chunk_id": r.chunk_id, "error": r.error} for r in failed] if failed else None,
        }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "SubagentType",
    "TaskPriority",
    # Data classes
    "ChunkSpec",
    "ChunkBoundary",
    "SubagentTask",
    "SubagentResult",
    # Progress
    "ProgressListener",
    "ProgressTracker",
    # Chunking
    "estimate_tokens",
    "find_chunk_boundaries",
    "split_into_chunks",
    "ChunkCoordinator",
    # Orchestration
    "SubagentExecutorProtocol",
    "SubagentOrchestrator",
    # Aggregation
    "ResultAggregator",
    "AnalysisAggregator",
    "TransformAggregator",
]
