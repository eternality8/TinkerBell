"""Pipeline stage: Finish.

This module provides the finish stage of the turn pipeline, responsible for
collecting metrics and assembling the final TurnOutput.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from ..types import (
    ModelResponse,
    TurnMetrics,
    TurnOutput,
    ToolCallRecord,
    AnalyzedTurn,
)
from .tools import ToolResults

__all__ = [
    "collect_metrics",
    "finish_turn",
    "finish_turn_with_error",
    "aggregate_tool_records",
    "calculate_total_tokens",
]


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


def calculate_total_tokens(
    prompt_tokens: int,
    completion_tokens: int,
) -> int:
    """Calculate total tokens from prompt and completion.

    Args:
        prompt_tokens: Tokens in the prompt.
        completion_tokens: Tokens in the completion.

    Returns:
        Total token count.
    """
    return prompt_tokens + completion_tokens


def aggregate_tool_records(
    *record_sequences: Sequence[ToolCallRecord],
) -> tuple[ToolCallRecord, ...]:
    """Aggregate multiple sequences of tool call records.

    Combines tool call records from multiple iterations into a single tuple.

    Args:
        *record_sequences: Variable number of record sequences to combine.

    Returns:
        Combined tuple of all tool call records.
    """
    all_records: list[ToolCallRecord] = []
    for seq in record_sequences:
        all_records.extend(seq)
    return tuple(all_records)


# -----------------------------------------------------------------------------
# Metrics Collection
# -----------------------------------------------------------------------------


def collect_metrics(
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    duration_ms: float = 0.0,
    tool_call_count: int = 0,
    iteration_count: int = 1,
    model_name: str | None = None,
    analysis_ran: bool = False,
    started_at: datetime | None = None,
    finished_at: datetime | None = None,
    response: ModelResponse | None = None,
    tool_results: ToolResults | None = None,
    analyzed_turn: AnalyzedTurn | None = None,
) -> TurnMetrics:
    """Collect and assemble turn execution metrics.

    This function can accept metrics directly or extract them from pipeline
    objects like ModelResponse, ToolResults, and AnalyzedTurn.

    Args:
        prompt_tokens: Tokens in the prompt (or extracted from response).
        completion_tokens: Tokens in the completion (or extracted from response).
        duration_ms: Total turn execution time in milliseconds.
        tool_call_count: Number of tool calls (or extracted from response/results).
        iteration_count: Number of tool loop iterations.
        model_name: Model used (or extracted from response).
        analysis_ran: Whether analysis was executed (or extracted from analyzed_turn).
        started_at: When the turn started.
        finished_at: When the turn finished.
        response: Optional ModelResponse to extract token counts from.
        tool_results: Optional ToolResults to extract tool counts from.
        analyzed_turn: Optional AnalyzedTurn to extract analysis info from.

    Returns:
        Assembled TurnMetrics.
    """
    # Extract from response if provided
    if response is not None:
        if prompt_tokens == 0:
            prompt_tokens = response.prompt_tokens
        if completion_tokens == 0:
            completion_tokens = response.completion_tokens
        if model_name is None:
            model_name = response.model
        if tool_call_count == 0:
            tool_call_count = len(response.tool_calls)

    # Extract from tool_results if provided
    if tool_results is not None:
        if tool_call_count == 0:
            tool_call_count = len(tool_results.results)

    # Extract from analyzed_turn if provided
    if analyzed_turn is not None:
        if not analysis_ran:
            analysis_ran = analyzed_turn.analysis_ran

    # Calculate total tokens
    total_tokens = calculate_total_tokens(prompt_tokens, completion_tokens)

    # Default timestamps
    if started_at is None:
        started_at = _utcnow()
    if finished_at is None:
        finished_at = _utcnow()

    return TurnMetrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        duration_ms=duration_ms,
        tool_call_count=tool_call_count,
        iteration_count=iteration_count,
        model_name=model_name,
        analysis_ran=analysis_ran,
        started_at=started_at,
        finished_at=finished_at,
    )


# -----------------------------------------------------------------------------
# Turn Finishing
# -----------------------------------------------------------------------------


def finish_turn(
    response: ModelResponse,
    tool_results: ToolResults | None = None,
    metrics: TurnMetrics | None = None,
    *,
    tool_records: Sequence[ToolCallRecord] = (),
    metadata: Mapping[str, Any] | None = None,
) -> TurnOutput:
    """Assemble the final TurnOutput from execution results.

    Combines the model response, tool execution results, and metrics into
    a TurnOutput that represents the completed turn.

    Args:
        response: The model's response.
        tool_results: Results from tool executions (if any).
        metrics: Collected turn metrics.
        tool_records: Pre-built tool call records (alternative to tool_results).
        metadata: Additional output metadata.

    Returns:
        The assembled TurnOutput.
    """
    # Build tool call records
    final_records: tuple[ToolCallRecord, ...]
    if tool_records:
        final_records = tuple(tool_records)
    elif tool_results is not None:
        # Convert tool results to records using response tool calls for arguments
        final_records = tool_results.to_records(response.tool_calls)
    else:
        final_records = ()

    # Use provided metrics or create minimal ones
    final_metrics = metrics or collect_metrics(response=response, tool_results=tool_results)

    return TurnOutput(
        response=response.text,
        tool_calls=final_records,
        metrics=final_metrics,
        success=True,
        error=None,
        metadata=dict(metadata) if metadata else {},
    )


def finish_turn_with_error(
    error: str | Exception,
    *,
    metrics: TurnMetrics | None = None,
    partial_response: str = "",
    tool_records: Sequence[ToolCallRecord] = (),
    metadata: Mapping[str, Any] | None = None,
) -> TurnOutput:
    """Create a TurnOutput representing a failed turn.

    Used when the turn encounters an unrecoverable error.

    Args:
        error: The error that occurred (string or exception).
        metrics: Collected metrics up to the point of failure.
        partial_response: Any partial response text received.
        tool_records: Any tool calls that were executed before failure.
        metadata: Additional output metadata.

    Returns:
        A TurnOutput with success=False and error details.
    """
    error_message = str(error) if not isinstance(error, str) else error

    # Ensure we have metrics
    final_metrics = metrics or TurnMetrics(finished_at=_utcnow())

    return TurnOutput(
        response=partial_response,
        tool_calls=tuple(tool_records),
        metrics=final_metrics,
        success=False,
        error=error_message,
        metadata=dict(metadata) if metadata else {},
    )


# -----------------------------------------------------------------------------
# Timing Helpers
# -----------------------------------------------------------------------------


class TurnTimer:
    """Context manager for timing turn execution.

    Usage:
        timer = TurnTimer()
        with timer:
            # ... execute turn ...
        metrics = collect_metrics(
            duration_ms=timer.duration_ms,
            started_at=timer.started_at,
            finished_at=timer.finished_at,
        )
    """

    def __init__(self) -> None:
        self.started_at: datetime | None = None
        self.finished_at: datetime | None = None
        self._start_perf: float = 0.0
        self._end_perf: float = 0.0

    def __enter__(self) -> TurnTimer:
        self.started_at = _utcnow()
        self._start_perf = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end_perf = time.perf_counter()
        self.finished_at = _utcnow()

    @property
    def duration_ms(self) -> float:
        """Get the duration in milliseconds."""
        if self._end_perf == 0.0:
            # Still running, get current duration
            return (time.perf_counter() - self._start_perf) * 1000
        return (self._end_perf - self._start_perf) * 1000

    def start(self) -> None:
        """Manually start the timer."""
        self.started_at = _utcnow()
        self._start_perf = time.perf_counter()

    def stop(self) -> None:
        """Manually stop the timer."""
        self._end_perf = time.perf_counter()
        self.finished_at = _utcnow()
