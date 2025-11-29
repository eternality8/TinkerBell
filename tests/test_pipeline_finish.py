"""Tests for orchestration/pipeline/finish.py."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from tinkerbell.ai.orchestration.pipeline.finish import (
    TurnTimer,
    aggregate_tool_records,
    calculate_total_tokens,
    collect_metrics,
    finish_turn,
    finish_turn_with_error,
)
from tinkerbell.ai.orchestration.pipeline.tools import (
    ToolExecutionResult,
    ToolResults,
)
from tinkerbell.ai.orchestration.types import (
    AnalyzedTurn,
    BudgetEstimate,
    Message,
    ModelResponse,
    ParsedToolCall,
    PreparedTurn,
    ToolCallRecord,
    TurnMetrics,
    TurnOutput,
)


# -----------------------------------------------------------------------------
# Test Fixtures and Helpers
# -----------------------------------------------------------------------------


def make_model_response(
    text: str = "Response text",
    tool_calls: list[ParsedToolCall] | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    model: str | None = "gpt-4o",
) -> ModelResponse:
    """Helper to create a ModelResponse."""
    return ModelResponse(
        text=text,
        tool_calls=tuple(tool_calls) if tool_calls else (),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        model=model,
    )


def make_tool_results(
    results: list[tuple[str, str, bool]] | None = None,
) -> ToolResults:
    """Helper to create ToolResults.
    
    Args:
        results: List of (call_id, name, success) tuples.
    """
    if results is None:
        return ToolResults()
    
    exec_results = [
        ToolExecutionResult.from_success(call_id, name, f"result_{name}")
        if success
        else ToolExecutionResult.from_error(call_id, name, "failed")
        for call_id, name, success in results
    ]
    return ToolResults(results=tuple(exec_results))


def make_tool_call(
    name: str = "tool",
    call_id: str = "c1",
    arguments: str = "{}",
) -> ParsedToolCall:
    """Helper to create a ParsedToolCall."""
    return ParsedToolCall(call_id=call_id, name=name, arguments=arguments, index=0)


def make_tool_record(
    name: str = "tool",
    call_id: str = "c1",
    success: bool = True,
) -> ToolCallRecord:
    """Helper to create a ToolCallRecord."""
    return ToolCallRecord(
        call_id=call_id,
        name=name,
        arguments="{}",
        result="ok" if success else "error",
        success=success,
        duration_ms=10.0,
        error=None if success else "failed",
    )


def make_analyzed_turn(analysis_ran: bool = True) -> AnalyzedTurn:
    """Helper to create an AnalyzedTurn."""
    prepared = PreparedTurn(
        messages=(Message.user("test"),),
        budget=BudgetEstimate(
            prompt_tokens=100,
            completion_budget=4000,
            total_budget=128000,
        ),
    )
    return AnalyzedTurn(
        prepared=prepared,
        hints=("hint1",) if analysis_ran else (),
        analysis_ran=analysis_ran,
    )


# -----------------------------------------------------------------------------
# Tests: calculate_total_tokens
# -----------------------------------------------------------------------------


class TestCalculateTotalTokens:
    """Tests for calculate_total_tokens function."""

    def test_simple_addition(self) -> None:
        """Test basic token addition."""
        assert calculate_total_tokens(100, 50) == 150

    def test_zero_values(self) -> None:
        """Test with zero values."""
        assert calculate_total_tokens(0, 0) == 0
        assert calculate_total_tokens(100, 0) == 100
        assert calculate_total_tokens(0, 50) == 50

    def test_large_values(self) -> None:
        """Test with large token counts."""
        assert calculate_total_tokens(100_000, 50_000) == 150_000


# -----------------------------------------------------------------------------
# Tests: aggregate_tool_records
# -----------------------------------------------------------------------------


class TestAggregateToolRecords:
    """Tests for aggregate_tool_records function."""

    def test_empty_sequences(self) -> None:
        """Test with no sequences."""
        result = aggregate_tool_records()
        assert result == ()

    def test_single_sequence(self) -> None:
        """Test with single sequence."""
        records = [make_tool_record("t1", "c1"), make_tool_record("t2", "c2")]
        result = aggregate_tool_records(records)
        assert len(result) == 2

    def test_multiple_sequences(self) -> None:
        """Test combining multiple sequences."""
        seq1 = [make_tool_record("t1", "c1")]
        seq2 = [make_tool_record("t2", "c2"), make_tool_record("t3", "c3")]
        seq3 = [make_tool_record("t4", "c4")]
        result = aggregate_tool_records(seq1, seq2, seq3)
        assert len(result) == 4
        assert result[0].name == "t1"
        assert result[1].name == "t2"
        assert result[2].name == "t3"
        assert result[3].name == "t4"

    def test_returns_tuple(self) -> None:
        """Test that result is a tuple."""
        result = aggregate_tool_records([make_tool_record()])
        assert isinstance(result, tuple)


# -----------------------------------------------------------------------------
# Tests: collect_metrics
# -----------------------------------------------------------------------------


class TestCollectMetrics:
    """Tests for collect_metrics function."""

    def test_direct_values(self) -> None:
        """Test providing values directly."""
        metrics = collect_metrics(
            prompt_tokens=100,
            completion_tokens=50,
            duration_ms=1500.0,
            tool_call_count=3,
            iteration_count=2,
            model_name="gpt-4o",
            analysis_ran=True,
        )
        assert metrics.prompt_tokens == 100
        assert metrics.completion_tokens == 50
        assert metrics.total_tokens == 150
        assert metrics.duration_ms == 1500.0
        assert metrics.tool_call_count == 3
        assert metrics.iteration_count == 2
        assert metrics.model_name == "gpt-4o"
        assert metrics.analysis_ran is True

    def test_extract_from_response(self) -> None:
        """Test extracting values from ModelResponse."""
        response = make_model_response(
            prompt_tokens=200,
            completion_tokens=100,
            model="gpt-4o-mini",
            tool_calls=[make_tool_call("t1"), make_tool_call("t2")],
        )
        metrics = collect_metrics(response=response)
        assert metrics.prompt_tokens == 200
        assert metrics.completion_tokens == 100
        assert metrics.total_tokens == 300
        assert metrics.model_name == "gpt-4o-mini"
        assert metrics.tool_call_count == 2

    def test_extract_from_tool_results(self) -> None:
        """Test extracting tool count from ToolResults."""
        tool_results = make_tool_results([
            ("c1", "t1", True),
            ("c2", "t2", True),
            ("c3", "t3", False),
        ])
        metrics = collect_metrics(tool_results=tool_results)
        assert metrics.tool_call_count == 3

    def test_extract_from_analyzed_turn(self) -> None:
        """Test extracting analysis_ran from AnalyzedTurn."""
        analyzed = make_analyzed_turn(analysis_ran=True)
        metrics = collect_metrics(analyzed_turn=analyzed)
        assert metrics.analysis_ran is True

    def test_direct_values_override_extracted(self) -> None:
        """Test that direct values take precedence over extracted."""
        response = make_model_response(prompt_tokens=100)
        metrics = collect_metrics(
            prompt_tokens=200,  # Direct value should win
            response=response,
        )
        assert metrics.prompt_tokens == 200

    def test_timestamps_provided(self) -> None:
        """Test with provided timestamps."""
        start = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2025, 1, 1, 12, 0, 5, tzinfo=timezone.utc)
        metrics = collect_metrics(started_at=start, finished_at=end)
        assert metrics.started_at == start
        assert metrics.finished_at == end

    def test_timestamps_defaulted(self) -> None:
        """Test that timestamps default to now."""
        metrics = collect_metrics()
        assert metrics.started_at is not None
        assert metrics.finished_at is not None

    def test_returns_turn_metrics(self) -> None:
        """Test that return type is TurnMetrics."""
        metrics = collect_metrics()
        assert isinstance(metrics, TurnMetrics)


# -----------------------------------------------------------------------------
# Tests: finish_turn
# -----------------------------------------------------------------------------


class TestFinishTurn:
    """Tests for finish_turn function."""

    def test_basic_response(self) -> None:
        """Test finishing with basic response."""
        response = make_model_response(text="Hello!")
        output = finish_turn(response)
        assert output.response == "Hello!"
        assert output.success is True
        assert output.error is None
        assert output.tool_calls == ()

    def test_with_tool_results(self) -> None:
        """Test finishing with tool results."""
        response = make_model_response(
            text="Done",
            tool_calls=[make_tool_call("read_file", "c1")],
        )
        tool_results = make_tool_results([("c1", "read_file", True)])
        output = finish_turn(response, tool_results)
        assert output.response == "Done"
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0].name == "read_file"

    def test_with_metrics(self) -> None:
        """Test finishing with provided metrics."""
        response = make_model_response()
        metrics = TurnMetrics(
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
        )
        output = finish_turn(response, metrics=metrics)
        assert output.metrics.prompt_tokens == 500
        assert output.metrics.total_tokens == 700

    def test_with_pre_built_records(self) -> None:
        """Test finishing with pre-built tool records."""
        response = make_model_response()
        records = [
            make_tool_record("t1", "c1"),
            make_tool_record("t2", "c2"),
        ]
        output = finish_turn(response, tool_records=records)
        assert len(output.tool_calls) == 2

    def test_records_take_precedence_over_results(self) -> None:
        """Test that tool_records override tool_results."""
        response = make_model_response()
        tool_results = make_tool_results([("c1", "from_results", True)])
        records = [make_tool_record("from_records", "c2")]
        output = finish_turn(response, tool_results, tool_records=records)
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0].name == "from_records"

    def test_with_metadata(self) -> None:
        """Test finishing with metadata."""
        response = make_model_response()
        output = finish_turn(response, metadata={"key": "value", "num": 42})
        assert output.metadata["key"] == "value"
        assert output.metadata["num"] == 42

    def test_returns_turn_output(self) -> None:
        """Test that return type is TurnOutput."""
        response = make_model_response()
        output = finish_turn(response)
        assert isinstance(output, TurnOutput)

    def test_metrics_extracted_from_response(self) -> None:
        """Test that metrics are extracted from response when not provided."""
        response = make_model_response(
            prompt_tokens=100,
            completion_tokens=50,
            model="test-model",
        )
        output = finish_turn(response)
        assert output.metrics.prompt_tokens == 100
        assert output.metrics.completion_tokens == 50
        assert output.metrics.model_name == "test-model"


# -----------------------------------------------------------------------------
# Tests: finish_turn_with_error
# -----------------------------------------------------------------------------


class TestFinishTurnWithError:
    """Tests for finish_turn_with_error function."""

    def test_string_error(self) -> None:
        """Test with string error message."""
        output = finish_turn_with_error("Something went wrong")
        assert output.success is False
        assert output.error == "Something went wrong"
        assert output.response == ""

    def test_exception_error(self) -> None:
        """Test with exception object."""
        exc = ValueError("Invalid value")
        output = finish_turn_with_error(exc)
        assert output.success is False
        assert "Invalid value" in output.error

    def test_with_partial_response(self) -> None:
        """Test with partial response text."""
        output = finish_turn_with_error(
            "Network error",
            partial_response="I was saying...",
        )
        assert output.response == "I was saying..."

    def test_with_tool_records(self) -> None:
        """Test with tool records from before failure."""
        records = [make_tool_record("completed_tool")]
        output = finish_turn_with_error("Failed after tool", tool_records=records)
        assert len(output.tool_calls) == 1
        assert output.tool_calls[0].name == "completed_tool"

    def test_with_metrics(self) -> None:
        """Test with metrics captured before failure."""
        metrics = TurnMetrics(duration_ms=1000.0)
        output = finish_turn_with_error("Error", metrics=metrics)
        assert output.metrics.duration_ms == 1000.0

    def test_with_metadata(self) -> None:
        """Test with metadata."""
        output = finish_turn_with_error(
            "Error",
            metadata={"attempt": 3},
        )
        assert output.metadata["attempt"] == 3

    def test_default_metrics_have_finished_at(self) -> None:
        """Test that default metrics have finished_at set."""
        output = finish_turn_with_error("Error")
        assert output.metrics.finished_at is not None


# -----------------------------------------------------------------------------
# Tests: TurnTimer
# -----------------------------------------------------------------------------


class TestTurnTimer:
    """Tests for TurnTimer context manager."""

    def test_context_manager_usage(self) -> None:
        """Test using TurnTimer as context manager."""
        timer = TurnTimer()
        with timer:
            time.sleep(0.01)  # 10ms
        assert timer.started_at is not None
        assert timer.finished_at is not None
        assert timer.duration_ms > 0

    def test_duration_approximately_correct(self) -> None:
        """Test that duration is approximately correct."""
        timer = TurnTimer()
        with timer:
            time.sleep(0.05)  # 50ms
        # Allow some tolerance
        assert 40 < timer.duration_ms < 100

    def test_manual_start_stop(self) -> None:
        """Test manual start/stop methods."""
        timer = TurnTimer()
        timer.start()
        time.sleep(0.01)
        timer.stop()
        assert timer.started_at is not None
        assert timer.finished_at is not None
        assert timer.duration_ms > 0

    def test_duration_while_running(self) -> None:
        """Test getting duration while timer is running."""
        timer = TurnTimer()
        timer.start()
        time.sleep(0.01)
        # Should return current duration without stopping
        current = timer.duration_ms
        assert current > 0
        time.sleep(0.01)
        assert timer.duration_ms > current

    def test_timestamps_are_utc(self) -> None:
        """Test that timestamps are UTC."""
        timer = TurnTimer()
        with timer:
            pass
        assert timer.started_at.tzinfo == timezone.utc
        assert timer.finished_at.tzinfo == timezone.utc

    def test_initial_state(self) -> None:
        """Test initial timer state."""
        timer = TurnTimer()
        assert timer.started_at is None
        assert timer.finished_at is None


# -----------------------------------------------------------------------------
# Tests: Integration Scenarios
# -----------------------------------------------------------------------------


class TestIntegrationScenarios:
    """Integration tests for finish module."""

    def test_full_successful_turn(self) -> None:
        """Test assembling a fully successful turn."""
        # Simulate a turn with tool calls
        response = make_model_response(
            text="I've read the file for you.",
            tool_calls=[make_tool_call("read_file", "c1", '{"path": "/test.txt"}')],
            prompt_tokens=500,
            completion_tokens=100,
        )
        tool_results = make_tool_results([("c1", "read_file", True)])
        
        analyzed = make_analyzed_turn(analysis_ran=True)
        
        metrics = collect_metrics(
            response=response,
            tool_results=tool_results,
            analyzed_turn=analyzed,
            duration_ms=1500.0,
            iteration_count=1,
        )
        
        output = finish_turn(response, tool_results, metrics)
        
        assert output.success is True
        assert output.response == "I've read the file for you."
        assert len(output.tool_calls) == 1
        assert output.metrics.prompt_tokens == 500
        assert output.metrics.analysis_ran is True

    def test_multi_iteration_turn(self) -> None:
        """Test assembling a turn with multiple tool iterations."""
        # Simulate collecting records from multiple iterations
        iter1_records = [make_tool_record("t1", "c1")]
        iter2_records = [make_tool_record("t2", "c2"), make_tool_record("t3", "c3")]
        
        all_records = aggregate_tool_records(iter1_records, iter2_records)
        
        response = make_model_response(text="All done!")
        metrics = collect_metrics(
            tool_call_count=3,
            iteration_count=2,
        )
        
        output = finish_turn(response, metrics=metrics, tool_records=all_records)
        
        assert len(output.tool_calls) == 3
        assert output.metrics.iteration_count == 2

    def test_failed_turn_with_partial_work(self) -> None:
        """Test creating output for a failed turn with partial work."""
        # Some tools ran before failure
        completed_records = [make_tool_record("successful_tool", "c1")]
        
        metrics = collect_metrics(
            prompt_tokens=200,
            tool_call_count=1,
            duration_ms=500.0,
        )
        
        output = finish_turn_with_error(
            "Rate limit exceeded",
            metrics=metrics,
            partial_response="I started to...",
            tool_records=completed_records,
        )
        
        assert output.success is False
        assert output.error == "Rate limit exceeded"
        assert output.response == "I started to..."
        assert len(output.tool_calls) == 1
        assert output.metrics.prompt_tokens == 200
