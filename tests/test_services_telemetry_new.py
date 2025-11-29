"""Tests for the telemetry service.

This module tests the TelemetryService class that provides telemetry
recording for the turn pipeline.
"""

from __future__ import annotations

import time
import pytest
from typing import Any, Mapping

from tinkerbell.ai.orchestration.services.telemetry import (
    TelemetryService,
    TelemetryConfig,
    TelemetryEvent,
    TelemetrySink,
    InMemoryTelemetrySink,
)
from tinkerbell.ai.orchestration.types import ToolCallRecord, TurnMetrics


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config() -> TelemetryConfig:
    """Create a test telemetry configuration."""
    return TelemetryConfig()


@pytest.fixture
def service(config: TelemetryConfig) -> TelemetryService:
    """Create a telemetry service with test config."""
    return TelemetryService(config)


@pytest.fixture
def disabled_service() -> TelemetryService:
    """Create a disabled telemetry service."""
    return TelemetryService(TelemetryConfig.disabled())


@pytest.fixture
def metrics() -> TurnMetrics:
    """Create test turn metrics."""
    return TurnMetrics(
        prompt_tokens=1000,
        completion_tokens=500,
        total_tokens=1500,
        duration_ms=250.0,
        tool_call_count=3,
        model_name="test-model",
    )


@pytest.fixture
def tool_record() -> ToolCallRecord:
    """Create a test tool call record."""
    return ToolCallRecord(
        call_id="call-1",
        name="test_tool",
        arguments={"arg1": "value1"},
        result="success",
        success=True,
        duration_ms=50.0,
    )


# =============================================================================
# TelemetryConfig Tests
# =============================================================================


class TestTelemetryConfig:
    """Tests for TelemetryConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = TelemetryConfig()
        
        assert config.enabled is True
        assert config.record_tool_calls is True
        assert config.record_metrics is True
        assert config.sink_capacity == 200

    def test_custom_values(self):
        """Should accept custom values."""
        config = TelemetryConfig(
            enabled=False,
            record_tool_calls=False,
            record_metrics=False,
            sink_capacity=100,
        )
        
        assert config.enabled is False
        assert config.record_tool_calls is False
        assert config.record_metrics is False
        assert config.sink_capacity == 100

    def test_frozen(self):
        """Config should be frozen."""
        config = TelemetryConfig()
        
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore

    def test_disabled_factory(self):
        """disabled() should create disabled config."""
        config = TelemetryConfig.disabled()
        
        assert config.enabled is False


# =============================================================================
# TelemetryEvent Tests
# =============================================================================


class TestTelemetryEvent:
    """Tests for TelemetryEvent dataclass."""

    def test_creation(self):
        """Should create event with required fields."""
        event = TelemetryEvent(event_type="test")
        
        assert event.event_type == "test"
        assert event.timestamp > 0
        assert event.turn_id is None
        assert event.document_id is None
        assert event.payload == {}

    def test_creation_with_all_fields(self):
        """Should create event with all fields."""
        event = TelemetryEvent(
            event_type="test",
            timestamp=1234567890.0,
            turn_id="turn-1",
            document_id="doc-1",
            payload={"key": "value"},
        )
        
        assert event.event_type == "test"
        assert event.timestamp == 1234567890.0
        assert event.turn_id == "turn-1"
        assert event.document_id == "doc-1"
        assert event.payload == {"key": "value"}

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        event = TelemetryEvent(
            event_type="test",
            timestamp=1234567890.0,
            turn_id="turn-1",
            document_id="doc-1",
            payload={"key": "value"},
        )
        
        result = event.to_dict()
        
        assert result["event_type"] == "test"
        assert result["timestamp"] == 1234567890.0
        assert result["turn_id"] == "turn-1"
        assert result["document_id"] == "doc-1"
        assert result["payload"] == {"key": "value"}


# =============================================================================
# InMemoryTelemetrySink Tests
# =============================================================================


class TestInMemoryTelemetrySink:
    """Tests for InMemoryTelemetrySink."""

    def test_record_and_retrieve(self):
        """Should record and retrieve events."""
        sink = InMemoryTelemetrySink()
        event = TelemetryEvent(event_type="test")
        
        sink.record(event)
        
        assert len(sink) == 1
        assert sink.events()[0] is event

    def test_capacity_limit(self):
        """Should respect capacity limit."""
        sink = InMemoryTelemetrySink(capacity=10)
        
        for i in range(15):
            sink.record(TelemetryEvent(event_type=f"event-{i}"))
        
        assert len(sink) == 10
        # First 5 should have been evicted
        events = sink.events()
        assert events[0].event_type == "event-5"
        assert events[-1].event_type == "event-14"

    def test_events_by_type(self):
        """Should filter events by type."""
        sink = InMemoryTelemetrySink()
        sink.record(TelemetryEvent(event_type="a"))
        sink.record(TelemetryEvent(event_type="b"))
        sink.record(TelemetryEvent(event_type="a"))
        
        result = sink.events_by_type("a")
        
        assert len(result) == 2
        assert all(e.event_type == "a" for e in result)

    def test_tail(self):
        """Should return most recent events."""
        sink = InMemoryTelemetrySink()
        for i in range(10):
            sink.record(TelemetryEvent(event_type=f"event-{i}"))
        
        result = sink.tail(3)
        
        assert len(result) == 3
        assert result[0].event_type == "event-7"
        assert result[1].event_type == "event-8"
        assert result[2].event_type == "event-9"

    def test_tail_all(self):
        """tail(None) should return all events."""
        sink = InMemoryTelemetrySink()
        for i in range(5):
            sink.record(TelemetryEvent(event_type=f"event-{i}"))
        
        result = sink.tail(None)
        
        assert len(result) == 5

    def test_clear(self):
        """clear() should remove all events."""
        sink = InMemoryTelemetrySink()
        sink.record(TelemetryEvent(event_type="test"))
        sink.record(TelemetryEvent(event_type="test"))
        
        count = sink.clear()
        
        assert count == 2
        assert len(sink) == 0


# =============================================================================
# TelemetryService Basic Tests
# =============================================================================


class TestTelemetryServiceBasic:
    """Tests for basic TelemetryService operations."""

    def test_init_with_default_config(self):
        """Should initialize with default config."""
        service = TelemetryService()
        
        assert service.enabled is True
        assert service.turn_count == 0
        assert service.tool_call_count == 0

    def test_init_with_custom_config(self, config: TelemetryConfig):
        """Should initialize with custom config."""
        service = TelemetryService(config)
        
        assert service.config is config

    def test_init_disabled(self, disabled_service: TelemetryService):
        """Should initialize in disabled state."""
        assert disabled_service.enabled is False

    def test_init_with_custom_sink(self):
        """Should use custom sink if provided."""
        sink = InMemoryTelemetrySink(capacity=50)
        service = TelemetryService(sink=sink)
        
        assert service.sink is sink


# =============================================================================
# TelemetryService.record_turn_start Tests
# =============================================================================


class TestTelemetryServiceRecordTurnStart:
    """Tests for TelemetryService.record_turn_start."""

    def test_records_event(self, service: TelemetryService):
        """Should record turn start event."""
        service.record_turn_start("turn-1", document_id="doc-1", model_name="test")
        
        events = service.get_events("turn_start")
        assert len(events) == 1
        assert events[0].turn_id == "turn-1"
        assert events[0].document_id == "doc-1"
        assert events[0].payload["model_name"] == "test"

    def test_disabled_does_nothing(self, disabled_service: TelemetryService):
        """Should not record when disabled."""
        disabled_service.record_turn_start("turn-1")
        
        events = disabled_service.get_events()
        assert len(events) == 0


# =============================================================================
# TelemetryService.record_turn_complete Tests
# =============================================================================


class TestTelemetryServiceRecordTurnComplete:
    """Tests for TelemetryService.record_turn_complete."""

    def test_records_event(self, service: TelemetryService, metrics: TurnMetrics):
        """Should record turn complete event."""
        service.record_turn_complete("turn-1", metrics, document_id="doc-1")
        
        events = service.get_events("turn_complete")
        assert len(events) == 1
        assert events[0].turn_id == "turn-1"
        assert events[0].payload["success"] is True
        assert events[0].payload["metrics"]["prompt_tokens"] == 1000

    def test_increments_turn_count(self, service: TelemetryService, metrics: TurnMetrics):
        """Should increment turn count."""
        service.record_turn_complete("turn-1", metrics)
        service.record_turn_complete("turn-2", metrics)
        
        assert service.turn_count == 2

    def test_disabled_does_nothing(
        self, disabled_service: TelemetryService, metrics: TurnMetrics
    ):
        """Should not record when disabled."""
        disabled_service.record_turn_complete("turn-1", metrics)
        
        assert disabled_service.turn_count == 0

    def test_respects_record_metrics_config(self, metrics: TurnMetrics):
        """Should not record when record_metrics is False."""
        config = TelemetryConfig(record_metrics=False)
        service = TelemetryService(config)
        
        service.record_turn_complete("turn-1", metrics)
        
        events = service.get_events("turn_complete")
        assert len(events) == 0

    def test_calls_external_emitter(self, metrics: TurnMetrics):
        """Should call external emitter."""
        emissions: list[tuple[str, Mapping[str, Any]]] = []
        
        def emitter(event: str, payload: Mapping[str, Any]) -> None:
            emissions.append((event, payload))
        
        service = TelemetryService(external_emitter=emitter)
        service.record_turn_complete("turn-1", metrics)
        
        assert len(emissions) == 1
        assert emissions[0][0] == "turn_complete"


# =============================================================================
# TelemetryService.record_turn_error Tests
# =============================================================================


class TestTelemetryServiceRecordTurnError:
    """Tests for TelemetryService.record_turn_error."""

    def test_records_event(self, service: TelemetryService):
        """Should record turn error event."""
        service.record_turn_error("turn-1", "Something went wrong", error_type="RuntimeError")
        
        events = service.get_events("turn_error")
        assert len(events) == 1
        assert events[0].payload["error"] == "Something went wrong"
        assert events[0].payload["error_type"] == "RuntimeError"


# =============================================================================
# TelemetryService.record_tool_call Tests
# =============================================================================


class TestTelemetryServiceRecordToolCall:
    """Tests for TelemetryService.record_tool_call."""

    def test_records_event(
        self, service: TelemetryService, tool_record: ToolCallRecord
    ):
        """Should record tool call event."""
        service.record_tool_call("turn-1", tool_record)
        
        events = service.get_events("tool_call")
        assert len(events) == 1
        assert events[0].payload["tool"]["name"] == "test_tool"

    def test_increments_tool_call_count(
        self, service: TelemetryService, tool_record: ToolCallRecord
    ):
        """Should increment tool call count."""
        service.record_tool_call("turn-1", tool_record)
        service.record_tool_call("turn-1", tool_record)
        
        assert service.tool_call_count == 2

    def test_respects_record_tool_calls_config(self, tool_record: ToolCallRecord):
        """Should not record when record_tool_calls is False."""
        config = TelemetryConfig(record_tool_calls=False)
        service = TelemetryService(config)
        
        service.record_tool_call("turn-1", tool_record)
        
        events = service.get_events("tool_call")
        assert len(events) == 0


# =============================================================================
# TelemetryService.record_analysis Tests
# =============================================================================


class TestTelemetryServiceRecordAnalysis:
    """Tests for TelemetryService.record_analysis."""

    def test_records_event(self, service: TelemetryService):
        """Should record analysis event."""
        service.record_analysis(
            "turn-1",
            cache_hit=True,
            duration_ms=50.0,
        )
        
        events = service.get_events("analysis")
        assert len(events) == 1
        assert events[0].payload["cache_hit"] is True
        assert events[0].payload["duration_ms"] == 50.0


# =============================================================================
# TelemetryService.record_budget_evaluation Tests
# =============================================================================


class TestTelemetryServiceRecordBudgetEvaluation:
    """Tests for TelemetryService.record_budget_evaluation."""

    def test_records_event(self, service: TelemetryService):
        """Should record budget evaluation event."""
        service.record_budget_evaluation(
            "turn-1",
            verdict="ok",
            prompt_tokens=5000,
            budget_tokens=10000,
        )
        
        events = service.get_events("budget_evaluation")
        assert len(events) == 1
        assert events[0].payload["verdict"] == "ok"
        assert events[0].payload["prompt_tokens"] == 5000


# =============================================================================
# TelemetryService.record_custom Tests
# =============================================================================


class TestTelemetryServiceRecordCustom:
    """Tests for TelemetryService.record_custom."""

    def test_records_event(self, service: TelemetryService):
        """Should record custom event."""
        service.record_custom(
            "my_event",
            turn_id="turn-1",
            custom_field="value",
        )
        
        events = service.get_events("my_event")
        assert len(events) == 1
        assert events[0].payload["custom_field"] == "value"


# =============================================================================
# TelemetryService.get_events Tests
# =============================================================================


class TestTelemetryServiceGetEvents:
    """Tests for TelemetryService.get_events."""

    def test_get_all_events(self, service: TelemetryService):
        """Should get all events."""
        service.record_turn_start("turn-1")
        service.record_custom("other")
        
        events = service.get_events()
        
        assert len(events) == 2

    def test_get_events_by_type(self, service: TelemetryService):
        """Should filter by event type."""
        service.record_turn_start("turn-1")
        service.record_turn_start("turn-2")
        service.record_custom("other")
        
        events = service.get_events("turn_start")
        
        assert len(events) == 2

    def test_get_events_with_limit(self, service: TelemetryService):
        """Should respect limit."""
        for i in range(10):
            service.record_custom(f"event-{i}")
        
        events = service.get_events(limit=3)
        
        assert len(events) == 3


# =============================================================================
# TelemetryService.summary Tests
# =============================================================================


class TestTelemetryServiceSummary:
    """Tests for TelemetryService.summary."""

    def test_returns_summary(
        self,
        service: TelemetryService,
        metrics: TurnMetrics,
        tool_record: ToolCallRecord,
    ):
        """Should return summary dict."""
        service.record_turn_complete("turn-1", metrics)
        service.record_tool_call("turn-1", tool_record)
        
        summary = service.summary()
        
        assert summary["enabled"] is True
        assert summary["turn_count"] == 1
        assert summary["tool_call_count"] == 1
        assert summary["event_count"] == 2
