"""Telemetry service for the orchestration pipeline.

This module provides a service layer for recording telemetry events
during turn execution, wrapping the existing telemetry infrastructure.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..types import ToolCallRecord, TurnMetrics

__all__ = [
    "TelemetryService",
    "TelemetryConfig",
    "TelemetryEvent",
    "TelemetrySink",
    "InMemoryTelemetrySink",
]

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Sink Protocol
# -----------------------------------------------------------------------------


class TelemetrySink(Protocol):
    """Protocol for telemetry event sinks."""

    def record(self, event: "TelemetryEvent") -> None:
        """Record a telemetry event."""
        ...


# -----------------------------------------------------------------------------
# Telemetry Event
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class TelemetryEvent:
    """A telemetry event recorded during turn execution.

    Attributes:
        event_type: Type of event (turn, tool_call, error, etc.).
        timestamp: When the event occurred.
        turn_id: Optional turn identifier.
        document_id: Optional document identifier.
        payload: Event-specific data.
    """

    event_type: str
    timestamp: float = field(default_factory=time.time)
    turn_id: str | None = None
    document_id: str | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "turn_id": self.turn_id,
            "document_id": self.document_id,
            "payload": dict(self.payload),
        }


# -----------------------------------------------------------------------------
# In-Memory Sink
# -----------------------------------------------------------------------------


class InMemoryTelemetrySink:
    """Thread-safe in-memory telemetry sink for testing and inspection.

    Stores events in a ring buffer with configurable capacity.
    """

    def __init__(self, capacity: int = 200) -> None:
        """Initialize the sink.

        Args:
            capacity: Maximum number of events to retain.
        """
        self._capacity = max(10, capacity)
        self._events: list[TelemetryEvent] = []
        self._lock = Lock()

    @property
    def capacity(self) -> int:
        """Maximum event capacity."""
        return self._capacity

    def record(self, event: TelemetryEvent) -> None:
        """Record a telemetry event.

        Args:
            event: The event to record.
        """
        with self._lock:
            self._events.append(event)
            # Trim to capacity
            while len(self._events) > self._capacity:
                self._events.pop(0)

    def events(self) -> list[TelemetryEvent]:
        """Get all recorded events.

        Returns:
            List of events in chronological order.
        """
        with self._lock:
            return list(self._events)

    def events_by_type(self, event_type: str) -> list[TelemetryEvent]:
        """Get events filtered by type.

        Args:
            event_type: The event type to filter by.

        Returns:
            List of matching events.
        """
        with self._lock:
            return [e for e in self._events if e.event_type == event_type]

    def tail(self, limit: int | None = None) -> list[TelemetryEvent]:
        """Get the most recent events.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of most recent events.
        """
        with self._lock:
            if limit is None or limit >= len(self._events):
                return list(self._events)
            return self._events[-limit:]

    def clear(self) -> int:
        """Clear all events.

        Returns:
            Number of events cleared.
        """
        with self._lock:
            count = len(self._events)
            self._events.clear()
            return count

    def __len__(self) -> int:
        """Get the number of recorded events."""
        with self._lock:
            return len(self._events)

    def __bool__(self) -> bool:
        """Sink is always truthy (exists as a container)."""
        return True


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class TelemetryConfig:
    """Configuration for the telemetry service.

    Attributes:
        enabled: Whether telemetry is enabled.
        record_tool_calls: Whether to record individual tool calls.
        record_metrics: Whether to record turn metrics.
        sink_capacity: Capacity for in-memory sink.
    """

    enabled: bool = True
    record_tool_calls: bool = True
    record_metrics: bool = True
    sink_capacity: int = 200

    @classmethod
    def disabled(cls) -> "TelemetryConfig":
        """Create a disabled configuration."""
        return cls(enabled=False)


# -----------------------------------------------------------------------------
# Telemetry Service
# -----------------------------------------------------------------------------


class TelemetryService:
    """Service for recording telemetry during turn execution.

    Provides a high-level interface for recording various telemetry events
    including turn metrics, tool calls, and custom events.

    Example:
        >>> service = TelemetryService()
        >>> service.record_turn_start("turn-1", document_id="doc-1")
        >>> service.record_tool_call("turn-1", tool_record)
        >>> service.record_turn_complete("turn-1", metrics)
    """

    def __init__(
        self,
        config: TelemetryConfig | None = None,
        *,
        sink: TelemetrySink | None = None,
        external_emitter: Callable[[str, Mapping[str, Any]], Any] | None = None,
    ) -> None:
        """Initialize the telemetry service.

        Args:
            config: Telemetry configuration.
            sink: Custom telemetry sink. If not provided, creates an in-memory sink.
            external_emitter: Optional callback for external telemetry systems.
        """
        self._config = config or TelemetryConfig()
        self._sink = sink or InMemoryTelemetrySink(self._config.sink_capacity)
        self._external_emitter = external_emitter
        self._turn_count = 0
        self._tool_call_count = 0
        self._lock = Lock()

    @property
    def config(self) -> TelemetryConfig:
        """The service configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Whether telemetry is enabled."""
        return self._config.enabled

    @property
    def sink(self) -> TelemetrySink:
        """The telemetry sink."""
        return self._sink

    @property
    def turn_count(self) -> int:
        """Number of turns recorded."""
        with self._lock:
            return self._turn_count

    @property
    def tool_call_count(self) -> int:
        """Number of tool calls recorded."""
        with self._lock:
            return self._tool_call_count

    def record_turn_start(
        self,
        turn_id: str,
        *,
        document_id: str | None = None,
        model_name: str | None = None,
        **extra: Any,
    ) -> None:
        """Record the start of a turn.

        Args:
            turn_id: Unique identifier for the turn.
            document_id: Optional document context.
            model_name: Model being used.
            **extra: Additional metadata.
        """
        if not self._config.enabled:
            return

        event = TelemetryEvent(
            event_type="turn_start",
            turn_id=turn_id,
            document_id=document_id,
            payload={
                "model_name": model_name,
                **extra,
            },
        )
        self._record(event)

    def record_turn_complete(
        self,
        turn_id: str,
        metrics: TurnMetrics,
        *,
        document_id: str | None = None,
        success: bool = True,
        **extra: Any,
    ) -> None:
        """Record the completion of a turn.

        Args:
            turn_id: Unique identifier for the turn.
            metrics: Turn execution metrics.
            document_id: Optional document context.
            success: Whether the turn succeeded.
            **extra: Additional metadata.
        """
        if not self._config.enabled:
            return

        if not self._config.record_metrics:
            return

        with self._lock:
            self._turn_count += 1

        event = TelemetryEvent(
            event_type="turn_complete",
            turn_id=turn_id,
            document_id=document_id,
            payload={
                "success": success,
                "metrics": metrics.to_dict(),
                **extra,
            },
        )
        self._record(event)
        self._emit_external("turn_complete", event.to_dict())

    def record_turn_error(
        self,
        turn_id: str,
        error: str,
        *,
        document_id: str | None = None,
        error_type: str | None = None,
        **extra: Any,
    ) -> None:
        """Record a turn error.

        Args:
            turn_id: Unique identifier for the turn.
            error: Error message.
            document_id: Optional document context.
            error_type: Type/category of error.
            **extra: Additional metadata.
        """
        if not self._config.enabled:
            return

        event = TelemetryEvent(
            event_type="turn_error",
            turn_id=turn_id,
            document_id=document_id,
            payload={
                "error": error,
                "error_type": error_type,
                **extra,
            },
        )
        self._record(event)
        self._emit_external("turn_error", event.to_dict())

    def record_tool_call(
        self,
        turn_id: str,
        record: ToolCallRecord,
        *,
        document_id: str | None = None,
        **extra: Any,
    ) -> None:
        """Record a tool call.

        Args:
            turn_id: The turn this call belongs to.
            record: The tool call record.
            document_id: Optional document context.
            **extra: Additional metadata.
        """
        if not self._config.enabled:
            return

        if not self._config.record_tool_calls:
            return

        with self._lock:
            self._tool_call_count += 1

        event = TelemetryEvent(
            event_type="tool_call",
            turn_id=turn_id,
            document_id=document_id,
            payload={
                "tool": record.to_dict(),
                **extra,
            },
        )
        self._record(event)

    def record_analysis(
        self,
        turn_id: str,
        *,
        document_id: str | None = None,
        cache_hit: bool = False,
        duration_ms: float | None = None,
        **extra: Any,
    ) -> None:
        """Record analysis execution.

        Args:
            turn_id: The turn this analysis belongs to.
            document_id: Optional document context.
            cache_hit: Whether analysis came from cache.
            duration_ms: Analysis duration in milliseconds.
            **extra: Additional metadata.
        """
        if not self._config.enabled:
            return

        event = TelemetryEvent(
            event_type="analysis",
            turn_id=turn_id,
            document_id=document_id,
            payload={
                "cache_hit": cache_hit,
                "duration_ms": duration_ms,
                **extra,
            },
        )
        self._record(event)

    def record_budget_evaluation(
        self,
        turn_id: str,
        *,
        document_id: str | None = None,
        verdict: str = "ok",
        prompt_tokens: int = 0,
        budget_tokens: int = 0,
        **extra: Any,
    ) -> None:
        """Record a budget evaluation.

        Args:
            turn_id: The turn this evaluation belongs to.
            document_id: Optional document context.
            verdict: Budget decision (ok, needs_summary, reject).
            prompt_tokens: Tokens in the prompt.
            budget_tokens: Configured budget limit.
            **extra: Additional metadata.
        """
        if not self._config.enabled:
            return

        event = TelemetryEvent(
            event_type="budget_evaluation",
            turn_id=turn_id,
            document_id=document_id,
            payload={
                "verdict": verdict,
                "prompt_tokens": prompt_tokens,
                "budget_tokens": budget_tokens,
                **extra,
            },
        )
        self._record(event)

    def record_custom(
        self,
        event_type: str,
        *,
        turn_id: str | None = None,
        document_id: str | None = None,
        **payload: Any,
    ) -> None:
        """Record a custom telemetry event.

        Args:
            event_type: Type of event.
            turn_id: Optional turn context.
            document_id: Optional document context.
            **payload: Event payload.
        """
        if not self._config.enabled:
            return

        event = TelemetryEvent(
            event_type=event_type,
            turn_id=turn_id,
            document_id=document_id,
            payload=payload,
        )
        self._record(event)

    def get_events(
        self,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> Sequence[TelemetryEvent]:
        """Get recorded events.

        Args:
            event_type: Optional filter by event type.
            limit: Maximum number of events to return.

        Returns:
            List of matching events.
        """
        if isinstance(self._sink, InMemoryTelemetrySink):
            if event_type is not None:
                events = self._sink.events_by_type(event_type)
            else:
                events = self._sink.events()
            if limit is not None:
                events = events[-limit:]
            return events
        return []

    def summary(self) -> dict[str, Any]:
        """Get a summary of telemetry activity.

        Returns:
            Dictionary with telemetry summary.
        """
        event_count = 0
        if isinstance(self._sink, InMemoryTelemetrySink):
            event_count = len(self._sink)

        return {
            "enabled": self._config.enabled,
            "turn_count": self.turn_count,
            "tool_call_count": self.tool_call_count,
            "event_count": event_count,
        }

    def _record(self, event: TelemetryEvent) -> None:
        """Record an event to the sink."""
        try:
            self._sink.record(event)
        except Exception:
            LOGGER.debug("Failed to record telemetry event", exc_info=True)

    def _emit_external(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """Emit to external telemetry system."""
        if self._external_emitter is not None:
            try:
                self._external_emitter(event_type, payload)
            except Exception:
                LOGGER.debug("Failed to emit external telemetry", exc_info=True)
