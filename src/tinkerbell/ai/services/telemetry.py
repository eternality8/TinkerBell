"""Telemetry utilities for AI runtime instrumentation."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock
from typing import Iterable, Protocol, Sequence


@dataclass(slots=True)
class ContextUsageEvent:
    """Represents a single AI turn's token usage and metadata."""

    document_id: str | None
    model: str
    prompt_tokens: int
    tool_tokens: int
    response_reserve: int | None
    timestamp: float
    conversation_length: int
    tool_names: tuple[str, ...]
    run_id: str


class TelemetrySink(Protocol):
    """Sink interface used to collect telemetry events."""

    def record(self, event: ContextUsageEvent) -> None:
        ...


class InMemoryTelemetrySink:
    """Simple ring-buffer telemetry sink for local inspection and tests."""

    def __init__(self, capacity: int = 200) -> None:
        self._capacity = max(10, capacity)
        self._buffer: deque[ContextUsageEvent] = deque(maxlen=self._capacity)
        self._lock = Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def record(self, event: ContextUsageEvent) -> None:
        with self._lock:
            self._buffer.append(event)

    def tail(self, limit: int | None = None) -> list[ContextUsageEvent]:
        with self._lock:
            events = list(self._buffer)
        if limit is None or limit >= len(events):
            return events
        return events[-limit:]

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


def snapshot_events(sink: TelemetrySink, limit: int | None = None) -> Sequence[ContextUsageEvent]:
    """Best-effort helper to retrieve events from arbitrary sinks."""

    if hasattr(sink, "tail"):
        tail = getattr(sink, "tail")
        try:
            return list(tail(limit))  # type: ignore[misc]
        except TypeError:
            return list(tail())  # type: ignore[misc]
    raise NotImplementedError("Telemetry sink does not support snapshotting")


__all__ = ["ContextUsageEvent", "TelemetrySink", "InMemoryTelemetrySink", "snapshot_events"]
