"""Application-level telemetry helpers bridging AI usage data to the UI."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence
LOGGER = logging.getLogger(__name__)

_EVENT_LISTENERS: dict[str, list[Callable[[dict[str, Any]], None]]] = {}


from ..ai import client as ai_client
from ..ai.ai_types import TokenCounterProtocol
from ..ai.client import TokenCounterRegistry
_TELEMETRY_DIR = Path.home() / ".tinkerbell" / "telemetry"
_DEFAULT_TELEMETRY_PATH = _TELEMETRY_DIR / "context_usage.json"


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

    def record(self, event: ContextUsageEvent) -> None:  # pragma: no cover - protocol stub
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



@dataclass(slots=True)
class TokenCounterStatus:
    """Describes whether precise token counting is available."""

    precise: bool
    source: str


@dataclass(slots=True)
class TokenUsageSummary:
    """Summarized view of a context usage event for display widgets."""

    model: str
    prompt_tokens: int
    tool_tokens: int
    response_reserve: int | None
    tool_names: tuple[str, ...]

    @property
    def last_tool(self) -> str | None:
        return self.tool_names[-1] if self.tool_names else None

    def as_status_text(self) -> str:
        parts = [f"Prompt {self.prompt_tokens:,}", f"Tools {self.tool_tokens:,}"]
        if self.response_reserve:
            parts.append(f"Reserve {self.response_reserve:,}")
        if self.last_tool:
            parts.append(f"Last tool {self.last_tool}")
        return " · ".join(parts)


@dataclass(slots=True)
class TokenUsageTotals:
    """Aggregated token totals across multiple usage events."""

    prompt_tokens: int
    tool_tokens: int
    event_count: int

    def as_status_text(self) -> str:
        parts = [f"Σ Prompt {self.prompt_tokens:,}", f"Σ Tools {self.tool_tokens:,}"]
        parts.append(f"Events {self.event_count}")
        return " · ".join(parts)


@dataclass(slots=True)
class UsageDashboard:
    """Bundled summary/totals payload for context usage widgets."""

    summary: TokenUsageSummary
    totals: TokenUsageTotals

    @property
    def summary_text(self) -> str:
        return self.summary.as_status_text()

    @property
    def totals_text(self) -> str:
        return self.totals.as_status_text()


def get_token_counter_registry() -> TokenCounterRegistry:
    """Return the shared token counter registry used across the app."""

    return TokenCounterRegistry.global_instance()


def get_token_counter(model: str | None = None) -> TokenCounterProtocol:
    """Retrieve the preferred counter for a model (or fallback)."""

    registry = get_token_counter_registry()
    return registry.get(model)


def count_text_tokens(text: str, *, model: str | None = None, estimate_only: bool = False) -> int:
    """Count (or deterministically estimate) tokens for arbitrary text."""

    if not text:
        return 0
    counter = get_token_counter(model)
    if estimate_only:
        return counter.estimate(text)
    try:
        return counter.count(text)
    except Exception:
        return counter.estimate(text)


def token_counter_status() -> TokenCounterStatus:
    """Expose whether precise tokenizers (tiktoken) are available."""

    precise = getattr(ai_client, "tiktoken", None) is not None
    source = "tiktoken" if precise else "approximate byte counter (4 bytes ≈ 1 token)"
    return TokenCounterStatus(precise=precise, source=source)


def summarize_usage_event(event: ContextUsageEvent | None) -> TokenUsageSummary | None:
    """Convert a context usage event into a display-friendly summary."""

    if event is None:
        return None
    return TokenUsageSummary(
        model=event.model,
        prompt_tokens=max(0, int(event.prompt_tokens)),
        tool_tokens=max(0, int(event.tool_tokens)),
        response_reserve=event.response_reserve,
        tool_names=tuple(event.tool_names or ()),
    )


def summarize_usage_events(events: Sequence[ContextUsageEvent] | Iterable[ContextUsageEvent] | None) -> TokenUsageSummary | None:
    """Return the latest usage summary from an iterable of events."""

    if events is None:
        return None
    if isinstance(events, Sequence):
        if not events:
            return None
        return summarize_usage_event(events[-1])
    # Fallback for generic iterables
    latest: ContextUsageEvent | None = None
    for event in events:
        latest = event
    return summarize_usage_event(latest)


def format_usage_summary(events: Sequence[ContextUsageEvent] | Iterable[ContextUsageEvent] | None) -> str | None:
    """Return a human-friendly status string from usage events."""

    summary = summarize_usage_events(events)
    if summary is None:
        return None
    return summary.as_status_text()


def summarize_usage_totals(events: Sequence[ContextUsageEvent] | Iterable[ContextUsageEvent] | None) -> TokenUsageTotals | None:
    """Aggregate prompt/tool totals from recorded context usage events."""

    if events is None:
        return None
    prompt_total = 0
    tool_total = 0
    event_count = 0
    for event in events:
        if event is None:
            continue
        event_count += 1
        prompt_total += max(0, int(event.prompt_tokens))
        tool_total += max(0, int(event.tool_tokens))
    if event_count == 0:
        return None
    return TokenUsageTotals(prompt_tokens=prompt_total, tool_tokens=tool_total, event_count=event_count)


def build_usage_dashboard(events: Sequence[ContextUsageEvent] | Iterable[ContextUsageEvent] | None) -> UsageDashboard | None:
    """Return combined summary + totals payload for UI widgets."""

    if events is None:
        return None
    if not isinstance(events, Sequence):
        events = tuple(events)
    summary = summarize_usage_events(events)
    totals = summarize_usage_totals(events)
    if summary is None or totals is None:
        return None
    return UsageDashboard(summary=summary, totals=totals)


def register_event_listener(event_name: str, callback: Callable[[dict[str, Any]], None]) -> None:
    """Register a callback invoked whenever :func:`emit` fires *event_name*."""

    if not event_name or callback is None:
        return
    listeners = _EVENT_LISTENERS.setdefault(event_name, [])
    if callback not in listeners:
        listeners.append(callback)


def emit(event_name: str, payload: Mapping[str, Any] | None = None) -> None:
    """Broadcast a structured telemetry event to in-process listeners."""

    if not event_name:
        return
    event_payload = {"event": event_name}
    if payload:
        event_payload.update(payload)
    listeners = list(_EVENT_LISTENERS.get(event_name, ()))
    for callback in listeners:
        try:
            callback(dict(event_payload))
        except Exception:  # pragma: no cover - listeners must not break emitters
            LOGGER.debug("Telemetry listener %s failed", callback, exc_info=True)
    LOGGER.debug("Telemetry emit %s: %s", event_name, event_payload)


def default_telemetry_path() -> Path:
    """Return the default on-disk telemetry buffer location."""

    return _DEFAULT_TELEMETRY_PATH


def load_persistent_events(path: str | Path | None = None, *, limit: int | None = None) -> list[ContextUsageEvent]:
    """Load telemetry events from disk, returning at most *limit* entries."""

    target = Path(path or _DEFAULT_TELEMETRY_PATH)
    if not target.exists():
        return []
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    events: list[ContextUsageEvent] = []
    for payload in raw:
        event = _event_from_payload(payload)
        if event is not None:
            events.append(event)
    if limit is not None and limit > 0:
        return events[-limit:]
    return events


class PersistentTelemetrySink(TelemetrySink):
    """Telemetry sink that mirrors an in-memory buffer to disk."""

    def __init__(self, path: str | Path | None = None, *, capacity: int = 200) -> None:
        self._path = Path(path or _DEFAULT_TELEMETRY_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._memory = InMemoryTelemetrySink(capacity=capacity)
        self._lock = Lock()
        self._load_existing()

    def record(self, event: ContextUsageEvent) -> None:
        with self._lock:
            self._memory.record(event)
            self._flush_locked()

    def tail(self, limit: int | None = None) -> list[ContextUsageEvent]:
        return self._memory.tail(limit)

    def __len__(self) -> int:
        return len(self._memory)

    @property
    def path(self) -> Path:
        return self._path

    def _load_existing(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for payload in raw[-self._memory.capacity :]:
            event = _event_from_payload(payload)
            if event is not None:
                self._memory.record(event)

    def _flush_locked(self) -> None:
        events = self._memory.tail()
        serialized = [_event_to_payload(event) for event in events]
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")
        tmp_path.replace(self._path)


def _event_to_payload(event: ContextUsageEvent) -> dict[str, Any]:
    data = asdict(event)
    data["tool_names"] = list(event.tool_names)
    return data


def _event_from_payload(payload: object) -> ContextUsageEvent | None:
    if not isinstance(payload, dict):
        return None
    try:
        return ContextUsageEvent(
            document_id=payload.get("document_id"),
            model=str(payload.get("model") or ""),
            prompt_tokens=int(payload.get("prompt_tokens", 0)),
            tool_tokens=int(payload.get("tool_tokens", 0)),
            response_reserve=payload.get("response_reserve"),
            timestamp=float(payload.get("timestamp", 0.0)),
            conversation_length=int(payload.get("conversation_length", 0)),
            tool_names=tuple(payload.get("tool_names") or ()),
            run_id=str(payload.get("run_id") or ""),
        )
    except (TypeError, ValueError):
        return None


__all__ = [
    "ContextUsageEvent",
    "TokenCounterStatus",
    "TokenUsageSummary",
    "TokenUsageTotals",
    "UsageDashboard",
    "TelemetrySink",
    "InMemoryTelemetrySink",
    "count_text_tokens",
    "default_telemetry_path",
    "format_usage_summary",
    "get_token_counter",
    "get_token_counter_registry",
    "load_persistent_events",
    "PersistentTelemetrySink",
    "build_usage_dashboard",
    "snapshot_events",
    "summarize_usage_event",
    "summarize_usage_events",
    "summarize_usage_totals",
    "token_counter_status",
    "emit",
    "register_event_listener",
]
