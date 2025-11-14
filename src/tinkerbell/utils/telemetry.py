"""Optional telemetry helpers."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

__all__ = ["TelemetryClient", "TelemetryEvent", "telemetry_enabled"]

_DEFAULT_TELEMETRY_DIR = Path.home() / ".tinkerbell" / "telemetry"
_TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(slots=True)
class TelemetryEvent:
    """Represents a single telemetry event prior to serialization."""

    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def serialize(self, session_id: str) -> str:
        payload = {
            "session_id": session_id,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "properties": self.properties,
        }
        return json.dumps(payload, default=_json_default, ensure_ascii=False)


@dataclass(slots=True)
class TelemetryClient:
    """Client that buffers events and flushes them as JSONL to disk when enabled."""

    enabled: bool = False
    storage_dir: Path | str | None = None
    max_buffer: int = 32
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    _buffer: list[TelemetryEvent] = field(default_factory=list, init=False, repr=False)

    def track_event(self, name: str, **props: Any) -> None:
        """Record an event if telemetry is enabled."""

        if not self.enabled:
            return
        event = TelemetryEvent(name=name, properties=_sanitize_props(props))
        self._buffer.append(event)
        if len(self._buffer) >= self.max_buffer:
            self.flush()

    def flush(self) -> Path | None:
        """Persist buffered events to disk and clear the buffer."""

        if not self.enabled or not self._buffer:
            return None

        target_dir = _resolve_storage_dir(self.storage_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        log_path = target_dir / "telemetry.jsonl"
        with log_path.open("a", encoding="utf-8") as handle:
            for event in self._buffer:
                handle.write(event.serialize(self.session_id))
                handle.write("\n")
        self._buffer.clear()
        return log_path

    def pending_events(self) -> int:
        """Return the number of events waiting to be flushed (primarily for tests)."""

        return len(self._buffer)


def telemetry_enabled(settings: Any | None = None) -> bool:
    """Return ``True`` if telemetry should be enabled for the current session."""

    env_value = os.environ.get("TINKERBELL_TELEMETRY")
    if env_value is not None:
        return env_value.strip().lower() in _TRUE_VALUES
    if settings is None:
        return False
    return bool(getattr(settings, "telemetry_opt_in", False))


def _sanitize_props(props: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    for key, value in props.items():
        if isinstance(value, Path):
            sanitized[key] = str(value)
        elif isinstance(value, datetime):
            sanitized[key] = value.isoformat()
        else:
            sanitized[key] = value
    return sanitized


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


def _resolve_storage_dir(storage_dir: Path | str | None) -> Path:
    env_override = os.environ.get("TINKERBELL_TELEMETRY_DIR")
    return Path(storage_dir or env_override or _DEFAULT_TELEMETRY_DIR).expanduser()


