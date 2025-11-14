"""Optional telemetry helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TelemetryClient:
    """Minimal telemetry client placeholder."""

    enabled: bool = False

    def track_event(self, name: str, **props) -> None:
        del name, props

