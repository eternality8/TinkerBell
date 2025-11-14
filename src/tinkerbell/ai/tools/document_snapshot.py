"""Tool returning the current document snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SnapshotProvider(Protocol):
    """Protocol implemented by the document bridge for retrieving snapshots."""

    def generate_snapshot(self, delta_only: bool = False) -> dict:
        ...


@dataclass(slots=True)
class DocumentSnapshotTool:
    """Simple synchronous tool returning document snapshots."""

    provider: SnapshotProvider

    def run(self, delta_only: bool = False) -> dict:
        return self.provider.generate_snapshot(delta_only=delta_only)

