"""Tool returning the current document snapshot."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class SnapshotProvider(Protocol):
    """Protocol implemented by the document bridge for retrieving snapshots."""

    def generate_snapshot(self, *, delta_only: bool = False) -> Mapping[str, Any]:
        ...

    @property
    def last_diff_summary(self) -> str | None:
        ...

    @property
    def last_snapshot_version(self) -> str | None:
        ...


@dataclass(slots=True)
class DocumentSnapshotTool:
    """Simple synchronous tool returning document snapshots."""

    provider: SnapshotProvider

    def run(self, delta_only: bool = False, include_diff: bool = True) -> dict:
        snapshot = deepcopy(dict(self.provider.generate_snapshot(delta_only=delta_only)))

        if include_diff:
            diff = getattr(self.provider, "last_diff_summary", None)
            if diff is not None:
                snapshot["diff_summary"] = diff

        version = snapshot.get("version") or getattr(self.provider, "last_snapshot_version", None)
        if version is not None:
            snapshot["version"] = version

        return snapshot

