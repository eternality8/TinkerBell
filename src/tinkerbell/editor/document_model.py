"""Dataclasses representing editor document state and snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""

    return datetime.now(timezone.utc)


@dataclass(slots=True)
class DocumentMetadata:
    """Metadata describing the currently loaded document."""

    path: Optional[Path] = None
    language: str = "markdown"
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class SelectionRange:
    """Represents the current selection inside the editor widget."""

    start: int = 0
    end: int = 0

    def as_tuple(self) -> tuple[int, int]:
        """Return the selection as a tuple for serialization."""

        return (self.start, self.end)


@dataclass(slots=True)
class DocumentState:
    """Full snapshot of the editor, used by the AI bridge."""

    text: str = ""
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    selection: SelectionRange = field(default_factory=SelectionRange)
    dirty: bool = False

    def update_text(self, new_text: str) -> None:
        """Update the document text and mark it dirty."""

        self.text = new_text
        self.dirty = True
        self.metadata.updated_at = _utcnow()

    def snapshot(self, *, delta_only: bool = False) -> Dict[str, Any]:
        """Return a serializable snapshot consumed by agent tools."""

        payload: Dict[str, Any] = {
            "text": self.text if not delta_only else "",
            "selection": self.selection.as_tuple(),
            "language": self.metadata.language,
            "dirty": self.dirty,
        }
        if self.metadata.path:
            payload["path"] = str(self.metadata.path)
        return payload
