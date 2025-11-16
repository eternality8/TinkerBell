"""Dataclasses representing editor document state and snapshots."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""

    return datetime.now(timezone.utc)


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class DocumentVersion:
    """Lightweight metadata describing a document snapshot."""

    document_id: str
    version_id: int
    content_hash: str
    edited_ranges: tuple[tuple[int, int], ...] = ()


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
    document_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    version_id: int = 1
    content_hash: str = field(default_factory=str)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = _hash_text(self.text)

    def update_text(self, new_text: str) -> None:
        """Update the document text and mark it dirty."""

        self.text = new_text
        self.dirty = True
        self.metadata.updated_at = _utcnow()
        self.version_id += 1
        self.content_hash = _hash_text(new_text)

    def snapshot(self, *, delta_only: bool = False) -> Dict[str, Any]:
        """Return a serializable snapshot consumed by agent tools."""

        payload: Dict[str, Any] = {
            "text": self.text if not delta_only else "",
            "selection": self.selection.as_tuple(),
            "language": self.metadata.language,
            "dirty": self.dirty,
        }
        payload["document_id"] = self.document_id
        payload["version_id"] = self.version_id
        payload["content_hash"] = self.content_hash
        if self.metadata.path:
            payload["path"] = str(self.metadata.path)
        return payload

    def version_info(self, *, edited_ranges: tuple[tuple[int, int], ...] = ()) -> DocumentVersion:
        return DocumentVersion(
            document_id=self.document_id,
            version_id=self.version_id,
            content_hash=self.content_hash,
            edited_ranges=edited_ranges,
        )

    def version_signature(self) -> str:
        info = self.version_info()
        return f"{info.document_id}:{info.version_id}:{info.content_hash}"
