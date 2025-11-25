"""Facade exposing read-only selection snapshots for AI tooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .workspace import DocumentWorkspace


@dataclass(slots=True)
class SelectionSnapshot:
    """Read-only view of the active editor selection."""

    tab_id: str | None
    document_id: str | None
    content_hash: str
    selection_start: int
    selection_end: int
    length: int
    line_start_offsets: tuple[int, ...]


class SelectionSnapshotProvider(Protocol):
    """Protocol implemented by selection gateways consumed by AI tools."""

    def capture(self, *, tab_id: str | None = None) -> SelectionSnapshot:
        ...


@dataclass(slots=True)
class SelectionGateway(SelectionSnapshotProvider):
    """Gateway confined to editor internals for live selection reads."""

    workspace: DocumentWorkspace

    def capture(self, *, tab_id: str | None = None) -> SelectionSnapshot:
        tab = self._resolve_tab(tab_id)
        document = tab.document()
        text = document.text or ""
        length = len(text)
        start, end = tab.editor.selection_span()
        start, end = self._clamp_range(start, end, length)
        line_start_offsets = self._normalize_offsets(text)
        return SelectionSnapshot(
            tab_id=tab.id,
            document_id=document.document_id,
            content_hash=document.content_hash,
            selection_start=start,
            selection_end=end,
            length=length,
            line_start_offsets=line_start_offsets,
        )

    def _resolve_tab(self, tab_id: str | None):
        if tab_id is None:
            return self.workspace.require_active_tab()
        return self.workspace.get_tab(tab_id)

    @staticmethod
    def _clamp_range(start: int, end: int, length: int) -> tuple[int, int]:
        start = max(0, min(int(start), length))
        end = max(0, min(int(end), length))
        if end < start:
            start, end = end, start
        return start, end

    @staticmethod
    def _normalize_offsets(text: str) -> tuple[int, ...]:
        offsets = [0]
        if text:
            cursor = 0
            for segment in text.splitlines(keepends=True):
                cursor += len(segment)
                offsets.append(cursor)
            if offsets[-1] < len(text):
                offsets.append(len(text))
        return tuple(_dedupe_non_decreasing(offsets))


def _dedupe_non_decreasing(values: Sequence[int]) -> list[int]:
    """Ensure offsets are monotonically increasing with no duplicates."""

    normalized: list[int] = []
    last = 0
    for value in values:
        cursor = max(0, int(value))
        if normalized and cursor < last:
            cursor = last
        normalized.append(cursor)
        last = cursor
    if not normalized:
        return [0]
    return normalized


__all__ = [
    "SelectionGateway",
    "SelectionSnapshot",
    "SelectionSnapshotProvider",
]
