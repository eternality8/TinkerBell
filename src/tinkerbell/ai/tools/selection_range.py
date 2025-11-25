"""Tool reporting the current selection range without mutating the caret."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import ClassVar, Sequence

from ...editor.selection_gateway import SelectionSnapshotProvider
from ...services.telemetry import emit as telemetry_emit


@dataclass(slots=True)
class SelectionRangeTool:
    """Return zero-based line bounds for the active selection."""

    gateway: SelectionSnapshotProvider
    summarizable: ClassVar[bool] = True

    def run(self, *, tab_id: str | None = None) -> dict[str, int | str]:
        snapshot = self.gateway.capture(tab_id=tab_id)
        selection = self._clamp_range(snapshot.selection_start, snapshot.selection_end, snapshot.length)
        offsets = self._normalize_offsets(snapshot.line_start_offsets, snapshot.length)
        start_line = self._line_for_offset(selection[0], offsets)
        end_anchor = selection[1] - 1 if selection[1] > selection[0] else selection[1]
        end_line = self._line_for_offset(max(end_anchor, selection[0]), offsets)
        payload = {
            "start_line": start_line,
            "end_line": end_line,
            "content_hash": snapshot.content_hash,
        }
        telemetry_emit(
            "span_snapshot_requested",
            {
                "document_id": snapshot.document_id,
                "tab_id": tab_id or snapshot.tab_id,
                "start_line": start_line,
                "end_line": end_line,
                "snapshot_span": {"start": selection[0], "end": selection[1]},
                "span_length": max(0, selection[1] - selection[0]),
                "content_hash": snapshot.content_hash,
            },
        )
        return payload

    @staticmethod
    def _line_for_offset(offset: int, offsets: Sequence[int]) -> int:
        if not offsets:
            return 0
        cursor = max(0, offset)
        index = bisect_right(offsets, cursor) - 1
        return max(0, index)

    @staticmethod
    def _clamp_range(start: int, end: int, length: int) -> tuple[int, int]:
        start_int = max(0, min(int(start), length))
        end_int = max(0, min(int(end), length))
        if end_int < start_int:
            start_int, end_int = end_int, start_int
        return start_int, end_int

    @staticmethod
    def _normalize_offsets(raw_offsets: Sequence[int], length: int) -> Sequence[int]:
        if not raw_offsets:
            return (0, length)
        normalized: list[int] = []
        for value in raw_offsets:
            try:
                cursor = int(value)
            except (TypeError, ValueError):
                continue
            cursor = max(0, min(cursor, length))
            if normalized and cursor < normalized[-1]:
                cursor = normalized[-1]
            normalized.append(cursor)
        if not normalized:
            normalized = [0, length]
        if normalized[0] != 0:
            normalized.insert(0, 0)
        if normalized[-1] != length:
            normalized.append(length)
        return normalized


__all__ = ["SelectionRangeTool"]
