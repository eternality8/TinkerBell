"""Tool reporting the current selection range without mutating the caret."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence, cast

from .document_snapshot import SnapshotProvider
from ...services.telemetry import emit as telemetry_emit


@dataclass(slots=True)
class SelectionRangeTool:
    """Return zero-based line bounds for the active selection."""

    provider: SnapshotProvider
    summarizable: ClassVar[bool] = True

    def run(self, *, tab_id: str | None = None) -> dict[str, int | str]:
        snapshot = dict(self._invoke_snapshot(tab_id=tab_id))
        length = self._resolve_length(snapshot)
        selection = self._selection_span(snapshot.get("selection"), length)
        line_offsets = self._resolve_line_offsets(snapshot, length)
        start_line = self._line_for_offset(selection[0], line_offsets)
        end_anchor = selection[1] - 1 if selection[1] > selection[0] else selection[1]
        end_line = self._line_for_offset(max(end_anchor, selection[0]), line_offsets)
        content_hash = self._resolve_content_hash(snapshot)
        payload = {
            "start_line": start_line,
            "end_line": end_line,
            "content_hash": content_hash,
        }
        telemetry_emit(
            "selection_snapshot_requested",
            {
                "document_id": snapshot.get("document_id"),
                "tab_id": tab_id or snapshot.get("tab_id"),
                "start_line": start_line,
                "end_line": end_line,
                "selection_span": {"start": selection[0], "end": selection[1]},
                "selection_length": max(0, selection[1] - selection[0]),
                "content_hash": content_hash,
            },
        )
        return payload

    def _invoke_snapshot(self, *, tab_id: str | None) -> Mapping[str, Any]:
        snapshot_fn = getattr(self.provider, "generate_snapshot", None)
        if not callable(snapshot_fn):  # pragma: no cover - defensive guard
            raise ValueError("Snapshot provider is missing generate_snapshot()")
        kwargs = {"delta_only": False, "tab_id": tab_id, "include_text": False}
        try:
            result = snapshot_fn(**kwargs)
        except TypeError:
            kwargs.pop("tab_id")
            try:
                result = snapshot_fn(**kwargs)
            except TypeError:
                result = snapshot_fn(delta_only=False)
        return cast(Mapping[str, Any], result)

    @staticmethod
    def _resolve_length(snapshot: Mapping[str, Any]) -> int:
        token = snapshot.get("length")
        try:
            return max(0, int(token)) if token is not None else 0
        except (TypeError, ValueError):
            return 0

    def _selection_span(self, selection: Any, length: int) -> tuple[int, int]:
        start, end = 0, 0
        if isinstance(selection, Mapping):
            start = selection.get("start", 0)
            end = selection.get("end", 0)
        elif isinstance(selection, Sequence) and len(selection) == 2:
            start, end = selection[0], selection[1]
        return self._clamp_range(start, end, length)

    def _resolve_line_offsets(self, snapshot: Mapping[str, Any], length: int) -> list[int]:
        raw_offsets = snapshot.get("line_offsets")
        offsets: list[int]
        if isinstance(raw_offsets, Sequence):
            offsets = []
            for value in raw_offsets:
                try:
                    cursor = int(value)
                except (TypeError, ValueError):
                    continue
                cursor = max(0, cursor)
                if offsets and cursor < offsets[-1]:
                    cursor = offsets[-1]
                offsets.append(cursor)
        else:
            text = snapshot.get("text")
            if isinstance(text, str) and text:
                offsets = self._build_line_offsets(text)
            else:
                offsets = []
        if not offsets or offsets[0] != 0:
            offsets.insert(0, 0)
        if offsets[-1] < length:
            offsets.append(length)
        elif offsets[-1] > length:
            offsets[-1] = length
        return offsets

    @staticmethod
    def _build_line_offsets(text: str) -> list[int]:
        offsets = [0]
        cursor = 0
        for segment in text.splitlines(keepends=True):
            cursor += len(segment)
            offsets.append(cursor)
        if offsets[-1] < len(text):
            offsets.append(len(text))
        return offsets

    @staticmethod
    def _line_for_offset(offset: int, offsets: Sequence[int]) -> int:
        if not offsets:
            return 0
        cursor = max(0, offset)
        index = bisect_right(offsets, cursor) - 1
        return max(0, index)

    @staticmethod
    def _clamp_range(start: Any, end: Any, length: int) -> tuple[int, int]:
        try:
            start_int = int(start)
        except (TypeError, ValueError):
            start_int = 0
        try:
            end_int = int(end)
        except (TypeError, ValueError):
            end_int = start_int
        start_int = max(0, min(start_int, length))
        end_int = max(0, min(end_int, length))
        if end_int < start_int:
            start_int, end_int = end_int, start_int
        return start_int, end_int

    @staticmethod
    def _resolve_content_hash(snapshot: Mapping[str, Any]) -> str:
        token = snapshot.get("content_hash")
        if isinstance(token, str) and token.strip():
            return token.strip()
        raise ValueError("Snapshot did not expose content_hash; call document_snapshot before requesting selection_range")


__all__ = ["SelectionRangeTool"]
