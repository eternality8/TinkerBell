"""Helper tool that converts raw text pairs into unified diffs."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Callable, ClassVar, Mapping, Sequence


@dataclass(slots=True)
class DiffBuilderTool:
    """Build a unified diff string that can be fed into DocumentEdit patch calls."""

    default_filename: str = "document.txt"
    default_context_lines: int = 5
    summarizable: ClassVar[bool] = True

    def run(self, original: str, updated: str, *, filename: str | None = None, context: int | None = None) -> str:
        if original is None or updated is None:
            raise ValueError("Both original and updated text must be provided")

        source_name = self._normalize_filename(filename)
        context_lines = self._normalize_context(context)
        from_label = f"a/{source_name}"
        to_label = f"b/{source_name}"
        original_lines = original.splitlines(keepends=True)
        updated_lines = updated.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines,
            updated_lines,
            fromfile=from_label,
            tofile=to_label,
            lineterm="",
            n=context_lines,
        )
        diff_text = "\n".join(diff)
        if not diff_text.strip():
            raise ValueError("No differences detected between the provided texts")
        return diff_text

    def _normalize_context(self, value: int | None) -> int:
        candidate = self.default_context_lines if value is None else int(value)
        return max(0, candidate)

    def _normalize_filename(self, name: str | None) -> str:
        if isinstance(name, str) and name.strip():
            return name.strip()
        return self.default_filename


@dataclass(slots=True)
class StreamedEditRequest:
    """Represents a pending replacement referencing absolute document offsets."""

    start: int
    end: int
    replacement: str
    match_text: str | None = None
    chunk_id: str | None = None
    chunk_hash: str | None = None


@dataclass(slots=True)
class StreamedPatchRange:
    """Normalized range payload emitted by :class:`StreamedDiffBuilder`."""

    start: int
    end: int
    replacement: str
    match_text: str
    chunk_id: str | None = None
    chunk_hash: str | None = None


@dataclass(slots=True)
class StreamedDiffStats:
    """Lightweight telemetry describing a streamed diff payload."""

    range_count: int
    replaced_chars: int
    inserted_chars: int


@dataclass(slots=True)
class StreamedDiffResult:
    """Encapsulates the normalized ranges plus aggregate stats."""

    ranges: tuple[StreamedPatchRange, ...]
    stats: StreamedDiffStats


@dataclass(slots=True)
class StreamedDiffBuilder:
    """Build range-based patch payloads without materializing full documents."""

    summarizable: ClassVar[bool] = False

    def build(
        self,
        requests: Sequence[StreamedEditRequest],
        *,
        window_loader: Callable[[int, int], Mapping[str, object]],
        manifest: Mapping[str, object] | None = None,
    ) -> StreamedDiffResult:
        if not requests:
            raise ValueError("At least one edit range must be supplied")

        normalized = tuple(sorted(requests, key=lambda entry: (entry.start, entry.end)))
        self._validate_non_overlapping(normalized)

        ranges: list[StreamedPatchRange] = []
        replaced_total = 0
        inserted_total = 0
        for request in normalized:
            start, end = self._clamp_range(request.start, request.end)
            match_text = request.match_text
            chunk_id = request.chunk_id
            chunk_hash = request.chunk_hash
            if match_text is None and end > start:
                window = window_loader(start, end)
                match_text = self._extract_slice(window, start, end)
                resolved_chunk = self._resolve_chunk_metadata(window, manifest, start, end)
                chunk_id = chunk_id or resolved_chunk[0]
                chunk_hash = chunk_hash or resolved_chunk[1]
            elif match_text is None:
                match_text = ""
                if chunk_id is None or chunk_hash is None:
                    resolved_chunk = self._resolve_chunk_metadata(None, manifest, start, end)
                    chunk_id = chunk_id or resolved_chunk[0]
                    chunk_hash = chunk_hash or resolved_chunk[1]

            if match_text is None:
                raise ValueError("Streamed patch ranges require match_text for non-insert edits")

            ranges.append(
                StreamedPatchRange(
                    start=start,
                    end=end,
                    replacement=request.replacement,
                    match_text=match_text,
                    chunk_id=chunk_id,
                    chunk_hash=chunk_hash,
                )
            )
            replaced_total += len(match_text)
            inserted_total += len(request.replacement)

        stats = StreamedDiffStats(
            range_count=len(ranges),
            replaced_chars=replaced_total,
            inserted_chars=inserted_total,
        )
        return StreamedDiffResult(ranges=tuple(ranges), stats=stats)

    @staticmethod
    def _validate_non_overlapping(entries: Sequence[StreamedEditRequest]) -> None:
        previous_end = -1
        for entry in entries:
            start, end = StreamedDiffBuilder._clamp_range(entry.start, entry.end)
            if start < previous_end:
                raise ValueError("Streamed patch ranges may not overlap")
            previous_end = max(previous_end, end)

    @staticmethod
    def _clamp_range(start: int, end: int) -> tuple[int, int]:
        start = max(0, int(start))
        end = max(0, int(end))
        if end < start:
            start, end = end, start
        return start, end

    @staticmethod
    def _extract_slice(snapshot: Mapping[str, object], start: int, end: int) -> str:
        text = snapshot.get("text")
        if not isinstance(text, str):
            raise ValueError("Snapshot window did not include text content")
        window = snapshot.get("window")
        window_start = int(window.get("start", 0)) if isinstance(window, Mapping) else 0
        offset = max(0, start - window_start)
        length = max(0, end - start)
        slice_text = text[offset : offset + length]
        if len(slice_text) != length:
            raise ValueError("Snapshot window did not cover the requested range")
        return slice_text

    @staticmethod
    def _resolve_chunk_metadata(
        snapshot: Mapping[str, object] | None,
        manifest: Mapping[str, object] | None,
        start: int,
        end: int,
    ) -> tuple[str | None, str | None]:
        chunk = StreamedDiffBuilder._match_chunk(snapshot, start, end)
        if chunk is None:
            chunk = StreamedDiffBuilder._match_chunk(manifest, start, end)
        if chunk is None:
            return None, None
        chunk_id = chunk.get("id") if isinstance(chunk.get("id"), str) else None
        chunk_hash = chunk.get("hash") if isinstance(chunk.get("hash"), str) else None
        return chunk_id, chunk_hash

    @staticmethod
    def _match_chunk(source: Mapping[str, object] | None, start: int, end: int) -> Mapping[str, object] | None:
        if source is None:
            return None
        chunks = None
        if isinstance(source, Mapping):
            if isinstance(source.get("chunks"), Sequence):
                chunks = source.get("chunks")
            elif isinstance(source.get("chunk_manifest"), Mapping):
                manifest = source.get("chunk_manifest")
                chunks = manifest.get("chunks") if isinstance(manifest.get("chunks"), Sequence) else None
        if not isinstance(chunks, Sequence):
            return None
        for entry in chunks:
            if not isinstance(entry, Mapping):
                continue
            chunk_start = int(entry.get("start", 0))
            chunk_end = int(entry.get("end", chunk_start))
            if start >= chunk_start and end <= chunk_end:
                return entry
        return None


__all__ = [
    "DiffBuilderTool",
    "StreamedDiffBuilder",
    "StreamedDiffResult",
    "StreamedDiffStats",
    "StreamedEditRequest",
    "StreamedPatchRange",
]
