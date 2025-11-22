"""Post-edit inspection utilities for detecting corrupted AI edits."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(slots=True)
class InspectionResult:
    """Represents the outcome of a post-edit inspection pass."""

    ok: bool
    reason: str | None = None
    detail: str | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _Paragraph:
    text: str
    normalized: str
    start: int
    end: int


@dataclass(slots=True)
class _Window:
    text: str
    normalized: str
    start: int
    end: int
    hash: str
    line_count: int
    line_index: int


class PostEditInspector:
    """Detects duplicate paragraphs and boundary corruption after patches."""

    _PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")
    _SPLIT_TOKEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"[a-z]{2}\n#"), "Word split before heading marker"),
        (re.compile(r"[a-z]{2}\n[A-Z]"), "Word split across newline without whitespace"),
    )
    _WINDOW_LINE_TARGET = 10
    _MIN_WINDOW_LINES = 4

    def __init__(
        self,
        *,
        duplicate_threshold: int = 2,
        token_drift: float = 0.05,
        window_padding: int = 256,
    ) -> None:
        self._duplicate_threshold = max(2, int(duplicate_threshold))
        self._token_drift = max(0.0, float(token_drift))
        self._window_padding = max(32, int(window_padding))

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def configure(
        self,
        *,
        duplicate_threshold: int | None = None,
        token_drift: float | None = None,
    ) -> None:
        if duplicate_threshold is not None:
            self._duplicate_threshold = max(2, int(duplicate_threshold))
        if token_drift is not None:
            self._token_drift = max(0.0, float(token_drift))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def inspect(
        self,
        *,
        before_text: str,
        after_text: str,
        spans: Sequence[tuple[int, int]] | None,
        range_hint: tuple[int, int] | None = None,
    ) -> InspectionResult:
        window_start, window_end = self._resolve_window(spans, len(after_text), range_hint)
        before_window = before_text[window_start:window_end]
        after_window = after_text[window_start:window_end]
        diagnostics: dict[str, object] = {
            "window": {"start": window_start, "end": window_end},
        }
        token_counts = {
            "before": self._count_tokens(before_window),
            "after": self._count_tokens(after_window),
        }
        diagnostics["token_counts"] = token_counts

        duplicate_window = self._detect_duplicate_window_run(after_window)
        if duplicate_window and not self._window_run_in_text(before_window, duplicate_window["hash"]):
            diagnostics["duplicate"] = duplicate_window
            detail = (
                f"Window repeated {duplicate_window['count']} times after edit (lines={duplicate_window['line_count']})"
            )
            return InspectionResult(
                ok=False,
                reason="duplicate_windows",
                detail=detail,
                diagnostics=diagnostics,
            )

        duplicate = self._detect_duplicate_paragraph_run(after_window)
        if duplicate and not self._duplicate_run_in_text(before_window, duplicate["normalized"], duplicate["count"]):
            diagnostics["duplicate"] = duplicate
            detail = f"Paragraph repeated {duplicate['count']} times after edit"
            return InspectionResult(
                ok=False,
                reason="duplicate_paragraphs",
                detail=detail,
                diagnostics=diagnostics,
            )

        boundary = self._detect_boundary_loss(before_window, after_window)
        if boundary is not None:
            diagnostics["boundary"] = boundary
            detail = boundary.get("detail") or "Paragraph boundary removed"
            return InspectionResult(
                ok=False,
                reason="boundary_dropped",
                detail=detail,
                diagnostics=diagnostics,
            )

        pattern_split = self._detect_split_token_patterns(after_window)
        if pattern_split is not None:
            diagnostics["split_pattern"] = pattern_split
            detail = pattern_split.get("detail") or "Split token heuristic triggered"
            return InspectionResult(
                ok=False,
                reason="split_token_regex",
                detail=detail,
                diagnostics=diagnostics,
            )

        split = self._detect_joined_tokens(after_text, spans)
        if split is not None:
            delta = self._token_delta_ratio(token_counts)
            diagnostics["split"] = {**split, "delta": delta}
            if delta >= self._token_drift:
                detail = split.get("detail") or "Tokens merged without whitespace"
                return InspectionResult(
                    ok=False,
                    reason="split_tokens",
                    detail=detail,
                    diagnostics=diagnostics,
                )

        return InspectionResult(ok=True, diagnostics=diagnostics)

    # ------------------------------------------------------------------
    # Detection helpers
    # ------------------------------------------------------------------
    def _resolve_window(
        self,
        spans: Sequence[tuple[int, int]] | None,
        text_length: int,
        range_hint: tuple[int, int] | None,
    ) -> tuple[int, int]:
        if range_hint is not None and (range_hint[0] or range_hint[1]):
            start = max(0, range_hint[0] - self._window_padding)
            end = min(text_length, range_hint[1] + self._window_padding)
            if start < end:
                return (start, end)
        if spans:
            starts = [max(0, span[0] - self._window_padding) for span in spans]
            ends = [min(text_length, span[1] + self._window_padding) for span in spans]
            return (min(starts), max(ends))
        return (0, min(text_length, 4_096))

    def _detect_duplicate_paragraph_run(self, text: str) -> Mapping[str, object] | None:
        paragraphs = self._extract_paragraphs(text)
        last_norm = ""
        run = 0
        anchor: _Paragraph | None = None
        for paragraph in paragraphs:
            normalized = paragraph.normalized
            if not normalized:
                run = 0
                last_norm = ""
                anchor = None
                continue
            if normalized == last_norm:
                run += 1
            else:
                run = 1
                last_norm = normalized
                anchor = paragraph
            if run >= self._duplicate_threshold:
                return {
                    "paragraph": (anchor.text if anchor else paragraph.text),
                    "normalized": normalized,
                    "count": run,
                }
        return None

    def _duplicate_run_in_text(self, text: str, normalized: str, count: int) -> bool:
        paragraphs = self._extract_paragraphs(text)
        run = 0
        for paragraph in paragraphs:
            if paragraph.normalized == normalized and normalized:
                run += 1
                if run >= count:
                    return True
            else:
                run = 0
        return False

    def _detect_duplicate_window_run(self, text: str) -> Mapping[str, object] | None:
        windows = self._build_line_windows(text)
        seen: dict[str, _Window] = {}
        max_gap = self._WINDOW_LINE_TARGET + self._MIN_WINDOW_LINES
        for window in windows:
            if window.line_count < self._MIN_WINDOW_LINES or not window.hash:
                continue
            previous = seen.get(window.hash)
            if previous is not None:
                line_gap = window.line_index - previous.line_index
                if line_gap <= max_gap:
                    return {
                        "hash": window.hash,
                        "line_count": window.line_count,
                        "count": 2,
                        "start": previous.start,
                        "end": window.end,
                        "line_gap": line_gap,
                    }
            else:
                seen[window.hash] = window
        return None

    def _window_run_in_text(self, text: str, window_hash: str) -> bool:
        if not window_hash:
            return False
        windows = self._build_line_windows(text)
        matches = sum(
            1
            for window in windows
            if window.hash == window_hash and window.line_count >= self._MIN_WINDOW_LINES
        )
        return matches >= self._duplicate_threshold

    def _detect_boundary_loss(self, before: str, after: str) -> Mapping[str, object] | None:
        before_breaks = self._count_boundaries(before)
        after_breaks = self._count_boundaries(after)
        if before_breaks > 0 and after_breaks == 0:
            return {
                "before": before_breaks,
                "after": after_breaks,
                "detail": "Edit removed the blank line separating paragraphs",
            }
        return None

    def _detect_joined_tokens(
        self,
        text: str,
        spans: Sequence[tuple[int, int]] | None,
    ) -> Mapping[str, object] | None:
        if not text or not spans:
            return None
        for start, end in spans:
            issue = self._check_boundary(text, start, end)
            if issue is not None:
                return issue
        return None

    def _detect_split_token_patterns(self, text: str) -> Mapping[str, object] | None:
        if not text:
            return None
        for pattern, description in self._SPLIT_TOKEN_PATTERNS:
            match = pattern.search(text)
            if match is None:
                continue
            return {
                "pattern": pattern.pattern,
                "match": match.group(0),
                "start": match.start(),
                "detail": description,
            }
        return None

    def _check_boundary(self, text: str, start: int, end: int) -> Mapping[str, object] | None:
        if 0 < start < len(text):
            left = text[start - 1]
            right = text[start]
            if self._is_token_char(left) and self._is_token_char(right):
                return {
                    "edge": "start",
                    "position": start,
                    "detail": "Missing whitespace before inserted text",
                }
        if 0 < end < len(text):
            left = text[end - 1]
            right = text[end]
            if self._is_token_char(left) and self._is_token_char(right):
                return {
                    "edge": "end",
                    "position": end,
                    "detail": "Missing whitespace after inserted text",
                }
        return None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _extract_paragraphs(self, text: str) -> list[_Paragraph]:
        if not text:
            return []
        segments: list[_Paragraph] = []
        cursor = 0
        for match in self._PARAGRAPH_BREAK.finditer(text):
            end = match.start()
            paragraph = self._build_paragraph(text[cursor:end], cursor, end)
            if paragraph is not None:
                segments.append(paragraph)
            cursor = match.end()
        if cursor <= len(text):
            paragraph = self._build_paragraph(text[cursor:], cursor, len(text))
            if paragraph is not None:
                segments.append(paragraph)
        return segments

    def _build_line_windows(self, text: str, lines_per_window: int | None = None) -> list[_Window]:
        if not text:
            return []
        target = lines_per_window or self._WINDOW_LINE_TARGET
        target = max(self._MIN_WINDOW_LINES, int(target))
        segments = text.splitlines(keepends=True)
        if not segments:
            return []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for segment in segments:
            start = cursor
            cursor += len(segment)
            offsets.append((start, cursor))
        windows: list[_Window] = []
        if len(segments) < self._MIN_WINDOW_LINES:
            window = self._build_window(segments, 0, offsets[-1][1], line_index=0)
            return [window] if window is not None else []
        step = 1
        for start_idx in range(0, len(segments), step):
            end_idx = min(len(segments), start_idx + target)
            if end_idx - start_idx < self._MIN_WINDOW_LINES:
                continue
            start = offsets[start_idx][0]
            end = offsets[end_idx - 1][1]
            window = self._build_window(segments[start_idx:end_idx], start, end, line_index=start_idx)
            if window is not None:
                windows.append(window)
        return windows

    def _build_window(self, buffer: Sequence[str], start: int, end: int, *, line_index: int) -> _Window | None:
        text = "".join(buffer).strip()
        if not text:
            return None
        normalized = " ".join(text.split())
        lowered = normalized.lower()
        window_hash = hashlib.sha1(lowered.encode("utf-8")).hexdigest() if lowered else ""
        return _Window(
            text=text,
            normalized=lowered,
            start=start,
            end=end,
            hash=window_hash,
            line_count=len(buffer),
            line_index=line_index,
        )

    def _build_paragraph(self, text: str, start: int, end: int) -> _Paragraph | None:
        stripped = text.strip()
        if not stripped:
            return None
        normalized = " ".join(stripped.split())
        return _Paragraph(text=stripped, normalized=normalized.lower(), start=start, end=end)

    def _count_boundaries(self, text: str) -> int:
        if not text:
            return 0
        return len(self._PARAGRAPH_BREAK.findall(text))

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(re.findall(r"\w+", text))

    def _token_delta_ratio(self, token_counts: Mapping[str, int]) -> float:
        before = max(1, int(token_counts.get("before", 0) or 0))
        after = max(0, int(token_counts.get("after", 0) or 0))
        delta = max(0, before - after)
        return delta / before if before else 0.0

    @staticmethod
    def _is_token_char(value: str) -> bool:
        return bool(value and value[0].isalnum())


__all__ = ["PostEditInspector", "InspectionResult"]
