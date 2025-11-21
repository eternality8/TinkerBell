"""Post-edit inspection utilities for detecting corrupted AI edits."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


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


class PostEditInspector:
    """Detects duplicate paragraphs and boundary corruption after patches."""

    _PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")

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

        duplicate = self._detect_duplicate_run(after_window)
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

    def _detect_duplicate_run(self, text: str) -> Mapping[str, object] | None:
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
