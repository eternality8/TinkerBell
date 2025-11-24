"""Unified diff parser and patch application helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional, Sequence, Tuple

_PATCH_HEADER_RE = re.compile(
    r"@@\s+-?(?P<old_start>\d+)(?:,(?P<old_len>\d+))?\s+\+(?P<new_start>\d+)(?:,(?P<new_len>\d+))?\s+@@"
)


class PatchApplyError(RuntimeError):
    """Raised when a unified diff cannot be applied cleanly."""

    def __init__(
        self,
        message: str,
        *,
        reason: str = "context_mismatch",
        expected: str | None = None,
        actual: str | None = None,
        hunk_header: str | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.expected = expected
        self.actual = actual
        self.hunk_header = hunk_header

    def details(self) -> dict[str, str | None]:
        return {
            "reason": self.reason,
            "expected": self.expected,
            "actual": self.actual,
            "hunk": self.hunk_header,
        }


@dataclass(slots=True)
class PatchResult:
    """Result of applying a unified diff to a document."""

    text: str
    spans: Tuple[Tuple[int, int], ...]
    summary: str


@dataclass(slots=True)
class RangePatch:
    """Structured replacement descriptor for streamed multi-range patches."""

    start: int
    end: int
    replacement: str
    match_text: str
    chunk_id: Optional[str] = None
    chunk_hash: Optional[str] = None


@dataclass(slots=True)
class _HunkLine:
    op: str
    text: str


@dataclass(slots=True)
class _Hunk:
    header: str
    old_start: int
    new_start: int
    lines: Tuple[_HunkLine, ...]


def apply_unified_diff(original_text: str, diff: str) -> PatchResult:
    """Apply ``diff`` to ``original_text`` and return the patched document."""

    hunks, newline_hint, target_trailing_newline = _parse_unified_diff(diff)
    if not hunks:
        raise PatchApplyError("Diff does not contain any hunks", reason="empty_diff")

    lines, original_trailing_newline, newline = _split_lines(original_text)
    newline = newline_hint or newline
    result_lines: List[str] = []
    original_index = 0

    for hunk in hunks:
        hunk_index = _locate_hunk(lines, hunk, original_index)
        # Copy unchanged lines before this hunk
        while original_index < hunk_index and original_index < len(lines):
            result_lines.append(lines[original_index])
            original_index += 1

        temp_index = original_index
        for entry in hunk.lines:
            if entry.op == " ":
                if entry.text == "" and not lines:
                    continue
                _assert_line_matches(entry, lines, temp_index, hunk.header)
                if temp_index >= len(lines):
                    raise PatchApplyError(
                        "Context extends beyond end of document",
                        reason="unexpected_eof",
                        hunk_header=hunk.header,
                    )
                result_lines.append(lines[temp_index])
                temp_index += 1
            elif entry.op == "-":
                _assert_line_matches(entry, lines, temp_index, hunk.header)
                temp_index += 1
            elif entry.op == "+":
                result_lines.append(entry.text)
            else:  # pragma: no cover - invalid diff op guarded earlier
                raise PatchApplyError(
                    f"Unsupported diff opcode: {entry.op}", reason="invalid_opcode", hunk_header=hunk.header
                )

        original_index = temp_index

    # Copy any remaining lines after the last hunk
    result_lines.extend(lines[original_index:])

    trailing_newline = target_trailing_newline
    if trailing_newline is None:
        trailing_newline = original_trailing_newline

    patched_text = newline.join(result_lines)
    if trailing_newline and (result_lines or original_trailing_newline):
        patched_text += newline

    spans = _compute_spans(original_text, patched_text)
    summary = _summarize_patch(original_text, patched_text)
    return PatchResult(text=patched_text, spans=spans, summary=summary)


def apply_streamed_ranges(original_text: str, ranges: Sequence[RangePatch]) -> PatchResult:
    """Apply range-based replacements without constructing a unified diff."""

    if not ranges:
        raise PatchApplyError("Range patches require at least one entry", reason="empty_range_patch")

    normalized = tuple(sorted(ranges, key=lambda item: (item.start, item.end)))
    _ensure_non_overlapping(normalized)

    updated_text = original_text
    for entry in reversed(normalized):
        start = max(0, entry.start)
        end = max(0, entry.end)
        if end < start:
            start, end = end, start
        if end > len(updated_text):
            raise PatchApplyError(
                "Patch range exceeds document length",
                reason="range_overflow",
                expected=entry.match_text,
                actual=None,
            )
        current_slice = updated_text[start:end]
        if current_slice != entry.match_text:
            raise PatchApplyError(
                "Streamed patch range content mismatch",
                reason="range_mismatch",
                expected=entry.match_text,
                actual=current_slice,
            )
        updated_text = updated_text[:start] + entry.replacement + updated_text[end:]

    spans = _compute_spans(original_text, updated_text)
    summary = _summarize_patch(original_text, updated_text)
    return PatchResult(text=updated_text, spans=spans, summary=summary)


def _assert_line_matches(entry: _HunkLine, lines: Sequence[str], index: int, header: str) -> None:
    if index >= len(lines):
        raise PatchApplyError(
            "Patch context lines exceed document length",
            reason="context_overflow",
            actual=None,
            expected=entry.text,
            hunk_header=header,
        )
    probe = lines[index]
    if probe != entry.text:
        raise PatchApplyError(
            "Context mismatch while applying patch",
            reason="context_mismatch",
            expected=entry.text,
            actual=probe,
            hunk_header=header,
        )


def _split_lines(text: str) -> tuple[list[str], bool, str]:
    if not text:
        return [], False, "\n"
    newline = _detect_newline(text)
    trailing_newline = text.endswith(("\n", "\r"))
    lines = text.splitlines()
    if trailing_newline and text.endswith("\r\n"):
        newline = "\r\n"
    return lines, trailing_newline, newline


def _detect_newline(text: str) -> str:
    if "\r\n" in text:
        return "\r\n"
    if "\n" in text:
        return "\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _compute_spans(before: str, after: str) -> Tuple[Tuple[int, int], ...]:
    matcher = SequenceMatcher(a=before, b=after, autojunk=False)
    spans: list[tuple[int, int]] = []
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal" or j1 == j2:
            continue
        spans.append((j1, j2))
    return tuple(spans)


def _summarize_patch(before: str, after: str) -> str:
    delta = len(after) - len(before)
    if delta == 0:
        return "patch: Δ0"
    sign = "+" if delta > 0 else "-"
    return f"patch: {sign}{abs(delta)} chars"


def _locate_hunk(lines: Sequence[str], hunk: _Hunk, start_index: int) -> int:
    """Return the best-fit index for ``hunk`` starting at ``start_index``.

    Unified diffs generated from small snippets often contain inaccurate line
    numbers. We fall back to scanning for the hunk's context block so patches
    can still apply even when ``old_start`` drifts.
    """

    anchor = max(0, hunk.old_start - 1)
    if anchor < start_index:
        anchor = start_index

    old_lines = _hunk_old_lines(hunk)
    if not old_lines:
        return min(anchor, len(lines))

    match_index = _find_sequence(lines, old_lines, anchor)
    if match_index is None and anchor != start_index:
        match_index = _find_sequence(lines, old_lines, start_index)
    if match_index is None and start_index != 0:
        match_index = _find_sequence(lines, old_lines, 0)
        if match_index is not None and match_index < start_index:
            match_index = None

    if match_index is None:
        preview = old_lines[0] if old_lines else None
        if _sequence_occurs_multiple_times(lines, old_lines):
            raise PatchApplyError(
                "Context matched multiple locations; provide explicit range/anchors before retrying",
                reason="context_ambiguous",
                expected=preview,
                hunk_header=hunk.header,
            )
        raise PatchApplyError(
            "Context mismatch while applying patch",
            reason="context_mismatch",
            expected=preview,
            hunk_header=hunk.header,
        )

    return match_index


def _hunk_old_lines(hunk: _Hunk) -> Tuple[str, ...]:
    lines: list[str] = []
    blank_context_active = False
    for entry in hunk.lines:
        if entry.op == "-":
            lines.append(entry.text)
            blank_context_active = False
        elif entry.op == " ":
            if entry.text == "":
                if not blank_context_active:
                    lines.append("")
                    blank_context_active = True
            else:
                lines.append(entry.text)
                blank_context_active = False
        else:
            blank_context_active = False
    if any(segment != "" for segment in lines):
        return tuple(lines)
    return tuple()


def _find_sequence(lines: Sequence[str], needle: Sequence[str], start: int) -> int | None:
    if not needle:
        return start
    segment = list(needle)
    limit = len(lines) - len(segment) + 1
    if limit < start:
        return None
    for index in range(start, limit + 1):
        if lines[index : index + len(segment)] == segment:
            return index
    return None


def _sequence_occurs_multiple_times(lines: Sequence[str], needle: Sequence[str]) -> bool:
    if not needle:
        return False
    segment = list(needle)
    if not segment:
        return False
    limit = len(lines) - len(segment) + 1
    if limit <= 0:
        return False
    matches = 0
    for index in range(0, limit + 1):
        if lines[index : index + len(segment)] == segment:
            matches += 1
            if matches >= 2:
                return True
    return False


def _parse_unified_diff(diff: str) -> tuple[tuple[_Hunk, ...], str | None, bool | None]:
    lines = diff.splitlines()
    hunks: list[_Hunk] = []
    newline_hint: str | None = None
    target_trailing_newline: bool | None = None

    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("@@"):
            hunk, index, trailing_hint = _parse_hunk(lines, index)
            hunks.append(hunk)
            if trailing_hint is not None:
                target_trailing_newline = trailing_hint
            continue
        if line.startswith(("---", "+++", "diff", "index", "@@")):
            index += 1
            continue
        if not line.strip():
            index += 1
            continue
        index += 1

    # Attempt to infer newline preference from diff literal lines
    for hunk in hunks:
        for entry in hunk.lines:
            if "\r" in entry.text:
                newline_hint = "\r\n" if "\r\n" in entry.text else "\r"
                break
        if newline_hint:
            break

    return tuple(hunks), newline_hint, target_trailing_newline


def _parse_hunk(lines: Sequence[str], start_index: int) -> tuple[_Hunk, int, bool | None]:
    header = lines[start_index]
    match = _PATCH_HEADER_RE.match(header)
    if not match:
        raise PatchApplyError("Malformed hunk header", reason="invalid_header", hunk_header=header)

    old_start = int(match.group("old_start"))
    new_start = int(match.group("new_start"))

    index = start_index + 1
    hunk_lines: list[_HunkLine] = []
    trailing_newline_hint: bool | None = None
    last_op: str | None = None

    while index < len(lines):
        line = lines[index]
        if line.startswith("@@") and hunk_lines:
            break
        if line.startswith("\\ No newline"):
            if last_op == "+":
                trailing_newline_hint = False
            index += 1
            continue
        if not line:
            index += 1
            continue
        op = line[0]
        if op not in {" ", "+", "-"}:
            break
        text = line[1:]
        hunk_lines.append(_HunkLine(op=op, text=text))
        last_op = op
        index += 1

    if not hunk_lines:
        raise PatchApplyError("Hunk contains no body", reason="empty_hunk", hunk_header=header)

    hunk = _Hunk(header=header, old_start=old_start, new_start=new_start, lines=tuple(hunk_lines))
    return hunk, index, trailing_newline_hint


def _ensure_non_overlapping(ranges: Sequence[RangePatch]) -> None:
    previous_end = -1
    for entry in ranges:
        start = max(0, entry.start)
        end = max(0, entry.end)
        if end < start:
            start, end = end, start
        if start < previous_end:
            raise PatchApplyError("Range patches may not overlap", reason="range_overlap")
        previous_end = max(previous_end, end)


__all__ = [
    "PatchApplyError",
    "PatchResult",
    "apply_unified_diff",
    "apply_streamed_ranges",
    "RangePatch",
]
