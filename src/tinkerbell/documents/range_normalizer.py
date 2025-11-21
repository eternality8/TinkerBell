"""Helpers for normalizing edit ranges prior to patch construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class NormalizedTextRange:
    """Represents an edit span widened to safe word/paragraph boundaries."""

    start: int
    end: int
    slice_text: str


def normalize_text_range(
    text: str,
    start: int,
    end: int,
    *,
    replacement: str | None = None,
) -> NormalizedTextRange:
    """Expand ``start``/``end`` to safe boundaries and capture the original slice."""

    document = text or ""
    length = len(document)
    clamped_start = _clamp(start, length)
    clamped_end = _clamp(end, length)
    if clamped_end < clamped_start:
        clamped_start, clamped_end = clamped_end, clamped_start

    expanded_start, expanded_end = _expand_word_boundaries(document, clamped_start, clamped_end)
    if expanded_end <= expanded_start:
        slice_text = document[expanded_start:expanded_end]
        return NormalizedTextRange(start=expanded_start, end=expanded_end, slice_text=slice_text)

    if _slice_contains_newline(document, expanded_start, expanded_end) or _text_contains_newline(replacement):
        expanded_start, expanded_end = _expand_paragraph_boundaries(document, expanded_start, expanded_end)

    slice_text = document[expanded_start:expanded_end]
    return NormalizedTextRange(start=expanded_start, end=expanded_end, slice_text=slice_text)


def compose_normalized_replacement(
    text: str,
    normalized: NormalizedTextRange,
    replacement: str,
    *,
    original_start: int,
    original_end: int,
) -> str:
    """Return a replacement string aligned to the widened span."""

    document = text or ""
    length = len(document)
    widened_start = max(0, min(normalized.start, length))
    widened_end = max(widened_start, min(normalized.end, length))

    orig_start = max(widened_start, min(int(original_start), widened_end))
    orig_end = max(orig_start, min(int(original_end), widened_end))

    prefix = document[widened_start:orig_start]
    suffix = document[orig_end:widened_end]
    return f"{prefix}{replacement}{suffix}"


def _clamp(value: int, length: int) -> int:
    return max(0, min(int(value), length))


def _expand_word_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    if not text:
        return start, end
    length = len(text)
    left = min(max(start, 0), length)
    while left > 0:
        prev_char = text[left - 1]
        curr_char = text[left] if left < length else ""
        if not (_is_word_char(prev_char) and _is_word_char(curr_char)):
            break
        left -= 1
    right = min(max(end, 0), length)
    while right < length and right > 0:
        prev_char = text[right - 1]
        curr_char = text[right]
        if not (_is_word_char(prev_char) and _is_word_char(curr_char)):
            break
        right += 1
    return left, min(right, length)


def _expand_paragraph_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    left = _find_previous_blank_line(text, start)
    right = _find_next_blank_line(text, end)
    return left, right


def _slice_contains_newline(text: str, start: int, end: int) -> bool:
    segment = text[start:end]
    return "\n" in segment or "\r" in segment


def _text_contains_newline(value: str | None) -> bool:
    if not value:
        return False
    return "\n" in value or "\r" in value


def _find_previous_blank_line(text: str, index: int) -> int:
    probe = max(0, min(index, len(text)))
    while probe > 0:
        newline = text.rfind("\n", 0, probe)
        if newline == -1:
            return 0
        cursor = newline - 1
        while cursor >= 0 and text[cursor] in {" ", "\t", "\r"}:
            cursor -= 1
        if cursor >= 0 and text[cursor] == "\n":
            return newline + 1
        probe = newline
    return 0


def _find_next_blank_line(text: str, index: int) -> int:
    length = len(text)
    probe = max(0, min(index, length))
    while probe < length:
        newline = text.find("\n", probe)
        if newline == -1:
            return length
        cursor = newline + 1
        while cursor < length and text[cursor] in {" ", "\t", "\r"}:
            cursor += 1
        if cursor < length and text[cursor] == "\n":
            return min(length, newline + 1)
        probe = newline + 1
    return length


def _is_word_char(char: str) -> bool:
    return char.isalnum() or char == "_"


__all__ = ["NormalizedTextRange", "compose_normalized_replacement", "normalize_text_range"]
