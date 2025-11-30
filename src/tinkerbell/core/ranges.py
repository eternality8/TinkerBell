"""Structured helpers for representing text spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(slots=True, frozen=True)
class TextRange(Sequence[int]):
    """Canonical representation of a text selection using absolute offsets."""

    start: int
    end: int

    def __post_init__(self) -> None:
        start = self._coerce_index(self.start, "start")
        end = self._coerce_index(self.end, "end")
        if end < start:
            start, end = end, start
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @staticmethod
    def _coerce_index(value: Any, label: str) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"TextRange {label} must be an integer") from exc
        if number < 0:
            return 0
        return number

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return self.to_tuple()[index]
        if index == 0:
            return self.start
        if index == 1:
            return self.end
        raise IndexError("TextRange index out of range")

    def __iter__(self) -> Iterator[int]:
        yield self.start
        yield self.end

    @property
    def length(self) -> int:
        """Return the width of the range."""

        return self.end - self.start

    @property
    def is_caret(self) -> bool:
        """Return ``True`` when the range collapses to a caret."""

        return self.start == self.end

    def to_tuple(self) -> tuple[int, int]:
        """Return the range as a ``(start, end)`` tuple."""

        return (self.start, self.end)

    def to_list(self) -> list[int]:
        """Return the range as a JSON-friendly list."""

        return [self.start, self.end]

    def to_dict(self) -> dict[str, int]:
        """Return the range as an object compatible with directive schemas."""

        return {"start": self.start, "end": self.end}

    def as_payload(self, *, prefer_object: bool = True) -> list[int] | dict[str, int]:
        """Serialize the range for JSON payloads."""

        return self.to_dict() if prefer_object else self.to_list()

    def clamp(self, *, lower: int = 0, upper: int | None = None) -> TextRange:
        """Clamp the range to ``[lower, upper]`` bounds."""

        start = max(lower, self.start)
        end = max(lower, self.end)
        if upper is not None:
            start = min(start, upper)
            end = min(end, upper)
        return TextRange(start=start, end=end)

    def expand(self, *, before: int = 0, after: int = 0) -> TextRange:
        """Return a new range widened by ``before``/``after`` characters."""

        return TextRange(start=max(0, self.start - max(0, before)), end=self.end + max(0, after))

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        fallback: tuple[int, int] | None = None,
    ) -> TextRange:
        """Coerce ``value`` into a :class:`TextRange`."""

        if isinstance(value, TextRange):
            return value
        if value is None:
            if fallback is None:
                raise ValueError("TextRange value is required")
            return cls(*fallback)
        if isinstance(value, Mapping):
            start = value.get("start")
            end = value.get("end")
            if start is None or end is None:
                if fallback is None:
                    raise ValueError("TextRange mappings require start and end keys")
                if start is None:
                    start = fallback[0]
                if end is None:
                    end = fallback[1]
            return cls(start, end)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            seq = list(value)
            if len(seq) != 2:
                raise ValueError("TextRange sequences must have exactly two entries")
            return cls(seq[0], seq[1])
        start = getattr(value, "start", None)
        end = getattr(value, "end", None)
        if start is not None and end is not None:
            return cls(start, end)
        raise TypeError("Unsupported TextRange input")

    @classmethod
    def zero(cls) -> TextRange:
        """Return a caret-aligned range at offset 0."""

        return cls(0, 0)


@dataclass(slots=True, frozen=True)
class LineRange(Sequence[int]):
    """Line-based representation of a selection using inclusive bounds."""

    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        start = self._coerce_index(self.start_line, "start_line")
        end = self._coerce_index(self.end_line, "end_line")
        if end < start:
            start, end = end, start
        object.__setattr__(self, "start_line", start)
        object.__setattr__(self, "end_line", end)

    @staticmethod
    def _coerce_index(value: Any, label: str) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"LineRange {label} must be an integer") from exc
        if number < 0:
            return 0
        return number

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return self.to_tuple()[index]
        if index == 0:
            return self.start_line
        if index == 1:
            return self.end_line
        raise IndexError("LineRange index out of range")

    def __iter__(self) -> Iterator[int]:
        yield self.start_line
        yield self.end_line

    @property
    def line_count(self) -> int:
        """Return the number of lines covered by the span (inclusive)."""

        return (self.end_line - self.start_line) + 1

    def to_tuple(self) -> tuple[int, int]:
        """Return the span as a ``(start_line, end_line)`` tuple."""

        return (self.start_line, self.end_line)

    def to_list(self) -> list[int]:
        """Return the span as a JSON-friendly list."""

        return [self.start_line, self.end_line]

    def to_dict(self) -> dict[str, int]:
        """Return the span as an object with ``start_line``/``end_line`` keys."""

        return {"start_line": self.start_line, "end_line": self.end_line}

    def clamp(self, *, lower: int = 0, upper: int | None = None) -> "LineRange":
        """Clamp the span to ``[lower, upper]`` bounds when provided."""

        start = max(lower, self.start_line)
        end = max(lower, self.end_line)
        if upper is not None:
            start = min(start, upper)
            end = min(end, upper)
        return LineRange(start, end)

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        fallback: tuple[int, int] | None = None,
    ) -> "LineRange":
        """Coerce ``value`` into a :class:`LineRange`."""

        if isinstance(value, LineRange):
            return value
        if value is None:
            if fallback is None:
                raise ValueError("LineRange value is required")
            return cls(*fallback)
        if isinstance(value, Mapping):
            start = value.get("start_line")
            end = value.get("end_line")
            if start is None or end is None:
                if fallback is None:
                    raise ValueError("LineRange mappings require start_line and end_line")
                if start is None:
                    start = fallback[0]
                if end is None:
                    end = fallback[1]
            return cls(start, end)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            seq = list(value)
            if len(seq) != 2:
                raise ValueError("LineRange sequences must have exactly two entries")
            return cls(seq[0], seq[1])
        start = getattr(value, "start_line", None)
        end = getattr(value, "end_line", None)
        if start is not None and end is not None:
            return cls(start, end)
        raise TypeError("Unsupported LineRange input")

    @classmethod
    def zero(cls) -> "LineRange":
        """Return a caret-aligned line span at index 0."""

        return cls(0, 0)


__all__ = ["TextRange", "LineRange"]
