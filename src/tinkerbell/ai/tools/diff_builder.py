"""Helper tool that converts raw text pairs into unified diffs."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any, ClassVar


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


__all__ = ["DiffBuilderTool"]
