"""Helper tool that converts raw text pairs into unified diffs."""

from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass(slots=True)
class DiffBuilderTool:
    """Build a unified diff string that can be fed into DocumentEdit patch calls."""

    default_filename: str = "document.txt"

    def run(self, original: str, updated: str, *, filename: str | None = None, context: int = 3) -> str:
        if original is None or updated is None:
            raise ValueError("Both original and updated text must be provided")

        source_name = filename or self.default_filename
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
            n=max(0, context),
        )
        diff_text = "\n".join(diff)
        if not diff_text.strip():
            raise ValueError("No differences detected between the provided texts")
        return diff_text


__all__ = ["DiffBuilderTool"]
