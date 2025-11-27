"""Insert Lines Tool for AI operations.

Provides functionality to insert content at specific positions in documents
with version validation and drift recovery capabilities.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

from .base import WriteTool, ToolContext
from .errors import (
    InvalidLineRangeError,
    LineOutOfBoundsError,
    ContentRequiredError,
    NoMatchesError,
    TooManyMatchesError,
    ToolError,
)
from .version import VersionManager, VersionToken

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Drift Recovery
# -----------------------------------------------------------------------------


def find_anchor_text(
    lines: list[str],
    anchor_text: str,
    *,
    expected_line: int | None = None,
    search_radius: int = 20,
) -> tuple[int, float]:
    """Find anchor text in document, handling drift.

    Args:
        lines: Document lines.
        anchor_text: Text pattern to find.
        expected_line: Expected line number (for prioritization).
        search_radius: How far from expected line to search.

    Returns:
        Tuple of (found_line, confidence_score).

    Raises:
        NoMatchesError: If anchor text not found.
        TooManyMatchesError: If multiple matches found.
    """
    matches: list[tuple[int, float]] = []

    # Normalize anchor text for comparison
    anchor_normalized = anchor_text.strip()

    for i, line in enumerate(lines):
        line_normalized = line.strip()

        if anchor_normalized in line_normalized:
            # Calculate confidence based on match quality and distance
            if line_normalized == anchor_normalized:
                base_confidence = 1.0  # Exact match
            elif line_normalized.startswith(anchor_normalized):
                base_confidence = 0.9  # Prefix match
            else:
                base_confidence = 0.7  # Substring match

            # Apply distance penalty if expected line provided
            if expected_line is not None:
                distance = abs(i - expected_line)
                if distance == 0:
                    distance_confidence = 1.0
                elif distance <= search_radius:
                    distance_confidence = 1.0 - (distance / search_radius * 0.3)
                else:
                    distance_confidence = 0.5  # Outside radius but found
            else:
                distance_confidence = 1.0

            final_confidence = base_confidence * distance_confidence
            matches.append((i, final_confidence))

    if not matches:
        raise NoMatchesError(
            message=f"Anchor text not found: '{anchor_text[:50]}...' (truncated)" if len(anchor_text) > 50 else f"Anchor text not found: '{anchor_text}'",
            pattern=anchor_text,
        )

    if len(matches) > 1:
        # Sort by confidence and check if top match is significantly better
        matches.sort(key=lambda x: x[1], reverse=True)
        if matches[0][1] - matches[1][1] < 0.2:
            # Ambiguous - multiple similar-confidence matches
            raise TooManyMatchesError(
                message=f"Multiple matches for anchor text (found {len(matches)})",
                match_count=len(matches),
            )

    return matches[0]


def split_lines(text: str) -> list[str]:
    """Split text into lines, preserving empty last line."""
    if not text:
        return []
    # Split but don't create extra empty line for trailing newline
    lines = text.split("\n")
    return lines


# -----------------------------------------------------------------------------
# Document Editor Protocol
# -----------------------------------------------------------------------------


class DocumentEditor(Protocol):
    """Protocol for editing document content."""

    def set_document_text(self, tab_id: str, new_text: str) -> None:
        """Set the complete text content of a document."""
        ...


# -----------------------------------------------------------------------------
# Insert Lines Tool
# -----------------------------------------------------------------------------


@dataclass
class InsertLinesTool(WriteTool):
    """Tool for inserting content at a specific position in a document.

    Inserts new content after a specified line number. Never deletes
    existing content - only adds new lines.

    Parameters:
        tab_id: Target tab ID (optional, defaults to active tab).
        version: Version token from a previous read (required).
        after_line: Line number after which to insert (-1 for start of document).
            0-indexed, so after_line=0 inserts after the first line.
            Use after_line=-1 to insert at the very beginning.
        content: The content to insert (required).
        match_text: Optional anchor text for drift recovery.
            If provided and after_line has drifted, the tool will
            search for this text and adjust the insertion point.
        dry_run: If True, preview changes without applying (default False).

    Returns:
        inserted_at: Information about where content was inserted.
            - after_line: The line after which content was inserted.
            - lines_added: Number of lines added.
            - new_lines: {start, end} line range of inserted content.
        drift_detected: True if anchor text caused line adjustment.
        drift_from: Original line if drift was detected.
        version: New version token (or unchanged if dry_run).
    """

    name: ClassVar[str] = "insert_lines"
    summarizable: ClassVar[bool] = False

    version_manager: VersionManager
    document_editor: DocumentEditor | None = None

    def validate(self, params: dict[str, Any]) -> None:
        """Validate parameters."""
        super().validate(params)

        content = params.get("content")
        if content is None:
            raise ContentRequiredError(
                message="Parameter 'content' is required for insertion",
                field_name="content",
            )

    def write(
        self,
        context: ToolContext,
        params: dict[str, Any],
        token: VersionToken,
    ) -> dict[str, Any]:
        """Perform the insertion."""
        tab_id = token.tab_id
        after_line = params.get("after_line", -1)
        content = params["content"]
        match_text = params.get("match_text")

        # Get current document content
        doc_content = context.document_provider.get_document_content(tab_id)
        if doc_content is None:
            from .errors import TabNotFoundError
            raise TabNotFoundError(
                message=f"Document not found: {tab_id}",
                tab_id=tab_id,
            )

        lines = split_lines(doc_content)
        total_lines = len(lines) if lines else 0

        # Handle drift recovery
        drift_detected = False
        drift_from = None

        if match_text and after_line >= 0:
            try:
                found_line, confidence = find_anchor_text(
                    lines,
                    match_text,
                    expected_line=after_line,
                )
                if found_line != after_line:
                    drift_detected = True
                    drift_from = after_line
                    after_line = found_line
                    LOGGER.info(
                        "Drift detected: adjusted insertion from line %d to %d (confidence=%.2f)",
                        drift_from, after_line, confidence
                    )
            except (NoMatchesError, TooManyMatchesError):
                # If match_text fails, fall back to original line number
                LOGGER.warning(
                    "Anchor text not found, using original line %d", after_line
                )

        # Validate line range
        if after_line < -1:
            raise InvalidLineRangeError(
                message=f"after_line must be >= -1, got {after_line}",
                start_line=after_line,
            )

        if after_line >= total_lines and total_lines > 0:
            raise LineOutOfBoundsError(
                message=f"after_line {after_line} is beyond document end (total: {total_lines} lines)",
                line=after_line,
                total_lines=total_lines,
            )

        # Split content to insert into lines
        new_lines = split_lines(content)
        lines_added = len(new_lines)

        # Perform insertion
        insert_index = after_line + 1  # Insert after the specified line
        if total_lines == 0:
            # Empty document
            result_lines = new_lines
        else:
            result_lines = lines[:insert_index] + new_lines + lines[insert_index:]

        # Build new document text
        new_text = "\n".join(result_lines)

        # Apply edit if not dry-run and editor available
        if self.document_editor:
            self.document_editor.set_document_text(tab_id, new_text)

        # Calculate new line range
        new_lines_start = insert_index
        new_lines_end = insert_index + lines_added - 1

        result: dict[str, Any] = {
            "inserted_at": {
                "after_line": after_line,
                "lines_added": lines_added,
                "new_lines": {
                    "start": new_lines_start,
                    "end": new_lines_end,
                },
            },
            "_new_text": new_text,  # For version increment
        }

        if drift_detected:
            result["drift_detected"] = True
            result["drift_from"] = drift_from

        return result

    def preview(
        self,
        context: ToolContext,
        params: dict[str, Any],
        token: VersionToken,
    ) -> dict[str, Any]:
        """Preview the insertion without applying."""
        tab_id = token.tab_id
        after_line = params.get("after_line", -1)
        content = params["content"]
        match_text = params.get("match_text")

        # Get current document content
        doc_content = context.document_provider.get_document_content(tab_id)
        if doc_content is None:
            from .errors import TabNotFoundError
            raise TabNotFoundError(
                message=f"Document not found: {tab_id}",
                tab_id=tab_id,
            )

        lines = split_lines(doc_content)
        total_lines = len(lines) if lines else 0

        # Handle drift recovery
        drift_detected = False
        drift_from = None

        if match_text and after_line >= 0:
            try:
                found_line, confidence = find_anchor_text(
                    lines,
                    match_text,
                    expected_line=after_line,
                )
                if found_line != after_line:
                    drift_detected = True
                    drift_from = after_line
                    after_line = found_line
            except (NoMatchesError, TooManyMatchesError):
                pass

        # Split content to insert
        new_lines = split_lines(content)
        lines_added = len(new_lines)

        # Calculate new line range
        insert_index = after_line + 1
        new_lines_start = insert_index
        new_lines_end = insert_index + lines_added - 1

        # Build preview context
        context_before: list[str] = []
        context_after: list[str] = []

        if total_lines > 0:
            # Get up to 3 lines before insertion point
            start = max(0, insert_index - 3)
            context_before = lines[start:insert_index]

            # Get up to 3 lines after insertion point
            end = min(total_lines, insert_index + 3)
            context_after = lines[insert_index:end]

        # Build preview text
        preview_lines = (
            context_before +
            [f"+{line}" for line in new_lines[:5]] +  # Show first 5 new lines
            (["...(more lines)..."] if lines_added > 5 else []) +
            context_after
        )

        result: dict[str, Any] = {
            "preview": {
                "insertion_point": after_line,
                "lines_to_add": lines_added,
                "new_lines_range": {
                    "start": new_lines_start,
                    "end": new_lines_end,
                },
                "context": "\n".join(preview_lines),
            },
        }

        if drift_detected:
            result["drift_detected"] = True
            result["drift_from"] = drift_from

        return result
