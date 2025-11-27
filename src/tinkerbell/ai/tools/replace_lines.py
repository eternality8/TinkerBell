"""Replace Lines Tool for AI operations.

Provides functionality to replace a range of lines in documents
with version validation and drift recovery capabilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from .base import WriteTool, ToolContext
from .errors import (
    InvalidLineRangeError,
    LineOutOfBoundsError,
    NoMatchesError,
    TooManyMatchesError,
    TabNotFoundError,
)
from .version import VersionManager, VersionToken
from .insert_lines import find_anchor_text, split_lines, DocumentEditor

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Replace Lines Tool
# -----------------------------------------------------------------------------


@dataclass
class ReplaceLinesTool(WriteTool):
    """Tool for replacing a range of lines in a document.

    Replaces content between start_line and end_line (inclusive) with
    new content. If content is empty, this effectively deletes lines.

    Parameters:
        tab_id: Target tab ID (optional, defaults to active tab).
        version: Version token from a previous read (required).
        start_line: First line to replace (0-indexed, inclusive, required).
        end_line: Last line to replace (0-indexed, inclusive, required).
        content: Replacement content (empty string for deletion, default "").
        match_text: Optional anchor text for drift recovery.
            Should be text from the region being replaced.
        dry_run: If True, preview changes without applying (default False).

    Returns:
        lines_affected: Information about the change.
            - removed: Number of lines removed.
            - added: Number of lines added.
            - net_change: Net change in line count.
        drift_detected: True if anchor text caused line adjustment.
        drift_from: Original start line if drift was detected.
        version: New version token (or unchanged if dry_run).
    """

    name: ClassVar[str] = "replace_lines"
    summarizable: ClassVar[bool] = False

    version_manager: VersionManager
    document_editor: DocumentEditor | None = None

    def validate(self, params: dict[str, Any]) -> None:
        """Validate parameters."""
        super().validate(params)

        start_line = params.get("start_line")
        end_line = params.get("end_line")

        if start_line is None:
            from .errors import MissingParameterError
            raise MissingParameterError(
                message="Parameter 'start_line' is required",
                parameter="start_line",
            )

        if end_line is None:
            from .errors import MissingParameterError
            raise MissingParameterError(
                message="Parameter 'end_line' is required",
                parameter="end_line",
            )

        if not isinstance(start_line, int) or not isinstance(end_line, int):
            raise InvalidLineRangeError(
                message="start_line and end_line must be integers",
                start_line=start_line,
                end_line=end_line,
            )

        if start_line < 0:
            raise InvalidLineRangeError(
                message=f"start_line must be >= 0, got {start_line}",
                start_line=start_line,
                end_line=end_line,
            )

        if end_line < start_line:
            raise InvalidLineRangeError(
                message=f"end_line ({end_line}) cannot be less than start_line ({start_line})",
                start_line=start_line,
                end_line=end_line,
            )

    def write(
        self,
        context: ToolContext,
        params: dict[str, Any],
        token: VersionToken,
    ) -> dict[str, Any]:
        """Perform the replacement."""
        tab_id = token.tab_id
        start_line = params["start_line"]
        end_line = params["end_line"]
        content = params.get("content", "")
        match_text = params.get("match_text")

        # Get current document content
        doc_content = context.document_provider.get_document_content(tab_id)
        if doc_content is None:
            raise TabNotFoundError(
                message=f"Document not found: {tab_id}",
                tab_id=tab_id,
            )

        lines = split_lines(doc_content)
        total_lines = len(lines) if lines else 0

        # Handle empty document
        if total_lines == 0:
            if start_line != 0 or end_line != 0:
                raise LineOutOfBoundsError(
                    message="Cannot replace lines in empty document (use insert_lines instead)",
                    line=start_line,
                    total_lines=0,
                )
            # For empty doc, treat as insertion
            new_text = content
            lines_removed = 0
            new_lines = split_lines(content)
            lines_added = len(new_lines) if new_lines else 0
        else:
            # Handle drift recovery
            drift_detected = False
            drift_from = None

            if match_text:
                try:
                    found_line, confidence = find_anchor_text(
                        lines,
                        match_text,
                        expected_line=start_line,
                    )
                    if found_line != start_line:
                        # Calculate drift amount and adjust both start and end
                        drift_amount = found_line - start_line
                        drift_detected = True
                        drift_from = start_line
                        start_line = found_line
                        end_line = end_line + drift_amount
                        LOGGER.info(
                            "Drift detected: adjusted from lines %d-%d to %d-%d (confidence=%.2f)",
                            drift_from, params["end_line"], start_line, end_line, confidence
                        )
                except (NoMatchesError, TooManyMatchesError):
                    LOGGER.warning(
                        "Anchor text not found, using original lines %d-%d",
                        start_line, end_line
                    )

            # Validate line range against document
            if start_line >= total_lines:
                raise LineOutOfBoundsError(
                    message=f"start_line {start_line} is beyond document end (total: {total_lines} lines)",
                    line=start_line,
                    total_lines=total_lines,
                )

            # Clamp end_line to valid range
            end_line = min(end_line, total_lines - 1)

            # Calculate affected lines
            lines_removed = end_line - start_line + 1

            # Split replacement content
            new_lines = split_lines(content) if content else []
            lines_added = len(new_lines)

            # Perform replacement
            result_lines = lines[:start_line] + new_lines + lines[end_line + 1:]
            new_text = "\n".join(result_lines)

        # Apply edit if editor available
        if self.document_editor:
            self.document_editor.set_document_text(tab_id, new_text)

        result: dict[str, Any] = {
            "lines_affected": {
                "removed": lines_removed,
                "added": lines_added,
                "net_change": lines_added - lines_removed,
            },
            "_new_text": new_text,
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
        """Preview the replacement without applying."""
        tab_id = token.tab_id
        start_line = params["start_line"]
        end_line = params["end_line"]
        content = params.get("content", "")
        match_text = params.get("match_text")

        # Get current document content
        doc_content = context.document_provider.get_document_content(tab_id)
        if doc_content is None:
            raise TabNotFoundError(
                message=f"Document not found: {tab_id}",
                tab_id=tab_id,
            )

        lines = split_lines(doc_content)
        total_lines = len(lines) if lines else 0

        # Handle drift recovery
        drift_detected = False
        drift_from = None

        if match_text and total_lines > 0:
            try:
                found_line, confidence = find_anchor_text(
                    lines,
                    match_text,
                    expected_line=start_line,
                )
                if found_line != start_line:
                    drift_amount = found_line - start_line
                    drift_detected = True
                    drift_from = start_line
                    start_line = found_line
                    end_line = end_line + drift_amount
            except (NoMatchesError, TooManyMatchesError):
                pass

        # Clamp to valid range
        if total_lines > 0:
            end_line = min(end_line, total_lines - 1)

        # Calculate affected lines
        lines_removed = end_line - start_line + 1 if total_lines > 0 else 0
        new_lines = split_lines(content) if content else []
        lines_added = len(new_lines)

        # Build preview
        preview_parts: list[str] = []

        # Context before
        if total_lines > 0:
            ctx_start = max(0, start_line - 2)
            for i in range(ctx_start, start_line):
                preview_parts.append(f"  {i}: {lines[i]}")

        # Removed lines (show first few)
        removed_preview = min(3, lines_removed)
        if total_lines > 0:
            for i in range(removed_preview):
                if start_line + i < total_lines:
                    preview_parts.append(f"- {start_line + i}: {lines[start_line + i]}")
        if lines_removed > removed_preview:
            preview_parts.append(f"  ... ({lines_removed - removed_preview} more lines removed)")

        # Added lines (show first few)
        added_preview = min(3, lines_added)
        for i in range(added_preview):
            preview_parts.append(f"+ {start_line + i}: {new_lines[i]}")
        if lines_added > added_preview:
            preview_parts.append(f"  ... ({lines_added - added_preview} more lines added)")

        # Context after
        if total_lines > 0:
            ctx_end = min(total_lines, end_line + 3)
            for i in range(end_line + 1, ctx_end):
                preview_parts.append(f"  {i}: {lines[i]}")

        result: dict[str, Any] = {
            "preview": {
                "start_line": start_line,
                "end_line": end_line,
                "lines_removed": lines_removed,
                "lines_added": lines_added,
                "net_change": lines_added - lines_removed,
                "diff_preview": "\n".join(preview_parts),
            },
        }

        if drift_detected:
            result["drift_detected"] = True
            result["drift_from"] = drift_from

        return result
