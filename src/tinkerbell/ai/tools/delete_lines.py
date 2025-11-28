"""Delete Lines Tool for AI operations.

Provides functionality to delete a range of lines from documents.
Implemented as a specialized wrapper around replace_lines with empty content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from .base import WriteTool, ToolContext
from .errors import (
    InvalidLineRangeError,
    LineOutOfBoundsError,
    TabNotFoundError,
    MissingParameterError,
    NoMatchesError,
    TooManyMatchesError,
)
from .version import VersionManager, VersionToken
from .insert_lines import find_anchor_text, split_lines, DocumentEditor

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Delete Lines Tool
# -----------------------------------------------------------------------------


@dataclass
class DeleteLinesTool(WriteTool):
    """Tool for deleting a range of lines from a document.

    Deletes content between start_line and end_line (inclusive).
    This is semantically equivalent to replace_lines with empty content,
    but provides a clearer API for deletion operations.

    Parameters:
        tab_id: Target tab ID (optional, defaults to active tab).
        version: Version token from a previous read (required).
        start_line: First line to delete (0-indexed, inclusive, required).
        end_line: Last line to delete (0-indexed, inclusive, required).
        match_text: Optional anchor text for drift recovery.
            Should be text from the region being deleted.
        dry_run: If True, preview changes without applying (default False).

    Returns:
        lines_deleted: Number of lines removed.
        deleted_content: The content that was deleted (for undo reference).
        version: New version token (or unchanged if dry_run).
    """

    name: ClassVar[str] = "delete_lines"
    summarizable: ClassVar[bool] = False

    version_manager: VersionManager
    document_editor: DocumentEditor | None = None

    def validate(self, params: dict[str, Any]) -> None:
        """Validate parameters."""
        super().validate(params)

        start_line = params.get("start_line")
        end_line = params.get("end_line")

        if start_line is None:
            raise MissingParameterError(
                message="Parameter 'start_line' is required",
                parameter="start_line",
            )

        if end_line is None:
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
        """Perform the deletion."""
        tab_id = token.tab_id
        start_line = params["start_line"]
        end_line = params["end_line"]
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
            raise LineOutOfBoundsError(
                message="Cannot delete lines from empty document",
                line=start_line,
                total_lines=0,
            )

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
        lines_deleted = end_line - start_line + 1

        # Capture deleted content for reference
        deleted_lines = lines[start_line:end_line + 1]
        deleted_content = "\n".join(deleted_lines)

        # Perform deletion
        result_lines = lines[:start_line] + lines[end_line + 1:]
        new_text = "\n".join(result_lines) if result_lines else ""

        # Apply edit via document provider
        if hasattr(context.document_provider, 'set_document_content'):
            context.document_provider.set_document_content(tab_id, new_text)
        elif self.document_editor:
            self.document_editor.set_document_text(tab_id, new_text)

        result: dict[str, Any] = {
            "lines_deleted": lines_deleted,
            "deleted_content": deleted_content,
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
        """Preview the deletion without applying."""
        tab_id = token.tab_id
        start_line = params["start_line"]
        end_line = params["end_line"]
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
        lines_deleted = end_line - start_line + 1 if total_lines > 0 else 0

        # Build preview
        preview_parts: list[str] = []

        # Context before
        if total_lines > 0:
            ctx_start = max(0, start_line - 2)
            for i in range(ctx_start, start_line):
                preview_parts.append(f"  {i}: {lines[i]}")

        # Deleted lines (show first few)
        deleted_preview = min(5, lines_deleted)
        if total_lines > 0:
            for i in range(deleted_preview):
                if start_line + i < total_lines:
                    preview_parts.append(f"- {start_line + i}: {lines[start_line + i]}")
        if lines_deleted > deleted_preview:
            preview_parts.append(f"  ... ({lines_deleted - deleted_preview} more lines to delete)")

        # Context after
        if total_lines > 0:
            ctx_end = min(total_lines, end_line + 3)
            for i in range(end_line + 1, ctx_end):
                preview_parts.append(f"  {i}: {lines[i]}")

        # Get deleted content preview
        deleted_content = ""
        if total_lines > 0:
            deleted_lines = lines[start_line:min(end_line + 1, total_lines)]
            deleted_content = "\n".join(deleted_lines[:10])
            if len(deleted_lines) > 10:
                deleted_content += f"\n... ({len(deleted_lines) - 10} more lines)"

        result: dict[str, Any] = {
            "preview": {
                "start_line": start_line,
                "end_line": end_line,
                "lines_to_delete": lines_deleted,
                "deleted_content_preview": deleted_content,
                "diff_preview": "\n".join(preview_parts),
            },
        }

        if drift_detected:
            result["drift_detected"] = True
            result["drift_from"] = drift_from

        return result
