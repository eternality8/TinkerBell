"""Write Document Tool for AI operations.

Provides functionality to replace the entire content of a document.
Use this for complete rewrites rather than incremental edits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

from .base import WriteTool, ToolContext
from .errors import (
    ContentRequiredError,
    TabNotFoundError,
)
from .version import VersionManager, VersionToken
from .insert_lines import split_lines, DocumentEditor

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Write Document Tool
# -----------------------------------------------------------------------------


@dataclass
class WriteDocumentTool(WriteTool):
    """Tool for replacing the entire content of a document.

    Completely replaces all document content with new content.
    Use this for full rewrites; prefer replace_lines for partial edits.

    Parameters:
        tab_id: Target tab ID (optional, defaults to active tab).
        version: Version token from a previous read (required).
        content: The complete new content for the document (required).
        dry_run: If True, preview changes without applying (default False).

    Returns:
        lines_affected: Information about the change.
            - previous: Line count before replacement.
            - current: Line count after replacement.
        size_affected: Information about size change.
            - previous: Character count before replacement.
            - current: Character count after replacement.
        version: New version token (or unchanged if dry_run).
    """

    name: ClassVar[str] = "write_document"
    summarizable: ClassVar[bool] = False

    version_manager: VersionManager
    document_editor: DocumentEditor | None = None

    def validate(self, params: dict[str, Any]) -> None:
        """Validate parameters."""
        super().validate(params)

        if "content" not in params:
            raise ContentRequiredError(
                message="Parameter 'content' is required for write_document",
                field_name="content",
            )

    def write(
        self,
        context: ToolContext,
        params: dict[str, Any],
        token: VersionToken,
    ) -> dict[str, Any]:
        """Perform the full document replacement."""
        tab_id = token.tab_id
        new_content = params["content"]

        # Get current document content
        doc_content = context.document_provider.get_document_content(tab_id)
        if doc_content is None:
            raise TabNotFoundError(
                message=f"Document not found: {tab_id}",
                tab_id=tab_id,
            )

        # Calculate metrics
        old_lines = split_lines(doc_content)
        new_lines = split_lines(new_content)

        previous_line_count = len(old_lines) if old_lines else 0
        current_line_count = len(new_lines) if new_lines else 0

        previous_size = len(doc_content)
        current_size = len(new_content)

        # Apply the edit via document provider
        if hasattr(context.document_provider, 'set_document_content'):
            context.document_provider.set_document_content(tab_id, new_content)
        elif self.document_editor:
            self.document_editor.set_document_text(tab_id, new_content)

        return {
            "lines_affected": {
                "previous": previous_line_count,
                "current": current_line_count,
            },
            "size_affected": {
                "previous": previous_size,
                "current": current_size,
            },
            "_new_text": new_content,
        }

    def preview(
        self,
        context: ToolContext,
        params: dict[str, Any],
        token: VersionToken,
    ) -> dict[str, Any]:
        """Preview the full document replacement without applying."""
        tab_id = token.tab_id
        new_content = params["content"]

        # Get current document content
        doc_content = context.document_provider.get_document_content(tab_id)
        if doc_content is None:
            raise TabNotFoundError(
                message=f"Document not found: {tab_id}",
                tab_id=tab_id,
            )

        # Calculate metrics
        old_lines = split_lines(doc_content)
        new_lines = split_lines(new_content)

        previous_line_count = len(old_lines) if old_lines else 0
        current_line_count = len(new_lines) if new_lines else 0

        previous_size = len(doc_content)
        current_size = len(new_content)

        # Build diff preview showing first/last few lines of change
        preview_parts: list[str] = []

        # Show summary
        preview_parts.append(f"=== Document Replacement Summary ===")
        preview_parts.append(f"Lines: {previous_line_count} -> {current_line_count}")
        preview_parts.append(f"Size: {previous_size} -> {current_size} chars")
        preview_parts.append("")

        # Show first few lines of old content
        preview_parts.append("--- Previous content (first 5 lines):")
        for i, line in enumerate(old_lines[:5]):
            preview_parts.append(f"- {i}: {line}")
        if previous_line_count > 5:
            preview_parts.append(f"  ... ({previous_line_count - 5} more lines)")
        preview_parts.append("")

        # Show first few lines of new content
        preview_parts.append("+++ New content (first 5 lines):")
        for i, line in enumerate(new_lines[:5]):
            preview_parts.append(f"+ {i}: {line}")
        if current_line_count > 5:
            preview_parts.append(f"  ... ({current_line_count - 5} more lines)")

        return {
            "preview": {
                "lines_previous": previous_line_count,
                "lines_current": current_line_count,
                "size_previous": previous_size,
                "size_current": current_size,
                "diff_preview": "\n".join(preview_parts),
            },
        }
