"""Tool for reading document content with automatic pagination.

WS2.2: Implements line-range reading with automatic pagination,
token estimation, and proper handling of empty/unsupported files.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..utils.tokens import CHARS_PER_TOKEN, estimate_tokens
from .base import ReadOnlyTool, ToolContext
from .errors import (
    InvalidLineRangeError,
    InvalidTabIdError,
    LineOutOfBoundsError,
    TabNotFoundError,
    UnsupportedFileTypeError,
)
from .list_tabs import detect_file_type, is_supported_file_type
from .version import VersionManager, compute_content_hash, get_version_manager


# Default token window for automatic pagination (~6000 tokens = ~24000 chars)
DEFAULT_TOKEN_WINDOW = 6000
DEFAULT_CHAR_WINDOW = int(DEFAULT_TOKEN_WINDOW * CHARS_PER_TOKEN)

# Maximum lines to return in a single response
MAX_LINES_PER_RESPONSE = 2000


def split_lines(text: str) -> list[str]:
    """Split text into lines, preserving line structure.

    Args:
        text: The text to split.

    Returns:
        List of lines (without trailing newlines).
    """
    if not text:
        return []
    return text.split("\n")


@dataclass
class ReadDocumentTool(ReadOnlyTool):
    """Read document content with automatic pagination and line metadata.

    This tool reads document content with support for:
    - Line range selection (0-indexed, inclusive)
    - Automatic pagination for large documents
    - Token estimation for context budgeting
    - Version tokens for optimistic concurrency

    Parameters:
        tab_id: Target tab ID (optional, defaults to active tab)
        start_line: First line to read, 0-indexed (optional, defaults to 0)
        end_line: Last line to read, inclusive (optional, defaults to auto-paginate)
        max_tokens: Maximum tokens to return (optional, defaults to 6000)

    Response includes:
        - content: The requested text content
        - lines: Line range metadata (start, end, total)
        - tokens: Token estimates (returned, total_estimate)
        - has_more: Whether more content is available
        - continuation_hint: Suggested next start_line if has_more
        - version: Version token for subsequent write operations
    """

    name: ClassVar[str] = "read_document"
    description: ClassVar[str] = "Read document content with automatic pagination"
    summarizable: ClassVar[bool] = True

    version_manager: VersionManager = field(default_factory=get_version_manager)

    def read(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the read_document tool.

        Args:
            context: Tool execution context with document provider.
            params: Tool parameters.

        Returns:
            Document content with metadata.

        Raises:
            InvalidTabIdError: If tab_id is invalid.
            TabNotFoundError: If the specified tab doesn't exist.
            UnsupportedFileTypeError: If the file type is not supported.
            InvalidLineRangeError: If the line range is invalid.
            LineOutOfBoundsError: If lines are out of document bounds.
        """
        # Resolve tab ID
        tab_id = context.resolve_tab_id(params.get("tab_id"))
        if tab_id is None:
            raise InvalidTabIdError(
                message="No tab specified and no active tab available",
                tab_id=params.get("tab_id"),
            )

        # Get document content
        content = context.document_provider.get_document_content(tab_id)
        if content is None:
            raise TabNotFoundError(
                message=f"Tab '{tab_id}' not found",
                tab_id=tab_id,
            )

        # Get file metadata for type checking
        metadata = context.document_provider.get_document_metadata(tab_id)
        path = metadata.get("path") if metadata else None
        language = metadata.get("language") if metadata else None
        file_type = detect_file_type(path, language)

        if not is_supported_file_type(file_type):
            raise UnsupportedFileTypeError(
                message=f"Cannot read {file_type} files",
                file_type=file_type,
                file_path=path,
            )

        # Parse parameters
        start_line = params.get("start_line", 0)
        end_line = params.get("end_line")  # None means auto-paginate
        max_tokens = params.get("max_tokens", DEFAULT_TOKEN_WINDOW)

        # Validate start_line
        if not isinstance(start_line, int) or start_line < 0:
            raise InvalidLineRangeError(
                message="start_line must be a non-negative integer",
                start_line=start_line,
                end_line=end_line,
            )

        # Split content into lines
        lines = split_lines(content)
        total_lines = len(lines)
        content_hash = compute_content_hash(content)

        # Handle empty document
        if total_lines == 0:
            # Register tab with version manager if not already
            if not self.version_manager.get_current_token(tab_id):
                doc_id = uuid.uuid4().hex
                self.version_manager.register_tab(tab_id, doc_id, content_hash)

            token = self.version_manager.get_current_token(tab_id)
            return {
                "content": "",
                "lines": {
                    "start": 0,
                    "end": 0,
                    "total": 0,
                    "returned": 0,
                },
                "tokens": {
                    "returned": 0,
                    "total_estimate": 0,
                },
                "has_more": False,
                "continuation_hint": None,
                "version": token.to_string() if token else None,
                "tab_id": tab_id,
                "file_type": file_type,
            }

        # Validate start_line bounds
        if start_line >= total_lines:
            raise LineOutOfBoundsError(
                message=f"start_line {start_line} is beyond document end (total: {total_lines} lines)",
                line=start_line,
                total_lines=total_lines,
            )

        # Validate end_line if provided
        if end_line is not None:
            if not isinstance(end_line, int) or end_line < 0:
                raise InvalidLineRangeError(
                    message="end_line must be a non-negative integer",
                    start_line=start_line,
                    end_line=end_line,
                )
            if end_line < start_line:
                raise InvalidLineRangeError(
                    message="end_line cannot be less than start_line",
                    start_line=start_line,
                    end_line=end_line,
                    total_lines=total_lines,
                )
            # Clamp end_line to document bounds
            end_line = min(end_line, total_lines - 1)
        else:
            # Auto-paginate: find end_line based on token budget
            end_line = self._find_pagination_end(
                lines,
                start_line,
                max_tokens=max_tokens,
                max_lines=MAX_LINES_PER_RESPONSE,
            )

        # Extract requested lines
        selected_lines = lines[start_line : end_line + 1]
        selected_content = "\n".join(selected_lines)

        # Calculate token estimates
        returned_tokens = estimate_tokens(selected_content)
        total_tokens = estimate_tokens(content)

        # Determine if there's more content
        has_more = end_line < total_lines - 1

        # Build continuation hint
        continuation_hint = None
        if has_more:
            continuation_hint = {
                "start_line": end_line + 1,
                "remaining_lines": total_lines - end_line - 1,
            }

        # Register/update version for this tab
        if not self.version_manager.get_current_token(tab_id):
            doc_id = uuid.uuid4().hex
            self.version_manager.register_tab(tab_id, doc_id, content_hash)
        token = self.version_manager.get_current_token(tab_id)

        return {
            "content": selected_content,
            "lines": {
                "start": start_line,
                "end": end_line,
                "total": total_lines,
                "returned": len(selected_lines),
            },
            "tokens": {
                "returned": returned_tokens,
                "total_estimate": total_tokens,
            },
            "has_more": has_more,
            "continuation_hint": continuation_hint,
            "version": token.to_string() if token else None,
            "tab_id": tab_id,
            "file_type": file_type,
        }

    def _find_pagination_end(
        self,
        lines: list[str],
        start_line: int,
        *,
        max_tokens: int,
        max_lines: int,
    ) -> int:
        """Find the end line for automatic pagination.

        Args:
            lines: All document lines.
            start_line: Starting line index.
            max_tokens: Maximum tokens to include.
            max_lines: Maximum lines to include.

        Returns:
            The end line index (inclusive).
        """
        total_lines = len(lines)
        max_end = min(start_line + max_lines - 1, total_lines - 1)

        # Build up content until we hit the token limit
        accumulated_chars = 0
        max_chars = int(max_tokens * CHARS_PER_TOKEN)

        for i in range(start_line, max_end + 1):
            line_chars = len(lines[i]) + 1  # +1 for newline
            if accumulated_chars + line_chars > max_chars and i > start_line:
                # Would exceed limit, stop at previous line
                return i - 1
            accumulated_chars += line_chars

        return max_end


# Factory function for creating the tool
def create_read_document_tool(version_manager: VersionManager | None = None) -> ReadDocumentTool:
    """Create a ReadDocumentTool instance.

    Args:
        version_manager: Optional version manager (uses global singleton if not provided).

    Returns:
        Configured ReadDocumentTool instance.
    """
    if version_manager is None:
        version_manager = get_version_manager()
    return ReadDocumentTool(version_manager=version_manager)
