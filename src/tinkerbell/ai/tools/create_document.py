"""Create Document Tool for AI operations.

Provides functionality to create new document tabs with optional
initial content and file type hints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Protocol

from .base import BaseTool, ToolContext
from .errors import (
    InvalidParameterError,
    ToolError,
)
from .version import VersionManager

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Error Types
# -----------------------------------------------------------------------------


class TitleExistsError(ToolError):
    """Raised when a document with the given title already exists."""

    severity: ClassVar[str] = "error"

    def __init__(
        self,
        *,
        title: str,
        existing_tab_id: str | None = None,
    ) -> None:
        super().__init__(
            error_code="title_exists",
            message=f"A document with title '{title}' already exists",
            details={
                "title": title,
                "existing_tab_id": existing_tab_id,
            } if existing_tab_id else {"title": title},
            suggestion="Use a different title or open the existing document",
        )


class DocumentCreationError(ToolError):
    """Raised when document creation fails."""

    severity: ClassVar[str] = "error"

    def __init__(
        self,
        *,
        message: str,
        reason: str | None = None,
    ) -> None:
        super().__init__(
            error_code="document_creation_failed",
            message=message,
            details={"reason": reason} if reason else {},
        )


# -----------------------------------------------------------------------------
# Document Creator Protocol
# -----------------------------------------------------------------------------


class DocumentCreator(Protocol):
    """Protocol for creating new document tabs."""

    def create_document(
        self,
        title: str,
        content: str = "",
        file_type: str | None = None,
    ) -> str:
        """Create a new document tab.

        Args:
            title: The title/filename for the new document.
            content: Optional initial content.
            file_type: Optional file type hint (e.g., 'markdown', 'json').

        Returns:
            The tab_id of the newly created document.

        Raises:
            TitleExistsError: If a document with this title already exists.
            DocumentCreationError: If creation fails for other reasons.
        """
        ...

    def document_exists(self, title: str) -> tuple[bool, str | None]:
        """Check if a document with the given title exists.

        Returns:
            Tuple of (exists, tab_id or None).
        """
        ...


# -----------------------------------------------------------------------------
# File Type Detection
# -----------------------------------------------------------------------------


def infer_file_type(title: str, hint: str | None = None) -> str:
    """Infer file type from title/filename or hint.

    Args:
        title: Document title or filename.
        hint: Optional explicit file type hint.

    Returns:
        Inferred file type string.
    """
    if hint:
        return hint.lower()

    # Try to get extension from title
    path = Path(title)
    ext = path.suffix.lower()

    extension_map = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "plain_text",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".xml": "xml",
        ".html": "html",
        ".htm": "html",
        ".css": "css",
        ".js": "javascript",
        ".ts": "typescript",
        ".py": "python",
        ".rst": "restructuredtext",
    }

    return extension_map.get(ext, "plain_text")


def suggest_extension(file_type: str) -> str:
    """Suggest file extension for a file type.

    Args:
        file_type: The file type string.

    Returns:
        Suggested extension (with dot).
    """
    type_map = {
        "markdown": ".md",
        "plain_text": ".txt",
        "json": ".json",
        "yaml": ".yaml",
        "xml": ".xml",
        "html": ".html",
        "css": ".css",
        "javascript": ".js",
        "typescript": ".ts",
        "python": ".py",
        "restructuredtext": ".rst",
    }

    return type_map.get(file_type.lower(), ".txt")


# -----------------------------------------------------------------------------
# Create Document Tool
# -----------------------------------------------------------------------------


@dataclass
class CreateDocumentTool(BaseTool):
    """Tool for creating new document tabs.

    Creates a new document tab with the specified title and optional
    initial content. Returns the tab_id and initial version token.

    Parameters:
        title: The title/filename for the new document (required).
        content: Optional initial content for the document.
        file_type: Optional file type hint (e.g., 'markdown', 'json').
            If not provided, inferred from title extension.

    Returns:
        tab_id: ID of the newly created tab.
        version: Initial version token for the document.
        title: The document title.
        file_type: The detected/specified file type.
        lines: Number of lines in initial content.
    """

    name: ClassVar[str] = "create_document"
    summarizable: ClassVar[bool] = False

    version_manager: VersionManager
    document_creator: DocumentCreator | None = None

    def validate(self, params: dict[str, Any]) -> None:
        """Validate parameters."""
        title = params.get("title")
        if not title:
            raise InvalidParameterError(
                message="Parameter 'title' is required",
                parameter="title",
                expected="Non-empty string",
                value=title,
            )

        # Validate title doesn't have problematic characters
        invalid_chars = set('<>:"/\\|?*')
        found_invalid = [c for c in str(title) if c in invalid_chars]
        if found_invalid:
            raise InvalidParameterError(
                message=f"Title contains invalid characters: {found_invalid}",
                parameter="title",
                expected="Valid filename characters",
                value=title,
            )

    def execute(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        """Create a new document tab."""
        title = params["title"]
        content = params.get("content", "")
        file_type_hint = params.get("file_type")

        # Infer file type
        file_type = infer_file_type(title, file_type_hint)

        # Check if document exists (if creator available)
        if self.document_creator:
            exists, existing_id = self.document_creator.document_exists(title)
            if exists:
                raise TitleExistsError(title=title, existing_tab_id=existing_id)

            # Create the document
            try:
                tab_id = self.document_creator.create_document(
                    title=title,
                    content=content,
                    file_type=file_type,
                )
            except TitleExistsError:
                raise
            except Exception as exc:
                LOGGER.exception("Failed to create document: %s", title)
                raise DocumentCreationError(
                    message=f"Failed to create document '{title}'",
                    reason=str(exc),
                ) from exc
        else:
            # No document creator configured - this is a configuration error
            # that should be caught early, not silently worked around
            LOGGER.error(
                "CreateDocumentTool.document_creator is None - tool was not properly wired. "
                "Cannot create document '%s'. This indicates a configuration bug.",
                title,
            )
            raise DocumentCreationError(
                message=f"Cannot create document '{title}': document creator not configured",
                reason="The create_document tool was not properly initialized with a document_creator. "
                       "This is an internal configuration error.",
            )

        # Initialize version tracking
        from .version import compute_content_hash
        content_hash = compute_content_hash(content)

        # Register with version manager (creates new tab entry)
        # Use title as document_id for simplicity
        token = self.version_manager.register_tab(tab_id, title, content_hash)

        # Count lines
        lines = content.split("\n") if content else []
        line_count = len(lines) if lines and lines != [""] else 0

        return {
            "tab_id": tab_id,
            "version": token.to_string(),
            "title": title,
            "file_type": file_type,
            "lines": line_count,
            "size_chars": len(content),
        }
