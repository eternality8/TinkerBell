"""Tool for listing all open tabs with enhanced metadata.

WS2.1: Updated to include version, size_chars, line_count, is_active, and file_type.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Protocol, Sequence

from .base import BaseTool, ToolContext, ToolResult
from .version import VersionManager, get_version_manager


class TabListingProvider(Protocol):
    """Protocol describing the workspace facade required to enumerate tabs."""

    def list_tabs(self) -> Sequence[Mapping[str, Any]]:
        """Return a list of tab metadata dictionaries."""
        ...

    def active_tab_id(self) -> str | None:
        """Return the currently active tab ID, or None if no tab is active."""
        ...

    def get_tab_content(self, tab_id: str) -> str | None:
        """Return the full text content of a tab, or None if not found."""
        ...


# File type detection based on extension
_FILE_TYPE_MAP: dict[str, str] = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "plain_text",
    ".text": "plain_text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".htm": "html",
    ".xml": "xml",
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".css": "css",
    ".csv": "csv",
    ".rst": "restructuredtext",
    ".tex": "latex",
    ".log": "log",
}

# Binary file extensions that should not be processed
_BINARY_EXTENSIONS: set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".exe", ".dll", ".so", ".dylib",
    ".bin", ".dat",
}


def detect_file_type(path: str | None, language: str | None = None) -> str:
    """Detect file type from path extension or language hint.

    Args:
        path: File path (may be None for untitled documents).
        language: Language hint from document metadata.

    Returns:
        A string identifying the file type: 'markdown', 'plain_text', 'json',
        'yaml', 'binary', 'unknown', or the language hint if recognized.
    """
    # Check language hint first
    if language and language.lower() in ("markdown", "json", "yaml", "plain_text"):
        return language.lower()

    if path is None:
        # Untitled documents default to plain_text (no assumed structure)
        return "plain_text"

    ext = os.path.splitext(path)[1].lower()

    # Check for binary files
    if ext in _BINARY_EXTENSIONS:
        return "binary"

    # Check known extensions
    if ext in _FILE_TYPE_MAP:
        return _FILE_TYPE_MAP[ext]

    # Fall back to language hint or unknown
    if language:
        return language.lower()

    return "unknown"


def is_supported_file_type(file_type: str) -> bool:
    """Check if a file type is supported for AI operations.

    Args:
        file_type: The file type string from detect_file_type().

    Returns:
        True if the file type is supported for reading/editing.
    """
    return file_type not in ("binary", "unknown")


@dataclass
class ListTabsTool(BaseTool):
    """Return enhanced metadata for every open editor tab.

    This tool provides a comprehensive listing of all open tabs including:
    - Tab identification (tab_id, title, path)
    - Content metrics (size_chars, line_count)
    - Version information for optimistic concurrency
    - File type detection for AI tool compatibility
    - Active status indicator

    The response enables the AI to:
    1. Choose which document to work with
    2. Understand document sizes for planning
    3. Detect file types for appropriate tool selection
    """

    name: ClassVar[str] = "list_tabs"
    description: ClassVar[str] = "List all open editor tabs with metadata"
    summarizable: ClassVar[bool] = True

    provider: TabListingProvider
    version_manager: VersionManager = field(default_factory=get_version_manager)

    def execute(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the list_tabs tool.

        Args:
            context: Tool execution context (unused for this tool).
            params: Tool parameters (none required).

        Returns:
            Dictionary containing:
            - tabs: List of tab metadata objects
            - active_tab_id: ID of the currently active tab
            - total: Total number of open tabs
        """
        raw_tabs = self.provider.list_tabs()
        active_tab_id = self.provider.active_tab_id()

        tabs = []
        for idx, entry in enumerate(raw_tabs, start=1):
            tab_data = self._build_tab_entry(entry, index=idx, active_tab_id=active_tab_id)
            tabs.append(tab_data)

        return {
            "tabs": tabs,
            "active_tab_id": active_tab_id,
            "total": len(tabs),
        }

    def _build_tab_entry(
        self,
        entry: Mapping[str, Any],
        *,
        index: int,
        active_tab_id: str | None,
    ) -> dict[str, Any]:
        """Build an enhanced tab entry from raw tab data.

        Args:
            entry: Raw tab metadata from the provider.
            index: 1-based tab index for display.
            active_tab_id: ID of the active tab for comparison.

        Returns:
            Enhanced tab metadata dictionary.
        """
        tab_id = str(entry.get("tab_id") or entry.get("id") or "").strip()
        title = str(entry.get("title") or "Untitled").strip()
        path = entry.get("path")
        if path is not None:
            path = str(path)

        language = entry.get("language")
        file_type = detect_file_type(path, language)

        # Get content metrics
        size_chars = 0
        line_count = 0
        try:
            content = self.provider.get_tab_content(tab_id)
            if content is not None:
                size_chars = len(content)
                line_count = content.count("\n") + (1 if content else 0)
                # Handle edge case: empty string has 0 lines
                if not content:
                    line_count = 0
        except Exception:
            # If we can't get content, use defaults
            pass

        # Get version token if available
        version = None
        try:
            token = self.version_manager.get_current_token(tab_id)
            if token:
                version = token.to_string()
        except Exception:
            pass

        return {
            # Identification
            "tab_id": tab_id,
            "title": title,
            "path": path,
            "tab_number": index,
            "label": f"Tab {index}: {title}",
            # Status
            "is_active": tab_id == active_tab_id,
            "dirty": bool(entry.get("dirty", False)),
            # Content metrics
            "size_chars": size_chars,
            "line_count": line_count,
            # Type information
            "file_type": file_type,
            "supported": is_supported_file_type(file_type),
            "language": language,
            # Version for optimistic concurrency
            "version": version,
            # Legacy fields
            "untitled_index": entry.get("untitled_index"),
        }


# Backward compatibility: Allow direct instantiation without base class features
def create_list_tabs_tool(provider: TabListingProvider) -> ListTabsTool:
    """Factory function to create a ListTabsTool instance.

    Args:
        provider: The tab listing provider implementation.

    Returns:
        A configured ListTabsTool instance.
    """
    return ListTabsTool(provider=provider)
