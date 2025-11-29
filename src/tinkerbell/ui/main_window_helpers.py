"""Helper classes for MainWindow that implement protocols and adapters.

These are small adapter classes that bridge between MainWindow and other
subsystems like the editor lock manager and tool dispatcher. Also includes
pure utility functions extracted from MainWindow.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence

from ..ai.orchestration.editor_lock import LockableTab, TabProvider
from ..ai.orchestration.tool_dispatcher import DispatchResult
from ..ai.tools.tool_registry import get_tool_registry
from ..chat.commands import DIRECTIVE_SCHEMA

if TYPE_CHECKING:
    from ..chat.message_model import ChatMessage
    from ..editor.workspace import DocumentTab


__all__ = [
    # Protocol implementations
    "WriteToolDispatchListener",
    "EditorTabWrapper",
    "WorkspaceTabProvider",
    # Pure utility functions
    "condense_whitespace",
    "line_column_from_offset",
    "coerce_stream_text",
    "infer_language",
    "directive_parameters_schema",
    "serialize_chat_history",
    "history_signature",
]


# ---------------------------------------------------------------------------
# Pure utility functions (extracted from MainWindow static methods)
# ---------------------------------------------------------------------------

def condense_whitespace(text: str) -> str:
    """Collapse all whitespace in text to single spaces."""
    return " ".join(text.split())


def line_column_from_offset(text: str, caret: int) -> tuple[int, int]:
    """Convert a character offset to 1-based (line, column) position."""
    if not text:
        return (1, 1)
    length = len(text)
    caret = max(0, min(int(caret), length))
    line = text.count("\n", 0, caret) + 1
    last_newline = text.rfind("\n", 0, caret)
    column = caret + 1 if last_newline == -1 else caret - last_newline
    if column <= 0:
        column = 1
    return (line, column)


def coerce_stream_text(payload: Any) -> str:
    """Extract text from various payload formats (streaming AI responses).
    
    Handles strings, mappings with text/content/value keys, sequences,
    and objects with text/content attributes.
    """
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, Mapping):
        for key in ("text", "content", "value"):
            if key in payload:
                text = coerce_stream_text(payload[key])
                if text:
                    return text
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        parts = [coerce_stream_text(item) for item in payload]
        return "".join(part for part in parts if part)
    text_attr = getattr(payload, "text", None)
    if text_attr:
        return coerce_stream_text(text_attr)
    content_attr = getattr(payload, "content", None)
    if content_attr is not None and content_attr is not payload:
        text = coerce_stream_text(content_attr)
        if text:
            return text
    return str(payload)


def infer_language(path: Path) -> str:
    """Infer a simple language identifier from the file suffix."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".json":
        return "json"
    if suffix in {".txt", ""}:
        return "text"
    return "plain"


def directive_parameters_schema() -> Dict[str, Any]:
    """Return a copy of the directive schema used by the document edit tool."""
    schema = deepcopy(DIRECTIVE_SCHEMA)
    schema.setdefault(
        "description",
        "Structured edit directive containing action, content, optional rationale, and target range. Prefer action='patch' with a unified diff and document_version when modifying existing text.",
    )
    return schema


def serialize_chat_history(
    history: "Sequence[ChatMessage]",
    limit: int = 10,
    *,
    exclude_latest: bool = False,
) -> list[dict[str, str]]:
    """Serialize chat history to a list of role/content dictionaries.
    
    Args:
        history: Sequence of ChatMessage objects
        limit: Maximum number of recent messages to include
        exclude_latest: If True, exclude the most recent message
        
    Returns:
        List of dicts with 'role' and 'content' keys
    """
    messages = list(history)
    if exclude_latest and messages:
        messages = messages[:-1]
    windowed = messages[-limit:] if limit else messages
    serialized: list[dict[str, str]] = []
    for message in windowed:
        text = (message.content or "").strip()
        if not text:
            continue
        serialized.append({"role": message.role, "content": text})
    return serialized


def history_signature(history: "Sequence[ChatMessage]") -> str | None:
    """Compute a SHA-256 digest of chat history for caching.
    
    Args:
        history: Sequence of ChatMessage objects
        
    Returns:
        Hex digest string, or None if history is empty
    """
    payload = serialize_chat_history(history)
    if not payload:
        return None
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest


# ---------------------------------------------------------------------------
# Protocol implementation classes
# ---------------------------------------------------------------------------

class WriteToolDispatchListener:
    """Listens for tool dispatch events to track document write operations.
    
    When a tool with writes_document=True completes successfully, this listener
    records the edit in the pending turn review so that changes are not dropped
    as "no-edits".
    """
    
    __slots__ = ("_window", "_registry")
    
    def __init__(self, main_window: Any) -> None:
        self._window = main_window
        self._registry = get_tool_registry()
    
    def on_tool_start(self, tool_name: str, arguments: Mapping[str, Any]) -> None:
        """Called when a tool starts execution."""
        # We only care about completion, not start
        pass
    
    def on_tool_complete(self, result: DispatchResult) -> None:
        """Called when a tool completes - increment edit count if it was a write tool."""
        if not result.success:
            return
        if not self._registry.is_write_tool(result.tool_name):
            return
        # Track this as an edit in the pending turn review
        self._window._record_write_tool_edit(result)
    
    def on_tool_error(self, tool_name: str, error: Any) -> None:
        """Called when a tool fails."""
        # Errors don't count as edits
        pass


class EditorTabWrapper:
    """Wraps a DocumentTab to implement LockableTab protocol."""
    
    __slots__ = ("_tab",)
    
    def __init__(self, tab: "DocumentTab") -> None:
        self._tab = tab
    
    @property
    def id(self) -> str:
        return self._tab.id
    
    def set_readonly(self, readonly: bool) -> None:
        """Set the tab's read-only state via its editor widget."""
        editor = self._tab.editor
        set_ro = getattr(editor, "set_readonly", None)
        if callable(set_ro):
            set_ro(readonly)
    
    def is_readonly(self) -> bool:
        """Check if the tab is read-only."""
        editor = self._tab.editor
        is_ro = getattr(editor, "is_readonly", None)
        if callable(is_ro):
            return is_ro()
        return False


class WorkspaceTabProvider:
    """Implements TabProvider protocol for the workspace."""
    
    __slots__ = ("_workspace",)
    
    def __init__(self, workspace: Any) -> None:
        self._workspace = workspace
    
    def get_all_tabs(self) -> Sequence[LockableTab]:
        """Get all open tabs as lockable wrappers."""
        tabs = list(self._workspace.iter_tabs())
        return [EditorTabWrapper(tab) for tab in tabs]
    
    def get_active_tab(self) -> LockableTab | None:
        """Get the currently active tab."""
        active = self._workspace.active_tab
        if active is None:
            return None
        return EditorTabWrapper(active)
