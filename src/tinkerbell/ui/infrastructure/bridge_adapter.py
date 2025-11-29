"""Bridge adapter for workspace bridge operations with event emission.

This module wraps WorkspaceBridgeRouter with event emission through
the EventBus for decoupled communication.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Sequence, TYPE_CHECKING

from ..events import EditApplied, EditFailed, EventBus

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ...chat.message_model import EditDirective
    from ...editor.document_model import DocumentState
    from ...editor.workspace import DocumentTab, DocumentWorkspace
    from ...services.bridge_router import WorkspaceBridgeRouter

_LOGGER = logging.getLogger(__name__)


class BridgeAdapter:
    """Adapter wrapping WorkspaceBridgeRouter with event emission.

    This adapter wraps the WorkspaceBridgeRouter to emit events through
    the EventBus when edits are applied or fail. This decouples the
    bridge operations from UI components that need to react to edit events.

    The adapter provides the same interface as WorkspaceBridgeRouter for
    generating snapshots and managing edits, while adding event-based
    notifications.

    Attributes:
        _router: The underlying WorkspaceBridgeRouter.
        _workspace: The document workspace.
        _event_bus: The event bus for publishing events.
        _edit_counter: Counter for generating unique edit IDs.
    """

    __slots__ = (
        "_router",
        "_workspace",
        "_event_bus",
        "_edit_counter",
    )

    def __init__(
        self,
        workspace: DocumentWorkspace,
        event_bus: EventBus,
    ) -> None:
        """Initialize the bridge adapter.

        Args:
            workspace: The document workspace.
            event_bus: The event bus for publishing events.
        """
        try:
            from ...services.bridge_router import WorkspaceBridgeRouter
        except ImportError:  # pragma: no cover - dependency guard
            _LOGGER.warning("Bridge router not available")
            raise

        self._workspace = workspace
        self._event_bus = event_bus
        self._router = WorkspaceBridgeRouter(workspace)
        self._edit_counter = 0

        # Register listeners for edit events
        self._router.add_edit_listener(self._on_edit_applied)
        self._router.add_failure_listener(self._on_edit_failed)

    # ------------------------------------------------------------------
    # Public API - Snapshot generation
    # ------------------------------------------------------------------

    def generate_snapshot(
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_open_documents: bool = False,
        window: Mapping[str, Any] | str | None = None,
        chunk_profile: str | None = None,
        max_tokens: int | None = None,
        include_text: bool = True,
    ) -> dict[str, Any]:
        """Generate a document snapshot.

        Args:
            delta_only: Whether to include only changes since last snapshot.
            tab_id: Optional tab ID to get snapshot for (defaults to active).
            include_open_documents: Whether to include list of open documents.
            window: Optional window specification for partial snapshots.
            chunk_profile: Optional chunk profile name.
            max_tokens: Optional maximum tokens for the snapshot.
            include_text: Whether to include document text.

        Returns:
            A dictionary containing the document snapshot.
        """
        return self._router.generate_snapshot(
            delta_only=delta_only,
            tab_id=tab_id,
            include_open_documents=include_open_documents,
            window=window,
            chunk_profile=chunk_profile,
            max_tokens=max_tokens,
            include_text=include_text,
        )

    def generate_snapshots(
        self,
        tab_ids: Sequence[str],
        *,
        delta_only: bool = False,
        include_open_documents: bool = False,
        window: Mapping[str, Any] | str | None = None,
        chunk_profile: str | None = None,
        max_tokens: int | None = None,
        include_text: bool = True,
    ) -> list[dict[str, Any]]:
        """Generate snapshots for multiple tabs.

        Args:
            tab_ids: List of tab IDs to generate snapshots for.
            delta_only: Whether to include only changes since last snapshot.
            include_open_documents: Whether to include list of open documents.
            window: Optional window specification for partial snapshots.
            chunk_profile: Optional chunk profile name.
            max_tokens: Optional maximum tokens for the snapshot.
            include_text: Whether to include document text.

        Returns:
            A list of snapshot dictionaries.
        """
        return self._router.generate_snapshots(
            tab_ids,
            delta_only=delta_only,
            include_open_documents=include_open_documents,
            window=window,
            chunk_profile=chunk_profile,
            max_tokens=max_tokens,
            include_text=include_text,
        )

    # ------------------------------------------------------------------
    # Public API - Edit operations
    # ------------------------------------------------------------------

    def queue_edit(self, directive: EditDirective, *, tab_id: str | None = None) -> None:
        """Queue an edit directive for application.

        Args:
            directive: The edit directive to queue.
            tab_id: Optional tab ID to apply edit to (defaults to active).
        """
        self._router.queue_edit(directive, tab_id=tab_id)

    # ------------------------------------------------------------------
    # Public API - Tab management
    # ------------------------------------------------------------------

    def track_tab(self, tab: DocumentTab) -> None:
        """Track a new tab for bridge operations.

        Args:
            tab: The tab to track.
        """
        self._router.track_tab(tab)

    def list_tabs(self) -> list[dict[str, object]]:
        """List all open tabs.

        Returns:
            A list of tab information dictionaries.
        """
        return self._router.list_tabs()

    def active_tab_id(self) -> str | None:
        """Get the active tab ID.

        Returns:
            The active tab ID, or None if no tabs are open.
        """
        return self._router.active_tab_id()

    def get_active_tab_id(self) -> str | None:
        """Alias for active_tab_id() for compatibility.

        Returns:
            The active tab ID, or None if no tabs are open.
        """
        return self._router.get_active_tab_id()

    def get_tab_content(self, tab_id: str) -> str | None:
        """Get the content of a specific tab.

        Args:
            tab_id: The tab identifier.

        Returns:
            Document text content, or None if tab not found.
        """
        return self._router.get_tab_content(tab_id)

    def get_document_content(self, tab_id: str) -> str | None:
        """Get document content for a tab.

        Args:
            tab_id: The tab identifier.

        Returns:
            Document text content, or None if tab not found.
        """
        return self._router.get_document_content(tab_id)

    def set_document_content(self, tab_id: str, content: str) -> None:
        """Set document content for a tab.

        Args:
            tab_id: The tab identifier.
            content: The new document content.

        Raises:
            KeyError: If the tab is not found.
        """
        self._router.set_document_content(tab_id, content)

    def get_version_token(self, tab_id: str) -> str | None:
        """Get current version token for a tab.

        Args:
            tab_id: The tab identifier.

        Returns:
            Version token string, or None if not available.
        """
        return self._router.get_version_token(tab_id)

    # ------------------------------------------------------------------
    # Public API - Listener management
    # ------------------------------------------------------------------

    def add_edit_listener(self, listener: Callable[..., None]) -> None:
        """Add a listener for edit events.

        Args:
            listener: The listener callback.
        """
        self._router.add_edit_listener(listener)

    def add_failure_listener(self, listener: Callable[..., None]) -> None:
        """Add a listener for edit failure events.

        Args:
            listener: The listener callback.
        """
        self._router.add_failure_listener(listener)

    def set_main_thread_executor(self, executor: Any) -> None:
        """Set the main thread executor for edit operations.

        Args:
            executor: The executor for main thread callbacks.
        """
        self._router.set_main_thread_executor(executor)

    # ------------------------------------------------------------------
    # Public API - Metrics and state
    # ------------------------------------------------------------------

    @property
    def last_diff_summary(self) -> Any:
        """Get the last diff summary."""
        return self._router.last_diff_summary

    def get_last_diff_summary(self, tab_id: str | None = None) -> Any:
        """Get the last diff summary for a tab.

        Args:
            tab_id: Optional tab ID (defaults to active).

        Returns:
            The last diff summary.
        """
        return self._router.get_last_diff_summary(tab_id)

    @property
    def last_snapshot_version(self) -> Any:
        """Get the last snapshot version."""
        return self._router.last_snapshot_version

    def get_last_snapshot_version(self, tab_id: str | None = None) -> Any:
        """Get the last snapshot version for a tab.

        Args:
            tab_id: Optional tab ID (defaults to active).

        Returns:
            The last snapshot version.
        """
        return self._router.get_last_snapshot_version(tab_id)

    @property
    def last_edit_context(self) -> Any:
        """Get the last edit context."""
        return self._router.last_edit_context

    def get_last_edit_context(self, tab_id: str | None = None) -> Any:
        """Get the last edit context for a tab.

        Args:
            tab_id: Optional tab ID (defaults to active).

        Returns:
            The last edit context.
        """
        return self._router.get_last_edit_context(tab_id)

    @property
    def patch_metrics(self) -> Any:
        """Get patch metrics."""
        return self._router.patch_metrics

    def get_patch_metrics(self, tab_id: str | None = None) -> Any:
        """Get patch metrics for a tab.

        Args:
            tab_id: Optional tab ID (defaults to active).

        Returns:
            The patch metrics.
        """
        return self._router.get_patch_metrics(tab_id)

    def get_last_failure_metadata(self, tab_id: str | None = None) -> dict[str, Any] | None:
        """Get the last failure metadata for a tab.

        Args:
            tab_id: Optional tab ID (defaults to active).

        Returns:
            The failure metadata, or None.
        """
        return self._router.get_last_failure_metadata(tab_id)

    @property
    def router(self) -> WorkspaceBridgeRouter:
        """Get the underlying router for advanced operations."""
        return self._router

    # ------------------------------------------------------------------
    # Internal event handlers
    # ------------------------------------------------------------------

    def _on_edit_applied(
        self,
        directive: EditDirective,
        document: DocumentState,
        diff: str,
    ) -> None:
        """Handle edit applied events and emit to event bus.

        Args:
            directive: The edit directive that was applied.
            document: The document state after the edit.
            diff: The diff representation of the edit.
        """
        self._edit_counter += 1
        edit_id = f"edit-{self._edit_counter}"

        # Extract tab_id from directive or use active
        tab_id = getattr(directive, "tab_id", None) or self._router.active_tab_id() or ""
        document_id = document.document_id

        # Extract action type from directive
        action = getattr(directive, "action", None)
        if action is not None:
            action_str = action.value if hasattr(action, "value") else str(action)
        else:
            action_str = "unknown"

        # Extract range from directive
        start = getattr(directive, "start", 0) or 0
        end = getattr(directive, "end", 0) or 0
        edit_range = (start, end)

        # Emit event
        event = EditApplied(
            tab_id=tab_id,
            document_id=document_id,
            edit_id=edit_id,
            action=action_str,
            range=edit_range,
            diff=diff[:1000] if diff else "",  # Truncate diff for event
        )
        self._event_bus.publish(event)

    def _on_edit_failed(
        self,
        directive: EditDirective,
        message: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Handle edit failure events and emit to event bus.

        Args:
            directive: The edit directive that failed.
            message: The failure message.
            metadata: Optional failure metadata.
        """
        # Extract tab_id from metadata or directive
        tab_id = ""
        if metadata:
            tab_id = str(metadata.get("tab_id", ""))
        if not tab_id:
            tab_id = getattr(directive, "tab_id", None) or self._router.active_tab_id() or ""

        # Extract document_id from metadata
        document_id = ""
        if metadata:
            document_id = str(metadata.get("document_id", ""))

        # Extract action type from directive
        action = getattr(directive, "action", None)
        if action is not None:
            action_str = action.value if hasattr(action, "value") else str(action)
        else:
            action_str = "unknown"

        # Emit event
        event = EditFailed(
            tab_id=tab_id,
            document_id=document_id,
            action=action_str,
            reason=message,
        )
        self._event_bus.publish(event)

    # ------------------------------------------------------------------
    # Attribute forwarding
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the underlying router."""
        return getattr(self._router, name)


__all__ = ["BridgeAdapter"]
