"""Document store domain manager.

Wraps DocumentWorkspace with event emission to decouple document management
from UI concerns. This is the single source of truth for document/tab state.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator

from ..events import (
    ActiveTabChanged,
    DocumentClosed,
    DocumentCreated,
    EventBus,
)

if TYPE_CHECKING:  # pragma: no cover
    from ...editor.document_model import DocumentState
    from ...editor.workspace import DocumentTab, DocumentWorkspace

LOGGER = logging.getLogger(__name__)


class DocumentStore:
    """Domain manager for document and tab lifecycle.

    Wraps a DocumentWorkspace and emits events on state changes.
    All document/tab operations should go through this class to ensure
    proper event emission and state tracking.

    Events Emitted:
        - DocumentCreated: When a new tab is created
        - DocumentClosed: When a tab is closed
        - ActiveTabChanged: When the active tab changes
    """

    def __init__(
        self,
        workspace: DocumentWorkspace,
        event_bus: EventBus,
    ) -> None:
        """Initialize the document store.

        Args:
            workspace: The underlying DocumentWorkspace to wrap.
            event_bus: The event bus for publishing events.
        """
        self._workspace = workspace
        self._bus = event_bus

        # Register as listener for active tab changes from workspace
        self._workspace.add_active_listener(self._on_active_tab_changed)

    # ------------------------------------------------------------------
    # Tab Lifecycle
    # ------------------------------------------------------------------

    def create_tab(
        self,
        document: DocumentState | None = None,
        title: str | None = None,
        path: Path | str | None = None,
        *,
        make_active: bool = True,
        untitled_index: int | None = None,
        tab_id: str | None = None,
    ) -> DocumentTab:
        """Create a new document tab.

        Args:
            document: Optional initial document state.
            title: Optional tab title.
            path: Optional file path for the document.
            make_active: Whether to make this the active tab.
            untitled_index: Optional index for untitled documents (for restoration).
            tab_id: Optional tab ID (for restoration to preserve original ID).

        Returns:
            The newly created DocumentTab.

        Emits:
            DocumentCreated: After the tab is created.
            ActiveTabChanged: If make_active is True.
        """
        tab = self._workspace.create_tab(
            document=document,
            title=title,
            path=path,
            make_active=make_active,
            untitled_index=untitled_index,
            tab_id=tab_id,
        )

        doc = tab.document()
        LOGGER.debug(
            "DocumentStore.create_tab: tab_id=%s, document_id=%s, path=%s",
            tab.id,
            doc.document_id,
            path,
        )

        self._bus.publish(DocumentCreated(
            tab_id=tab.id,
            document_id=doc.document_id,
        ))

        return tab

    def close_tab(self, tab_id: str) -> DocumentTab | None:
        """Close a document tab.

        Args:
            tab_id: The ID of the tab to close.

        Returns:
            The closed DocumentTab, or None if not found.

        Emits:
            DocumentClosed: After the tab is closed.
            ActiveTabChanged: If the closed tab was active.
        """
        try:
            tab = self._workspace.get_tab(tab_id)
        except KeyError:
            LOGGER.warning("DocumentStore.close_tab: unknown tab_id=%s", tab_id)
            return None

        doc = tab.document()
        document_id = doc.document_id

        closed_tab = self._workspace.close_tab(tab_id)

        LOGGER.debug(
            "DocumentStore.close_tab: tab_id=%s, document_id=%s",
            tab_id,
            document_id,
        )

        self._bus.publish(DocumentClosed(
            tab_id=tab_id,
            document_id=document_id,
        ))

        return closed_tab

    # ------------------------------------------------------------------
    # Tab Access
    # ------------------------------------------------------------------

    def get_tab(self, tab_id: str) -> DocumentTab:
        """Get a tab by ID.

        Args:
            tab_id: The ID of the tab to retrieve.

        Returns:
            The DocumentTab.

        Raises:
            KeyError: If the tab is not found.
        """
        return self._workspace.get_tab(tab_id)

    @property
    def active_tab(self) -> DocumentTab | None:
        """Get the currently active tab, if any."""
        return self._workspace.active_tab

    @property
    def active_tab_id(self) -> str | None:
        """Get the ID of the currently active tab, if any."""
        return self._workspace.active_tab_id

    def set_active_tab(self, tab_id: str) -> DocumentTab:
        """Set the active tab.

        Args:
            tab_id: The ID of the tab to make active.

        Returns:
            The newly active DocumentTab.

        Raises:
            KeyError: If the tab is not found.

        Emits:
            ActiveTabChanged: After the active tab changes.
        """
        # The workspace will notify via the listener callback
        return self._workspace.set_active_tab(tab_id)

    def iter_tabs(self) -> Iterator[DocumentTab]:
        """Iterate over all open tabs in order."""
        return self._workspace.iter_tabs()

    def tab_count(self) -> int:
        """Return the number of open tabs."""
        return self._workspace.tab_count()

    def tab_ids(self) -> tuple[str, ...]:
        """Return tuple of all tab IDs in order."""
        return tuple(self._workspace.tab_ids())

    # ------------------------------------------------------------------
    # Document Lookup
    # ------------------------------------------------------------------

    def find_tab_by_path(self, path: Path | str) -> DocumentTab | None:
        """Find a tab by its document's file path.

        Args:
            path: The file path to search for.

        Returns:
            The matching DocumentTab, or None if not found.
        """
        return self._workspace.find_tab_by_path(path)

    def find_document_by_id(self, document_id: str) -> DocumentState | None:
        """Find a document by its unique ID.

        Args:
            document_id: The document ID to search for.

        Returns:
            The matching DocumentState, or None if not found.
        """
        return self._workspace.find_document_by_id(document_id)

    # ------------------------------------------------------------------
    # Convenience Methods
    # ------------------------------------------------------------------

    def ensure_tab(self) -> DocumentTab:
        """Ensure at least one tab exists, creating one if needed.

        Returns:
            An existing or newly created tab.

        Emits:
            DocumentCreated: If a new tab was created.
        """
        if self._workspace.tab_count() > 0:
            return self._workspace.ensure_tab()

        # No tabs exist, create one through our method to emit events
        return self.create_tab()

    def require_active_tab(self) -> DocumentTab:
        """Get the active tab, raising if none exists.

        Returns:
            The active DocumentTab.

        Raises:
            RuntimeError: If no active tab exists.
        """
        return self._workspace.require_active_tab()

    def active_document(self) -> DocumentState:
        """Get the document from the active tab.

        Returns:
            The active DocumentState.

        Raises:
            RuntimeError: If no active tab exists.
        """
        return self._workspace.active_document()

    # ------------------------------------------------------------------
    # Workspace Access (for advanced use cases)
    # ------------------------------------------------------------------

    @property
    def workspace(self) -> DocumentWorkspace:
        """Access the underlying workspace for advanced operations.

        Note: Prefer using DocumentStore methods when possible to ensure
        proper event emission.
        """
        return self._workspace

    # ------------------------------------------------------------------
    # Internal Event Handling
    # ------------------------------------------------------------------

    def _on_active_tab_changed(self, tab: DocumentTab | None) -> None:
        """Handle active tab changes from the workspace."""
        if tab is None:
            # No active tab (all tabs closed) - could emit a different event
            LOGGER.debug("DocumentStore: active tab cleared (no tabs)")
            return

        doc = tab.document()
        LOGGER.debug(
            "DocumentStore: active tab changed to tab_id=%s, document_id=%s",
            tab.id,
            doc.document_id,
        )

        self._bus.publish(ActiveTabChanged(
            tab_id=tab.id,
            document_id=doc.document_id,
        ))


__all__ = ["DocumentStore"]
