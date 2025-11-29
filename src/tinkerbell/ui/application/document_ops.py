"""Document operation use cases.

This module provides use cases for document lifecycle operations:
- NewDocumentUseCase: Create a new untitled document
- OpenDocumentUseCase: Open a document from disk
- SaveDocumentUseCase: Save the current document
- CloseDocumentUseCase: Close a document tab
- RevertDocumentUseCase: Revert to the saved version
- RestoreWorkspaceUseCase: Restore workspace from saved session
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ...editor.document_model import DocumentMetadata, DocumentState
    from ...services.settings import Settings
    from ...services.unsaved_cache import UnsavedCache
    from ..domain.document_store import DocumentStore
    from ..domain.review_manager import ReviewManager
    from ..domain.overlay_manager import OverlayManager
    from ..domain.session_store import SessionStore
    from ..events import EventBus

LOGGER = logging.getLogger(__name__)


class DialogProvider(Protocol):
    """Protocol for UI dialog providers."""

    def prompt_open_path(
        self,
        start_dir: Path | None = None,
        token_budget: int | None = None,
    ) -> Path | None:
        """Prompt user to select a file to open."""
        ...

    def prompt_save_path(
        self,
        start_dir: Path | None = None,
        document_text: str | None = None,
        token_budget: int | None = None,
    ) -> Path | None:
        """Prompt user to select a save location."""
        ...


class NewDocumentUseCase:
    """Use case for creating a new untitled document.

    Creates a new tab with an empty document and makes it active.

    Events Emitted:
        - DocumentCreated: After the tab is created
        - ActiveTabChanged: When the new tab becomes active
    """

    __slots__ = ("_document_store", "_session_store", "_event_bus")

    def __init__(
        self,
        document_store: DocumentStore,
        session_store: SessionStore,
        event_bus: EventBus,
    ) -> None:
        """Initialize the use case.

        Args:
            document_store: Store for document/tab management.
            session_store: Store for session persistence.
            event_bus: Event bus for publishing events.
        """
        self._document_store = document_store
        self._session_store = session_store
        self._event_bus = event_bus

    def execute(self) -> str:
        """Create a new untitled document.

        Returns:
            The tab_id of the newly created tab.
        """
        tab = self._document_store.create_tab(make_active=True)
        LOGGER.debug("NewDocumentUseCase: created tab %s", tab.id)
        return tab.id


class OpenDocumentUseCase:
    """Use case for opening a document from disk.

    Handles:
    - File dialog if no path provided
    - Focusing existing tab if file already open
    - Restoring unsaved snapshots if available
    - Updating recent files list

    Events Emitted:
        - DocumentCreated: After the tab is created
        - DocumentOpened: After content is loaded
        - ActiveTabChanged: When the tab becomes active
    """

    __slots__ = (
        "_document_store",
        "_session_store",
        "_event_bus",
        "_dialog_provider",
        "_file_reader",
        "_settings_provider",
        "_cache_provider",
    )

    def __init__(
        self,
        document_store: DocumentStore,
        session_store: SessionStore,
        event_bus: EventBus,
        dialog_provider: DialogProvider | None = None,
        *,
        file_reader: Callable[[Path], str] | None = None,
        settings_provider: Callable[[], Settings | None] | None = None,
        cache_provider: Callable[[], UnsavedCache | None] | None = None,
    ) -> None:
        """Initialize the use case.

        Args:
            document_store: Store for document/tab management.
            session_store: Store for session persistence.
            event_bus: Event bus for publishing events.
            dialog_provider: Provider for file dialogs.
            file_reader: Function to read file content (defaults to file_io.read_text).
            settings_provider: Function to get current settings.
            cache_provider: Function to get unsaved cache.
        """
        self._document_store = document_store
        self._session_store = session_store
        self._event_bus = event_bus
        self._dialog_provider = dialog_provider
        self._file_reader = file_reader
        self._settings_provider = settings_provider or (lambda: None)
        self._cache_provider = cache_provider or (lambda: None)

    def execute(self, path: Path | str | None = None) -> str | None:
        """Open a document from disk.

        Args:
            path: Path to open, or None to show file dialog.

        Returns:
            The tab_id of the opened tab, or None if cancelled/failed.

        Raises:
            FileNotFoundError: If the specified path doesn't exist.
        """
        # If no path provided, prompt user
        if path is None:
            if self._dialog_provider is None:
                LOGGER.warning("OpenDocumentUseCase: no dialog provider")
                return None
            settings = self._settings_provider()
            start_dir = self._resolve_start_dir(settings)
            token_budget = self._resolve_token_budget(settings)
            path = self._dialog_provider.prompt_open_path(
                start_dir=start_dir,
                token_budget=token_budget,
            )
            if path is None:
                LOGGER.debug("OpenDocumentUseCase: user cancelled dialog")
                return None

        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(target)

        # Check if already open
        existing = self._document_store.find_tab_by_path(target)
        if existing is not None:
            self._document_store.set_active_tab(existing.id)
            LOGGER.debug("OpenDocumentUseCase: focused existing tab %s", existing.id)
            return existing.id

        # Read file content
        reader = self._file_reader
        if reader is None:
            from ...utils import file_io
            reader = file_io.read_text

        text = reader(target)

        # Create document
        from ...editor.document_model import DocumentMetadata, DocumentState

        language = self._session_store.infer_language(target)
        metadata = DocumentMetadata(path=target, language=language)
        document = DocumentState(text=text, metadata=metadata)
        document.dirty = False

        # Create tab
        tab = self._document_store.create_tab(
            document=document,
            path=target,
            title=target.name,
            make_active=True,
        )

        # Update session
        self._session_store.set_current_path(target)
        settings = self._settings_provider()
        if settings is not None:
            self._session_store.remember_recent_file(target, settings)

        # Check for unsaved snapshot to restore
        cache = self._cache_provider()
        snapshot = self._session_store.get_unsaved_snapshot(cache, target, tab.id)
        if snapshot is not None:
            self._apply_snapshot(tab.id, snapshot)
            LOGGER.debug(
                "OpenDocumentUseCase: restored unsaved snapshot for %s",
                target,
            )

        # Emit DocumentOpened event
        from ..events import DocumentOpened

        self._event_bus.publish(
            DocumentOpened(
                tab_id=tab.id,
                document_id=tab.document().document_id,
                path=str(target),
            )
        )

        LOGGER.debug("OpenDocumentUseCase: opened %s as tab %s", target, tab.id)
        return tab.id

    def _resolve_start_dir(self, settings: Settings | None) -> Path | None:
        """Resolve the starting directory for the file dialog."""
        current = self._session_store.current_path
        if current is not None and current.parent.exists():
            return current.parent
        if settings is not None:
            for entry in getattr(settings, "recent_files", []) or []:
                candidate = Path(entry).expanduser()
                if candidate.is_dir():
                    return candidate
                if candidate.exists():
                    return candidate.parent
        return Path.home()

    def _resolve_token_budget(self, settings: Settings | None) -> int | None:
        """Resolve the token budget from settings."""
        if settings is None:
            return None
        raw = getattr(settings, "max_context_tokens", None)
        return raw if isinstance(raw, int) else None

    def _apply_snapshot(self, tab_id: str, snapshot: dict[str, Any]) -> None:
        """Apply an unsaved snapshot to a tab."""
        tab = self._document_store.get_tab(tab_id)
        doc = tab.document()
        text = snapshot.get("text", "")
        doc.text = text
        doc.dirty = True


class SaveDocumentUseCase:
    """Use case for saving the current document.

    Handles:
    - Save As dialog if no path set
    - Creating parent directories
    - Updating recent files list
    - Clearing unsaved snapshots

    Events Emitted:
        - DocumentSaved: After the document is saved
    """

    __slots__ = (
        "_document_store",
        "_session_store",
        "_event_bus",
        "_dialog_provider",
        "_file_writer",
        "_settings_provider",
        "_cache_provider",
    )

    def __init__(
        self,
        document_store: DocumentStore,
        session_store: SessionStore,
        event_bus: EventBus,
        dialog_provider: DialogProvider | None = None,
        *,
        file_writer: Callable[[Path, str], None] | None = None,
        settings_provider: Callable[[], Settings | None] | None = None,
        cache_provider: Callable[[], UnsavedCache | None] | None = None,
    ) -> None:
        """Initialize the use case.

        Args:
            document_store: Store for document/tab management.
            session_store: Store for session persistence.
            event_bus: Event bus for publishing events.
            dialog_provider: Provider for file dialogs.
            file_writer: Function to write file content.
            settings_provider: Function to get current settings.
            cache_provider: Function to get unsaved cache.
        """
        self._document_store = document_store
        self._session_store = session_store
        self._event_bus = event_bus
        self._dialog_provider = dialog_provider
        self._file_writer = file_writer
        self._settings_provider = settings_provider or (lambda: None)
        self._cache_provider = cache_provider or (lambda: None)

    def execute(self, path: Path | str | None = None) -> Path:
        """Save the current document.

        Args:
            path: Path to save to, or None to use current/prompt.

        Returns:
            The path the document was saved to.

        Raises:
            RuntimeError: If save was cancelled or no active tab.
        """
        active_tab = self._document_store.active_tab
        if active_tab is None:
            raise RuntimeError("No active tab to save")

        document = active_tab.document()
        previous_path = document.metadata.path

        # Determine target path
        target_path: Path | None
        if path is not None:
            target_path = Path(path)
        else:
            target_path = document.metadata.path
            if target_path is None:
                target_path = self._session_store.current_path

        # If still no path, prompt user
        if target_path is None:
            if self._dialog_provider is None:
                raise RuntimeError("Save cancelled - no path and no dialog provider")
            settings = self._settings_provider()
            start_dir = self._resolve_start_dir(settings)
            token_budget = self._resolve_token_budget(settings)
            target_path = self._dialog_provider.prompt_save_path(
                start_dir=start_dir,
                document_text=document.text,
                token_budget=token_budget,
            )
            if target_path is None:
                raise RuntimeError("Save cancelled")

        # Ensure parent directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Write file
        writer = self._file_writer
        if writer is None:
            from ...utils import file_io
            writer = file_io.write_text

        writer(target_path, document.text)

        # Update document state
        document.metadata.path = target_path
        document.dirty = False

        # Update session
        self._session_store.set_current_path(target_path)
        settings = self._settings_provider()
        if settings is not None:
            self._session_store.remember_recent_file(target_path, settings)

        # Clear unsaved snapshots
        cache = self._cache_provider()
        if cache is not None:
            self._session_store.clear_unsaved_snapshot(cache, target_path, active_tab.id)
            if previous_path is None:
                # Was untitled - also clear untitled snapshot
                self._session_store.clear_unsaved_snapshot(cache, None, active_tab.id)
            self._session_store.persist_unsaved_cache(cache)

        # Emit DocumentSaved event
        from ..events import DocumentSaved

        self._event_bus.publish(
            DocumentSaved(
                tab_id=active_tab.id,
                document_id=document.document_id,
                path=str(target_path),
            )
        )

        LOGGER.debug("SaveDocumentUseCase: saved to %s", target_path)
        return target_path

    def _resolve_start_dir(self, settings: Settings | None) -> Path | None:
        """Resolve the starting directory for the file dialog."""
        current = self._session_store.current_path
        if current is not None and current.parent.exists():
            return current.parent
        if settings is not None:
            for entry in getattr(settings, "recent_files", []) or []:
                candidate = Path(entry).expanduser()
                if candidate.is_dir():
                    return candidate
                if candidate.exists():
                    return candidate.parent
        return Path.home()

    def _resolve_token_budget(self, settings: Settings | None) -> int | None:
        """Resolve the token budget from settings."""
        if settings is None:
            return None
        raw = getattr(settings, "max_context_tokens", None)
        return raw if isinstance(raw, int) else None


class CloseDocumentUseCase:
    """Use case for closing a document tab.

    Handles:
    - Cleaning up review sessions for the tab
    - Clearing overlays for the tab
    - Closing the tab in the document store

    Events Emitted:
        - DocumentClosed: After the tab is closed
        - ActiveTabChanged: If the closed tab was active
    """

    __slots__ = (
        "_document_store",
        "_session_store",
        "_review_manager",
        "_overlay_manager",
        "_event_bus",
    )

    def __init__(
        self,
        document_store: DocumentStore,
        session_store: SessionStore,
        review_manager: ReviewManager,
        overlay_manager: OverlayManager,
        event_bus: EventBus,
    ) -> None:
        """Initialize the use case.

        Args:
            document_store: Store for document/tab management.
            session_store: Store for session persistence.
            review_manager: Manager for pending reviews.
            overlay_manager: Manager for diff overlays.
            event_bus: Event bus for publishing events.
        """
        self._document_store = document_store
        self._session_store = session_store
        self._review_manager = review_manager
        self._overlay_manager = overlay_manager
        self._event_bus = event_bus

    def execute(self, tab_id: str | None = None) -> bool:
        """Close a document tab.

        Args:
            tab_id: The tab to close, or None to close active tab.

        Returns:
            True if a tab was closed, False if not.
        """
        # Resolve tab to close
        if tab_id is None:
            active = self._document_store.active_tab
            if active is None:
                LOGGER.debug("CloseDocumentUseCase: no active tab to close")
                return False
            tab_id = active.id

        # Mark any pending review session as orphaned
        pending = self._review_manager.pending_review
        if pending is not None and tab_id in pending.sessions:
            # The session will be cleaned up when review is finalized
            LOGGER.debug(
                "CloseDocumentUseCase: tab %s has pending review session",
                tab_id,
            )

        # Clear overlay for this tab
        if self._overlay_manager.has_overlay(tab_id):
            self._overlay_manager.clear_overlay(tab_id)

        # Close the tab
        closed = self._document_store.close_tab(tab_id)
        if closed is None:
            LOGGER.debug("CloseDocumentUseCase: tab %s not found", tab_id)
            return False

        LOGGER.debug("CloseDocumentUseCase: closed tab %s", tab_id)
        return True


class RevertDocumentUseCase:
    """Use case for reverting a document to its saved state.

    Reloads the document from disk, discarding unsaved changes.

    Events Emitted:
        - DocumentModified: After content is reverted
    """

    __slots__ = (
        "_document_store",
        "_session_store",
        "_event_bus",
        "_file_reader",
        "_cache_provider",
    )

    def __init__(
        self,
        document_store: DocumentStore,
        session_store: SessionStore,
        event_bus: EventBus,
        *,
        file_reader: Callable[[Path], str] | None = None,
        cache_provider: Callable[[], UnsavedCache | None] | None = None,
    ) -> None:
        """Initialize the use case.

        Args:
            document_store: Store for document/tab management.
            session_store: Store for session persistence.
            event_bus: Event bus for publishing events.
            file_reader: Function to read file content.
            cache_provider: Function to get unsaved cache.
        """
        self._document_store = document_store
        self._session_store = session_store
        self._event_bus = event_bus
        self._file_reader = file_reader
        self._cache_provider = cache_provider or (lambda: None)

    def execute(self) -> bool:
        """Revert the current document to its saved state.

        Returns:
            True if the document was reverted, False if not possible.
        """
        active_tab = self._document_store.active_tab
        if active_tab is None:
            LOGGER.debug("RevertDocumentUseCase: no active tab")
            return False

        document = active_tab.document()
        path = document.metadata.path
        if path is None:
            # Untitled document - can't revert
            LOGGER.debug("RevertDocumentUseCase: untitled document cannot be reverted")
            return False

        target = Path(path)
        if not target.exists():
            LOGGER.debug("RevertDocumentUseCase: file %s not found", target)
            return False

        # Read file content
        reader = self._file_reader
        if reader is None:
            from ...utils import file_io
            reader = file_io.read_text

        text = reader(target)

        # Update document
        document.text = text
        document.dirty = False

        # Clear unsaved snapshot
        cache = self._cache_provider()
        if cache is not None:
            self._session_store.clear_unsaved_snapshot(cache, target, active_tab.id)
            self._session_store.persist_unsaved_cache(cache)

        # Emit DocumentModified event
        from ..events import DocumentModified
        import hashlib

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        self._event_bus.publish(
            DocumentModified(
                tab_id=active_tab.id,
                document_id=document.document_id,
                version_id=document.version_id,
                content_hash=content_hash,
            )
        )

        LOGGER.debug("RevertDocumentUseCase: reverted %s", target)
        return True


class RestoreWorkspaceUseCase:
    """Use case for restoring workspace from saved session state.

    Handles:
    - Aborting pending reviews before restore
    - Closing existing tabs
    - Creating tabs from saved settings
    - Loading file content or applying unsaved snapshots
    - Setting active tab
    - Cleaning up orphan snapshots

    Events Emitted:
        - DocumentClosed: For each closed tab
        - DocumentCreated: For each restored tab
        - ActiveTabChanged: When active tab is set
        - WorkspaceRestored: After restore completes
    """

    __slots__ = (
        "_document_store",
        "_session_store",
        "_review_manager",
        "_overlay_manager",
        "_event_bus",
        "_file_reader",
    )

    def __init__(
        self,
        document_store: DocumentStore,
        session_store: SessionStore,
        review_manager: ReviewManager,
        overlay_manager: OverlayManager,
        event_bus: EventBus,
        *,
        file_reader: Callable[[Path], str] | None = None,
    ) -> None:
        """Initialize the use case.

        Args:
            document_store: Store for document/tab management.
            session_store: Store for session persistence.
            review_manager: Manager for pending reviews.
            overlay_manager: Manager for diff overlays.
            event_bus: Event bus for publishing events.
            file_reader: Function to read file content.
        """
        self._document_store = document_store
        self._session_store = session_store
        self._review_manager = review_manager
        self._overlay_manager = overlay_manager
        self._event_bus = event_bus
        self._file_reader = file_reader

    def execute(self, settings: Settings, cache: UnsavedCache | None) -> bool:
        """Restore workspace from saved session state.

        Args:
            settings: Settings containing saved workspace state.
            cache: Unsaved cache for snapshot restoration.

        Returns:
            True if workspace was restored, False otherwise.
        """
        # Get open_tabs from settings
        open_tabs = getattr(settings, "open_tabs", None)
        open_tabs_was_set = open_tabs is not None
        entries = [e for e in (open_tabs or []) if isinstance(e, dict)]

        LOGGER.debug(
            "RestoreWorkspaceUseCase: will restore %d tabs, open_tabs_was_set=%s",
            len(entries),
            open_tabs_was_set,
        )

        # Abort any pending review
        pending = self._review_manager.pending_review
        if pending is not None:
            self._review_manager.drop("workspace-restore")
            self._overlay_manager.clear_all_overlays()

        # Close existing tabs
        for tab_id in list(self._document_store.tab_ids()):
            self._document_store.close_tab(tab_id)

        # If open_tabs was explicitly empty, we're done
        if open_tabs_was_set and not entries:
            self._emit_workspace_restored(0, None)
            return True

        if not entries:
            return False

        # Restore tabs
        restored_ids: list[str] = []
        for entry in entries:
            tab = self._create_tab_from_entry(entry, cache)
            if tab is not None:
                restored_ids.append(tab.id)

        if not restored_ids:
            self._emit_workspace_restored(0, None)
            return False

        # Set active tab
        active_id = getattr(settings, "active_tab_id", None) or restored_ids[-1]
        if active_id not in self._document_store.tab_ids():
            active_id = restored_ids[-1]
        self._document_store.set_active_tab(active_id)

        # Restore untitled counter
        next_index = getattr(settings, "next_untitled_index", None)
        if isinstance(next_index, int):
            workspace = self._document_store.workspace
            setter = getattr(workspace, "set_next_untitled_index", None)
            if callable(setter):
                setter(next_index)

        # Cleanup orphan snapshots
        if cache is not None:
            if self._session_store.cleanup_orphan_snapshots(cache, entries):
                self._session_store.persist_unsaved_cache(cache)

        self._emit_workspace_restored(len(restored_ids), active_id)
        LOGGER.debug(
            "RestoreWorkspaceUseCase: restored %d tabs, active=%s",
            len(restored_ids),
            active_id,
        )
        return True

    def _create_tab_from_entry(
        self,
        entry: dict[str, Any],
        cache: UnsavedCache | None,
    ) -> Any | None:
        """Create a tab from a settings entry.

        Args:
            entry: Settings entry with tab info.
            cache: Unsaved cache for snapshot restoration.

        Returns:
            The created DocumentTab, or None if failed.
        """
        from ...editor.document_model import DocumentMetadata, DocumentState

        # Extract entry data
        title = str(entry.get("title") or "Untitled")
        path_value = entry.get("path")
        path = Path(path_value).expanduser() if path_value else None
        language = str(
            entry.get("language")
            or (self._session_store.infer_language(path) if path else "markdown")
        )

        # Create document
        document = DocumentState(
            text="",
            metadata=DocumentMetadata(path=path, language=language),
        )
        document.dirty = bool(entry.get("dirty", False))

        # Load file content if path exists
        if path is not None and path.exists():
            reader = self._file_reader
            if reader is None:
                from ...utils import file_io
                reader = file_io.read_text
            try:
                document.text = reader(path)
            except Exception as exc:
                LOGGER.warning(
                    "RestoreWorkspaceUseCase: failed to read %s: %s",
                    path,
                    exc,
                )
                document.text = ""

        # Create tab (with specific tab_id if provided)
        tab_id_hint = entry.get("tab_id")
        tab = self._document_store.create_tab(
            document=document,
            path=path,
            title=title,
            make_active=False,  # We'll set active later
        )

        # Apply unsaved snapshot if available
        if cache is not None:
            snapshot = self._session_store.get_unsaved_snapshot(
                cache,
                path,
                tab.id,
            )
            if snapshot is not None:
                text = snapshot.get("text", "")
                doc = tab.document()
                doc.text = text
                doc.dirty = True
                LOGGER.debug(
                    "RestoreWorkspaceUseCase: applied snapshot for tab %s",
                    tab.id,
                )

        return tab

    def _emit_workspace_restored(
        self,
        tab_count: int,
        active_tab_id: str | None,
    ) -> None:
        """Emit WorkspaceRestored event."""
        from ..events import WorkspaceRestored

        self._event_bus.publish(
            WorkspaceRestored(
                tab_count=tab_count,
                active_tab_id=active_tab_id,
            )
        )


__all__ = [
    "DialogProvider",
    "NewDocumentUseCase",
    "OpenDocumentUseCase",
    "SaveDocumentUseCase",
    "CloseDocumentUseCase",
    "RevertDocumentUseCase",
    "RestoreWorkspaceUseCase",
]
