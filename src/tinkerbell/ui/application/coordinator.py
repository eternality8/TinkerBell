"""Application coordinator facade.

This module provides the AppCoordinator - a facade that delegates
to individual use cases and provides a unified API for the presentation layer.

The coordinator:
- Owns all use case instances
- Provides facade methods that delegate to use cases
- Manages widget references for presentation layer callbacks
- Subscribes to domain events for cross-cutting concerns
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ...services.importers import FileImporter
    from ...services.settings import Settings
    from ...services.unsaved_cache import UnsavedCache
    from ..domain.ai_turn_manager import AITurnManager
    from ..domain.document_store import DocumentStore
    from ..domain.overlay_manager import OverlayManager
    from ..domain.review_manager import ReviewManager
    from ..domain.session_store import SessionStore
    from ..events import EventBus
    from ..models.ai_models import AITurnState
    from .ai_ops import SnapshotProvider
    from .document_ops import DialogProvider
    from .import_ops import ImportDialogProvider
    from .review_ops import (
        AcceptResult,
        ChatRestorer,
        DocumentRestorer,
        OverlayRestorer,
        RejectResult,
        WorkspaceSyncer,
    )

LOGGER = logging.getLogger(__name__)


class AppCoordinator:
    """Facade coordinating all application use cases.

    Provides a unified API for the presentation layer to interact with
    the application. All operations delegate to the appropriate use case,
    ensuring consistent behavior and proper event emission.

    The coordinator is the main entry point for application operations,
    replacing direct MainWindow method sprawl with focused, testable
    use case objects.

    Example:
        coordinator = AppCoordinator(
            event_bus=event_bus,
            document_store=document_store,
            session_store=session_store,
            ai_turn_manager=ai_turn_manager,
            review_manager=review_manager,
            overlay_manager=overlay_manager,
        )

        # Document operations
        coordinator.new_document()
        coordinator.open_document()
        coordinator.save_document()

        # AI operations
        await coordinator.run_ai_turn("Write a poem")
        coordinator.cancel_ai_turn()

        # Review operations
        coordinator.accept_review()
        coordinator.reject_review()
    """

    __slots__ = (
        # Core dependencies
        "_event_bus",
        "_document_store",
        "_session_store",
        "_ai_turn_manager",
        "_review_manager",
        "_overlay_manager",
        # Use cases (lazily constructed)
        "_new_document_uc",
        "_open_document_uc",
        "_save_document_uc",
        "_close_document_uc",
        "_revert_document_uc",
        "_restore_workspace_uc",
        "_run_ai_turn_uc",
        "_cancel_ai_turn_uc",
        "_accept_review_uc",
        "_reject_review_uc",
        "_import_document_uc",
        # Providers/callbacks for use case construction
        "_dialog_provider",
        "_import_dialog_provider",
        "_snapshot_provider",
        "_document_restorer",
        "_overlay_restorer",
        "_chat_restorer",
        "_workspace_syncer",
        "_file_importer",
        "_settings_provider",
        "_cache_provider",
        "_metadata_enricher",
        "_history_provider",
        "_stream_handler",
    )

    def __init__(
        self,
        event_bus: "EventBus",
        document_store: "DocumentStore",
        session_store: "SessionStore",
        ai_turn_manager: "AITurnManager",
        review_manager: "ReviewManager",
        overlay_manager: "OverlayManager",
        *,
        dialog_provider: "DialogProvider | None" = None,
        import_dialog_provider: "ImportDialogProvider | None" = None,
        snapshot_provider: "SnapshotProvider | None" = None,
        document_restorer: "DocumentRestorer | None" = None,
        overlay_restorer: "OverlayRestorer | None" = None,
        chat_restorer: "ChatRestorer | None" = None,
        workspace_syncer: "WorkspaceSyncer | None" = None,
        file_importer: "FileImporter | None" = None,
        settings_provider: Callable[[], "Settings | None"] | None = None,
        cache_provider: Callable[[], "UnsavedCache | None"] | None = None,
        metadata_enricher: Callable[[dict[str, Any]], None] | None = None,
        history_provider: Callable[[], Sequence[Mapping[str, str]] | None] | None = None,
        stream_handler: Callable[[Any], None] | None = None,
    ) -> None:
        """Initialize the coordinator.

        Args:
            event_bus: Event bus for publishing events.
            document_store: Manager for document/tab lifecycle.
            session_store: Manager for session persistence.
            ai_turn_manager: Manager for AI turn execution.
            review_manager: Manager for pending reviews.
            overlay_manager: Manager for diff overlays.
            dialog_provider: Provider for file dialogs.
            import_dialog_provider: Provider for import dialogs.
            snapshot_provider: Provider for document snapshots.
            document_restorer: Restorer for document state (reject).
            overlay_restorer: Restorer for overlay state (reject).
            chat_restorer: Restorer for chat state (reject).
            workspace_syncer: Syncer for workspace persistence.
            file_importer: Registry of import handlers.
            settings_provider: Function to get current settings.
            cache_provider: Function to get unsaved cache.
            metadata_enricher: Callback to enrich AI turn metadata.
            history_provider: Callback to get chat history.
            stream_handler: Callback for AI streaming events.
        """
        # Core dependencies
        self._event_bus = event_bus
        self._document_store = document_store
        self._session_store = session_store
        self._ai_turn_manager = ai_turn_manager
        self._review_manager = review_manager
        self._overlay_manager = overlay_manager

        # Providers/callbacks
        self._dialog_provider = dialog_provider
        self._import_dialog_provider = import_dialog_provider
        self._snapshot_provider = snapshot_provider
        self._document_restorer = document_restorer
        self._overlay_restorer = overlay_restorer
        self._chat_restorer = chat_restorer
        self._workspace_syncer = workspace_syncer
        self._file_importer = file_importer
        self._settings_provider = settings_provider
        self._cache_provider = cache_provider
        self._metadata_enricher = metadata_enricher
        self._history_provider = history_provider
        self._stream_handler = stream_handler

        # Use cases (lazy initialization)
        self._new_document_uc: Any = None
        self._open_document_uc: Any = None
        self._save_document_uc: Any = None
        self._close_document_uc: Any = None
        self._revert_document_uc: Any = None
        self._restore_workspace_uc: Any = None
        self._run_ai_turn_uc: Any = None
        self._cancel_ai_turn_uc: Any = None
        self._accept_review_uc: Any = None
        self._reject_review_uc: Any = None
        self._import_document_uc: Any = None

    # ------------------------------------------------------------------
    # Provider/Callback Setters (for presentation layer binding)
    # ------------------------------------------------------------------

    def set_dialog_provider(self, provider: "DialogProvider") -> None:
        """Set the dialog provider for file operations."""
        self._dialog_provider = provider
        # Invalidate cached use cases that depend on this
        self._open_document_uc = None
        self._save_document_uc = None

    def set_import_dialog_provider(self, provider: "ImportDialogProvider") -> None:
        """Set the dialog provider for import operations."""
        self._import_dialog_provider = provider
        self._import_document_uc = None

    def set_snapshot_provider(self, provider: "SnapshotProvider") -> None:
        """Set the snapshot provider for AI operations."""
        self._snapshot_provider = provider
        self._run_ai_turn_uc = None

    def set_document_restorer(self, restorer: "DocumentRestorer") -> None:
        """Set the document restorer for reject operations."""
        self._document_restorer = restorer
        self._reject_review_uc = None

    def set_overlay_restorer(self, restorer: "OverlayRestorer") -> None:
        """Set the overlay restorer for reject operations."""
        self._overlay_restorer = restorer
        self._reject_review_uc = None

    def set_chat_restorer(self, restorer: "ChatRestorer") -> None:
        """Set the chat restorer for reject operations."""
        self._chat_restorer = restorer
        self._reject_review_uc = None

    def set_workspace_syncer(self, syncer: "WorkspaceSyncer") -> None:
        """Set the workspace syncer for persistence."""
        self._workspace_syncer = syncer
        self._accept_review_uc = None
        self._reject_review_uc = None

    def set_file_importer(self, importer: "FileImporter") -> None:
        """Set the file importer for import operations."""
        self._file_importer = importer
        self._import_document_uc = None

    def set_metadata_enricher(
        self,
        enricher: Callable[[dict[str, Any]], None],
    ) -> None:
        """Set the metadata enricher for AI operations."""
        self._metadata_enricher = enricher
        self._run_ai_turn_uc = None

    def set_history_provider(
        self,
        provider: Callable[[], Sequence[Mapping[str, str]] | None],
    ) -> None:
        """Set the history provider for AI operations."""
        self._history_provider = provider
        self._run_ai_turn_uc = None

    def set_stream_handler(self, handler: Callable[[Any], None]) -> None:
        """Set the stream handler for AI operations."""
        self._stream_handler = handler
        self._run_ai_turn_uc = None

    # ------------------------------------------------------------------
    # Document Operations
    # ------------------------------------------------------------------

    def new_document(self) -> str:
        """Create a new untitled document.

        Returns:
            The tab_id of the newly created tab.
        """
        uc = self._get_new_document_uc()
        return uc.execute()

    def open_document(self, path: Path | str | None = None) -> str | None:
        """Open a document from disk.

        Args:
            path: Path to open, or None to prompt user.

        Returns:
            The tab_id of the opened document, or None if cancelled.
        """
        uc = self._get_open_document_uc()
        return uc.execute(path)

    def save_document(self, path: Path | str | None = None) -> Path | None:
        """Save the current document.

        Args:
            path: Path to save to, or None to use current/prompt.

        Returns:
            The path saved to, or None if cancelled/failed.
        """
        uc = self._get_save_document_uc()
        try:
            return uc.execute(path)
        except RuntimeError as exc:
            LOGGER.debug("AppCoordinator.save_document: %s", exc)
            return None

    def save_document_as(self) -> Path | None:
        """Save the current document to a new path.

        Always prompts for a new path, ignoring any existing path.

        Returns:
            The path saved to, or None if cancelled/failed.
        """
        # Force Save As by clearing current path temporarily
        uc = self._get_save_document_uc()
        active_tab = self._document_store.active_tab
        if active_tab is None:
            LOGGER.debug("AppCoordinator.save_document_as: no active tab")
            return None

        # Save original path
        document = active_tab.document()
        original_path = document.metadata.path

        # Clear path to force dialog
        document.metadata.path = None
        try:
            return uc.execute()
        except RuntimeError as exc:
            # Restore path on cancel
            document.metadata.path = original_path
            LOGGER.debug("AppCoordinator.save_document_as: %s", exc)
            return None

    def save_workspace_state(self) -> bool:
        """Save the current workspace state to settings.

        This method saves both workspace metadata (open tabs, active tab) and
        the content of dirty/untitled documents to the unsaved cache. This
        ensures that document content created by AI tools is preserved across
        app restarts.

        Returns:
            True if workspace state was saved successfully.
        """
        workspace = self._document_store.workspace
        workspace_state = workspace.serialize_state()
        settings = self._settings_provider() if self._settings_provider else None

        # Save dirty document content to the unsaved cache
        cache = self._cache_provider() if self._cache_provider else None
        if cache is not None:
            self._save_dirty_documents_to_cache(cache)
            self._session_store.persist_unsaved_cache(cache)

        return self._session_store.sync_workspace_state(
            workspace_state,
            settings,
            persist=True,
        )

    def _save_dirty_documents_to_cache(self, cache: "UnsavedCache") -> None:
        """Save content of dirty documents to the unsaved cache.

        For documents with a file path, saves to unsaved_snapshots keyed by path.
        For untitled documents (no path), saves to untitled_snapshots keyed by tab_id.

        Args:
            cache: The unsaved cache to update.
        """
        for tab in self._document_store.iter_tabs():
            document = tab.document()
            if not document.dirty:
                continue

            snapshot = {
                "text": document.text,
                "language": document.metadata.language,
            }

            path = document.metadata.path
            if path is not None:
                # Document has a file path - save to unsaved_snapshots
                key = self._session_store.normalize_snapshot_key(path, tab.id)
                if cache.unsaved_snapshots is None:
                    cache.unsaved_snapshots = {}
                cache.unsaved_snapshots[key] = snapshot
                LOGGER.debug(
                    "AppCoordinator: saved dirty document snapshot for path %s",
                    path,
                )
            else:
                # Untitled document - save to untitled_snapshots keyed by tab_id
                if cache.untitled_snapshots is None:
                    cache.untitled_snapshots = {}
                cache.untitled_snapshots[tab.id] = snapshot
                LOGGER.debug(
                    "AppCoordinator: saved untitled document snapshot for tab %s",
                    tab.id,
                )

    def close_document(self, tab_id: str | None = None) -> bool:
        """Close a document tab.

        Args:
            tab_id: Tab to close, or None for active tab.

        Returns:
            True if the tab was closed.
        """
        uc = self._get_close_document_uc()
        return uc.execute(tab_id)

    def revert_document(self) -> bool:
        """Revert the current document to its saved state.

        Returns:
            True if the document was reverted.
        """
        uc = self._get_revert_document_uc()
        return uc.execute()

    def restore_workspace(
        self,
        settings: "Settings",
        cache: "UnsavedCache",
    ) -> bool:
        """Restore workspace from saved session.

        Args:
            settings: Settings containing workspace state.
            cache: Unsaved document cache.

        Returns:
            True if workspace was restored.
        """
        uc = self._get_restore_workspace_uc()
        return uc.execute(settings, cache)

    # ------------------------------------------------------------------
    # AI Operations
    # ------------------------------------------------------------------

    async def run_ai_turn(
        self,
        prompt: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        chat_snapshot: Mapping[str, Any] | None = None,
    ) -> "AITurnState":
        """Execute an AI turn.

        Args:
            prompt: The user prompt to send to the AI.
            metadata: Optional metadata for the turn.
            chat_snapshot: Optional chat state for restoration.

        Returns:
            The AITurnState representing this turn.

        Raises:
            RuntimeError: If AI is unavailable or turn already running.
        """
        uc = self._get_run_ai_turn_uc()
        return await uc.execute(prompt, metadata, chat_snapshot=chat_snapshot)

    def cancel_ai_turn(self) -> bool:
        """Cancel a running AI turn.

        Returns:
            True if a turn was canceled.
        """
        uc = self._get_cancel_ai_turn_uc()
        return uc.execute()

    async def suggest_followups(
        self,
        history: Sequence[Mapping[str, str]],
        *,
        max_suggestions: int = 4,
    ) -> list[str]:
        """Generate follow-up suggestions based on chat history.

        Args:
            history: Conversation history as role/content mappings.
            max_suggestions: Maximum number of suggestions to generate.

        Returns:
            List of suggested follow-up prompts.
        """
        if self._ai_turn_manager is None:
            return []
        return await self._ai_turn_manager.suggest_followups(
            history,
            max_suggestions=max_suggestions,
        )

    # ------------------------------------------------------------------
    # Review Operations
    # ------------------------------------------------------------------

    def accept_review(self) -> "AcceptResult":
        """Accept pending AI edits.

        Returns:
            AcceptResult with operation details.
        """
        uc = self._get_accept_review_uc()
        return uc.execute()

    def reject_review(self) -> "RejectResult":
        """Reject pending AI edits and restore documents.

        Returns:
            RejectResult with operation details.
        """
        uc = self._get_reject_review_uc()
        return uc.execute()

    # ------------------------------------------------------------------
    # Import Operations
    # ------------------------------------------------------------------

    def import_document(self, path: Path | None = None) -> str | None:
        """Import a non-native file format.

        Args:
            path: Path to import, or None to prompt user.

        Returns:
            The tab_id of the imported document, or None if cancelled.
        """
        uc = self._get_import_document_uc()
        if uc is None:
            LOGGER.debug("AppCoordinator.import_document: no importer configured")
            return None
        result = uc.execute(path)
        return result.tab_id if result.success else None

    # ------------------------------------------------------------------
    # Snapshot Operations
    # ------------------------------------------------------------------

    def refresh_snapshot(
        self,
        *,
        tab_id: str | None = None,
        delta_only: bool = False,
    ) -> dict[str, Any] | None:
        """Generate a fresh document snapshot.

        Args:
            tab_id: Optional specific tab to snapshot.
            delta_only: If True, only include changes since last snapshot.

        Returns:
            Snapshot dictionary, or None if no snapshot provider.
        """
        if self._snapshot_provider is None:
            return None
        return self._snapshot_provider.generate_snapshot(
            tab_id=tab_id,
            delta_only=delta_only,
        )

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    @property
    def has_pending_review(self) -> bool:
        """Check if there is a pending review."""
        return self._review_manager.has_pending_review()

    @property
    def is_review_ready(self) -> bool:
        """Check if the pending review is ready for decision."""
        return self._review_manager.is_ready_for_review()

    @property
    def is_ai_running(self) -> bool:
        """Check if an AI turn is currently running."""
        return self._ai_turn_manager.is_running()

    @property
    def active_tab_id(self) -> str | None:
        """Get the active tab ID."""
        tab = self._document_store.active_tab
        return tab.id if tab else None

    @property
    def active_tab(self) -> Any:
        """Get the active tab, if any."""
        return self._document_store.active_tab

    # ------------------------------------------------------------------
    # Document Status
    # ------------------------------------------------------------------

    def get_document_descriptors(self) -> list["DocumentDescriptor"]:
        """Get descriptors for all open documents.

        Returns:
            List of DocumentDescriptor for each open tab.
        """
        from ..document_status import DocumentDescriptor

        descriptors = []
        for tab in self._document_store.iter_tabs():
            doc = tab.document
            label = tab.title or doc.path.name if doc.path else f"Untitled {tab.id}"
            descriptors.append(DocumentDescriptor(
                document_id=doc.document_id,
                label=label,
                tab_id=tab.id,
            ))
        return descriptors

    def get_document_status(self, document_id: str | None = None) -> dict[str, Any]:
        """Get the status payload for a document.

        Args:
            document_id: The document ID to get status for.
                        If None, uses the active document.

        Returns:
            Status payload dictionary with document metadata.
        """
        # Find the document
        doc = None
        tab = None

        if document_id is not None:
            doc = self._document_store.find_document_by_id(document_id)
            # Find the tab for this document
            for t in self._document_store.iter_tabs():
                if t.document.document_id == document_id:
                    tab = t
                    break
        else:
            tab = self._document_store.active_tab
            if tab is not None:
                doc = tab.document

        if doc is None:
            return {"document": {"document_id": document_id or "unknown"}, "error": "Document not found"}

        # Build basic document info
        doc_payload: dict[str, Any] = {
            "document_id": doc.document_id,
            "path": str(doc.path) if doc.path else None,
            "label": tab.title if tab else doc.path.name if doc.path else "Untitled",
            "dirty": doc.dirty,
            "line_count": len(doc.content.splitlines()) if doc.content else 0,
            "char_count": len(doc.content) if doc.content else 0,
        }

        # Try to get snapshot data for richer status
        snapshot_data: dict[str, Any] = {}
        if self._snapshot_provider is not None:
            tab_id = tab.id if tab else None
            try:
                snapshot_data = self._snapshot_provider.generate_snapshot(tab_id=tab_id) or {}
            except Exception:
                LOGGER.debug("Failed to generate snapshot for status", exc_info=True)

        return {
            "document": doc_payload,
            "chunks": snapshot_data.get("chunks", {}),
            "outline": snapshot_data.get("outline", {}),
            "plot": snapshot_data.get("plot", {}),
            "telemetry": snapshot_data.get("telemetry", {}),
            "concordance": snapshot_data.get("concordance", {}),
            "planner": snapshot_data.get("planner", {}),
            "summary": f"Document: {doc_payload['label']}",
        }

    # ------------------------------------------------------------------
    # Use Case Factory Methods (Lazy Initialization)
    # ------------------------------------------------------------------

    def _get_new_document_uc(self) -> Any:
        if self._new_document_uc is None:
            from .document_ops import NewDocumentUseCase

            self._new_document_uc = NewDocumentUseCase(
                document_store=self._document_store,
                session_store=self._session_store,
                event_bus=self._event_bus,
            )
        return self._new_document_uc

    def _get_open_document_uc(self) -> Any:
        if self._open_document_uc is None:
            from .document_ops import OpenDocumentUseCase

            self._open_document_uc = OpenDocumentUseCase(
                document_store=self._document_store,
                session_store=self._session_store,
                event_bus=self._event_bus,
                dialog_provider=self._dialog_provider,
                settings_provider=self._settings_provider,
                cache_provider=self._cache_provider,
            )
        return self._open_document_uc

    def _get_save_document_uc(self) -> Any:
        if self._save_document_uc is None:
            from .document_ops import SaveDocumentUseCase

            self._save_document_uc = SaveDocumentUseCase(
                document_store=self._document_store,
                session_store=self._session_store,
                event_bus=self._event_bus,
                dialog_provider=self._dialog_provider,
                settings_provider=self._settings_provider,
                cache_provider=self._cache_provider,
            )
        return self._save_document_uc

    def _get_close_document_uc(self) -> Any:
        if self._close_document_uc is None:
            from .document_ops import CloseDocumentUseCase

            self._close_document_uc = CloseDocumentUseCase(
                document_store=self._document_store,
                session_store=self._session_store,
                review_manager=self._review_manager,
                overlay_manager=self._overlay_manager,
                event_bus=self._event_bus,
            )
        return self._close_document_uc

    def _get_revert_document_uc(self) -> Any:
        if self._revert_document_uc is None:
            from .document_ops import RevertDocumentUseCase

            self._revert_document_uc = RevertDocumentUseCase(
                document_store=self._document_store,
                session_store=self._session_store,
                event_bus=self._event_bus,
            )
        return self._revert_document_uc

    def _get_restore_workspace_uc(self) -> Any:
        if self._restore_workspace_uc is None:
            from .document_ops import RestoreWorkspaceUseCase

            self._restore_workspace_uc = RestoreWorkspaceUseCase(
                document_store=self._document_store,
                session_store=self._session_store,
                review_manager=self._review_manager,
                overlay_manager=self._overlay_manager,
                event_bus=self._event_bus,
            )
        return self._restore_workspace_uc

    def _get_run_ai_turn_uc(self) -> Any:
        if self._run_ai_turn_uc is None:
            from .ai_ops import RunAITurnUseCase

            if self._snapshot_provider is None:
                raise RuntimeError("Cannot run AI turn without snapshot provider")

            self._run_ai_turn_uc = RunAITurnUseCase(
                ai_turn_manager=self._ai_turn_manager,
                review_manager=self._review_manager,
                snapshot_provider=self._snapshot_provider,
                event_bus=self._event_bus,
                metadata_enricher=self._metadata_enricher,
                history_provider=self._history_provider,
                stream_handler=self._stream_handler,
            )
        return self._run_ai_turn_uc

    def _get_cancel_ai_turn_uc(self) -> Any:
        if self._cancel_ai_turn_uc is None:
            from .ai_ops import CancelAITurnUseCase

            self._cancel_ai_turn_uc = CancelAITurnUseCase(
                ai_turn_manager=self._ai_turn_manager,
                review_manager=self._review_manager,
                event_bus=self._event_bus,
            )
        return self._cancel_ai_turn_uc

    def _get_accept_review_uc(self) -> Any:
        if self._accept_review_uc is None:
            from .review_ops import AcceptReviewUseCase

            self._accept_review_uc = AcceptReviewUseCase(
                review_manager=self._review_manager,
                overlay_manager=self._overlay_manager,
                event_bus=self._event_bus,
                workspace_syncer=self._workspace_syncer,
            )
        return self._accept_review_uc

    def _get_reject_review_uc(self) -> Any:
        if self._reject_review_uc is None:
            from .review_ops import RejectReviewUseCase

            if self._document_restorer is None:
                raise RuntimeError("Cannot reject review without document restorer")

            self._reject_review_uc = RejectReviewUseCase(
                review_manager=self._review_manager,
                overlay_manager=self._overlay_manager,
                document_restorer=self._document_restorer,
                event_bus=self._event_bus,
                overlay_restorer=self._overlay_restorer,
                chat_restorer=self._chat_restorer,
                workspace_syncer=self._workspace_syncer,
            )
        return self._reject_review_uc

    def _get_import_document_uc(self) -> Any | None:
        if self._import_document_uc is None:
            if self._file_importer is None or self._import_dialog_provider is None:
                return None

            from .import_ops import ImportDocumentUseCase

            self._import_document_uc = ImportDocumentUseCase(
                document_store=self._document_store,
                session_store=self._session_store,
                file_importer=self._file_importer,
                dialog_provider=self._import_dialog_provider,
                event_bus=self._event_bus,
            )
        return self._import_document_uc


__all__ = ["AppCoordinator"]
