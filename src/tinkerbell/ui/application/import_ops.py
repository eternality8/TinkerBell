"""Import operation use cases.

This module provides use cases for file import operations:
- ImportDocumentUseCase: Import a non-native file format into a new document
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ...editor.document_model import DocumentMetadata, DocumentState
    from ...services.importers import FileImporter, ImportResult
    from ..domain.document_store import DocumentStore
    from ..domain.session_store import SessionStore
    from ..events import EventBus

LOGGER = logging.getLogger(__name__)


class ImportDialogProvider(Protocol):
    """Protocol for import dialog providers."""

    def prompt_import_path(
        self,
        dialog_filter: str | None = None,
    ) -> Path | None:
        """Prompt user to select a file to import.

        Args:
            dialog_filter: File filter string for the dialog.

        Returns:
            Selected path or None if canceled.
        """
        ...


class DocumentFactory(Protocol):
    """Protocol for creating documents from import results."""

    def create_document(
        self,
        text: str,
        *,
        language: str = "text",
        dirty: bool = True,
    ) -> "DocumentState":
        """Create a new document with the given content.

        Args:
            text: The document text content.
            language: The document language/syntax.
            dirty: Whether to mark the document as dirty.

        Returns:
            A new DocumentState instance.
        """
        ...


@dataclass(slots=True, frozen=True)
class ImportResult:
    """Result of an import operation.

    Attributes:
        success: Whether the import succeeded.
        tab_id: The ID of the created tab (if successful).
        source_path: The path that was imported (if any).
        title: The title assigned to the imported document.
        message: Human-readable status message.
        notes: Additional notes from the importer.
    """

    success: bool
    tab_id: str | None
    source_path: Path | None
    title: str
    message: str
    notes: str | None = None


class ImportDocumentUseCase:
    """Use case for importing a non-native file format.

    Orchestrates importing an external file format:
    1. Prompt user to select a file to import
    2. Convert the file using the appropriate import handler
    3. Create a new document tab with the converted content
    4. Remember the source file in recent files
    5. Sync workspace state

    Events Emitted (via DocumentStore):
        - DocumentCreated: After the import tab is created
        - ActiveTabChanged: When the import tab becomes active
    """

    __slots__ = (
        "_document_store",
        "_session_store",
        "_file_importer",
        "_dialog_provider",
        "_event_bus",
    )

    def __init__(
        self,
        document_store: "DocumentStore",
        session_store: "SessionStore",
        file_importer: "FileImporter",
        dialog_provider: ImportDialogProvider,
        event_bus: "EventBus",
    ) -> None:
        """Initialize the use case.

        Args:
            document_store: Manager for document/tab lifecycle.
            session_store: Manager for session persistence.
            file_importer: Registry of import handlers.
            dialog_provider: Provider for import dialogs.
            event_bus: Event bus for publishing events.
        """
        self._document_store = document_store
        self._session_store = session_store
        self._file_importer = file_importer
        self._dialog_provider = dialog_provider
        self._event_bus = event_bus

    def execute(self, path: Path | None = None) -> ImportResult:
        """Execute the import operation.

        Args:
            path: Optional pre-selected file path. If None, prompts user.

        Returns:
            ImportResult with details about the operation.
        """
        # If no path provided, prompt user
        if path is None:
            try:
                dialog_filter = self._file_importer.dialog_filter()
            except Exception:  # pragma: no cover
                LOGGER.debug(
                    "ImportDocumentUseCase: dialog_filter failed",
                    exc_info=True,
                )
                dialog_filter = "All Files (*)"

            path = self._dialog_provider.prompt_import_path(dialog_filter)
            if path is None:
                LOGGER.debug("ImportDocumentUseCase: user canceled")
                return ImportResult(
                    success=False,
                    tab_id=None,
                    source_path=None,
                    title="",
                    message="Import canceled",
                )

        # Import the file
        try:
            import_result = self._file_importer.import_file(path)
        except FileNotFoundError:
            LOGGER.debug("ImportDocumentUseCase: file not found: %s", path)
            return ImportResult(
                success=False,
                tab_id=None,
                source_path=path,
                title="",
                message=f"File not found: {path}",
            )
        except Exception as exc:
            # Handle ImporterError and other exceptions
            message = str(exc).strip() or f"Unable to import {path.name}"
            LOGGER.debug(
                "ImportDocumentUseCase: import failed: %s",
                message,
                exc_info=True,
            )
            return ImportResult(
                success=False,
                tab_id=None,
                source_path=path,
                title="",
                message=message,
            )

        # Create document from import result
        language = (import_result.language or "text").strip() or "text"
        title = (
            import_result.title or path.stem or "Imported Document"
        ).strip() or "Imported Document"

        try:
            # Create the document via document_store
            # Note: document_store.create_document creates a DocumentState
            # and document_store.create_tab creates the tab
            from ...editor.document_model import DocumentMetadata, DocumentState

            document = DocumentState(
                text=import_result.text,
                metadata=DocumentMetadata(language=language),
            )
            document.dirty = True

            tab_id = self._document_store.create_tab(
                document=document,
                title=title,
                make_active=True,
            )

        except Exception:  # pragma: no cover
            LOGGER.debug(
                "ImportDocumentUseCase: failed to create tab",
                exc_info=True,
            )
            return ImportResult(
                success=False,
                tab_id=None,
                source_path=path,
                title=title,
                message="Import failed while creating tab",
            )

        # Remember the source file in recent files
        try:
            self._session_store.remember_recent_file(path)
        except Exception:  # pragma: no cover
            LOGGER.debug(
                "ImportDocumentUseCase: failed to remember recent file",
                exc_info=True,
            )

        # Clear the current path (imported docs are "new" files)
        self._session_store.set_current_path(None)

        # Sync workspace state
        try:
            self._session_store.sync_workspace_state()
        except Exception:  # pragma: no cover
            LOGGER.debug(
                "ImportDocumentUseCase: failed to sync workspace state",
                exc_info=True,
            )

        # Build success message
        status = f"Imported {path.name}"
        if import_result.notes:
            status = f"{status} – {import_result.notes}"

        LOGGER.debug(
            "ImportDocumentUseCase: imported %s -> tab_id=%s",
            path,
            tab_id,
        )

        return ImportResult(
            success=True,
            tab_id=tab_id,
            source_path=path,
            title=title,
            message=status,
            notes=import_result.notes,
        )

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """Get tuple of supported file extensions."""
        return self._file_importer.supported_extensions()

    @property
    def dialog_filter(self) -> str:
        """Get the dialog filter string for import file dialogs."""
        try:
            return self._file_importer.dialog_filter()
        except Exception:  # pragma: no cover
            return "All Files (*)"


__all__ = [
    "ImportDialogProvider",
    "DocumentFactory",
    "ImportResult",
    "ImportDocumentUseCase",
]
