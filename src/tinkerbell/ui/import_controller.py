"""Importer controller extracted from the main window."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from ..editor.document_model import DocumentMetadata, DocumentState
from ..services.importers import FileImporter, ImportResult, ImporterError

LOGGER = logging.getLogger(__name__)


def _noop_status(_: str) -> None:
    return None


def _missing_tab_factory(_: DocumentState, __: str) -> str:  # pragma: no cover - misconfiguration guard
    raise RuntimeError("new_tab_factory callback is required")


def _noop_document_callback(_: DocumentState) -> None:
    return None


def _noop_workspace_callback() -> None:
    return None


def _noop_remember_file(_: Path) -> None:
    return None


class ImportController:
    """Coordinates file import workflows for the main window."""

    def __init__(
        self,
        *,
        file_importer: FileImporter | None = None,
        prompt_for_path: Callable[[], Path | None] | None = None,
        new_tab_factory: Callable[[DocumentState, str], str] | None = None,
        status_updater: Callable[[str], None] | None = None,
        remember_recent_file: Callable[[Path], None] | None = None,
        refresh_window_title: Callable[[DocumentState], None] | None = None,
        sync_workspace_state: Callable[[], None] | None = None,
        update_autosave_indicator: Callable[[DocumentState], None] | None = None,
    ) -> None:
        self._file_importer = file_importer or FileImporter()
        self._prompt_for_path = prompt_for_path
        self._new_tab_factory = new_tab_factory or _missing_tab_factory
        self._status_updater = status_updater or _noop_status
        self._remember_recent_file = remember_recent_file or _noop_remember_file
        self._refresh_window_title = refresh_window_title or _noop_document_callback
        self._sync_workspace_state = sync_workspace_state or _noop_workspace_callback
        self._update_autosave_indicator = update_autosave_indicator or _noop_document_callback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def file_importer(self) -> FileImporter:
        return self._file_importer

    @file_importer.setter
    def file_importer(self, importer: FileImporter) -> None:
        self._file_importer = importer

    def dialog_filter(self) -> str:
        try:
            return self._file_importer.dialog_filter()
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.debug("Import dialog filter failed", exc_info=True)
            return "All Files (*)"

    def handle_import(self) -> None:
        if self._prompt_for_path is None:
            self._status_updater("Import dialog unavailable")
            return

        try:
            path = self._prompt_for_path()
        except RuntimeError:
            self._status_updater("Import dialog unavailable")
            return

        if path is None:
            self._status_updater("Import canceled")
            return

        try:
            result = self._file_importer.import_file(path)
        except FileNotFoundError:
            self._status_updater(f"File not found: {path}")
            return
        except ImporterError as exc:
            message = str(exc).strip() or f"Unable to import {path.name}"
            self._status_updater(message)
            return

        self._open_import_result(result, source_path=path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _open_import_result(self, result: ImportResult, *, source_path: Path) -> None:
        language = (result.language or "text").strip() or "text"
        document = DocumentState(text=result.text, metadata=DocumentMetadata(language=language))
        document.dirty = True
        title = (result.title or source_path.stem or "Imported Document").strip() or "Imported Document"

        try:
            self._new_tab_factory(document, title)
        except Exception:
            LOGGER.debug("Failed to create tab for import", exc_info=True)
            self._status_updater("Import failed while creating tab")
            return

        self._remember_recent_file(source_path)
        self._refresh_window_title(document)
        self._sync_workspace_state()
        self._update_autosave_indicator(document)
        status = f"Imported {source_path.name}"
        if result.notes:
            status = f"{status} – {result.notes}"
        self._status_updater(status)


__all__ = ["ImportController"]
