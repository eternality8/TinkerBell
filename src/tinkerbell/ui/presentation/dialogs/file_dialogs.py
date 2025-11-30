"""File dialog implementations for the presentation layer.

This module provides dialog provider classes that implement the
DialogProvider and ImportDialogProvider protocols from the application layer.

These classes wrap the Qt-based dialogs from tinkerbell.ui.presentation.widgets.dialogs
and provide a clean interface for the use cases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from PySide6.QtWidgets import QWidget

LOGGER = logging.getLogger(__name__)


class FileDialogProvider:
    """Provider for file open and save dialogs.

    Implements the DialogProvider protocol from the application layer,
    wrapping the Qt-based dialogs with a clean interface.

    Example:
        provider = FileDialogProvider(
            parent_provider=lambda: main_window,
            start_dir_resolver=lambda: Path.home(),
            token_budget_resolver=lambda: 128000,
        )

        path = provider.prompt_open_path()
        if path:
            # User selected a file
            ...
    """

    __slots__ = (
        "_parent_provider",
        "_start_dir_resolver",
        "_token_budget_resolver",
        "_enable_samples",
    )

    def __init__(
        self,
        *,
        parent_provider: Callable[[], "QWidget | None"] | None = None,
        start_dir_resolver: Callable[[], Path | None] | None = None,
        token_budget_resolver: Callable[[], int | None] | None = None,
        enable_samples: bool = True,
    ) -> None:
        """Initialize the dialog provider.

        Args:
            parent_provider: Function returning the parent widget for dialogs.
            start_dir_resolver: Function returning the starting directory.
            token_budget_resolver: Function returning the token budget.
            enable_samples: Whether to show sample documents in open dialog.
        """
        self._parent_provider = parent_provider
        self._start_dir_resolver = start_dir_resolver
        self._token_budget_resolver = token_budget_resolver
        self._enable_samples = enable_samples

    def prompt_open_path(
        self,
        start_dir: Path | None = None,
        token_budget: int | None = None,
    ) -> Path | None:
        """Prompt user to select a file to open.

        Args:
            start_dir: Starting directory (overrides resolver).
            token_budget: Token budget hint (overrides resolver).

        Returns:
            Selected path or None if canceled.
        """
        try:
            from tinkerbell.ui.presentation.widgets.dialogs import open_file_dialog
        except ImportError as exc:  # pragma: no cover
            LOGGER.warning("File dialogs require PySide6: %s", exc)
            return None

        parent = self._parent_provider() if self._parent_provider else None
        resolved_start_dir = start_dir
        if resolved_start_dir is None and self._start_dir_resolver:
            resolved_start_dir = self._start_dir_resolver()
        resolved_budget = token_budget
        if resolved_budget is None and self._token_budget_resolver:
            resolved_budget = self._token_budget_resolver()

        return open_file_dialog(
            parent=parent,
            start_dir=resolved_start_dir,
            token_budget=resolved_budget,
            enable_samples=self._enable_samples,
        )

    def prompt_save_path(
        self,
        start_dir: Path | None = None,
        document_text: str | None = None,
        token_budget: int | None = None,
    ) -> Path | None:
        """Prompt user to select a save location.

        Args:
            start_dir: Starting directory (overrides resolver).
            document_text: Document text for preview/validation.
            token_budget: Token budget hint (overrides resolver).

        Returns:
            Selected path or None if canceled.
        """
        try:
            from tinkerbell.ui.presentation.widgets.dialogs import save_file_dialog
        except ImportError as exc:  # pragma: no cover
            LOGGER.warning("File dialogs require PySide6: %s", exc)
            return None

        parent = self._parent_provider() if self._parent_provider else None
        resolved_start_dir = start_dir
        if resolved_start_dir is None and self._start_dir_resolver:
            resolved_start_dir = self._start_dir_resolver()
        resolved_budget = token_budget
        if resolved_budget is None and self._token_budget_resolver:
            resolved_budget = self._token_budget_resolver()

        return save_file_dialog(
            parent=parent,
            start_dir=resolved_start_dir,
            document_text=document_text,
            token_budget=resolved_budget,
        )


class ImportDialogProvider:
    """Provider for file import dialogs.

    Implements the ImportDialogProvider protocol from the application layer,
    wrapping the Qt-based dialogs with support for custom file filters.

    Example:
        provider = ImportDialogProvider(
            parent_provider=lambda: main_window,
            start_dir_resolver=lambda: Path.home(),
        )

        path = provider.prompt_import_path(dialog_filter="PDF Files (*.pdf)")
        if path:
            # User selected a file to import
            ...
    """

    __slots__ = (
        "_parent_provider",
        "_start_dir_resolver",
        "_token_budget_resolver",
    )

    def __init__(
        self,
        *,
        parent_provider: Callable[[], "QWidget | None"] | None = None,
        start_dir_resolver: Callable[[], Path | None] | None = None,
        token_budget_resolver: Callable[[], int | None] | None = None,
    ) -> None:
        """Initialize the import dialog provider.

        Args:
            parent_provider: Function returning the parent widget for dialogs.
            start_dir_resolver: Function returning the starting directory.
            token_budget_resolver: Function returning the token budget.
        """
        self._parent_provider = parent_provider
        self._start_dir_resolver = start_dir_resolver
        self._token_budget_resolver = token_budget_resolver

    def prompt_import_path(
        self,
        dialog_filter: str | None = None,
    ) -> Path | None:
        """Prompt user to select a file to import.

        Args:
            dialog_filter: File filter string (e.g., "PDF Files (*.pdf)").

        Returns:
            Selected path or None if canceled.
        """
        try:
            from tinkerbell.ui.presentation.widgets.dialogs import open_file_dialog
        except ImportError as exc:  # pragma: no cover
            LOGGER.warning("File dialogs require PySide6: %s", exc)
            return None

        parent = self._parent_provider() if self._parent_provider else None
        start_dir = self._start_dir_resolver() if self._start_dir_resolver else None
        token_budget = (
            self._token_budget_resolver() if self._token_budget_resolver else None
        )

        return open_file_dialog(
            parent=parent,
            caption="Import File",
            start_dir=start_dir,
            file_filter=dialog_filter,
            token_budget=token_budget,
            enable_samples=False,
        )


__all__ = [
    "FileDialogProvider",
    "ImportDialogProvider",
]
