"""Dialog widgets for file operations and settings."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import httpx  # Re-export for monkeypatching in tests

from PySide6.QtWidgets import QDialog, QWidget

from .common import (
    DEFAULT_FILE_FILTER,
    PREVIEW_CHAR_LIMIT,
)
from .sample_document import (
    SampleDocument,
    discover_sample_documents,
)
from .document_load import (
    DocumentLoadDialog,
)
from .document_export import (
    DocumentExportDialog,
)
from .validation_errors import (
    ValidationErrorsDialog,
    show_validation_errors,
)

# Import SettingsDialog and related from the new settings_dialog module
from .settings_dialog import (
    SettingsDialog,
    SettingsDialogResult,
    ValidationResult,
    SettingsValidator,
    SettingsTester,
    show_settings_dialog,
    test_ai_api_settings,
    test_embedding_settings,
)


def open_file_dialog(
    parent: QWidget | None = None,
    *,
    caption: str = "Open Document",
    start_dir: Path | str | None = None,
    file_filter: str | None = None,
    token_budget: int | None = None,
    enable_samples: bool = True,
    sample_documents: Sequence[SampleDocument] | None = None,
    preview_char_limit: int = PREVIEW_CHAR_LIMIT,
) -> Path | None:
    """Display the enhanced open/import dialog with previews and token budgets."""

    samples: Sequence[SampleDocument] | None = sample_documents
    if enable_samples and samples is None:
        samples = discover_sample_documents()

    dialog = DocumentLoadDialog(
        parent=parent,
        caption=caption,
        start_dir=start_dir,
        file_filter=file_filter,
        token_budget=token_budget,
        sample_documents=samples,
        preview_char_limit=preview_char_limit,
    )
    accepted = dialog.exec() == int(QDialog.DialogCode.Accepted)
    if not accepted:
        return None
    return dialog.selected_path()


def save_file_dialog(
    parent: QWidget | None = None,
    *,
    caption: str = "Save Document",
    start_dir: Path | str | None = None,
    file_filter: str | None = None,
    default_suffix: str = "md",
    document_text: str | None = None,
    token_budget: int | None = None,
    preview_char_limit: int = PREVIEW_CHAR_LIMIT,
) -> Path | None:
    """Display the export/save dialog and return the chosen path."""

    dialog = DocumentExportDialog(
        parent=parent,
        caption=caption,
        start_dir=start_dir,
        file_filter=file_filter,
        default_suffix=default_suffix,
        document_text=document_text,
        token_budget=token_budget,
        preview_char_limit=preview_char_limit,
    )
    accepted = dialog.exec() == int(QDialog.DialogCode.Accepted)
    if not accepted:
        return None
    selected = dialog.selected_path()
    if selected is None:
        return None
    resolved = Path(selected)
    if default_suffix and not resolved.suffix:
        resolved = resolved.with_suffix(f".{default_suffix}")
    return resolved


__all__ = [
    "DEFAULT_FILE_FILTER",
    "SampleDocument",
    "discover_sample_documents",
    "SettingsDialog",
    "SettingsDialogResult",
    "ValidationErrorsDialog",
    "ValidationResult",
    "SettingsValidator",
    "SettingsTester",
    "test_ai_api_settings",
    "test_embedding_settings",
    "open_file_dialog",
    "save_file_dialog",
    "show_settings_dialog",
    "show_validation_errors",
    "DocumentLoadDialog",
    "DocumentExportDialog",
    "httpx",
    "QDialog",
]
