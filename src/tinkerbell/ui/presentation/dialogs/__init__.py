"""Dialog implementations for the presentation layer.

This package contains reusable dialog components that implement
the DialogProvider and ImportDialogProvider protocols from the
application layer.

Dialogs:
    - FileDialogProvider: Open/Save file dialogs
    - ImportDialogProvider: Import file dialogs
    - CommandPaletteDialog: Searchable command palette
    - DocumentStatusWindow: Document status inspection dialog
"""

from __future__ import annotations

from .command_palette import (
    CommandPaletteDialog,
    PaletteCommand,
    build_palette_commands,
)
from .document_status_window import DocumentStatusWindow
from .file_dialogs import (
    FileDialogProvider,
    ImportDialogProvider,
)

__all__: list[str] = [
    "CommandPaletteDialog",
    "DocumentStatusWindow",
    "FileDialogProvider",
    "ImportDialogProvider",
    "PaletteCommand",
    "build_palette_commands",
]
