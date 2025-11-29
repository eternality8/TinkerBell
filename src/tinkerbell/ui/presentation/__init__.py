"""Presentation layer for UI architecture.

This package contains thin UI components that respond to domain events
and delegate user actions to the application layer. The presentation
layer is responsible for:

1. **Updaters**: Components that subscribe to events and update UI widgets
   - StatusBarUpdater: Updates status bar based on domain events
   - ChatPanelUpdater: Updates chat panel based on AI events

2. **Dialogs**: Reusable dialog implementations
   - File dialogs (open, save, import)
   - Command palette
   - Document status window

3. **Window Chrome**: Top-level window management
   - Window title updates
   - Menu bar management
   - Keyboard shortcuts

4. **MainWindow**: Thin orchestration shell
   - Widget creation and ownership
   - Event subscription and dispatch
   - Action callback wiring

Design Principles:
    - Thin components that delegate to AppCoordinator
    - Subscribe to events for reactive updates
    - No business logic - only UI updates
    - Testable in isolation with mock event bus
"""

from __future__ import annotations

from .status_updaters import (
    StatusBarProtocol,
    ChatPanelProtocol,
    StatusBarUpdater,
    ChatPanelUpdater,
)
from .window_chrome import WindowChrome, WindowChromeState
from .main_window import ThinMainWindow, WINDOW_APP_NAME, UNTITLED_DOCUMENT_NAME
from .dialogs import (
    CommandPaletteDialog,
    DocumentStatusWindow,
    FileDialogProvider,
    ImportDialogProvider,
    PaletteCommand,
    build_palette_commands,
)

__all__: list[str] = [
    # Protocols
    "StatusBarProtocol",
    "ChatPanelProtocol",
    # Updaters
    "StatusBarUpdater",
    "ChatPanelUpdater",
    # Window Chrome
    "WindowChrome",
    "WindowChromeState",
    # Main Window
    "ThinMainWindow",
    "WINDOW_APP_NAME",
    "UNTITLED_DOCUMENT_NAME",
    # Dialogs
    "CommandPaletteDialog",
    "DocumentStatusWindow",
    "FileDialogProvider",
    "ImportDialogProvider",
    "PaletteCommand",
    "build_palette_commands",
]
