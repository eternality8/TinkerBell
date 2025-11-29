"""Thin presentation layer main window.

This module provides a thin MainWindow shell that:
1. Creates core widgets (editor, chat panel, status bar)
2. Wires WindowChrome with action callbacks
3. Sets up StatusBarUpdater and ChatPanelUpdater
4. Subscribes to events for window title and cursor updates
5. Delegates all operations to AppCoordinator

This is a WS4.5 implementation as part of the UI architecture redesign.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ..application.coordinator import AppCoordinator
    from ..events import EventBus

LOGGER = logging.getLogger(__name__)

# Window constants
WINDOW_APP_NAME = "TinkerBell"
UNTITLED_DOCUMENT_NAME = "Untitled"

# Qt imports with headless fallback
try:  # pragma: no cover - PySide6 optional in CI
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
        QMessageBox,
        QWidget,
    )

    _QT_AVAILABLE = True
except Exception:  # pragma: no cover - runtime stubs keep tests headless
    _QT_AVAILABLE = False
    Qt = None  # type: ignore[assignment,misc]
    QTimer = None  # type: ignore[assignment,misc]
    QApplication = None  # type: ignore[assignment,misc]
    QMessageBox = None  # type: ignore[assignment,misc]
    QWidget = None  # type: ignore[assignment,misc]

    class QMainWindow:  # type: ignore[no-redef]
        """Stub for headless testing."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def setWindowTitle(self, title: str) -> None:
            pass

        def setCentralWidget(self, widget: Any) -> None:
            pass

        def setStatusBar(self, widget: Any) -> None:
            pass

        def menuBar(self) -> Any:
            return None

        def show(self) -> None:
            pass

        def close(self) -> None:
            pass


class ThinMainWindow(QMainWindow):
    """Thin presentation shell for the main application window.

    This window acts as a thin orchestration layer that:
    - Creates and owns core UI widgets
    - Wires action callbacks to AppCoordinator
    - Sets up event subscriptions for reactive updates
    - Delegates all business logic to the coordinator

    The window itself contains no business logic - it simply
    connects the presentation components to the application layer.

    Example:
        event_bus = EventBus()
        coordinator = AppCoordinator(...)

        window = ThinMainWindow(
            event_bus=event_bus,
            coordinator=coordinator,
        )
        window.show()
    """

    def __init__(
        self,
        event_bus: "EventBus",
        coordinator: "AppCoordinator",
        *,
        skip_widgets: bool = False,
    ) -> None:
        """Initialize the main window.

        Args:
            event_bus: The event bus for reactive updates.
            coordinator: The application coordinator for operations.
            skip_widgets: If True, skip widget creation (for testing).
        """
        super().__init__()
        self._event_bus = event_bus
        self._coordinator = coordinator
        self._async_loop: asyncio.AbstractEventLoop | None = None

        # Widgets (created lazily unless skip_widgets)
        self._editor: Any = None
        self._chat_panel: Any = None
        self._status_bar: Any = None
        self._chrome_state: Any = None

        # Updaters
        self._status_bar_updater: Any = None
        self._chat_panel_updater: Any = None

        # State
        self._subscribed = False
        self._last_status_message = ""

        # Initialize
        if not skip_widgets:
            self._create_widgets()
            self._setup_chrome()
            self._setup_updaters()
            self._setup_chat_listeners()
            self._setup_editor_listeners()

        # Always subscribe to events (even without widgets)
        self._subscribe_to_events()

        self._update_window_title()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def editor(self) -> Any:
        """The tabbed editor widget."""
        return self._editor

    @property
    def chat_panel(self) -> Any:
        """The chat panel widget."""
        return self._chat_panel

    @property
    def status_bar(self) -> Any:
        """The status bar widget."""
        return self._status_bar

    @property
    def actions(self) -> dict[str, Any]:
        """The action dictionary from WindowChrome."""
        if self._chrome_state is None:
            return {}
        return self._chrome_state.actions

    @property
    def last_status_message(self) -> str:
        """The last status message displayed."""
        return self._last_status_message

    # ------------------------------------------------------------------
    # Widget Creation
    # ------------------------------------------------------------------

    def _create_widgets(self) -> None:
        """Create the core UI widgets."""
        try:
            from ...chat.chat_panel import ChatPanel
            from ...editor.tabbed_editor import TabbedEditorWidget
            from ...widgets.status_bar import StatusBar

            self._editor = TabbedEditorWidget(skip_default_tab=True)
            self._chat_panel = ChatPanel()
            self._status_bar = StatusBar()

            LOGGER.debug("ThinMainWindow: created widgets")
        except ImportError as e:
            LOGGER.warning("ThinMainWindow: failed to import widgets: %s", e)

    def _setup_chrome(self) -> None:
        """Set up the window chrome (menus, toolbars, layout)."""
        if self._editor is None or self._chat_panel is None:
            return

        from .window_chrome import WindowChrome

        callbacks = self._build_action_callbacks()
        chrome = WindowChrome(
            window=self,
            editor=self._editor,
            chat_panel=self._chat_panel,
            status_bar=self._status_bar,
            action_callbacks=callbacks,
        )
        self._chrome_state = chrome.assemble()

        LOGGER.debug("ThinMainWindow: chrome assembled")

    def _build_action_callbacks(self) -> dict[str, Callable[[], Any]]:
        """Build the action callback mapping for WindowChrome."""
        return {
            "file_new_tab": self._handle_new_tab,
            "file_open": self._handle_open,
            "file_import": self._handle_import,
            "file_save": self._handle_save,
            "file_save_as": self._handle_save_as,
            "file_close_tab": self._handle_close_tab,
            "file_revert": self._handle_revert,
            "ai_snapshot": self._handle_snapshot,
            "ai_accept_changes": self._handle_accept,
            "ai_reject_changes": self._handle_reject,
            "settings_open": self._handle_settings,
            "command_palette": self._handle_command_palette,
            "view_document_status": self._handle_document_status,
        }

    def _setup_updaters(self) -> None:
        """Set up the reactive updaters."""
        if self._status_bar is None or self._chat_panel is None:
            return

        from .status_updaters import ChatPanelUpdater, StatusBarUpdater

        self._status_bar_updater = StatusBarUpdater(
            self._status_bar,
            self._event_bus,
        )
        self._chat_panel_updater = ChatPanelUpdater(
            self._chat_panel,
            self._event_bus,
        )

        LOGGER.debug("ThinMainWindow: updaters created")

    # ------------------------------------------------------------------
    # Event Subscriptions
    # ------------------------------------------------------------------

    def _subscribe_to_events(self) -> None:
        """Subscribe to events for window updates."""
        from ..events import (
            ActiveTabChanged,
            DocumentOpened,
            DocumentSaved,
            StatusMessage,
            WindowTitleChanged,
        )

        self._event_bus.subscribe(WindowTitleChanged, self._on_window_title_changed)
        self._event_bus.subscribe(DocumentOpened, self._on_document_opened)
        self._event_bus.subscribe(DocumentSaved, self._on_document_saved)
        self._event_bus.subscribe(ActiveTabChanged, self._on_active_tab_changed)
        self._event_bus.subscribe(StatusMessage, self._on_status_message)

        self._subscribed = True
        LOGGER.debug("ThinMainWindow: subscribed to events")

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events."""
        if not self._subscribed:
            return

        from ..events import (
            ActiveTabChanged,
            DocumentOpened,
            DocumentSaved,
            StatusMessage,
            WindowTitleChanged,
        )

        self._event_bus.unsubscribe(WindowTitleChanged, self._on_window_title_changed)
        self._event_bus.unsubscribe(DocumentOpened, self._on_document_opened)
        self._event_bus.unsubscribe(DocumentSaved, self._on_document_saved)
        self._event_bus.unsubscribe(ActiveTabChanged, self._on_active_tab_changed)
        self._event_bus.unsubscribe(StatusMessage, self._on_status_message)

        self._subscribed = False
        LOGGER.debug("ThinMainWindow: unsubscribed from events")

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    def _on_window_title_changed(self, event: Any) -> None:
        """Handle WindowTitleChanged events."""
        self.setWindowTitle(event.title)

    def _on_document_opened(self, event: Any) -> None:
        """Handle DocumentOpened events."""
        self._update_window_title()

    def _on_document_saved(self, event: Any) -> None:
        """Handle DocumentSaved events."""
        self._update_window_title()

    def _on_active_tab_changed(self, event: Any) -> None:
        """Handle ActiveTabChanged events."""
        self._update_window_title()

    def _on_status_message(self, event: Any) -> None:
        """Handle StatusMessage events (for last_status_message tracking)."""
        self._last_status_message = event.message

    def _update_window_title(self) -> None:
        """Update the window title based on current state."""
        active_tab_id = self._coordinator.active_tab_id
        if active_tab_id is None:
            self.setWindowTitle(WINDOW_APP_NAME)
            return

        # Try to get document info for title
        title = WINDOW_APP_NAME
        # Note: In the full implementation, we'd query the document store
        # for the active document's path/title. For now, keep it simple.
        self.setWindowTitle(title)

    # ------------------------------------------------------------------
    # Action Handlers (delegate to coordinator)
    # ------------------------------------------------------------------

    def _handle_new_tab(self) -> None:
        """Handle new tab action."""
        self._coordinator.new_document()

    def _handle_open(self) -> None:
        """Handle open file action."""
        self._coordinator.open_document()

    def _handle_import(self) -> None:
        """Handle import file action."""
        self._coordinator.import_document()

    def _handle_save(self) -> None:
        """Handle save action."""
        self._coordinator.save_document()

    def _handle_save_as(self) -> None:
        """Handle save as action."""
        self._coordinator.save_document_as()

    def _handle_close_tab(self) -> None:
        """Handle close tab action."""
        self._coordinator.close_document()

    def _handle_revert(self) -> None:
        """Handle revert action."""
        self._coordinator.revert_document()

    def _handle_snapshot(self) -> None:
        """Handle refresh snapshot action."""
        self._coordinator.refresh_snapshot()

    def _handle_accept(self) -> None:
        """Handle accept changes action."""
        self._coordinator.accept_review()

    def _handle_reject(self) -> None:
        """Handle reject changes action."""
        self._coordinator.reject_review()

    def _handle_settings(self) -> None:
        """Handle settings action."""
        # Settings dialog is not yet part of the new architecture
        # This will be wired in WS5 (Infrastructure)
        LOGGER.debug("ThinMainWindow: settings not yet implemented in new arch")

    def _handle_command_palette(self) -> None:
        """Handle command palette action."""
        # Command palette needs to be wired to the coordinator
        # For now, create and show it directly
        from .dialogs import CommandPaletteDialog, build_palette_commands

        if not hasattr(self, "_command_palette") or self._command_palette is None:
            self._command_palette = CommandPaletteDialog(parent=self)

        if self._chrome_state is not None:
            entries = build_palette_commands(
                self._chrome_state.actions,
                exclude=("command_palette",),
            )
            self._command_palette.set_entries(entries)

        self._command_palette.show()

    def _handle_document_status(self) -> None:
        """Handle document status action."""
        # Document status window needs document info from coordinator
        # This will be fully wired when document status service is integrated
        LOGGER.debug("ThinMainWindow: document status not yet implemented in new arch")

    # ------------------------------------------------------------------
    # Chat Panel Integration (WS6.2)
    # ------------------------------------------------------------------

    def _setup_chat_listeners(self) -> None:
        """Set up chat panel request and session reset listeners."""
        if self._chat_panel is None:
            return

        # Wire request listener: ChatPanel.add_request_listener -> run_ai_turn
        self._chat_panel.add_request_listener(self._on_chat_request)

        # Wire session reset listener: ChatPanel.add_session_reset_listener -> cancel_ai_turn
        self._chat_panel.add_session_reset_listener(self._on_session_reset)

        LOGGER.debug("ThinMainWindow: wired chat panel listeners")

    def _on_chat_request(self, prompt: str, metadata: dict[str, Any]) -> None:
        """Handle chat panel request to run AI turn.

        Args:
            prompt: The user prompt to send to the AI.
            metadata: Metadata from the chat composer context.
        """
        # Capture chat snapshot before AI turn
        chat_snapshot = None
        if self._chat_panel is not None:
            try:
                snapshot = self._chat_panel.capture_state()
                chat_snapshot = {
                    "messages": snapshot.messages,
                    "tool_traces": snapshot.tool_traces,
                    "composer_text": snapshot.composer_text,
                }
            except Exception:
                LOGGER.debug("Failed to capture chat snapshot", exc_info=True)

        # Schedule the async AI turn
        coro = self._coordinator.run_ai_turn(
            prompt,
            metadata,
            chat_snapshot=chat_snapshot,
        )
        self.schedule_coroutine(coro)

    def _on_session_reset(self) -> None:
        """Handle chat panel session reset to cancel AI turn."""
        self._coordinator.cancel_ai_turn()

    # ------------------------------------------------------------------
    # Editor Selection Integration (WS6.2)
    # ------------------------------------------------------------------

    def _setup_editor_listeners(self) -> None:
        """Set up editor selection listeners for status bar updates."""
        if self._editor is None or self._status_bar is None:
            return

        # Wire selection listener: TabbedEditorWidget.add_selection_listener -> status bar
        self._editor.add_selection_listener(self._on_editor_selection_changed)

        LOGGER.debug("ThinMainWindow: wired editor selection listener")

    def _on_editor_selection_changed(
        self,
        tab_id: str,
        selection: Any,
        line: int,
        column: int,
    ) -> None:
        """Handle editor selection changes to update status bar.

        Args:
            tab_id: The ID of the tab where selection changed.
            selection: The selection range.
            line: The current line number (1-indexed).
            column: The current column number (1-indexed).
        """
        if self._status_bar is None:
            return

        # Update cursor position in status bar
        try:
            # StatusBar should have a method like set_cursor_position
            setter = getattr(self._status_bar, "set_cursor_position", None)
            if callable(setter):
                setter(line, column)
            else:
                # Fallback: try set_position_label
                label_setter = getattr(self._status_bar, "set_position_label", None)
                if callable(label_setter):
                    label_setter(f"Ln {line}, Col {column}")
        except Exception:
            LOGGER.debug("Failed to update cursor position", exc_info=True)

    # ------------------------------------------------------------------
    # Async Support
    # ------------------------------------------------------------------

    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create the async event loop."""
        if self._async_loop is None:
            try:
                self._async_loop = asyncio.get_running_loop()
            except RuntimeError:
                self._async_loop = asyncio.get_event_loop()
        return self._async_loop

    def schedule_coroutine(self, coro: Any) -> Any:
        """Schedule a coroutine to run in the event loop.

        Args:
            coro: The coroutine to schedule.

        Returns:
            A Future representing the scheduled coroutine.
        """
        loop = self._get_event_loop()
        return asyncio.ensure_future(coro, loop=loop)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event: Any) -> None:
        """Handle window close event."""
        LOGGER.debug("ThinMainWindow: close event")

        # Dispose updaters
        if self._status_bar_updater is not None:
            self._status_bar_updater.dispose()
        if self._chat_panel_updater is not None:
            self._chat_panel_updater.dispose()

        # Unsubscribe from events
        self._unsubscribe_from_events()

        # Request application quit
        if _QT_AVAILABLE and QApplication is not None:
            app = QApplication.instance()
            if app is not None and not app.closingDown():
                app.quit()

        # Accept the close event
        if hasattr(event, "accept"):
            event.accept()

    def dispose(self) -> None:
        """Clean up resources.

        Call this method when the window is being destroyed
        to ensure proper cleanup of subscriptions and resources.
        """
        # Remove chat panel listeners
        if self._chat_panel is not None:
            try:
                self._chat_panel.remove_request_listener(self._on_chat_request)
                self._chat_panel.remove_session_reset_listener(self._on_session_reset)
            except Exception:
                LOGGER.debug("Failed to remove chat panel listeners", exc_info=True)

        if self._status_bar_updater is not None:
            self._status_bar_updater.dispose()
            self._status_bar_updater = None

        if self._chat_panel_updater is not None:
            self._chat_panel_updater.dispose()
            self._chat_panel_updater = None

        self._unsubscribe_from_events()

        LOGGER.debug("ThinMainWindow: disposed")


__all__ = ["ThinMainWindow", "WINDOW_APP_NAME", "UNTITLED_DOCUMENT_NAME"]
