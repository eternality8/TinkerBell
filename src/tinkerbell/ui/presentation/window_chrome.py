"""Window chrome assembly for the main application window.

This module provides the WindowChrome class which encapsulates the
construction of the main window's visual structure:
- Splitter layout (editor + chat panel)
- Menu bar with actions
- Toolbar setup
- Status bar integration

This is the presentation layer component responsible for assembling
the window "chrome" (menus, toolbars, layout) without business logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

try:
    from .chat.chat_panel import ChatPanel
    from ...editor.tabbed_editor import TabbedEditorWidget
    from .widgets.status_bar import StatusBar
except ImportError:  # pragma: no cover - handle isolated imports
    ChatPanel = Any  # type: ignore[misc,assignment]
    TabbedEditorWidget = Any  # type: ignore[misc,assignment]
    StatusBar = Any  # type: ignore[misc,assignment]

from ..models.actions import MenuSpec, SplitterState, ToolbarSpec, WindowAction


@dataclass(slots=True)
class WindowChromeState:
    """Internal snapshot of declarative UI structures for the main window.

    Attributes:
        splitter: The QSplitter widget (or SplitterState fallback).
        actions: Mapping of action names to WindowAction instances.
        menus: Mapping of menu names to MenuSpec definitions.
        toolbars: Mapping of toolbar names to ToolbarSpec definitions.
        qt_actions: Mapping of action names to QAction instances.
    """

    splitter: Any
    actions: dict[str, WindowAction]
    menus: dict[str, MenuSpec]
    toolbars: dict[str, ToolbarSpec]
    qt_actions: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ActionDefinition:
    """Static metadata describing a high-level menu or toolbar action."""

    name: str
    text: str
    shortcut: str | None
    status_tip: str | None


# Action definitions for menus and toolbars
_ACTION_DEFINITIONS: tuple[_ActionDefinition, ...] = (
    _ActionDefinition(
        name="file_new_tab",
        text="New Tab",
        shortcut="Ctrl+N",
        status_tip="Create a new untitled tab",
    ),
    _ActionDefinition(
        name="file_open",
        text="Open…",
        shortcut="Ctrl+O",
        status_tip="Open a document from disk",
    ),
    _ActionDefinition(
        name="file_import",
        text="Import…",
        shortcut="Ctrl+Shift+I",
        status_tip="Convert PDFs and other formats into editable text",
    ),
    _ActionDefinition(
        name="file_save",
        text="Save",
        shortcut="Ctrl+S",
        status_tip="Save the current document",
    ),
    _ActionDefinition(
        name="file_close_tab",
        text="Close Tab",
        shortcut="Ctrl+W",
        status_tip="Close the active tab",
    ),
    _ActionDefinition(
        name="file_revert",
        text="Revert",
        shortcut=None,
        status_tip="Discard unsaved changes and reload the file from disk",
    ),
    _ActionDefinition(
        name="file_save_as",
        text="Save As…",
        shortcut="Ctrl+Shift+S",
        status_tip="Save the document to a new location",
    ),
    _ActionDefinition(
        name="ai_snapshot",
        text="Refresh Snapshot",
        shortcut="Ctrl+Shift+R",
        status_tip="Capture the latest editor snapshot for the AI agent",
    ),
    _ActionDefinition(
        name="ai_accept_changes",
        text="Accept AI Changes",
        shortcut="Ctrl+Shift+Enter",
        status_tip="Accept all pending AI edits across tabs",
    ),
    _ActionDefinition(
        name="ai_reject_changes",
        text="Reject AI Changes",
        shortcut="Ctrl+Shift+Backspace",
        status_tip="Reject the pending AI turn and restore prior state",
    ),
    _ActionDefinition(
        name="settings_open",
        text="Preferences…",
        shortcut="Ctrl+Comma",
        status_tip="Configure AI and editor preferences",
    ),
    _ActionDefinition(
        name="command_palette",
        text="Command Palette…",
        shortcut="Ctrl+Shift+P",
        status_tip="Search for any command or action",
    ),
    _ActionDefinition(
        name="view_document_status",
        text="Document Status…",
        shortcut="Ctrl+Shift+D",
        status_tip="Inspect document readiness (chunks, outline, telemetry)",
    ),
)


# Default menu structure
_DEFAULT_MENUS: tuple[MenuSpec, ...] = (
    MenuSpec(
        name="file",
        title="&File",
        actions=(
            "file_new_tab",
            "file_open",
            "file_import",
            "file_save",
            "file_save_as",
            "file_close_tab",
            "file_revert",
        ),
    ),
    MenuSpec(
        name="view",
        title="&View",
        actions=("command_palette", "view_document_status"),
    ),
    MenuSpec(
        name="settings",
        title="&Settings",
        actions=("settings_open",),
    ),
    MenuSpec(
        name="ai",
        title="&AI",
        actions=("ai_snapshot", "ai_accept_changes", "ai_reject_changes"),
    ),
)


# Default toolbar structure
_DEFAULT_TOOLBARS: tuple[ToolbarSpec, ...] = (
    ToolbarSpec(
        name="file",
        actions=("file_new_tab", "file_open", "file_save", "file_close_tab"),
    ),
    ToolbarSpec(
        name="ai",
        actions=("ai_snapshot",),
    ),
)


class WindowChrome:
    """Encapsulates splitter/build wiring so MainWindow stays lean.

    This class handles all the "chrome" assembly for the main window:
    - Building the splitter with editor and chat panel
    - Creating WindowAction instances from definitions
    - Installing Qt menus and toolbars
    - Setting up the status bar

    The WindowChrome separates presentation concerns from business logic,
    allowing the MainWindow to focus on coordination rather than UI assembly.

    Example:
        chrome = WindowChrome(
            window=main_window,
            editor=tabbed_editor,
            chat_panel=chat_panel,
            status_bar=status_bar,
            action_callbacks={
                "file_new_tab": lambda: coordinator.new_document(),
                "file_open": lambda: coordinator.open_document(),
                # ... etc
            },
        )
        state = chrome.assemble()
    """

    __slots__ = ("_window", "_editor", "_chat_panel", "_status_bar", "_callbacks")

    def __init__(
        self,
        *,
        window: Any,
        editor: TabbedEditorWidget,
        chat_panel: ChatPanel,
        status_bar: StatusBar,
        action_callbacks: Mapping[str, Callable[[], Any]],
    ) -> None:
        """Initialize the window chrome builder.

        Args:
            window: The QMainWindow instance.
            editor: The tabbed editor widget.
            chat_panel: The chat panel widget.
            status_bar: The status bar widget.
            action_callbacks: Mapping of action names to callback functions.
        """
        self._window = window
        self._editor = editor
        self._chat_panel = chat_panel
        self._status_bar = status_bar
        self._callbacks = action_callbacks

    def assemble(self) -> WindowChromeState:
        """Build splitter, actions, menus, and Qt bindings in one place.

        Returns:
            WindowChromeState containing all assembled UI components.

        Raises:
            KeyError: If a required action callback is missing.
        """
        splitter = self._build_splitter()
        self._window.setCentralWidget(splitter)
        self._apply_status_bar()
        actions = self._create_actions()
        menus = {spec.name: spec for spec in _DEFAULT_MENUS}
        toolbars = {spec.name: spec for spec in _DEFAULT_TOOLBARS}
        qt_actions = self._install_qt_menus(actions, menus)
        return WindowChromeState(
            splitter=splitter,
            actions=actions,
            menus=menus,
            toolbars=toolbars,
            qt_actions=qt_actions,
        )

    def _apply_status_bar(self) -> None:
        """Apply the status bar to the window."""
        qt_status_bar = getattr(self._status_bar, "widget", lambda: None)()
        try:
            self._window.setStatusBar(qt_status_bar or self._status_bar)  # type: ignore[arg-type]
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _build_splitter(self) -> Any:
        """Create the editor/chat splitter with a headless fallback.

        Returns:
            A QSplitter widget or SplitterState fallback for headless mode.
        """
        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication, QSplitter

            if QApplication.instance() is None:
                raise RuntimeError("QApplication must exist before constructing widgets")

            splitter = QSplitter()
            orientation = getattr(Qt, "Horizontal", None)
            if orientation is None:
                orientation = getattr(
                    getattr(Qt, "Orientation", object),
                    "Horizontal",
                    None,
                )
            if orientation is not None:
                try:
                    splitter.setOrientation(orientation)  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - Qt defensive guard
                    pass
            splitter.addWidget(self._editor)  # type: ignore[arg-type]
            splitter.addWidget(self._chat_panel)  # type: ignore[arg-type]
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
            return splitter
        except Exception:
            return SplitterState(editor=self._editor, chat_panel=self._chat_panel)

    def _create_actions(self) -> dict[str, WindowAction]:
        """Create WindowAction instances from definitions.

        Returns:
            Dict mapping action names to WindowAction instances.

        Raises:
            KeyError: If a required callback is missing.
        """
        actions: dict[str, WindowAction] = {}
        for definition in _ACTION_DEFINITIONS:
            callback = self._callbacks.get(definition.name)
            if callback is None:
                raise KeyError(f"Missing callback for action '{definition.name}'")
            actions[definition.name] = WindowAction(
                name=definition.name,
                text=definition.text,
                shortcut=definition.shortcut,
                status_tip=definition.status_tip,
                callback=callback,
            )
        return actions

    def _install_qt_menus(
        self,
        actions: dict[str, WindowAction],
        menus: dict[str, MenuSpec],
    ) -> dict[str, Any]:
        """Install Qt menus and actions on the window.

        Args:
            actions: The WindowAction instances to install.
            menus: The menu specifications to create.

        Returns:
            Dict mapping action names to QAction instances.
        """
        try:
            from PySide6.QtGui import QAction
            from PySide6.QtWidgets import QMenuBar
        except Exception:
            return {}

        menubar_factory = getattr(self._window, "menuBar", None)
        menubar = menubar_factory() if callable(menubar_factory) else None
        if menubar is None:
            menubar = QMenuBar(self._window)
        else:
            try:
                menubar.clear()
            except Exception:
                menubar = QMenuBar(self._window)

        qt_actions: dict[str, Any] = {}
        for action in actions.values():
            qt_action = QAction(action.text, self._window)
            if action.shortcut:
                try:
                    qt_action.setShortcut(action.shortcut)  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - Qt defensive guard
                    pass
            if action.status_tip:
                qt_action.setStatusTip(action.status_tip)
            try:
                qt_action.triggered.connect(action.trigger)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
            qt_actions[action.name] = qt_action

        for menu_spec in menus.values():
            menu = menubar.addMenu(menu_spec.title)
            for action_name in menu_spec.actions:
                qt_action = qt_actions.get(action_name)
                if qt_action is None:
                    continue
                menu.addAction(qt_action)

        setter = getattr(self._window, "setMenuBar", None)
        if callable(setter):
            setter(menubar)

        return qt_actions


__all__ = ["WindowChrome", "WindowChromeState"]
