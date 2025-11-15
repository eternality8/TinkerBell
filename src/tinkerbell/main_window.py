"""Main window implementation coordinating the editor and chat panes."""

from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Iterable, Mapping, Optional, TYPE_CHECKING

from .chat.chat_panel import ChatPanel
from .chat.commands import DIRECTIVE_SCHEMA
from .chat.message_model import ChatMessage, EditDirective, ToolTrace
from .editor.document_model import DocumentMetadata, DocumentState, SelectionRange
from .editor.editor_widget import EditorWidget
from .services.bridge import DocumentBridge
from .services.settings import Settings, SettingsStore
from .utils import file_io, logging as logging_utils
from .widgets.status_bar import StatusBar
from .ai.tools.document_snapshot import DocumentSnapshotTool
from .ai.tools.document_edit import DocumentEditTool
from .ai.tools.validation import validate_snippet

if TYPE_CHECKING:  # pragma: no cover - import only for static analysis
    from .ai.agents.executor import AIController
    from .ai.client import AIStreamEvent
    from .widgets.dialogs import SettingsDialogResult

QMainWindow: Any
QWidget: Any

try:  # pragma: no cover - PySide6 optional in CI
    from PySide6.QtWidgets import QMainWindow as _QtQMainWindow, QWidget as _QtQWidget
    QMainWindow = _QtQMainWindow
    QWidget = _QtQWidget
except Exception:  # pragma: no cover - runtime stubs keep tests headless

    class _StubQMainWindow:  # type: ignore
        """Fallback placeholder when PySide6 is unavailable."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            self._central_widget: Any = None
            self._status_bar: Any = None
            self._window_title: str = ""
            self._shown: bool = False

        def setCentralWidget(self, widget: Any) -> None:  # noqa: N802 - Qt API
            self._central_widget = widget

        def centralWidget(self) -> Any:  # noqa: N802 - Qt API
            return self._central_widget

        def setStatusBar(self, status_bar: Any) -> None:  # noqa: N802
            self._status_bar = status_bar

        def statusBar(self) -> Any:  # noqa: N802
            return self._status_bar

        def setWindowTitle(self, title: str) -> None:  # noqa: N802
            self._window_title = title

        def windowTitle(self) -> str:  # noqa: N802
            return self._window_title

        def show(self) -> None:
            self._shown = True


    class _StubQWidget:  # type: ignore
        """Fallback placeholder when PySide6 is unavailable."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs


    QMainWindow = _StubQMainWindow
    QWidget = _StubQWidget


_LOGGER = logging.getLogger(__name__)


WINDOW_APP_NAME = "TinkerBell"
UNTITLED_DOCUMENT_NAME = "Untitled"


@dataclass(slots=True)
class WindowAction:
    """Represents a high-level action exposed through menus/toolbars."""

    name: str
    text: str
    shortcut: Optional[str] = None
    status_tip: Optional[str] = None
    callback: Optional[Callable[[], Any]] = None

    def trigger(self) -> None:
        """Invoke the registered callback, if available."""

        if self.callback is not None:
            self.callback()


@dataclass(slots=True)
class MenuSpec:
    """Declarative menu definition used for headless + Qt builds."""

    name: str
    title: str
    actions: tuple[str, ...]


@dataclass(slots=True)
class ToolbarSpec:
    """Declarative toolbar definition mirroring the plan.md contract."""

    name: str
    actions: tuple[str, ...]


@dataclass(slots=True)
class SplitterState:
    """Simple structure describing the editor/chat splitter layout."""

    editor: Any
    chat_panel: Any
    orientation: str = "horizontal"
    stretch_factors: tuple[int, int] = (3, 2)


@dataclass(slots=True)
class WindowContext:
    """Shared context passed to the main window when constructing the UI."""

    settings: Optional[Settings] = None
    ai_controller: Optional["AIController"] = None
    settings_store: Optional[SettingsStore] = None


class MainWindow(QMainWindow):
    """Primary application window hosting the editor and chat splitter."""

    _WINDOW_BASE_TITLE = WINDOW_APP_NAME
    _UNTITLED_SNAPSHOT_KEY = "__untitled__"

    def __init__(self, context: WindowContext):  # noqa: D401 - doc inherited
        super().__init__()
        self._context = context
        initial_settings = context.settings
        show_tool_panel = bool(getattr(initial_settings, "show_tool_activity_panel", False))
        self._editor = EditorWidget()
        self._chat_panel = ChatPanel(show_tool_activity_panel=show_tool_panel)
        self._bridge = DocumentBridge(editor=self._editor)
        self._status_bar = StatusBar()
        self._splitter: Any = None
        self._actions: Dict[str, WindowAction] = {}
        self._menus: Dict[str, MenuSpec] = {}
        self._toolbars: Dict[str, ToolbarSpec] = {}
        self._qt_actions: Dict[str, Any] = {}
        self._last_snapshot: dict[str, Any] = {}
        self._last_status_message: str = ""
        self._current_document_path: Optional[Path] = None
        self._ai_task: asyncio.Task[Any] | None = None
        self._ai_stream_active = False
        self._unsaved_snapshot_digests: dict[str, str] = {}
        self._snapshot_persistence_block = 0
        self._debug_logging_enabled = bool(getattr(initial_settings, "debug_logging", False))
        self._active_theme = (getattr(initial_settings, "theme", "") or "default").strip() or "default"
        self._ai_client_signature: tuple[Any, ...] | None = self._ai_settings_signature(initial_settings)
        self._initialize_ui()

    # ------------------------------------------------------------------
    # UI setup helpers
    # ------------------------------------------------------------------
    def _initialize_ui(self) -> None:
        """Set up menus, toolbars, splitter layout, and status widgets."""

        self._refresh_window_title()
        self._splitter = self._build_splitter()
        self.setCentralWidget(self._splitter)
        qt_status_bar = getattr(self._status_bar, "widget", lambda: None)()
        self.setStatusBar(qt_status_bar or self._status_bar)  # type: ignore[arg-type]

        self._actions = self._create_actions()
        self._menus = self._create_menus()
        self._toolbars = self._create_toolbars()
        self._install_qt_menus()
        self._wire_signals()
        self._register_default_ai_tools()

        self.update_status("Ready")
        self._restore_last_session_document()

    def _build_splitter(self) -> Any:
        """Create the editor/chat splitter, falling back to a lightweight state."""

        try:
            from PySide6.QtCore import Qt
            from PySide6.QtWidgets import QApplication, QSplitter

            if QApplication.instance() is None:
                raise RuntimeError("QApplication must exist before constructing widgets")

            splitter = QSplitter()
            orientation = getattr(Qt, "Horizontal", None)
            if orientation is None:
                orientation = getattr(getattr(Qt, "Orientation", object), "Horizontal", None)
            if orientation is not None:
                try:
                    splitter.setOrientation(orientation)  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - defensive fallback
                    pass
            splitter.addWidget(self._editor)  # type: ignore[arg-type]
            splitter.addWidget(self._chat_panel)  # type: ignore[arg-type]
            splitter.setStretchFactor(0, 3)
            splitter.setStretchFactor(1, 2)
            return splitter
        except Exception:  # pragma: no cover - executed in headless tests
            return SplitterState(editor=self._editor, chat_panel=self._chat_panel)

    def _create_actions(self) -> Dict[str, WindowAction]:
        """Instantiate all menu/toolbar actions defined in the plan."""

        return {
            "file_open": self._build_action(
                name="file_open",
                text="Open…",
                shortcut="Ctrl+O",
                status_tip="Open a document from disk",
                callback=self._handle_open_requested,
            ),
            "file_save": self._build_action(
                name="file_save",
                text="Save",
                shortcut="Ctrl+S",
                status_tip="Save the current document",
                callback=self._handle_save_requested,
            ),
            "file_save_as": self._build_action(
                name="file_save_as",
                text="Save As…",
                shortcut="Ctrl+Shift+S",
                status_tip="Save the document to a new location",
                callback=self._handle_save_as_requested,
            ),
            "ai_snapshot": self._build_action(
                name="ai_snapshot",
                text="Refresh Snapshot",
                shortcut="Ctrl+Shift+R",
                status_tip="Capture the latest editor snapshot for the AI agent",
                callback=self._handle_snapshot_requested,
            ),
            "settings_open": self._build_action(
                name="settings_open",
                text="Preferences…",
                shortcut="Ctrl+Comma",
                status_tip="Configure AI and editor preferences",
                callback=self._handle_settings_requested,
            ),
        }

    def _create_menus(self) -> Dict[str, MenuSpec]:
        """Return declarative menu metadata used by future Qt wiring."""

        return {
            "file": MenuSpec(
                name="file",
                title="&File",
                actions=("file_open", "file_save", "file_save_as"),
            ),
            "settings": MenuSpec(name="settings", title="&Settings", actions=("settings_open",)),
            "ai": MenuSpec(name="ai", title="&AI", actions=("ai_snapshot",)),
        }

    def _create_toolbars(self) -> Dict[str, ToolbarSpec]:
        """Return declarative toolbar metadata used by future Qt wiring."""

        return {
            "file": ToolbarSpec(name="file", actions=("file_open", "file_save")),
            "ai": ToolbarSpec(name="ai", actions=("ai_snapshot",)),
        }

    def _install_qt_menus(self) -> None:
        """Create a QMenuBar with bound Qt actions when PySide6 is available."""

        try:
            from PySide6.QtGui import QAction
            from PySide6.QtWidgets import QMenuBar
        except Exception:  # pragma: no cover - PySide6 optional for tests
            self._qt_actions.clear()
            return

        menubar: Any
        menubar_factory = getattr(self, "menuBar", None)
        menubar = menubar_factory() if callable(menubar_factory) else None
        if menubar is None:
            menubar = QMenuBar(self)
        else:
            try:
                menubar.clear()
            except Exception:  # pragma: no cover - extremely defensive
                menubar = QMenuBar(self)

        self._qt_actions.clear()
        for action in self._actions.values():
            qt_action = QAction(action.text, self)
            if action.shortcut:
                try:
                    qt_action.setShortcut(action.shortcut)  # type: ignore[arg-type]
                except Exception:  # pragma: no cover - shortcut parsing varies
                    pass
            if action.status_tip:
                qt_action.setStatusTip(action.status_tip)
            try:
                qt_action.triggered.connect(action.trigger)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - Qt stubs during tests
                pass
            self._qt_actions[action.name] = qt_action

        for menu_spec in self.menu_specs():
            menu = menubar.addMenu(menu_spec.title)
            for action_name in menu_spec.actions:
                qt_action = self._qt_actions.get(action_name)
                if qt_action is None:
                    continue
                menu.addAction(qt_action)

        setter = getattr(self, "setMenuBar", None)
        if callable(setter):
            setter(menubar)

    def _build_action(
        self,
        *,
        name: str,
        text: str,
        shortcut: Optional[str],
        status_tip: Optional[str],
        callback: Optional[Callable[[], Any]],
    ) -> WindowAction:
        """Create an action descriptor and register it inside the window."""

        return WindowAction(
            name=name,
            text=text,
            shortcut=shortcut,
            status_tip=status_tip,
            callback=callback,
        )

    def _wire_signals(self) -> None:
        """Connect editor/chat events required for AI coordination."""

        self._editor.add_snapshot_listener(self._handle_editor_snapshot)
        self._editor.add_text_listener(self._handle_editor_text_changed)
        self._editor.add_selection_listener(self._handle_editor_selection_changed)
        self._chat_panel.add_request_listener(self._handle_chat_request)
        self._chat_panel.add_session_reset_listener(self._handle_chat_session_reset)
        self._bridge.add_edit_listener(self._handle_edit_applied)
        self._handle_editor_selection_changed(self._editor.to_document().selection)

    def _register_default_ai_tools(self) -> None:
        """Register the default document-aware tools with the AI controller."""

        controller = self._context.ai_controller
        if controller is None:
            return

        register = getattr(controller, "register_tool", None)
        if not callable(register):
            _LOGGER.debug("AI controller does not expose register_tool; skipping tool wiring.")
            return

        try:
            snapshot_tool = DocumentSnapshotTool(provider=self._bridge)
            register(
                "document_snapshot",
                snapshot_tool,
                description=(
                    "Return the latest editor snapshot including text, selection, metadata, and diff markers."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "delta_only": {
                            "type": "boolean",
                            "description": "Return only fields that changed since the previous snapshot.",
                        }
                    },
                    "additionalProperties": False,
                },
            )

            edit_tool = DocumentEditTool(bridge=self._bridge)
            register(
                "document_edit",
                edit_tool,
                description=(
                    "Apply a structured edit directive (insert, replace, annotate) against the active document."
                ),
                parameters=self._directive_parameters_schema(),
            )

            register(
                "validate_snippet",
                validate_snippet,
                description="Validate YAML/JSON snippets before inserting them into the document.",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Snippet contents that should be validated.",
                        },
                        "fmt": {
                            "type": "string",
                            "description": "Declared format of the snippet.",
                            "enum": ["yaml", "yml", "json"],
                        },
                    },
                    "required": ["text", "fmt"],
                    "additionalProperties": False,
                },
            )

            _LOGGER.debug("Default AI tools registered: document_snapshot, document_edit, validate_snippet")
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.warning("Failed to register default AI tools: %s", exc)

    @staticmethod
    def _directive_parameters_schema() -> Dict[str, Any]:
        """Return a copy of the directive schema used by the document edit tool."""

        schema = deepcopy(DIRECTIVE_SCHEMA)
        schema.setdefault(
            "description",
            "Structured edit directive containing action, content, optional rationale, and target range.",
        )
        return schema

    # ------------------------------------------------------------------
    # AI coordination
    # ------------------------------------------------------------------
    def _handle_chat_request(self, prompt: str, metadata: dict[str, Any]) -> None:
        controller = self._context.ai_controller
        if controller is None:
            self._post_assistant_notice(
                "AI assistant is unavailable. Open Settings to configure your API key and model."
            )
            self.update_status("AI unavailable")
            return

        if self._ai_task and not self._ai_task.done():
            self._cancel_active_ai_turn()
            self.update_status("Previous AI request canceled")

        snapshot = self._bridge.generate_snapshot()
        task = self._run_coroutine(
            self._run_ai_turn(controller, prompt, snapshot, metadata or {})
        )
        self._ai_task = task

    async def _run_ai_turn(
        self,
        controller: "AIController",
        prompt: str,
        snapshot: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> None:
        self._ai_stream_active = False
        normalized_metadata = self._normalize_metadata(metadata)
        self.update_status("AI thinking…")

        try:
            result = await controller.run_chat(
                prompt,
                snapshot,
                metadata=normalized_metadata,
                on_event=self._handle_ai_stream_event,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self._handle_ai_failure(exc)
            return

        response_text = (result or {}).get("response", "").strip()
        if not response_text:
            response_text = "The AI did not return any content."
        self._finalize_ai_response(response_text)
        self.update_status("AI response ready")

    async def _handle_ai_stream_event(self, event: "AIStreamEvent") -> None:
        self._process_stream_event(event)

    def _process_stream_event(self, event: "AIStreamEvent") -> None:
        event_type = getattr(event, "type", "") or ""
        if event_type in {"content.delta", "content.done", "refusal.delta", "refusal.done"}:
            chunk = getattr(event, "content", None)
            if chunk:
                self._chat_panel.append_ai_message(
                    ChatMessage(role="assistant", content=str(chunk)),
                    streaming=True,
                )
                self._ai_stream_active = True
            return

        if event_type.startswith("tool_calls") and getattr(event, "tool_name", None):
            arguments = (getattr(event, "arguments_delta", None) or getattr(event, "tool_arguments", None) or "").strip()
            if not arguments:
                arguments = "(no arguments)"
            trace = ToolTrace(
                name=str(getattr(event, "tool_name", "tool")),
                input_summary=arguments[:120],
                output_summary=(getattr(event, "tool_arguments", None) or arguments)[:120],
            )
            self._chat_panel.show_tool_trace(trace)

    def _finalize_ai_response(self, content: str) -> None:
        if self._ai_stream_active:
            self._chat_panel.append_ai_message(ChatMessage(role="assistant", content=""), streaming=False)
            self._ai_stream_active = False
        else:
            self._chat_panel.append_ai_message(ChatMessage(role="assistant", content=content))

    def _handle_ai_failure(self, exc: Exception) -> None:
        _LOGGER.error("AI request failed: %s", exc)
        self._ai_stream_active = False
        self._post_assistant_notice(f"AI request failed: {exc}")
        self.update_status("AI error")

    def _post_assistant_notice(self, message: str) -> None:
        message = message.strip()
        if not message:
            return
        self._chat_panel.append_ai_message(ChatMessage(role="assistant", content=message))

    def _run_coroutine(self, coro: Coroutine[Any, Any, None]) -> asyncio.Task[Any] | None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
            return None

        task = loop.create_task(coro)
        task.add_done_callback(self._on_ai_task_finished)
        return task

    def _cancel_active_ai_turn(self) -> None:
        controller = self._context.ai_controller
        cancel = getattr(controller, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:  # pragma: no cover - defensive logging
                _LOGGER.debug("AI controller cancel raised", exc_info=True)

        task = self._ai_task
        if task is not None and not task.done():
            try:
                task.cancel()
            except Exception:  # pragma: no cover - defensive logging
                _LOGGER.debug("AI task cancellation failed", exc_info=True)
        self._ai_task = None
        self._ai_stream_active = False

    def _handle_chat_session_reset(self) -> None:
        self._cancel_active_ai_turn()
        self.update_status("Chat reset")

    def _on_ai_task_finished(self, task: asyncio.Task[Any]) -> None:
        if task is self._ai_task:
            self._ai_task = None
        try:
            task.result()
        except Exception as exc:  # pragma: no cover - surfaced via handler
            self._handle_ai_failure(exc)

    def _normalize_metadata(self, metadata: Mapping[str, Any] | None) -> Dict[str, str] | None:
        if not metadata:
            return None
        normalized: Dict[str, str] = {}
        for key, value in metadata.items():
            normalized[str(key)] = str(value)
        return normalized or None

    def _handle_edit_applied(self, directive: EditDirective, _state: DocumentState, diff: str) -> None:
        summary = f"Applied {directive.action} ({diff})"
        self.update_status(summary)
        trace = ToolTrace(
            name=f"edit:{directive.action}",
            input_summary=f"range={directive.target_range}",
            output_summary=diff,
        )
        self._chat_panel.show_tool_trace(trace)

    # ------------------------------------------------------------------
    # Action callbacks
    # ------------------------------------------------------------------
    def _handle_editor_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Cache the latest snapshot for future agent requests."""

        self._last_snapshot = snapshot

    def _handle_editor_text_changed(self, text: str, state: DocumentState) -> None:
        """Update window title when the editor content or metadata shifts."""

        self._refresh_window_title(state)
        self._refresh_chat_suggestions(state=state)
        if self._snapshot_persistence_block > 0:
            return
        self._persist_unsaved_snapshot(state)

    def _handle_editor_selection_changed(self, selection: SelectionRange) -> None:
        """Refresh suggestions and composer context when the selection moves."""

        self._refresh_chat_suggestions(selection=selection)

    def _refresh_chat_suggestions(
        self,
        *,
        state: DocumentState | None = None,
        selection: SelectionRange | None = None,
    ) -> None:
        document = state or self._editor.to_document()
        active_selection = selection or document.selection
        start, end = active_selection.as_tuple()
        text = document.text[start:end]
        summary = self._summarize_selection_text(text)
        self._chat_panel.set_selection_summary(summary)
        suggestions = self._build_chat_suggestions(document, text)
        self._chat_panel.set_suggestions(suggestions)

    def _build_chat_suggestions(self, document: DocumentState, selection_text: str) -> list[str]:
        has_selection = bool(selection_text.strip())
        has_document_text = bool(document.text.strip())
        if has_selection:
            suggestions = [
                "Summarize the selected text.",
                "Rewrite the selected text for clarity.",
                "Extract action items from the selection.",
            ]
        elif has_document_text:
            suggestions = [
                "Summarize the current document.",
                "Suggest improvements to the document structure.",
                "Highlight inconsistencies or missing sections.",
            ]
        else:
            name = self._document_display_name(document)
            suggestions = [
                f"Draft an outline for {name}.",
                "Propose a starter paragraph for this document.",
                "List the key points this document should cover.",
            ]
        suggestions.append("Help me plan the next edits.")
        return suggestions

    def _document_display_name(self, document: DocumentState) -> str:
        path = document.metadata.path or self._current_document_path
        if path is not None:
            name = path.name
            return name if name else WINDOW_APP_NAME
        return UNTITLED_DOCUMENT_NAME

    def _summarize_selection_text(self, selection_text: str) -> Optional[str]:
        condensed = self._condense_whitespace(selection_text)
        if not condensed:
            return None
        if len(condensed) > 80:
            condensed = f"{condensed[:77].rstrip()}…"
        return condensed

    @staticmethod
    def _condense_whitespace(text: str) -> str:
        return " ".join(text.split())

    def _handle_snapshot_requested(self) -> None:
        """Force a snapshot refresh and log the event."""

        snapshot = self._editor.request_snapshot()
        self._last_snapshot = snapshot
        _LOGGER.debug("Snapshot refreshed: chars=%s", len(snapshot.get("text", "")))
        self.update_status("Snapshot refreshed")

    def _handle_open_requested(self) -> None:
        """Prompt the user for a document path and load it into the editor."""

        path = self._prompt_for_open_path()
        if path is None:
            self.update_status("Open canceled")
            return

        try:
            self.open_document(path)
        except FileNotFoundError:
            self.update_status(f"File not found: {path}")
            raise

    def _handle_save_requested(self) -> None:
        """Save the current document, prompting for a path if required."""

        try:
            self.save_document()
        except RuntimeError:
            # Status updates are handled inside save_document / prompt helpers.
            _LOGGER.debug("Save request aborted; no path selected.")

    def _handle_save_as_requested(self) -> None:
        """Trigger a Save As workflow backed by the standard dialog."""

        path = self._prompt_for_save_path()
        if path is None:
            self.update_status("Save canceled")
            return
        try:
            self.save_document(path)
        except RuntimeError:
            _LOGGER.debug("Save As aborted for path: %s", path)

    def _handle_settings_requested(self) -> None:
        """Show the settings dialog so users can configure integrations."""

        settings = self._context.settings or Settings()
        self._context.settings = settings

        try:
            result = self._show_settings_dialog(settings)
        except RuntimeError as exc:
            _LOGGER.warning("Settings dialog unavailable: %s", exc)
            self.update_status("Settings dialog unavailable")
            return

        if not getattr(result, "accepted", False):
            self.update_status("Settings unchanged")
            return

        self._context.settings = result.settings
        self._apply_runtime_settings(result.settings)
        self._persist_settings(result.settings)
        self.update_status("Settings updated")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def editor_widget(self) -> EditorWidget:
        """Expose the editor widget for tests and auxiliary services."""

        return self._editor

    @property
    def chat_panel(self) -> ChatPanel:
        """Return the chat panel instance."""

        return self._chat_panel

    @property
    def actions(self) -> Dict[str, WindowAction]:
        """Return the registered window actions keyed by identifier."""

        return dict(self._actions)

    @property
    def last_status_message(self) -> str:
        """Return the most recent status message emitted by the window."""

        return self._last_status_message

    def open_document(self, path: str | Path) -> None:
        """Open the provided document path inside the editor widget."""

        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(target)

        text = file_io.read_text(target)
        metadata = DocumentMetadata(path=target, language=self._infer_language(target))
        document = DocumentState(text=text, metadata=metadata)
        document.dirty = False

        self._snapshot_persistence_block += 1
        try:
            self._editor.load_document(document)
        finally:
            self._snapshot_persistence_block = max(0, self._snapshot_persistence_block - 1)
        self._current_document_path = target
        self._remember_recent_file(target)
        if self._apply_pending_snapshot_for_path(target):
            return
        self.update_status(f"Loaded {target.name}")

    def save_document(self, path: str | Path | None = None) -> Path:
        """Persist the current document to disk and return the saved path."""

        document = self._editor.to_document()
        previous_path = document.metadata.path or self._current_document_path
        target_path: Path | None
        if path is not None:
            target_path = Path(path)
        else:
            target_path = document.metadata.path or self._current_document_path

        if target_path is None:
            target_path = self._prompt_for_save_path()
            if target_path is None:
                self.update_status("Save canceled")
                raise RuntimeError("Save canceled")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        file_io.write_text(target_path, document.text)
        document.metadata.path = target_path
        document.dirty = False
        self._current_document_path = target_path
        self._remember_recent_file(target_path)
        self._clear_unsaved_snapshot(path=target_path)
        if previous_path is None:
            self._clear_unsaved_snapshot(path=None)
        self.update_status(f"Saved {target_path.name}")
        self._refresh_window_title(document)
        return target_path

    def update_status(self, message: str, *, timeout_ms: Optional[int] = None) -> None:
        """Update the window status bar and keep local bookkeeping."""

        self._last_status_message = message

        try:
            self._status_bar.set_message(message, timeout_ms=timeout_ms)
        except Exception:  # pragma: no cover - defensive logging
            pass

        _LOGGER.debug("Status: %s", message)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _infer_language(self, path: Path) -> str:
        """Infer a simple language identifier from the file suffix."""

        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return "markdown"
        if suffix in {".yaml", ".yml"}:
            return "yaml"
        if suffix == ".json":
            return "json"
        if suffix in {".txt", ""}:
            return "text"
        return "plain"

    def _prompt_for_open_path(self) -> Path | None:
        """Show the open-file dialog and return the selected path."""

        start_dir = self._resolve_open_start_dir(self._context.settings)
        try:
            from .widgets.dialogs import open_file_dialog
        except Exception as exc:  # pragma: no cover - depends on Qt availability
            raise RuntimeError(
                "File dialogs require the optional PySide6 dependency."
            ) from exc

        parent = self._qt_parent_widget()
        return open_file_dialog(parent=parent, start_dir=start_dir)

    def _prompt_for_save_path(self) -> Path | None:
        """Show the save-file dialog and return the chosen path."""

        start_dir = self._resolve_save_start_dir(self._context.settings)
        try:
            from .widgets.dialogs import save_file_dialog
        except Exception as exc:  # pragma: no cover - depends on Qt availability
            raise RuntimeError(
                "File dialogs require the optional PySide6 dependency."
            ) from exc

        parent = self._qt_parent_widget()
        return save_file_dialog(parent=parent, start_dir=start_dir)

    def _resolve_open_start_dir(self, settings: Optional[Settings]) -> Path | None:
        if self._current_document_path is not None:
            parent = self._current_document_path.parent
            if parent.exists():
                return parent
        if settings:
            for entry in settings.recent_files:
                candidate = Path(entry).expanduser()
                if candidate.is_dir():
                    return candidate
                if candidate.exists():
                    return candidate.parent
        return Path.home()

    def _resolve_save_start_dir(self, settings: Optional[Settings]) -> Path | None:
        return self._resolve_open_start_dir(settings)

    def _remember_recent_file(self, path: Path) -> None:
        settings = self._context.settings
        if settings is None:
            return

        normalized = str(path.expanduser().resolve())
        updated: list[str] = [normalized]
        for existing in settings.recent_files:
            existing_normalized = str(Path(existing).expanduser().resolve())
            if existing_normalized == normalized:
                continue
            updated.append(existing)
            if len(updated) >= 10:
                break
        settings.recent_files = updated
        settings.last_open_file = normalized
        self._persist_settings(settings)

    def _restore_last_session_document(self) -> None:
        settings = self._context.settings
        if settings is None:
            return

        restored_file = self._try_restore_last_file(settings)
        if restored_file:
            return
        self._restore_unsaved_snapshot(settings)

    def _try_restore_last_file(self, settings: Settings) -> bool:
        last_path = (settings.last_open_file or "").strip()
        if not last_path:
            return False

        target = Path(last_path).expanduser()
        if not target.exists() or target.is_dir():
            self._handle_missing_last_file(settings)
            return False

        try:
            self.open_document(target)
            return True
        except FileNotFoundError:
            self._handle_missing_last_file(settings)
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.warning("Failed to restore last-open file %s: %s", target, exc)
            self.update_status("Failed to restore last session file")
        return False

    def _restore_unsaved_snapshot(self, settings: Settings) -> None:
        snapshot = settings.unsaved_snapshot
        if not isinstance(snapshot, dict) or "text" not in snapshot:
            return

        self._load_snapshot_document(snapshot, path=None)
        self.update_status("Restored unsaved draft")

    def _apply_pending_snapshot_for_path(self, path: Path) -> bool:
        restored = self._restore_unsaved_snapshot_for_path(path)
        if restored:
            self.update_status(f"Restored unsaved changes for {path.name}")
        return restored

    def _restore_unsaved_snapshot_for_path(self, path: Path | str) -> bool:
        settings = self._context.settings
        if settings is None:
            return False

        normalized_key = self._snapshot_key(path)
        snapshot = (settings.unsaved_snapshots or {}).get(normalized_key)
        if not isinstance(snapshot, dict) or "text" not in snapshot:
            return False

        resolved_path = Path(normalized_key)
        self._load_snapshot_document(snapshot, path=resolved_path)
        return True

    def _load_snapshot_document(self, snapshot: dict[str, Any], *, path: Path | None) -> None:
        text = str(snapshot.get("text", ""))
        language = str(
            snapshot.get("language")
            or (self._infer_language(path) if path is not None else "markdown")
        )
        selection_raw = snapshot.get("selection")
        if isinstance(selection_raw, (tuple, list)) and len(selection_raw) == 2:
            selection = SelectionRange(int(selection_raw[0]), int(selection_raw[1]))
        else:
            selection = SelectionRange()

        document = DocumentState(
            text=text,
            metadata=DocumentMetadata(path=path, language=language),
            selection=selection,
            dirty=True,
        )
        self._editor.load_document(document)
        self._current_document_path = path
        digest_key = self._snapshot_key(path)
        self._unsaved_snapshot_digests[digest_key] = file_io.compute_text_digest(text)

    def _handle_missing_last_file(self, settings: Settings) -> None:
        if settings.last_open_file:
            settings.last_open_file = None
            self._persist_settings(settings)
        self.update_status("Last session file missing")

    def _persist_unsaved_snapshot(self, state: DocumentState | None = None) -> None:
        settings = self._context.settings
        if settings is None:
            return

        document = state or self._editor.to_document()
        path = document.metadata.path or self._current_document_path
        key = self._snapshot_key(path)

        if not document.dirty:
            self._clear_unsaved_snapshot(settings=settings, path=path)
            return

        snapshot = {
            "text": document.text,
            "language": document.metadata.language,
            "selection": list(document.selection.as_tuple()),
        }
        digest = file_io.compute_text_digest(snapshot["text"])
        if self._unsaved_snapshot_digests.get(key) == digest:
            existing = self._get_snapshot_entry(settings, key)
            if existing == snapshot:
                return

        if key == self._UNTITLED_SNAPSHOT_KEY:
            settings.unsaved_snapshot = snapshot
        else:
            snapshots = dict(settings.unsaved_snapshots or {})
            snapshots[key] = snapshot
            settings.unsaved_snapshots = snapshots

        self._unsaved_snapshot_digests[key] = digest
        self._persist_settings(settings)

    def _clear_unsaved_snapshot(
        self,
        *,
        settings: Settings | None = None,
        path: Path | str | None = None,
        persist: bool = True,
    ) -> None:
        key = self._snapshot_key(path)
        target_settings = settings or self._context.settings
        changed = False

        if target_settings is not None:
            if key == self._UNTITLED_SNAPSHOT_KEY:
                if target_settings.unsaved_snapshot is not None:
                    target_settings.unsaved_snapshot = None
                    changed = True
            else:
                snapshots = dict(target_settings.unsaved_snapshots or {})
                if snapshots.pop(key, None) is not None:
                    target_settings.unsaved_snapshots = snapshots
                    changed = True

        self._unsaved_snapshot_digests.pop(key, None)
        if changed and persist and target_settings is not None:
            self._persist_settings(target_settings)

    def _get_snapshot_entry(self, settings: Settings, key: str) -> dict[str, Any] | None:
        if key == self._UNTITLED_SNAPSHOT_KEY:
            return settings.unsaved_snapshot
        return (settings.unsaved_snapshots or {}).get(key)

    def _snapshot_key(self, path: Path | str | None) -> str:
        if path is None:
            return self._UNTITLED_SNAPSHOT_KEY
        return str(Path(path).expanduser().resolve())

    def _qt_parent_widget(self) -> Any | None:
        try:
            from PySide6.QtWidgets import QWidget
        except Exception:  # pragma: no cover - PySide optional during tests
            return None
        return self if isinstance(self, QWidget) else None

    def _show_settings_dialog(self, settings: Settings) -> "SettingsDialogResult":
        try:
            from .widgets.dialogs import show_settings_dialog
        except Exception as exc:  # pragma: no cover - depends on Qt availability
            raise RuntimeError("Settings dialog requires the PySide6 dependency.") from exc

        parent = self._qt_parent_widget()
        return show_settings_dialog(parent=parent, settings=settings)

    def _persist_settings(self, settings: Settings | None) -> None:
        if settings is None:
            return

        store = self._context.settings_store
        if store is None:
            return

        try:
            store.save(settings)
        except Exception as exc:  # pragma: no cover - disk write failures are rare
            _LOGGER.warning("Failed to persist settings: %s", exc)

    def _apply_runtime_settings(self, settings: Settings) -> None:
        self._apply_chat_panel_settings(settings)
        self._apply_debug_logging_setting(settings)
        self._apply_theme_setting(settings)
        self._refresh_ai_runtime(settings)

    def _apply_chat_panel_settings(self, settings: Settings) -> None:
        visible = bool(getattr(settings, "show_tool_activity_panel", False))
        setter = getattr(self._chat_panel, "set_tool_activity_visibility", None)
        if callable(setter):
            setter(visible)

    def _update_logging_configuration(self, debug_enabled: bool) -> None:
        level = logging.DEBUG if debug_enabled else logging.INFO
        try:
            logging_utils.setup_logging(level, force=True)
            _LOGGER.debug("Runtime logging level updated to %s", logging.getLevelName(level))
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.warning("Unable to update logging configuration: %s", exc)

    def _update_ai_debug_logging(self, debug_enabled: bool) -> None:
        controller = self._context.ai_controller
        if controller is None:
            return
        client = getattr(controller, "client", None)
        if client is None:
            return
        client_settings = getattr(client, "settings", None)
        if client_settings is None:
            return
        try:
            client_settings.debug_logging = debug_enabled
            _LOGGER.debug("AI client debug logging set to %s", debug_enabled)
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.debug("Unable to update AI client debug flag: %s", exc)

    def _apply_debug_logging_setting(self, settings: Settings) -> None:
        new_debug = bool(getattr(settings, "debug_logging", False))
        if new_debug != self._debug_logging_enabled:
            self._update_logging_configuration(new_debug)
            self._debug_logging_enabled = new_debug
        self._update_ai_debug_logging(new_debug)

    def _apply_theme_setting(self, settings: Settings) -> None:
        theme_name = (getattr(settings, "theme", "") or "default").strip() or "default"
        if theme_name == self._active_theme:
            return
        self._active_theme = theme_name
        try:
            self._editor.apply_theme(theme_name)
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Unable to apply editor theme %s: %s", theme_name, exc)
        self._apply_application_theme(theme_name)

    def _apply_application_theme(self, theme_name: str) -> None:
        try:
            from PySide6.QtWidgets import QApplication
        except Exception:  # pragma: no cover - PySide optional during tests
            return

        app = QApplication.instance()
        if app is None:
            return

        normalized = theme_name.lower()
        setter = getattr(app, "setStyle", None)
        if not callable(setter):
            return

        if normalized == "dark":
            try:
                setter("Fusion")
            except Exception:  # pragma: no cover - style availability varies
                pass
            return

        if normalized in {"default", "light", ""}:
            default_style = getattr(app, "_tinkerbell_default_style", None)
            if default_style is None:
                style_getter = getattr(app, "style", None)
                style_obj = style_getter() if callable(style_getter) else None
                object_name_getter = getattr(style_obj, "objectName", None)
                default_style = object_name_getter() if callable(object_name_getter) else None
                setattr(app, "_tinkerbell_default_style", default_style or "Fusion")
            try:
                setter(default_style or "Fusion")
            except Exception:  # pragma: no cover - defensive guard
                pass

    def _refresh_ai_runtime(self, settings: Settings) -> None:
        if not self._ai_settings_ready(settings):
            self._disable_ai_controller()
            return

        signature = self._ai_settings_signature(settings)
        controller = self._context.ai_controller

        if controller is None:
            controller = self._build_ai_controller_from_settings(settings)
            if controller is None:
                return
            self._context.ai_controller = controller
            self._ai_client_signature = signature
            self._register_default_ai_tools()
        elif signature != self._ai_client_signature:
            client = self._build_ai_client_from_settings(settings)
            if client is None:
                return
            controller.update_client(client)
            self._ai_client_signature = signature

        self._update_ai_debug_logging(bool(getattr(settings, "debug_logging", False)))

    def _ai_settings_ready(self, settings: Settings) -> bool:
        return bool((settings.api_key or "").strip() and (settings.base_url or "").strip() and (settings.model or "").strip())

    def _ai_settings_signature(self, settings: Settings | None) -> tuple[Any, ...] | None:
        if settings is None:
            return None
        headers = tuple(sorted((settings.default_headers or {}).items()))
        metadata = tuple(sorted((settings.metadata or {}).items()))
        return (
            settings.base_url,
            settings.api_key,
            settings.model,
            settings.organization,
            settings.request_timeout,
            settings.max_retries,
            settings.retry_min_seconds,
            settings.retry_max_seconds,
            headers,
            metadata,
        )

    def _build_ai_client_from_settings(self, settings: Settings):
        try:
            from .ai.client import AIClient, ClientSettings
        except Exception as exc:  # pragma: no cover - dependency guard
            _LOGGER.warning("AI client components unavailable: %s", exc)
            return None

        client_settings = ClientSettings(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
            organization=settings.organization,
            request_timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            retry_min_seconds=settings.retry_min_seconds,
            retry_max_seconds=settings.retry_max_seconds,
            default_headers=settings.default_headers or None,
            metadata=settings.metadata or None,
            debug_logging=bool(getattr(settings, "debug_logging", False)),
        )
        try:
            return AIClient(client_settings)
        except Exception as exc:
            _LOGGER.warning("Failed to build AI client: %s", exc)
            return None

    def _build_ai_controller_from_settings(self, settings: Settings):
        client = self._build_ai_client_from_settings(settings)
        if client is None:
            return None
        try:
            from .ai.agents.executor import AIController
        except Exception as exc:  # pragma: no cover - dependency guard
            _LOGGER.warning("AI controller unavailable: %s", exc)
            return None

        try:
            return AIController(client=client)
        except Exception as exc:
            _LOGGER.warning("Failed to initialize AI controller: %s", exc)
            return None

    def _disable_ai_controller(self) -> None:
        controller = self._context.ai_controller
        if controller is None and self._ai_client_signature is None:
            return
        if self._ai_task and not self._ai_task.done():
            try:
                self._ai_task.cancel()
            except Exception:  # pragma: no cover - defensive guard
                pass
        self._context.ai_controller = None
        self._ai_task = None
        self._ai_stream_active = False
        self._ai_client_signature = None
        _LOGGER.info("AI controller disabled until settings are completed.")

    def _refresh_window_title(self, state: Optional[DocumentState] = None) -> None:
        """Construct a descriptive window title for the active document."""

        document = state or self._editor.to_document()
        candidate_path: Optional[Path]
        if document.metadata.path is not None:
            candidate_path = Path(document.metadata.path)
        elif self._current_document_path is not None:
            candidate_path = Path(self._current_document_path)
        else:
            candidate_path = None

        if candidate_path is not None and candidate_path.name:
            filename = candidate_path.name
        elif candidate_path is not None:
            filename = candidate_path.name or str(candidate_path)
        else:
            filename = UNTITLED_DOCUMENT_NAME

        dirty_prefix = "*" if document.dirty else ""
        subject = f"{dirty_prefix}{filename}" if dirty_prefix else filename
        base_title = self._WINDOW_BASE_TITLE
        title = f"{subject} - {base_title}" if subject else base_title

        try:
            self.setWindowTitle(title)
        except Exception:  # pragma: no cover - defensive guard for Qt shims
            pass

    # API surface reserved for future extensions ---------------------------------
    def menu_specs(self) -> Iterable[MenuSpec]:
        """Return menu specifications for downstream Qt wiring."""

        return tuple(self._menus.values())

    def toolbar_specs(self) -> Iterable[ToolbarSpec]:
        """Return toolbar specifications for downstream Qt wiring."""

        return tuple(self._toolbars.values())

