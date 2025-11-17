"""Main window implementation coordinating the editor and chat panes."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import os
import time
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Iterable, Mapping, Optional, Sequence, TYPE_CHECKING, cast

from .chat.chat_panel import ChatPanel
from .chat.commands import (
    ActionType,
    DIRECTIVE_SCHEMA,
    ManualCommandRequest,
    ManualCommandType,
    parse_manual_command,
    resolve_tab_reference,
)
from .chat.message_model import ChatMessage, EditDirective, ToolTrace
from .editor.document_model import DocumentMetadata, DocumentState, SelectionRange
from .editor.workspace import DocumentTab
from .editor.tabbed_editor import TabbedEditorWidget
from .services import telemetry as telemetry_service
from .ai.memory import EmbeddingProvider, DocumentEmbeddingIndex, LangChainEmbeddingProvider, OpenAIEmbeddingProvider
from .ai.memory.buffers import DocumentSummaryMemory
from .ai.services import OutlineBuilderWorker
from .services.bridge_router import WorkspaceBridgeRouter
from .services.importers import FileImporter, ImportResult, ImporterError
from .services.settings import Settings, SettingsStore
from .utils import file_io, logging as logging_utils
from .widgets.status_bar import StatusBar
from .ai.tools.diff_builder import DiffBuilderTool
from .ai.tools.document_snapshot import DocumentSnapshotTool
from .ai.tools.document_edit import DocumentEditTool
from .ai.tools.document_apply_patch import DocumentApplyPatchTool
from .ai.tools.document_find_sections import DocumentFindSectionsTool
from .ai.tools.document_outline import DocumentOutlineTool
from .ai.tools.list_tabs import ListTabsTool
from .ai.tools.search_replace import SearchReplaceTool
from .ai.tools.validation import validate_snippet
from .theme import Theme, load_theme, theme_manager

if TYPE_CHECKING:  # pragma: no cover - import only for static analysis
    from .ai.agents.executor import AIController
    from .ai.client import AIStreamEvent
    from .widgets.dialogs import SettingsDialogResult

QApplication: Any
QMainWindow: Any
QWidget: Any

try:  # pragma: no cover - PySide6 optional in CI
    from PySide6.QtWidgets import QApplication as _QtQApplication, QMainWindow as _QtQMainWindow, QWidget as _QtQWidget
    QMainWindow = _QtQMainWindow
    QWidget = _QtQWidget
    QApplication = _QtQApplication
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
    
    class _StubQApplication:  # type: ignore
        """Minimal QApplication placeholder for headless tests."""

        @staticmethod
        def instance() -> Any:
            return None

    QApplication = _StubQApplication


_LOGGER = logging.getLogger(__name__)


WINDOW_APP_NAME = "TinkerBell"
UNTITLED_DOCUMENT_NAME = "Untitled"
SUGGESTION_LOADING_LABEL = "Generating personalized suggestions…"
OPENAI_API_BASE_URL = "https://api.openai.com/v1"


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


@dataclass(slots=True)
class _PendingToolTrace:
    """Bookkeeping for streaming tool call data before it is displayed."""

    name: str
    arguments_chunks: list[str] = field(default_factory=list)
    raw_input: str | None = None
    pending_output: str | None = None
    pending_parsed: Any | None = None
    trace: ToolTrace | None = None
    started_at: float | None = None
    tool_call_id: str | None = None


@dataclass(slots=True)
class OutlineStatusInfo:
    """Tracks outline freshness/latency metadata per document."""

    status_label: str = ""
    status_code: str = ""
    tooltip: str = ""
    version_id: int | None = None
    outline_hash: str | None = None
    latency_ms: float | None = None
    completed_at: float | None = None
    node_count: int | None = None
    token_count: int | None = None
    stale: bool = False
    stale_since: float | None = None


@dataclass(slots=True)
class EmbeddingRuntimeState:
    """Bookkeeping structure describing the active embedding backend."""

    backend: str = "disabled"
    model: str | None = None
    provider_label: str | None = None
    status: str = "disabled"
    detail: str | None = None
    error: str | None = None

    @property
    def label(self) -> str:
        if self.status in {"disabled", "unavailable"}:
            return "Off" if self.status == "disabled" else "Unavailable"
        if self.status == "error":
            return "Error"
        return self.provider_label or self.backend.title()

    def as_snapshot(self) -> dict[str, Any]:
        return {
            "embedding_backend": self.backend if self.backend != "disabled" else None,
            "embedding_model": self.model,
            "embedding_status": self.status,
            "embedding_detail": self.detail or self.error,
        }


@dataclass(slots=True, frozen=True)
class _LangChainProviderTemplate:
    """Describes heuristics for provider-specific LangChain wiring."""

    family: str
    match_keywords: tuple[str, ...]
    default_base_url: str | None = None
    api_key_env: str | None = None
    tokenizer: str | None = None
    default_headers: Mapping[str, str] | None = None
    dimensions: Mapping[str, int] = field(default_factory=dict)
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def matches(self, model_name: str, *, forced_family: str | None = None) -> bool:
        if forced_family:
            return forced_family == self.family
        if not self.match_keywords:
            return True
        lowered = model_name.lower()
        return any(keyword in lowered for keyword in self.match_keywords)

    def dimension_for(self, model_name: str) -> int | None:
        lookup = self.dimensions or {}
        if not lookup:
            return None
        return lookup.get(model_name.lower())


_DEFAULT_LANGCHAIN_PROVIDER = _LangChainProviderTemplate(
    family="openai",
    match_keywords=("text-embedding", "openai", "ada"),
    default_base_url=OPENAI_API_BASE_URL,
    tokenizer="cl100k_base",
    dimensions={
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    },
)

_LANGCHAIN_PROVIDER_TEMPLATES: tuple[_LangChainProviderTemplate, ...] = (
    _DEFAULT_LANGCHAIN_PROVIDER,
    _LangChainProviderTemplate(
        family="deepseek",
        match_keywords=("deepseek",),
        default_base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        tokenizer="cl100k_base",
        dimensions={"deepseek-embedding": 1536},
    ),
    _LangChainProviderTemplate(
        family="glm",
        match_keywords=("glm", "zhipu"),
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
        tokenizer="cl100k_base",
        dimensions={"glm-4-embed": 1024},
    ),
    _LangChainProviderTemplate(
        family="moonshot",
        match_keywords=("moonshot", "kimi"),
        default_base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        tokenizer="cl100k_base",
    ),
)

_LANGCHAIN_PROVIDER_ENV_VARS: tuple[str, ...] = tuple(
    sorted({template.api_key_env for template in _LANGCHAIN_PROVIDER_TEMPLATES if template.api_key_env})
)


class MainWindow(QMainWindow):
    """Primary application window hosting the editor and chat splitter."""

    _WINDOW_BASE_TITLE = WINDOW_APP_NAME
    _UNTITLED_SNAPSHOT_KEY = "__untitled__"

    def __init__(self, context: WindowContext):  # noqa: D401 - doc inherited
        super().__init__()
        self._context = context
        initial_settings = context.settings
        self._phase3_outline_enabled = bool(getattr(initial_settings, "phase3_outline_tools", False))
        show_tool_panel = bool(getattr(initial_settings, "show_tool_activity_panel", False))
        self._editor = TabbedEditorWidget()
        self._workspace = self._editor.workspace
        self._file_importer = FileImporter()
        self._chat_panel = ChatPanel(show_tool_activity_panel=show_tool_panel)
        self._bridge = WorkspaceBridgeRouter(self._workspace)
        self._editor.add_tab_created_listener(self._bridge.track_tab)
        self._workspace.add_active_listener(self._handle_active_tab_changed)
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
        self._pending_tool_traces: Dict[str, _PendingToolTrace] = {}
        self._tool_trace_index: Dict[str, ToolTrace] = {}
        self._last_compaction_stats: dict[str, int] | None = None
        self._suggestion_task: asyncio.Task[Any] | None = None
        self._suggestion_request_id = 0
        self._suggestion_cache_key: str | None = None
        self._suggestion_cache_values: tuple[str, ...] | None = None
        self._unsaved_snapshot_digests: dict[str, str] = {}
        self._outline_digest_cache: dict[str, str] = {}
        self._outline_status_by_document: dict[str, OutlineStatusInfo] = {}
        self._snapshot_persistence_block = 0
        self._debug_logging_enabled = bool(getattr(initial_settings, "debug_logging", False))
        self._active_theme: str | None = None
        self._active_theme_request: str | None = None
        self._auto_patch_tool: DocumentApplyPatchTool | None = None
        self._ai_client_signature: tuple[Any, ...] | None = self._ai_settings_signature(initial_settings)
        self._restoring_workspace = False
        self._tabs_with_overlay: set[str] = set()
        self._last_autosave_at: datetime | None = None
        self._outline_worker: OutlineBuilderWorker | None = None
        self._outline_tool: DocumentOutlineTool | None = None
        self._find_sections_tool: DocumentFindSectionsTool | None = None
        self._embedding_index: DocumentEmbeddingIndex | None = None
        self._embedding_state = EmbeddingRuntimeState()
        self._embedding_signature: tuple[Any, ...] | None = None
        self._embedding_snapshot_metadata: dict[str, Any] = {}
        self._embedding_resource: Any | None = None
        self._last_outline_status: tuple[str, str] | None = None
        self._initialize_ui()
        if initial_settings is not None:
            self._apply_theme_setting(initial_settings)
        self._refresh_embedding_runtime(initial_settings)
        if self._phase3_outline_enabled:
            self._outline_worker = self._create_outline_worker()
            self._propagate_embedding_index_to_worker()

    # ------------------------------------------------------------------
    # Qt lifecycle hooks
    # ------------------------------------------------------------------
    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        """Ensure background tasks are canceled before the window closes."""

        self._cancel_active_ai_turn()
        self._cancel_dynamic_suggestions()
        self._clear_suggestion_cache()
        self._shutdown_outline_worker()
        self._request_app_shutdown()

        super_close = getattr(super(), "closeEvent", None)
        if callable(super_close):  # pragma: no branch - defensive wiring
            try:
                super_close(event)
            except Exception:  # pragma: no cover - Qt stubs/tests
                _LOGGER.debug("Base closeEvent handler raised", exc_info=True)

        accept = getattr(event, "accept", None)
        if callable(accept):
            accept()

    def _request_app_shutdown(self) -> None:
        """Ask the QApplication to terminate so the qasync loop can stop."""

        qt_instance = None
        try:
            qt_instance_getter = getattr(QApplication, "instance", None)
            if callable(qt_instance_getter):
                qt_instance = qt_instance_getter()
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.debug("Failed to access QApplication instance", exc_info=True)
            return

        if not qt_instance:
            return

        closing_down = getattr(qt_instance, "closingDown", None)
        try:
            if callable(closing_down) and closing_down():
                return
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.debug("QApplication closingDown() check failed", exc_info=True)

        quit_fn = getattr(qt_instance, "quit", None)
        if callable(quit_fn):
            try:
                quit_fn()
            except Exception:  # pragma: no cover - defensive guard
                _LOGGER.debug("QApplication.quit() raised", exc_info=True)

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
        self._update_autosave_indicator(document=self._editor.to_document())

    def _create_outline_worker(self) -> OutlineBuilderWorker | None:
        loop = self._resolve_async_loop()
        if loop is None:
            return None
        if loop.is_running():
            return self._start_outline_worker(loop)
        try:
            loop.call_soon(self._start_outline_worker, loop)
        except RuntimeError:  # pragma: no cover - loop may be closed in tests
            return None
        return None

    def _start_outline_worker(self, loop: asyncio.AbstractEventLoop) -> OutlineBuilderWorker | None:
        if getattr(self, "_outline_worker", None):
            return self._outline_worker
        cache_dir = self._resolve_outline_cache_root()
        try:
            worker = OutlineBuilderWorker(
                document_provider=self._workspace.find_document_by_id,
                storage_dir=cache_dir,
                loop=loop,
            )
        except Exception:  # pragma: no cover - background worker optional in tests
            _LOGGER.debug("Outline worker unavailable; continuing without outlines.", exc_info=True)
            return None
        self._outline_worker = worker
        return worker

    def _shutdown_outline_worker(self) -> None:
        worker = getattr(self, "_outline_worker", None)
        if worker is None:
            return
        self._outline_worker = None
        close_coro = worker.aclose()
        loop = worker.loop
        if loop.is_running():
            loop.create_task(close_coro)
            return
        try:
            loop.run_until_complete(close_coro)
        except RuntimeError:
            asyncio.run(close_coro)

    def _outline_memory(self) -> DocumentSummaryMemory | None:
        worker = getattr(self, "_outline_worker", None)
        if worker is None:
            return None
        return getattr(worker, "memory", None)

    def _resolve_embedding_index(self) -> DocumentEmbeddingIndex | None:
        return getattr(self, "_embedding_index", None)

    def _safe_active_document(self) -> DocumentState | None:
        try:
            return self._workspace.active_document()
        except Exception:
            return None

    def _resolve_outline_digest(self, document_id: str | None) -> str | None:
        doc_id = str(document_id).strip() if document_id else ""
        if not doc_id:
            document = self._safe_active_document()
            if document is None:
                return None
            doc_id = document.document_id
        memory = self._outline_memory()
        if memory is None:
            return None
        record = memory.get(doc_id)
        return record.outline_hash if record else None

    def _resolve_async_loop(self) -> asyncio.AbstractEventLoop | None:
        try:
            return asyncio.get_event_loop()
        except RuntimeError:
            return None

    def _run_background_task(self, task_factory: Callable[[], Coroutine[Any, Any, Any]]) -> None:
        def _build_coroutine() -> Coroutine[Any, Any, Any]:
            coroutine = task_factory()
            if not asyncio.iscoroutine(coroutine):  # pragma: no cover - defensive check
                raise TypeError("Background task factory must return a coroutine")
            return coroutine

        try:
            loop = self._resolve_async_loop()
            if loop is not None and loop.is_running():
                loop.create_task(_build_coroutine())
                return
            try:
                asyncio.run(_build_coroutine())
                return
            except RuntimeError:
                new_loop = asyncio.new_event_loop()
                try:
                    new_loop.run_until_complete(_build_coroutine())
                finally:
                    new_loop.close()
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.debug("Background task failed", exc_info=True)

    def _resolve_outline_cache_root(self) -> Path:
        store = getattr(self._context, "settings_store", None)
        if store is not None:
            base_dir = store.path.parent
        else:
            base_dir = Path.home() / ".tinkerbell"
        return base_dir / "cache" / "outline_builder"

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
            "file_new_tab": self._build_action(
                name="file_new_tab",
                text="New Tab",
                shortcut="Ctrl+N",
                status_tip="Create a new untitled tab",
                callback=self._handle_new_tab_requested,
            ),
            "file_open": self._build_action(
                name="file_open",
                text="Open…",
                shortcut="Ctrl+O",
                status_tip="Open a document from disk",
                callback=self._handle_open_requested,
            ),
            "file_import": self._build_action(
                name="file_import",
                text="Import…",
                shortcut="Ctrl+Shift+I",
                status_tip="Convert PDFs and other formats into editable text",
                callback=self._handle_import_requested,
            ),
            "file_save": self._build_action(
                name="file_save",
                text="Save",
                shortcut="Ctrl+S",
                status_tip="Save the current document",
                callback=self._handle_save_requested,
            ),
            "file_close_tab": self._build_action(
                name="file_close_tab",
                text="Close Tab",
                shortcut="Ctrl+W",
                status_tip="Close the active tab",
                callback=self._handle_close_tab_requested,
            ),
            "file_revert": self._build_action(
                name="file_revert",
                text="Revert",
                shortcut=None,
                status_tip="Discard unsaved changes and reload the file from disk",
                callback=self._handle_revert_requested,
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
                actions=("file_new_tab", "file_open", "file_import", "file_save", "file_save_as", "file_close_tab", "file_revert"),
            ),
            "settings": MenuSpec(name="settings", title="&Settings", actions=("settings_open",)),
            "ai": MenuSpec(name="ai", title="&AI", actions=("ai_snapshot",)),
        }

    def _create_toolbars(self) -> Dict[str, ToolbarSpec]:
        """Return declarative toolbar metadata used by future Qt wiring."""

        return {
            "file": ToolbarSpec(name="file", actions=("file_new_tab", "file_open", "file_save", "file_close_tab")),
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
        self._chat_panel.add_suggestion_panel_listener(self._handle_suggestion_panel_toggled)
        self._chat_panel.set_stop_ai_callback(self._cancel_active_ai_turn)
        self._bridge.add_edit_listener(self._handle_edit_applied)
        self._bridge.add_failure_listener(self._handle_edit_failure)
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
            snapshot_tool = DocumentSnapshotTool(
                provider=self._bridge,
                outline_digest_resolver=self._resolve_outline_digest,
            )
            register(
                "document_snapshot",
                snapshot_tool,
                description=(
                    "Return the freshest document snapshot (text, metadata, diff summaries) for the active or specified tab."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "delta_only": {
                            "type": "boolean",
                            "description": "When true, include only the selection and surrounding context instead of the full document.",
                        },
                        "include_diff": {
                            "type": "boolean",
                            "description": "Attach the most recent diff summary when available (default true).",
                        },
                        "tab_id": {
                            "type": "string",
                            "description": "Target a specific tab instead of the active document.",
                        },
                        "source_tab_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of additional tab snapshots to gather alongside the active document.",
                        },
                        "include_open_documents": {
                            "type": "boolean",
                            "description": "When true, include a summary of all open documents in the snapshot payload.",
                        },
                    },
                    "additionalProperties": False,
                },
            )

            self._register_phase3_ai_tools(register_fn=register)

            edit_tool = DocumentEditTool(bridge=self._bridge, patch_only=True)
            register(
                "document_edit",
                edit_tool,
                description=(
                    "Apply a structured edit directive (insert, replace, annotate, or unified diff patch) against the active document."
                ),
                parameters=self._directive_parameters_schema(),
            )

            apply_patch_tool = DocumentApplyPatchTool(bridge=self._bridge, edit_tool=edit_tool)
            self._auto_patch_tool = apply_patch_tool
            register(
                "document_apply_patch",
                apply_patch_tool,
                description=(
                    "Replace a target_range with new content by automatically building and applying a unified diff."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Replacement text that should occupy the specified target_range.",
                        },
                        "target_range": DIRECTIVE_SCHEMA["properties"]["target_range"],
                        "document_version": {
                            "type": "string",
                            "description": "Document snapshot version captured before drafting the edit.",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "Optional explanation stored alongside the edit directive.",
                        },
                        "context_lines": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Override the number of context lines included in the generated diff.",
                        },
                        "tab_id": {
                            "type": "string",
                            "description": "Optional tab identifier; defaults to the active tab when omitted.",
                        },
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            )

            tab_listing_tool = ListTabsTool(provider=self._bridge)
            register(
                "list_tabs",
                tab_listing_tool,
                description="Enumerate the open tabs (tab_id, title, path, dirty) so agents can target specific documents.",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            )

            diff_tool = DiffBuilderTool()
            register(
                "diff_builder",
                diff_tool,
                description=(
                    "Return a unified diff given original and updated text snippets to drive patch directives."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "original": {
                            "type": "string",
                            "description": "The prior version of the text block.",
                        },
                        "updated": {
                            "type": "string",
                            "description": "The revised text block.",
                        },
                        "filename": {
                            "type": "string",
                            "description": "Optional virtual filename used in diff headers.",
                        },
                        "context": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Number of context lines to include in the diff (default 3).",
                        },
                    },
                    "required": ["original", "updated"],
                    "additionalProperties": False,
                },
            )

            search_tool = SearchReplaceTool(bridge=self._bridge)
            register(
                "search_replace",
                search_tool,
                description=(
                    "Search the current document or selection and optionally apply replacements with regex/literal matching."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Text or regex pattern to find.",
                        },
                        "replacement": {
                            "type": "string",
                            "description": "Content that will replace each match.",
                        },
                        "is_regex": {
                            "type": "boolean",
                            "description": "Interpret the pattern as a regular expression.",
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["document", "selection"],
                            "description": "Limit replacements to the entire document or just the current selection.",
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "When true, do not apply edits—only preview the outcome.",
                        },
                        "max_replacements": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Optional cap on the number of replacements to perform.",
                        },
                        "match_case": {
                            "type": "boolean",
                            "description": "Respect character casing when matching (defaults to true).",
                        },
                        "whole_word": {
                            "type": "boolean",
                            "description": "Only match full words when true.",
                        },
                    },
                    "required": ["pattern", "replacement"],
                    "additionalProperties": False,
                },
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
                            "enum": ["yaml", "yml", "json", "markdown", "md"],
                        },
                    },
                    "required": ["text", "fmt"],
                    "additionalProperties": False,
                },
            )

            registered = [
                "document_snapshot",
                "document_edit",
                "document_apply_patch",
                "diff_builder",
                "search_replace",
                "validate_snippet",
                "list_tabs",
            ]
            if self._phase3_outline_enabled:
                registered.insert(1, "document_outline")
                registered.insert(2, "document_find_sections")
            _LOGGER.debug("Default AI tools registered: %s", ", ".join(registered))
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.warning("Failed to register default AI tools: %s", exc)

    def _register_phase3_ai_tools(self, *, register_fn: Callable[..., Any] | None = None) -> None:
        if not self._phase3_outline_enabled:
            return

        controller = self._context.ai_controller
        if controller is None:
            return

        register = register_fn or getattr(controller, "register_tool", None)
        if not callable(register):
            _LOGGER.debug("AI controller does not expose register_tool; skipping phase3 tool wiring.")
            return

        outline_tool = self._ensure_outline_tool()
        if outline_tool is not None:
            register(
                "document_outline",
                outline_tool,
                description=(
                    "Return the most recent outline for the active document, including pointer IDs, blurbs, and staleness metadata."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Optional explicit document identifier; defaults to the active tab.",
                        },
                        "desired_levels": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Limit the outline depth to this heading level before budgeting.",
                        },
                        "include_blurbs": {
                            "type": "boolean",
                            "description": "When false, omit excerpt blurbs to conserve tokens.",
                        },
                        "max_nodes": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 1000,
                            "description": "Cap the number of nodes returned prior to budget enforcement.",
                        },
                    },
                    "additionalProperties": False,
                },
            )

        find_sections_tool = self._ensure_find_sections_tool()
        if find_sections_tool is not None:
            register(
                "document_find_sections",
                find_sections_tool,
                description=(
                    "Return the best-matching document chunks for a natural language query using embeddings or fallback heuristics."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "document_id": {
                            "type": "string",
                            "description": "Optional target document identifier; defaults to the active document.",
                        },
                        "query": {
                            "type": "string",
                            "minLength": 1,
                            "description": "Natural-language description of the section(s) to find.",
                        },
                        "top_k": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 12,
                            "description": "Maximum number of pointers to return.",
                        },
                        "min_confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": "Minimum embedding match score before falling back to heuristics.",
                        },
                        "filters": {
                            "type": "object",
                            "description": "Optional additional filtering hints understood by custom index providers.",
                        },
                        "include_outline_context": {
                            "type": "boolean",
                            "description": "When true, include nearby outline headings for each pointer.",
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            )

    def _unregister_phase3_ai_tools(self) -> None:
        controller = self._context.ai_controller
        if controller is None:
            return
        unregister = getattr(controller, "unregister_tool", None)
        if not callable(unregister):
            return
        for name in ("document_outline", "document_find_sections"):
            try:
                unregister(name)
            except Exception:  # pragma: no cover - defensive
                _LOGGER.debug("Failed to unregister tool %s", name, exc_info=True)
    def _ensure_outline_tool(self) -> DocumentOutlineTool | None:
        if not self._phase3_outline_enabled:
            return None
        if self._outline_tool is not None:
            return self._outline_tool
        try:
            def _pending_outline(document_id: str) -> bool:
                worker = getattr(self, "_outline_worker", None)
                if worker is None:
                    return False
                return worker.is_rebuild_pending(document_id)

            tool = DocumentOutlineTool(
                memory_resolver=self._outline_memory,
                document_lookup=self._workspace.find_document_by_id,
                active_document_provider=self._safe_active_document,
                budget_policy=None,
                pending_outline_checker=_pending_outline,
            )
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.debug("Unable to initialize DocumentOutlineTool", exc_info=True)
            return None
        self._outline_tool = tool
        return tool

    def _ensure_find_sections_tool(self) -> DocumentFindSectionsTool | None:
        if not self._phase3_outline_enabled:
            return None
        if self._find_sections_tool is not None:
            return self._find_sections_tool
        try:
            tool = DocumentFindSectionsTool(
                embedding_index_resolver=self._resolve_embedding_index,
                document_lookup=self._workspace.find_document_by_id,
                active_document_provider=self._safe_active_document,
                outline_memory=self._outline_memory,
            )
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.debug("Unable to initialize DocumentFindSectionsTool", exc_info=True)
            return None
        self._find_sections_tool = tool
        return tool

    @staticmethod
    def _directive_parameters_schema() -> Dict[str, Any]:
        """Return a copy of the directive schema used by the document edit tool."""

        schema = deepcopy(DIRECTIVE_SCHEMA)
        schema.setdefault(
            "description",
            "Structured edit directive containing action, content, optional rationale, and target range. Prefer action='patch' with a unified diff and document_version when modifying existing text.",
        )
        return schema

    # ------------------------------------------------------------------
    # AI coordination
    # ------------------------------------------------------------------
    def _handle_chat_request(self, prompt: str, metadata: dict[str, Any]) -> None:
        try:
            manual_request = parse_manual_command(prompt)
        except ValueError as exc:
            self._post_assistant_notice(str(exc))
            self.update_status("Manual command error")
            return
        if manual_request is not None:
            self._handle_manual_command(manual_request)
            return

        controller = self._context.ai_controller
        if controller is None:
            self._post_assistant_notice(
                "AI assistant is unavailable. Open Settings to configure your API key and model."
            )
            self.update_status("AI unavailable")
            self._chat_panel.set_ai_running(False)
            return

        if self._ai_task and not self._ai_task.done():
            self._cancel_active_ai_turn()
            self.update_status("Previous AI request canceled")

        snapshot = self._bridge.generate_snapshot()
        self._apply_embedding_metadata(snapshot)
        history_payload = self._serialize_chat_history(
            self._chat_panel.history(),
            limit=0,
            exclude_latest=True,
        )
        self._chat_panel.set_ai_running(True)
        task = self._run_coroutine(
            self._run_ai_turn(controller, prompt, snapshot, metadata or {}, history_payload)
        )
        self._ai_task = task
        if task is None:
            self._chat_panel.set_ai_running(False)

    def _handle_manual_command(self, request: ManualCommandRequest) -> None:
        if request.command is ManualCommandType.OUTLINE:
            self._handle_manual_outline_command(request)
            return
        if request.command is ManualCommandType.FIND_SECTIONS:
            self._handle_manual_find_sections_command(request)
            return
        self._post_assistant_notice(f"Unsupported manual command '{request.command.value}'.")
        self.update_status("Manual command unsupported")

    def _handle_manual_outline_command(self, request: ManualCommandRequest) -> None:
        if not self._phase3_outline_enabled:
            self._post_assistant_notice("Outline tooling is disabled. Enable it in Settings > AI to use /outline.")
            self.update_status("Outline disabled")
            return

        tool = self._ensure_outline_tool()
        if tool is None:
            self._post_assistant_notice("Outline tool is unavailable.")
            self.update_status("Outline unavailable")
            return

        args = dict(request.args)
        doc_reference = args.get("document_id")
        resolved_id = self._resolve_manual_document_id(doc_reference)
        if doc_reference and resolved_id is None:
            self._post_assistant_notice(f"Couldn't find a document matching '{doc_reference}'.")
            self.update_status("Document not found")
            return
        if resolved_id:
            args["document_id"] = resolved_id
        else:
            args.pop("document_id", None)

        try:
            response = tool.run(**args)
        except Exception as exc:  # pragma: no cover - defensive path
            _LOGGER.debug("Manual outline command failed", exc_info=True)
            self._post_assistant_notice(f"Outline command failed: {exc}")
            self.update_status("Outline command failed")
            return

        message = self._render_manual_outline_response(response, doc_reference)
        self._post_assistant_notice(message)
        status_text = str(response.get("status") or "ok") if isinstance(response, Mapping) else "ok"
        self._record_manual_tool_trace(
            name="manual:document_outline",
            input_summary=self._summarize_manual_input("document_outline", args),
            output_summary=status_text,
            args=args,
            response=response,
        )
        self.update_status("Outline ready")

    def _handle_manual_find_sections_command(self, request: ManualCommandRequest) -> None:
        if not self._phase3_outline_enabled:
            self._post_assistant_notice("Retrieval tooling is disabled. Enable it in Settings > AI to use /find.")
            self.update_status("Retrieval disabled")
            return

        tool = self._ensure_find_sections_tool()
        if tool is None:
            self._post_assistant_notice("Find sections tool is unavailable.")
            self.update_status("Retrieval unavailable")
            return

        args = dict(request.args)
        doc_reference = args.get("document_id")
        resolved_id = self._resolve_manual_document_id(doc_reference)
        if doc_reference and resolved_id is None:
            self._post_assistant_notice(f"Couldn't find a document matching '{doc_reference}'.")
            self.update_status("Document not found")
            return
        if resolved_id:
            args["document_id"] = resolved_id
        else:
            args.pop("document_id", None)

        try:
            response = tool.run(**args)
        except Exception as exc:  # pragma: no cover - defensive path
            _LOGGER.debug("Manual find sections command failed", exc_info=True)
            self._post_assistant_notice(f"Find sections command failed: {exc}")
            self.update_status("Find sections failed")
            return

        message = self._render_manual_retrieval_response(response, args.get("query"), doc_reference)
        self._post_assistant_notice(message)
        status_text = str(response.get("status") or "ok") if isinstance(response, Mapping) else "ok"
        self._record_manual_tool_trace(
            name="manual:document_find_sections",
            input_summary=self._summarize_manual_input("document_find_sections", args),
            output_summary=status_text,
            args=args,
            response=response,
        )
        self.update_status("Find sections ready")

    def _resolve_manual_document_id(self, reference: str | None) -> str | None:
        text = (reference or "").strip()
        if not text:
            return None
        lowered = text.lower()
        if lowered in {"active", "current"}:
            document = self._safe_active_document()
            return document.document_id if document is not None else None

        direct = self._workspace.find_document_by_id(text)
        if direct is not None:
            return direct.document_id

        try:
            tab = self._workspace.get_tab(text)
        except KeyError:
            tab = None
        if tab is not None:
            return tab.document().document_id

        entries: list[dict[str, Any]] = []
        for tab in self._workspace.iter_tabs():
            document = tab.document()
            entry: dict[str, Any] = {
                "tab_id": tab.id,
                "title": tab.title,
                "label": tab.title,
            }
            if tab.untitled_index is not None:
                entry["untitled_index"] = tab.untitled_index
            path = document.metadata.path
            if path is not None:
                entry["path"] = str(path)
            entries.append(entry)

        resolved_tab_id = resolve_tab_reference(text, entries, active_tab_id=self._workspace.active_tab_id)
        if resolved_tab_id:
            try:
                tab = self._workspace.get_tab(resolved_tab_id)
            except KeyError:
                tab = None
            if tab is not None:
                return tab.document().document_id
        return None

    def _document_label_from_id(self, document_id: str | None, fallback: str | None = None) -> str:
        if document_id:
            document = self._workspace.find_document_by_id(document_id)
            if document is not None:
                return self._document_display_name(document)
            return document_id
        if fallback:
            return fallback
        document = self._safe_active_document()
        if document is not None:
            return self._document_display_name(document)
        return WINDOW_APP_NAME

    def _render_manual_outline_response(
        self,
        response: Mapping[str, Any],
        requested_label: str | None,
    ) -> str:
        status = str(response.get("status") or "unknown")
        doc_label = self._document_label_from_id(response.get("document_id"), fallback=requested_label)
        parts = [f"Document outline ({status}) for {doc_label}."]
        reason = response.get("reason")
        if reason:
            parts.append(f"Reason: {reason}.")

        nodes = response.get("nodes") or []
        if nodes:
            parts.append("Headings:")
            parts.extend(self._render_outline_tree_lines(nodes))
        else:
            outline_available = response.get("outline_available")
            if outline_available is False:
                parts.append("No outline is available for this document yet.")

        notes: list[str] = []
        if response.get("trimmed"):
            reason_text = response.get("trimmed_reason") or "request limits"
            notes.append(f"trimmed={reason_text}")
        if response.get("is_stale"):
            notes.append("stale compared to current document")
        if notes:
            parts.append("Notes: " + ", ".join(notes) + ".")

        generated = response.get("generated_at")
        if generated:
            parts.append(f"Generated at {generated}.")
        outline_digest = response.get("outline_digest")
        if outline_digest:
            parts.append(f"Digest: {outline_digest}.")

        return "\n".join(part for part in parts if part)

    def _render_outline_tree_lines(self, nodes: Sequence[Mapping[str, Any]], limit: int = 24) -> list[str]:
        lines: list[str] = []
        truncated = False

        def visit(node: Mapping[str, Any], level_hint: int) -> None:
            nonlocal truncated
            if len(lines) >= limit:
                truncated = True
                return
            level = int(node.get("level") or level_hint or 1)
            indent = "  " * max(0, level - 1)
            text = str(node.get("text") or "Untitled").strip() or "Untitled"
            pointer = node.get("pointer_id") or node.get("id")
            suffix = f" ({pointer})" if pointer else ""
            lines.append(f"{indent}- {text}{suffix}")
            children = node.get("children") or []
            for child in children:
                if not isinstance(child, Mapping):
                    continue
                visit(child, level + 1)
                if truncated:
                    return

        for entry in nodes:
            if not isinstance(entry, Mapping):
                continue
            visit(entry, int(entry.get("level") or 1))
            if truncated:
                break

        if truncated:
            lines.append("  … additional headings omitted.")
        return lines

    def _render_manual_retrieval_response(
        self,
        response: Mapping[str, Any],
        requested_query: str | None,
        requested_document_label: str | None,
    ) -> str:
        status = str(response.get("status") or "unknown")
        doc_label = self._document_label_from_id(response.get("document_id"), fallback=requested_document_label)
        query_text = response.get("query") or (requested_query or "")
        if query_text:
            header = f"Find sections ({status}) for {doc_label} — \"{query_text}\""
        else:
            header = f"Find sections ({status}) for {doc_label}."
        parts = [header]

        details: list[str] = []
        strategy = response.get("strategy")
        if strategy:
            details.append(f"strategy={strategy}")
        fallback_reason = response.get("fallback_reason")
        if fallback_reason:
            details.append(f"fallback={fallback_reason}")
        latency = response.get("latency_ms")
        if isinstance(latency, (int, float)):
            details.append(f"latency={latency:.1f} ms")
        if details:
            parts.append("Details: " + ", ".join(details) + ".")

        pointers = response.get("pointers") or []
        if pointers:
            parts.append("Matches:")
            parts.extend(self._format_retrieval_pointers(pointers))
            extra = max(0, len(pointers) - 5)
            if extra:
                parts.append(f"… {extra} additional match(es).")
        else:
            parts.append("No matching sections were found.")

        return "\n".join(parts)

    def _format_retrieval_pointers(
        self,
        pointers: Sequence[Mapping[str, Any]],
        *,
        limit: int = 5,
    ) -> list[str]:
        lines: list[str] = []
        for index, pointer in enumerate(pointers[:limit], start=1):
            if not isinstance(pointer, Mapping):
                continue
            pointer_id = pointer.get("pointer_id") or pointer.get("chunk_id") or f"chunk-{index}"
            outline_context = pointer.get("outline_context")
            heading = outline_context.get("heading") if isinstance(outline_context, Mapping) else None
            label_parts = [str(pointer_id)]
            if heading:
                label_parts.append(str(heading))
            score = pointer.get("score")
            score_text = f"{float(score):.2f}" if isinstance(score, (int, float)) else "n/a"
            lines.append(f"{index}. {' · '.join(label_parts)} (score {score_text})")
            preview = pointer.get("preview")
            snippet = self._condense_whitespace(str(preview)) if isinstance(preview, str) else ""
            if snippet:
                if len(snippet) > 180:
                    snippet = f"{snippet[:177]}…"
                lines.append(f"    {snippet}")
        return lines

    def _summarize_manual_input(self, label: str, args: Mapping[str, Any]) -> str:
        if not args:
            return label
        preferred = ("document_id", "query", "top_k", "desired_levels", "max_nodes", "min_confidence")
        parts: list[str] = []
        for key in preferred:
            if key in args:
                parts.append(f"{key}={args[key]}")
        if not parts:
            for key, value in list(args.items())[:4]:
                parts.append(f"{key}={value}")
        summary = ", ".join(parts[:4])
        return f"{label} ({summary})" if summary else label

    def _record_manual_tool_trace(
        self,
        *,
        name: str,
        input_summary: str,
        output_summary: str,
        args: Mapping[str, Any],
        response: Mapping[str, Any] | Any,
    ) -> None:
        args_payload = dict(args) if isinstance(args, Mapping) else {"value": args}
        if isinstance(response, Mapping):
            response_payload: Mapping[str, Any] | Any = response
        else:
            response_payload = {"value": response}
        metadata = {
            "raw_input": json.dumps(args_payload, default=str),
            "raw_output": json.dumps(response_payload, default=str),
            "manual_command": True,
        }
        trace = ToolTrace(
            name=name,
            input_summary=input_summary,
            output_summary=output_summary,
            metadata=metadata,
        )
        self._chat_panel.show_tool_trace(trace)

    async def _run_ai_turn(
        self,
        controller: "AIController",
        prompt: str,
        snapshot: Mapping[str, Any],
        metadata: Mapping[str, Any],
        history: Sequence[Mapping[str, str]] | None,
    ) -> None:
        self._ai_stream_active = False
        normalized_metadata = self._normalize_metadata(metadata)
        self.update_status("AI thinking…")

        try:
            result = await controller.run_chat(
                prompt,
                snapshot,
                metadata=normalized_metadata,
                history=history,
                on_event=self._handle_ai_stream_event,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            self._handle_ai_failure(exc)
            return

        payload = result or {}
        tool_records = payload.get("tool_calls")
        self._annotate_tool_traces_with_compaction(tool_records)
        self._last_compaction_stats = payload.get("trace_compaction")
        response_text = payload.get("response", "").strip()
        if not response_text:
            response_text = "The AI did not return any content."
        self._finalize_ai_response(response_text)
        self._update_context_usage_status()
        self.update_status("AI response ready")

    async def _handle_ai_stream_event(self, event: "AIStreamEvent") -> None:
        self._process_stream_event(event)

    def _process_stream_event(self, event: "AIStreamEvent") -> None:
        event_type = getattr(event, "type", "") or ""
        if event_type in {"content.delta", "refusal.delta"}:
            chunk = self._coerce_stream_text(getattr(event, "content", None))
            if chunk:
                self._chat_panel.append_ai_message(
                    ChatMessage(role="assistant", content=chunk),
                    streaming=True,
                )
                self._ai_stream_active = True
            return

        if event_type in {"content.done", "refusal.done"}:
            if self._ai_stream_active:
                return
            chunk = self._coerce_stream_text(getattr(event, "content", None))
            if chunk:
                self._chat_panel.append_ai_message(
                    ChatMessage(role="assistant", content=chunk),
                    streaming=True,
                )
                self._ai_stream_active = True
            return

        if event_type == "tool_calls.function.arguments.delta":
            self._record_tool_call_arguments_delta(event)
            return

        if event_type == "tool_calls.function.arguments.done":
            self._finalize_tool_call_arguments(event)
            return

        if event_type == "tool_calls.result":
            self._record_tool_call_result(event)
            return

    def _record_tool_call_arguments_delta(self, event: Any) -> None:
        key = self._tool_call_key(event)
        if not key:
            return
        state = self._pending_tool_traces.get(key)
        if state is None:
            state = _PendingToolTrace(name=self._normalize_tool_name(event))
            self._pending_tool_traces[key] = state
        state.tool_call_id = key
        delta = getattr(event, "arguments_delta", None) or getattr(event, "tool_arguments", None)
        if delta:
            state.arguments_chunks.append(str(delta))

    def _finalize_tool_call_arguments(self, event: Any) -> None:
        key = self._tool_call_key(event)
        if not key:
            return
        state = self._pending_tool_traces.get(key)
        if state is None:
            state = _PendingToolTrace(name=self._normalize_tool_name(event))
            self._pending_tool_traces[key] = state
        state.tool_call_id = key
        arguments_text = getattr(event, "tool_arguments", None)
        if not arguments_text:
            arguments_text = "".join(state.arguments_chunks)
        state.arguments_chunks.clear()
        state.raw_input = str(arguments_text or "")
        metadata: Dict[str, Any] = {"raw_input": state.raw_input}
        metadata["tool_call_id"] = state.tool_call_id or key
        trace = ToolTrace(
            name=state.name,
            input_summary=self._summarize_tool_io(state.raw_input),
            output_summary="(running…)",
            metadata=metadata,
        )
        state.trace = trace
        state.started_at = time.perf_counter()
        self._chat_panel.show_tool_trace(trace)
        if state.pending_output is not None:
            self._apply_tool_result_to_trace(key, state)

    def _update_context_usage_status(self) -> None:
        controller = self._context.ai_controller
        settings = self._context.settings
        if controller is None or settings is None:
            return
        debug_settings = getattr(settings, "debug", None)
        if not getattr(debug_settings, "token_logging_enabled", False):
            return
        limit = getattr(debug_settings, "token_log_limit", 1)
        try:
            limit_value = max(1, int(limit))
        except (TypeError, ValueError):
            limit_value = 1
        events = controller.get_recent_context_events(limit=limit_value)
        dashboard = telemetry_service.build_usage_dashboard(events)
        if dashboard is None:
            return
        summary_text = dashboard.summary_text
        compaction_stats = self._last_compaction_stats
        if isinstance(compaction_stats, Mapping):
            compactions = int(compaction_stats.get("total_compactions", 0))
            tokens_saved = int(compaction_stats.get("tokens_saved", 0))
            if compactions or tokens_saved:
                stats_bits = f"Compactions {compactions}"
                if tokens_saved:
                    stats_bits = f"{stats_bits} (saved {tokens_saved:,})"
                summary_text = f"{summary_text} · {stats_bits}" if summary_text else stats_bits
        budget_snapshot = None
        getter = getattr(controller, "get_budget_status", None)
        if callable(getter):
            budget_snapshot = getter()
        if isinstance(budget_snapshot, Mapping):
            budget_text = str(budget_snapshot.get("summary_text") or "").strip()
            if budget_text:
                summary_text = f"{summary_text} · {budget_text}"
        self._status_bar.set_memory_usage(
            summary_text,
            totals=dashboard.totals_text,
            last_tool=dashboard.summary.last_tool,
        )

    def _update_autosave_indicator(
        self,
        *,
        autosaved: bool = False,
        document: DocumentState | None = None,
    ) -> None:
        doc = document or self._editor.to_document()
        if autosaved:
            self._last_autosave_at = datetime.now(timezone.utc)
        status, detail = self._format_autosave_label(doc)
        try:
            self._status_bar.set_autosave_state(status, detail=detail)
        except Exception:  # pragma: no cover - defensive guard
            pass

    def _format_autosave_label(self, document: DocumentState) -> tuple[str, str]:
        name = self._document_display_name(document)
        if not document.dirty:
            return ("Saved", name)
        if self._last_autosave_at is None:
            return ("Unsaved changes", name)
        elapsed = datetime.now(timezone.utc) - self._last_autosave_at
        return (f"Autosaved {self._format_elapsed(elapsed)}", name)

    @staticmethod
    def _format_elapsed(delta: timedelta) -> str:
        seconds = max(0, int(delta.total_seconds()))
        if seconds < 5:
            return "just now"
        if seconds < 60:
            return f"{seconds}s ago"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"

    def _record_tool_call_result(self, event: Any) -> None:
        key = self._tool_call_key(event)
        if not key:
            return
        state = self._pending_tool_traces.get(key)
        if state is None:
            state = _PendingToolTrace(name=self._normalize_tool_name(event))
            self._pending_tool_traces[key] = state
        state.tool_call_id = key
        content = getattr(event, "content", None) or getattr(event, "tool_arguments", None) or ""
        state.pending_output = str(content)
        state.pending_parsed = getattr(event, "parsed", None)
        self._apply_tool_result_to_trace(key, state)

    def _apply_tool_result_to_trace(self, key: str, state: _PendingToolTrace) -> None:
        trace = state.trace
        if trace is None or state.pending_output is None:
            return
        trace.output_summary = self._summarize_tool_io(state.pending_output)
        metadata = dict(trace.metadata)
        if state.raw_input is not None:
            metadata.setdefault("raw_input", state.raw_input)
        metadata["raw_output"] = state.pending_output
        if state.pending_parsed is not None:
            metadata["parsed_output"] = state.pending_parsed
        trace.metadata = metadata
        tool_call_id = metadata.get("tool_call_id")
        if isinstance(tool_call_id, str):
            self._tool_trace_index[tool_call_id] = trace
        if state.started_at is not None:
            elapsed = max(0.0, time.perf_counter() - state.started_at)
            trace.duration_ms = int(elapsed * 1000)
        self._chat_panel.update_tool_trace(trace)
        self._pending_tool_traces.pop(key, None)

    def _annotate_tool_traces_with_compaction(self, records: Sequence[Mapping[str, Any]] | None) -> None:
        if not records:
            return
        for record in records:
            if not isinstance(record, Mapping):
                continue
            pointer = record.get("pointer")
            if not pointer:
                continue
            call_id = record.get("id") or record.get("tool_call_id")
            if not call_id:
                continue
            trace = self._tool_trace_index.get(str(call_id))
            if trace is None:
                continue
            metadata = dict(trace.metadata or {})
            metadata["compacted"] = True
            metadata["pointer"] = pointer
            instructions = pointer.get("rehydrate_instructions") if isinstance(pointer, Mapping) else None
            if instructions:
                metadata["pointer_instructions"] = instructions
            summary = pointer.get("display_text") if isinstance(pointer, Mapping) else None
            if summary:
                metadata["pointer_summary"] = summary
                trace.output_summary = summary
            trace.metadata = metadata
            self._chat_panel.update_tool_trace(trace)

    def _tool_call_key(self, event: Any) -> str:
        identifier = getattr(event, "tool_call_id", None) or getattr(event, "id", None)
        if identifier:
            return str(identifier)
        index = getattr(event, "tool_index", None)
        return f"{self._normalize_tool_name(event)}:{index if index is not None else 0}"

    @staticmethod
    def _normalize_tool_name(event: Any) -> str:
        name = getattr(event, "tool_name", None) or "tool"
        text = str(name).strip()
        return text or "tool"

    @staticmethod
    def _summarize_tool_io(payload: Any, limit: int = 140) -> str:
        if payload is None:
            return "(empty)"
        text = str(payload).strip()
        if not text:
            return "(empty)"
        condensed = " ".join(text.split())
        if not condensed:
            return "(empty)"
        if len(condensed) <= limit:
            return condensed
        return f"{condensed[: limit - 1].rstrip()}…"

    def _coerce_stream_text(self, payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            return payload
        if isinstance(payload, Mapping):
            for key in ("text", "content", "value"):
                if key in payload:
                    text = self._coerce_stream_text(payload[key])
                    if text:
                        return text
        if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
            parts = [self._coerce_stream_text(item) for item in payload]
            return "".join(part for part in parts if part)
        text_attr = getattr(payload, "text", None)
        if text_attr:
            return self._coerce_stream_text(text_attr)
        content_attr = getattr(payload, "content", None)
        if content_attr is not None and content_attr is not payload:
            text = self._coerce_stream_text(content_attr)
            if text:
                return text
        return str(payload)

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
        self._chat_panel.set_ai_running(False)

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
        self._chat_panel.set_ai_running(False)

    def _handle_chat_session_reset(self) -> None:
        self._cancel_active_ai_turn()
        self._cancel_dynamic_suggestions()
        self._clear_suggestion_cache()
        self._tool_trace_index.clear()
        self._last_compaction_stats = None
        self._refresh_chat_suggestions()
        self.update_status("Chat reset")

    def _handle_suggestion_panel_toggled(self, is_open: bool) -> None:
        if not is_open:
            self._cancel_dynamic_suggestions()
            return

        history = self._chat_panel.history()
        if not history:
            self._refresh_chat_suggestions()
            return

        history_signature = self._history_signature(history)
        if history_signature is None:
            self._refresh_chat_suggestions()
            return

        if (
            history_signature == self._suggestion_cache_key
            and self._suggestion_cache_values
        ):
            self._chat_panel.set_suggestions(list(self._suggestion_cache_values))
            self.update_status("Loaded cached suggestions")
            return

        controller = self._context.ai_controller
        if controller is None:
            self._refresh_chat_suggestions()
            self.update_status("AI suggestions unavailable")
            return

        self._cancel_dynamic_suggestions()
        self._chat_panel.set_suggestions([SUGGESTION_LOADING_LABEL])
        self._suggestion_request_id += 1
        request_id = self._suggestion_request_id
        history_snapshot = tuple(history)

        task = self._run_coroutine(self._generate_dynamic_suggestions(history_snapshot, request_id))
        if isinstance(task, asyncio.Task):
            self._suggestion_task = task
            task.add_done_callback(self._on_suggestion_task_finished)
        else:
            self._suggestion_task = None

    def _cancel_dynamic_suggestions(self) -> None:
        task = self._suggestion_task
        if task is not None and not task.done():
            try:
                task.cancel()
            except Exception:  # pragma: no cover - defensive logging
                _LOGGER.debug("Failed to cancel suggestion task", exc_info=True)
        self._suggestion_task = None

    def _on_suggestion_task_finished(self, task: asyncio.Task[Any]) -> None:
        if task is self._suggestion_task:
            self._suggestion_task = None

    def _clear_suggestion_cache(self) -> None:
        self._suggestion_cache_key = None
        self._suggestion_cache_values = None

    def _store_suggestion_cache(self, history: Sequence[ChatMessage], suggestions: Sequence[str]) -> None:
        signature = self._history_signature(history)
        if signature is None:
            self._clear_suggestion_cache()
            return
        self._suggestion_cache_key = signature
        self._suggestion_cache_values = tuple(suggestions)

    def _history_signature(self, history: Sequence[ChatMessage]) -> str | None:
        payload = self._serialize_chat_history(history)
        if not payload:
            return None
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return digest

    def _serialize_chat_history(
        self,
        history: Sequence[ChatMessage],
        limit: int = 10,
        *,
        exclude_latest: bool = False,
    ) -> list[dict[str, str]]:
        messages = list(history)
        if exclude_latest and messages:
            messages = messages[:-1]
        windowed = messages[-limit:] if limit else messages
        serialized: list[dict[str, str]] = []
        for message in windowed:
            text = (message.content or "").strip()
            if not text:
                continue
            serialized.append({"role": message.role, "content": text})
        return serialized

    async def _generate_dynamic_suggestions(
        self,
        history: Sequence[ChatMessage],
        request_id: int,
    ) -> None:
        controller = self._context.ai_controller
        if controller is None:
            return

        payload = self._serialize_chat_history(history)
        if not payload:
            return

        suggestions: list[str] = []
        try:
            suggestions = await controller.suggest_followups(payload, max_suggestions=4)
        except asyncio.CancelledError:  # pragma: no cover - propagated cancellation
            raise
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.warning("Contextual suggestions failed: %s", exc)

        if request_id != self._suggestion_request_id:
            return

        if suggestions:
            self._store_suggestion_cache(history, suggestions)
            self._chat_panel.set_suggestions(suggestions)
            self.update_status("Contextual suggestions ready")
            return

        self._clear_suggestion_cache()
        self._refresh_chat_suggestions()
        self.update_status("Using preset suggestions")

    def _on_ai_task_finished(self, task: asyncio.Task[Any]) -> None:
        if task is self._ai_task:
            self._ai_task = None
        try:
            task.result()
        except Exception as exc:  # pragma: no cover - surfaced via handler
            self._handle_ai_failure(exc)
        finally:
            self._chat_panel.set_ai_running(False)

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
        metadata: Dict[str, Any] = {}
        range_hint: tuple[int, int] = directive.target_range
        context = getattr(self._bridge, "last_edit_context", None)
        if directive.action == ActionType.PATCH.value and context is not None:
            range_hint = context.target_range
        if directive.action == "replace":
            if context is not None and context.action == "replace":
                metadata["text_before"] = context.replaced_text
                metadata["text_after"] = context.content
            else:
                metadata["text_after"] = directive.content
        elif directive.action == ActionType.PATCH.value:
            if context is not None and context.diff:
                metadata["diff_preview"] = context.diff
            if context is not None and context.spans:
                metadata["spans"] = context.spans
        trace = ToolTrace(
            name=f"edit:{directive.action}",
            input_summary=f"range={range_hint}",
            output_summary=diff,
            metadata=metadata,
        )
        self._chat_panel.show_tool_trace(trace)
        self._apply_diff_overlay(trace, document=_state, range_hint=range_hint)
        self._update_autosave_indicator(document=_state)

    def _handle_edit_failure(self, directive: EditDirective, message: str) -> None:
        action = (directive.action or "").strip().lower()
        if action == ActionType.PATCH.value:
            notice = message or "Patch rejected"
            self.update_status("Patch rejected – ask TinkerBell to re-sync")
            self._post_assistant_notice(f"Patch apply failed: {notice}. Please request a fresh snapshot.")
        else:
            self.update_status(f"Edit failed: {message or 'Unknown error'}")

    def _apply_diff_overlay(self, trace: ToolTrace, *, document: DocumentState, range_hint: tuple[int, int]) -> None:
        tab_id = self._find_tab_id_for_document(document)
        if not tab_id:
            return
        metadata = trace.metadata if isinstance(trace.metadata, Mapping) else {}
        spans = self._coerce_overlay_spans(metadata.get("spans"), fallback_range=range_hint)
        diff_payload = metadata.get("diff_preview") if isinstance(metadata, Mapping) else None
        label = str(diff_payload or trace.output_summary or trace.name)
        try:
            self._editor.show_diff_overlay(
                label,
                spans=spans,
                summary=trace.output_summary,
                source=trace.name,
                tab_id=tab_id,
            )
        except Exception:  # pragma: no cover - defensive guard
            return
        self._tabs_with_overlay.add(tab_id)

    def _coerce_overlay_spans(
        self,
        raw_spans: Any,
        *,
        fallback_range: tuple[int, int] | None,
    ) -> tuple[tuple[int, int], ...]:
        spans: list[tuple[int, int]] = []
        if isinstance(raw_spans, Sequence):
            for entry in raw_spans:
                if not isinstance(entry, Sequence) or len(entry) != 2:
                    continue
                try:
                    start = int(entry[0])
                    end = int(entry[1])
                except (TypeError, ValueError):
                    continue
                if start == end:
                    continue
                if end < start:
                    start, end = end, start
                spans.append((start, end))
        if not spans and fallback_range is not None and fallback_range[0] != fallback_range[1]:
            start, end = fallback_range
            if end < start:
                start, end = end, start
            spans.append((start, end))
        return tuple(spans)

    def _maybe_clear_diff_overlay(self, state: DocumentState) -> None:
        tab_id = self._find_tab_id_for_document(state)
        if not tab_id or tab_id not in self._tabs_with_overlay:
            return
        try:
            tab = self._workspace.get_tab(tab_id)
        except KeyError:
            self._tabs_with_overlay.discard(tab_id)
            return
        change_source = getattr(tab.editor, "last_change_source", "")
        if change_source != "user":
            return
        self._clear_diff_overlay(tab_id=tab_id)

    def _clear_diff_overlay(self, *, tab_id: str | None = None) -> None:
        target_id = tab_id or self._workspace.active_tab_id
        if not target_id or target_id not in self._tabs_with_overlay:
            return
        try:
            self._editor.clear_diff_overlay(tab_id=target_id)
        except KeyError:  # pragma: no cover - tab already gone
            pass
        self._tabs_with_overlay.discard(target_id)

    def _find_tab_id_for_document(self, document: DocumentState) -> str | None:
        document_id = getattr(document, "document_id", None)
        for tab in self._workspace.iter_tabs():
            try:
                tab_document = tab.document()
            except Exception:  # pragma: no cover - defensive guard
                continue
            if document_id and tab_document.document_id == document_id:
                return tab.id
        return self._workspace.active_tab_id

    # ------------------------------------------------------------------
    # Action callbacks
    # ------------------------------------------------------------------
    def _handle_editor_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Cache the latest snapshot for future agent requests."""

        self._last_snapshot = snapshot
        digest = str(snapshot.get("outline_digest") or "").strip()
        document_id = str(snapshot.get("document_id") or "").strip()
        if not document_id:
            document = self._safe_active_document()
            if document is not None:
                document_id = document.document_id
        if not digest or not document_id:
            return
        if self._outline_digest_cache.get(document_id) == digest:
            return
        self._outline_digest_cache[document_id] = digest
        self._emit_outline_timeline_event(document_id, digest)

    def _handle_editor_text_changed(self, text: str, state: DocumentState) -> None:
        """Update window title when the editor content or metadata shifts."""

        self._refresh_window_title(state)
        self._refresh_chat_suggestions(state=state)
        if self._snapshot_persistence_block > 0:
            self._update_autosave_indicator(document=state)
            self._maybe_clear_diff_overlay(state)
            return
        self._persist_unsaved_snapshot(state)
        self._update_autosave_indicator(document=state)
        self._maybe_clear_diff_overlay(state)

    def _handle_editor_selection_changed(self, selection: SelectionRange) -> None:
        """Refresh suggestions and composer context when the selection moves."""

        self._refresh_chat_suggestions(selection=selection)

    def _handle_active_tab_changed(self, tab: DocumentTab | None) -> None:
        if tab is None:
            self._current_document_path = None
            self._sync_settings_workspace_state()
            return
        document = tab.document()
        self._current_document_path = document.metadata.path
        self._refresh_window_title(document)
        self._refresh_chat_suggestions(state=document)
        self._update_autosave_indicator(document=document)
        self._sync_settings_workspace_state()

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

    def _emit_outline_timeline_event(self, document_id: str, outline_digest: str) -> None:
        try:
            document = self._workspace.find_document_by_id(document_id)
        except Exception:
            document = None
        label = self._document_display_name(document) if document is not None else document_id
        trace = ToolTrace(
            name="document_outline",
            input_summary=label,
            output_summary="Outline updated",
            metadata={
                "document_id": document_id,
                "outline_digest": outline_digest,
                "timestamp": time.time(),
            },
        )
        try:
            self._chat_panel.show_tool_trace(trace)
        except Exception:  # pragma: no cover - UI optional in tests
            _LOGGER.debug("Unable to surface outline availability trace", exc_info=True)

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

    def _handle_import_requested(self) -> None:
        """Import a non-native file format by converting it to plain text."""

        try:
            path = self._prompt_for_import_path()
        except RuntimeError:
            self.update_status("Import dialog unavailable")
            return

        if path is None:
            self.update_status("Import canceled")
            return

        try:
            result = self._file_importer.import_file(path)
        except FileNotFoundError:
            self.update_status(f"File not found: {path}")
            return
        except ImporterError as exc:
            message = str(exc).strip() or f"Unable to import {path.name}"
            self.update_status(message)
            return

        self._open_import_result(result, source_path=path)

    def _handle_new_tab_requested(self) -> None:
        with self._suspend_snapshot_persistence():
            tab = self._editor.create_tab()
        self._workspace.set_active_tab(tab.id)
        self._current_document_path = None
        self._sync_settings_workspace_state()
        self._update_autosave_indicator(document=self._editor.to_document())
        self.update_status("New tab created")

    def _handle_close_tab_requested(self) -> None:
        active_tab = self._workspace.active_tab
        if active_tab is None:
            self.update_status("No tab to close")
            return
        closed = self._editor.close_active_tab()
        document = closed.document()
        self._clear_unsaved_snapshot(path=document.metadata.path, tab_id=closed.id)
        self._tabs_with_overlay.discard(closed.id)
        if self._workspace.tab_count() == 0:
            self._handle_new_tab_requested()
        else:
            new_active = self._workspace.active_tab
            if new_active is not None:
                self._current_document_path = new_active.document().metadata.path
        self._sync_settings_workspace_state()
        self._update_autosave_indicator(document=self._editor.to_document())
        self.update_status(f"Closed tab {closed.title}")

    def _handle_save_requested(self) -> None:
        """Save the current document, prompting for a path if required."""

        try:
            self.save_document()
        except RuntimeError:
            # Status updates are handled inside save_document / prompt helpers.
            _LOGGER.debug("Save request aborted; no path selected.")

    def _handle_save_as_requested(self) -> None:
        """Trigger a Save As workflow backed by the standard dialog."""

        path = self._prompt_for_save_path(document=self._editor.to_document())
        if path is None:
            self.update_status("Save canceled")
            return
        try:
            self.save_document(path)
        except RuntimeError:
            _LOGGER.debug("Save As aborted for path: %s", path)

    def _handle_revert_requested(self) -> None:
        """Reload the active document from disk, discarding unsaved edits."""

        path = self._resolve_active_document_path()
        if path is None:
            self.update_status("No file to revert")
            return

        try:
            text = file_io.read_text(path)
        except FileNotFoundError:
            self.update_status(f"File not found: {path}")
            return

        metadata = DocumentMetadata(path=path, language=self._infer_language(path))
        document = DocumentState(text=text, metadata=metadata)
        document.dirty = False

        with self._suspend_snapshot_persistence():
            self._editor.load_document(document)

        self._current_document_path = path
        self._clear_unsaved_snapshot(path=path)
        self._refresh_window_title(document)
        self.update_status(f"Reverted {path.name}")
        self._sync_settings_workspace_state()
        self._update_autosave_indicator(document=document)

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
    def editor_widget(self) -> TabbedEditorWidget:
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

        existing = self._workspace.find_tab_by_path(target)
        if existing is not None:
            self._workspace.set_active_tab(existing.id)
            self.update_status(f"Focused {target.name}")
            return

        text = file_io.read_text(target)
        metadata = DocumentMetadata(path=target, language=self._infer_language(target))
        document = DocumentState(text=text, metadata=metadata)
        document.dirty = False

        with self._suspend_snapshot_persistence():
            tab = self._editor.create_tab(document=document, path=target, title=target.name, make_active=True)
        self._workspace.set_active_tab(tab.id)
        self._current_document_path = target
        self._remember_recent_file(target)
        if self._apply_pending_snapshot_for_path(target):
            self._sync_settings_workspace_state()
            self._update_autosave_indicator(document=self._editor.to_document())
            return
        self.update_status(f"Loaded {target.name}")
        self._sync_settings_workspace_state()
        self._update_autosave_indicator(document=document)

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
            target_path = self._prompt_for_save_path(document=document)
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
        active_tab = self._workspace.active_tab
        if previous_path is None:
            self._clear_unsaved_snapshot(path=None, tab_id=active_tab.id if active_tab else None)
        if active_tab is not None:
            self._editor.refresh_tab_title(active_tab.id)
        self.update_status(f"Saved {target_path.name}")
        self._refresh_window_title(document)
        self._sync_settings_workspace_state()
        self._update_autosave_indicator(document=document)
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
        token_budget = None
        settings = self._context.settings
        if settings is not None:
            raw_budget = getattr(settings, "max_context_tokens", None)
            if isinstance(raw_budget, int):
                token_budget = raw_budget
        return open_file_dialog(parent=parent, start_dir=start_dir, token_budget=token_budget)

    def _prompt_for_import_path(self) -> Path | None:
        """Display the import dialog constrained to supported file types."""

        start_dir = self._resolve_open_start_dir(self._context.settings)
        try:
            from .widgets.dialogs import open_file_dialog
        except Exception as exc:  # pragma: no cover - depends on Qt availability
            raise RuntimeError(
                "File dialogs require the optional PySide6 dependency."
            ) from exc

        parent = self._qt_parent_widget()
        file_filter = self._file_importer.dialog_filter()
        token_budget = None
        settings = self._context.settings
        if settings is not None:
            raw_budget = getattr(settings, "max_context_tokens", None)
            if isinstance(raw_budget, int):
                token_budget = raw_budget
        return open_file_dialog(
            parent=parent,
            caption="Import File",
            start_dir=start_dir,
            file_filter=file_filter,
            token_budget=token_budget,
            enable_samples=False,
        )

    def _resolve_active_document_path(self) -> Path | None:
        document = self._editor.to_document()
        if document.metadata.path is not None:
            return Path(document.metadata.path)
        if self._current_document_path is not None:
            return Path(self._current_document_path)
        return None

    def _prompt_for_save_path(self, *, document: DocumentState | None = None) -> Path | None:
        """Show the save-file dialog and return the chosen path."""

        start_dir = self._resolve_save_start_dir(self._context.settings)
        try:
            from .widgets.dialogs import save_file_dialog
        except Exception as exc:  # pragma: no cover - depends on Qt availability
            raise RuntimeError(
                "File dialogs require the optional PySide6 dependency."
            ) from exc

        parent = self._qt_parent_widget()
        token_budget = None
        settings = self._context.settings
        if settings is not None:
            raw_budget = getattr(settings, "max_context_tokens", None)
            if isinstance(raw_budget, int):
                token_budget = raw_budget

        document_text: str | None = None
        selection_text: str | None = None
        if document is not None:
            document_text = document.text
            selection = document.selection
            if selection.end > selection.start:
                text = document.text
                start = max(0, min(len(text), selection.start))
                end = max(start, min(len(text), selection.end))
                selection_text = text[start:end]
        return save_file_dialog(
            parent=parent,
            start_dir=start_dir,
            document_text=document_text,
            selection_text=selection_text,
            token_budget=token_budget,
        )

    def _open_import_result(self, result: ImportResult, *, source_path: Path) -> None:
        """Create a new tab populated with imported text."""

        language = (result.language or "text").strip() or "text"
        document = DocumentState(text=result.text, metadata=DocumentMetadata(language=language))
        document.dirty = True
        title = (result.title or source_path.stem or "Imported Document").strip() or "Imported Document"

        with self._suspend_snapshot_persistence():
            tab = self._editor.create_tab(document=document, title=title, make_active=True)
        self._workspace.set_active_tab(tab.id)
        self._current_document_path = None
        self._remember_recent_file(source_path)
        self._refresh_window_title(document)
        self._sync_settings_workspace_state()
        self._update_autosave_indicator(document=document)
        status = f"Imported {source_path.name}"
        if result.notes:
            status = f"{status} – {result.notes}"
        self.update_status(status)

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

        if self._restore_workspace_tabs(settings):
            self._sync_settings_workspace_state(persist=False)
            return

        next_index = getattr(settings, "next_untitled_index", None)
        if isinstance(next_index, int):
            self._workspace.set_next_untitled_index(next_index)

        if self._try_restore_last_file(settings):
            self._sync_settings_workspace_state(persist=False)
            return

        self._restore_unsaved_snapshot(settings)
        self._sync_settings_workspace_state(persist=False)

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

    def _restore_unsaved_snapshot(self, settings: Settings) -> bool:
        snapshot = settings.unsaved_snapshot
        if not isinstance(snapshot, dict) or "text" not in snapshot:
            return False

        tab = self._workspace.active_tab or self._workspace.ensure_tab()
        self._load_snapshot_document(snapshot, path=None, tab_id=tab.id)
        self.update_status("Restored unsaved draft")
        return True

    def _restore_workspace_tabs(self, settings: Settings) -> bool:
        entries = [entry for entry in (settings.open_tabs or []) if isinstance(entry, dict)]
        if not entries:
            return False

        self._restoring_workspace = True
        self._tabs_with_overlay.clear()
        try:
            for tab_id in list(self._workspace.tab_ids()):
                try:
                    self._editor.close_tab(tab_id)
                except KeyError:  # pragma: no cover - defensive guard
                    continue

            restored_ids: list[str] = []
            for entry in entries:
                tab = self._create_tab_from_settings_entry(entry)
                if tab is not None:
                    restored_ids.append(tab.id)

            if not restored_ids:
                return False

            active_id = settings.active_tab_id or restored_ids[-1]
            if active_id not in self._workspace.tab_ids():
                active_id = restored_ids[-1]
            self._workspace.set_active_tab(active_id)
            next_index = getattr(settings, "next_untitled_index", None)
            if isinstance(next_index, int):
                self._workspace.set_next_untitled_index(next_index)
            return True
        finally:
            self._restoring_workspace = False

    def _create_tab_from_settings_entry(self, entry: Mapping[str, Any]) -> DocumentTab | None:
        title = str(entry.get("title") or UNTITLED_DOCUMENT_NAME)
        path_value = entry.get("path")
        path = Path(path_value).expanduser() if path_value else None
        language = str(entry.get("language") or (self._infer_language(path) if path else "markdown"))
        document = DocumentState(text="", metadata=DocumentMetadata(path=path, language=language))
        document.dirty = bool(entry.get("dirty", False))

        if path is not None:
            try:
                document.text = file_io.read_text(path)
            except Exception as exc:  # pragma: no cover - defensive logging
                _LOGGER.debug("Unable to restore %s from disk: %s", path, exc)
                document.text = ""

        raw_untitled_index = entry.get("untitled_index")
        untitled_index_value: int | None = None
        if raw_untitled_index is not None:
            try:
                untitled_index_value = int(raw_untitled_index)
            except (TypeError, ValueError):
                untitled_index_value = None

        with self._suspend_snapshot_persistence():
            tab = self._editor.create_tab(
                document=document,
                path=path,
                title=title,
                make_active=False,
                tab_id=entry.get("tab_id"),
                untitled_index=untitled_index_value,
            )

        if path is not None:
            self._apply_pending_snapshot_for_path(path, tab_id=tab.id, silent=True)
        else:
            self._apply_untitled_snapshot(tab.id)

        self._editor.refresh_tab_title(tab.id)
        return tab

    def _apply_pending_snapshot_for_path(
        self,
        path: Path | str,
        *,
        tab_id: str | None = None,
        silent: bool = False,
    ) -> bool:
        settings = self._context.settings
        if settings is None:
            return False

        normalized_key = self._snapshot_key(path)
        snapshot = (settings.unsaved_snapshots or {}).get(normalized_key)
        if not isinstance(snapshot, dict) or "text" not in snapshot:
            return False

        resolved_path = Path(normalized_key)
        self._load_snapshot_document(snapshot, path=resolved_path, tab_id=tab_id)
        if not silent:
            self.update_status(f"Restored unsaved changes for {resolved_path.name}")
        return True

    def _apply_untitled_snapshot(self, tab_id: str) -> bool:
        settings = self._context.settings
        if settings is None:
            return False

        snapshot = None
        if settings.untitled_snapshots:
            snapshot = (settings.untitled_snapshots or {}).get(tab_id)
        if snapshot is None:
            snapshot = settings.unsaved_snapshot
        if not isinstance(snapshot, dict) or "text" not in snapshot:
            return False
        self._load_snapshot_document(snapshot, path=None, tab_id=tab_id)
        return True

    def _load_snapshot_document(self, snapshot: dict[str, Any], *, path: Path | None, tab_id: str | None) -> None:
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
        with self._suspend_snapshot_persistence():
            self._editor.load_document(document, tab_id=tab_id)

        if tab_id is None or self._workspace.active_tab_id == tab_id:
            self._current_document_path = path
        digest_key = self._snapshot_key(path, tab_id=tab_id)
        self._unsaved_snapshot_digests[digest_key] = file_io.compute_text_digest(text)
        if tab_id is not None:
            self._editor.refresh_tab_title(tab_id)
        self._update_autosave_indicator(document=document)

    @contextmanager
    def _suspend_snapshot_persistence(self) -> Any:
        self._snapshot_persistence_block += 1
        try:
            yield
        finally:
            self._snapshot_persistence_block = max(0, self._snapshot_persistence_block - 1)

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
        active_tab = self._workspace.active_tab
        tab_id = active_tab.id if active_tab is not None else None
        key = self._snapshot_key(path, tab_id=tab_id)

        if not document.dirty:
            self._clear_unsaved_snapshot(settings=settings, path=path, tab_id=tab_id)
            return

        snapshot = {
            "text": document.text,
            "language": document.metadata.language,
            "selection": list(document.selection.as_tuple()),
        }
        digest = file_io.compute_text_digest(snapshot["text"])
        if self._unsaved_snapshot_digests.get(key) == digest:
            existing = self._get_snapshot_entry(settings, path=path, tab_id=tab_id)
            if existing == snapshot:
                return

        if path is None:
            if tab_id is not None:
                snapshots = dict(settings.untitled_snapshots or {})
                snapshots[tab_id] = snapshot
                settings.untitled_snapshots = snapshots
            else:
                settings.unsaved_snapshot = snapshot
        else:
            snapshots = dict(settings.unsaved_snapshots or {})
            snapshots[key] = snapshot
            settings.unsaved_snapshots = snapshots

        self._unsaved_snapshot_digests[key] = digest
        self._sync_settings_workspace_state(persist=False)
        self._persist_settings(settings)
        self._update_autosave_indicator(autosaved=True, document=document)

    def _clear_unsaved_snapshot(
        self,
        *,
        settings: Settings | None = None,
        path: Path | str | None = None,
        tab_id: str | None = None,
        persist: bool = True,
    ) -> None:
        key = self._snapshot_key(path, tab_id=tab_id)
        target_settings = settings or self._context.settings
        changed = False

        if target_settings is not None:
            if path is None and tab_id is not None:
                snapshots = dict(target_settings.untitled_snapshots or {})
                if snapshots.pop(tab_id, None) is not None:
                    target_settings.untitled_snapshots = snapshots
                    changed = True
            elif path is None and tab_id is None:
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

    def _get_snapshot_entry(
        self,
        settings: Settings,
        *,
        path: Path | str | None,
        tab_id: str | None,
    ) -> dict[str, Any] | None:
        if path is None:
            if tab_id is not None:
                return (settings.untitled_snapshots or {}).get(tab_id)
            return settings.unsaved_snapshot
        key = self._snapshot_key(path)
        return (settings.unsaved_snapshots or {}).get(key)

    def _snapshot_key(self, path: Path | str | None, *, tab_id: str | None = None) -> str:
        if path is None:
            if tab_id:
                return f"{self._UNTITLED_SNAPSHOT_KEY}:{tab_id}"
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
            module = importlib.import_module(".widgets.dialogs", __package__)
        except Exception as exc:  # pragma: no cover - depends on Qt availability
            raise RuntimeError("Settings dialog requires the PySide6 dependency.") from exc

        show_settings_dialog = getattr(module, "show_settings_dialog", None)
        if not callable(show_settings_dialog):  # pragma: no cover - defensive guard
            raise RuntimeError("Settings dialog is unavailable.")

        parent = self._qt_parent_widget()
        result = show_settings_dialog(parent=parent, settings=settings)
        return cast("SettingsDialogResult", result)

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

    def _sync_settings_workspace_state(self, *, persist: bool = True) -> None:
        if self._restoring_workspace:
            return
        settings = self._context.settings
        if settings is None:
            return

        state = self._workspace.serialize_state()
        settings.open_tabs = state.get("open_tabs", [])
        settings.active_tab_id = state.get("active_tab_id")
        untitled_counter = state.get("untitled_counter")
        if isinstance(untitled_counter, int):
            settings.next_untitled_index = untitled_counter
        if persist:
            self._persist_settings(settings)

    def _refresh_embedding_runtime(self, settings: Settings | None) -> None:
        if settings is None or not self._phase3_outline_enabled:
            self._teardown_embedding_runtime(reason="Phase 3 tools disabled", hide_status=not self._phase3_outline_enabled)
            return

        backend = (getattr(settings, "embedding_backend", "auto") or "auto").strip().lower()
        if backend in {"", "auto"}:
            backend = "openai"
        if backend == "disabled":
            self._teardown_embedding_runtime(reason="Embeddings disabled in settings")
            return

        model_name = (getattr(settings, "embedding_model_name", "") or "").strip() or "text-embedding-3-large"
        signature = self._embedding_settings_signature(settings, backend, model_name)
        if signature == self._embedding_signature and self._embedding_index is not None:
            self._set_embedding_state(self._embedding_state)
            return

        self._teardown_embedding_runtime(reason="Reinitializing embedding backend", keep_status=True)

        provider, provider_state = self._build_embedding_provider(backend, model_name, settings)
        if provider is None:
            self._set_embedding_state(provider_state)
            return

        cache_root = self._resolve_embedding_cache_root()
        index = self._create_embedding_index(provider, storage_dir=cache_root)
        if index is None:
            error_state = EmbeddingRuntimeState(
                backend=backend,
                model=model_name,
                provider_label=provider_state.provider_label,
                status="error",
                error=f"Unable to initialize embedding cache under {cache_root}",
            )
            self._set_embedding_state(error_state)
            return

        self._embedding_index = index
        self._embedding_signature = signature
        self._set_embedding_state(provider_state)
        self._propagate_embedding_index_to_worker()

    def _embedding_settings_signature(self, settings: Settings, backend: str, model: str) -> tuple[Any, ...]:
        metadata = getattr(settings, "metadata", {}) or {}
        class_override = metadata.get("langchain_embeddings_class")
        kwargs_payload = metadata.get("langchain_embeddings_kwargs")
        provider_hint = metadata.get("langchain_provider_family")
        env_class = os.environ.get("TINKERBELL_LANGCHAIN_EMBEDDINGS_CLASS")
        env_kwargs = os.environ.get("TINKERBELL_LANGCHAIN_EMBEDDINGS_KWARGS")
        env_provider_family = os.environ.get("TINKERBELL_LANGCHAIN_PROVIDER_FAMILY")
        provider_env_secrets = tuple(os.environ.get(var) for var in _LANGCHAIN_PROVIDER_ENV_VARS)
        return (
            backend,
            model,
            getattr(settings, "base_url", None),
            getattr(settings, "api_key", None),
            getattr(settings, "organization", None),
            class_override,
            kwargs_payload,
            env_class,
            env_kwargs,
            provider_hint,
            env_provider_family,
            provider_env_secrets,
        )

    def _build_embedding_provider(
        self,
        backend: str,
        model_name: str,
        settings: Settings,
    ) -> tuple[EmbeddingProvider | None, EmbeddingRuntimeState]:
        if backend == "openai":
            return self._build_openai_embedding_provider(model_name, settings)
        if backend == "langchain":
            return self._build_langchain_embedding_provider(model_name, settings)
        state = EmbeddingRuntimeState(backend=backend, model=model_name, status="error", error=f"Unknown backend '{backend}'")
        return None, state

    def _build_openai_embedding_provider(
        self,
        model_name: str,
        settings: Settings,
    ) -> tuple[EmbeddingProvider | None, EmbeddingRuntimeState]:
        state = EmbeddingRuntimeState(backend="openai", model=model_name, provider_label="OpenAI", status="error")
        api_key = (getattr(settings, "api_key", "") or "").strip()
        base_url = (getattr(settings, "base_url", "") or "").strip() or OPENAI_API_BASE_URL
        if not api_key:
            state.error = "API key required for OpenAI embeddings"
            return None, state
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            state.error = f"openai package unavailable: {exc}"
            return None, state

        try:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                organization=getattr(settings, "organization", None),
                timeout=getattr(settings, "request_timeout", 90.0),
            )
            provider = OpenAIEmbeddingProvider(client=client, model=model_name)
        except Exception as exc:
            state.error = f"Failed to initialize OpenAI embeddings: {exc}"
            return None, state

        state.status = "ready"
        state.detail = model_name
        self._embedding_resource = client
        return provider, state

    def _build_langchain_embedding_provider(
        self,
        model_name: str,
        settings: Settings,
    ) -> tuple[EmbeddingProvider | None, EmbeddingRuntimeState]:
        state = EmbeddingRuntimeState(backend="langchain", model=model_name, provider_label="LangChain")
        try:
            embeddings, provider_template = self._instantiate_langchain_embeddings(model_name, settings)
        except Exception as exc:
            state.status = "error"
            state.error = str(exc)
            return None, state
        provider = LangChainEmbeddingProvider(embeddings=embeddings)
        label = getattr(embeddings, "__class__", type(embeddings)).__name__
        template_label = provider_template.family.title() if provider_template else label
        state.provider_label = f"LangChain/{template_label}"
        model_detail = getattr(embeddings, "model", None) or getattr(embeddings, "model_name", None)
        state.detail = str(model_detail or model_name)
        state.status = "ready"
        return provider, state

    def _instantiate_langchain_embeddings(self, model_name: str, settings: Settings) -> tuple[Any, _LangChainProviderTemplate]:
        metadata = getattr(settings, "metadata", {}) or {}
        class_override = metadata.get("langchain_embeddings_class") or os.environ.get("TINKERBELL_LANGCHAIN_EMBEDDINGS_CLASS")
        raw_kwargs = metadata.get("langchain_embeddings_kwargs") or os.environ.get("TINKERBELL_LANGCHAIN_EMBEDDINGS_KWARGS")
        extra_kwargs = self._parse_langchain_kwargs(raw_kwargs)
        provider_template = self._detect_langchain_provider_template(model_name, metadata)
        params = dict(extra_kwargs)
        params.setdefault("model", model_name)
        self._apply_langchain_provider_template(params, provider_template, model_name, settings, metadata)
        if class_override:
            module_name, _, attr_name = class_override.rpartition(".")
            if not module_name or not attr_name:
                raise RuntimeError("LangChain embeddings class override must include module path, e.g., 'langchain_openai.OpenAIEmbeddings'")
            module = importlib.import_module(module_name)
            factory = getattr(module, attr_name)
            embeddings = factory(**params)
            return embeddings, provider_template
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "langchain-openai package is required for the LangChain embedding backend. "
                "Install it or set metadata['langchain_embeddings_class'] to a custom implementation."
            ) from exc

        embeddings = OpenAIEmbeddings(**params)
        return embeddings, provider_template

    def _detect_langchain_provider_template(
        self,
        model_name: str,
        metadata: Mapping[str, Any],
    ) -> _LangChainProviderTemplate:
        forced = str(
            metadata.get("langchain_provider_family")
            or os.environ.get("TINKERBELL_LANGCHAIN_PROVIDER_FAMILY")
            or ""
        ).strip().lower()
        forced_family = forced or None
        for template in _LANGCHAIN_PROVIDER_TEMPLATES:
            if template.matches(model_name, forced_family=forced_family):
                return template
        return _DEFAULT_LANGCHAIN_PROVIDER

    def _apply_langchain_provider_template(
        self,
        params: dict[str, Any],
        template: _LangChainProviderTemplate,
        model_name: str,
        settings: Settings,
        metadata: Mapping[str, Any],
    ) -> None:
        metadata_base_override = str(metadata.get(f"{template.family}_base_url") or "").strip()
        configured_base_url = metadata_base_override or (getattr(settings, "base_url", "") or "").strip()
        if template.family != "openai" and template.default_base_url:
            if self._should_override_base_url(configured_base_url, template.default_base_url):
                params.setdefault("base_url", template.default_base_url)
            elif configured_base_url:
                params.setdefault("base_url", configured_base_url)
            else:
                params.setdefault("base_url", template.default_base_url)
        else:
            if configured_base_url:
                params.setdefault("base_url", configured_base_url)
            elif template.default_base_url:
                params.setdefault("base_url", template.default_base_url)

        metadata_key = metadata.get(f"{template.family}_api_key")
        api_key = str(metadata_key).strip() if isinstance(metadata_key, str) else ""
        if not api_key:
            api_key = (getattr(settings, "api_key", "") or "").strip()
        if not api_key and template.api_key_env:
            api_key = (os.environ.get(template.api_key_env, "") or "").strip()
        if api_key:
            params.setdefault("api_key", api_key)

        organization = getattr(settings, "organization", None)
        if organization:
            params.setdefault("organization", organization)

        headers: dict[str, str] = {}
        if template.default_headers:
            headers.update(template.default_headers)
        user_headers = getattr(settings, "default_headers", None) or {}
        headers.update({str(k): str(v) for k, v in user_headers.items()})
        if headers:
            params.setdefault("default_headers", headers)

        if template.tokenizer:
            params.setdefault("tiktoken_model_name", template.tokenizer)

        dimension_override = template.dimension_for(model_name)
        if dimension_override:
            params.setdefault("dimensions", dimension_override)

        for key, value in (template.extra_kwargs or {}).items():
            params.setdefault(key, value)

    @staticmethod
    def _should_override_base_url(current: str, candidate: str) -> bool:
        normalized_current = (current or "").strip().lower()
        normalized_candidate = (candidate or "").strip().lower()
        if not normalized_candidate:
            return False
        if not normalized_current:
            return True
        openai_default = OPENAI_API_BASE_URL.lower()
        return normalized_current == openai_default and normalized_candidate != openai_default

    @staticmethod
    def _parse_langchain_kwargs(payload: object) -> dict[str, Any]:
        if not payload:
            return {}
        if isinstance(payload, Mapping):
            return dict(payload)
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return {}
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}
        return {}

    def _create_embedding_index(self, provider: EmbeddingProvider, *, storage_dir: Path) -> DocumentEmbeddingIndex | None:
        try:
            loop = self._resolve_async_loop()
            if loop is not None:
                return DocumentEmbeddingIndex(storage_dir=storage_dir, provider=provider, loop=loop)
            return DocumentEmbeddingIndex(storage_dir=storage_dir, provider=provider)
        except Exception as exc:
            _LOGGER.warning("Failed to initialize embedding index: %s", exc)
            return None

    def _teardown_embedding_runtime(
        self,
        *,
        reason: str,
        hide_status: bool = False,
        keep_status: bool = False,
    ) -> None:
        if self._embedding_index is not None:
            self._close_embedding_index(self._embedding_index)
        self._embedding_index = None
        self._embedding_signature = None
        self._embedding_snapshot_metadata = {}
        self._propagate_embedding_index_to_worker()
        self._dispose_embedding_resource()
        if keep_status:
            return
        fallback_backend = "disabled" if self._phase3_outline_enabled else "unavailable"
        state = EmbeddingRuntimeState(
            backend=fallback_backend,
            model=None,
            status=fallback_backend if fallback_backend != "unavailable" else "unavailable",
            detail=reason,
        )
        self._set_embedding_state(state, hide_status=hide_status)

    def _close_embedding_index(self, index: DocumentEmbeddingIndex) -> None:
        async_close = cast(Callable[[], Coroutine[Any, Any, Any]], index.aclose)
        self._run_background_task(async_close)

    def _dispose_embedding_resource(self) -> None:
        resource = self._embedding_resource
        self._embedding_resource = None
        if resource is None:
            return
        close = getattr(resource, "aclose", None)
        if callable(close):
            try:
                async_close = cast(Callable[[], Coroutine[Any, Any, Any]], close)
                self._run_background_task(async_close)
            except Exception:
                _LOGGER.debug("Failed to dispose async embedding resource", exc_info=True)
            return
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _set_embedding_state(self, state: EmbeddingRuntimeState, *, hide_status: bool = False) -> None:
        self._embedding_state = state
        snapshot = {k: v for k, v in state.as_snapshot().items() if v not in (None, "")}
        self._embedding_snapshot_metadata = snapshot
        label = "" if hide_status else state.label
        detail = state.detail or state.error or ""
        try:
            self._status_bar.set_embedding_status(label, detail=detail)
        except Exception:  # pragma: no cover - defensive guard
            pass

    def _propagate_embedding_index_to_worker(self) -> None:
        worker = getattr(self, "_outline_worker", None)
        if worker is None:
            return
        updater = getattr(worker, "update_embedding_index", None)
        if callable(updater):
            try:
                updater(self._embedding_index)
            except Exception:
                _LOGGER.debug("Failed to propagate embedding index to worker", exc_info=True)

    def _apply_embedding_metadata(self, snapshot: dict[str, Any]) -> None:
        if not self._embedding_snapshot_metadata:
            for field in ("embedding_backend", "embedding_model", "embedding_status", "embedding_detail"):
                snapshot.pop(field, None)
            return
        snapshot.update(self._embedding_snapshot_metadata)

    def _resolve_embedding_cache_root(self) -> Path:
        return self._resolve_outline_cache_root().parent

    def _apply_runtime_settings(self, settings: Settings) -> None:
        self._apply_chat_panel_settings(settings)
        self._apply_phase3_outline_setting(settings)
        self._apply_debug_logging_setting(settings)
        self._apply_theme_setting(settings)
        self._refresh_embedding_runtime(settings)
        self._refresh_ai_runtime(settings)

    def _apply_phase3_outline_setting(self, settings: Settings) -> None:
        enabled = bool(getattr(settings, "phase3_outline_tools", False))
        if enabled == self._phase3_outline_enabled:
            return

        self._phase3_outline_enabled = enabled
        if enabled:
            self._outline_worker = self._outline_worker or self._create_outline_worker()
            if self._status_bar is not None:
                self._status_bar.set_outline_status("")
            self._register_phase3_ai_tools()
            return

        self._shutdown_outline_worker()
        self._outline_worker = None
        self._outline_tool = None
        self._find_sections_tool = None
        self._outline_digest_cache.clear()
        if self._status_bar is not None:
            self._status_bar.set_outline_status("")
        self._unregister_phase3_ai_tools()

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

    def _apply_max_tool_iterations(self, controller: Any, limit: int) -> None:
        setter = getattr(controller, "set_max_tool_iterations", None)
        if not callable(setter):
            return
        try:
            setter(limit)
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Unable to update max tool iterations: %s", exc)

    def _apply_context_window_settings(self, controller: Any, settings: Settings) -> None:
        configurator = getattr(controller, "configure_context_window", None)
        if not callable(configurator):
            return
        try:
            configurator(
                max_context_tokens=getattr(settings, "max_context_tokens", 128_000),
                response_token_reserve=getattr(settings, "response_token_reserve", 16_000),
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Unable to update context window settings: %s", exc)

    def _apply_context_policy_settings(self, controller: Any, settings: Settings) -> None:
        builder = getattr(self, "_build_context_budget_policy", None)
        if not callable(builder):
            return
        policy = builder(settings)
        configurator = getattr(controller, "configure_budget_policy", None)
        if callable(configurator):
            try:
                configurator(policy)
            except Exception as exc:  # pragma: no cover - defensive guard
                _LOGGER.debug("Unable to update context budget policy: %s", exc)
        outline_tool = getattr(self, "_outline_tool", None)
        if outline_tool is not None:
            outline_tool.budget_policy = policy

    def _build_context_budget_policy(self, settings: Settings):
        try:
            from .ai.services.context_policy import ContextBudgetPolicy
        except Exception as exc:  # pragma: no cover - dependency guard
            _LOGGER.debug("Context budget policy unavailable: %s", exc)
            return None

        policy_settings = getattr(settings, "context_policy", None)
        max_context = getattr(settings, "max_context_tokens", 128_000)
        reserve = getattr(settings, "response_token_reserve", 16_000)
        model_name = getattr(settings, "model", None)
        return ContextBudgetPolicy.from_settings(
            policy_settings,
            model_name=model_name,
            max_context_tokens=max_context,
            response_token_reserve=reserve,
        )

    @staticmethod
    def _resolve_max_tool_iterations(settings: Settings | None) -> int:
        raw = getattr(settings, "max_tool_iterations", 8) if settings else 8
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 8
        return max(1, min(value, 50))

    def _apply_debug_logging_setting(self, settings: Settings) -> None:
        new_debug = bool(getattr(settings, "debug_logging", False))
        if new_debug != self._debug_logging_enabled:
            self._update_logging_configuration(new_debug)
            self._debug_logging_enabled = new_debug
        self._update_ai_debug_logging(new_debug)

    def _apply_theme_setting(self, settings: Settings) -> None:
        requested_name = (getattr(settings, "theme", "") or "default").strip() or "default"
        normalized_request = requested_name.lower()
        if normalized_request == self._active_theme_request:
            return
        self._active_theme_request = normalized_request
        theme = load_theme(requested_name)
        self._active_theme = theme.name
        try:
            self._editor.apply_theme(theme)
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Unable to apply editor theme %s: %s", theme.name, exc)
        theme_manager.apply_to_application(theme)

    def _refresh_ai_runtime(self, settings: Settings) -> None:
        if not self._ai_settings_ready(settings):
            self._disable_ai_controller()
            return

        signature = self._ai_settings_signature(settings)
        controller = self._context.ai_controller
        iteration_limit = self._resolve_max_tool_iterations(settings)

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

        if controller is not None:
            self._apply_max_tool_iterations(controller, iteration_limit)
            self._apply_context_window_settings(controller, settings)
            self._apply_context_policy_settings(controller, settings)

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
            limit = self._resolve_max_tool_iterations(settings)
            policy = self._build_context_budget_policy(settings)
            return AIController(
                client=client,
                max_tool_iterations=limit,
                max_context_tokens=getattr(settings, "max_context_tokens", 128_000),
                response_token_reserve=getattr(settings, "response_token_reserve", 16_000),
                budget_policy=policy,
            )
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

