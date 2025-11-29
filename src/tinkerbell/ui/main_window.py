"""Main window implementation coordinating the editor and chat panes."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Dict, Iterable, Mapping, Optional, Sequence, TYPE_CHECKING, cast

from ..ai.analysis import AnalysisAdvice
from ..chat.chat_panel import ChatPanel, ChatTurnSnapshot
from ..chat.commands import (
    ActionType,
    ManualCommandRequest,
    ManualCommandType,
    parse_manual_command,
    resolve_tab_reference,
)
from ..chat.message_model import ChatMessage, EditDirective, ToolTrace
from ..editor.document_model import DocumentMetadata, DocumentState
from ..editor.selection_gateway import SelectionGateway
from ..editor.workspace import DocumentTab
from ..documents.ranges import TextRange
from ..editor.tabbed_editor import TabbedEditorWidget
from ..ai.memory.buffers import DocumentSummaryMemory
from ..services.bridge import DocumentBridge
from ..services.bridge_router import WorkspaceBridgeRouter
from ..services.importers import FileImporter
from ..services.settings import Settings, SettingsStore
from ..utils import file_io
from ..services import telemetry as telemetry_service
from ..widgets.status_bar import StatusBar
from ..ai.tools.tool_wiring import (
    ToolWiringContext,
    register_new_tools,
)
from ..ai.tools.tool_registry import get_tool_registry
from ..ai.orchestration.tool_dispatcher import DispatchListener, DispatchResult
from .settings_runtime import SettingsRuntime
from ..ai.orchestration.editor_lock import (
    EditorLockManager,
    LockableTab,
    LockState,
    LockStatusUpdater,
    TabProvider,
)
from .ai_review_controller import AIReviewController, EditSummary, PendingReviewSession, PendingTurnReview
from .ai_turn_coordinator import AITurnCoordinator
from .document_session_service import DocumentSessionService
from .document_state_monitor import DocumentStateMonitor
from .document_status_service import DocumentStatusService
from .embedding_controller import EmbeddingController, EmbeddingRuntimeState
from .import_controller import ImportController
from .main_window_helpers import (
    WriteToolDispatchListener,
    WorkspaceTabProvider,
    # Pure utility functions
    condense_whitespace,
    line_column_from_offset,
    coerce_stream_text,
    infer_language,
    directive_parameters_schema,
    serialize_chat_history,
    history_signature,
)
from .models.actions import MenuSpec, ToolbarSpec, WindowAction
from .models.window_state import OutlineStatusInfo, WindowContext
from .review_overlay_manager import ReviewOverlayManager
from .outline_runtime import OutlineRuntime
from .telemetry_controller import TelemetryController
from .tool_trace_presenter import ToolTracePresenter
from .tools.provider import ToolProvider
from .window_shell import WindowChrome
from .widgets import CommandPaletteDialog, DocumentStatusWindow, PaletteCommand, build_palette_commands

if TYPE_CHECKING:  # pragma: no cover - import only for type hints
    from ..ai.orchestration import AIOrchestrator
    from ..ai.memory import DocumentEmbeddingIndex
    from ..ai.memory.plot_state import DocumentPlotStateStore
    from ..ai.memory.character_map import CharacterMapStore
    from ..widgets.dialogs import SettingsDialogResult

QApplication: Any
QMainWindow: Any
QWidget: Any

try:  # pragma: no cover - PySide6 optional in CI
    from PySide6.QtWidgets import QApplication as _QtQApplication, QMainWindow as _QtQMainWindow, QWidget as _QtQWidget

    QMainWindow = _QtQMainWindow
    QWidget = _QtQWidget
    QApplication = _QtQApplication
except Exception:  # pragma: no cover - runtime stubs keep tests headless

    class _StubQMainWindow:  # type: ignore[misc]
        """Fallback placeholder when PySide6 is unavailable."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs


_LOGGER = logging.getLogger(__name__)

WINDOW_APP_NAME = "TinkerBell"
UNTITLED_DOCUMENT_NAME = "Untitled"
SUGGESTION_LOADING_LABEL = "Generating personalized suggestions…"
OPENAI_API_BASE_URL = "https://api.openai.com/v1"

# Import helper classes from extracted module
from .main_window_helpers import (
    WriteToolDispatchListener as _WriteToolDispatchListener,
    EditorTabWrapper as _EditorTabWrapper,
    WorkspaceTabProvider as _WorkspaceTabProvider,
)


class MainWindow(QMainWindow):
    """Primary application window hosting the editor and chat splitter."""

    _WINDOW_BASE_TITLE = WINDOW_APP_NAME
    _UNTITLED_SNAPSHOT_KEY = "__untitled__"

    def __init__(self, context: WindowContext):  # noqa: D401 - doc inherited
        super().__init__()
        self._context = context
        self._async_loop: asyncio.AbstractEventLoop | None = None
        initial_settings = context.settings
        show_tool_panel = bool(getattr(initial_settings, "show_tool_activity_panel", False))
        self._editor = TabbedEditorWidget(skip_default_tab=True)
        self._editor.set_tab_close_handler(self._handle_tab_close_request)
        self._workspace = self._editor.workspace
        self._chat_panel = ChatPanel(show_tool_activity_panel=show_tool_panel)
        self._selection_gateway = SelectionGateway(workspace=self._workspace)
        self._bridge = WorkspaceBridgeRouter(self._workspace)
        self._editor.add_tab_created_listener(self._bridge.track_tab)
        self._status_bar = StatusBar()
        self._status_bar.set_document_status_callback(self._handle_document_status_clicked)
        self._document_monitor: DocumentStateMonitor | None = None
        self._document_session = DocumentSessionService(
            context=self._context,
            editor=self._editor,
            workspace=self._workspace,
            document_monitor_resolver=lambda: self._document_monitor,
            open_document=lambda path: self.open_document(path),
            status_updater=self.update_status,
            qt_parent_provider=self._qt_parent_widget,
            untitled_document_name=UNTITLED_DOCUMENT_NAME,
            untitled_snapshot_key=self._UNTITLED_SNAPSHOT_KEY,
        )
        self._review_overlay_manager: ReviewOverlayManager | None = None
        self._document_monitor = DocumentStateMonitor(
            editor=self._editor,
            workspace=self._workspace,
            chat_panel=self._chat_panel,
            status_bar=self._status_bar,
            unsaved_cache_provider=lambda: self._context.unsaved_cache,
            unsaved_cache_persister=self._document_session.persist_unsaved_cache,
            refresh_window_title=self._refresh_window_title,
            sync_workspace_state=lambda persist: self._document_session.sync_workspace_state(persist=persist),
            current_path_getter=self._document_session.get_current_path,
            current_path_setter=self._document_session.set_current_path,
            last_snapshot_setter=lambda snapshot: setattr(self, "_last_snapshot", snapshot),
            active_document_provider=self._safe_active_document,
            maybe_clear_diff_overlay=self._maybe_clear_diff_overlay,
            window_app_name=WINDOW_APP_NAME,
            untitled_document_name=UNTITLED_DOCUMENT_NAME,
            untitled_snapshot_key=self._UNTITLED_SNAPSHOT_KEY,
        )
        self._workspace.add_active_listener(self._document_monitor.handle_active_tab_changed)
        self._workspace.add_active_listener(self._handle_active_tab_for_review)
        self._workspace.add_active_listener(self._handle_active_tab_for_document_status)
        self._workspace.add_active_listener(self._handle_active_tab_for_cursor)
        self._telemetry_controller = TelemetryController(
            status_bar=self._status_bar,
            context=self._context,
            chat_panel=self._chat_panel,
        )
        self._review_controller = AIReviewController(
            status_bar=self._status_bar,
            chat_panel=self._chat_panel,
            workspace=self._workspace,
            clear_diff_overlay=lambda tab_id=None: self._clear_diff_overlay(tab_id=tab_id),
            update_status=self.update_status,
            post_assistant_notice=self._post_assistant_notice,
            accept_callback=self._handle_accept_ai_changes,
            reject_callback=self._handle_reject_ai_changes,
        )
        self._document_session.set_review_controller(self._review_controller)
        self._review_overlay_manager = ReviewOverlayManager(
            editor=self._editor,
            workspace=self._workspace,
            review_controller=self._review_controller,
            chat_panel=self._chat_panel,
            status_updater=self.update_status,
            notice_poster=self._post_assistant_notice,
            window_title_refresher=self._refresh_window_title,
            autosave_updater=lambda document: self._document_monitor.update_autosave_indicator(document=document),
            sync_workspace_state=lambda: self._document_session.sync_workspace_state(),
        )
        self._document_session.set_review_overlay_manager(self._review_overlay_manager)
        file_importer = FileImporter()
        self._import_controller = ImportController(
            file_importer=file_importer,
            prompt_for_path=lambda: self._prompt_for_import_path(),
            new_tab_factory=self._create_import_tab,
            status_updater=self.update_status,
            remember_recent_file=self._document_session.remember_recent_file,
            refresh_window_title=self._refresh_window_title,
            sync_workspace_state=lambda: self._document_session.sync_workspace_state(),
            update_autosave_indicator=lambda document: self._document_monitor.update_autosave_indicator(document=document),
        )
        self._document_session.set_import_dialog_filter_provider(lambda: self._import_controller.dialog_filter())
        self._splitter: Any = None
        self._actions: Dict[str, WindowAction] = {}
        self._menus: Dict[str, MenuSpec] = {}
        self._toolbars: Dict[str, ToolbarSpec] = {}
        self._qt_actions: Dict[str, Any] = {}
        self._last_snapshot: dict[str, Any] = {}
        self._last_status_message: str = ""
        self._ai_task: asyncio.Task[Any] | None = None
        self._ai_stream_active = False
        self._pending_turn_snapshot: ChatTurnSnapshot | None = None
        self._tool_trace_index: Dict[str, ToolTrace] = {}
        self._tool_trace_presenter = ToolTracePresenter(
            chat_panel=self._chat_panel,
            tool_trace_index=self._tool_trace_index,
        )
        self._document_status_service: DocumentStatusService | None = None
        self._document_status_window: DocumentStatusWindow | None = None
        self._command_palette: CommandPaletteDialog | None = None
        self._document_status_events_registered = False
        self._suggestion_task: asyncio.Task[Any] | None = None
        self._suggestion_request_id = 0
        self._suggestion_cache_key: str | None = None
        self._suggestion_cache_values: tuple[str, ...] | None = None
        self._outline_status_by_document: dict[str, OutlineStatusInfo] = {}
        self._suppress_cancel_abort = False
        self._outline_runtime: OutlineRuntime | None = None
        self._tool_provider: ToolProvider | None = None
        self._embedding_controller = EmbeddingController(
            status_bar=self._status_bar,
            cache_root_resolver=self._resolve_embedding_cache_root,
            outline_worker_resolver=lambda: self._outline_runtime.worker() if self._outline_runtime else None,
            async_loop_resolver=self._resolve_async_loop,
            background_task_runner=self._run_background_task,
        )
        self._outline_runtime = OutlineRuntime(
            document_provider=self._workspace.find_document_by_id,
            storage_root=self._resolve_outline_cache_root(),
            loop_resolver=self._resolve_async_loop,
            index_propagator=lambda: self._embedding_controller.propagate_index_to_worker(),
        )
        self._tool_provider = ToolProvider(
            controller_resolver=lambda: self._context.ai_orchestrator,
            bridge=self._bridge,
            workspace=self._workspace,
            selection_gateway=self._selection_gateway,
            editor=self._editor,  # For Qt-aware tab creation
        )
        # Initialize editor lock manager for AI turn coordination
        self._editor_lock_manager = EditorLockManager(
            tab_provider=_WorkspaceTabProvider(self._workspace),
            status_updater=self._update_lock_status,
        )
        self._ai_turn_coordinator = AITurnCoordinator(
            controller_resolver=lambda: self._context.ai_orchestrator,
            chat_panel=self._chat_panel,
            review_controller=self._review_controller,
            telemetry_controller=self._telemetry_controller,
            status_updater=self.update_status,
            failure_handler=self._handle_ai_failure,
            response_finalizer=self._finalize_ai_response,
            tool_trace_presenter=self._tool_trace_presenter,
            stream_state_setter=self._set_ai_stream_active,
            stream_text_coercer=coerce_stream_text,
            lock_manager=self._editor_lock_manager,
        )
        self._settings_runtime = SettingsRuntime(
            context=self._context,
            editor=self._editor,
            telemetry_controller=self._telemetry_controller,
            embedding_controller=self._embedding_controller,
            register_default_ai_tools=self._register_default_ai_tools,
            ai_task_getter=lambda: self._ai_task,
            ai_task_setter=lambda task: setattr(self, "_ai_task", task),
            ai_stream_state_setter=self._set_ai_stream_active,
            initial_settings=initial_settings,
        )
        self._last_outline_status: tuple[str, str] | None = None
        self._initialize_ui()
        self._telemetry_controller.register_subagent_listeners()
        self._telemetry_controller.register_chunk_flow_listeners()
        self._telemetry_controller.update_subagent_indicator()
        self._telemetry_controller.reset_chunk_flow_state()
        self._register_document_status_listeners()
        self._handle_active_tab_for_cursor(self._workspace.active_tab)
        if initial_settings is not None:
            self._settings_runtime.apply_theme_setting(initial_settings)
        self._embedding_controller.refresh_runtime(initial_settings)
        # Start outline runtime by default (Phase 3 tools are always enabled now)
        if self._outline_runtime is not None:
            self._outline_runtime.ensure_started()
            self._embedding_controller.propagate_index_to_worker()

    # ------------------------------------------------------------------
    # Qt lifecycle hooks
    # ------------------------------------------------------------------
    def closeEvent(self, event: Any) -> None:  # noqa: N802 - Qt naming
        """Ensure background tasks are canceled before the window closes."""

        self._cancel_active_ai_turn()
        self._cancel_dynamic_suggestions()
        self._clear_suggestion_cache()
        if self._outline_runtime is not None:
            self._outline_runtime.shutdown()
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

        self._ensure_asyncio_loop_stops()

    def _ensure_asyncio_loop_stops(self) -> None:
        """Force the qasync loop to receive a stop signal if Qt skips it."""

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        is_running = getattr(loop, "is_running", None)
        if not callable(is_running) or not is_running():
            return

        stopper = getattr(loop, "stop", None)
        if not callable(stopper):
            return

        try:
            loop.call_soon(stopper)
        except RuntimeError:
            try:
                stopper()
            except Exception:  # pragma: no cover - defensive guard
                _LOGGER.debug("loop.stop() failed during shutdown", exc_info=True)

    # ------------------------------------------------------------------
    # UI setup helpers
    # ------------------------------------------------------------------
    def _initialize_ui(self) -> None:
        """Set up menus, toolbars, splitter layout, and status widgets."""

        self._refresh_window_title()
        chrome = WindowChrome(
            window=self,
            editor=self._editor,
            chat_panel=self._chat_panel,
            status_bar=self._status_bar,
            action_callbacks=self._action_callbacks(),
        )
        chrome_state = chrome.assemble()
        self._splitter = chrome_state.splitter
        self._actions = chrome_state.actions
        self._menus = chrome_state.menus
        self._toolbars = chrome_state.toolbars
        self._qt_actions = chrome_state.qt_actions
        self._initialize_command_palette()
        self._wire_signals()
        self._register_default_ai_tools()

        self.update_status("Ready")
        self._document_session.restore_last_session_document()
        if self._workspace.active_tab is not None:
            self._document_monitor.update_autosave_indicator(document=self._editor.to_document())
        self._refresh_document_status_badge()

    def _action_callbacks(self) -> Dict[str, Callable[[], Any]]:
        """Return the callbacks wired into menu and toolbar actions."""

        return {
            "file_new_tab": self._handle_new_tab_requested,
            "file_open": self._handle_open_requested,
            "file_import": self._handle_import_requested,
            "file_save": self._handle_save_requested,
            "file_close_tab": self._handle_close_tab_requested,
            "file_revert": self._handle_revert_requested,
            "file_save_as": self._handle_save_as_requested,
            "ai_snapshot": self._handle_snapshot_requested,
            "ai_accept_changes": self._handle_accept_ai_changes,
            "ai_reject_changes": self._handle_reject_ai_changes,
            "settings_open": self._handle_settings_requested,
            "view_document_status": self._handle_document_status_action,
            "command_palette": self._handle_command_palette_requested,
        }

    def _initialize_command_palette(self) -> None:
        """Instantiate the palette and seed it with the current actions."""

        palette = CommandPaletteDialog(parent=self)
        palette.set_entries(self._build_palette_entries())
        self._command_palette = palette

    def _build_palette_entries(self) -> list[PaletteCommand]:
        return build_palette_commands(self._actions, exclude=("command_palette",))

    def _handle_command_palette_requested(self) -> None:
        if self._command_palette is None:
            self._initialize_command_palette()
        if self._command_palette is not None:
            self._command_palette.set_entries(self._build_palette_entries())
            self._command_palette.show()

    def _wire_signals(self) -> None:
        """Connect editor and chat events to the window handlers."""

        monitor = self._document_monitor
        if monitor is not None:
            self._editor.add_snapshot_listener(monitor.handle_editor_snapshot)
            self._editor.add_text_listener(monitor.handle_editor_text_changed)
            self._editor.add_selection_listener(self._handle_editor_selection_changed)
        else:  # pragma: no cover - monitor is always set in production
            def _noop(*args: Any, **kwargs: Any) -> None:  # noqa: ANN002, ANN003 - fallback stub
                return None

            self._editor.add_snapshot_listener(_noop)
            self._editor.add_text_listener(_noop)
            self._editor.add_selection_listener(_noop)

        self._chat_panel.add_request_listener(self._handle_chat_request)
        self._chat_panel.add_session_reset_listener(self._handle_chat_session_reset)
        self._chat_panel.add_suggestion_panel_listener(self._handle_suggestion_panel_toggled)
        self._chat_panel.set_stop_ai_callback(self._cancel_active_ai_turn)
        self._bridge.add_edit_listener(self._handle_edit_applied)
        self._bridge.add_failure_listener(self._handle_edit_failure)
        if monitor is not None:
            monitor.refresh_chat_suggestions()

    def _handle_editor_selection_changed(
        self,
        tab_id: str,
        _selection: object,
        line: int,
        column: int,
    ) -> None:
        if tab_id != self._workspace.active_tab_id:
            return
        self._status_bar.update_cursor(line, column)

    def _outline_memory(self) -> DocumentSummaryMemory | None:
        runtime = self._outline_runtime
        if runtime is None:
            return None
        return runtime.outline_memory()

    def _resolve_embedding_index(self) -> DocumentEmbeddingIndex | None:
        return self._embedding_controller.resolve_index()

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
        loop: asyncio.AbstractEventLoop | None = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = None
        cached = self._async_loop
        if loop is None and cached is not None and not cached.is_closed():
            loop = cached
        if loop is None:
            _LOGGER.warning("No asyncio event loop is currently available; outline worker startup is deferred.")
            return None
        if loop.is_closed():
            _LOGGER.warning("Resolved asyncio event loop is closed; outline worker cannot start.")
            return None
        self._async_loop = loop
        return loop

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

    def _tool_wiring_context(self) -> ToolWiringContext:
        if self._tool_provider is None:
            raise RuntimeError("Tool provider is not initialized")
        return self._tool_provider.build_tool_wiring_context()

    def _register_default_ai_tools(self, *, register_fn: Callable[..., Any] | None = None) -> None:
        context = self._tool_wiring_context()
        result = register_new_tools(context)
        if result.registered:
            _LOGGER.debug("AI tools registered: %s", ", ".join(result.registered))
        if result.failed:
            _LOGGER.warning("Some AI tools failed to register: %s", ", ".join(result.failed))
            self.update_status("Some AI tools unavailable")
            self._post_assistant_notice("Some AI tools could not be registered; check logs for details.")
        
        # Configure the tool dispatcher for the new tool system
        controller = context.controller
        if controller is not None and hasattr(controller, "configure_tool_dispatcher"):
            controller.configure_tool_dispatcher(context_provider=context.bridge)
            # Wire up dispatch listener to track write tool completions for edit counting
            dispatcher = getattr(controller, "tool_dispatcher", None)
            if dispatcher is not None and hasattr(dispatcher, "set_listener"):
                dispatcher.set_listener(_WriteToolDispatchListener(self))

    def _resolve_plot_state_store(self) -> DocumentPlotStateStore | None:
        orchestrator = self._context.ai_orchestrator
        if orchestrator is None:
            return None
        return getattr(orchestrator, "plot_state_store", None)

    def _resolve_character_map_store(self) -> CharacterMapStore | None:
        orchestrator = self._context.ai_orchestrator
        if orchestrator is None:
            return None
        return getattr(orchestrator, "character_map_store", None)

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

        snapshot = self._chat_panel.consume_turn_snapshot()
        if snapshot is None:
            snapshot = self._chat_panel.capture_state()
        self._pending_turn_snapshot = snapshot

        orchestrator = self._context.ai_orchestrator
        if orchestrator is None:
            self._post_assistant_notice(
                "AI assistant is unavailable. Open Settings to configure your API key and model."
            )
            self.update_status("AI unavailable")
            self._chat_panel.set_ai_running(False)
            return

        self._review_controller.auto_accept_pending_review(reason="new-turn")
        self._review_controller.begin_pending_turn_review(
            prompt=prompt,
            prompt_metadata=metadata,
            chat_snapshot=snapshot,
        )

        if self._ai_task and not self._ai_task.done():
            self._suppress_cancel_abort = True
            try:
                self._cancel_active_ai_turn()
            finally:
                self._suppress_cancel_abort = False
            self.update_status("Previous AI request canceled")

        snapshot = self._bridge.generate_snapshot()
        self._apply_embedding_metadata(snapshot)
        history_payload = serialize_chat_history(
            self._chat_panel.history(),
            limit=0,
            exclude_latest=True,
        )
        self._chat_panel.set_ai_running(True)
        task = self._run_coroutine(
            self._ai_turn_coordinator.run_ai_turn(
                prompt=prompt,
                snapshot=snapshot,
                metadata=metadata or {},
                history=history_payload,
            )
        )
        self._ai_task = task
        if task is None:
            self._chat_panel.set_ai_running(False)

    def _handle_manual_command(self, request: ManualCommandRequest) -> None:
        if request.command is ManualCommandType.OUTLINE:
            self._post_assistant_notice("Outline tooling has been removed. Use /find for document search.")
            self.update_status("Outline removed")
            return
        if request.command is ManualCommandType.FIND_SECTIONS:
            self._post_assistant_notice("Find sections tool has been removed. The AI assistant uses search_document tool automatically.")
            self.update_status("Find sections removed")
            return
        if request.command is ManualCommandType.ANALYZE:
            self._handle_manual_analyze_command(request)
            return
        if request.command is ManualCommandType.STATUS:
            self._handle_manual_status_command(request)
            return
        self._post_assistant_notice(f"Unsupported manual command '{request.command.value}'.")
        self.update_status("Manual command unsupported")

    def _handle_manual_analyze_command(self, request: ManualCommandRequest) -> None:
        orchestrator = self._context.ai_orchestrator
        if orchestrator is None:
            self._post_assistant_notice("AI assistant is unavailable.")
            self.update_status("Analysis unavailable")
            return
        enabled_probe = getattr(orchestrator, "analysis_enabled", None)
        if callable(enabled_probe) and not enabled_probe():
            self._post_assistant_notice("Preflight analysis is disabled. Enable it in Settings > AI to use /analyze.")
            self.update_status("Analysis disabled")
            return

        args = dict(request.args)
        doc_reference = args.get("document_id")
        resolved_id = None
        if doc_reference:
            resolved_id = self._resolve_manual_document_id(doc_reference)
            if resolved_id is None:
                self._post_assistant_notice(f"Couldn't find a document matching '{doc_reference}'.")
                self.update_status("Document not found")
                return
        if resolved_id is None:
            document = self._safe_active_document()
            if document is None:
                self._post_assistant_notice("No active document to analyze.")
                self.update_status("Analysis unavailable")
                return
            resolved_id = document.document_id

        target_tab_id: str | None = None
        target_document: DocumentState | None = None
        for tab in self._workspace.iter_tabs():
            document = tab.document()
            if document.document_id == resolved_id:
                target_tab_id = tab.id
                target_document = document
                break
        if target_document is None:
            target_document = self._workspace.find_document_by_id(resolved_id)
        if target_document is None:
            self._post_assistant_notice("Document is not open in the workspace.")
            self.update_status("Analysis aborted")
            return

        try:
            snapshot = (
                self._bridge.generate_snapshot(tab_id=target_tab_id)
                if target_tab_id is not None
                else target_document.snapshot()
            )
        except Exception as exc:  # pragma: no cover - defensive path
            _LOGGER.debug("Snapshot for manual analysis failed", exc_info=True)
            self._post_assistant_notice(f"Unable to capture snapshot for analysis: {exc}")
            self.update_status("Analysis failed")
            return

        force_refresh = bool(args.get("force_refresh"))
        analyze_reason = (args.get("reason") or "").strip() or None
        telemetry_payload = {
            "document_id": resolved_id,
            "force_refresh": force_refresh,
            "reason": analyze_reason,
            "source": "manual_command",
        }
        telemetry_service.emit("analysis.ui_override.requested", dict(telemetry_payload))

        try:
            advice = controller.request_analysis_advice(
                document_id=resolved_id,
                snapshot=snapshot,
                force_refresh=force_refresh,
                reason=analyze_reason,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Manual analysis command failed", exc_info=True)
            failure_payload = dict(telemetry_payload)
            failure_payload["status"] = "error"
            failure_payload["error"] = str(exc)
            telemetry_service.emit("analysis.ui_override.failed", failure_payload)
            self._post_assistant_notice(f"Analysis command failed: {exc}")
            self.update_status("Analysis failed")
            return

        if advice is None:
            completion_payload = dict(telemetry_payload)
            completion_payload["status"] = "empty"
            telemetry_service.emit("analysis.ui_override.completed", completion_payload)
            self._post_assistant_notice("Preflight analysis did not return any advice.")
            self.update_status("Analysis unavailable")
            return

        label = self._document_label_from_id(resolved_id)
        message = self._format_analysis_notice(advice, document_label=label)
        self._post_assistant_notice(message)
        completion_payload = dict(telemetry_payload)
        completion_payload.update(
            {
                "status": "ok",
                "cache_state": advice.cache_state,
                "chunk_profile": advice.chunk_profile,
                "required_tools": list(advice.required_tools),
                "optional_tools": list(advice.optional_tools),
                "warning_codes": [warning.code for warning in advice.warnings],
            }
        )
        telemetry_service.emit("analysis.ui_override.completed", completion_payload)
        self.update_status("Analysis ready")
        self._telemetry_controller.refresh_analysis_state(resolved_id, document_label=label)

    def _handle_manual_status_command(self, request: ManualCommandRequest) -> None:
        service = self._ensure_document_status_service()
        if service is None:
            self._post_assistant_notice("Document status tooling is unavailable. Install the UI dependencies and restart.")
            self.update_status("Document status unavailable")
            return

        args = dict(request.args)
        doc_reference = args.get("document_id")
        resolved_id = self._resolve_manual_document_id(doc_reference) if doc_reference else None
        if doc_reference and resolved_id is None:
            self._post_assistant_notice(f"Couldn't find a document matching '{doc_reference}'.")
            self.update_status("Document not found")
            return

        as_json = bool(args.get("as_json"))
        payload = self._show_document_status_window(document_id=resolved_id)
        if payload is None:
            try:
                payload = service.build_status_payload(resolved_id)
                self._apply_document_status_badge(payload)
            except ValueError:
                self._post_assistant_notice("Open a document before requesting status.")
                self.update_status("Document status unavailable")
                return
            except Exception as exc:  # pragma: no cover - defensive guard
                _LOGGER.debug("Manual status command failed", exc_info=True)
                self._post_assistant_notice(f"Document status command failed: {exc}")
                self.update_status("Document status failed")
                return

        summary_text = str(payload.get("summary") or "Document status ready.").strip()
        if as_json:
            body = json.dumps(payload, indent=2, ensure_ascii=False)
            message = f"```json\n{body}\n```"
        else:
            message = summary_text
        self._post_assistant_notice(message)
        self.update_status("Document status ready")

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
                return self._document_monitor.document_display_name(document)
            return document_id
        if fallback:
            return fallback
        document = self._safe_active_document()
        if document is not None:
            return self._document_monitor.document_display_name(document)
        return WINDOW_APP_NAME

    def _format_analysis_notice(
        self,
        advice: AnalysisAdvice,
        *,
        document_label: str | None = None,
    ) -> str:
        doc_label = document_label or self._document_label_from_id(advice.document_id)
        header_bits = [f"Preflight analysis for {doc_label}"]
        chunk_profile = (advice.chunk_profile or "auto").strip()
        if chunk_profile:
            header_bits.append(f"profile {chunk_profile}")
        cache_state = advice.cache_state.strip() if advice.cache_state else ""
        if cache_state:
            header_bits.append(f"cache={cache_state}")
        header = " · ".join(header_bits) + "."

        required_tools = ", ".join(advice.required_tools) if advice.required_tools else "none"
        parts = [header, f"Required tools: {required_tools}."]
        if advice.optional_tools:
            parts.append(f"Optional tools: {', '.join(advice.optional_tools)}.")
        refresh_text = "yes" if advice.must_refresh_outline else "no"
        parts.append(f"Outline refresh required: {refresh_text}.")
        if advice.plot_state_status:
            parts.append(f"Plot state: {advice.plot_state_status}.")
        if advice.concordance_status:
            parts.append(f"Concordance: {advice.concordance_status}.")

        warnings = advice.warnings or ()
        if warnings:
            parts.append("Warnings:")
            for warning in warnings:
                parts.append(f"- ({warning.severity}) {warning.code}: {warning.message}")

        traces = advice.rule_trace or ()
        if traces:
            max_traces = 5
            parts.append("Rule trace:")
            for trace in traces[:max_traces]:
                parts.append(f"- {trace}")
            remaining = len(traces) - max_traces
            if remaining > 0:
                parts.append(f"… {remaining} additional rule(s).")

        if advice.generated_at:
            generated_at = datetime.fromtimestamp(advice.generated_at, tz=timezone.utc)
            iso_text = generated_at.isoformat().replace("+00:00", "Z")
            parts.append(f"Generated at {iso_text}.")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Document status helpers
    # ------------------------------------------------------------------
    def _handle_document_status_action(self) -> None:
        payload = self._show_document_status_window()
        if payload is not None:
            self.update_status("Document status ready")

    def _handle_document_status_clicked(self) -> None:
        self._handle_document_status_action()

    def _show_document_status_window(self, document_id: str | None = None) -> Mapping[str, Any] | None:
        service = self._ensure_document_status_service()
        if service is None:
            return None
        descriptors = service.list_document_descriptors()
        if not descriptors:
            self._status_bar.set_document_status_badge("", detail=None)
            self._post_assistant_notice("Open a document before viewing status.")
            self.update_status("Document status unavailable")
            return None
        window = self._ensure_document_status_window()
        if window is None:
            return None
        window.update_documents(descriptors)
        payload = window.show(document_id=document_id)
        if payload is not None:
            self._apply_document_status_badge(payload)
        return payload

    def _ensure_document_status_window(self) -> DocumentStatusWindow | None:
        if self._document_status_window is not None:
            return self._document_status_window
        service = self._ensure_document_status_service()
        if service is None:
            return None
        parent = self._qt_parent_widget()
        try:
            self._document_status_window = DocumentStatusWindow(
                documents=service.list_document_descriptors(),
                status_loader=self._load_document_status_payload,
                parent=parent,
            )
        except Exception:  # pragma: no cover - optional Qt dependency
            _LOGGER.debug("Document status window initialization failed", exc_info=True)
            self._document_status_window = None
        return self._document_status_window

    def _load_document_status_payload(self, document_id: str | None) -> Mapping[str, Any]:
        service = self._ensure_document_status_service()
        if service is None:
            raise RuntimeError("Document status service unavailable")
        payload = service.build_status_payload(document_id)
        self._apply_document_status_badge(payload)
        return payload

    def _apply_document_status_badge(self, payload: Mapping[str, Any] | None) -> None:
        status_bar = self._status_bar
        if status_bar is None:
            return
        badge: Mapping[str, Any] | None = None
        if isinstance(payload, Mapping):
            candidate = payload.get("badge")
            if isinstance(candidate, Mapping):
                badge = candidate
        if badge is None:
            status_bar.set_document_status_badge("", detail=None)
            return
        status_text = str(badge.get("status") or "").strip()
        detail_value = badge.get("detail") if isinstance(badge, Mapping) else None
        if detail_value is None:
            detail_text = None
        elif isinstance(detail_value, str):
            detail_text = detail_value
        else:
            detail_text = str(detail_value)
        severity_value = badge.get("severity") if isinstance(badge, Mapping) else None
        severity = str(severity_value).strip() if isinstance(severity_value, str) else None
        status_bar.set_document_status_badge(status_text, detail=detail_text, severity=severity)

    def _refresh_document_status_badge(self, document_id: str | None = None) -> None:
        status_bar = self._status_bar
        if status_bar is None:
            return
        service = self._document_status_service or self._ensure_document_status_service()
        if service is None:
            status_bar.set_document_status_badge("", detail=None)
            return
        try:
            payload = service.build_status_payload(document_id)
        except ValueError:
            status_bar.set_document_status_badge("", detail=None)
            return
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.debug("Refreshing document status badge failed", exc_info=True)
            return
        self._apply_document_status_badge(payload)

    def _handle_active_tab_for_document_status(self, tab: DocumentTab | None) -> None:
        document_id = None
        if tab is not None:
            try:
                document_id = tab.document().document_id
            except Exception:  # pragma: no cover - defensive guard
                document_id = None
        self._refresh_document_status_badge(document_id=document_id)

    def _handle_active_tab_for_cursor(self, tab: DocumentTab | None) -> None:
        if tab is None:
            self._status_bar.update_cursor(1, 1)
            return
        try:
            selection = tab.editor.selection_span()
        except Exception:
            selection = (0, 0)
        caret = selection[1]
        try:
            document = tab.document()
            text = document.text or ""
        except Exception:
            text = ""
        line, column = line_column_from_offset(text, caret)
        self._status_bar.update_cursor(line, column)

    def _handle_document_status_signal(self, payload: Mapping[str, Any] | None) -> None:
        document_id: str | None = None
        if isinstance(payload, Mapping):
            for key in ("document_id", "documentId", "document"):
                candidate = payload.get(key)
                if candidate:
                    document_id = str(candidate)
                    break
        self._refresh_document_status_badge(document_id=document_id)
        window = self._document_status_window
        if window is not None:
            try:
                window.refresh(document_id=document_id)
            except Exception:  # pragma: no cover - defensive guard
                _LOGGER.debug("Document status window refresh failed", exc_info=True)

    def _register_document_status_listeners(self) -> None:
        if self._document_status_events_registered:
            return
        for event_name in (
            "chunk_flow.requested",
            "chunk_flow.escaped_full_snapshot",
            "chunk_flow.retry_success",
            "analysis.preflight.completed",
            "analysis.preflight.failed",
            "analysis.ui_override.completed",
        ):
            telemetry_service.register_event_listener(event_name, self._handle_document_status_signal)
        self._document_status_events_registered = True

    def _ensure_document_status_service(self) -> DocumentStatusService | None:
        if self._document_status_service is not None:
            return self._document_status_service
        if self._workspace is None:
            return None
        outline_resolver = self._outline_memory
        self._document_status_service = DocumentStatusService(
            workspace=self._workspace,
            bridge=self._bridge,
            telemetry=self._telemetry_controller,
            controller_resolver=lambda: self._context.ai_orchestrator,
            outline_memory_resolver=outline_resolver,
            plot_state_resolver=self._resolve_plot_state_store,
            character_map_resolver=self._resolve_character_map_store,
        )
        return self._document_status_service

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
            header = f"Find text ({status}) for {doc_label} — \"{query_text}\""
        else:
            header = f"Find text ({status}) for {doc_label}."
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
            parts.append("No matching spans were found.")

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
            snippet = condense_whitespace(str(preview)) if isinstance(preview, str) else ""
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




    





    def _set_ai_stream_active(self, active: bool) -> None:
        self._ai_stream_active = active

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
        self.update_status("AI error – pending edits discarded")
        self._chat_panel.set_ai_running(False)
        self._review_controller.abort_pending_review(
            reason="ai-failure",
            restore_composer=True,
            clear_overlays=True,
        )

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
        orchestrator = self._context.ai_orchestrator
        cancel = getattr(orchestrator, "cancel", None)
        if callable(cancel):
            try:
                cancel()
            except Exception:  # pragma: no cover - defensive logging
                _LOGGER.debug("AI orchestrator cancel raised", exc_info=True)

        task = self._ai_task
        if task is not None and not task.done():
            try:
                task.cancel()
            except Exception:  # pragma: no cover - defensive logging
                _LOGGER.debug("AI task cancellation failed", exc_info=True)
        self._ai_task = None
        self._ai_stream_active = False
        self._chat_panel.set_ai_running(False)
        
        # Force release the editor lock when canceling
        self._ai_turn_coordinator.force_release_lock()
        
        if self._suppress_cancel_abort:
            return
        self._review_controller.abort_pending_review(
            reason="ai-canceled",
            status="Canceled AI request; pending edits discarded",
            restore_composer=True,
            clear_overlays=True,
        )

    def _handle_chat_session_reset(self) -> None:
        self._cancel_active_ai_turn()
        self._cancel_dynamic_suggestions()
        self._clear_suggestion_cache()
        self._tool_trace_presenter.reset()
        self._telemetry_controller.set_compaction_stats(None)
        self._document_monitor.refresh_chat_suggestions()
        self.update_status("Chat reset")

    def _handle_suggestion_panel_toggled(self, is_open: bool) -> None:
        if not is_open:
            self._cancel_dynamic_suggestions()
            return

        history = self._chat_panel.history()
        if not history:
            self._document_monitor.refresh_chat_suggestions()
            return

        hist_sig = history_signature(history)
        if hist_sig is None:
            self._document_monitor.refresh_chat_suggestions()
            return

        if (
            hist_sig == self._suggestion_cache_key
            and self._suggestion_cache_values
        ):
            self._chat_panel.set_suggestions(list(self._suggestion_cache_values))
            self.update_status("Loaded cached suggestions")
            return

        orchestrator = self._context.ai_orchestrator
        if orchestrator is None:
            self._document_monitor.refresh_chat_suggestions()
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
        signature = history_signature(history)
        if signature is None:
            self._clear_suggestion_cache()
            return
        self._suggestion_cache_key = signature
        self._suggestion_cache_values = tuple(suggestions)

    async def _generate_dynamic_suggestions(
        self,
        history: Sequence[ChatMessage],
        request_id: int,
    ) -> None:
        orchestrator = self._context.ai_orchestrator
        if orchestrator is None:
            return

        payload = serialize_chat_history(history)
        if not payload:
            return

        suggestions: list[str] = []
        try:
            suggestions = await orchestrator.suggest_followups(payload, max_suggestions=4)
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
        self._document_monitor.refresh_chat_suggestions()
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

    def _handle_edit_applied(self, directive: EditDirective, _state: DocumentState, diff: str) -> None:
        self._emit_guardrail_notice(None, None)
        summary = f"Applied {directive.action} ({diff})"
        self.update_status(summary)
        metadata: Dict[str, Any] = {}
        range_hint: TextRange = directive.target_range
        context = getattr(self._bridge, "last_edit_context", None)
        if directive.action == ActionType.PATCH.value and context is not None:
            range_hint = context.target_range
        range_summary = range_hint.to_tuple()
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
            input_summary=f"range={range_summary}",
            output_summary=diff,
            metadata=metadata,
        )
        spans = self._review_overlay_manager.coerce_overlay_spans(metadata.get("spans"), fallback_range=range_hint)
        diff_preview = metadata.get("diff_preview")
        overlay_label = str(diff_preview or trace.output_summary or trace.name)
        tab_id = self._review_overlay_manager.find_tab_id_for_document(_state)
        tab: DocumentTab | None = None
        if tab_id:
            try:
                tab = self._workspace.get_tab(tab_id)
            except KeyError:
                tab = None
        current_document = tab.document() if tab is not None else None

        turn = self._review_controller.pending_turn_review
        if turn is not None and tab_id:
            session = self._review_controller.ensure_pending_review_session(
                tab_id=tab_id,
                document_snapshot=_state,
                tab=tab,
            )
            if session is not None:
                edit_summary = EditSummary(
                    directive=directive,
                    diff=diff,
                    spans=spans,
                    range_hint=range_hint,
                )
                session.applied_edits.append(edit_summary)
                turn.total_edit_count += 1
                if spans:
                    session.merged_spans = (
                        self._review_overlay_manager.merge_overlay_spans(session.merged_spans, spans)
                        if session.merged_spans
                        else spans
                    )
                session.last_overlay_label = overlay_label
                if current_document is not None:
                    session.latest_version_signature = current_document.version_signature()

        self._chat_panel.show_tool_trace(trace)
        self._review_overlay_manager.apply_diff_overlay(
            trace,
            document=_state,
            range_hint=range_hint,
            tab_id=tab_id,
            spans_override=spans,
            label_override=overlay_label,
        )
        self._document_monitor.update_autosave_indicator(document=current_document or _state)

    def _record_write_tool_edit(self, result: DispatchResult) -> None:
        """Record a write tool completion as an edit in the pending turn review.
        
        Called by _WriteToolDispatchListener when a tool with writes_document=True
        completes successfully. This ensures that write_document and similar tools
        increment the edit count so changes aren't dropped as "no-edits".
        
        Args:
            result: The dispatch result from the completed tool.
        """
        turn = self._review_controller.pending_turn_review
        if turn is None:
            _LOGGER.debug(
                "write tool %s completed but no pending turn review",
                result.tool_name,
            )
            return
        
        # Extract tab_id from the result metadata or tool arguments
        tab_id: str | None = None
        if result.metadata:
            tab_id = result.metadata.get("tab_id")
        if tab_id is None and isinstance(result.result, dict):
            tab_id = result.result.get("tab_id")
        
        # If we still don't have a tab_id, try to get the active tab
        if tab_id is None:
            tab_id = self._workspace.active_tab_id
        
        if tab_id is None:
            _LOGGER.debug(
                "write tool %s completed but could not determine tab_id",
                result.tool_name,
            )
            # Still increment edit count even if we can't track the tab
            turn.total_edit_count += 1
            return
        
        # Get the document state for the session snapshot
        try:
            tab = self._workspace.get_tab(tab_id)
            doc_state = tab.editor.to_document()
        except KeyError:
            _LOGGER.debug(
                "write tool %s completed but tab %s not found",
                result.tool_name,
                tab_id,
            )
            turn.total_edit_count += 1
            return
        
        # Ensure we have a review session for this tab
        session = self._review_controller.ensure_pending_review_session(
            tab_id=tab_id,
            document_snapshot=doc_state,
            tab=tab,
        )
        
        # Increment edit count
        turn.total_edit_count += 1
        _LOGGER.debug(
            "Recorded write tool %s edit for tab %s (total_edit_count=%d)",
            result.tool_name,
            tab_id,
            turn.total_edit_count,
        )

    def _handle_edit_failure(
        self,
        directive: EditDirective,
        message: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        action = (directive.action or "").strip().lower()
        context = dict(metadata or {})
        reason = context.get("reason") or message or "Edit failed"
        tab_hint = context.get("tab_id")
        detail_bits: list[str] = []
        status_hint = context.get("status")
        cause_hint = context.get("cause")
        range_count = context.get("range_count")
        if status_hint:
            detail_bits.append(f"status={status_hint}")
        if cause_hint:
            detail_bits.append(f"cause={cause_hint}")
        if isinstance(range_count, int):
            detail_bits.append(f"ranges={range_count}")
        if context.get("streamed"):
            detail_bits.append("streamed")
        if tab_hint:
            detail_bits.append(f"tab={tab_hint}")
        detail_suffix = f" ({', '.join(detail_bits)})" if detail_bits else ""

        if action == ActionType.PATCH.value:
            guardrail_notice = self._describe_patch_guardrail(context, reason)
            if guardrail_notice is not None:
                status_text, detail_text, toast_text = guardrail_notice
                status_message = status_text if not detail_suffix else f"{status_text}{detail_suffix}"
                self.update_status(status_message)
                self._emit_guardrail_notice(status_text, detail_text)
                if toast_text:
                    self._post_assistant_notice(toast_text)
                return

            self._emit_guardrail_notice(None, None)
            self.update_status(f"Patch rejected{detail_suffix}")
            sentences = [f"Patch apply failed: {reason}."]
            if detail_bits:
                sentences.append(f"Details: {', '.join(detail_bits)}.")
            sentences.append("Request a fresh snapshot before retrying.")
            self._post_assistant_notice(" ".join(sentences))
        else:
            self.update_status(f"Edit failed: {reason}{detail_suffix}")
            if detail_bits:
                self._post_assistant_notice(
                    f"Edit failed ({', '.join(detail_bits)}): {reason}"
                )

    def _describe_patch_guardrail(
        self,
        metadata: Mapping[str, Any] | None,
        reason: str,
    ) -> tuple[str, str, str] | None:
        safe_notice = self._describe_safe_edit_guardrail(metadata, reason)
        if safe_notice is not None:
            return safe_notice
        if not metadata:
            return None
        cause = str(metadata.get("cause") or "").strip()
        reason_text = condense_whitespace(str(metadata.get("reason") or reason or ""))
        if cause == DocumentBridge.CAUSE_HASH_MISMATCH:
            return self._build_snapshot_guardrail_notice(reason_text)
        if cause == DocumentBridge.CAUSE_CHUNK_HASH_MISMATCH:
            return self._build_chunk_guardrail_notice(reason_text)
        status_hint = str(metadata.get("status") or "").strip().lower()
        if status_hint == "stale":
            return self._build_snapshot_guardrail_notice(reason_text)
        return None

    def _describe_safe_edit_guardrail(
        self,
        metadata: Mapping[str, Any] | None,
        reason: str,
    ) -> tuple[str, str, str] | None:
        if not metadata:
            return None
        cause = str(metadata.get("cause") or "").strip()
        if cause != DocumentBridge.CAUSE_INSPECTOR_FAILURE:
            return None
        diagnostics = metadata.get("diagnostics")
        diag_map = diagnostics if isinstance(diagnostics, Mapping) else {}
        reason_text = condense_whitespace(str(metadata.get("reason") or reason or ""))
        detail_bits: list[str] = []
        if reason_text:
            detail_bits.append(reason_text)
        duplicate = diag_map.get("duplicate")
        if isinstance(duplicate, Mapping):
            count = duplicate.get("count")
            snippet = self._summarize_inspector_snippet(duplicate.get("paragraph"))
            if count and snippet:
                detail_bits.append(f"Repeated paragraph ({count}×): {snippet}")
        boundary = diag_map.get("boundary")
        if isinstance(boundary, Mapping):
            detail = boundary.get("detail")
            if detail:
                detail_bits.append(str(detail))
        split = diag_map.get("split")
        if isinstance(split, Mapping):
            detail = split.get("detail")
            if detail:
                detail_bits.append(str(detail))
        window_hint = ""
        window = diag_map.get("window")
        if isinstance(window, Mapping):
            start = window.get("start")
            end = window.get("end")
            if isinstance(start, int) and isinstance(end, int) and end > start:
                length = max(1, end - start)
                window_hint = f"Inspect characters {start:,}–{end:,} (~{length:,} chars)."
                detail_bits.append(window_hint)
        detail_text = " ".join(bit for bit in detail_bits if bit).strip()
        reason_label = self._summarize_guardrail_reason(reason_text)
        status_text = f"Safe Edit: {reason_label}" if reason_label else "Safe Edit Blocked"
        toast_lines = ["Safe edit guardrail rejected the patch."]
        if reason_text:
            toast_lines.append(reason_text)
        if window_hint:
            toast_lines.append(window_hint)
        toast_lines.append("Request a fresh snapshot before retrying.")
        toast_text = " ".join(line for line in toast_lines if line)
        return (status_text, detail_text or status_text, toast_text)

    def _build_snapshot_guardrail_notice(self, reason_text: str) -> tuple[str, str, str]:
        detail = reason_text or "Document snapshot no longer matches the live buffer."
        guidance = "Refresh document_snapshot (AI ▸ Snapshot) before retrying."
        return self._build_guardrail_notice(prefix="Snapshot mismatch", detail=detail, guidance=guidance)

    def _build_chunk_guardrail_notice(self, reason_text: str) -> tuple[str, str, str]:
        detail = reason_text or "Streamed chunk content changed before the patch could apply."
        guidance = "Request a fresh snapshot (with chunk manifest) and rebuild the streamed patch before retrying."
        return self._build_guardrail_notice(prefix="Chunk hash mismatch", detail=detail, guidance=guidance)

    def _build_guardrail_notice(
        self,
        *,
        prefix: str,
        detail: str,
        guidance: str,
    ) -> tuple[str, str, str]:
        detail_text = condense_whitespace(detail) or prefix
        guidance_text = condense_whitespace(guidance)
        reason_label = self._summarize_guardrail_reason(detail_text)
        suffix = ""
        if reason_label:
            maybe_prefixed = reason_label.lower().startswith(prefix.lower())
            suffix = "" if maybe_prefixed else f" – {reason_label}"
        status_text = f"Guardrail: {prefix}{suffix}"
        combined_detail = f"{detail_text} {guidance_text}".strip()
        toast_lines = ["Patch blocked by guardrail.", detail_text, guidance_text]
        toast_text = " ".join(line for line in toast_lines if line)
        return status_text, combined_detail, toast_text

    def _emit_guardrail_notice(self, status: str | None, detail: str | None) -> None:
        if self._status_bar is not None:
            self._status_bar.set_guardrail_notice(status, detail=detail)
        setter = getattr(self._chat_panel, "set_guardrail_state", None)
        if callable(setter):
            setter(status, detail=detail, category="safe_edit")

    def _summarize_guardrail_reason(self, reason: str, *, limit: int = 60) -> str:
        text = condense_whitespace(reason)
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return f"{text[:limit - 1].rstrip()}…"

    def _summarize_inspector_snippet(self, value: Any, *, limit: int = 80) -> str:
        if value is None:
            return ""
        text = condense_whitespace(str(value))
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return f"{text[:limit - 1].rstrip()}…"

    def _maybe_clear_diff_overlay(self, state: DocumentState) -> None:
        if self._review_overlay_manager is not None:
            self._review_overlay_manager.maybe_clear_diff_overlay(state)

    def _clear_diff_overlay(self, *, tab_id: str | None = None) -> None:
        if self._review_overlay_manager is not None:
            self._review_overlay_manager.clear_diff_overlay(tab_id)

    def _handle_active_tab_for_review(self, _tab: DocumentTab | None) -> None:
        pending_turn = self._review_controller.pending_turn_review
        if pending_turn and pending_turn.ready_for_review:
            self._review_controller.show_review_controls()

    def _handle_snapshot_requested(self) -> None:
        """Force a snapshot refresh and log the event."""

        snapshot = self._editor.request_snapshot()
        self._last_snapshot = snapshot
        _LOGGER.debug("Snapshot refreshed: chars=%s", len(snapshot.get("text", "")))
        self.update_status("Snapshot refreshed")
    def _handle_accept_ai_changes(self) -> None:
        if self._review_overlay_manager is not None:
            self._review_overlay_manager.handle_accept_ai_changes()

    def _handle_reject_ai_changes(self) -> None:
        if self._review_overlay_manager is not None:
            self._review_overlay_manager.handle_reject_ai_changes()

    def _handle_open_requested(self) -> None:
        """Prompt the user for a document path and load it into the editor."""

        path = self._document_session.prompt_for_open_path()
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

        self._import_controller.handle_import()

    def _handle_new_tab_requested(self) -> None:
        with self._document_monitor.suspend_snapshot_persistence():
            tab = self._editor.create_tab()
        self._workspace.set_active_tab(tab.id)
        self._document_session.set_current_path(None)
        self._document_session.sync_workspace_state()
        self._document_monitor.update_autosave_indicator(document=self._editor.to_document())
        self.update_status("New tab created")

    def _create_import_tab(self, document: DocumentState, title: str) -> str:
        """Create a tab for imported content and return its identifier."""

        with self._document_monitor.suspend_snapshot_persistence():
            tab = self._editor.create_tab(document=document, title=title, make_active=True)
        self._workspace.set_active_tab(tab.id)
        self._document_session.set_current_path(None)
        return tab.id

    def _handle_close_tab_requested(self) -> None:
        active_tab = self._workspace.active_tab
        if active_tab is None:
            self.update_status("No tab to close")
            return
        handled = self._close_tab(active_tab.id)
        if not handled:
            self.update_status("No tab to close")

    def _handle_tab_close_request(self, tab_id: str) -> bool:
        return self._close_tab(tab_id)

    def _close_tab(self, tab_id: str) -> bool:
        was_active = self._workspace.active_tab_id == tab_id
        try:
            closed = self._editor.close_tab(tab_id)
        except KeyError:
            return False

        # Handle orphaned tabs (existed in Qt widget but not in workspace)
        if closed is None:
            # Still sync workspace state to persist the current tabs
            self._document_session.sync_workspace_state()
            self.update_status(f"Closed orphaned tab {tab_id}")
            return True

        self._review_controller.mark_pending_session_orphaned(closed.id, reason="tab-closed")
        document = closed.document()
        self._document_monitor.clear_unsaved_snapshot(path=document.metadata.path, tab_id=closed.id)
        if self._review_overlay_manager is not None:
            self._review_overlay_manager.discard_overlay(closed.id)
        if self._workspace.tab_count() == 0:
            self._document_session.set_current_path(None)
        elif was_active:
            new_active = self._workspace.active_tab
            if new_active is not None:
                self._document_session.set_current_path(new_active.document().metadata.path)
        self._document_session.sync_workspace_state()
        active_tab = self._workspace.active_tab
        if active_tab is not None:
            self._document_monitor.update_autosave_indicator(document=active_tab.document())
        self.update_status(f"Closed tab {closed.title}")
        return True

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

        metadata = DocumentMetadata(path=path, language=infer_language(path))
        document = DocumentState(text=text, metadata=metadata)
        document.dirty = False

        with self._document_monitor.suspend_snapshot_persistence():
            self._editor.load_document(document)

        self._document_session.set_current_path(path)
        self._document_monitor.clear_unsaved_snapshot(path=path)
        self._refresh_window_title(document)
        self.update_status(f"Reverted {path.name}")
        self._document_session.sync_workspace_state()
        self._document_monitor.update_autosave_indicator(document=document)

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
        self._settings_runtime.apply_runtime_settings(
            result.settings,
            chat_panel_handler=self._apply_chat_panel_settings,
        )
        # Sync workspace state to ensure open_tabs reflects current tabs
        # (the dialog may have been open while tabs were closed)
        self._document_session.sync_workspace_state()
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
    def _file_importer(self) -> FileImporter:
        """Backwards-compatible access to the importer facade (tests)."""

        return self._import_controller.file_importer

    @_file_importer.setter
    def _file_importer(self, importer: FileImporter) -> None:
        self._import_controller.file_importer = importer

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
        metadata = DocumentMetadata(path=target, language=infer_language(target))
        document = DocumentState(text=text, metadata=metadata)
        document.dirty = False

        with self._document_monitor.suspend_snapshot_persistence():
            tab = self._editor.create_tab(document=document, path=target, title=target.name, make_active=True)
        self._workspace.set_active_tab(tab.id)
        self._document_session.set_current_path(target)
        self._document_session.remember_recent_file(target)
        if self._document_session.apply_pending_snapshot_for_path(target):
            self._document_session.sync_workspace_state()
            self._document_monitor.update_autosave_indicator(document=self._editor.to_document())
            return
        self.update_status(f"Loaded {target.name}")
        self._document_session.sync_workspace_state()
        self._document_monitor.update_autosave_indicator(document=document)

    def save_document(self, path: str | Path | None = None) -> Path:
        """Persist the current document to disk and return the saved path."""

        document = self._editor.to_document()
        previous_path = document.metadata.path or self._document_session.get_current_path()
        target_path: Path | None
        if path is not None:
            target_path = Path(path)
        else:
            target_path = document.metadata.path or self._document_session.get_current_path()

        if target_path is None:
            target_path = self._prompt_for_save_path(document=document)
            if target_path is None:
                self.update_status("Save canceled")
                raise RuntimeError("Save canceled")

        target_path.parent.mkdir(parents=True, exist_ok=True)
        file_io.write_text(target_path, document.text)
        document.metadata.path = target_path
        document.dirty = False
        self._document_session.set_current_path(target_path)
        self._document_session.remember_recent_file(target_path)
        self._document_monitor.clear_unsaved_snapshot(path=target_path)
        active_tab = self._workspace.active_tab
        if previous_path is None:
            self._document_monitor.clear_unsaved_snapshot(path=None, tab_id=active_tab.id if active_tab else None)
        if active_tab is not None:
            self._editor.refresh_tab_title(active_tab.id)
        self.update_status(f"Saved {target_path.name}")
        self._refresh_window_title(document)
        self._document_session.sync_workspace_state()
        self._document_monitor.update_autosave_indicator(document=document)
        return target_path

    def update_status(self, message: str, *, timeout_ms: Optional[int] = None) -> None:
        """Update the window status bar and keep local bookkeeping."""

        self._last_status_message = message

        try:
            self._status_bar.set_message(message, timeout_ms=timeout_ms)
        except Exception:  # pragma: no cover - defensive logging
            pass

        _LOGGER.debug("Status: %s", message)

    def _update_lock_status(self, message: str, is_locked: bool) -> None:
        """Update status bar with editor lock state.
        
        This is called by the EditorLockManager when lock state changes.
        """
        _LOGGER.debug("Editor lock status: %s (locked=%s)", message, is_locked)
        if self._status_bar is not None:
            self._status_bar.set_editor_lock_state(is_locked, message)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _prompt_for_import_path(self) -> Path | None:
        """Delegate import dialog handling to the session service."""

        return self._document_session.prompt_for_import_path()

    def _prompt_for_save_path(self, *, document: DocumentState | None = None) -> Path | None:
        """Delegate save dialog handling to the session service."""

        return self._document_session.prompt_for_save_path(document=document)

    def _resolve_active_document_path(self) -> Path | None:
        document = self._editor.to_document()
        if document.metadata.path is not None:
            return Path(document.metadata.path)
        fallback = self._document_session.get_current_path()
        if fallback is not None:
            return Path(fallback)
        return None

    def _qt_parent_widget(self) -> Any | None:
        try:
            from PySide6.QtWidgets import QWidget
        except Exception:  # pragma: no cover - PySide optional during tests
            return None
        return self if isinstance(self, QWidget) else None

    def _show_settings_dialog(self, settings: Settings) -> "SettingsDialogResult":
        try:
            module = importlib.import_module("tinkerbell.widgets.dialogs")
        except Exception as exc:  # pragma: no cover - depends on Qt availability
            raise RuntimeError("Settings dialog requires the PySide6 dependency.") from exc

        show_settings_dialog = getattr(module, "show_settings_dialog", None)
        if not callable(show_settings_dialog):  # pragma: no cover - defensive guard
            raise RuntimeError("Settings dialog is unavailable.")

        parent = self._qt_parent_widget()
        result = show_settings_dialog(parent=parent, settings=settings)
        return cast("SettingsDialogResult", result)

    def _apply_embedding_metadata(self, snapshot: dict[str, Any]) -> None:
        self._embedding_controller.apply_snapshot_metadata(snapshot)

    def _resolve_embedding_cache_root(self) -> Path:
        return self._resolve_outline_cache_root().parent

    def _resolve_outline_cache_root(self) -> Path:
        store = getattr(self._context, "settings_store", None)
        if store is not None:
            base_dir = store.path.parent
        else:
            base_dir = Path.home() / ".tinkerbell"
        return base_dir / "cache" / "outline_builder"

    def _apply_chat_panel_settings(self, settings: Settings) -> None:
        visible = bool(getattr(settings, "show_tool_activity_panel", False))
        setter = getattr(self._chat_panel, "set_tool_activity_visibility", None)
        if callable(setter):
            setter(visible)

    def _build_ai_client_from_settings(self, settings: Settings):
        return self._settings_runtime.build_ai_client_from_settings(settings)

    def _build_ai_orchestrator_from_settings(self, settings: Settings):
        return self._settings_runtime.build_ai_orchestrator_from_settings(settings)

    @property
    def _ai_client_signature(self) -> tuple[Any, ...] | None:
        return self._settings_runtime.ai_client_signature

    def _refresh_window_title(self, state: Optional[DocumentState] = None) -> None:
        """Construct a descriptive window title for the active document."""

        if state is None and self._workspace.active_tab is None:
            self.setWindowTitle(self._WINDOW_BASE_TITLE)
            return

        document = state or self._editor.to_document()
        candidate_path: Optional[Path]
        if document.metadata.path is not None:
            candidate_path = Path(document.metadata.path)
        else:
            candidate_path = self._document_session.get_current_path()

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

