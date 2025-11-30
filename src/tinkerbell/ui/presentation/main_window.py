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
        workspace: Any = None,
        skip_widgets: bool = False,
    ) -> None:
        """Initialize the main window.

        Args:
            event_bus: The event bus for reactive updates.
            coordinator: The application coordinator for operations.
            workspace: The document workspace for the editor.
            skip_widgets: If True, skip widget creation (for testing).
        """
        super().__init__()
        self._event_bus = event_bus
        self._coordinator = coordinator
        self._workspace = workspace
        self._async_loop: asyncio.AbstractEventLoop | None = None

        # Widgets (created lazily unless skip_widgets)
        self._editor: Any = None
        self._chat_panel: Any = None
        self._status_bar: Any = None
        self._chrome_state: Any = None

        # Updaters
        self._status_bar_updater: Any = None
        self._chat_panel_updater: Any = None

        # Dialogs (created on demand)
        self._document_status_window: Any = None

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
            self._setup_default_suggestions()

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
            from .chat.chat_panel import ChatPanel
            from ...editor.tabbed_editor import TabbedEditorWidget
            from .widgets.status_bar import StatusBar

            self._editor = TabbedEditorWidget(
                workspace=self._workspace,
                skip_default_tab=True,
            )

            # Get settings to configure chat panel
            show_tool_activity = False
            if self._coordinator._settings_provider:
                settings = self._coordinator._settings_provider()
                if settings:
                    show_tool_activity = settings.show_tool_activity_panel

            self._chat_panel = ChatPanel(show_tool_activity_panel=show_tool_activity)
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
            accept_callback=self._handle_accept,
            reject_callback=self._handle_reject,
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
            AITurnToolExecuted,
            DocumentOpened,
            DocumentSaved,
            EditorLockChanged,
            SettingsChanged,
            StatusMessage,
            WindowTitleChanged,
        )

        self._event_bus.subscribe(WindowTitleChanged, self._on_window_title_changed)
        self._event_bus.subscribe(DocumentOpened, self._on_document_opened)
        self._event_bus.subscribe(DocumentSaved, self._on_document_saved)
        self._event_bus.subscribe(ActiveTabChanged, self._on_active_tab_changed)
        self._event_bus.subscribe(StatusMessage, self._on_status_message)
        self._event_bus.subscribe(SettingsChanged, self._on_settings_changed)
        self._event_bus.subscribe(AITurnToolExecuted, self._on_tool_executed)
        self._event_bus.subscribe(EditorLockChanged, self._on_editor_lock_changed)

        self._subscribed = True
        LOGGER.debug("ThinMainWindow: subscribed to events")

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events."""
        if not self._subscribed:
            return

        from ..events import (
            ActiveTabChanged,
            AITurnToolExecuted,
            DocumentOpened,
            DocumentSaved,
            EditorLockChanged,
            SettingsChanged,
            StatusMessage,
            WindowTitleChanged,
        )

        self._event_bus.unsubscribe(WindowTitleChanged, self._on_window_title_changed)
        self._event_bus.unsubscribe(DocumentOpened, self._on_document_opened)
        self._event_bus.unsubscribe(DocumentSaved, self._on_document_saved)
        self._event_bus.unsubscribe(ActiveTabChanged, self._on_active_tab_changed)
        self._event_bus.unsubscribe(StatusMessage, self._on_status_message)
        self._event_bus.unsubscribe(SettingsChanged, self._on_settings_changed)
        self._event_bus.unsubscribe(AITurnToolExecuted, self._on_tool_executed)
        self._event_bus.unsubscribe(EditorLockChanged, self._on_editor_lock_changed)

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
        self._update_document_format(getattr(event, "path", None))

    def _on_document_saved(self, event: Any) -> None:
        """Handle DocumentSaved events."""
        self._update_window_title()
        self._update_document_format(getattr(event, "path", None))

    def _on_active_tab_changed(self, event: Any) -> None:
        """Handle ActiveTabChanged events."""
        self._update_window_title()
        self._update_document_format_from_active_tab()

    def _update_document_format(self, path: str | Path | None) -> None:
        """Update the status bar document format based on file path."""
        if self._status_bar is None:
            return

        from ...ai.tools.list_tabs import detect_file_type

        path_str = str(path) if path else None
        file_type = detect_file_type(path_str)
        try:
            self._status_bar.set_document_format(file_type)
        except Exception:
            LOGGER.debug("Failed to update document format", exc_info=True)

    def _update_document_format_from_active_tab(self) -> None:
        """Update document format from the currently active tab."""
        if self._coordinator is None:
            return

        try:
            tab = self._coordinator.active_tab
            if tab is not None:
                self._update_document_format(tab.path)
            else:
                self._update_document_format(None)
        except Exception:
            LOGGER.debug("Failed to get active tab for format update", exc_info=True)

    def _on_status_message(self, event: Any) -> None:
        """Handle StatusMessage events (for last_status_message tracking)."""
        self._last_status_message = event.message

    def _on_editor_lock_changed(self, event: Any) -> None:
        """Handle EditorLockChanged events to lock/unlock the editor."""
        if self._editor is None:
            return

        locked = getattr(event, "locked", False)
        try:
            self._editor.set_all_readonly(locked)
        except Exception:
            LOGGER.debug("ThinMainWindow: failed to set editor readonly state", exc_info=True)

    def _on_settings_changed(self, event: Any) -> None:
        """Handle SettingsChanged events to update UI based on new settings."""
        if self._chat_panel is None:
            return

        settings = event.settings
        if settings is None:
            return

        # Update tool activity panel visibility
        show_tool_activity = getattr(settings, "show_tool_activity_panel", False)
        self._chat_panel.set_tool_activity_visibility(show_tool_activity)
        LOGGER.debug("ThinMainWindow: updated tool activity visibility to %s", show_tool_activity)

    def _on_tool_executed(self, event: Any) -> None:
        """Handle AITurnToolExecuted events to update the tool activity panel."""
        if self._chat_panel is None:
            return

        from .chat.message_model import ToolTrace

        # Get event details
        tool_name = getattr(event, "tool_name", "") or "tool"
        tool_call_id = getattr(event, "tool_call_id", "") or ""
        arguments = getattr(event, "arguments", "") or ""
        result = getattr(event, "result", "") or ""
        success = getattr(event, "success", True)
        duration_ms = getattr(event, "duration_ms", 0.0) or 0.0

        # Check if this is a "running" event (arguments.done) or a result event
        is_running = result == "(running…)"

        if is_running:
            # Tool is starting - create a new pending trace
            input_summary = arguments[:100] if arguments else ""
            trace = ToolTrace(
                name=tool_name,
                input_summary=input_summary,
                output_summary="(running…)",
                duration_ms=0,
                metadata={
                    "tool_call_id": tool_call_id,
                    "raw_input": arguments,
                    "raw_output": "",
                },
            )
            # Store pending trace for later update
            if not hasattr(self, "_pending_tool_traces"):
                self._pending_tool_traces: dict[str, Any] = {}
            self._pending_tool_traces[tool_call_id] = trace
            self._chat_panel.show_tool_trace(trace)
        else:
            # Tool completed - update existing trace if we have one
            pending_traces = getattr(self, "_pending_tool_traces", {})
            existing_trace = pending_traces.get(tool_call_id)

            if existing_trace is not None:
                # Update the existing trace
                output_summary = result if result else "(no output)"
                if not success:
                    output_summary = f"Error: {result}" if result else "Error"

                existing_trace.output_summary = output_summary
                existing_trace.duration_ms = int(duration_ms)
                if existing_trace.metadata:
                    existing_trace.metadata["raw_output"] = result

                # Tell the panel to refresh
                self._chat_panel.update_tool_trace(existing_trace)

                # Remove from pending
                pending_traces.pop(tool_call_id, None)
            else:
                # No pending trace found - create a new one (shouldn't happen normally)
                input_summary = arguments[:100] if arguments else ""
                output_summary = result if result else "(no output)"
                if not success:
                    output_summary = f"Error: {result}" if result else "Error"

                trace = ToolTrace(
                    name=tool_name,
                    input_summary=input_summary,
                    output_summary=output_summary,
                    duration_ms=int(duration_ms),
                    metadata={
                        "tool_call_id": tool_call_id,
                        "raw_input": arguments,
                        "raw_output": result,
                    },
                )
                self._chat_panel.show_tool_trace(trace)

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
        try:
            from tinkerbell.ui.presentation.widgets.dialogs import SettingsDialog, ValidationResult
            from tinkerbell.services.settings import Settings
        except ImportError:
            LOGGER.warning("Settings dialog requires PySide6")
            return

        # Get current settings from coordinator's settings provider
        settings = self._coordinator._settings_provider()
        if settings is None:
            LOGGER.warning("No settings available")
            return

        # Create test functions for the dialog
        validator = self._create_settings_validator()
        api_tester = self._create_api_tester()
        embedding_tester = self._create_embedding_tester()

        # Show settings dialog
        dialog = SettingsDialog(
            settings=settings,
            parent=self,
            validator=validator,
            api_tester=api_tester,
            embedding_tester=embedding_tester,
        )
        if dialog.exec() == 1:  # QDialog.Accepted == 1
            # Gather new settings and apply them
            new_settings = dialog.gather_settings()
            if new_settings is not None:
                # Update settings in context
                self._apply_settings_changes(settings, new_settings)

    def _create_settings_validator(self) -> Any:
        """Create a function to validate settings/API key."""
        from tinkerbell.ui.presentation.widgets.dialogs import ValidationResult

        def validate_settings(settings: Any) -> ValidationResult:
            """Validate API key and settings format."""
            errors = []

            # Validate base URL
            base_url = getattr(settings, "base_url", "") or ""
            if not base_url:
                errors.append("Base URL is required.")
            elif not base_url.startswith(("http://", "https://")):
                errors.append("Base URL must start with http:// or https://")

            # Validate API key
            api_key = getattr(settings, "api_key", "") or ""
            if not api_key:
                errors.append("API key is required.")
            elif len(api_key) < 10:
                errors.append("API key seems too short.")

            # Validate model
            model = getattr(settings, "model", "") or ""
            if not model:
                errors.append("Model name is required.")

            if errors:
                return ValidationResult(
                    ok=False,
                    message="Validation failed: " + "; ".join(errors),
                )

            return ValidationResult(
                ok=True,
                message="Settings validated successfully.",
            )

        return validate_settings

    def _create_api_tester(self) -> Any:
        """Create a function to test API connection."""
        from tinkerbell.ui.presentation.widgets.dialogs import ValidationResult

        def test_api(settings: Any) -> ValidationResult:
            """Test API connection by listing models."""
            try:
                import httpx

                base_url = (settings.base_url or "").rstrip("/")
                api_key = settings.api_key or ""

                if not base_url:
                    return ValidationResult(
                        ok=False,
                        message="Base URL is required.",
                    )

                # Use synchronous httpx to test connection
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                with httpx.Client(timeout=10.0) as http_client:
                    response = http_client.get(
                        f"{base_url}/models",
                        headers=headers,
                    )
                    if response.status_code == 200:
                        try:
                            data = response.json()
                            models = data.get("data", [])
                            return ValidationResult(
                                ok=True,
                                message=f"API connection successful! Found {len(models)} model(s).",
                            )
                        except Exception:
                            return ValidationResult(
                                ok=True,
                                message="API connection successful.",
                            )
                    elif response.status_code == 401:
                        return ValidationResult(
                            ok=False,
                            message="API authentication failed. Check your API key.",
                        )
                    else:
                        return ValidationResult(
                            ok=False,
                            message=f"API returned status {response.status_code}.",
                        )

            except httpx.ConnectError as exc:
                return ValidationResult(
                    ok=False,
                    message=f"Cannot connect to API: {exc}",
                )
            except Exception as exc:
                return ValidationResult(
                    ok=False,
                    message=f"API connection failed: {exc}",
                )

        return test_api

    def _create_embedding_tester(self) -> Any:
        """Create a function to test embedding service."""
        from tinkerbell.ui.presentation.widgets.dialogs import ValidationResult

        def test_embeddings(settings: Any) -> ValidationResult:
            """Test embedding service by actually generating embeddings."""
            try:
                metadata = getattr(settings, "metadata", {}) or {}
                mode = metadata.get("embedding_mode", "disabled")

                if mode == "disabled":
                    return ValidationResult(
                        ok=True,
                        message="Embeddings are disabled.",
                    )

                if mode == "local":
                    # Actually test loading the model and generating embeddings
                    model_path = metadata.get("st_model_path", "")
                    if not model_path:
                        return ValidationResult(
                            ok=False,
                            message="Local mode requires a model path or HuggingFace model ID.",
                        )

                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        return ValidationResult(
                            ok=False,
                            message=f"sentence-transformers is required for local embeddings: {exc}",
                        )

                    # Get optional settings
                    device = metadata.get("st_device", "cpu")
                    cache_dir = metadata.get("st_cache_dir") or None
                    dtype_name = metadata.get("st_dtype", "")

                    # Parse dtype
                    import torch
                    dtype_value = None
                    if dtype_name:
                        dtype_map = {
                            "float32": torch.float32,
                            "float16": torch.float16,
                            "bfloat16": torch.bfloat16,
                        }
                        dtype_value = dtype_map.get(dtype_name)

                    model_kwargs = {"dtype": dtype_value} if dtype_value else {}

                    try:
                        # Load the model
                        model = SentenceTransformer(
                            model_path,
                            device=device,
                            cache_folder=cache_dir,
                            model_kwargs=model_kwargs,
                        )

                        # Generate a test embedding
                        test_text = "This is a test sentence for embedding generation."
                        embedding = model.encode([test_text], convert_to_numpy=True)

                        if embedding is not None and len(embedding) > 0:
                            dims = len(embedding[0])
                            return ValidationResult(
                                ok=True,
                                message=f"Embedding test successful! Model loaded, {dims}-dimensional vectors.",
                            )
                        else:
                            return ValidationResult(
                                ok=False,
                                message="Model loaded but failed to generate embeddings.",
                            )

                    except Exception as exc:
                        return ValidationResult(
                            ok=False,
                            message=f"Failed to load embedding model: {exc}",
                        )

                if mode == "custom-api":
                    # Try to connect to custom embedding API and generate a test embedding
                    api_config = metadata.get("embedding_api", {})
                    base_url = api_config.get("base_url", "")
                    api_key = api_config.get("api_key", "")
                    model_name = api_config.get("model", "text-embedding-ada-002")

                    if not base_url:
                        return ValidationResult(
                            ok=False,
                            message="Custom API mode requires a base URL.",
                        )

                    import httpx
                    try:
                        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                        headers["Content-Type"] = "application/json"

                        # Try to generate a test embedding
                        with httpx.Client(timeout=30.0) as http_client:
                            response = http_client.post(
                                f"{base_url.rstrip('/')}/embeddings",
                                headers=headers,
                                json={
                                    "input": "Test embedding generation.",
                                    "model": model_name,
                                },
                            )
                            if response.status_code == 200:
                                data = response.json()
                                embeddings = data.get("data", [])
                                if embeddings:
                                    dims = len(embeddings[0].get("embedding", []))
                                    return ValidationResult(
                                        ok=True,
                                        message=f"Embedding API test successful! {dims}-dimensional vectors.",
                                    )
                                return ValidationResult(
                                    ok=True,
                                    message="Embedding API responded successfully.",
                                )
                            elif response.status_code == 401:
                                return ValidationResult(
                                    ok=False,
                                    message="Embedding API authentication failed. Check your API key.",
                                )
                            else:
                                return ValidationResult(
                                    ok=False,
                                    message=f"Embedding API error: {response.status_code} - {response.text[:100]}",
                                )
                    except httpx.ConnectError as exc:
                        return ValidationResult(
                            ok=False,
                            message=f"Cannot connect to embedding API: {exc}",
                        )
                    except Exception as exc:
                        return ValidationResult(
                            ok=False,
                            message=f"Embedding API test failed: {exc}",
                        )

                return ValidationResult(
                    ok=False,
                    message=f"Unknown embedding mode: {mode}",
                )

            except Exception as exc:
                return ValidationResult(
                    ok=False,
                    message=f"Embedding test failed: {exc}",
                )

        return test_embeddings

    def _apply_settings_changes(self, old_settings: Any, new_settings: Any) -> None:
        """Apply settings changes after dialog acceptance."""
        from dataclasses import fields
        
        # Copy new values to the existing settings object
        for field in fields(new_settings):
            setattr(old_settings, field.name, getattr(new_settings, field.name))

        # Persist settings
        context = getattr(self._coordinator, "_context", None)
        if context is None:
            # Try to get settings store from coordinator's internal state
            session_store = getattr(self._coordinator, "_session_store", None)
            if session_store:
                session_store.persist_settings(old_settings)
        
        # Publish settings changed event
        from ..events import SettingsChanged
        self._event_bus.publish(SettingsChanged(settings=old_settings))
        
        LOGGER.debug("ThinMainWindow: settings applied")

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
        """Handle document status action - show the document status window."""
        try:
            from .dialogs.document_status_window import DocumentStatusWindow
        except ImportError:
            LOGGER.warning("DocumentStatusWindow requires PySide6")
            return

        # Create or update the window
        if self._document_status_window is None:
            documents = self._coordinator.get_document_descriptors()
            self._document_status_window = DocumentStatusWindow(
                documents=documents,
                status_loader=self._coordinator.get_document_status,
                parent=self,
            )
        else:
            # Update documents list in case tabs changed
            documents = self._coordinator.get_document_descriptors()
            self._document_status_window.update_documents(documents)

        # Get active document ID
        active_doc_id = None
        active_tab = self._coordinator.active_tab_id
        if active_tab:
            # Find document ID for active tab
            for desc in self._coordinator.get_document_descriptors():
                if desc.tab_id == active_tab:
                    active_doc_id = desc.document_id
                    break

        self._document_status_window.show(document_id=active_doc_id)
        LOGGER.debug("ThinMainWindow: document status window shown")

    # ------------------------------------------------------------------
    # Suggestions Setup
    # ------------------------------------------------------------------

    # Default suggestions shown when no context-specific suggestions are available
    DEFAULT_SUGGESTIONS: tuple[str, ...] = (
        "Summarize the document",
        "Continue writing from where you left off",
        "Improve the writing style",
        "Fix grammar and spelling errors",
        "Expand on this topic",
    )

    def _setup_default_suggestions(self) -> None:
        """Set up default suggestions for the chat panel."""
        if self._chat_panel is None:
            return

        try:
            self._chat_panel.set_suggestions(self.DEFAULT_SUGGESTIONS)
            LOGGER.debug("ThinMainWindow: default suggestions set")
        except Exception:
            LOGGER.debug("Failed to set default suggestions", exc_info=True)

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

        # Wire stop AI callback: ChatPanel stop button -> cancel_ai_turn
        self._chat_panel.set_stop_ai_callback(self._on_stop_ai_requested)

        # Wire suggestion panel listener: generate AI suggestions when panel is opened
        self._chat_panel.add_suggestion_panel_listener(self._on_suggestion_panel_toggled)

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

    def _on_stop_ai_requested(self) -> None:
        """Handle stop button click to cancel the active AI turn."""
        self._coordinator.cancel_ai_turn()

    def _on_suggestion_panel_toggled(self, is_open: bool) -> None:
        """Handle suggestion panel open/close to generate AI suggestions.

        When the panel is opened and there's chat history, generates
        context-aware suggestions from the AI. Falls back to default
        suggestions if generation fails or there's no history.

        Args:
            is_open: True if the panel was opened, False if closed.
        """
        if not is_open or self._chat_panel is None:
            return

        # Get chat history
        try:
            history = self._chat_panel.history()
            if not history:
                # No history, keep default suggestions
                LOGGER.debug("ThinMainWindow: no chat history, keeping defaults")
                return

            # Convert ChatMessage to role/content dict format
            history_dicts = [
                {"role": msg.role, "content": msg.content}
                for msg in history
                if msg.role and msg.content
            ]

            if not history_dicts:
                return

            # Schedule async suggestion generation
            coro = self._generate_suggestions(history_dicts)
            self.schedule_coroutine(coro)

        except Exception:
            LOGGER.debug("Failed to get chat history for suggestions", exc_info=True)

    async def _generate_suggestions(self, history: list[dict[str, str]]) -> None:
        """Generate AI-based suggestions from chat history.

        Args:
            history: Chat history as role/content dicts.
        """
        try:
            suggestions = await self._coordinator.suggest_followups(history)
            if suggestions and self._chat_panel is not None:
                self._chat_panel.set_suggestions(suggestions)
                LOGGER.debug(
                    "ThinMainWindow: updated suggestions from AI (%d items)",
                    len(suggestions),
                )
        except Exception:
            LOGGER.debug("Failed to generate AI suggestions", exc_info=True)

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

    def _save_workspace_state(self) -> None:
        """Save workspace state before closing."""
        try:
            if self._coordinator.save_workspace_state():
                LOGGER.debug("ThinMainWindow: saved workspace state")
            else:
                LOGGER.debug("ThinMainWindow: workspace state not saved (no settings)")
        except Exception:
            LOGGER.debug("ThinMainWindow: failed to save workspace state", exc_info=True)

    def closeEvent(self, event: Any) -> None:
        """Handle window close event."""
        LOGGER.debug("ThinMainWindow: close event")

        # Save workspace state before closing
        self._save_workspace_state()

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
