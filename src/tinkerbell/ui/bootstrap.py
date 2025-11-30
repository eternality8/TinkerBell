"""Application bootstrap module.

This module provides the factory function that creates and wires together
all components of the UI architecture, returning a fully configured
application ready to run.

The bootstrap process:
1. Creates the event bus
2. Instantiates all domain stores
3. Creates infrastructure adapters
4. Instantiates the coordinator with all dependencies
5. Creates the main window
6. Returns configured components

Usage:
    from tinkerbell.ui.bootstrap import create_application
    from tinkerbell.ui.models.window_state import WindowContext

    context = WindowContext(settings=settings, ...)
    event_bus, coordinator, window = create_application(context)
    window.show()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

from .events import EventBus
from .models.window_state import WindowContext

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from .application.coordinator import AppCoordinator
    from .presentation.main_window import ThinMainWindow

_LOGGER = logging.getLogger(__name__)


def create_application(
    context: WindowContext,
    *,
    workspace: Any | None = None,
    editor: Any | None = None,
    orchestrator_provider: Callable[[], Any] | None = None,
    skip_widgets: bool = False,
) -> tuple[EventBus, "AppCoordinator", "ThinMainWindow"]:
    """Create and wire all application components.

    This is the main entry point for bootstrapping the UI application.
    It creates all layers (domain, application, infrastructure, presentation)
    and wires them together via the event bus and dependency injection.

    Args:
        context: The window context with settings and shared state.
        workspace: Optional pre-created DocumentWorkspace. If None, a new
                   one will be created.
        editor: Optional pre-created TabbedEditorWidget. If None, the
                main window will create one.
        orchestrator_provider: Optional callback to get the AI orchestrator.
                               If None, uses context.ai_orchestrator.
        skip_widgets: If True, skip widget creation (for headless testing).

    Returns:
        A tuple of (event_bus, coordinator, main_window).

    Example:
        context = WindowContext(
            settings=settings,
            settings_store=settings_store,
            unsaved_cache=cache,
            unsaved_cache_store=cache_store,
        )
        event_bus, coordinator, window = create_application(context)
        window.show()
    """
    _LOGGER.info("Bootstrapping UI application...")

    # =========================================================================
    # 1. Create Event Bus
    # =========================================================================
    event_bus = EventBus()
    _LOGGER.debug("Created event bus")

    # =========================================================================
    # 2. Create Document Workspace (if not provided)
    # =========================================================================
    if workspace is None:
        try:
            from ..editor.workspace import DocumentWorkspace

            workspace = DocumentWorkspace()
            _LOGGER.debug("Created document workspace")
        except ImportError:
            _LOGGER.warning("Could not import DocumentWorkspace")
            raise

    # =========================================================================
    # 3. Create Domain Stores
    # =========================================================================
    from .domain import (
        AITurnManager,
        DocumentStore,
        EmbeddingStore,
        OutlineStore,
        OverlayManager,
        ReviewManager,
        SessionStore,
    )

    # DocumentStore wraps the workspace with event emission
    document_store = DocumentStore(workspace=workspace, event_bus=event_bus)
    _LOGGER.debug("Created document store")

    # SessionStore for persistence
    session_store = SessionStore(
        settings_store=context.settings_store,
        unsaved_cache_store=context.unsaved_cache_store,
        event_bus=event_bus,
    )
    _LOGGER.debug("Created session store")

    # AITurnManager for AI execution
    if orchestrator_provider is None:
        orchestrator_provider = lambda: context.ai_orchestrator  # noqa: E731

    ai_turn_manager = AITurnManager(
        orchestrator_provider=orchestrator_provider,
        event_bus=event_bus,
    )
    _LOGGER.debug("Created AI turn manager")

    # ReviewManager for pending reviews
    review_manager = ReviewManager(event_bus=event_bus)
    _LOGGER.debug("Created review manager")

    # EditTracker to wire EditApplied events to managers
    from .presentation.status_updaters import EditTracker

    edit_tracker = EditTracker(
        ai_turn_manager=ai_turn_manager,
        review_manager=review_manager,
        event_bus=event_bus,
    )
    _LOGGER.debug("Created edit tracker")

    # OverlayManager for diff overlays
    # We create stub callbacks initially - they'll be wired to the actual
    # editor after the main window is created
    def _stub_show_overlay(
        label: str,
        spans: tuple[tuple[int, int], ...],
        summary: str | None,
        source: str | None,
        tab_id: str | None,
    ) -> None:
        _LOGGER.debug("OverlayManager: show_overlay stub called (editor not wired)")

    def _stub_clear_overlay(tab_id: str | None) -> None:
        _LOGGER.debug("OverlayManager: clear_overlay stub called (editor not wired)")

    overlay_manager = OverlayManager(
        show_overlay=_stub_show_overlay,
        clear_overlay=_stub_clear_overlay,
        event_bus=event_bus,
    )
    _LOGGER.debug("Created overlay manager")

    # EmbeddingStore for document embeddings
    cache_root = None
    if context.settings_store:
        try:
            cache_root = context.settings_store.path.parent / "embedding_cache"
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    embedding_store = EmbeddingStore(
        cache_root=cache_root,
        event_bus=event_bus,
    )
    _LOGGER.debug("Created embedding store")

    # OutlineStore for document outlines
    # document_provider will be set up later if needed
    outline_store = OutlineStore(
        document_provider=None,
        storage_root=None,
        loop_resolver=None,
        event_bus=event_bus,
    )
    _LOGGER.debug("Created outline store")

    # =========================================================================
    # 4. Create Infrastructure Adapters
    # =========================================================================
    from .infrastructure import (
        BridgeAdapter,
        SettingsAdapter,
        TelemetryAdapter,
    )

    # BridgeAdapter wraps WorkspaceBridgeRouter with event emission
    bridge_adapter = BridgeAdapter(workspace=workspace, event_bus=event_bus)
    _LOGGER.debug("Created bridge adapter")

    # SettingsAdapter for settings management
    settings_adapter = SettingsAdapter(
        context=context,
        event_bus=event_bus,
        editor=editor,
        initial_settings=context.settings,
    )
    _LOGGER.debug("Created settings adapter")

    # TelemetryAdapter for telemetry forwarding
    telemetry_adapter = TelemetryAdapter(event_bus=event_bus)
    telemetry_adapter.register_telemetry_listeners()
    _LOGGER.debug("Created telemetry adapter")

    # =========================================================================
    # 5. Create Application Coordinator
    # =========================================================================
    from .application import AppCoordinator

    # Settings/cache providers
    def settings_provider() -> Any:
        return context.settings

    def cache_provider() -> Any:
        return context.unsaved_cache

    coordinator = AppCoordinator(
        event_bus=event_bus,
        document_store=document_store,
        session_store=session_store,
        ai_turn_manager=ai_turn_manager,
        review_manager=review_manager,
        overlay_manager=overlay_manager,
        settings_provider=settings_provider,
        cache_provider=cache_provider,
        # Providers will be set after window creation
    )
    _LOGGER.debug("Created application coordinator")

    # =========================================================================
    # 6. Create Main Window
    # =========================================================================
    from .presentation.main_window import ThinMainWindow

    main_window = ThinMainWindow(
        event_bus=event_bus,
        coordinator=coordinator,
        workspace=workspace,
        skip_widgets=skip_widgets,
    )
    _LOGGER.debug("Created main window")

    # =========================================================================
    # 7. Wire Up Remaining Dependencies
    # =========================================================================
    # Wire editor callbacks to overlay manager if window created widgets
    if main_window.editor is not None:
        editor = main_window.editor

        def _show_overlay(
            label: str,
            spans: tuple[tuple[int, int], ...],
            summary: str | None,
            source: str | None,
            tab_id: str | None,
        ) -> None:
            editor.show_diff_overlay(
                label,
                spans=spans,
                summary=summary,
                source=source,
                tab_id=tab_id,
            )

        def _clear_overlay(tab_id: str | None) -> None:
            editor.clear_diff_overlay(tab_id=tab_id)

        overlay_manager.set_callbacks(_show_overlay, _clear_overlay)
        settings_adapter.set_editor(editor)
        _LOGGER.debug("Wired editor to overlay manager and settings adapter")

    # Wire dialog providers to coordinator
    if not skip_widgets:
        from .presentation.dialogs import FileDialogProvider, ImportDialogProvider

        # Create dialog providers with window as parent
        file_dialog_provider = FileDialogProvider(
            parent_provider=lambda: main_window,
            start_dir_resolver=lambda: session_store.current_path,
            token_budget_resolver=lambda: getattr(context.settings, "max_context_tokens", 128_000),
        )
        coordinator.set_dialog_provider(file_dialog_provider)

        import_dialog_provider = ImportDialogProvider(
            parent_provider=lambda: main_window,
            start_dir_resolver=lambda: session_store.current_path,
        )
        coordinator.set_import_dialog_provider(import_dialog_provider)
        _LOGGER.debug("Wired dialog providers to coordinator")

    # Set snapshot provider on coordinator
    coordinator.set_snapshot_provider(bridge_adapter)

    # =========================================================================
    # 8. Configure AI Tools
    # =========================================================================
    orchestrator = context.ai_orchestrator
    if orchestrator is not None:
        try:
            from ..editor.selection_gateway import SelectionGateway
            from .infrastructure.tool_adapter import ToolAdapter

            # Create selection gateway for tool wiring
            selection_gateway = SelectionGateway(workspace=workspace)

            # Create tool adapter
            tool_adapter = ToolAdapter(
                controller_resolver=lambda: orchestrator,
                bridge=bridge_adapter,
                workspace=workspace,
                selection_gateway=selection_gateway,
                editor=main_window.editor,
                event_bus=event_bus,
            )

            # Configure the tool dispatcher on the orchestrator
            orchestrator.configure_tool_dispatcher(
                context_provider=bridge_adapter,
            )

            # Register all tools
            result = tool_adapter.register_tools()
            _LOGGER.info(
                "Tool registration: %d registered, %d failed, %d skipped",
                len(result.registered),
                len(result.failed),
                len(result.skipped),
            )

        except ImportError as exc:
            _LOGGER.warning("Tool registration unavailable: %s", exc)
        except Exception as exc:
            _LOGGER.warning("Tool registration failed: %s", exc)

    # Restore workspace from saved session (or create default tab)
    if not skip_widgets:
        restored = coordinator.restore_workspace(context.settings, context.unsaved_cache)
        if not restored:
            # No workspace to restore - create a default empty tab
            coordinator.new_document()
            _LOGGER.debug("Created default empty document")
        else:
            _LOGGER.debug("Restored workspace from saved session")

    _LOGGER.info("Application bootstrap complete")
    return event_bus, coordinator, main_window


def create_application_headless(
    context: WindowContext,
    workspace: Any | None = None,
) -> tuple[EventBus, "AppCoordinator"]:
    """Create application components without UI for testing.

    This creates all domain stores and the coordinator without creating
    any Qt widgets, suitable for headless unit testing.

    Args:
        context: The window context with settings and shared state.
        workspace: Optional pre-created DocumentWorkspace.

    Returns:
        A tuple of (event_bus, coordinator).
    """
    event_bus, coordinator, _ = create_application(
        context,
        workspace=workspace,
        skip_widgets=True,
    )
    return event_bus, coordinator


__all__ = [
    "create_application",
    "create_application_headless",
]
