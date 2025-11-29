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

    # OverlayManager for diff overlays (needs editor reference later)
    overlay_manager = OverlayManager(
        editor=editor,
        workspace=workspace,
        event_bus=event_bus,
    )
    _LOGGER.debug("Created overlay manager")

    # EmbeddingStore for document embeddings
    cache_root = None
    if context.settings_store:
        try:
            cache_root = context.settings_store.path.parent / "embedding_cache"
        except Exception:
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

    # Store adapters on coordinator for later access
    coordinator._bridge_adapter = bridge_adapter  # type: ignore[attr-defined]
    coordinator._settings_adapter = settings_adapter  # type: ignore[attr-defined]
    coordinator._telemetry_adapter = telemetry_adapter  # type: ignore[attr-defined]
    coordinator._embedding_store = embedding_store  # type: ignore[attr-defined]
    coordinator._outline_store = outline_store  # type: ignore[attr-defined]

    # =========================================================================
    # 6. Create Main Window
    # =========================================================================
    from .presentation.main_window import ThinMainWindow

    main_window = ThinMainWindow(
        event_bus=event_bus,
        coordinator=coordinator,
        skip_widgets=skip_widgets,
    )
    _LOGGER.debug("Created main window")

    # =========================================================================
    # 7. Wire Up Remaining Dependencies
    # =========================================================================
    # Set editor reference on overlay manager if window created widgets
    if main_window.editor is not None:
        overlay_manager.set_editor(main_window.editor)
        settings_adapter.set_editor(main_window.editor)
        _LOGGER.debug("Wired editor to overlay manager and settings adapter")

    # Set snapshot provider on coordinator
    coordinator.set_snapshot_provider(bridge_adapter)

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
