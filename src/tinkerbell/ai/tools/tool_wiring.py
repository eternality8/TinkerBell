"""Tool Wiring Module - Extracted from main_window.py.

This module handles all AI tool registration, separating tool wiring
concerns from the main window UI logic. It provides a clean interface
for registering both legacy and new tools.

WS7.1: Tool Wiring Extraction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence, Mapping, runtime_checkable

LOGGER = logging.getLogger(__name__)


# =============================================================================
# Provider Protocols - What tool wiring needs from the application
# =============================================================================


@runtime_checkable
class DocumentBridge(Protocol):
    """Protocol for document bridge operations."""

    def get_document(self, tab_id: str) -> Any: ...
    def get_active_document(self) -> Any: ...
    def list_tabs(self) -> Sequence[Mapping[str, Any]]: ...


@runtime_checkable
class WorkspaceProvider(Protocol):
    """Protocol for workspace operations."""

    def find_document_by_id(self, document_id: str) -> Any: ...
    def active_document(self) -> Any | None: ...


@runtime_checkable
class SelectionProvider(Protocol):
    """Protocol for selection/gateway operations."""

    def get_selection(self, tab_id: str | None = None) -> Any: ...


@runtime_checkable
class AIControllerProvider(Protocol):
    """Protocol for AI controller access."""

    def register_tool(
        self,
        name: str,
        tool: Any,
        *,
        description: str,
        parameters: Mapping[str, Any],
    ) -> None: ...

    def unregister_tool(self, name: str) -> None: ...


# =============================================================================
# Tool Wiring Context - Dependencies passed to tool registration
# =============================================================================


@dataclass
class ToolWiringContext:
    """All dependencies needed for tool registration.
    
    This dataclass captures everything the tool wiring system needs
    from the main window, allowing tool registration to be fully
    decoupled from UI code.
    """

    # Core providers
    controller: AIControllerProvider | None
    bridge: DocumentBridge
    workspace: WorkspaceProvider
    selection_gateway: SelectionProvider | None = None

    # Resolvers (lazy initialization callbacks)
    embedding_index_resolver: Callable[[], Any] | None = None
    active_document_provider: Callable[[], Any] | None = None


# =============================================================================
# Tool Registration Result
# =============================================================================


@dataclass
class ToolRegistrationResult:
    """Result of tool registration attempt."""

    registered: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if no failures occurred."""
        return len(self.failed) == 0

    def __str__(self) -> str:
        parts = []
        if self.registered:
            parts.append(f"registered={self.registered}")
        if self.failed:
            parts.append(f"failed={self.failed}")
        if self.skipped:
            parts.append(f"skipped={self.skipped}")
        return f"ToolRegistrationResult({', '.join(parts)})"


# =============================================================================
# Legacy Tool Registration (DEPRECATED - NO-OP)
# =============================================================================


def register_legacy_tools(ctx: ToolWiringContext) -> ToolRegistrationResult:
    """Register legacy tools with the AI controller.
    
    .. deprecated:: WS8.1
        Use :func:`register_new_tools` instead. Legacy tools have been
        removed in WS8.1. This function now returns an empty result
        and emits a deprecation warning.
    
    Args:
        ctx: Tool wiring context (unused).
        
    Returns:
        Empty ToolRegistrationResult.
    """
    import warnings
    warnings.warn(
        "register_legacy_tools is deprecated and is now a no-op. "
        "Use register_new_tools instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ToolRegistrationResult()


# =============================================================================
# New Tool Registration (WS1-6 Tools)
# =============================================================================


def register_new_tools(ctx: ToolWiringContext) -> ToolRegistrationResult:
    """Register new WS1-6 tools with the AI controller.
    
    This function registers the new tool implementations created in
    workstreams 1-6. These are designed to replace the legacy tools.
    
    Tools are registered in TWO places:
    1. Controller's legacy registry (for LLM schema generation)
    2. New ToolRegistry (for ToolDispatcher routing)
    
    Args:
        ctx: Tool wiring context with all dependencies.
        
    Returns:
        Result indicating which tools were registered/failed/skipped.
    """
    from .version import get_version_manager
    from .list_tabs import create_list_tabs_tool
    from .read_document import create_read_document_tool
    from .search_document import create_search_document_tool
    from .get_outline import create_get_outline_tool
    from .tool_registry import (
        LIST_TABS_SCHEMA,
        READ_DOCUMENT_SCHEMA,
        SEARCH_DOCUMENT_SCHEMA,
        GET_OUTLINE_SCHEMA,
        get_tool_registry,
    )

    result = ToolRegistrationResult()
    
    controller = ctx.controller
    if controller is None:
        LOGGER.debug("No AI controller available; skipping new tool registration.")
        return result

    register = getattr(controller, "register_tool", None)
    if not callable(register):
        LOGGER.debug("AI controller does not expose register_tool; skipping.")
        return result

    version_manager = get_version_manager()
    tool_registry = get_tool_registry()

    def _safe_register(name: str, tool: Any, schema: Any) -> bool:
        try:
            # Register with controller's legacy registry (for LLM schema)
            register(
                name,
                tool,
                description=schema.description,
                parameters=schema.to_json_schema(),
            )
            # Also register with new ToolRegistry (for ToolDispatcher)
            tool_registry.register(tool, schema=schema)
            result.registered.append(name)
            return True
        except Exception as exc:
            LOGGER.warning("Failed to register %s: %s", name, exc)
            result.failed.append(name)
            return False

    # -------------------------------------------------------------------------
    # list_tabs
    # -------------------------------------------------------------------------
    try:
        list_tabs_tool = create_list_tabs_tool(provider=ctx.bridge)
        _safe_register("list_tabs", list_tabs_tool, LIST_TABS_SCHEMA)
    except Exception as exc:
        LOGGER.warning("Failed to create list_tabs tool: %s", exc)
        result.failed.append("list_tabs")

    # -------------------------------------------------------------------------
    # read_document (replaces document_snapshot)
    # -------------------------------------------------------------------------
    try:
        read_tool = create_read_document_tool(version_manager)
        _safe_register("read_document", read_tool, READ_DOCUMENT_SCHEMA)
    except Exception as exc:
        LOGGER.warning("Failed to create read_document tool: %s", exc)
        result.failed.append("read_document")

    # -------------------------------------------------------------------------
    # search_document (replaces document_find_text)
    # -------------------------------------------------------------------------
    try:
        search_tool = create_search_document_tool(version_manager)
        _safe_register("search_document", search_tool, SEARCH_DOCUMENT_SCHEMA)
    except Exception as exc:
        LOGGER.warning("Failed to create search_document tool: %s", exc)
        result.failed.append("search_document")

    # -------------------------------------------------------------------------
    # get_outline (replaces document_outline)
    # -------------------------------------------------------------------------
    try:
        outline_tool = create_get_outline_tool(version_manager)
        _safe_register("get_outline", outline_tool, GET_OUTLINE_SCHEMA)
    except Exception as exc:
        LOGGER.warning("Failed to create get_outline tool: %s", exc)
        result.failed.append("get_outline")

    LOGGER.info("New tool registration complete: %s", result)
    return result


# =============================================================================
# Unregistration
# =============================================================================


def unregister_tools(
    controller: Any,
    tool_names: Sequence[str],
) -> list[str]:
    """Unregister tools from the AI controller.
    
    Args:
        controller: AI controller instance.
        tool_names: Names of tools to unregister.
        
    Returns:
        List of successfully unregistered tool names.
    """
    if controller is None:
        return []

    unregister = getattr(controller, "unregister_tool", None)
    if not callable(unregister):
        return []

    unregistered = []
    for name in tool_names:
        try:
            unregister(name)
            unregistered.append(name)
        except Exception:
            LOGGER.debug("Failed to unregister %s", name, exc_info=True)

    return unregistered


__all__ = [
    # Protocols
    "DocumentBridge",
    "WorkspaceProvider",
    "SelectionProvider",
    "AIControllerProvider",
    # Context
    "ToolWiringContext",
    "ToolRegistrationResult",
    # Registration functions
    "register_legacy_tools",
    "register_new_tools",
    "unregister_tools",
]
