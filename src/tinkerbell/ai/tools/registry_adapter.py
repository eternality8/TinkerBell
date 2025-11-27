"""Adapter to wire new WS1-6 tools into the existing AI controller interface.

This module bridges the new tool system (WS1-6) with the legacy registration
interface used by main_window.py. It allows gradual migration from legacy
tools to the new unified tool system.

WS7.1: Migration Adapter
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .tool_registry import (
    ToolSchema,
    ALL_TOOL_SCHEMAS,
    LIST_TABS_SCHEMA,
    READ_DOCUMENT_SCHEMA,
    SEARCH_DOCUMENT_SCHEMA,
    GET_OUTLINE_SCHEMA,
)
from .version import VersionManager, get_version_manager

# New tool imports (WS2-5)
from .read_document import create_read_document_tool
from .search_document import create_search_document_tool
from .get_outline import create_get_outline_tool

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolRegistrationFailure:
    """Represents a single tool registration error."""

    name: str
    error: Exception


class ToolRegistrationError(RuntimeError):
    """Aggregated exception emitted when some tools fail to register."""

    def __init__(self, failures: Sequence[ToolRegistrationFailure]):
        names = ", ".join(f.name for f in failures) or "unknown"
        super().__init__(f"Failed to register tool(s): {names}")
        self.failures = tuple(failures)


@dataclass(slots=True)
class NewToolRegistryContext:
    """Runtime dependencies for registering new tools.
    
    This is similar to ToolRegistryContext in registry.py but works
    with the new tool implementations.
    """

    controller: Any
    version_manager: VersionManager | None = None
    
    # Feature flags
    enable_read_tools: bool = True
    enable_write_tools: bool = True
    enable_analysis_tools: bool = False
    enable_transform_tools: bool = False


def register_new_tools(
    context: NewToolRegistryContext,
    *,
    register_fn: Callable[..., Any] | None = None,
) -> list[str]:
    """Register the new WS1-6 tools with the AI controller.
    
    This function registers the new tool implementations alongside or
    instead of legacy tools. It uses the same controller.register_tool()
    interface for compatibility.
    
    Args:
        context: Runtime dependencies and feature flags.
        register_fn: Optional override for the registration function.
        
    Returns:
        List of successfully registered tool names.
    """
    controller = context.controller
    if controller is None:
        return []

    register = register_fn or getattr(controller, "register_tool", None)
    if not callable(register):
        LOGGER.debug("AI controller does not expose register_tool; skipping new tool wiring.")
        return []

    failures: list[ToolRegistrationFailure] = []
    registered: list[str] = []
    
    # Get or create version manager
    version_manager = context.version_manager or get_version_manager()

    def _record_failure(name: str, exc: Exception) -> None:
        failures.append(ToolRegistrationFailure(name=name, error=exc))
        LOGGER.warning("Failed to register new tool %s: %s", name, exc, exc_info=True)

    def _safe_register(name: str, tool: Any, schema: ToolSchema) -> None:
        try:
            register(
                name,
                tool,
                description=schema.description,
                parameters=schema.to_json_schema(),
            )
        except Exception as exc:
            _record_failure(name, exc)
        else:
            registered.append(name)

    # -------------------------------------------------------------------------
    # WS2: Navigation & Reading Tools
    # -------------------------------------------------------------------------
    
    if context.enable_read_tools:
        # read_document - New tool (replaces document_snapshot for reading)
        try:
            read_doc_tool = create_read_document_tool(version_manager)
            _safe_register("read_document", read_doc_tool, READ_DOCUMENT_SCHEMA)
        except Exception as exc:
            _record_failure("read_document", exc)

        # search_document - New tool (replaces document_find_text)
        try:
            search_tool = create_search_document_tool(version_manager)
            _safe_register("search_document", search_tool, SEARCH_DOCUMENT_SCHEMA)
        except Exception as exc:
            _record_failure("search_document", exc)

        # get_outline - New tool (replaces document_outline)
        try:
            outline_tool = create_get_outline_tool(version_manager)
            _safe_register("get_outline", outline_tool, GET_OUTLINE_SCHEMA)
        except Exception as exc:
            _record_failure("get_outline", exc)

    # -------------------------------------------------------------------------
    # WS3: Writing Tools
    # -------------------------------------------------------------------------
    
    if context.enable_write_tools:
        # Note: Writing tools require DocumentEditor protocol implementation
        # which needs to be adapted from the bridge
        LOGGER.debug("Writing tools (WS3) require bridge adapter - skipping for now")

    # -------------------------------------------------------------------------
    # WS5: Subagent Tools (Analysis & Transform)
    # -------------------------------------------------------------------------
    
    if context.enable_analysis_tools:
        LOGGER.debug("Analysis tools (WS5) require AI client - skipping for now")

    if context.enable_transform_tools:
        LOGGER.debug("Transform tools (WS5) require AI client - skipping for now")

    # Log results
    if registered:
        LOGGER.info("New AI tools registered (WS1-6): %s", ", ".join(registered))
    if failures:
        LOGGER.warning(
            "Some new tools failed to register: %s",
            ", ".join(f.name for f in failures)
        )

    return registered


def get_new_tool_schemas() -> dict[str, ToolSchema]:
    """Return all new tool schemas for reference.
    
    This can be used to inspect available tools and their parameters
    without actually registering them.
    """
    return dict(ALL_TOOL_SCHEMAS)


__all__ = [
    "NewToolRegistryContext",
    "ToolRegistrationFailure",
    "ToolRegistrationError",
    "register_new_tools",
    "get_new_tool_schemas",
]
