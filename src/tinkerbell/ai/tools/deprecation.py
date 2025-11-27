"""Deprecation utilities for legacy AI tools.

This module provides deprecation warnings and migration guidance for
legacy tool implementations that are being replaced by the new WS1-6
tool system.

WS7.1: Legacy Tool Deprecation
"""

from __future__ import annotations

import functools
import logging
import warnings
from typing import Any, Callable, TypeVar

LOGGER = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Mapping of legacy tool names to their replacements
LEGACY_TOOL_REPLACEMENTS: dict[str, str] = {
    # Legacy tool name -> New tool name
    "document_snapshot": "read_document",
    "document_edit": "replace_lines / insert_lines / delete_lines",
    "document_apply_patch": "replace_lines",
    "document_chunk": "read_document (with offset)",
    "document_find_text": "search_document",
    "document_outline": "get_outline",
    "document_insert": "insert_lines",
    "document_replace_all": "write_document",
    "search_replace": "find_and_replace",
    "selection_range": "(removed - use read_document)",
}


class DeprecatedToolWarning(DeprecationWarning):
    """Warning for deprecated AI tools."""

    pass


def deprecated_tool(
    replacement: str | None = None,
    removal_version: str = "2.0.0",
) -> Callable[[F], F]:
    """Decorator to mark a tool class or function as deprecated.
    
    Args:
        replacement: Name of the replacement tool.
        removal_version: Version when the tool will be removed.
        
    Returns:
        Decorator function.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tool_name = getattr(func, "__name__", str(func))
            msg = f"{tool_name} is deprecated and will be removed in version {removal_version}."
            if replacement:
                msg += f" Use {replacement} instead."
            warnings.warn(msg, DeprecatedToolWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator


def emit_deprecation_warning(
    tool_name: str,
    replacement: str | None = None,
    removal_version: str = "2.0.0",
) -> None:
    """Emit a deprecation warning for a legacy tool.
    
    Args:
        tool_name: Name of the deprecated tool.
        replacement: Name of the replacement tool.
        removal_version: Version when the tool will be removed.
    """
    msg = f"{tool_name} is deprecated and will be removed in version {removal_version}."
    if replacement:
        msg += f" Use {replacement} instead."
    warnings.warn(msg, DeprecatedToolWarning, stacklevel=3)


def get_replacement_tool(legacy_name: str) -> str | None:
    """Get the replacement tool name for a legacy tool.
    
    Args:
        legacy_name: Name of the legacy tool.
        
    Returns:
        Name of the replacement tool, or None if no replacement exists.
    """
    return LEGACY_TOOL_REPLACEMENTS.get(legacy_name)


# Enable deprecation warnings by default in development
warnings.filterwarnings("default", category=DeprecatedToolWarning)


__all__ = [
    "LEGACY_TOOL_REPLACEMENTS",
    "DeprecatedToolWarning",
    "deprecated_tool",
    "emit_deprecation_warning",
    "get_replacement_tool",
]
