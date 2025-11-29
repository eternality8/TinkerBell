"""Tool registry for the orchestration pipeline.

This module provides a registry for managing tool registrations,
allowing tools to be registered, retrieved, and listed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Mapping, Sequence

from .types import Tool, ToolSpec, SimpleTool, ToolHandler, AsyncToolHandler

__all__ = [
    "ToolRegistry",
    "ToolRegistration",
    "DuplicateToolError",
    "ToolNotFoundError",
]

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class DuplicateToolError(Exception):
    """Raised when attempting to register a tool with a name that already exists."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Tool '{name}' is already registered")


class ToolNotFoundError(Exception):
    """Raised when a requested tool is not found in the registry."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Tool '{name}' not found")


# -----------------------------------------------------------------------------
# Tool Registration
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class ToolRegistration:
    """Record of a registered tool.

    Attributes:
        name: Tool name.
        tool: The tool implementation.
        spec: Tool specification.
        enabled: Whether the tool is currently enabled.
        metadata: Additional registration metadata.
    """

    name: str
    tool: Tool
    spec: ToolSpec
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Tool Registry
# -----------------------------------------------------------------------------


class ToolRegistry:
    """Registry for managing tool registrations.

    Provides methods for registering, retrieving, and listing tools.
    Supports both Tool implementations and simple function handlers.

    Example:
        registry = ToolRegistry()

        # Register a Tool instance
        registry.register(my_tool)

        # Register a function with spec
        registry.register_function(
            spec=ToolSpec(name="greet", description="Greet"),
            handler=lambda args: f"Hello, {args['name']}!",
        )

        # Get and execute a tool
        tool = registry.get("greet")
        result = await tool.execute({"name": "World"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolRegistration] = {}

    def register(
        self,
        tool: Tool,
        *,
        enabled: bool = True,
        allow_override: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolRegistration:
        """Register a tool implementation.

        Args:
            tool: The tool to register.
            enabled: Whether the tool is enabled.
            allow_override: If True, allows overriding existing registration.
            metadata: Additional metadata to store with registration.

        Returns:
            The tool registration record.

        Raises:
            DuplicateToolError: If tool name already registered and allow_override is False.
        """
        name = tool.name
        if name in self._tools and not allow_override:
            raise DuplicateToolError(name)

        registration = ToolRegistration(
            name=name,
            tool=tool,
            spec=tool.spec,
            enabled=enabled,
            metadata=dict(metadata) if metadata else {},
        )
        self._tools[name] = registration
        LOGGER.debug("Registered tool: %s", name)
        return registration

    def register_function(
        self,
        spec: ToolSpec,
        handler: ToolHandler | AsyncToolHandler,
        *,
        enabled: bool = True,
        allow_override: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> ToolRegistration:
        """Register a function as a tool.

        Convenience method for registering simple function handlers
        without implementing a full Tool class.

        Args:
            spec: Tool specification.
            handler: Function to handle tool calls (sync or async).
            enabled: Whether the tool is enabled.
            allow_override: If True, allows overriding existing registration.
            metadata: Additional metadata.

        Returns:
            The tool registration record.

        Raises:
            DuplicateToolError: If tool name already registered.
        """
        tool = SimpleTool(spec=spec, handler=handler)
        return self.register(
            tool,
            enabled=enabled,
            allow_override=allow_override,
            metadata=metadata,
        )

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name.

        Args:
            name: The tool name to unregister.

        Returns:
            True if the tool was unregistered, False if not found.
        """
        if name in self._tools:
            del self._tools[name]
            LOGGER.debug("Unregistered tool: %s", name)
            return True
        return False

    def get(self, name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            name: The tool name.

        Returns:
            The tool if found and enabled, None otherwise.
        """
        registration = self._tools.get(name)
        if registration is None or not registration.enabled:
            return None
        return registration.tool

    def get_required(self, name: str) -> Tool:
        """Get a tool by name, raising if not found.

        Args:
            name: The tool name.

        Returns:
            The tool.

        Raises:
            ToolNotFoundError: If tool not found or disabled.
        """
        tool = self.get(name)
        if tool is None:
            raise ToolNotFoundError(name)
        return tool

    def get_registration(self, name: str) -> ToolRegistration | None:
        """Get the full registration record for a tool.

        Args:
            name: The tool name.

        Returns:
            The registration record if found, None otherwise.
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool is registered and enabled.

        Args:
            name: The tool name.

        Returns:
            True if the tool exists and is enabled.
        """
        registration = self._tools.get(name)
        return registration is not None and registration.enabled

    def list_tools(self, *, include_disabled: bool = False) -> list[ToolSpec]:
        """List all registered tool specifications.

        Args:
            include_disabled: If True, includes disabled tools.

        Returns:
            List of tool specifications.
        """
        specs: list[ToolSpec] = []
        for registration in self._tools.values():
            if registration.enabled or include_disabled:
                specs.append(registration.spec)
        return specs

    def list_names(self, *, include_disabled: bool = False) -> list[str]:
        """List all registered tool names.

        Args:
            include_disabled: If True, includes disabled tools.

        Returns:
            List of tool names.
        """
        names: list[str] = []
        for registration in self._tools.values():
            if registration.enabled or include_disabled:
                names.append(registration.name)
        return names

    def list_registrations(
        self,
        *,
        include_disabled: bool = False,
    ) -> list[ToolRegistration]:
        """List all tool registrations.

        Args:
            include_disabled: If True, includes disabled tools.

        Returns:
            List of registration records.
        """
        registrations: list[ToolRegistration] = []
        for registration in self._tools.values():
            if registration.enabled or include_disabled:
                registrations.append(registration)
        return registrations

    def get_openai_tools(
        self,
        *,
        filter_names: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Get tool definitions in OpenAI format.

        Args:
            filter_names: If provided, only include these tools.

        Returns:
            List of tool definitions for OpenAI API.
        """
        tools: list[dict[str, Any]] = []
        for registration in self._tools.values():
            if not registration.enabled:
                continue
            if filter_names is not None and registration.name not in filter_names:
                continue
            tools.append(registration.spec.to_openai_tool())
        return tools

    def enable(self, name: str) -> bool:
        """Enable a tool by name.

        Args:
            name: The tool name.

        Returns:
            True if the tool was enabled, False if not found.
        """
        registration = self._tools.get(name)
        if registration is None:
            return False
        registration.enabled = True
        return True

    def disable(self, name: str) -> bool:
        """Disable a tool by name.

        Args:
            name: The tool name.

        Returns:
            True if the tool was disabled, False if not found.
        """
        registration = self._tools.get(name)
        if registration is None:
            return False
        registration.enabled = False
        return True

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()

    def __len__(self) -> int:
        """Get the number of registered tools (including disabled)."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if a tool name is registered."""
        return name in self._tools
