"""Tool system types for the orchestration pipeline.

This module defines the core types used by the tool execution system.
These types are designed to work with the pipeline's ToolExecutor protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Mapping, Protocol, runtime_checkable

__all__ = [
    "ToolSpec",
    "ToolHandler",
    "AsyncToolHandler",
    "Tool",
    "ToolCategory",
]


# -----------------------------------------------------------------------------
# Tool Categories
# -----------------------------------------------------------------------------


class ToolCategory:
    """Standard tool categories for organization."""

    READ = "read"
    WRITE = "write"
    SEARCH = "search"
    ANALYSIS = "analysis"
    UTILITY = "utility"
    SYSTEM = "system"


# -----------------------------------------------------------------------------
# Tool Specification
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ToolSpec:
    """Specification for a tool's interface.

    This is used to generate the tool definition for the model API
    and for documentation purposes.

    Attributes:
        name: Unique identifier for the tool.
        description: Human-readable description of what the tool does.
        parameters: JSON Schema for the tool's parameters.
        category: Tool category for organization.
        requires_document: Whether the tool requires an active document.
        is_write: Whether the tool modifies document state.
    """

    name: str
    description: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    category: str = ToolCategory.UTILITY
    requires_document: bool = False
    is_write: bool = False

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to OpenAI tool definition format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters) if self.parameters else {
                    "type": "object",
                    "properties": {},
                },
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters) if self.parameters else {},
            "category": self.category,
            "requires_document": self.requires_document,
            "is_write": self.is_write,
        }


# -----------------------------------------------------------------------------
# Tool Handler Types
# -----------------------------------------------------------------------------

# Synchronous tool handler
ToolHandler = Callable[[Mapping[str, Any]], Any]

# Asynchronous tool handler
AsyncToolHandler = Callable[[Mapping[str, Any]], Coroutine[Any, Any, Any]]


# -----------------------------------------------------------------------------
# Tool Protocol
# -----------------------------------------------------------------------------


@runtime_checkable
class Tool(Protocol):
    """Protocol for tool implementations.

    Tools can be implemented as classes conforming to this protocol,
    or as simple functions registered with a ToolSpec.

    Attributes:
        name: Unique identifier for the tool.
        spec: Tool specification with metadata and parameters.
    """

    @property
    def name(self) -> str:
        """Get the tool's unique name."""
        ...

    @property
    def spec(self) -> ToolSpec:
        """Get the tool's specification."""
        ...

    async def execute(self, arguments: Mapping[str, Any]) -> Any:
        """Execute the tool with the given arguments.

        Args:
            arguments: Tool arguments as a dictionary.

        Returns:
            The tool's result (any type).

        Raises:
            Exception: If tool execution fails.
        """
        ...


# -----------------------------------------------------------------------------
# Simple Tool Implementation
# -----------------------------------------------------------------------------


@dataclass
class SimpleTool:
    """Simple tool implementation wrapping a callable.

    This provides a convenient way to create tools from functions
    without implementing a full class.

    Example:
        def my_handler(args: dict) -> str:
            return f"Hello, {args.get('name', 'World')}!"

        tool = SimpleTool(
            spec=ToolSpec(name="greet", description="Greet someone"),
            handler=my_handler,
        )
    """

    spec: ToolSpec
    handler: ToolHandler | AsyncToolHandler
    _is_async: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        import asyncio
        self._is_async = asyncio.iscoroutinefunction(self.handler)

    @property
    def name(self) -> str:
        """Get the tool's name from its spec."""
        return self.spec.name

    async def execute(self, arguments: Mapping[str, Any]) -> Any:
        """Execute the tool handler."""
        if self._is_async:
            return await self.handler(arguments)  # type: ignore
        else:
            return self.handler(arguments)
