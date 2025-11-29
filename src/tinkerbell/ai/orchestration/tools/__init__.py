"""Tool system for the orchestration pipeline.

This package provides the tool registry, executor, and related types
for managing and executing tools within the turn pipeline.

Example:
    from tinkerbell.ai.orchestration.tools import (
        ToolRegistry,
        ToolExecutor,
        ToolSpec,
        SimpleTool,
    )

    # Create registry and register tools
    registry = ToolRegistry()
    registry.register_function(
        spec=ToolSpec(name="greet", description="Greet someone"),
        handler=lambda args: f"Hello, {args.get('name', 'World')}!",
    )

    # Create executor and run tools
    executor = ToolExecutor(registry)
    result = await executor.execute("greet", {"name": "Alice"})
"""

from .types import (
    Tool,
    ToolSpec,
    ToolHandler,
    AsyncToolHandler,
    SimpleTool,
    ToolCategory,
)

from .registry import (
    ToolRegistry,
    ToolRegistration,
    DuplicateToolError,
    ToolNotFoundError,
)

from .executor import (
    ToolExecutor,
    ExecutorConfig,
    ToolExecutionError,
)

__all__ = [
    # types.py
    "Tool",
    "ToolSpec",
    "ToolHandler",
    "AsyncToolHandler",
    "SimpleTool",
    "ToolCategory",
    # registry.py
    "ToolRegistry",
    "ToolRegistration",
    "DuplicateToolError",
    "ToolNotFoundError",
    # executor.py
    "ToolExecutor",
    "ExecutorConfig",
    "ToolExecutionError",
]
