"""Tool executor for the orchestration pipeline.

This module provides the ToolExecutor class that conforms to the
pipeline's ToolExecutor protocol and handles tool execution with
proper error handling and logging.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .registry import ToolRegistry, ToolNotFoundError
from .types import Tool

__all__ = [
    "ToolExecutor",
    "ExecutorConfig",
    "ToolExecutionError",
]

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class ToolExecutionError(Exception):
    """Raised when tool execution fails."""

    def __init__(
        self,
        message: str,
        tool_name: str = "",
        cause: Exception | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.cause = cause
        super().__init__(message)


# -----------------------------------------------------------------------------
# Executor Configuration
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class ExecutorConfig:
    """Configuration for the tool executor.

    Attributes:
        default_timeout: Default timeout for tool execution in seconds.
        log_arguments: Whether to log tool arguments (may contain sensitive data).
        log_results: Whether to log tool results.
        strict_mode: If True, raise on unknown tools; if False, return error result.
    """

    default_timeout: float | None = 30.0
    log_arguments: bool = False
    log_results: bool = False
    strict_mode: bool = False


# -----------------------------------------------------------------------------
# Tool Executor
# -----------------------------------------------------------------------------


class ToolExecutor:
    """Executor for running tools from a registry.

    This class conforms to the pipeline's ToolExecutor protocol and
    provides the execute() method for running tools by name.

    Example:
        registry = ToolRegistry()
        registry.register(my_tool)

        executor = ToolExecutor(registry)
        result = await executor.execute("my_tool", {"arg": "value"})
    """

    def __init__(
        self,
        registry: ToolRegistry,
        config: ExecutorConfig | None = None,
    ) -> None:
        """Initialize the executor.

        Args:
            registry: The tool registry to use.
            config: Optional executor configuration.
        """
        self._registry = registry
        self._config = config or ExecutorConfig()

    @property
    def registry(self) -> ToolRegistry:
        """Get the underlying tool registry."""
        return self._registry

    @property
    def config(self) -> ExecutorConfig:
        """Get the executor configuration."""
        return self._config

    async def execute(
        self,
        name: str,
        arguments: Mapping[str, Any],
        *,
        call_id: str = "",
        timeout: float | None = None,
    ) -> Any:
        """Execute a tool by name with arguments.

        This method conforms to the pipeline's ToolExecutor protocol.

        Args:
            name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            call_id: Optional call ID for tracing.
            timeout: Optional timeout override.

        Returns:
            The tool's result.

        Raises:
            ToolNotFoundError: If tool not found and strict_mode is True.
            ToolExecutionError: If execution fails.
            asyncio.TimeoutError: If execution times out.
        """
        # Log the call
        if self._config.log_arguments:
            LOGGER.debug(
                "Executing tool %s (call_id=%s) with arguments: %s",
                name,
                call_id,
                arguments,
            )
        else:
            LOGGER.debug("Executing tool %s (call_id=%s)", name, call_id)

        # Get the tool
        tool = self._registry.get(name)
        if tool is None:
            error_msg = f"Tool '{name}' not found or disabled"
            if self._config.strict_mode:
                raise ToolNotFoundError(name)
            LOGGER.warning(error_msg)
            return {"error": "tool_not_found", "message": error_msg}

        # Determine timeout
        effective_timeout = timeout if timeout is not None else self._config.default_timeout

        # Execute with timeout
        start_time = time.perf_counter()
        try:
            if effective_timeout is not None and effective_timeout > 0:
                result = await asyncio.wait_for(
                    tool.execute(arguments),
                    timeout=effective_timeout,
                )
            else:
                result = await tool.execute(arguments)

            duration_ms = (time.perf_counter() - start_time) * 1000

            if self._config.log_results:
                LOGGER.debug(
                    "Tool %s completed in %.1fms with result: %s",
                    name,
                    duration_ms,
                    result,
                )
            else:
                LOGGER.debug("Tool %s completed in %.1fms", name, duration_ms)

            return result

        except asyncio.TimeoutError:
            duration_ms = (time.perf_counter() - start_time) * 1000
            LOGGER.warning(
                "Tool %s timed out after %.1fms (timeout=%.1fs)",
                name,
                duration_ms,
                effective_timeout,
            )
            raise

        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            LOGGER.warning(
                "Tool %s failed after %.1fms: %s",
                name,
                duration_ms,
                e,
            )
            raise ToolExecutionError(
                message=str(e),
                tool_name=name,
                cause=e,
            ) from e

    async def execute_many(
        self,
        calls: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        parallel: bool = False,
        timeout: float | None = None,
    ) -> list[Any]:
        """Execute multiple tool calls.

        Args:
            calls: Sequence of (name, arguments) tuples.
            parallel: Whether to execute in parallel.
            timeout: Optional timeout per call.

        Returns:
            List of results in the same order as calls.
        """
        if parallel:
            tasks = [
                self.execute(name, args, timeout=timeout)
                for name, args in calls
            ]
            return list(await asyncio.gather(*tasks, return_exceptions=True))
        else:
            results: list[Any] = []
            for name, args in calls:
                try:
                    result = await self.execute(name, args, timeout=timeout)
                    results.append(result)
                except Exception as e:
                    results.append(e)
            return results

    def has_tool(self, name: str) -> bool:
        """Check if a tool is available.

        Args:
            name: The tool name.

        Returns:
            True if the tool exists and is enabled.
        """
        return self._registry.has(name)

    def list_tools(self) -> list[str]:
        """List available tool names.

        Returns:
            List of enabled tool names.
        """
        return self._registry.list_names()
