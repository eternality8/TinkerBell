"""Base classes for AI tools.

This module provides abstract base classes that standardize tool interfaces,
error handling, and telemetry integration across all AI tools.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Protocol, TypeVar

from .version import VersionManager, VersionMismatchError, VersionToken
from .errors import ToolError, InvalidTabIdError


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class DocumentProvider(Protocol):
    """Protocol for accessing document content and state."""

    def get_document_text(self, tab_id: str | None = None) -> str:
        """Get the text content of a document."""
        ...

    def get_active_tab_id(self) -> str | None:
        """Get the ID of the currently active tab."""
        ...

    def get_document_content(self, tab_id: str) -> str | None:
        """Get the content of a specific tab, or None if not found."""
        ...

    def get_document_metadata(self, tab_id: str) -> dict[str, Any] | None:
        """Get metadata for a specific tab (path, language, etc.)."""
        ...


class TelemetryEmitter(Protocol):
    """Protocol for emitting telemetry events."""

    def emit(self, event_name: str, payload: Mapping[str, Any]) -> None:
        """Emit a telemetry event with the given payload."""
        ...


@dataclass(slots=True)
class ToolResult:
    """Standardized result container for tool execution.

    Attributes:
        success: Whether the tool completed successfully.
        data: The result data if successful.
        error: Error details if unsuccessful.
        duration_ms: Execution time in milliseconds.
        metadata: Additional metadata about the execution.
    """

    success: bool
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for JSON responses."""
        if self.success:
            result = dict(self.data) if self.data else {}
            if self.metadata:
                result["_metadata"] = dict(self.metadata)
            return result
        else:
            result = self.error.to_dict() if self.error else {"error": "unknown", "message": "Unknown error"}
            if self.metadata:
                result["_metadata"] = dict(self.metadata)
            return result


@dataclass(slots=True)
class ToolContext:
    """Runtime context provided to tool execution.

    Attributes:
        tab_id: Target tab ID (None for active tab).
        document_provider: Access to document content.
        version_manager: Version tracking system.
        telemetry: Optional telemetry emitter.
        request_id: Unique identifier for this request (for tracing).
    """

    document_provider: DocumentProvider
    version_manager: VersionManager
    tab_id: str | None = None
    telemetry: TelemetryEmitter | None = None
    request_id: str | None = None

    def resolve_tab_id(self, explicit_tab_id: str | None = None) -> str | None:
        """Resolve the effective tab ID, preferring explicit over context over active.

        Args:
            explicit_tab_id: Explicitly provided tab ID from parameters.

        Returns:
            The resolved tab ID, or None if no tab can be resolved.
        """
        # Priority: explicit parameter > context tab_id > active tab
        if explicit_tab_id:
            return explicit_tab_id
        if self.tab_id:
            return self.tab_id
        return self.document_provider.get_active_tab_id()

    def require_tab_id(self, explicit_tab_id: str | None = None) -> str:
        """Resolve the effective tab ID, raising if none available.

        Args:
            explicit_tab_id: Explicitly provided tab ID from parameters.

        Returns:
            The resolved tab ID.

        Raises:
            InvalidTabIdError: If no tab can be resolved.
        """
        resolved = self.resolve_tab_id(explicit_tab_id)
        if not resolved:
            raise InvalidTabIdError(
                message="No tab_id provided and no active tab available",
                tab_id=explicit_tab_id,
            )
        return resolved


class BaseTool(ABC):
    """Abstract base class for all AI tools.

    Provides standardized execution flow with:
    - Automatic timing and telemetry
    - Consistent error handling
    - Input validation hooks
    - Summarizability flag for context compaction

    Subclasses must implement:
    - `name`: Tool identifier
    - `execute()`: Core tool logic

    Example:
        class MyTool(BaseTool):
            name = "my_tool"

            def execute(self, context: ToolContext, params: dict) -> dict:
                return {"result": "done"}
    """

    # Class-level configuration
    name: ClassVar[str] = ""
    summarizable: ClassVar[bool] = True

    def run(
        self,
        context: ToolContext,
        params: Mapping[str, Any] | None = None,
    ) -> ToolResult:
        """Execute the tool with standardized error handling and telemetry.

        This is the main entry point that wraps `execute()` with timing,
        error handling, and telemetry emission.

        Args:
            context: Runtime context including document access and version manager.
            params: Tool-specific parameters.

        Returns:
            ToolResult containing success/failure status and data or error.
        """
        start_time = time.perf_counter()
        params = dict(params) if params else {}

        try:
            # Validate inputs
            self.validate(params)

            # Execute the tool
            result_data = self.execute(context, params)

            duration_ms = (time.perf_counter() - start_time) * 1000.0
            result = ToolResult(
                success=True,
                data=result_data,
                duration_ms=duration_ms,
            )

            # Emit success telemetry
            self._emit_telemetry(context, params, result)

            return result

        except ToolError as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            result = ToolResult(
                success=False,
                error=exc,
                duration_ms=duration_ms,
            )
            self._emit_telemetry(context, params, result)
            return result

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            LOGGER.exception("Tool %s failed unexpectedly", self.name)
            error = ToolError(
                error_code="internal_error",
                message=f"Internal error: {exc}",
            )
            result = ToolResult(
                success=False,
                error=error,
                duration_ms=duration_ms,
            )
            self._emit_telemetry(context, params, result)
            return result

    @abstractmethod
    def execute(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool's core logic.

        Subclasses implement this method to perform the actual work.

        Args:
            context: Runtime context including document access.
            params: Validated tool parameters.

        Returns:
            Dictionary containing the tool's result data.

        Raises:
            ToolError: For expected error conditions.
            Exception: For unexpected errors (will be wrapped).
        """
        ...

    def validate(self, params: dict[str, Any]) -> None:
        """Validate tool parameters before execution.

        Override to add custom validation. Raise ToolError for invalid inputs.

        Args:
            params: Parameters to validate.

        Raises:
            ToolError: If parameters are invalid.
        """
        pass

    def _emit_telemetry(
        self,
        context: ToolContext,
        params: dict[str, Any],
        result: ToolResult,
    ) -> None:
        """Emit telemetry event for this tool execution."""
        if context.telemetry is None:
            return

        payload: dict[str, Any] = {
            "tool": self.name,
            "success": result.success,
            "duration_ms": round(result.duration_ms, 3),
        }

        if context.request_id:
            payload["request_id"] = context.request_id
        if context.tab_id:
            payload["tab_id"] = context.tab_id
        if not result.success and result.error:
            payload["error_code"] = result.error.error_code
            payload["error_message"] = result.error.message

        try:
            context.telemetry.emit(f"tool.{self.name}", payload)
        except Exception:
            LOGGER.debug("Failed to emit telemetry for tool %s", self.name, exc_info=True)


class ReadOnlyTool(BaseTool):
    """Base class for tools that read document state without modifications.

    Read-only tools:
    - Do not require version token validation
    - Always include the current version in their response
    - Cannot modify document content

    Example:
        class ReadDocumentTool(ReadOnlyTool):
            name = "read_document"

            def execute(self, context: ToolContext, params: dict) -> dict:
                tab_id = context.resolve_tab_id()
                text = context.document_provider.get_document_text(tab_id)
                version = context.version_manager.get_current_token(tab_id)
                return {
                    "content": text,
                    "version": version.to_string() if version else None,
                }
    """

    def execute(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the read operation and attach version information.

        Subclasses should override `read()` instead of this method.
        """
        result = self.read(context, params)

        # Ensure version is included in response
        if "version" not in result:
            try:
                tab_id = context.resolve_tab_id()
                token = context.version_manager.get_current_token(tab_id)
                if token:
                    result["version"] = token.to_string()
            except Exception:
                pass  # Version attachment is best-effort

        return result

    @abstractmethod
    def read(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        """Perform the read operation.

        Args:
            context: Runtime context.
            params: Tool parameters.

        Returns:
            Dictionary with read results. Version will be auto-attached.
        """
        ...


class WriteTool(BaseTool):
    """Base class for tools that modify document content.

    Write tools:
    - Require version token validation before making changes
    - Automatically increment the version after successful edits
    - Support dry-run mode to preview changes without applying them
    - Return the new version token in their response

    Subclasses must implement:
    - `write()`: The actual modification logic
    - `preview()`: Optional preview logic for dry-run mode

    Example:
        class ReplaceLinesTool(WriteTool):
            name = "replace_lines"

            def write(self, context, params, token):
                # Apply the replacement
                new_text = apply_replacement(...)
                return {
                    "lines_affected": {"removed": 5, "added": 3},
                }
    """

    def execute(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the write operation with version validation.

        1. Validates the provided version token
        2. If dry_run=True, returns preview without applying changes
        3. Otherwise, applies changes and increments version
        """
        # Extract and validate version token
        version_str = params.get("version")
        if not version_str:
            from .errors import ContentRequiredError
            raise ContentRequiredError(
                message="Write operations require a 'version' token from a previous read",
                field_name="version",
            )

        try:
            token = VersionToken.from_string(version_str)
        except ValueError as exc:
            from .errors import InvalidVersionTokenError
            raise InvalidVersionTokenError(
                message=str(exc),
                token=version_str,
            ) from exc

        # Validate token is current
        try:
            context.version_manager.validate_token(token)
        except VersionMismatchError as exc:
            from .errors import VersionMismatchToolError
            raise VersionMismatchToolError.from_version_error(exc) from exc

        # Check for dry-run mode
        dry_run = params.get("dry_run", False)
        if dry_run:
            preview_result = self.preview(context, params, token)
            preview_result["dry_run"] = True
            preview_result["version"] = token.to_string()  # Version not consumed
            return preview_result

        # Execute the write
        result = self.write(context, params, token)

        # Increment version after successful write
        new_text = result.pop("_new_text", None)
        if new_text is not None:
            from .version import compute_content_hash
            new_hash = compute_content_hash(new_text)
            new_token = context.version_manager.increment_version(token.tab_id, new_hash)
            result["version"] = new_token.to_string()
        else:
            # If no new text provided, still include current version
            current = context.version_manager.get_current_token(token.tab_id)
            result["version"] = current.to_string() if current else token.to_string()

        return result

    @abstractmethod
    def write(
        self,
        context: ToolContext,
        params: dict[str, Any],
        token: VersionToken,
    ) -> dict[str, Any]:
        """Perform the write operation.

        Args:
            context: Runtime context.
            params: Tool parameters (version already validated).
            token: Validated version token.

        Returns:
            Dictionary with write results. Include `_new_text` key with
            the updated document content to trigger version increment.

        Raises:
            ToolError: For write-related errors.
        """
        ...

    def preview(
        self,
        context: ToolContext,
        params: dict[str, Any],
        token: VersionToken,
    ) -> dict[str, Any]:
        """Preview the write operation without applying changes.

        Override to provide custom preview logic. Default returns
        a basic preview message.

        Args:
            context: Runtime context.
            params: Tool parameters.
            token: The version token (not consumed in preview).

        Returns:
            Dictionary with preview information.
        """
        return {
            "preview": "Write preview not implemented for this tool",
        }


class SubagentTool(BaseTool):
    """Base class for tools that spawn subagents for complex operations.

    Subagent tools:
    - Can spawn multiple parallel workers for chunk processing
    - Provide progress tracking during execution
    - Aggregate results from multiple subagents
    - Handle partial failures gracefully

    Subclasses must implement:
    - `plan()`: Determine subagent tasks to spawn
    - `aggregate()`: Combine subagent results

    Example:
        class AnalyzeDocumentTool(SubagentTool):
            name = "analyze_document"

            def plan(self, context, params):
                # Return list of chunk specs to analyze
                return [{"chunk_id": "1", ...}, ...]

            def aggregate(self, results):
                # Combine chunk analysis results
                return {"characters": [...], "themes": [...]}
    """

    # Maximum concurrent subagents
    max_parallel: ClassVar[int] = 4

    def execute(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the subagent workflow.

        1. Plan the subagent tasks
        2. Execute subagents (potentially in parallel)
        3. Aggregate results
        """
        # Plan the work
        tasks = self.plan(context, params)
        if not tasks:
            return {"status": "no_tasks", "results": []}

        # Execute subagents
        results = []
        errors = []
        for task in tasks:
            try:
                result = self.execute_subagent(context, task)
                results.append(result)
            except Exception as exc:
                LOGGER.warning("Subagent task failed: %s", exc)
                errors.append({"task": task, "error": str(exc)})

        # Aggregate results
        aggregated = self.aggregate(results)

        # Include error info if any
        if errors:
            aggregated["partial_errors"] = errors
            aggregated["completed"] = len(results)
            aggregated["total"] = len(tasks)

        return aggregated

    @abstractmethod
    def plan(
        self,
        context: ToolContext,
        params: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Plan the subagent tasks to execute.

        Args:
            context: Runtime context.
            params: Tool parameters.

        Returns:
            List of task specifications for subagents.
        """
        ...

    @abstractmethod
    def aggregate(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate results from completed subagents.

        Args:
            results: List of results from individual subagents.

        Returns:
            Aggregated result dictionary.
        """
        ...

    def execute_subagent(
        self,
        context: ToolContext,
        task: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a single subagent task.

        Override for custom subagent execution logic.

        Args:
            context: Runtime context.
            task: Task specification from plan().

        Returns:
            Subagent result.
        """
        raise NotImplementedError("Subclasses must implement execute_subagent()")


__all__ = [
    "BaseTool",
    "ReadOnlyTool",
    "WriteTool",
    "SubagentTool",
    "ToolResult",
    "ToolContext",
    "DocumentProvider",
    "TelemetryEmitter",
]
