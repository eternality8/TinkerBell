"""Tool Dispatcher for AI Controller Integration.

Provides routing and execution of tools from the new registry,
integrating with the transaction system for atomic operations.

WS6.2: Controller Integration
- Tool dispatch to new implementations
- Transaction wrapping for write operations
- Error handling with proper codes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..tools.base import BaseTool, ReadOnlyTool, WriteTool, ToolContext, ToolResult
from ..tools.errors import (
    ErrorCode,
    ToolError,
    VersionMismatchToolError,
    InvalidVersionTokenError,
    TabNotFoundError,
)
from ..tools.version import VersionManager, get_version_manager
from ..tools.tool_registry import (
    ToolRegistry,
    ToolRegistration,
    ToolCategory,
    get_tool_registry,
)
from .transaction import Transaction, TransactionState, TransactionManager
from .editor_lock import EditorLockManager, LockReason

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Dispatch Result
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class DispatchResult:
    """Result of a tool dispatch operation.

    Attributes:
        success: Whether the tool executed successfully.
        result: The tool's return value.
        error: Error if execution failed.
        tool_name: Name of the tool executed.
        execution_time_ms: Execution time in milliseconds.
        transaction_id: Transaction ID if write operation.
    """

    success: bool
    result: Any
    error: ToolError | None = None
    tool_name: str = ""
    execution_time_ms: float = 0.0
    transaction_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        data: dict[str, Any] = {
            "success": self.success,
            "tool_name": self.tool_name,
            "execution_time_ms": self.execution_time_ms,
        }
        if self.success:
            data["result"] = self.result
        else:
            data["error"] = self.error.to_dict() if self.error else {"message": "Unknown error"}
        if self.transaction_id:
            data["transaction_id"] = self.transaction_id
        if self.metadata:
            data["metadata"] = self.metadata
        return data


# -----------------------------------------------------------------------------
# Tool Context Provider
# -----------------------------------------------------------------------------


class ToolContextProvider(Protocol):
    """Protocol for providing tool execution context."""

    def get_active_tab_id(self) -> str | None:
        """Get the currently active tab ID."""
        ...

    def get_document_content(self, tab_id: str) -> str | None:
        """Get document content for a tab."""
        ...

    def set_document_content(self, tab_id: str, content: str) -> None:
        """Set document content for a tab."""
        ...

    def get_version_token(self, tab_id: str) -> str | None:
        """Get current version token for a tab."""
        ...


# -----------------------------------------------------------------------------
# Dispatch Listener
# -----------------------------------------------------------------------------


class DispatchListener(Protocol):
    """Callback protocol for dispatch events."""

    def on_tool_start(self, tool_name: str, arguments: Mapping[str, Any]) -> None:
        """Called when a tool starts execution."""
        ...

    def on_tool_complete(self, result: DispatchResult) -> None:
        """Called when a tool completes."""
        ...

    def on_tool_error(self, tool_name: str, error: ToolError) -> None:
        """Called when a tool fails."""
        ...


# -----------------------------------------------------------------------------
# Tool Dispatcher
# -----------------------------------------------------------------------------


class ToolDispatcher:
    """Dispatches tool calls to appropriate implementations.

    The dispatcher routes tool calls to registered implementations,
    wrapping write operations in transactions and handling errors
    with proper error codes.

    Example:
        dispatcher = ToolDispatcher(
            registry=get_tool_registry(),
            context_provider=bridge,
        )
        result = await dispatcher.dispatch("read_document", {"tab_id": "tab1"})
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        context_provider: ToolContextProvider | None = None,
        version_manager: VersionManager | None = None,
        transaction_manager: TransactionManager | None = None,
        lock_manager: EditorLockManager | None = None,
        listener: DispatchListener | None = None,
    ) -> None:
        """Initialize the dispatcher.

        Args:
            registry: Tool registry to use.
            context_provider: Provider for document context.
            version_manager: Version manager for tokens.
            transaction_manager: Transaction manager for write ops.
            lock_manager: Editor lock manager.
            listener: Dispatch event listener.
        """
        self._registry = registry or get_tool_registry()
        self._context_provider = context_provider
        self._version_manager = version_manager or get_version_manager()
        self._transaction_manager = transaction_manager
        self._lock_manager = lock_manager
        self._listener = listener

    def set_listener(self, listener: DispatchListener | None) -> None:
        """Set or replace the dispatch event listener.
        
        Args:
            listener: Dispatch event listener to receive tool start/complete/error events.
        """
        self._listener = listener

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        tool_name: str,
        arguments: Mapping[str, Any],
        *,
        context: ToolContext | None = None,
    ) -> DispatchResult:
        """Dispatch a tool call.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            context: Optional tool context.

        Returns:
            DispatchResult with execution outcome.
        """
        start_time = datetime.now(timezone.utc)

        # Notify listener
        if self._listener:
            try:
                self._listener.on_tool_start(tool_name, arguments)
            except Exception:
                LOGGER.debug("Listener on_tool_start failed", exc_info=True)

        # Get registration
        registration = self._registry.get_registration(tool_name)
        if not registration or not registration.enabled:
            error = ToolError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message=f"Tool '{tool_name}' is not registered or disabled.",
            )
            result = self._create_error_result(tool_name, error, start_time)
            self._notify_error(tool_name, error)
            return result

        # Determine if write operation
        is_write = registration.schema.writes_document
        requires_version = registration.schema.requires_version

        # Validate version token if required
        if requires_version:
            version_token = arguments.get("version_token")
            if not version_token:
                error = InvalidVersionTokenError(
                    message="version_token is required for this operation.",
                    token=None,
                )
                result = self._create_error_result(tool_name, error, start_time)
                self._notify_error(tool_name, error)
                return result

            # Validate token format and freshness
            validation_error = self._validate_version_token(version_token)
            if validation_error:
                result = self._create_error_result(tool_name, validation_error, start_time)
                self._notify_error(tool_name, validation_error)
                return result

        # Execute with or without transaction
        if is_write and self._transaction_manager:
            result = await self._dispatch_with_transaction(
                registration, arguments, context, start_time
            )
        else:
            result = await self._dispatch_direct(
                registration, arguments, context, start_time
            )

        # Notify listener
        if self._listener:
            try:
                self._listener.on_tool_complete(result)
            except Exception:
                LOGGER.debug("Listener on_tool_complete failed", exc_info=True)

        return result

    async def _dispatch_direct(
        self,
        registration: ToolRegistration,
        arguments: Mapping[str, Any],
        context: ToolContext | None,
        start_time: datetime,
    ) -> DispatchResult:
        """Execute tool without transaction wrapping."""
        tool = registration.impl
        tool_name = registration.name

        try:
            # Build context if not provided
            if context is None:
                context = self._build_context(arguments)
            
            # If no context could be built (no provider), execute without context
            if context is None:
                # For callables, execute without context
                if callable(tool):
                    result = await self._execute_callable(tool, arguments)
                    return self._create_success_result(tool_name, result, start_time)
                else:
                    raise ValueError("Cannot execute BaseTool without document provider")

            # Execute tool
            if isinstance(tool, BaseTool):
                result = await self._execute_base_tool(tool, arguments, context)
            elif callable(tool):
                result = await self._execute_callable(tool, arguments)
            else:
                raise TypeError(f"Tool {tool_name} is not callable")

            return self._create_success_result(tool_name, result, start_time)

        except ToolError as e:
            return self._create_error_result(tool_name, e, start_time)
        except Exception as e:
            LOGGER.exception("Tool %s failed unexpectedly", tool_name)
            error = ToolError(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=str(e),
            )
            return self._create_error_result(tool_name, error, start_time)

    async def _dispatch_with_transaction(
        self,
        registration: ToolRegistration,
        arguments: Mapping[str, Any],
        context: ToolContext | None,
        start_time: datetime,
    ) -> DispatchResult:
        """Execute write tool with transaction wrapping."""
        tool = registration.impl
        tool_name = registration.name

        # Extract tab_id for transaction
        tab_id = self._extract_tab_id(arguments)
        if not tab_id and self._context_provider:
            tab_id = self._get_active_tab_from_provider()

        if not tab_id:
            error = TabNotFoundError(
                message="Could not determine target tab for write operation.",
            )
            return self._create_error_result(tool_name, error, start_time)

        # Create transaction (this also begins it with the tab_ids)
        transaction = self._transaction_manager.create_transaction(tab_ids=[tab_id])
        transaction_id = transaction.transaction_id

        try:
            # Build context
            if context is None:
                context = self._build_context(arguments)

            # Execute tool
            if isinstance(tool, WriteTool):
                result = await self._execute_write_tool(tool, arguments, context, transaction)
            elif isinstance(tool, BaseTool):
                result = await self._execute_base_tool(tool, arguments, context)
            elif callable(tool):
                result = await self._execute_callable(tool, arguments)
            else:
                raise TypeError(f"Tool {tool_name} is not callable")

            # Commit transaction
            transaction.commit()

            dispatch_result = self._create_success_result(tool_name, result, start_time)
            dispatch_result.transaction_id = transaction_id
            return dispatch_result

        except ToolError as e:
            # Rollback on tool error
            if transaction.state == TransactionState.ACTIVE:
                transaction.rollback(reason=str(e))
            return self._create_error_result(tool_name, e, start_time, transaction_id)

        except Exception as e:
            # Rollback on unexpected error
            if transaction.state == TransactionState.ACTIVE:
                transaction.rollback(reason=str(e))
            LOGGER.exception("Tool %s failed in transaction", tool_name)
            error = ToolError(
                error_code=ErrorCode.INTERNAL_ERROR,
                message=str(e),
            )
            return self._create_error_result(tool_name, error, start_time, transaction_id)

    # ------------------------------------------------------------------
    # Tool Execution
    # ------------------------------------------------------------------

    async def _execute_base_tool(
        self,
        tool: BaseTool,
        arguments: Mapping[str, Any],
        context: ToolContext,
    ) -> Any:
        """Execute a BaseTool implementation."""
        # Validate arguments
        validation_error = tool.validate(arguments)
        if validation_error:
            raise ToolError(
                error_code=ErrorCode.INVALID_PARAMETER,
                message=validation_error,
            )

        # Execute - context first, then params (matches BaseTool.execute signature)
        result = tool.execute(context, dict(arguments))

        # Handle async result
        if hasattr(result, "__await__"):
            result = await result

        # Convert ToolResult to dict
        if isinstance(result, ToolResult):
            return result.to_dict()

        return result

    async def _execute_write_tool(
        self,
        tool: WriteTool,
        arguments: Mapping[str, Any],
        context: ToolContext,
        transaction: Transaction,
    ) -> Any:
        """Execute a WriteTool with transaction integration."""
        # The WriteTool should stage changes in the transaction
        # For now, execute normally and let the tool handle staging
        return await self._execute_base_tool(tool, arguments, context)

    async def _execute_callable(
        self,
        tool: Callable[..., Any],
        arguments: Mapping[str, Any],
    ) -> Any:
        """Execute a callable tool."""
        result = tool(**arguments)

        # Handle async result
        if hasattr(result, "__await__"):
            result = await result

        return result

    # ------------------------------------------------------------------
    # Context Building
    # ------------------------------------------------------------------

    def _build_context(self, arguments: Mapping[str, Any]) -> ToolContext | None:
        """Build tool context from arguments and provider.
        
        Returns None if no document provider is available.
        """
        tab_id = self._extract_tab_id(arguments)
        if not tab_id and self._context_provider:
            tab_id = self._get_active_tab_from_provider()

        # Need a document provider for ToolContext
        doc_provider = self._create_document_provider()
        if not doc_provider:
            return None

        return ToolContext(
            document_provider=doc_provider,
            version_manager=self._version_manager,
            tab_id=tab_id,
        )

    def _create_document_provider(self) -> Any:
        """Create or retrieve document provider for context.
        
        Returns the context_provider if it implements DocumentProvider,
        or creates an adapter around it.
        """
        if self._context_provider is None:
            return None
        
        # If context_provider already implements DocumentProvider interface, use it
        if hasattr(self._context_provider, 'get_document_text'):
            return self._context_provider
        
        # Create an adapter that wraps ToolContextProvider
        return _DocumentProviderAdapter(self._context_provider)

    def _get_active_tab_from_provider(self) -> str | None:
        """Get active tab ID from context provider.
        
        Handles different provider types:
        - ToolContextProvider with get_active_tab_id() method
        - DocumentBridge with _tab_id attribute
        """
        if self._context_provider is None:
            return None
        
        # Standard ToolContextProvider interface
        if hasattr(self._context_provider, 'get_active_tab_id'):
            return self._context_provider.get_active_tab_id()
        
        # DocumentBridge stores tab_id as _tab_id
        if hasattr(self._context_provider, '_tab_id'):
            return getattr(self._context_provider, '_tab_id', None)
        
        return None

    def _extract_tab_id(self, arguments: Mapping[str, Any]) -> str | None:
        """Extract tab_id from arguments or version_token."""
        # Direct tab_id
        tab_id = arguments.get("tab_id")
        if tab_id:
            return str(tab_id)

        # From version_token
        version_token = arguments.get("version_token")
        if version_token and ":" in version_token:
            return version_token.split(":")[0]

        return None

    # ------------------------------------------------------------------
    # Version Token Validation
    # ------------------------------------------------------------------

    def _validate_version_token(self, token: str) -> ToolError | None:
        """Validate a version token.

        Returns:
            ToolError if invalid, None if valid.
        """
        if not token or ":" not in token:
            return InvalidVersionTokenError(
                message="Invalid version_token format. Expected 'tab_id:version_id:hash'.",
                token=token,
            )

        parts = token.split(":")
        if len(parts) < 2:
            return InvalidVersionTokenError(
                message="Invalid version_token format. Expected 'tab_id:version_id:hash'.",
                token=token,
            )

        tab_id = parts[0]

        # Check if token is current
        try:
            parsed = self._version_manager.parse_token(token)
            if not self._version_manager.is_current(parsed):
                current = self._version_manager.get_current_token(tab_id)
                return VersionMismatchToolError(
                    message="Version token is stale. Document has been modified.",
                    current_version={"token": current.to_string()} if current else None,
                    your_version={"token": token},
                )
        except Exception:
            # Token parsing failed - might be legacy format
            LOGGER.debug("Version token validation failed for: %s", token)

        return None

    # ------------------------------------------------------------------
    # Result Building
    # ------------------------------------------------------------------

    def _create_success_result(
        self,
        tool_name: str,
        result: Any,
        start_time: datetime,
    ) -> DispatchResult:
        """Create a successful dispatch result."""
        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return DispatchResult(
            success=True,
            result=result,
            tool_name=tool_name,
            execution_time_ms=elapsed_ms,
        )

    def _create_error_result(
        self,
        tool_name: str,
        error: ToolError,
        start_time: datetime,
        transaction_id: str | None = None,
    ) -> DispatchResult:
        """Create an error dispatch result."""
        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        return DispatchResult(
            success=False,
            result=None,
            error=error,
            tool_name=tool_name,
            execution_time_ms=elapsed_ms,
            transaction_id=transaction_id,
        )

    def _notify_error(self, tool_name: str, error: ToolError) -> None:
        """Notify listener of error."""
        if self._listener:
            try:
                self._listener.on_tool_error(tool_name, error)
            except Exception:
                LOGGER.debug("Listener on_tool_error failed", exc_info=True)

    def _notify_success(self, result: DispatchResult) -> None:
        """Notify listener of success."""
        if self._listener:
            try:
                self._listener.on_tool_complete(result)
            except Exception:
                LOGGER.debug("Listener on_tool_complete failed", exc_info=True)

    # ------------------------------------------------------------------
    # Batch Dispatch
    # ------------------------------------------------------------------

    async def dispatch_batch(
        self,
        calls: Sequence[tuple[str, Mapping[str, Any]]],
        *,
        context: ToolContext | None = None,
        stop_on_error: bool = False,
    ) -> list[DispatchResult]:
        """Dispatch multiple tool calls.

        Args:
            calls: List of (tool_name, arguments) tuples.
            context: Shared tool context.
            stop_on_error: Stop batch on first error.

        Returns:
            List of dispatch results.
        """
        results: list[DispatchResult] = []

        for tool_name, arguments in calls:
            result = await self.dispatch(tool_name, arguments, context=context)
            results.append(result)

            if stop_on_error and not result.success:
                break

        return results


class _DocumentProviderAdapter:
    """Adapter that wraps ToolContextProvider for DocumentProvider compatibility.
    
    Provides both `get_document_text` and `get_document_content` as aliases
    to support different tool interface expectations.
    """
    
    def __init__(self, provider: ToolContextProvider) -> None:
        self._provider = provider
    
    def get_active_tab_id(self) -> str | None:
        return self._provider.get_active_tab_id()
    
    def get_document_text(self, tab_id: str | None = None) -> str | None:
        """Get document text for a tab."""
        if tab_id is None:
            tab_id = self.get_active_tab_id()
        if tab_id is None:
            return None
        return self._provider.get_document_content(tab_id)
    
    def get_document_content(self, tab_id: str) -> str | None:
        """Alias for get_document_text for tools that expect this method name."""
        return self._provider.get_document_content(tab_id)
    
    def get_document_metadata(self, tab_id: str) -> dict[str, Any] | None:
        """Get metadata for a tab (path, language, etc.)."""
        # Return minimal metadata - real implementation would query the provider
        return {"tab_id": tab_id}
    
    def set_document_text(self, tab_id: str, content: str) -> None:
        """Set document text for a tab."""
        self._provider.set_document_content(tab_id, content)


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------


def create_tool_dispatcher(
    *,
    context_provider: ToolContextProvider | None = None,
    transaction_manager: TransactionManager | None = None,
    lock_manager: EditorLockManager | None = None,
) -> ToolDispatcher:
    """Create a configured tool dispatcher.

    Args:
        context_provider: Document context provider.
        transaction_manager: Transaction manager for writes.
        lock_manager: Editor lock manager.

    Returns:
        Configured ToolDispatcher instance.
    """
    return ToolDispatcher(
        registry=get_tool_registry(),
        context_provider=context_provider,
        version_manager=get_version_manager(),
        transaction_manager=transaction_manager,
        lock_manager=lock_manager,
    )


__all__ = [
    "DispatchResult",
    "ToolContextProvider",
    "DispatchListener",
    "ToolDispatcher",
    "create_tool_dispatcher",
]
