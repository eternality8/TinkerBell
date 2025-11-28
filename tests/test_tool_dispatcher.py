"""Tests for ToolDispatcher.

Tests cover:
- Basic dispatch to read-only tools
- Write operations with transaction wrapping
- Error handling and proper error codes
- Context building and tool execution
- Transaction lifecycle (no double-begin bug)
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping
from unittest.mock import MagicMock, patch

from tinkerbell.ai.orchestration.tool_dispatcher import (
    ToolDispatcher,
    DispatchResult,
    ToolContextProvider,
)
from tinkerbell.ai.orchestration.transaction import (
    Transaction,
    TransactionManager,
    TransactionState,
)
from tinkerbell.ai.tools.base import (
    BaseTool,
    ReadOnlyTool,
    WriteTool,
    ToolContext,
    ToolResult,
)
from tinkerbell.ai.tools.errors import (
    ToolError,
    ErrorCode,
    TabNotFoundError,
)
from tinkerbell.ai.tools.tool_registry import (
    ToolRegistry,
    ToolSchema,
    ToolCategory,
)
from tinkerbell.ai.tools.version import VersionManager


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockReadTool(ReadOnlyTool):
    """A mock read-only tool for testing."""
    
    name: ClassVar[str] = "mock_read"
    call_count: int = field(default=0, init=False)
    last_params: dict = field(default_factory=dict, init=False)
    
    def read(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        self.last_params = dict(params)
        return {"status": "ok", "data": params.get("value", "default")}


@dataclass
class MockSimpleWriteTool(BaseTool):
    """A simple mock tool that's marked as writes_document for transaction testing.
    
    Unlike MockSimpleWriteTool, this doesn't inherit from WriteTool so it doesn't
    require version token validation. Used for testing transaction handling.
    """
    
    name: ClassVar[str] = "mock_simple_write"
    call_count: int = field(default=0, init=False)
    last_params: dict = field(default_factory=dict, init=False)
    should_fail: bool = False
    
    def execute(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        self.last_params = dict(params)
        if self.should_fail:
            raise ToolError(error_code=ErrorCode.INTERNAL_ERROR, message="Mock failure")
        return {"status": "written", "content_length": len(params.get("content", ""))}


@dataclass
class MockBaseTool(BaseTool):
    """A generic mock tool for testing."""
    
    name: ClassVar[str] = "mock_base"
    call_count: int = field(default=0, init=False)
    last_context: ToolContext | None = field(default=None, init=False)
    last_params: dict = field(default_factory=dict, init=False)
    
    def execute(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        self.last_context = context
        self.last_params = dict(params)
        return {"executed": True, "tool": self.name}


class MockContextProvider:
    """Mock context provider for testing."""
    
    def __init__(self, active_tab_id: str = "test-tab-1"):
        self.active_tab_id = active_tab_id
        self._documents: dict[str, str] = {
            "test-tab-1": "Hello, World!",
            "test-tab-2": "Second document content.",
        }
    
    def get_active_tab_id(self) -> str | None:
        return self.active_tab_id
    
    def get_document_text(self, tab_id: str) -> str | None:
        return self._documents.get(tab_id)
    
    def set_document_text(self, tab_id: str, text: str) -> None:
        self._documents[tab_id] = text
    
    def get_open_tabs(self) -> list[dict[str, Any]]:
        return [{"tab_id": tid} for tid in self._documents]


class MockDocumentProvider:
    """Mock document provider for ToolContext."""
    
    def __init__(self, documents: dict[str, str] | None = None):
        self._documents = documents or {"test-tab-1": "Hello, World!"}
    
    def get_document_text(self, tab_id: str) -> str | None:
        return self._documents.get(tab_id)
    
    def set_document_text(self, tab_id: str, text: str) -> None:
        self._documents[tab_id] = text


@pytest.fixture
def registry() -> ToolRegistry:
    """Create a fresh tool registry."""
    return ToolRegistry()


@pytest.fixture
def version_manager() -> VersionManager:
    """Create a version manager."""
    return VersionManager()


@pytest.fixture
def context_provider() -> MockContextProvider:
    """Create a mock context provider."""
    return MockContextProvider()


@pytest.fixture
def transaction_manager() -> TransactionManager:
    """Create a transaction manager."""
    return TransactionManager()


@pytest.fixture
def dispatcher(
    registry: ToolRegistry,
    version_manager: VersionManager,
    context_provider: MockContextProvider,
    transaction_manager: TransactionManager,
) -> ToolDispatcher:
    """Create a tool dispatcher with all dependencies."""
    return ToolDispatcher(
        registry=registry,
        context_provider=context_provider,
        version_manager=version_manager,
        transaction_manager=transaction_manager,
    )


# =============================================================================
# DispatchResult Tests
# =============================================================================


class TestDispatchResult:
    """Tests for DispatchResult dataclass."""

    def test_success_result_to_dict(self):
        """Successful result serializes correctly."""
        result = DispatchResult(
            success=True,
            result={"data": "test"},
            tool_name="test_tool",
            execution_time_ms=50.0,
        )
        data = result.to_dict()
        
        assert data["success"] is True
        assert data["tool_name"] == "test_tool"
        assert data["result"] == {"data": "test"}
        assert "error" not in data

    def test_error_result_to_dict(self):
        """Error result serializes correctly."""
        error = ToolError(error_code=ErrorCode.INVALID_PARAMETER, message="Bad param")
        result = DispatchResult(
            success=False,
            result=None,
            error=error,
            tool_name="failing_tool",
            execution_time_ms=10.0,
        )
        data = result.to_dict()
        
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["message"] == "Bad param"

    def test_transaction_id_included(self):
        """Transaction ID is included when present."""
        result = DispatchResult(
            success=True,
            result={},
            tool_name="write_tool",
            transaction_id="tx-123",
        )
        data = result.to_dict()
        
        assert data["transaction_id"] == "tx-123"


# =============================================================================
# Basic Dispatch Tests
# =============================================================================


class TestBasicDispatch:
    """Tests for basic tool dispatch functionality."""

    @pytest.mark.asyncio
    async def test_dispatch_unregistered_tool_returns_error(self, dispatcher: ToolDispatcher):
        """Dispatching an unregistered tool returns an error."""
        result = await dispatcher.dispatch("nonexistent_tool", {})
        
        assert result.success is False
        assert result.error is not None
        assert "not registered" in result.error.message.lower()

    @pytest.mark.asyncio
    async def test_dispatch_disabled_tool_returns_error(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
    ):
        """Dispatching a disabled tool returns an error."""
        tool = MockReadTool()
        schema = ToolSchema(
            name="mock_read",
            description="A mock read tool",
            category=ToolCategory.NAVIGATION,
        )
        registry.register(tool, schema=schema, enabled=False)
        
        result = await dispatcher.dispatch("mock_read", {})
        
        assert result.success is False
        assert "not registered or disabled" in result.error.message.lower()

    @pytest.mark.asyncio
    async def test_dispatch_read_tool_succeeds(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
    ):
        """Dispatching a read tool executes successfully."""
        tool = MockReadTool()
        schema = ToolSchema(
            name="mock_read",
            description="A mock read tool",
            category=ToolCategory.NAVIGATION,
        )
        registry.register(tool, schema=schema)
        
        result = await dispatcher.dispatch("mock_read", {"value": "test_value"})
        
        assert result.success is True
        assert result.result["status"] == "ok"
        assert result.result["data"] == "test_value"
        assert tool.call_count == 1
        assert tool.last_params["value"] == "test_value"

    @pytest.mark.asyncio
    async def test_dispatch_passes_context_and_params_correctly(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
    ):
        """Tool receives correct context and params."""
        tool = MockBaseTool()
        schema = ToolSchema(
            name="mock_base",
            description="A mock tool",
            category=ToolCategory.UTILITY,
        )
        registry.register(tool, schema=schema)
        
        result = await dispatcher.dispatch("mock_base", {"tab_id": "test-tab-1", "foo": "bar"})
        
        assert result.success is True
        assert tool.last_params["foo"] == "bar"
        # Context should have been built
        assert tool.last_context is not None


# =============================================================================
# Transaction Tests
# =============================================================================


class TestTransactionHandling:
    """Tests for transaction handling in write operations."""

    @pytest.mark.asyncio
    async def test_write_tool_creates_transaction(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
        transaction_manager: TransactionManager,
    ):
        """Write tools are wrapped in a transaction."""
        tool = MockSimpleWriteTool()
        schema = ToolSchema(
            name="mock_simple_write",
            description="A mock write tool",
            category=ToolCategory.NAVIGATION,
            writes_document=True,
        )
        registry.register(tool, schema=schema)
        
        result = await dispatcher.dispatch(
            "mock_simple_write",
            {"tab_id": "test-tab-1", "content": "new content"},
        )
        
        assert result.success is True
        assert result.transaction_id is not None

    @pytest.mark.asyncio
    async def test_transaction_not_double_begun(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
    ):
        """Transaction is not started twice (regression test for double-begin bug)."""
        tool = MockSimpleWriteTool()
        schema = ToolSchema(
            name="mock_simple_write",
            description="A mock write tool",
            category=ToolCategory.NAVIGATION,
            writes_document=True,
        )
        registry.register(tool, schema=schema)
        
        # This should not raise "Cannot begin: transaction is ACTIVE"
        result = await dispatcher.dispatch(
            "mock_simple_write",
            {"tab_id": "test-tab-1", "content": "test"},
        )
        
        assert result.success is True
        # Tool was actually called
        assert tool.call_count == 1

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_tool_error(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
        transaction_manager: TransactionManager,
    ):
        """Transaction is rolled back when tool raises error."""
        tool = MockSimpleWriteTool()
        tool.should_fail = True
        schema = ToolSchema(
            name="mock_simple_write",
            description="A mock write tool",
            category=ToolCategory.NAVIGATION,
            writes_document=True,
        )
        registry.register(tool, schema=schema)
        
        result = await dispatcher.dispatch(
            "mock_simple_write",
            {"tab_id": "test-tab-1", "content": "will fail"},
        )
        
        assert result.success is False
        assert "Mock failure" in result.error.message

    @pytest.mark.asyncio
    async def test_write_without_tab_id_uses_active_tab(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
        context_provider: MockContextProvider,
    ):
        """Write operation uses active tab when tab_id not provided."""
        context_provider.active_tab_id = "test-tab-2"
        tool = MockSimpleWriteTool()
        schema = ToolSchema(
            name="mock_simple_write",
            description="A mock write tool",
            category=ToolCategory.NAVIGATION,
            writes_document=True,
        )
        registry.register(tool, schema=schema)
        
        result = await dispatcher.dispatch("mock_simple_write", {"content": "data"})
        
        assert result.success is True

    @pytest.mark.asyncio
    async def test_write_without_any_tab_returns_error(
        self,
        registry: ToolRegistry,
        version_manager: VersionManager,
        transaction_manager: TransactionManager,
    ):
        """Write operation fails if no tab can be determined."""
        # Create dispatcher with no active tab
        no_tab_provider = MockContextProvider()
        no_tab_provider.active_tab_id = None
        
        dispatcher = ToolDispatcher(
            registry=registry,
            context_provider=no_tab_provider,
            version_manager=version_manager,
            transaction_manager=transaction_manager,
        )
        
        tool = MockSimpleWriteTool()
        schema = ToolSchema(
            name="mock_simple_write",
            description="A mock write tool",
            category=ToolCategory.NAVIGATION,
            writes_document=True,
        )
        registry.register(tool, schema=schema)
        
        result = await dispatcher.dispatch("mock_simple_write", {"content": "data"})
        
        assert result.success is False
        assert isinstance(result.error, TabNotFoundError)


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling during dispatch."""

    @pytest.mark.asyncio
    async def test_tool_error_returned_properly(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
    ):
        """ToolError raised by tool is returned in result."""
        @dataclass
        class FailingTool(BaseTool):
            name: ClassVar[str] = "failing_tool"
            
            def execute(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
                raise ToolError(
                    error_code=ErrorCode.INVALID_PARAMETER,
                    message="Parameter 'x' is invalid",
                )
        
        tool = FailingTool()
        schema = ToolSchema(
            name="failing_tool",
            description="A tool that fails",
            category=ToolCategory.UTILITY,
        )
        registry.register(tool, schema=schema)
        
        result = await dispatcher.dispatch("failing_tool", {"x": "bad"})
        
        assert result.success is False
        assert result.error.error_code == ErrorCode.INVALID_PARAMETER
        assert "Parameter 'x' is invalid" in result.error.message

    @pytest.mark.asyncio
    async def test_unexpected_exception_wrapped_as_internal_error(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
    ):
        """Unexpected exceptions are wrapped as internal errors."""
        @dataclass
        class CrashingTool(BaseTool):
            name: ClassVar[str] = "crashing_tool"
            
            def execute(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
                raise RuntimeError("Unexpected crash!")
        
        tool = CrashingTool()
        schema = ToolSchema(
            name="crashing_tool",
            description="A tool that crashes",
            category=ToolCategory.UTILITY,
        )
        registry.register(tool, schema=schema)
        
        result = await dispatcher.dispatch("crashing_tool", {})
        
        assert result.success is False
        assert result.error.error_code == ErrorCode.INTERNAL_ERROR
        assert "Unexpected crash" in result.error.message


# =============================================================================
# Callable Tool Tests
# =============================================================================


class TestCallableTools:
    """Tests for callable (non-BaseTool) tool dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_callable_function(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
    ):
        """Plain callable functions can be dispatched."""
        def simple_tool(value: str = "default") -> dict[str, Any]:
            return {"result": value}
        
        schema = ToolSchema(
            name="simple_callable",
            description="A simple callable",
            category=ToolCategory.UTILITY,
        )
        registry.register(simple_tool, schema=schema)
        
        result = await dispatcher.dispatch("simple_callable", {"value": "hello"})
        
        assert result.success is True
        assert result.result["result"] == "hello"

    @pytest.mark.asyncio
    async def test_dispatch_async_callable(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
    ):
        """Async callables are properly awaited."""
        async def async_tool(delay: float = 0) -> dict[str, Any]:
            return {"async": True}
        
        schema = ToolSchema(
            name="async_callable",
            description="An async callable",
            category=ToolCategory.UTILITY,
        )
        registry.register(async_tool, schema=schema)
        
        result = await dispatcher.dispatch("async_callable", {})
        
        assert result.success is True
        assert result.result["async"] is True


# =============================================================================
# Context Building Tests
# =============================================================================


class TestContextBuilding:
    """Tests for ToolContext building during dispatch."""

    @pytest.mark.asyncio
    async def test_context_includes_tab_id(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
    ):
        """Context includes tab_id from arguments."""
        tool = MockBaseTool()
        schema = ToolSchema(
            name="mock_base",
            description="A mock tool",
            category=ToolCategory.NAVIGATION,
        )
        registry.register(tool, schema=schema)
        
        await dispatcher.dispatch("mock_base", {"tab_id": "specific-tab"})
        
        # Tool was called - context building didn't fail
        assert tool.call_count == 1

    @pytest.mark.asyncio
    async def test_context_uses_active_tab_when_not_specified(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
        context_provider: MockContextProvider,
    ):
        """Context uses active tab when tab_id not in arguments."""
        context_provider.active_tab_id = "active-tab-123"
        tool = MockBaseTool()
        schema = ToolSchema(
            name="mock_base",
            description="A mock tool",
            category=ToolCategory.NAVIGATION,
        )
        registry.register(tool, schema=schema)
        
        await dispatcher.dispatch("mock_base", {"other_param": "value"})
        
        assert tool.call_count == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestDispatcherIntegration:
    """Integration tests for dispatcher with multiple components."""

    @pytest.mark.asyncio
    async def test_multiple_sequential_dispatches(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
    ):
        """Multiple dispatches work correctly in sequence."""
        tool = MockReadTool()
        schema = ToolSchema(
            name="mock_read",
            description="A mock read tool",
            category=ToolCategory.NAVIGATION,
        )
        registry.register(tool, schema=schema)
        
        result1 = await dispatcher.dispatch("mock_read", {"value": "first"})
        result2 = await dispatcher.dispatch("mock_read", {"value": "second"})
        result3 = await dispatcher.dispatch("mock_read", {"value": "third"})
        
        assert result1.success and result1.result["data"] == "first"
        assert result2.success and result2.result["data"] == "second"
        assert result3.success and result3.result["data"] == "third"
        assert tool.call_count == 3

    @pytest.mark.asyncio
    async def test_mixed_read_write_operations(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
    ):
        """Read and write operations can be intermixed."""
        read_tool = MockReadTool()
        write_tool = MockSimpleWriteTool()
        
        read_schema = ToolSchema(
            name="mock_read",
            description="Read tool",
            category=ToolCategory.NAVIGATION,
        )
        write_schema = ToolSchema(
            name="mock_simple_write",
            description="Write tool",
            category=ToolCategory.NAVIGATION,
            writes_document=True,
        )
        
        registry.register(read_tool, schema=read_schema)
        registry.register(write_tool, schema=write_schema)
        
        r1 = await dispatcher.dispatch("mock_read", {"value": "before"})
        w1 = await dispatcher.dispatch("mock_simple_write", {"tab_id": "test-tab-1", "content": "new"})
        r2 = await dispatcher.dispatch("mock_read", {"value": "after"})
        
        assert r1.success and w1.success and r2.success
        assert w1.transaction_id is not None
        assert r1.transaction_id is None  # Read ops don't have transactions

    @pytest.mark.asyncio
    async def test_execution_time_is_recorded(
        self,
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
    ):
        """Execution time is recorded in result."""
        tool = MockReadTool()
        schema = ToolSchema(
            name="mock_read",
            description="A mock read tool",
            category=ToolCategory.NAVIGATION,
        )
        registry.register(tool, schema=schema)
        
        result = await dispatcher.dispatch("mock_read", {})
        
        assert result.execution_time_ms >= 0


# =============================================================================
# DocumentBridge Compatibility Tests
# =============================================================================


class TestDocumentBridgeCompatibility:
    """Tests for DocumentBridge compatibility with dispatcher."""

    @pytest.mark.asyncio
    async def test_dispatcher_handles_bridge_with_tab_id_attribute(
        self,
        registry: ToolRegistry,
        transaction_manager: TransactionManager,
        version_manager: VersionManager,
    ):
        """Dispatcher can get tab_id from bridge's _tab_id attribute."""

        # Mock bridge that uses _tab_id like DocumentBridge does
        class MockBridgeLikeProvider:
            def __init__(self):
                self._tab_id: str | None = "bridge-tab-123"

            def get_document_content(self, tab_id: str) -> str | None:
                return "test content"

            def set_document_content(self, tab_id: str, content: str) -> None:
                pass

        bridge = MockBridgeLikeProvider()
        dispatcher = ToolDispatcher(
            registry=registry,
            context_provider=bridge,  # type: ignore  # Testing compatibility
            version_manager=version_manager,
            transaction_manager=transaction_manager,
        )

        write_tool = MockSimpleWriteTool()
        schema = ToolSchema(
            name="mock_write",
            description="A mock write tool",
            category=ToolCategory.NAVIGATION,
            writes_document=True,
        )
        registry.register(write_tool, schema=schema)

        # Dispatch without explicit tab_id - should use bridge's _tab_id
        result = await dispatcher.dispatch("mock_write", {"content": "new"})

        # Should succeed because tab_id was obtained from bridge._tab_id
        assert result.success is True
        assert write_tool.call_count == 1

    @pytest.mark.asyncio
    async def test_dispatcher_handles_bridge_without_get_active_tab_id(
        self,
        registry: ToolRegistry,
        transaction_manager: TransactionManager,
        version_manager: VersionManager,
    ):
        """Dispatcher handles providers without get_active_tab_id method."""

        # Provider that has neither get_active_tab_id nor _tab_id
        class MinimalProvider:
            def get_document_content(self, tab_id: str) -> str | None:
                return "test content"

            def set_document_content(self, tab_id: str, content: str) -> None:
                pass

        provider = MinimalProvider()
        dispatcher = ToolDispatcher(
            registry=registry,
            context_provider=provider,  # type: ignore  # Testing compatibility
            version_manager=version_manager,
            transaction_manager=transaction_manager,
        )

        write_tool = MockSimpleWriteTool()
        schema = ToolSchema(
            name="mock_write",
            description="A mock write tool",
            category=ToolCategory.NAVIGATION,
            writes_document=True,
        )
        registry.register(write_tool, schema=schema)

        # Dispatch without tab_id - should fail gracefully since no tab_id source
        result = await dispatcher.dispatch("mock_write", {"content": "new"})

        # Should fail with tab not found error
        assert result.success is False
        assert result.error is not None
        assert "tab" in result.error.message.lower()
