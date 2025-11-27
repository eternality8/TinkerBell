"""WS7 Integration Tests - End-to-End Validation of New Tool System.

This module tests the complete tool workflow including:
- Multi-tool workflows
- Version token handling across operations
- Transaction commit/rollback
- Tool dispatcher integration
- Tool registry exports

WS7.2: New Test Suite
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, Mapping
from unittest.mock import MagicMock, AsyncMock

from tinkerbell.ai.tools.base import (
    BaseTool,
    ReadOnlyTool,
    WriteTool,
    ToolContext,
    ToolResult,
    DocumentProvider,
)
from tinkerbell.ai.tools.errors import (
    ToolError,
    ErrorCode,
    VersionMismatchToolError,
    InvalidVersionTokenError,
    TabNotFoundError,
)
from tinkerbell.ai.tools.version import (
    VersionManager,
    VersionToken,
    get_version_manager,
    reset_version_manager,
)
from tinkerbell.ai.tools.tool_registry import (
    ToolRegistry,
    ToolSchema,
    ParameterSchema,
    ToolCategory,
    get_tool_registry,
    reset_tool_registry,
    ALL_TOOL_SCHEMAS,
    LIST_TABS_SCHEMA,
    READ_DOCUMENT_SCHEMA,
    INSERT_LINES_SCHEMA,
    REPLACE_LINES_SCHEMA,
    DELETE_LINES_SCHEMA,
)
from tinkerbell.ai.orchestration.tool_dispatcher import (
    ToolDispatcher,
    DispatchResult,
    ToolContextProvider,
    create_tool_dispatcher,
)
from tinkerbell.ai.orchestration.transaction import (
    Transaction,
    TransactionState,
    TransactionManager,
    get_transaction_manager,
    reset_transaction_manager,
)
from tinkerbell.ai.orchestration.editor_lock import (
    EditorLockManager,
    LockState,
    LockReason,
    get_lock_manager,
    reset_lock_manager,
)


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state before each test."""
    reset_version_manager()
    reset_tool_registry()
    reset_transaction_manager()
    reset_lock_manager()
    yield


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Create a fresh tool registry."""
    return ToolRegistry()


# -----------------------------------------------------------------------------
# Test: Tool Schema Exports
# -----------------------------------------------------------------------------


class TestToolSchemaExports:
    """Tests for tool schema exports from tool_registry."""

    def test_all_schemas_contains_core_tools(self) -> None:
        """ALL_TOOL_SCHEMAS should contain all core tool schemas."""
        schema_names = list(ALL_TOOL_SCHEMAS.keys())
        assert "list_tabs" in schema_names
        assert "read_document" in schema_names
        assert "insert_lines" in schema_names
        assert "replace_lines" in schema_names
        assert "delete_lines" in schema_names

    def test_list_tabs_schema_structure(self) -> None:
        """list_tabs schema should have proper structure."""
        assert LIST_TABS_SCHEMA.name == "list_tabs"
        assert LIST_TABS_SCHEMA.description
        assert LIST_TABS_SCHEMA.writes_document is False

    def test_read_document_schema_structure(self) -> None:
        """read_document schema should have proper structure."""
        assert READ_DOCUMENT_SCHEMA.name == "read_document"
        assert READ_DOCUMENT_SCHEMA.description
        param_names = [p.name for p in READ_DOCUMENT_SCHEMA.parameters]
        assert "tab_id" in param_names

    def test_insert_lines_schema_requires_version(self) -> None:
        """insert_lines should require version token."""
        assert INSERT_LINES_SCHEMA.name == "insert_lines"
        assert INSERT_LINES_SCHEMA.writes_document is True
        assert INSERT_LINES_SCHEMA.requires_version is True

    def test_schema_to_json_schema(self) -> None:
        """ToolSchema should convert to JSON Schema."""
        json_schema = LIST_TABS_SCHEMA.to_json_schema()
        assert json_schema["type"] == "object"
        assert "properties" in json_schema


# -----------------------------------------------------------------------------
# Test: Tool Registry
# -----------------------------------------------------------------------------


class TestToolRegistryOperations:
    """Tests for ToolRegistry class operations."""

    def test_register_callable(self, tool_registry: ToolRegistry) -> None:
        """Can register a callable as a tool."""
        tool_registry.register(
            tool=lambda: {"result": "ok"},
            schema=ToolSchema(name="my_tool", description="A test tool"),
        )

        assert tool_registry.has_tool("my_tool")

    def test_list_tools(self, tool_registry: ToolRegistry) -> None:
        """Can list registered tools."""
        tool_registry.register(
            tool=lambda: {},
            schema=ToolSchema(name="tool_a", description="Tool A"),
        )
        tool_registry.register(
            tool=lambda: {},
            schema=ToolSchema(name="tool_b", description="Tool B"),
        )

        tools = tool_registry.list_tools()
        assert "tool_a" in tools
        assert "tool_b" in tools

    def test_enable_disable_tool(self, tool_registry: ToolRegistry) -> None:
        """Can enable/disable tools."""
        tool_registry.register(
            tool=lambda: {},
            schema=ToolSchema(name="toggle_tool", description="Toggle me"),
        )

        # Default enabled
        assert tool_registry.has_tool("toggle_tool")

        # Disable
        tool_registry.disable_tool("toggle_tool")
        assert not tool_registry.has_tool("toggle_tool")  # has_tool returns False for disabled

        # Re-enable
        tool_registry.enable_tool("toggle_tool")
        assert tool_registry.has_tool("toggle_tool")

    def test_to_openai_tools(self, tool_registry: ToolRegistry) -> None:
        """Can export to OpenAI tools format."""
        tool_registry.register(
            tool=lambda x: {"doubled": x * 2},
            schema=ToolSchema(
                name="doubler",
                description="Double a number",
                parameters=[
                    ParameterSchema(name="x", type="integer", description="The number to double", required=True),
                ],
            ),
        )

        tools = tool_registry.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "doubler"
        assert "x" in tools[0]["function"]["parameters"]["properties"]

    def test_disabled_tools_excluded_from_openai(self, tool_registry: ToolRegistry) -> None:
        """Disabled tools excluded from OpenAI export."""
        tool_registry.register(
            tool=lambda: {},
            schema=ToolSchema(name="active", description="Active tool"),
        )
        tool_registry.register(
            tool=lambda: {},
            schema=ToolSchema(name="inactive", description="Inactive tool"),
        )
        tool_registry.disable_tool("inactive")

        tools = tool_registry.to_openai_tools()
        names = [t["function"]["name"] for t in tools]
        assert "active" in names
        assert "inactive" not in names

    def test_category_organization(self, tool_registry: ToolRegistry) -> None:
        """Tools can be organized by category."""
        tool_registry.register(
            tool=lambda: {},
            schema=ToolSchema(name="nav_tool", description="Navigate", category=ToolCategory.NAVIGATION),
        )
        tool_registry.register(
            tool=lambda: {},
            schema=ToolSchema(name="write_tool", description="Write", writes_document=True, category=ToolCategory.WRITING),
        )

        # Verify tools are registered
        assert tool_registry.has_tool("nav_tool")
        assert tool_registry.has_tool("write_tool")

        # Schemas should have correct categories
        nav_schema = tool_registry.get_schema("nav_tool")
        write_schema = tool_registry.get_schema("write_tool")
        assert nav_schema is not None
        assert nav_schema.category == ToolCategory.NAVIGATION
        assert write_schema is not None
        assert write_schema.category == ToolCategory.WRITING


# -----------------------------------------------------------------------------
# Test: Tool Dispatcher
# -----------------------------------------------------------------------------


class TestToolDispatcher:
    """Tests for ToolDispatcher class."""

    @pytest.fixture
    def dispatcher(self, tool_registry: ToolRegistry) -> ToolDispatcher:
        """Create a dispatcher with test tools."""
        tool_registry.register(
            tool=lambda: {"status": "ok"},
            schema=ToolSchema(name="simple_tool", description="A simple tool"),
        )
        tool_registry.register(
            tool=lambda x, y: {"sum": x + y},
            schema=ToolSchema(
                name="adder",
                description="Add two numbers",
                parameters=[
                    ParameterSchema(name="x", type="integer", description="First number", required=True),
                    ParameterSchema(name="y", type="integer", description="Second number", required=True),
                ],
            ),
        )
        return ToolDispatcher(registry=tool_registry)

    @pytest.mark.asyncio
    async def test_dispatch_simple_tool(self, dispatcher: ToolDispatcher) -> None:
        """Can dispatch a simple tool."""
        result = await dispatcher.dispatch("simple_tool", {})

        assert result.success
        assert result.result == {"status": "ok"}
        assert result.tool_name == "simple_tool"

    @pytest.mark.asyncio
    async def test_dispatch_with_args(self, dispatcher: ToolDispatcher) -> None:
        """Can dispatch a tool with arguments."""
        result = await dispatcher.dispatch("adder", {"x": 5, "y": 3})

        assert result.success
        assert result.result == {"sum": 8}

    @pytest.mark.asyncio
    async def test_dispatch_unregistered_tool(self, dispatcher: ToolDispatcher) -> None:
        """Dispatching unregistered tool returns error."""
        result = await dispatcher.dispatch("nonexistent", {})

        assert not result.success
        assert result.error is not None
        assert result.error.error_code == ErrorCode.INVALID_PARAMETER

    @pytest.mark.asyncio
    async def test_dispatch_tracks_time(self, dispatcher: ToolDispatcher) -> None:
        """Dispatch tracks execution time."""
        result = await dispatcher.dispatch("simple_tool", {})

        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_batch_dispatch(self, dispatcher: ToolDispatcher) -> None:
        """Can batch dispatch multiple tools."""
        calls = [
            ("simple_tool", {}),
            ("adder", {"x": 1, "y": 2}),
        ]

        results = await dispatcher.dispatch_batch(calls)

        assert len(results) == 2
        assert results[0].success
        assert results[1].success
        assert results[1].result == {"sum": 3}

    @pytest.mark.asyncio
    async def test_batch_stop_on_error(self, tool_registry: ToolRegistry) -> None:
        """Batch can stop on first error."""

        def failing(**kwargs: Any) -> dict[str, Any]:
            raise ToolError(error_code=ErrorCode.INTERNAL_ERROR, message="Boom")

        tool_registry.register(
            tool=lambda: {"ok": True},
            schema=ToolSchema(name="ok_tool", description="OK"),
        )
        tool_registry.register(
            tool=failing,
            schema=ToolSchema(name="fail_tool", description="Fails"),
        )

        dispatcher = ToolDispatcher(registry=tool_registry)
        calls = [
            ("ok_tool", {}),
            ("fail_tool", {}),
            ("ok_tool", {}),  # Should not run
        ]

        results = await dispatcher.dispatch_batch(calls, stop_on_error=True)

        assert len(results) == 2
        assert results[0].success
        assert not results[1].success


# -----------------------------------------------------------------------------
# Test: Error Handling
# -----------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error handling in tool dispatch."""

    @pytest.mark.asyncio
    async def test_tool_error_preserved(self, tool_registry: ToolRegistry) -> None:
        """ToolError is preserved in dispatch result."""

        def error_tool(**kwargs: Any) -> dict[str, Any]:
            raise TabNotFoundError(message="No tab", tab_id="missing")

        tool_registry.register(
            tool=error_tool,
            schema=ToolSchema(name="error_tool", description="Errors"),
        )

        dispatcher = ToolDispatcher(registry=tool_registry)
        result = await dispatcher.dispatch("error_tool", {})

        assert not result.success
        assert result.error is not None
        assert result.error.error_code == ErrorCode.TAB_NOT_FOUND

    @pytest.mark.asyncio
    async def test_exception_wrapped(self, tool_registry: ToolRegistry) -> None:
        """Unexpected exceptions wrapped in ToolError."""

        def broken(**kwargs: Any) -> dict[str, Any]:
            raise ValueError("Something broke")

        tool_registry.register(
            tool=broken,
            schema=ToolSchema(name="broken", description="Broken"),
        )

        dispatcher = ToolDispatcher(registry=tool_registry)
        result = await dispatcher.dispatch("broken", {})

        assert not result.success
        assert result.error is not None
        assert result.error.error_code == ErrorCode.INTERNAL_ERROR

    def test_dispatch_result_to_dict(self) -> None:
        """DispatchResult serializes correctly."""
        # Success case
        success = DispatchResult(
            success=True,
            result={"data": "value"},
            tool_name="test",
            execution_time_ms=10.5,
        )
        success_dict = success.to_dict()
        assert success_dict["success"] is True
        assert success_dict["result"] == {"data": "value"}

        # Error case
        error = DispatchResult(
            success=False,
            result=None,
            error=ToolError(error_code=ErrorCode.INTERNAL_ERROR, message="Failed"),
            tool_name="test",
        )
        error_dict = error.to_dict()
        assert error_dict["success"] is False
        assert "error" in error_dict


# -----------------------------------------------------------------------------
# Test: Editor Lock Integration
# -----------------------------------------------------------------------------


class TestEditorLockIntegration:
    """Tests for editor lock operations."""

    def test_lock_manager_singleton(self) -> None:
        """Lock manager is singleton."""
        manager1 = get_lock_manager()
        manager2 = get_lock_manager()
        assert manager1 is manager2

    def test_acquire_release_lock(self) -> None:
        """Can acquire and release lock."""
        manager = get_lock_manager()

        session = manager.acquire(LockReason.AI_TURN)
        assert session is not None
        assert manager.is_locked  # is_locked is a property

        # Release via manager, not session
        manager.release(session.session_id)
        assert not manager.is_locked

    def test_lock_prevents_concurrent(self) -> None:
        """Lock prevents concurrent access."""
        manager = get_lock_manager()

        session1 = manager.acquire(LockReason.AI_TURN)
        assert session1 is not None

        # Second acquire attempt should return None when locked
        session2 = manager.acquire(LockReason.AI_TURN)
        assert session2 is None

        manager.release(session1.session_id)


# -----------------------------------------------------------------------------
# Test: Transaction Integration
# -----------------------------------------------------------------------------


class TestTransactionIntegration:
    """Tests for transaction operations."""

    def test_transaction_manager_singleton(self) -> None:
        """Transaction manager is singleton."""
        manager1 = get_transaction_manager()
        manager2 = get_transaction_manager()
        assert manager1 is manager2

    def test_transaction_lifecycle(self) -> None:
        """Transaction has proper lifecycle."""
        manager = get_transaction_manager()
        # create_transaction automatically begins the transaction
        tx = manager.create_transaction(tab_ids=["tab1"])

        # Transaction starts in ACTIVE state after creation
        assert tx.state == TransactionState.ACTIVE

        tx.commit()
        assert tx.state == TransactionState.COMMITTED

    def test_transaction_rollback(self) -> None:
        """Transaction can rollback."""
        manager = get_transaction_manager()
        tx = manager.create_transaction(tab_ids=["tab1"])

        # Transaction is already active
        assert tx.state == TransactionState.ACTIVE
        tx.stage_change("tab1", "old", "new")
        tx.rollback(reason="Test rollback")

        assert tx.state == TransactionState.ROLLED_BACK


# -----------------------------------------------------------------------------
# Test: Full Integration Scenarios
# -----------------------------------------------------------------------------


class TestFullIntegrationScenarios:
    """End-to-end integration scenarios."""

    @pytest.mark.asyncio
    async def test_multi_tool_workflow(self, tool_registry: ToolRegistry) -> None:
        """Test a realistic multi-tool workflow."""
        # Simulate document state
        documents = {"tab1": "Line 1\nLine 2\nLine 3\n"}

        # Register tools that interact with the state
        tool_registry.register(
            tool=lambda tab_id="tab1": {
                "content": documents.get(tab_id, ""),
                "line_count": len(documents.get(tab_id, "").split("\n")),
            },
            schema=ToolSchema(name="read_doc", description="Read document"),
        )

        def write_doc(tab_id: str, content: str) -> dict[str, Any]:
            documents[tab_id] = content
            return {"success": True, "new_line_count": len(content.split("\n"))}

        tool_registry.register(
            tool=write_doc,
            schema=ToolSchema(name="write_doc", description="Write document", writes_document=True),
        )

        dispatcher = ToolDispatcher(registry=tool_registry)

        # Workflow: read, modify, write
        read_result = await dispatcher.dispatch("read_doc", {"tab_id": "tab1"})
        assert read_result.success
        assert "Line 1" in read_result.result["content"]

        write_result = await dispatcher.dispatch(
            "write_doc",
            {"tab_id": "tab1", "content": "Modified content\n"},
        )
        assert write_result.success

        # Verify change
        verify_result = await dispatcher.dispatch("read_doc", {"tab_id": "tab1"})
        assert verify_result.success
        assert "Modified content" in verify_result.result["content"]

    def test_registry_exports_match_schemas(self) -> None:
        """Exported schemas match expected tools."""
        expected_tools = [
            "list_tabs",
            "read_document",
            "search_document",
            "get_outline",
            "create_document",
            "insert_lines",
            "replace_lines",
            "delete_lines",
            "write_document",
            "find_and_replace",
            "analyze_document",
            "transform_document",
        ]

        for tool_name in expected_tools:
            assert tool_name in ALL_TOOL_SCHEMAS, f"Missing schema for {tool_name}"
            schema = ALL_TOOL_SCHEMAS[tool_name]
            assert schema.name == tool_name
            assert schema.description


# -----------------------------------------------------------------------------
# Test: Version Manager (Light Integration)
# -----------------------------------------------------------------------------


class TestVersionManagerIntegration:
    """Light integration tests for version manager."""

    def test_version_manager_singleton(self) -> None:
        """Version manager is singleton."""
        manager1 = get_version_manager()
        manager2 = get_version_manager()
        assert manager1 is manager2

    def test_can_register_and_get_token(self) -> None:
        """Can register tab and get version token."""
        manager = get_version_manager()

        token = manager.register_tab(
            tab_id="test_tab",
            document_id="doc1",
            content_hash="abc123",
        )

        assert token is not None
        assert token.tab_id == "test_tab"
        assert token.version_id >= 1

    def test_token_serialization(self) -> None:
        """Token can be serialized and parsed."""
        manager = get_version_manager()

        token = manager.register_tab(
            tab_id="serial_tab",
            document_id="doc2",
            content_hash="def456",
        )

        token_str = token.to_string()
        assert "serial_tab" in token_str

        # Use VersionToken.from_string() for parsing
        parsed = VersionToken.from_string(token_str)
        assert parsed.tab_id == "serial_tab"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
