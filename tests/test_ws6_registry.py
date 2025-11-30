"""Tests for WS6: Tool Registry & Integration.

Tests cover:
- WS6.1: Tool Registry (ParameterSchema, ToolSchema, ToolRegistry)
- WS6.2: Tool Dispatcher (dispatch, transactions, errors)
- WS6.3: Prompt Updates (system_prompt_v2, formatting)
"""

from __future__ import annotations

import pytest
from typing import Any, Mapping
from unittest.mock import MagicMock, AsyncMock

from tinkerbell.ai.tools.tool_registry import (
    ToolCategory,
    ParameterSchema,
    ToolSchema,
    ToolRegistration,
    ToolRegistry,
    RegistrationFailure,
    RegistrationError,
    get_tool_registry,
    reset_tool_registry,
    # Schema constants
    VERSION_TOKEN_PARAM,
    TAB_ID_PARAM,
    LIST_TABS_SCHEMA,
    READ_DOCUMENT_SCHEMA,
    INSERT_LINES_SCHEMA,
    ALL_TOOL_SCHEMAS,
)
from tinkerbell.ai.tools.base import BaseTool, ReadOnlyTool, WriteTool, ToolContext, ToolResult
from tinkerbell.ai.tools.errors import ErrorCode, ToolError, InvalidVersionTokenError
from tinkerbell.ai.tools.version import VersionManager
from tinkerbell.ai.orchestration.tool_dispatcher import (
    ToolDispatcher,
    DispatchResult,
    create_tool_dispatcher,
)
from tinkerbell.ai.prompts import (
    system_prompt_v2,
    format_document_context,
    format_error_context,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def version_manager() -> VersionManager:
    """Fresh version manager for testing."""
    return VersionManager()


@pytest.fixture
def tool_registry() -> ToolRegistry:
    """Fresh tool registry for testing."""
    reset_tool_registry()
    return get_tool_registry()


# =============================================================================
# WS6.1: Tool Registry Tests
# =============================================================================


class TestParameterSchema:
    """Tests for ParameterSchema."""

    def test_basic_string_param(self) -> None:
        """Test basic string parameter schema."""
        param = ParameterSchema(
            name="query",
            type="string",
            description="Search query",
            required=True,
        )
        schema = param.to_json_schema()
        assert schema["type"] == "string"
        assert schema["description"] == "Search query"
        assert "required" not in schema  # required is handled at parent level

    def test_integer_param_with_bounds(self) -> None:
        """Test integer parameter with min/max."""
        param = ParameterSchema(
            name="count",
            type="integer",
            description="Item count",
            minimum=1,
            maximum=100,
        )
        schema = param.to_json_schema()
        assert schema["type"] == "integer"
        assert schema["minimum"] == 1
        assert schema["maximum"] == 100

    def test_enum_param(self) -> None:
        """Test parameter with enum values."""
        param = ParameterSchema(
            name="mode",
            type="string",
            description="Search mode",
            enum=["exact", "regex", "semantic"],
        )
        schema = param.to_json_schema()
        assert schema["enum"] == ["exact", "regex", "semantic"]

    def test_default_value(self) -> None:
        """Test parameter with default value."""
        param = ParameterSchema(
            name="enabled",
            type="boolean",
            description="Enable feature",
            default=True,
        )
        schema = param.to_json_schema()
        assert schema["default"] is True

    def test_string_length_constraints(self) -> None:
        """Test string with length constraints."""
        param = ParameterSchema(
            name="name",
            type="string",
            description="Name field",
            min_length=1,
            max_length=100,
        )
        schema = param.to_json_schema()
        assert schema["minLength"] == 1
        assert schema["maxLength"] == 100


class TestToolSchema:
    """Tests for ToolSchema."""

    def test_basic_schema(self) -> None:
        """Test basic tool schema."""
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            category=ToolCategory.UTILITY,
        )
        assert schema.name == "test_tool"
        assert schema.description == "A test tool"
        assert schema.category == ToolCategory.UTILITY

    def test_schema_with_parameters(self) -> None:
        """Test schema with parameters."""
        schema = ToolSchema(
            name="search",
            description="Search tool",
            parameters=[
                ParameterSchema(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
                ParameterSchema(
                    name="limit",
                    type="integer",
                    description="Result limit",
                    required=False,
                    default=10,
                ),
            ],
        )
        json_schema = schema.to_json_schema()
        assert "properties" in json_schema
        assert "query" in json_schema["properties"]
        assert "limit" in json_schema["properties"]
        assert json_schema["required"] == ["query"]

    def test_schema_json_output(self) -> None:
        """Test JSON schema output format."""
        schema = ToolSchema(
            name="test",
            description="Test tool",
            parameters=[
                ParameterSchema(
                    name="input",
                    type="string",
                    description="Input",
                    required=True,
                ),
            ],
        )
        json_schema = schema.to_json_schema()
        assert json_schema["type"] == "object"
        assert json_schema["additionalProperties"] is False
        assert "input" in json_schema["properties"]

    def test_empty_parameters(self) -> None:
        """Test schema with no parameters."""
        schema = ToolSchema(
            name="list_tabs",
            description="List all tabs",
            parameters=[],
        )
        json_schema = schema.to_json_schema()
        assert json_schema["properties"] == {}
        assert "required" not in json_schema


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self, tool_registry: ToolRegistry) -> None:
        """Test basic tool registration."""
        mock_tool = MagicMock()
        schema = ToolSchema(
            name="test_tool",
            description="Test",
            category=ToolCategory.UTILITY,
        )
        tool_registry.register(mock_tool, schema=schema)
        
        assert tool_registry.has_tool("test_tool")
        assert tool_registry.get_tool("test_tool") is mock_tool

    def test_register_with_params(self, tool_registry: ToolRegistry) -> None:
        """Test registration with parameter dict."""
        mock_tool = MagicMock()
        mock_tool.name = "my_tool"
        tool_registry.register(
            mock_tool,
            name="my_tool",
            description="My tool",
            parameters={
                "properties": {
                    "input": {"type": "string", "description": "Input"},
                },
                "required": ["input"],
            },
        )
        
        schema = tool_registry.get_schema("my_tool")
        assert schema is not None
        assert schema.name == "my_tool"

    def test_unregister_tool(self, tool_registry: ToolRegistry) -> None:
        """Test tool unregistration."""
        mock_tool = MagicMock()
        tool_registry.register(
            mock_tool,
            schema=ToolSchema(name="temp", description="Temporary"),
        )
        
        assert tool_registry.has_tool("temp")
        result = tool_registry.unregister("temp")
        assert result is True
        assert not tool_registry.has_tool("temp")

    def test_unregister_nonexistent(self, tool_registry: ToolRegistry) -> None:
        """Test unregistering nonexistent tool."""
        result = tool_registry.unregister("nonexistent")
        assert result is False

    def test_list_tools(self, tool_registry: ToolRegistry) -> None:
        """Test listing registered tools."""
        tool_registry.register(
            MagicMock(),
            schema=ToolSchema(name="tool1", description="Tool 1", category=ToolCategory.NAVIGATION),
        )
        tool_registry.register(
            MagicMock(),
            schema=ToolSchema(name="tool2", description="Tool 2", category=ToolCategory.WRITING),
        )
        
        all_tools = tool_registry.list_tools()
        assert "tool1" in all_tools
        assert "tool2" in all_tools

    def test_list_by_category(self, tool_registry: ToolRegistry) -> None:
        """Test listing tools by category."""
        tool_registry.register(
            MagicMock(),
            schema=ToolSchema(name="nav1", description="Nav 1", category=ToolCategory.NAVIGATION),
        )
        tool_registry.register(
            MagicMock(),
            schema=ToolSchema(name="write1", description="Write 1", category=ToolCategory.WRITING),
        )
        
        nav_tools = tool_registry.list_tools(category=ToolCategory.NAVIGATION)
        assert "nav1" in nav_tools
        assert "write1" not in nav_tools

    def test_enable_disable_tool(self, tool_registry: ToolRegistry) -> None:
        """Test enabling/disabling tools."""
        tool_registry.register(
            MagicMock(),
            schema=ToolSchema(name="toggle", description="Toggle"),
        )
        
        assert tool_registry.has_tool("toggle")
        
        tool_registry.disable_tool("toggle")
        assert not tool_registry.has_tool("toggle")  # has_tool checks enabled
        
        tool_registry.enable_tool("toggle")
        assert tool_registry.has_tool("toggle")

    def test_to_openai_tools(self, tool_registry: ToolRegistry) -> None:
        """Test OpenAI tools format generation."""
        tool_registry.register(
            MagicMock(),
            schema=ToolSchema(
                name="test",
                description="Test tool",
                parameters=[
                    ParameterSchema(
                        name="input",
                        type="string",
                        description="Input",
                        required=True,
                    ),
                ],
            ),
        )
        
        openai_tools = tool_registry.to_openai_tools()
        assert len(openai_tools) == 1
        assert openai_tools[0]["type"] == "function"
        assert openai_tools[0]["function"]["name"] == "test"
        assert openai_tools[0]["function"]["strict"] is True

    def test_clear_registry(self, tool_registry: ToolRegistry) -> None:
        """Test clearing all tools."""
        tool_registry.register(
            MagicMock(),
            schema=ToolSchema(name="tool1", description="Tool 1"),
        )
        tool_registry.register(
            MagicMock(),
            schema=ToolSchema(name="tool2", description="Tool 2"),
        )
        
        tool_registry.clear()
        assert tool_registry.list_tools() == []


class TestPredefinedSchemas:
    """Tests for predefined tool schemas."""

    def test_version_token_param(self) -> None:
        """Test VERSION_TOKEN_PARAM schema."""
        assert VERSION_TOKEN_PARAM.name == "version_token"
        assert VERSION_TOKEN_PARAM.type == "string"
        assert VERSION_TOKEN_PARAM.required is True

    def test_tab_id_param(self) -> None:
        """Test TAB_ID_PARAM schema."""
        assert TAB_ID_PARAM.name == "tab_id"
        assert TAB_ID_PARAM.required is False

    def test_list_tabs_schema(self) -> None:
        """Test LIST_TABS_SCHEMA."""
        assert LIST_TABS_SCHEMA.name == "list_tabs"
        assert LIST_TABS_SCHEMA.category == ToolCategory.NAVIGATION
        assert LIST_TABS_SCHEMA.writes_document is False

    def test_read_document_schema(self) -> None:
        """Test READ_DOCUMENT_SCHEMA."""
        assert READ_DOCUMENT_SCHEMA.name == "read_document"
        assert READ_DOCUMENT_SCHEMA.requires_version is False
        json_schema = READ_DOCUMENT_SCHEMA.to_json_schema()
        assert "tab_id" in json_schema["properties"]

    def test_insert_lines_schema(self) -> None:
        """Test INSERT_LINES_SCHEMA."""
        assert INSERT_LINES_SCHEMA.name == "insert_lines"
        assert INSERT_LINES_SCHEMA.requires_version is True
        assert INSERT_LINES_SCHEMA.writes_document is True
        json_schema = INSERT_LINES_SCHEMA.to_json_schema()
        assert "version_token" in json_schema["required"]
        assert "after_line" in json_schema["required"]
        assert "content" in json_schema["required"]

    def test_all_schemas_dict(self) -> None:
        """Test ALL_TOOL_SCHEMAS contains all tools."""
        expected_tools = [
            "list_tabs", "read_document", "search_document", "get_outline",
            "create_document", "insert_lines", "replace_lines", "delete_lines",
            "write_document", "find_and_replace",
            "analyze_document", "transform_document",
            "diff_builder", "validate_snippet",
        ]
        for tool_name in expected_tools:
            assert tool_name in ALL_TOOL_SCHEMAS, f"Missing: {tool_name}"


class TestRegistrationError:
    """Tests for registration error handling."""

    def test_registration_failure(self) -> None:
        """Test RegistrationFailure dataclass."""
        failure = RegistrationFailure(
            name="broken_tool",
            error=ValueError("Something went wrong"),
        )
        assert failure.name == "broken_tool"
        assert isinstance(failure.error, ValueError)

    def test_registration_error(self) -> None:
        """Test RegistrationError exception."""
        failures = [
            RegistrationFailure(name="tool1", error=ValueError("Error 1")),
            RegistrationFailure(name="tool2", error=TypeError("Error 2")),
        ]
        error = RegistrationError(failures)
        assert "tool1" in str(error)
        assert "tool2" in str(error)
        assert len(error.failures) == 2


# =============================================================================
# WS6.2: Tool Dispatcher Tests
# =============================================================================


class TestDispatchResult:
    """Tests for DispatchResult."""

    def test_success_result(self) -> None:
        """Test successful dispatch result."""
        result = DispatchResult(
            success=True,
            result={"data": "test"},
            tool_name="test_tool",
            execution_time_ms=50.0,
        )
        assert result.success
        assert result.result == {"data": "test"}
        assert result.error is None

    def test_error_result(self) -> None:
        """Test error dispatch result."""
        error = ToolError(error_code=ErrorCode.INVALID_PARAMETER, message="Bad param")
        result = DispatchResult(
            success=False,
            result=None,
            error=error,
            tool_name="test_tool",
        )
        assert not result.success
        assert result.error is error

    def test_to_dict_success(self) -> None:
        """Test to_dict for success."""
        result = DispatchResult(
            success=True,
            result={"key": "value"},
            tool_name="test",
            execution_time_ms=10.0,
        )
        data = result.to_dict()
        assert data["success"] is True
        assert data["result"] == {"key": "value"}
        assert "error" not in data

    def test_to_dict_error(self) -> None:
        """Test to_dict for error."""
        error = ToolError(error_code=ErrorCode.TAB_NOT_FOUND, message="Tab missing")
        result = DispatchResult(
            success=False,
            result=None,
            error=error,
            tool_name="test",
        )
        data = result.to_dict()
        assert data["success"] is False
        assert "error" in data
        assert data["error"]["error"] == "tab_not_found"  # ErrorCode is lowercase


class TestToolDispatcher:
    """Tests for ToolDispatcher."""

    @pytest.fixture
    def dispatcher(self, tool_registry: ToolRegistry) -> ToolDispatcher:
        """Create dispatcher with fresh registry."""
        return ToolDispatcher(registry=tool_registry)

    @pytest.mark.asyncio
    async def test_dispatch_unregistered_tool(self, dispatcher: ToolDispatcher) -> None:
        """Test dispatching to unregistered tool."""
        result = await dispatcher.dispatch("nonexistent", {})
        assert not result.success
        assert result.error is not None
        assert "not registered" in result.error.message

    @pytest.mark.asyncio
    async def test_dispatch_callable_tool(
        self, dispatcher: ToolDispatcher, tool_registry: ToolRegistry
    ) -> None:
        """Test dispatching to a simple callable tool."""
        # Create mock tool that returns a dict
        def mock_tool(**kwargs: Any) -> dict[str, Any]:
            return {"content": "Hello", "input": kwargs.get("input", "")}
        
        tool_registry.register(
            mock_tool,
            schema=ToolSchema(
                name="simple_tool",
                description="Simple tool",
                category=ToolCategory.NAVIGATION,
                writes_document=False,
            ),
        )
        
        result = await dispatcher.dispatch("simple_tool", {"input": "test"})
        assert result.success
        assert result.result["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_dispatch_requires_version_missing(
        self, dispatcher: ToolDispatcher, tool_registry: ToolRegistry
    ) -> None:
        """Test tool requiring version token fails without token."""
        mock_tool = MagicMock()
        tool_registry.register(
            mock_tool,
            schema=ToolSchema(
                name="write_tool",
                description="Write tool",
                requires_version=True,
                writes_document=True,
            ),
        )
        
        # Without version token
        result = await dispatcher.dispatch("write_tool", {"content": "test"})
        assert not result.success
        assert "version_token" in result.error.message.lower()


class TestToolDispatcherBatch:
    """Tests for batch dispatch."""

    @pytest.fixture
    def dispatcher(self, tool_registry: ToolRegistry) -> ToolDispatcher:
        """Create dispatcher with test tools."""
        def mock_tool(**kwargs: Any) -> dict[str, Any]:
            return {"ok": True}
        
        tool_registry.register(
            mock_tool,
            schema=ToolSchema(name="tool1", description="Tool 1"),
        )
        tool_registry.register(
            mock_tool,
            schema=ToolSchema(name="tool2", description="Tool 2"),
        )
        
        return ToolDispatcher(registry=tool_registry)

    @pytest.mark.asyncio
    async def test_batch_dispatch(self, dispatcher: ToolDispatcher) -> None:
        """Test batch dispatch of multiple tools."""
        calls = [
            ("tool1", {"input": "a"}),
            ("tool2", {"input": "b"}),
        ]
        results = await dispatcher.dispatch_batch(calls)
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_batch_stop_on_error(
        self, dispatcher: ToolDispatcher, tool_registry: ToolRegistry
    ) -> None:
        """Test batch stops on error when requested."""
        # Add a failing tool - use a function that raises ToolError
        def failing_tool(**kwargs: Any) -> dict[str, Any]:
            raise ToolError(
                error_code=ErrorCode.INTERNAL_ERROR,
                message="Intentional failure",
            )
        
        tool_registry.register(
            failing_tool,
            schema=ToolSchema(name="failing", description="Fails"),
        )
        
        calls = [
            ("tool1", {}),
            ("failing", {}),
            ("tool2", {}),
        ]
        results = await dispatcher.dispatch_batch(calls, stop_on_error=True)
        # Should stop after failing tool
        assert len(results) == 2
        assert results[0].success
        assert not results[1].success


# =============================================================================
# WS6.3: Prompt Tests
# =============================================================================


class TestSystemPromptV2:
    """Tests for system_prompt_v2."""

    def test_contains_personality(self) -> None:
        """Test prompt contains personality section."""
        prompt = system_prompt_v2()
        assert "TinkerBell" in prompt
        assert "fairy" in prompt.lower()

    def test_contains_tool_sections(self) -> None:
        """Test prompt contains tool documentation."""
        prompt = system_prompt_v2()
        assert "Navigation Tools" in prompt
        assert "Writing Tools" in prompt
        assert "Analysis Tools" in prompt

    def test_contains_workflow(self) -> None:
        """Test prompt contains workflow instructions."""
        prompt = system_prompt_v2()
        assert "Read First" in prompt or "read_document" in prompt
        assert "version_token" in prompt

    def test_contains_error_handling(self) -> None:
        """Test prompt contains error handling."""
        prompt = system_prompt_v2()
        assert "version_mismatch" in prompt
        assert "Recovery" in prompt or "recovery" in prompt


class TestFormatDocumentContext:
    """Tests for format_document_context."""

    def test_basic_formatting(self) -> None:
        """Test basic document context formatting."""
        snapshot = {
            "path": "/path/to/doc.md",
            "file_type": "markdown",
            "total_lines": 100,
            "total_chars": 5000,
            "version_token": "tab1:v1:abc",
        }
        result = format_document_context(snapshot)
        assert "doc.md" in result
        assert "markdown" in result
        assert "100" in result
        assert "5000" in result or "5,000" in result

    def test_large_document_note(self) -> None:
        """Test large document gets special note."""
        snapshot = {
            "path": "large.txt",
            "total_chars": 50000,
            "total_lines": 2000,
        }
        result = format_document_context(snapshot)
        assert "large" in result.lower()

    def test_window_info(self) -> None:
        """Test window info for partial reads."""
        snapshot = {
            "path": "doc.txt",
            "total_lines": 1000,
            "total_chars": 10000,
            "offset": 100,
            "window_lines": 50,
        }
        result = format_document_context(snapshot)
        assert "100" in result
        assert "Window" in result or "window" in result


class TestFormatErrorContext:
    """Tests for format_error_context."""

    def test_basic_error(self) -> None:
        """Test basic error formatting."""
        error = {
            "code": "VERSION_MISMATCH",
            "message": "Document has changed",
        }
        result = format_error_context(error)
        assert "VERSION_MISMATCH" in result
        assert "Document has changed" in result

    def test_error_with_recovery(self) -> None:
        """Test error with recovery hint."""
        error = {
            "code": "STALE_VERSION",
            "message": "Version is stale",
            "recovery_hint": "Call read_document to get fresh token",
        }
        result = format_error_context(error)
        assert "Recovery" in result
        assert "read_document" in result


# =============================================================================
# Integration Tests
# =============================================================================


class TestWS6Integration:
    """Integration tests for WS6 components."""

    def test_registry_with_all_schemas(self) -> None:
        """Test registry can load all predefined schemas."""
        registry = ToolRegistry()
        
        for name, schema in ALL_TOOL_SCHEMAS.items():
            mock_tool = MagicMock()
            registry.register(mock_tool, schema=schema)
        
        assert len(registry.list_tools()) == len(ALL_TOOL_SCHEMAS)

    def test_openai_format_all_tools(self) -> None:
        """Test all tools can be converted to OpenAI format."""
        registry = ToolRegistry()
        
        for name, schema in ALL_TOOL_SCHEMAS.items():
            registry.register(MagicMock(), schema=schema)
        
        openai_tools = registry.to_openai_tools()
        assert len(openai_tools) == len(ALL_TOOL_SCHEMAS)
        
        for tool in openai_tools:
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    @pytest.mark.asyncio
    async def test_dispatcher_with_callable(self) -> None:
        """Test dispatcher works with callable implementation."""
        reset_tool_registry()
        registry = get_tool_registry()
        
        # Create a simple callable tool
        def test_tool(**kwargs: Any) -> dict[str, Any]:
            return {"content": "test content", "input": kwargs.get("input", "")}
        
        registry.register(
            test_tool,
            schema=ToolSchema(
                name="test_read",
                description="Test read tool",
                category=ToolCategory.NAVIGATION,
            ),
        )
        
        dispatcher = ToolDispatcher(registry=registry)
        result = await dispatcher.dispatch("test_read", {"input": "hello"})
        
        assert result.success
        assert result.result["content"] == "test content"

    def test_category_organization(self) -> None:
        """Test tools are properly organized by category."""
        registry = ToolRegistry()
        
        # Register tools from different categories
        for name, schema in ALL_TOOL_SCHEMAS.items():
            registry.register(MagicMock(), schema=schema)
        
        nav_tools = registry.list_tools(category=ToolCategory.NAVIGATION)
        write_tools = registry.list_tools(category=ToolCategory.WRITING)
        
        assert "list_tabs" in nav_tools
        assert "read_document" in nav_tools
        assert "insert_lines" in write_tools
        assert "replace_lines" in write_tools
