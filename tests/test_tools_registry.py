"""Tests for orchestration/tools/registry.py."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from tinkerbell.ai.orchestration.tools import (
    DuplicateToolError,
    SimpleTool,
    Tool,
    ToolCategory,
    ToolNotFoundError,
    ToolRegistry,
    ToolRegistration,
    ToolSpec,
)


# -----------------------------------------------------------------------------
# Test Fixtures and Helpers
# -----------------------------------------------------------------------------


def make_spec(
    name: str = "test_tool",
    description: str = "A test tool",
    category: str = ToolCategory.UTILITY,
) -> ToolSpec:
    """Helper to create a ToolSpec."""
    return ToolSpec(
        name=name,
        description=description,
        parameters={"type": "object", "properties": {"arg": {"type": "string"}}},
        category=category,
    )


def make_simple_tool(
    name: str = "test_tool",
    description: str = "A test tool",
    handler: Any = None,
) -> SimpleTool:
    """Helper to create a SimpleTool."""
    if handler is None:
        handler = lambda args: f"result for {name}"
    return SimpleTool(
        spec=make_spec(name, description),
        handler=handler,
    )


class MockTool:
    """Mock tool implementation for testing."""

    def __init__(self, name: str = "mock_tool") -> None:
        self._name = name
        self._spec = make_spec(name, f"Mock tool: {name}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(self, arguments: Mapping[str, Any]) -> Any:
        return {"mock_result": self._name, "args": dict(arguments)}


# -----------------------------------------------------------------------------
# Tests: ToolSpec
# -----------------------------------------------------------------------------


class TestToolSpec:
    """Tests for ToolSpec dataclass."""

    def test_basic_creation(self) -> None:
        """Test basic ToolSpec creation."""
        spec = ToolSpec(name="test", description="A test")
        assert spec.name == "test"
        assert spec.description == "A test"
        assert spec.category == ToolCategory.UTILITY

    def test_with_parameters(self) -> None:
        """Test ToolSpec with parameters."""
        params = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        spec = ToolSpec(name="read", description="Read file", parameters=params)
        assert spec.parameters["type"] == "object"
        assert "path" in spec.parameters["properties"]

    def test_to_openai_tool(self) -> None:
        """Test conversion to OpenAI format."""
        spec = ToolSpec(
            name="test_func",
            description="Test function",
            parameters={"type": "object", "properties": {}},
        )
        openai_format = spec.to_openai_tool()
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "test_func"
        assert openai_format["function"]["description"] == "Test function"

    def test_to_openai_tool_default_parameters(self) -> None:
        """Test OpenAI format with default parameters."""
        spec = ToolSpec(name="test", description="Test")
        openai_format = spec.to_openai_tool()
        assert openai_format["function"]["parameters"]["type"] == "object"

    def test_to_dict(self) -> None:
        """Test serialization to dict."""
        spec = ToolSpec(
            name="test",
            description="Test",
            category=ToolCategory.READ,
            requires_document=True,
            is_write=False,
        )
        data = spec.to_dict()
        assert data["name"] == "test"
        assert data["category"] == ToolCategory.READ
        assert data["requires_document"] is True

    def test_frozen(self) -> None:
        """Test that ToolSpec is frozen."""
        spec = ToolSpec(name="test", description="Test")
        with pytest.raises(AttributeError):
            spec.name = "modified"  # type: ignore


# -----------------------------------------------------------------------------
# Tests: SimpleTool
# -----------------------------------------------------------------------------


class TestSimpleTool:
    """Tests for SimpleTool class."""

    def test_sync_handler(self) -> None:
        """Test SimpleTool with synchronous handler."""
        tool = SimpleTool(
            spec=make_spec("sync_tool"),
            handler=lambda args: args.get("x", 0) * 2,
        )
        assert tool.name == "sync_tool"
        assert tool.spec.name == "sync_tool"

    def test_async_handler(self) -> None:
        """Test SimpleTool with async handler."""
        async def async_handler(args: dict) -> str:
            return f"async result: {args}"

        tool = SimpleTool(
            spec=make_spec("async_tool"),
            handler=async_handler,
        )
        assert tool.name == "async_tool"

    @pytest.mark.asyncio
    async def test_execute_sync(self) -> None:
        """Test executing synchronous handler."""
        tool = SimpleTool(
            spec=make_spec("sync"),
            handler=lambda args: args.get("value", 0) + 10,
        )
        result = await tool.execute({"value": 5})
        assert result == 15

    @pytest.mark.asyncio
    async def test_execute_async(self) -> None:
        """Test executing asynchronous handler."""
        async def handler(args: dict) -> str:
            return f"async: {args.get('msg', '')}"

        tool = SimpleTool(spec=make_spec("async"), handler=handler)
        result = await tool.execute({"msg": "hello"})
        assert result == "async: hello"


# -----------------------------------------------------------------------------
# Tests: ToolRegistry - Basic Operations
# -----------------------------------------------------------------------------


class TestToolRegistryBasic:
    """Tests for basic ToolRegistry operations."""

    def test_empty_registry(self) -> None:
        """Test empty registry."""
        registry = ToolRegistry()
        assert len(registry) == 0
        assert registry.list_tools() == []

    def test_register_tool(self) -> None:
        """Test registering a tool."""
        registry = ToolRegistry()
        tool = MockTool("my_tool")
        registration = registry.register(tool)
        assert registration.name == "my_tool"
        assert registration.enabled is True
        assert len(registry) == 1

    def test_register_duplicate_raises(self) -> None:
        """Test that duplicate registration raises."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"))
        with pytest.raises(DuplicateToolError) as exc_info:
            registry.register(MockTool("tool"))
        assert exc_info.value.name == "tool"

    def test_register_with_override(self) -> None:
        """Test registering with allow_override."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"))
        registry.register(MockTool("tool"), allow_override=True)
        assert len(registry) == 1

    def test_register_with_metadata(self) -> None:
        """Test registering with metadata."""
        registry = ToolRegistry()
        registration = registry.register(
            MockTool("tool"),
            metadata={"version": "1.0", "author": "test"},
        )
        assert registration.metadata["version"] == "1.0"
        assert registration.metadata["author"] == "test"

    def test_register_function(self) -> None:
        """Test registering a function."""
        registry = ToolRegistry()
        registration = registry.register_function(
            spec=make_spec("func_tool"),
            handler=lambda args: "result",
        )
        assert registration.name == "func_tool"
        assert registry.has("func_tool")


# -----------------------------------------------------------------------------
# Tests: ToolRegistry - Retrieval
# -----------------------------------------------------------------------------


class TestToolRegistryRetrieval:
    """Tests for ToolRegistry retrieval operations."""

    def test_get_existing(self) -> None:
        """Test getting an existing tool."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"))
        tool = registry.get("tool")
        assert tool is not None
        assert tool.name == "tool"

    def test_get_nonexistent(self) -> None:
        """Test getting a nonexistent tool."""
        registry = ToolRegistry()
        tool = registry.get("nonexistent")
        assert tool is None

    def test_get_disabled(self) -> None:
        """Test getting a disabled tool returns None."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"), enabled=False)
        tool = registry.get("tool")
        assert tool is None

    def test_get_required_existing(self) -> None:
        """Test get_required with existing tool."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"))
        tool = registry.get_required("tool")
        assert tool.name == "tool"

    def test_get_required_nonexistent_raises(self) -> None:
        """Test get_required raises for nonexistent tool."""
        registry = ToolRegistry()
        with pytest.raises(ToolNotFoundError) as exc_info:
            registry.get_required("missing")
        assert exc_info.value.name == "missing"

    def test_get_registration(self) -> None:
        """Test getting registration record."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"), metadata={"key": "value"})
        registration = registry.get_registration("tool")
        assert registration is not None
        assert registration.metadata["key"] == "value"

    def test_has_enabled(self) -> None:
        """Test has() with enabled tool."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"))
        assert registry.has("tool") is True

    def test_has_disabled(self) -> None:
        """Test has() with disabled tool."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"), enabled=False)
        assert registry.has("tool") is False

    def test_contains(self) -> None:
        """Test __contains__ operator."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"))
        assert "tool" in registry
        assert "missing" not in registry


# -----------------------------------------------------------------------------
# Tests: ToolRegistry - Listing
# -----------------------------------------------------------------------------


class TestToolRegistryListing:
    """Tests for ToolRegistry listing operations."""

    def test_list_tools(self) -> None:
        """Test listing tool specs."""
        registry = ToolRegistry()
        registry.register(MockTool("tool1"))
        registry.register(MockTool("tool2"))
        specs = registry.list_tools()
        assert len(specs) == 2
        names = [s.name for s in specs]
        assert "tool1" in names
        assert "tool2" in names

    def test_list_tools_excludes_disabled(self) -> None:
        """Test that list_tools excludes disabled by default."""
        registry = ToolRegistry()
        registry.register(MockTool("enabled"))
        registry.register(MockTool("disabled"), enabled=False)
        specs = registry.list_tools()
        assert len(specs) == 1
        assert specs[0].name == "enabled"

    def test_list_tools_include_disabled(self) -> None:
        """Test listing tools with include_disabled."""
        registry = ToolRegistry()
        registry.register(MockTool("enabled"))
        registry.register(MockTool("disabled"), enabled=False)
        specs = registry.list_tools(include_disabled=True)
        assert len(specs) == 2

    def test_list_names(self) -> None:
        """Test listing tool names."""
        registry = ToolRegistry()
        registry.register(MockTool("a"))
        registry.register(MockTool("b"))
        names = registry.list_names()
        assert set(names) == {"a", "b"}

    def test_list_registrations(self) -> None:
        """Test listing registrations."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"))
        registrations = registry.list_registrations()
        assert len(registrations) == 1
        assert isinstance(registrations[0], ToolRegistration)


# -----------------------------------------------------------------------------
# Tests: ToolRegistry - OpenAI Format
# -----------------------------------------------------------------------------


class TestToolRegistryOpenAI:
    """Tests for OpenAI format generation."""

    def test_get_openai_tools(self) -> None:
        """Test getting tools in OpenAI format."""
        registry = ToolRegistry()
        registry.register(MockTool("tool1"))
        registry.register(MockTool("tool2"))
        tools = registry.get_openai_tools()
        assert len(tools) == 2
        assert all(t["type"] == "function" for t in tools)

    def test_get_openai_tools_filtered(self) -> None:
        """Test filtering OpenAI tools."""
        registry = ToolRegistry()
        registry.register(MockTool("tool1"))
        registry.register(MockTool("tool2"))
        registry.register(MockTool("tool3"))
        tools = registry.get_openai_tools(filter_names=["tool1", "tool3"])
        assert len(tools) == 2
        names = [t["function"]["name"] for t in tools]
        assert "tool2" not in names

    def test_get_openai_tools_excludes_disabled(self) -> None:
        """Test that disabled tools are excluded from OpenAI format."""
        registry = ToolRegistry()
        registry.register(MockTool("enabled"))
        registry.register(MockTool("disabled"), enabled=False)
        tools = registry.get_openai_tools()
        assert len(tools) == 1


# -----------------------------------------------------------------------------
# Tests: ToolRegistry - Enable/Disable
# -----------------------------------------------------------------------------


class TestToolRegistryEnableDisable:
    """Tests for enable/disable operations."""

    def test_enable(self) -> None:
        """Test enabling a tool."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"), enabled=False)
        assert registry.has("tool") is False
        result = registry.enable("tool")
        assert result is True
        assert registry.has("tool") is True

    def test_enable_nonexistent(self) -> None:
        """Test enabling nonexistent tool."""
        registry = ToolRegistry()
        result = registry.enable("missing")
        assert result is False

    def test_disable(self) -> None:
        """Test disabling a tool."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"))
        assert registry.has("tool") is True
        result = registry.disable("tool")
        assert result is True
        assert registry.has("tool") is False

    def test_disable_nonexistent(self) -> None:
        """Test disabling nonexistent tool."""
        registry = ToolRegistry()
        result = registry.disable("missing")
        assert result is False


# -----------------------------------------------------------------------------
# Tests: ToolRegistry - Unregister/Clear
# -----------------------------------------------------------------------------


class TestToolRegistryUnregisterClear:
    """Tests for unregister and clear operations."""

    def test_unregister(self) -> None:
        """Test unregistering a tool."""
        registry = ToolRegistry()
        registry.register(MockTool("tool"))
        result = registry.unregister("tool")
        assert result is True
        assert "tool" not in registry

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering nonexistent tool."""
        registry = ToolRegistry()
        result = registry.unregister("missing")
        assert result is False

    def test_clear(self) -> None:
        """Test clearing all tools."""
        registry = ToolRegistry()
        registry.register(MockTool("tool1"))
        registry.register(MockTool("tool2"))
        registry.clear()
        assert len(registry) == 0
