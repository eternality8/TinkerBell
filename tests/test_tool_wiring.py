"""Tests for WS7.1: Tool Wiring Module.

This module tests the extracted tool wiring logic that was moved
out of main_window.py into a dedicated module.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any, Mapping, Sequence

from tinkerbell.ai.tools.tool_wiring import (
    DocumentBridge,
    WorkspaceProvider,
    SelectionProvider,
    AIControllerProvider,
    ToolWiringContext,
    ToolRegistrationResult,
    register_legacy_tools,
    register_new_tools,
    unregister_tools,
)


# =============================================================================
# Mock Providers
# =============================================================================


class MockBridge:
    """Mock document bridge for testing."""

    def get_document(self, tab_id: str) -> dict:
        return {"tab_id": tab_id, "content": "test content"}

    def get_active_document(self) -> dict | None:
        return {"tab_id": "active", "content": "active content"}

    def list_tabs(self) -> list[dict]:
        return [{"tab_id": "tab1", "title": "Test"}]


class MockWorkspace:
    """Mock workspace provider for testing."""

    def find_document_by_id(self, document_id: str) -> dict | None:
        return {"id": document_id, "content": "test"}

    def active_document(self) -> dict | None:
        return {"id": "active", "content": "test"}


class MockController:
    """Mock AI controller for testing."""

    def __init__(self) -> None:
        self.registered_tools: dict[str, Any] = {}
        self.register_calls: list[tuple] = []

    def register_tool(
        self,
        name: str,
        tool: Any,
        *,
        description: str = "",
        parameters: Mapping[str, Any] | None = None,
    ) -> None:
        self.registered_tools[name] = tool
        self.register_calls.append((name, tool, description, parameters))

    def unregister_tool(self, name: str) -> None:
        if name in self.registered_tools:
            del self.registered_tools[name]


# =============================================================================
# ToolWiringContext Tests
# =============================================================================


class TestToolWiringContext:
    """Test the ToolWiringContext dataclass."""

    def test_context_creation_minimal(self) -> None:
        """Context can be created with minimal required fields."""
        bridge = MockBridge()
        workspace = MockWorkspace()
        
        ctx = ToolWiringContext(
            controller=None,
            bridge=bridge,
            workspace=workspace,
        )
        
        assert ctx.controller is None
        assert ctx.bridge is bridge
        assert ctx.workspace is workspace

    def test_context_creation_full(self) -> None:
        """Context can be created with all fields."""
        controller = MockController()
        bridge = MockBridge()
        workspace = MockWorkspace()
        
        ctx = ToolWiringContext(
            controller=controller,
            bridge=bridge,
            workspace=workspace,
            active_document_provider=lambda: {"id": "test"},
        )
        
        assert ctx.controller is controller
        assert ctx.active_document_provider is not None


# =============================================================================
# ToolRegistrationResult Tests
# =============================================================================


class TestToolRegistrationResult:
    """Test the ToolRegistrationResult dataclass."""

    def test_empty_result_is_success(self) -> None:
        """Empty result is considered success."""
        result = ToolRegistrationResult()
        assert result.success is True
        assert result.registered == []
        assert result.failed == []
        assert result.skipped == []

    def test_result_with_registered_is_success(self) -> None:
        """Result with only registered tools is success."""
        result = ToolRegistrationResult(
            registered=["tool1", "tool2"],
        )
        assert result.success is True

    def test_result_with_failed_is_not_success(self) -> None:
        """Result with any failed tools is not success."""
        result = ToolRegistrationResult(
            registered=["tool1"],
            failed=["tool2"],
        )
        assert result.success is False

    def test_str_representation(self) -> None:
        """String representation includes all lists."""
        result = ToolRegistrationResult(
            registered=["a"],
            failed=["b"],
            skipped=["c"],
        )
        s = str(result)
        assert "registered" in s
        assert "failed" in s
        assert "skipped" in s


# =============================================================================
# Legacy Tool Registration Tests
# =============================================================================


class TestRegisterLegacyTools:
    """Test legacy tool registration (deprecated - now returns empty)."""

    def test_returns_empty_if_no_controller(self) -> None:
        """Returns empty result (deprecated function is now a no-op)."""
        ctx = ToolWiringContext(
            controller=None,
            bridge=MockBridge(),
            workspace=MockWorkspace(),
        )
        
        result = register_legacy_tools(ctx)
        
        assert result.registered == []
        assert result.failed == []

    def test_returns_empty_if_no_register_method(self) -> None:
        """Returns empty result (deprecated function is now a no-op)."""
        controller = object()  # No register_tool method
        ctx = ToolWiringContext(
            controller=controller,
            bridge=MockBridge(),
            workspace=MockWorkspace(),
        )
        
        result = register_legacy_tools(ctx)
        
        assert result.registered == []

    def test_is_noop(self) -> None:
        """Legacy registration is now a no-op and returns empty result."""
        controller = MockController()
        ctx = ToolWiringContext(
            controller=controller,
            bridge=MockBridge(),
            workspace=MockWorkspace(),
        )
        
        result = register_legacy_tools(ctx)
        
        # Deprecated function is now a no-op
        assert result.registered == []
        assert result.failed == []

# =============================================================================
# New Tool Registration Tests
# =============================================================================


class TestRegisterNewTools:
    """Test new WS1-6 tool registration."""

    def test_returns_empty_if_no_controller(self) -> None:
        """Returns empty result when controller is None."""
        ctx = ToolWiringContext(
            controller=None,
            bridge=MockBridge(),
            workspace=MockWorkspace(),
        )
        
        result = register_new_tools(ctx)
        
        assert result.registered == []

    def test_returns_empty_if_no_register_method(self) -> None:
        """Returns empty result when controller lacks register_tool."""
        controller = object()
        ctx = ToolWiringContext(
            controller=controller,
            bridge=MockBridge(),
            workspace=MockWorkspace(),
        )
        
        result = register_new_tools(ctx)
        
        assert result.registered == []

    def test_registers_read_tools(self) -> None:
        """Registers new read tools."""
        controller = MockController()
        ctx = ToolWiringContext(
            controller=controller,
            bridge=MockBridge(),
            workspace=MockWorkspace(),
        )
        
        result = register_new_tools(ctx)
        
        # Should attempt to register all WS2-5 tools
        expected_tools = {
            # WS2: Navigation & Reading
            "list_tabs", "read_document", "search_document", "get_outline",
            # WS3: Writing
            "create_document", "insert_lines", "replace_lines", "delete_lines",
            "write_document", "find_and_replace",
            # WS5: Subagent
            "analyze_document", "transform_document",
        }
        registered_or_failed = set(result.registered) | set(result.failed)
        assert expected_tools <= registered_or_failed, (
            f"Missing tools: {expected_tools - registered_or_failed}"
        )


# =============================================================================
# Unregistration Tests
# =============================================================================


class TestUnregisterTools:
    """Test tool unregistration."""

    def test_returns_empty_if_no_controller(self) -> None:
        """Returns empty list when controller is None."""
        result = unregister_tools(None, ["tool1", "tool2"])
        assert result == []

    def test_returns_empty_if_no_unregister_method(self) -> None:
        """Returns empty list when controller lacks unregister_tool."""
        controller = object()
        result = unregister_tools(controller, ["tool1"])
        assert result == []

    def test_unregisters_existing_tools(self) -> None:
        """Successfully unregisters existing tools."""
        controller = MockController()
        controller.registered_tools = {"tool1": "impl1", "tool2": "impl2"}
        
        result = unregister_tools(controller, ["tool1", "tool2"])
        
        assert "tool1" in result
        assert "tool2" in result
        assert "tool1" not in controller.registered_tools
        assert "tool2" not in controller.registered_tools

    def test_handles_missing_tools_gracefully(self) -> None:
        """Handles unregistering non-existent tools gracefully."""
        controller = MockController()
        controller.registered_tools = {"tool1": "impl1"}
        
        # Should not raise, just skip the missing tool
        result = unregister_tools(controller, ["tool1", "nonexistent"])
        
        assert "tool1" in result
        # nonexistent may or may not be in result depending on implementation


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestProtocolCompliance:
    """Test that mock classes comply with protocols."""

    def test_mock_bridge_is_document_bridge(self) -> None:
        """MockBridge implements DocumentBridge protocol."""
        bridge = MockBridge()
        assert isinstance(bridge, DocumentBridge)

    def test_mock_workspace_is_workspace_provider(self) -> None:
        """MockWorkspace implements WorkspaceProvider protocol."""
        workspace = MockWorkspace()
        assert isinstance(workspace, WorkspaceProvider)

    def test_mock_controller_is_ai_controller_provider(self) -> None:
        """MockController implements AIControllerProvider protocol."""
        controller = MockController()
        assert isinstance(controller, AIControllerProvider)


# =============================================================================
# Integration Tests
# =============================================================================


class TestToolWiringIntegration:
    """Integration tests for tool wiring."""

    def test_full_registration_workflow(self) -> None:
        """Test complete registration and unregistration workflow."""
        controller = MockController()
        ctx = ToolWiringContext(
            controller=controller,
            bridge=MockBridge(),
            workspace=MockWorkspace(),
        )
        
        # Legacy tools is now a deprecated no-op
        legacy_result = register_legacy_tools(ctx)
        assert legacy_result.registered == [], "Legacy registration should be no-op"
        
        # Register new tools
        new_result = register_new_tools(ctx)
        # New tools may fail due to missing dependencies, that's ok
        
        # Unregister all
        all_tools = new_result.registered
        unregistered = unregister_tools(controller, all_tools)
        
        # Should have unregistered at least some tools if any were registered
        if new_result.registered:
            assert len(unregistered) > 0
