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
    AIToolRegistrar,
    ToolWiringContext,
    ToolRegistrationResult,
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


# =============================================================================
# Tool Registration Tests
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

    def test_mock_controller_is_ai_tool_registrar(self) -> None:
        """MockController implements AIToolRegistrar protocol."""
        controller = MockController()
        assert isinstance(controller, AIToolRegistrar)


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
        
        # Register tools
        result = register_new_tools(ctx)
        # New tools may fail due to missing dependencies, that's ok
        
        # Unregister all
        all_tools = result.registered
        unregistered = unregister_tools(controller, all_tools)
        
        # Should have unregistered at least some tools if any were registered
        if result.registered:
            assert len(unregistered) > 0


# =============================================================================
# DocumentCreatorAdapter Integration Tests
# =============================================================================


class TestDocumentCreatorAdapterIntegration:
    """Integration tests for DocumentCreatorAdapter with real DocumentWorkspace.
    
    These tests ensure the adapter correctly interfaces with the actual
    workspace implementation, catching API mismatches like using .tabs
    instead of .iter_tabs().
    """

    @pytest.fixture
    def workspace(self):
        """Create a real DocumentWorkspace for testing."""
        from tinkerbell.editor.workspace import DocumentWorkspace
        from tinkerbell.editor.editor_widget import EditorWidget
        from tinkerbell.services.bridge import DocumentBridge
        
        # Create workspace with mock factories that don't require Qt
        def mock_editor_factory():
            editor = MagicMock(spec=EditorWidget)
            editor.to_document.return_value = MagicMock(
                text="",
                dirty=False,
                metadata=MagicMock(path=None, language="markdown")
            )
            return editor
        
        def mock_bridge_factory(editor):
            return MagicMock(spec=DocumentBridge)
        
        return DocumentWorkspace(
            editor_factory=mock_editor_factory,
            bridge_factory=mock_bridge_factory,
        )

    def test_adapter_from_ui_tools_provider(self, workspace):
        """Test _DocumentCreatorAdapter from ui.tools.provider module."""
        from tinkerbell.ui.tools.provider import _DocumentCreatorAdapter
        
        adapter = _DocumentCreatorAdapter(workspace)
        
        # Test document_exists on empty workspace
        exists, tab_id = adapter.document_exists("NonExistent")
        assert exists is False
        assert tab_id is None

    def test_adapter_document_exists_finds_tab(self, workspace):
        """Test that document_exists correctly finds existing tabs."""
        from tinkerbell.ui.tools.provider import _DocumentCreatorAdapter
        
        # Create a tab in the workspace
        tab = workspace.create_tab(title="My Document")
        
        adapter = _DocumentCreatorAdapter(workspace)
        
        # Should find the document by its actual title (workspace may modify it)
        # The title includes untitled_index suffix for untitled docs
        exists, found_id = adapter.document_exists(tab.title)
        assert exists is True
        assert found_id == tab.id

    def test_adapter_document_exists_no_match(self, workspace):
        """Test that document_exists returns False for non-matching titles."""
        from tinkerbell.ui.tools.provider import _DocumentCreatorAdapter
        
        # Create a tab with a different title
        workspace.create_tab(title="Other Document")
        
        adapter = _DocumentCreatorAdapter(workspace)
        
        # Should not find a different title
        exists, found_id = adapter.document_exists("My Document")
        assert exists is False
        assert found_id is None

    def test_adapter_create_document(self, workspace):
        """Test that create_document creates a real tab."""
        from tinkerbell.ui.tools.provider import _DocumentCreatorAdapter
        
        adapter = _DocumentCreatorAdapter(workspace)
        
        # Create a document
        tab_id = adapter.create_document(
            title="New Doc",
            content="Hello World",
            file_type="markdown",
        )
        
        # Verify the tab was created
        assert tab_id is not None
        assert workspace.tab_count() == 1
        
        # Verify we can find it by the actual tab title
        tab = workspace.get_tab(tab_id)
        exists, found_id = adapter.document_exists(tab.title)
        assert exists is True
        assert found_id == tab_id

    def test_adapter_iterates_multiple_tabs(self, workspace):
        """Test that document_exists correctly iterates over multiple tabs."""
        from tinkerbell.ui.tools.provider import _DocumentCreatorAdapter
        
        # Create multiple tabs
        workspace.create_tab(title="Doc A")
        workspace.create_tab(title="Doc B")
        tab_c = workspace.create_tab(title="Doc C")
        
        adapter = _DocumentCreatorAdapter(workspace)
        
        # Should find the last one by its actual title
        exists, found_id = adapter.document_exists(tab_c.title)
        assert exists is True
        assert found_id == tab_c.id
        
        # Should not find non-existent
        exists, found_id = adapter.document_exists("Completely Different Name")
        assert exists is False


# =============================================================================
# AIClientProviderAdapter Tests
# =============================================================================


class TestAIClientProviderAdapter:
    """Tests for _AIClientProviderAdapter."""

    def test_get_ai_client_returns_client_from_controller(self):
        """Test that get_ai_client returns the client attribute from controller."""
        from tinkerbell.ui.tools.provider import _AIClientProviderAdapter
        
        # Create a mock controller with a client attribute
        mock_client = MagicMock()
        mock_controller = MagicMock()
        mock_controller.client = mock_client
        
        adapter = _AIClientProviderAdapter(lambda: mock_controller)
        
        result = adapter.get_ai_client()
        assert result is mock_client

    def test_get_ai_client_returns_none_when_controller_is_none(self):
        """Test that get_ai_client returns None when controller is None."""
        from tinkerbell.ui.tools.provider import _AIClientProviderAdapter
        
        adapter = _AIClientProviderAdapter(lambda: None)
        
        result = adapter.get_ai_client()
        assert result is None

    def test_get_ai_client_returns_none_when_no_client_attribute(self):
        """Test that get_ai_client returns None when controller has no client."""
        from tinkerbell.ui.tools.provider import _AIClientProviderAdapter
        
        # Create a controller without a client attribute
        mock_controller = MagicMock(spec=[])  # No attributes
        
        adapter = _AIClientProviderAdapter(lambda: mock_controller)
        
        result = adapter.get_ai_client()
        assert result is None

    def test_adapter_is_used_in_tool_wiring_context(self):
        """Test that ToolProvider includes ai_client_provider in context."""
        from tinkerbell.ui.tools.provider import ToolProvider
        from tinkerbell.editor.selection_gateway import SelectionSnapshotProvider
        
        mock_controller = MagicMock()
        mock_controller.client = MagicMock()
        
        provider = ToolProvider(
            controller_resolver=lambda: mock_controller,
            bridge=MockBridge(),
            workspace=MockWorkspace(),
            selection_gateway=MagicMock(spec=SelectionSnapshotProvider),
        )
        
        ctx = provider.build_tool_wiring_context()
        
        # Verify ai_client_provider is set
        assert ctx.ai_client_provider is not None
        
        # Verify it can get the client
        client = ctx.ai_client_provider.get_ai_client()
        assert client is mock_controller.client
