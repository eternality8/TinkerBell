"""Tests for WS7.1: Legacy Tool Deprecation and Migration.

This module tests:
- Deprecation warnings for legacy tools
- Migration adapter for new tools
- Legacy tool replacement mapping
"""

from __future__ import annotations

import warnings
import pytest
from unittest.mock import MagicMock, patch

from tinkerbell.ai.tools.deprecation import (
    LEGACY_TOOL_REPLACEMENTS,
    DeprecatedToolWarning,
    deprecated_tool,
    emit_deprecation_warning,
    get_replacement_tool,
)
from tinkerbell.ai.tools.registry_adapter import (
    NewToolRegistryContext,
    ToolRegistrationFailure,
    ToolRegistrationError,
    register_new_tools,
    get_new_tool_schemas,
)


# =============================================================================
# Deprecation Tests
# =============================================================================


class TestLegacyToolReplacements:
    """Test the legacy tool replacement mapping."""

    def test_replacement_mapping_exists(self) -> None:
        """Verify replacement mapping is populated."""
        assert len(LEGACY_TOOL_REPLACEMENTS) > 0
        
    def test_key_legacy_tools_have_replacements(self) -> None:
        """Verify key legacy tools have replacements defined."""
        key_tools = [
            "document_snapshot",
            "document_edit",
            "document_apply_patch",
            "document_find_text",
            "document_outline",
        ]
        for tool in key_tools:
            assert tool in LEGACY_TOOL_REPLACEMENTS, f"Missing replacement for {tool}"
            assert LEGACY_TOOL_REPLACEMENTS[tool], f"Empty replacement for {tool}"

    def test_get_replacement_tool_found(self) -> None:
        """Get replacement returns correct value for known tools."""
        result = get_replacement_tool("document_snapshot")
        assert result == "read_document"
        
    def test_get_replacement_tool_not_found(self) -> None:
        """Get replacement returns None for unknown tools."""
        result = get_replacement_tool("unknown_tool")
        assert result is None


class TestDeprecatedToolWarning:
    """Test the deprecation warning system."""

    def test_warning_is_deprecation_warning(self) -> None:
        """DeprecatedToolWarning inherits from DeprecationWarning."""
        assert issubclass(DeprecatedToolWarning, DeprecationWarning)

    def test_emit_deprecation_warning_simple(self) -> None:
        """emit_deprecation_warning emits correct warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_deprecation_warning("test_tool")
            
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecatedToolWarning)
            assert "test_tool" in str(w[0].message)
            assert "deprecated" in str(w[0].message).lower()

    def test_emit_deprecation_warning_with_replacement(self) -> None:
        """emit_deprecation_warning includes replacement in message."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_deprecation_warning("old_tool", replacement="new_tool")
            
            assert len(w) == 1
            assert "new_tool" in str(w[0].message)

    def test_emit_deprecation_warning_with_version(self) -> None:
        """emit_deprecation_warning includes removal version."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_deprecation_warning("test_tool", removal_version="3.0.0")
            
            assert len(w) == 1
            assert "3.0.0" in str(w[0].message)


class TestDeprecatedToolDecorator:
    """Test the @deprecated_tool decorator."""

    def test_decorator_emits_warning(self) -> None:
        """Decorated function emits warning when called."""
        @deprecated_tool()
        def my_old_function() -> str:
            return "result"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = my_old_function()
            
            assert result == "result"
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecatedToolWarning)

    def test_decorator_with_replacement(self) -> None:
        """Decorated function mentions replacement."""
        @deprecated_tool(replacement="new_function")
        def old_function() -> None:
            pass
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            old_function()
            
            assert "new_function" in str(w[0].message)

    def test_decorator_preserves_function_metadata(self) -> None:
        """Decorator preserves function name and docstring."""
        @deprecated_tool()
        def documented_function() -> None:
            """This is the docstring."""
            pass
        
        assert documented_function.__name__ == "documented_function"
        assert "docstring" in (documented_function.__doc__ or "")


# =============================================================================
# Registry Adapter Tests
# =============================================================================


class TestNewToolRegistryContext:
    """Test the NewToolRegistryContext dataclass."""

    def test_context_creation(self) -> None:
        """Context can be created with required fields."""
        controller = MagicMock()
        context = NewToolRegistryContext(controller=controller)
        
        assert context.controller is controller
        assert context.version_manager is None
        assert context.enable_read_tools is True
        assert context.enable_write_tools is True
        assert context.enable_analysis_tools is False
        assert context.enable_transform_tools is False

    def test_context_with_all_flags_disabled(self) -> None:
        """Context can disable all feature flags."""
        context = NewToolRegistryContext(
            controller=MagicMock(),
            enable_read_tools=False,
            enable_write_tools=False,
            enable_analysis_tools=False,
            enable_transform_tools=False,
        )
        
        assert not context.enable_read_tools
        assert not context.enable_write_tools


class TestToolRegistrationFailure:
    """Test ToolRegistrationFailure dataclass."""

    def test_failure_creation(self) -> None:
        """Failure can be created with name and error."""
        error = ValueError("test error")
        failure = ToolRegistrationFailure(name="test_tool", error=error)
        
        assert failure.name == "test_tool"
        assert failure.error is error


class TestToolRegistrationError:
    """Test ToolRegistrationError exception."""

    def test_error_message_includes_tool_names(self) -> None:
        """Error message lists failed tool names."""
        failures = [
            ToolRegistrationFailure(name="tool1", error=ValueError("err1")),
            ToolRegistrationFailure(name="tool2", error=ValueError("err2")),
        ]
        error = ToolRegistrationError(failures)
        
        assert "tool1" in str(error)
        assert "tool2" in str(error)
        assert len(error.failures) == 2

    def test_error_with_empty_failures(self) -> None:
        """Error handles empty failure list."""
        error = ToolRegistrationError([])
        assert "unknown" in str(error)


class TestRegisterNewTools:
    """Test the register_new_tools function."""

    def test_returns_empty_list_if_no_controller(self) -> None:
        """Returns empty list when controller is None."""
        context = NewToolRegistryContext(controller=None)
        result = register_new_tools(context)
        assert result == []

    def test_returns_empty_list_if_no_register_tool(self) -> None:
        """Returns empty list when controller lacks register_tool."""
        controller = MagicMock(spec=[])  # No register_tool attribute
        context = NewToolRegistryContext(controller=controller)
        result = register_new_tools(context)
        assert result == []

    def test_registers_read_tools_when_enabled(self) -> None:
        """Registers reading tools when enable_read_tools is True."""
        controller = MagicMock()
        register_fn = MagicMock()
        controller.register_tool = register_fn
        
        context = NewToolRegistryContext(
            controller=controller,
            enable_read_tools=True,
        )
        
        result = register_new_tools(context)
        
        # Should attempt to register read_document, search_document, get_outline
        assert register_fn.call_count >= 1
        registered_names = [call[0][0] for call in register_fn.call_args_list]
        
        # At least one of the read tools should be registered
        read_tools = {"read_document", "search_document", "get_outline"}
        assert any(name in read_tools for name in registered_names)

    def test_skips_read_tools_when_disabled(self) -> None:
        """Skips reading tools when enable_read_tools is False."""
        controller = MagicMock()
        register_fn = MagicMock()
        controller.register_tool = register_fn
        
        context = NewToolRegistryContext(
            controller=controller,
            enable_read_tools=False,
            enable_write_tools=False,
        )
        
        result = register_new_tools(context)
        
        # No read tools should be registered
        for call in register_fn.call_args_list:
            name = call[0][0]
            assert name not in {"read_document", "search_document", "get_outline"}

    def test_uses_custom_register_fn(self) -> None:
        """Uses custom register_fn when provided."""
        controller = MagicMock()
        custom_register = MagicMock()
        
        context = NewToolRegistryContext(
            controller=controller,
            enable_read_tools=True,
        )
        
        result = register_new_tools(context, register_fn=custom_register)
        
        # Custom function should be called instead of controller.register_tool
        assert custom_register.call_count >= 1
        assert not controller.register_tool.called


class TestGetNewToolSchemas:
    """Test the get_new_tool_schemas function."""

    def test_returns_dict_of_schemas(self) -> None:
        """Returns a dictionary of tool schemas."""
        schemas = get_new_tool_schemas()
        
        assert isinstance(schemas, dict)
        assert len(schemas) > 0

    def test_contains_expected_tools(self) -> None:
        """Contains schemas for expected tools."""
        schemas = get_new_tool_schemas()
        
        expected_tools = [
            "list_tabs",
            "read_document",
            "search_document",
            "get_outline",
            "create_document",
            "insert_lines",
            "replace_lines",
            "delete_lines",
        ]
        
        for tool in expected_tools:
            assert tool in schemas, f"Missing schema for {tool}"

    def test_schemas_have_required_fields(self) -> None:
        """Each schema has name, description, and parameters."""
        schemas = get_new_tool_schemas()
        
        for name, schema in schemas.items():
            assert hasattr(schema, "name"), f"{name} missing name"
            assert hasattr(schema, "description"), f"{name} missing description"
            assert hasattr(schema, "parameters"), f"{name} missing parameters"
            assert schema.description, f"{name} has empty description"


# =============================================================================
# Integration Tests
# =============================================================================


class TestDeprecationAndMigrationIntegration:
    """Integration tests for deprecation + migration flow."""

    def test_replacement_lookup_and_warning(self) -> None:
        """Can look up replacement and emit appropriate warning."""
        legacy_tool = "document_snapshot"
        replacement = get_replacement_tool(legacy_tool)
        
        assert replacement is not None
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            emit_deprecation_warning(legacy_tool, replacement=replacement)
            
            assert len(w) == 1
            assert replacement in str(w[0].message)

    def test_new_schemas_cover_replacements(self) -> None:
        """New tool schemas exist for most replacement tools."""
        schemas = get_new_tool_schemas()
        
        # Check that at least some replacements have schemas
        for legacy, replacement in LEGACY_TOOL_REPLACEMENTS.items():
            # Skip complex replacements (e.g., "replace_lines / insert_lines")
            if "/" not in replacement and "(" not in replacement:
                assert replacement in schemas, (
                    f"Replacement '{replacement}' for '{legacy}' has no schema"
                )
