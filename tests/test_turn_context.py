"""Tests for TurnContext and dispatcher integration."""

from __future__ import annotations

from typing import Any, cast

import pytest
from datetime import datetime, timezone

from tinkerbell.ai.orchestration.turn_context import TurnContext, _extract_tab_id_from_version


class TestTurnContext:
    """Tests for TurnContext creation and behavior."""

    def test_default_creation(self) -> None:
        """TurnContext can be created with defaults."""
        ctx = TurnContext()
        
        assert ctx.turn_id  # UUID generated
        assert ctx.pinned_tab_id is None
        assert ctx.document_id is None
        assert isinstance(ctx.started_at, datetime)
        assert ctx.metadata == {}

    def test_from_snapshot_extracts_tab_id(self) -> None:
        """from_snapshot extracts tab_id from snapshot."""
        snapshot = {
            "tab_id": "t1",
            "document_id": "doc-123",
            "version": "t1:abcd:5",
        }
        
        ctx = TurnContext.from_snapshot(snapshot)
        
        assert ctx.pinned_tab_id == "t1"
        assert ctx.document_id == "doc-123"
        assert ctx.metadata.get("snapshot_version") == "t1:abcd:5"

    def test_from_snapshot_uses_version_fallback(self) -> None:
        """from_snapshot extracts tab_id from version when tab_id not present."""
        snapshot = {
            "version": "t2:efgh:10",
            "document_id": "doc-456",
        }
        
        ctx = TurnContext.from_snapshot(snapshot)
        
        assert ctx.pinned_tab_id == "t2"
        assert ctx.document_id == "doc-456"

    def test_from_snapshot_handles_empty(self) -> None:
        """from_snapshot handles empty/None snapshot gracefully."""
        ctx1 = TurnContext.from_snapshot(None)
        ctx2 = TurnContext.from_snapshot({})
        
        assert ctx1.pinned_tab_id is None
        assert ctx2.pinned_tab_id is None

    def test_from_snapshot_explicit_turn_id(self) -> None:
        """from_snapshot accepts explicit turn_id."""
        ctx = TurnContext.from_snapshot({"tab_id": "t1"}, turn_id="custom-turn-id")
        
        assert ctx.turn_id == "custom-turn-id"

    def test_with_tab_id_returns_new_instance(self) -> None:
        """with_tab_id returns a new TurnContext with updated tab_id."""
        original = TurnContext(
            turn_id="turn-1",
            pinned_tab_id="t1",
            document_id="doc-1",
            metadata={"key": "value"},
        )
        
        modified = original.with_tab_id("t2")
        
        # Modified has new tab_id
        assert modified.pinned_tab_id == "t2"
        # But preserves other fields
        assert modified.turn_id == "turn-1"
        assert modified.document_id == "doc-1"
        assert modified.metadata == {"key": "value"}
        # Original unchanged
        assert original.pinned_tab_id == "t1"

    def test_to_dict_serialization(self) -> None:
        """to_dict returns serializable representation."""
        ctx = TurnContext(
            turn_id="turn-123",
            pinned_tab_id="t1",
            document_id="doc-abc",
            metadata={"foo": "bar"},
        )
        
        data = ctx.to_dict()
        
        assert data["turn_id"] == "turn-123"
        assert data["pinned_tab_id"] == "t1"
        assert data["document_id"] == "doc-abc"
        assert data["metadata"] == {"foo": "bar"}
        assert "started_at" in data  # ISO format string

    def test_pin_tab_if_empty_pins_when_none(self) -> None:
        """pin_tab_if_empty pins tab when pinned_tab_id is None."""
        ctx = TurnContext(pinned_tab_id=None)
        
        result = ctx.pin_tab_if_empty("new-tab")
        
        assert result is True
        assert ctx.pinned_tab_id == "new-tab"

    def test_pin_tab_if_empty_does_not_override(self) -> None:
        """pin_tab_if_empty does not override existing pinned_tab_id."""
        ctx = TurnContext(pinned_tab_id="existing-tab")
        
        result = ctx.pin_tab_if_empty("new-tab")
        
        assert result is False
        assert ctx.pinned_tab_id == "existing-tab"

    def test_pin_tab_if_empty_only_pins_once(self) -> None:
        """pin_tab_if_empty only pins the first time."""
        ctx = TurnContext(pinned_tab_id=None)
        
        # First call should pin
        result1 = ctx.pin_tab_if_empty("first-tab")
        assert result1 is True
        assert ctx.pinned_tab_id == "first-tab"
        
        # Second call should not override
        result2 = ctx.pin_tab_if_empty("second-tab")
        assert result2 is False
        assert ctx.pinned_tab_id == "first-tab"


class TestExtractTabIdFromVersion:
    """Tests for _extract_tab_id_from_version helper."""

    def test_extracts_from_valid_token(self) -> None:
        """Extracts tab_id from valid version token."""
        assert _extract_tab_id_from_version("t1:abcd:5") == "t1"
        assert _extract_tab_id_from_version("tab-123:hash:1") == "tab-123"

    def test_handles_none(self) -> None:
        """Returns None for None input."""
        assert _extract_tab_id_from_version(None) is None

    def test_handles_empty_string(self) -> None:
        """Returns None for empty string."""
        assert _extract_tab_id_from_version("") is None

    def test_handles_non_string(self) -> None:
        """Returns None for non-string input."""
        assert _extract_tab_id_from_version(cast(Any, 123)) is None

    def test_handles_no_colon(self) -> None:
        """Returns the whole string if no colon."""
        assert _extract_tab_id_from_version("just-tab-id") == "just-tab-id"


class TestDispatcherTurnContextIntegration:
    """Integration tests for TurnContext with ToolDispatcher."""

    @pytest.fixture
    def mock_context_provider(self):
        """Create a mock context provider."""
        class MockProvider:
            def __init__(self):
                self.active_tab_id = "active-tab"
            
            def get_active_tab_id(self) -> str | None:
                return self.active_tab_id
            
            def get_document_content(self, tab_id: str) -> str | None:
                return f"content for {tab_id}"
            
            def set_document_content(self, tab_id: str, content: str) -> None:
                pass
            
            def get_version_token(self, tab_id: str) -> str | None:
                return f"{tab_id}:hash:1"
            
            def get_document_text(self, tab_id: str | None = None) -> str:
                return f"text for {tab_id}"
        
        return MockProvider()

    @pytest.fixture
    def dispatcher(self, mock_context_provider):
        """Create a dispatcher with mock provider."""
        from tinkerbell.ai.orchestration.tool_dispatcher import ToolDispatcher
        from tinkerbell.ai.tools.tool_registry import ToolRegistry
        
        return ToolDispatcher(
            registry=ToolRegistry(),
            context_provider=mock_context_provider,
        )

    def test_set_and_clear_turn_context(self, dispatcher) -> None:
        """Dispatcher can set and clear turn context."""
        ctx = TurnContext(pinned_tab_id="t1")
        
        assert dispatcher.turn_context is None
        
        dispatcher.set_turn_context(ctx)
        assert dispatcher.turn_context is ctx
        assert dispatcher.turn_context.pinned_tab_id == "t1"
        
        dispatcher.clear_turn_context()
        assert dispatcher.turn_context is None

    def test_build_context_uses_pinned_tab_id(self, dispatcher) -> None:
        """_build_context uses pinned tab_id from turn context."""
        turn_ctx = TurnContext(pinned_tab_id="pinned-tab")
        dispatcher.set_turn_context(turn_ctx)
        
        # Build context without explicit tab_id in arguments
        tool_ctx = dispatcher._build_context({})
        
        assert tool_ctx is not None
        assert tool_ctx.tab_id == "pinned-tab"

    def test_explicit_tab_id_overrides_pinned(self, dispatcher) -> None:
        """Explicit tab_id in arguments overrides pinned tab_id."""
        turn_ctx = TurnContext(pinned_tab_id="pinned-tab")
        dispatcher.set_turn_context(turn_ctx)
        
        # Build context with explicit tab_id
        tool_ctx = dispatcher._build_context({"tab_id": "explicit-tab"})
        
        assert tool_ctx is not None
        assert tool_ctx.tab_id == "explicit-tab"

    def test_falls_back_to_active_when_no_pinned(self, dispatcher, mock_context_provider) -> None:
        """Falls back to active tab when turn context has no pinned tab_id."""
        turn_ctx = TurnContext(pinned_tab_id=None)  # No pinned tab
        dispatcher.set_turn_context(turn_ctx)
        mock_context_provider.active_tab_id = "current-active"
        
        tool_ctx = dispatcher._build_context({})
        
        assert tool_ctx is not None
        assert tool_ctx.tab_id == "current-active"

    def test_no_turn_context_uses_active_tab(self, dispatcher, mock_context_provider) -> None:
        """Without turn context, uses active tab from provider."""
        mock_context_provider.active_tab_id = "provider-active"
        
        tool_ctx = dispatcher._build_context({})
        
        assert tool_ctx is not None
        assert tool_ctx.tab_id == "provider-active"

    def test_simulates_tab_switch_scenario(self, dispatcher, mock_context_provider) -> None:
        """Simulates user switching tabs during AI turn."""
        # At turn start: pin the active tab
        mock_context_provider.active_tab_id = "original-tab"
        turn_ctx = TurnContext.from_snapshot({"tab_id": "original-tab"})
        dispatcher.set_turn_context(turn_ctx)
        
        # User switches tab during turn
        mock_context_provider.active_tab_id = "user-switched-to-this"
        
        # Tool should still use pinned tab, not current active
        tool_ctx = dispatcher._build_context({})
        
        assert tool_ctx is not None
        assert tool_ctx.tab_id == "original-tab"  # Not "user-switched-to-this"


class TestMaybePinCreatedTab:
    """Tests for _maybe_pin_created_tab functionality."""

    @pytest.fixture
    def mock_context_provider(self):
        """Create a mock context provider."""
        class MockProvider:
            def __init__(self):
                self.active_tab_id = None
            
            def get_active_tab_id(self) -> str | None:
                return self.active_tab_id
            
            def get_document_text(self, tab_id: str | None = None) -> str:
                return ""
        
        return MockProvider()

    @pytest.fixture
    def dispatcher(self, mock_context_provider):
        """Create a dispatcher with mock provider."""
        from tinkerbell.ai.orchestration.tool_dispatcher import ToolDispatcher, DispatchResult
        from tinkerbell.ai.tools.tool_registry import ToolRegistry
        
        return ToolDispatcher(
            registry=ToolRegistry(),
            context_provider=mock_context_provider,
        )

    def test_pins_created_tab_when_no_pinned_tab(self, dispatcher) -> None:
        """_maybe_pin_created_tab pins newly created tab when turn has no pinned tab."""
        from tinkerbell.ai.orchestration.tool_dispatcher import DispatchResult
        
        # Set up turn context with no pinned tab (started with no documents)
        turn_ctx = TurnContext(pinned_tab_id=None)
        dispatcher.set_turn_context(turn_ctx)
        
        # Simulate successful create_document result
        result = DispatchResult(
            success=True,
            result={"tab_id": "new-tab-123", "title": "New Doc"},
            tool_name="create_document",
            execution_time_ms=10.0,
        )
        
        dispatcher._maybe_pin_created_tab("create_document", result)
        
        # The turn context should now have the new tab pinned
        assert dispatcher.turn_context.pinned_tab_id == "new-tab-123"

    def test_does_not_pin_when_tab_already_pinned(self, dispatcher) -> None:
        """_maybe_pin_created_tab does not override existing pinned tab."""
        from tinkerbell.ai.orchestration.tool_dispatcher import DispatchResult
        
        # Set up turn context with existing pinned tab
        turn_ctx = TurnContext(pinned_tab_id="existing-tab")
        dispatcher.set_turn_context(turn_ctx)
        
        # Simulate successful create_document result
        result = DispatchResult(
            success=True,
            result={"tab_id": "new-tab-123", "title": "New Doc"},
            tool_name="create_document",
            execution_time_ms=10.0,
        )
        
        dispatcher._maybe_pin_created_tab("create_document", result)
        
        # The original pinned tab should remain
        assert dispatcher.turn_context.pinned_tab_id == "existing-tab"

    def test_does_not_pin_for_other_tools(self, dispatcher) -> None:
        """_maybe_pin_created_tab only affects create_document tool."""
        from tinkerbell.ai.orchestration.tool_dispatcher import DispatchResult
        
        # Set up turn context with no pinned tab
        turn_ctx = TurnContext(pinned_tab_id=None)
        dispatcher.set_turn_context(turn_ctx)
        
        # Simulate some other tool returning a tab_id
        result = DispatchResult(
            success=True,
            result={"tab_id": "some-tab"},
            tool_name="some_other_tool",
            execution_time_ms=10.0,
        )
        
        dispatcher._maybe_pin_created_tab("some_other_tool", result)
        
        # Pinned tab should still be None
        assert dispatcher.turn_context.pinned_tab_id is None

    def test_does_not_pin_on_failure(self, dispatcher) -> None:
        """_maybe_pin_created_tab does not pin on failed create_document."""
        from tinkerbell.ai.orchestration.tool_dispatcher import DispatchResult
        
        # Set up turn context with no pinned tab
        turn_ctx = TurnContext(pinned_tab_id=None)
        dispatcher.set_turn_context(turn_ctx)
        
        # Simulate failed create_document
        result = DispatchResult(
            success=False,
            result=None,
            tool_name="create_document",
            execution_time_ms=10.0,
        )
        
        dispatcher._maybe_pin_created_tab("create_document", result)
        
        # Pinned tab should still be None
        assert dispatcher.turn_context.pinned_tab_id is None

    def test_does_not_pin_when_no_turn_context(self, dispatcher) -> None:
        """_maybe_pin_created_tab does nothing when no turn context."""
        from tinkerbell.ai.orchestration.tool_dispatcher import DispatchResult
        
        # No turn context set
        assert dispatcher.turn_context is None
        
        # Simulate successful create_document
        result = DispatchResult(
            success=True,
            result={"tab_id": "new-tab"},
            tool_name="create_document",
            execution_time_ms=10.0,
        )
        
        # Should not raise
        dispatcher._maybe_pin_created_tab("create_document", result)
        
        # Still no turn context
        assert dispatcher.turn_context is None

    def test_does_not_pin_when_result_missing_tab_id(self, dispatcher) -> None:
        """_maybe_pin_created_tab handles missing tab_id in result."""
        from tinkerbell.ai.orchestration.tool_dispatcher import DispatchResult
        
        # Set up turn context with no pinned tab
        turn_ctx = TurnContext(pinned_tab_id=None)
        dispatcher.set_turn_context(turn_ctx)
        
        # Simulate result without tab_id
        result = DispatchResult(
            success=True,
            result={"title": "New Doc"},  # No tab_id
            tool_name="create_document",
            execution_time_ms=10.0,
        )
        
        dispatcher._maybe_pin_created_tab("create_document", result)
        
        # Pinned tab should still be None
        assert dispatcher.turn_context.pinned_tab_id is None


__all__ = ["TestTurnContext", "TestExtractTabIdFromVersion", "TestDispatcherTurnContextIntegration", "TestMaybePinCreatedTab"]
