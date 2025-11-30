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


__all__ = ["TestTurnContext", "TestExtractTabIdFromVersion", "TestDispatcherTurnContextIntegration"]
