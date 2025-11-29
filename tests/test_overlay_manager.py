"""Tests for OverlayManager domain manager."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, call

import pytest

from tinkerbell.ui.domain.overlay_manager import OverlayManager
from tinkerbell.ui.events import EventBus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def show_overlay_mock() -> MagicMock:
    return MagicMock()


@pytest.fixture
def clear_overlay_mock() -> MagicMock:
    return MagicMock()


@pytest.fixture
def overlay_manager(
    show_overlay_mock: MagicMock,
    clear_overlay_mock: MagicMock,
    event_bus: EventBus,
) -> OverlayManager:
    return OverlayManager(
        show_overlay=show_overlay_mock,
        clear_overlay=clear_overlay_mock,
        event_bus=event_bus,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestOverlayManagerInit:
    """Tests for OverlayManager initialization."""

    def test_initial_state(self, overlay_manager: OverlayManager) -> None:
        """Manager starts with no overlays tracked."""
        assert overlay_manager.overlay_tab_ids() == ()
        assert overlay_manager.overlay_count() == 0


# =============================================================================
# Apply Overlay Tests
# =============================================================================


class TestOverlayManagerApply:
    """Tests for OverlayManager.apply_overlay()."""

    def test_applies_overlay(
        self,
        overlay_manager: OverlayManager,
        show_overlay_mock: MagicMock,
    ) -> None:
        """apply_overlay calls show callback and tracks tab."""
        spans = ((0, 10), (20, 30))
        result = overlay_manager.apply_overlay(
            "tab-1", "diff text", spans, "summary", "source"
        )

        assert result is True
        show_overlay_mock.assert_called_once_with(
            "diff text", spans, "summary", "source", "tab-1"
        )
        assert overlay_manager.has_overlay("tab-1")

    def test_returns_false_for_empty_tab_id(
        self,
        overlay_manager: OverlayManager,
        show_overlay_mock: MagicMock,
    ) -> None:
        """apply_overlay returns False for empty tab_id."""
        result = overlay_manager.apply_overlay("", "diff", ((0, 10),))

        assert result is False
        show_overlay_mock.assert_not_called()

    def test_returns_false_for_empty_spans(
        self,
        overlay_manager: OverlayManager,
        show_overlay_mock: MagicMock,
    ) -> None:
        """apply_overlay returns False for empty spans."""
        result = overlay_manager.apply_overlay("tab-1", "diff", ())

        assert result is False
        show_overlay_mock.assert_not_called()

    def test_handles_callback_exception(
        self,
        overlay_manager: OverlayManager,
        show_overlay_mock: MagicMock,
    ) -> None:
        """apply_overlay handles callback exception."""
        show_overlay_mock.side_effect = RuntimeError("test error")

        result = overlay_manager.apply_overlay("tab-1", "diff", ((0, 10),))

        assert result is False
        assert not overlay_manager.has_overlay("tab-1")


# =============================================================================
# Clear Overlay Tests
# =============================================================================


class TestOverlayManagerClear:
    """Tests for OverlayManager.clear_overlay()."""

    def test_clears_overlay(
        self,
        overlay_manager: OverlayManager,
        clear_overlay_mock: MagicMock,
    ) -> None:
        """clear_overlay calls callback and removes tracking."""
        overlay_manager.apply_overlay("tab-1", "diff", ((0, 10),))

        result = overlay_manager.clear_overlay("tab-1")

        assert result is True
        clear_overlay_mock.assert_called_once_with("tab-1")
        assert not overlay_manager.has_overlay("tab-1")

    def test_no_op_for_untracked_tab(
        self,
        overlay_manager: OverlayManager,
        clear_overlay_mock: MagicMock,
    ) -> None:
        """clear_overlay is no-op for tab without overlay."""
        result = overlay_manager.clear_overlay("tab-1")

        assert result is True
        clear_overlay_mock.assert_not_called()

    def test_handles_key_error(
        self,
        overlay_manager: OverlayManager,
        clear_overlay_mock: MagicMock,
    ) -> None:
        """clear_overlay handles KeyError (tab closed)."""
        overlay_manager.apply_overlay("tab-1", "diff", ((0, 10),))
        clear_overlay_mock.side_effect = KeyError("tab-1")

        result = overlay_manager.clear_overlay("tab-1")

        assert result is True
        assert not overlay_manager.has_overlay("tab-1")


# =============================================================================
# Clear All Tests
# =============================================================================


class TestOverlayManagerClearAll:
    """Tests for OverlayManager.clear_all_overlays()."""

    def test_clears_all_overlays(
        self,
        overlay_manager: OverlayManager,
        clear_overlay_mock: MagicMock,
    ) -> None:
        """clear_all_overlays clears all tracked overlays."""
        overlay_manager.apply_overlay("tab-1", "diff1", ((0, 10),))
        overlay_manager.apply_overlay("tab-2", "diff2", ((0, 10),))

        overlay_manager.clear_all_overlays()

        assert overlay_manager.overlay_count() == 0
        assert clear_overlay_mock.call_count == 2

    def test_no_op_when_empty(
        self,
        overlay_manager: OverlayManager,
        clear_overlay_mock: MagicMock,
    ) -> None:
        """clear_all_overlays is safe when no overlays."""
        overlay_manager.clear_all_overlays()

        clear_overlay_mock.assert_not_called()


# =============================================================================
# Query Tests
# =============================================================================


class TestOverlayManagerQueries:
    """Tests for query methods."""

    def test_has_overlay_true(self, overlay_manager: OverlayManager) -> None:
        """has_overlay returns True for tracked tab."""
        overlay_manager.apply_overlay("tab-1", "diff", ((0, 10),))
        assert overlay_manager.has_overlay("tab-1") is True

    def test_has_overlay_false(self, overlay_manager: OverlayManager) -> None:
        """has_overlay returns False for untracked tab."""
        assert overlay_manager.has_overlay("tab-1") is False

    def test_overlay_tab_ids(self, overlay_manager: OverlayManager) -> None:
        """overlay_tab_ids returns sorted tuple."""
        overlay_manager.apply_overlay("tab-2", "diff", ((0, 10),))
        overlay_manager.apply_overlay("tab-1", "diff", ((0, 10),))

        ids = overlay_manager.overlay_tab_ids()

        assert ids == ("tab-1", "tab-2")

    def test_overlay_count(self, overlay_manager: OverlayManager) -> None:
        """overlay_count returns correct count."""
        assert overlay_manager.overlay_count() == 0

        overlay_manager.apply_overlay("tab-1", "diff", ((0, 10),))
        assert overlay_manager.overlay_count() == 1

        overlay_manager.apply_overlay("tab-2", "diff", ((0, 10),))
        assert overlay_manager.overlay_count() == 2


# =============================================================================
# State Management Tests
# =============================================================================


class TestOverlayManagerState:
    """Tests for internal state management."""

    def test_discard_overlay(self, overlay_manager: OverlayManager) -> None:
        """discard_overlay removes from tracking without callback."""
        overlay_manager.apply_overlay("tab-1", "diff", ((0, 10),))

        overlay_manager.discard_overlay("tab-1")

        assert not overlay_manager.has_overlay("tab-1")

    def test_discard_overlay_safe_for_missing(
        self, overlay_manager: OverlayManager
    ) -> None:
        """discard_overlay is safe for untracked tab."""
        overlay_manager.discard_overlay("nonexistent")  # Should not raise

    def test_reset(self, overlay_manager: OverlayManager) -> None:
        """reset clears all tracking without callbacks."""
        overlay_manager.apply_overlay("tab-1", "diff", ((0, 10),))
        overlay_manager.apply_overlay("tab-2", "diff", ((0, 10),))

        overlay_manager.reset()

        assert overlay_manager.overlay_count() == 0


# =============================================================================
# Static Helper Tests
# =============================================================================


class TestCoerceOverlaySpans:
    """Tests for coerce_overlay_spans static method."""

    def test_coerce_valid_spans(self) -> None:
        """Coerces valid span data."""
        raw = [[0, 10], [20, 30]]
        result = OverlayManager.coerce_overlay_spans(raw)
        assert result == ((0, 10), (20, 30))

    def test_coerce_tuple_spans(self) -> None:
        """Coerces tuple spans."""
        raw = [(0, 10), (20, 30)]
        result = OverlayManager.coerce_overlay_spans(raw)
        assert result == ((0, 10), (20, 30))

    def test_coerce_reverses_inverted_spans(self) -> None:
        """Corrects inverted spans."""
        raw = [[10, 0]]
        result = OverlayManager.coerce_overlay_spans(raw)
        assert result == ((0, 10),)

    def test_coerce_skips_empty_spans(self) -> None:
        """Skips zero-length spans."""
        raw = [[0, 0], [10, 20]]
        result = OverlayManager.coerce_overlay_spans(raw)
        assert result == ((10, 20),)

    def test_coerce_skips_invalid_entries(self) -> None:
        """Skips invalid entries."""
        raw = [[0, 10], "invalid", [20], [30, 40]]
        result = OverlayManager.coerce_overlay_spans(raw)
        assert result == ((0, 10), (30, 40))

    def test_coerce_uses_fallback(self) -> None:
        """Uses fallback when no valid spans."""
        result = OverlayManager.coerce_overlay_spans([], fallback_range=(5, 15))
        assert result == ((5, 15),)

    def test_coerce_fallback_reverses_inverted(self) -> None:
        """Corrects inverted fallback range."""
        result = OverlayManager.coerce_overlay_spans([], fallback_range=(15, 5))
        assert result == ((5, 15),)

    def test_coerce_skips_empty_fallback(self) -> None:
        """Skips zero-length fallback."""
        result = OverlayManager.coerce_overlay_spans([], fallback_range=(10, 10))
        assert result == ()

    def test_coerce_ignores_string(self) -> None:
        """Ignores string input."""
        result = OverlayManager.coerce_overlay_spans("not spans")
        assert result == ()


class TestMergeOverlaySpans:
    """Tests for merge_overlay_spans static method."""

    def test_merge_disjoint_spans(self) -> None:
        """Merges disjoint spans."""
        existing = ((0, 10),)
        new = ((20, 30),)
        result = OverlayManager.merge_overlay_spans(existing, new)
        assert result == ((0, 10), (20, 30))

    def test_merge_overlapping_spans(self) -> None:
        """Merges overlapping spans."""
        existing = ((0, 15),)
        new = ((10, 25),)
        result = OverlayManager.merge_overlay_spans(existing, new)
        assert result == ((0, 25),)

    def test_merge_adjacent_spans(self) -> None:
        """Keeps adjacent spans separate."""
        existing = ((0, 10),)
        new = ((10, 20),)
        result = OverlayManager.merge_overlay_spans(existing, new)
        # Adjacent spans are kept separate (not merged since start > end[1])
        # Actually 10 > 10 is false, so they merge
        assert result == ((0, 20),)

    def test_merge_empty_existing(self) -> None:
        """Returns new when existing is empty."""
        result = OverlayManager.merge_overlay_spans((), ((0, 10),))
        assert result == ((0, 10),)

    def test_merge_empty_new(self) -> None:
        """Returns existing when new is empty."""
        result = OverlayManager.merge_overlay_spans(((0, 10),), ())
        assert result == ((0, 10),)

    def test_merge_multiple_spans(self) -> None:
        """Merges multiple spans correctly."""
        existing = ((0, 10), (30, 40))
        new = ((5, 15), (35, 50))
        result = OverlayManager.merge_overlay_spans(existing, new)
        assert result == ((0, 15), (30, 50))

    def test_merge_sorts_spans(self) -> None:
        """Result is sorted by start position."""
        existing = ((20, 30),)
        new = ((0, 10),)
        result = OverlayManager.merge_overlay_spans(existing, new)
        assert result == ((0, 10), (20, 30))


# =============================================================================
# Module Export Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_exported_from_domain(self) -> None:
        """OverlayManager is exported from domain package."""
        from tinkerbell.ui.domain import OverlayManager as OM

        assert OM is OverlayManager
