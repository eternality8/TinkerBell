"""Overlay manager domain service.

Manages diff overlay state tracking across document tabs. This is the
domain layer abstraction for overlay bookkeeping, independent of the
actual UI widget implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from ..events import EventBus

if TYPE_CHECKING:  # pragma: no cover
    from ...documents.ranges import TextRange

LOGGER = logging.getLogger(__name__)


class OverlayManager:
    """Domain manager for diff overlay state tracking.

    Tracks which tabs have active diff overlays and provides methods
    for applying, clearing, and querying overlay state. The actual
    overlay rendering is delegated to the editor widget via callbacks.

    This manager does not own the overlay rendering logic - it tracks
    state and coordinates with the editor through provided callbacks.
    """

    def __init__(
        self,
        show_overlay: Callable[[str, tuple[tuple[int, int], ...], str | None, str | None, str | None], None],
        clear_overlay: Callable[[str | None], None],
        event_bus: EventBus,
    ) -> None:
        """Initialize the overlay manager.

        Args:
            show_overlay: Callback to show an overlay on a tab.
                Signature: (label, spans, summary, source, tab_id) -> None
            clear_overlay: Callback to clear an overlay from a tab.
                Signature: (tab_id) -> None
            event_bus: The event bus for publishing events.
        """
        self._show_overlay = show_overlay
        self._clear_overlay = clear_overlay
        self._bus = event_bus
        self._tabs_with_overlay: set[str] = set()

    # ------------------------------------------------------------------
    # Overlay Lifecycle
    # ------------------------------------------------------------------

    def apply_overlay(
        self,
        tab_id: str,
        label: str,
        spans: tuple[tuple[int, int], ...],
        summary: str | None = None,
        source: str | None = None,
    ) -> bool:
        """Apply a diff overlay to a tab.

        Args:
            tab_id: The tab to apply the overlay to.
            label: The overlay label/diff text.
            spans: Tuple of (start, end) spans to highlight.
            summary: Optional summary text.
            source: Optional source identifier.

        Returns:
            True if overlay was applied successfully.
        """
        if not tab_id:
            LOGGER.warning("OverlayManager.apply_overlay: no tab_id provided")
            return False

        if not spans:
            LOGGER.debug("OverlayManager.apply_overlay: no spans, skipping")
            return False

        try:
            self._show_overlay(label, spans, summary, source, tab_id)
            self._tabs_with_overlay.add(tab_id)
            LOGGER.debug(
                "OverlayManager.apply_overlay: tab_id=%s, spans=%d",
                tab_id,
                len(spans),
            )
            return True
        except Exception:  # pragma: no cover
            LOGGER.debug(
                "OverlayManager.apply_overlay: failed for tab_id=%s",
                tab_id,
                exc_info=True,
            )
            return False

    def clear_overlay(self, tab_id: str | None = None) -> bool:
        """Clear the overlay from a specific tab.

        Args:
            tab_id: The tab to clear, or None for active tab handling
                (passed through to the callback).

        Returns:
            True if overlay was cleared (or tab had no overlay).
        """
        if tab_id and tab_id not in self._tabs_with_overlay:
            return True  # No overlay to clear

        try:
            self._clear_overlay(tab_id)
            if tab_id:
                self._tabs_with_overlay.discard(tab_id)
                LOGGER.debug("OverlayManager.clear_overlay: tab_id=%s", tab_id)
            return True
        except KeyError:  # pragma: no cover - tab already closed
            if tab_id:
                self._tabs_with_overlay.discard(tab_id)
            return True
        except Exception:  # pragma: no cover
            LOGGER.debug(
                "OverlayManager.clear_overlay: failed for tab_id=%s",
                tab_id,
                exc_info=True,
            )
            return False

    def clear_all_overlays(self) -> None:
        """Clear overlays from all tabs."""
        for tab_id in list(self._tabs_with_overlay):
            self.clear_overlay(tab_id)
        self._tabs_with_overlay.clear()
        LOGGER.debug("OverlayManager.clear_all_overlays")

    # ------------------------------------------------------------------
    # Query Methods
    # ------------------------------------------------------------------

    def has_overlay(self, tab_id: str) -> bool:
        """Check if a tab has an active overlay.

        Args:
            tab_id: The tab to check.

        Returns:
            True if the tab has an active overlay.
        """
        return tab_id in self._tabs_with_overlay

    def overlay_tab_ids(self) -> tuple[str, ...]:
        """Get tuple of all tab IDs with active overlays."""
        return tuple(sorted(self._tabs_with_overlay))

    def overlay_count(self) -> int:
        """Get the number of tabs with active overlays."""
        return len(self._tabs_with_overlay)

    # ------------------------------------------------------------------
    # Internal State Management
    # ------------------------------------------------------------------

    def discard_overlay(self, tab_id: str) -> None:
        """Remove a tab from overlay tracking without clearing.

        Used when a tab is closed or overlay is cleared externally.

        Args:
            tab_id: The tab to remove from tracking.
        """
        self._tabs_with_overlay.discard(tab_id)

    def reset(self) -> None:
        """Reset all overlay tracking state.

        Clears the internal tracking set without calling clear callbacks.
        """
        self._tabs_with_overlay.clear()

    # ------------------------------------------------------------------
    # Static Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def coerce_overlay_spans(
        raw_spans: Any,
        *,
        fallback_range: tuple[int, int] | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Coerce raw span data into normalized overlay spans.

        Args:
            raw_spans: Raw span data (list of lists/tuples).
            fallback_range: Optional fallback range if no spans found.

        Returns:
            Tuple of (start, end) spans.
        """
        spans: list[tuple[int, int]] = []

        if isinstance(raw_spans, Sequence) and not isinstance(raw_spans, str):
            for entry in raw_spans:
                if not isinstance(entry, Sequence) or len(entry) != 2:
                    continue
                try:
                    start = int(entry[0])
                    end = int(entry[1])
                except (TypeError, ValueError):
                    continue
                if start == end:
                    continue
                if end < start:
                    start, end = end, start
                spans.append((start, end))

        if not spans and fallback_range is not None:
            start, end = fallback_range
            if start != end:
                if end < start:
                    start, end = end, start
                spans.append((start, end))

        return tuple(spans)

    @staticmethod
    def merge_overlay_spans(
        existing: tuple[tuple[int, int], ...],
        new_spans: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        """Merge two sets of spans into non-overlapping sorted spans.

        Args:
            existing: Existing spans.
            new_spans: New spans to merge.

        Returns:
            Merged tuple of non-overlapping spans.
        """
        if not existing:
            return new_spans
        if not new_spans:
            return existing

        ordered = sorted(existing + new_spans, key=lambda span: span[0])
        merged: list[list[int]] = []

        for start, end in ordered:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        return tuple((start, end) for start, end in merged)


__all__ = ["OverlayManager"]
