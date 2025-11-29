"""Review manager domain service.

Manages pending AI edit reviews, tracking the state of edits awaiting
user acceptance or rejection. This is the single source of truth for
review state.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any

from ..events import (
    EventBus,
    ReviewAccepted,
    ReviewRejected,
    ReviewStateChanged,
)
from ..models.review_models import PendingEdit, PendingReview, ReviewSession

if TYPE_CHECKING:  # pragma: no cover
    from ...editor.document_model import DocumentState
    from ...editor.editor_widget import DiffOverlayState

LOGGER = logging.getLogger(__name__)


class ReviewManager:
    """Domain manager for AI edit review workflow.

    Manages the lifecycle of pending reviews, from beginning a new review
    to accepting or rejecting edits. All state transitions emit events
    through the event bus.

    Events Emitted:
        - ReviewStateChanged: When review state changes (ready, finalized, etc.)
        - ReviewAccepted: When user accepts pending edits
        - ReviewRejected: When user rejects pending edits
    """

    def __init__(self, event_bus: EventBus) -> None:
        """Initialize the review manager.

        Args:
            event_bus: The event bus for publishing events.
        """
        self._bus = event_bus
        self._pending: PendingReview | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def pending_review(self) -> PendingReview | None:
        """Get the current pending review, if any."""
        return self._pending

    def has_pending_review(self) -> bool:
        """Check if there is a pending review."""
        return self._pending is not None

    def is_ready_for_review(self) -> bool:
        """Check if the pending review is ready for user decision."""
        return self._pending is not None and self._pending.ready

    # ------------------------------------------------------------------
    # Review Lifecycle
    # ------------------------------------------------------------------

    def begin_review(
        self,
        turn_id: str,
        prompt: str,
        metadata: dict[str, Any] | None = None,
    ) -> PendingReview:
        """Begin a new pending review for an AI turn.

        If there is an existing pending review, it will be dropped
        (auto-accepted implicitly).

        Args:
            turn_id: The unique identifier for this AI turn.
            prompt: The user prompt that initiated the turn.
            metadata: Optional metadata about the turn.

        Returns:
            The newly created PendingReview.

        Emits:
            ReviewStateChanged: To indicate a new review has started.
        """
        # Drop any existing review
        if self._pending is not None:
            LOGGER.debug(
                "ReviewManager: dropping existing review %s for new review %s",
                self._pending.turn_id,
                turn_id,
            )
            self._pending = None

        # Create new review
        self._pending = PendingReview(
            turn_id=turn_id,
            prompt=prompt,
        )

        LOGGER.debug("ReviewManager.begin_review: turn_id=%s", turn_id)

        self._emit_state_changed()
        return self._pending

    def record_edit(
        self,
        tab_id: str,
        document_snapshot: DocumentState,
        edit: PendingEdit,
        previous_overlay: DiffOverlayState | None = None,
    ) -> None:
        """Record an edit applied during the current turn.

        Ensures a session exists for the tab and adds the edit to it.

        Args:
            tab_id: The tab where the edit was applied.
            document_snapshot: Snapshot of document before the edit.
            edit: The edit that was applied.
            previous_overlay: The overlay state before this edit.
        """
        if self._pending is None:
            LOGGER.warning("ReviewManager.record_edit: no pending review")
            return

        session = self.ensure_session(tab_id, document_snapshot, previous_overlay)
        if session is not None:
            session.edits.append(edit)
            self._pending.total_edit_count += 1

            LOGGER.debug(
                "ReviewManager.record_edit: tab_id=%s, edit_id=%s, total=%d",
                tab_id,
                edit.edit_id,
                self._pending.total_edit_count,
            )

    def ensure_session(
        self,
        tab_id: str,
        document_snapshot: DocumentState,
        previous_overlay: DiffOverlayState | None = None,
    ) -> ReviewSession | None:
        """Ensure a review session exists for the given tab.

        Creates a new session if one doesn't exist, using the provided
        document snapshot as the baseline.

        Args:
            tab_id: The tab identifier.
            document_snapshot: The document state snapshot.
            previous_overlay: The overlay state before this session.

        Returns:
            The existing or newly created ReviewSession, or None if
            no pending review exists.
        """
        if self._pending is None:
            return None

        existing = self._pending.get_session(tab_id)
        if existing is not None:
            return existing

        # Create new session with snapshot copy
        snapshot_copy = deepcopy(document_snapshot)
        overlay_copy = deepcopy(previous_overlay) if previous_overlay else None

        session = ReviewSession(
            tab_id=tab_id,
            document_id=document_snapshot.document_id,
            snapshot=snapshot_copy,
            previous_overlay=overlay_copy,
        )

        self._pending.add_session(session)

        LOGGER.debug(
            "ReviewManager.ensure_session: created session for tab_id=%s",
            tab_id,
        )

        return session

    def finalize(self, success: bool) -> None:
        """Finalize the current review after AI turn completion.

        If successful and there are edits, marks the review as ready
        for user decision. Otherwise drops the review.

        Args:
            success: Whether the AI turn completed successfully.

        Emits:
            ReviewStateChanged: When the review is ready or dropped.
        """
        if self._pending is None:
            LOGGER.debug("ReviewManager.finalize: no pending review")
            return

        turn_id = self._pending.turn_id

        if not success:
            LOGGER.debug("ReviewManager.finalize: turn failed, dropping")
            self.drop("turn-failed")
            return

        if self._pending.total_edit_count <= 0:
            LOGGER.debug("ReviewManager.finalize: no edits, dropping")
            self.drop("no-edits")
            return

        # Mark ready for review
        self._pending.ready = True

        LOGGER.debug(
            "ReviewManager.finalize: turn_id=%s ready with %d edits across %d tabs",
            turn_id,
            self._pending.total_edit_count,
            self._pending.affected_tab_count,
        )

        self._emit_state_changed()

    def accept(self) -> list[str]:
        """Accept the pending review edits.

        Returns:
            List of tab IDs that were affected by the accepted edits.

        Emits:
            ReviewAccepted: After the review is accepted.
            ReviewStateChanged: After the review is cleared.
        """
        if self._pending is None:
            LOGGER.debug("ReviewManager.accept: no pending review")
            return []

        turn_id = self._pending.turn_id
        affected_tabs = list(self._pending.sessions.keys())

        LOGGER.debug(
            "ReviewManager.accept: turn_id=%s, tabs=%s",
            turn_id,
            affected_tabs,
        )

        # Clear the pending review
        self._pending = None

        # Emit events
        self._bus.publish(ReviewAccepted(
            turn_id=turn_id,
            tabs=tuple(affected_tabs),
        ))
        self._emit_state_changed()

        return affected_tabs

    def reject(self) -> dict[str, DocumentState]:
        """Reject the pending review edits.

        Returns:
            Dict mapping tab_id to the document snapshot to restore.

        Emits:
            ReviewRejected: After the review is rejected.
            ReviewStateChanged: After the review is cleared.
        """
        if self._pending is None:
            LOGGER.debug("ReviewManager.reject: no pending review")
            return {}

        turn_id = self._pending.turn_id
        snapshots: dict[str, DocumentState] = {}

        # Collect snapshots for restoration
        for tab_id, session in self._pending.sessions.items():
            snapshots[tab_id] = session.snapshot

        affected_tabs = list(snapshots.keys())

        LOGGER.debug(
            "ReviewManager.reject: turn_id=%s, tabs=%s",
            turn_id,
            affected_tabs,
        )

        # Clear the pending review
        self._pending = None

        # Emit events
        self._bus.publish(ReviewRejected(
            turn_id=turn_id,
            tabs=tuple(affected_tabs),
        ))
        self._emit_state_changed()

        return snapshots

    def drop(self, reason: str) -> None:
        """Drop the pending review without accepting or rejecting.

        Used when the review is invalidated (e.g., new turn started,
        turn failed, or no edits applied).

        Args:
            reason: A description of why the review was dropped.

        Emits:
            ReviewStateChanged: After the review is cleared.
        """
        if self._pending is None:
            return

        turn_id = self._pending.turn_id

        LOGGER.debug(
            "ReviewManager.drop: turn_id=%s, reason=%s",
            turn_id,
            reason,
        )

        self._pending = None
        self._emit_state_changed()

    # ------------------------------------------------------------------
    # Session Helpers
    # ------------------------------------------------------------------

    def get_session(self, tab_id: str) -> ReviewSession | None:
        """Get the review session for a specific tab.

        Args:
            tab_id: The tab identifier.

        Returns:
            The ReviewSession for the tab, or None if not found.
        """
        if self._pending is None:
            return None
        return self._pending.get_session(tab_id)

    def get_snapshot(self, tab_id: str) -> DocumentState | None:
        """Get the document snapshot for a specific tab.

        Args:
            tab_id: The tab identifier.

        Returns:
            The DocumentState snapshot, or None if not found.
        """
        session = self.get_session(tab_id)
        if session is None:
            return None
        return session.snapshot

    def affected_tab_ids(self) -> tuple[str, ...]:
        """Get tuple of all tab IDs affected by the pending review."""
        if self._pending is None:
            return ()
        return tuple(self._pending.sessions.keys())

    def update_merged_spans(
        self,
        tab_id: str,
        spans: tuple[tuple[int, int], ...],
    ) -> None:
        """Update the merged spans for a tab's session.

        Args:
            tab_id: The tab identifier.
            spans: The new merged spans.
        """
        session = self.get_session(tab_id)
        if session is not None:
            session.merged_spans = spans

    # ------------------------------------------------------------------
    # Formatting Helpers
    # ------------------------------------------------------------------

    def format_summary(self) -> str:
        """Format a human-readable summary of the pending review.

        Returns:
            Summary string like "3 edits across 2 tabs".
        """
        if self._pending is None:
            return "0 edits across 0 tabs"

        edits = max(self._pending.total_edit_count, 0)
        tabs = max(self._pending.affected_tab_count, 1)
        edit_label = "edit" if edits == 1 else "edits"
        tab_label = "tab" if tabs == 1 else "tabs"
        return f"{edits} {edit_label} across {tabs} {tab_label}"

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _emit_state_changed(self) -> None:
        """Emit ReviewStateChanged event with current state."""
        if self._pending is None:
            self._bus.publish(ReviewStateChanged(
                turn_id="",
                ready=False,
                edit_count=0,
                tabs_affected=0,
            ))
        else:
            self._bus.publish(ReviewStateChanged(
                turn_id=self._pending.turn_id,
                ready=self._pending.ready,
                edit_count=self._pending.total_edit_count,
                tabs_affected=self._pending.affected_tab_count,
            ))


__all__ = ["ReviewManager"]
