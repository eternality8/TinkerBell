"""Review operation use cases.

This module provides use cases for review operations:
- AcceptReviewUseCase: Accept pending AI edits
- RejectReviewUseCase: Reject pending AI edits and restore documents
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ..domain.document_store import DocumentStore
    from ..domain.overlay_manager import OverlayManager
    from ..domain.review_manager import ReviewManager
    from ..events import EventBus

LOGGER = logging.getLogger(__name__)


class DocumentRestorer(Protocol):
    """Protocol for restoring document state."""

    def restore_document(
        self,
        tab_id: str,
        snapshot: Any,
    ) -> bool:
        """Restore a document to a previous snapshot.

        Args:
            tab_id: The tab to restore.
            snapshot: The document state snapshot to restore.

        Returns:
            True if restoration succeeded, False otherwise.
        """
        ...


class OverlayRestorer(Protocol):
    """Protocol for restoring overlay state."""

    def show_overlay(
        self,
        tab_id: str,
        diff: str,
        spans: tuple[tuple[int, int], ...],
        summary: str | None = None,
        source: str | None = None,
    ) -> bool:
        """Show a diff overlay on a tab.

        Args:
            tab_id: The tab to show the overlay on.
            diff: The diff text/label.
            spans: The spans to highlight.
            summary: Optional summary text.
            source: Optional source identifier.

        Returns:
            True if overlay was shown successfully.
        """
        ...


class ChatRestorer(Protocol):
    """Protocol for restoring chat state."""

    def restore_chat_state(
        self,
        snapshot: Any,
    ) -> bool:
        """Restore chat panel to a previous snapshot.

        Args:
            snapshot: The chat state snapshot to restore.

        Returns:
            True if restoration succeeded, False otherwise.
        """
        ...


class WorkspaceSyncer(Protocol):
    """Protocol for syncing workspace state to disk."""

    def sync_workspace(self) -> None:
        """Persist workspace state to disk."""
        ...


@dataclass(slots=True, frozen=True)
class AcceptResult:
    """Result of accepting a review.

    Attributes:
        success: Whether the accept operation succeeded.
        tabs_cleared: List of tab IDs where overlays were cleared.
        skipped_tabs: List of tab IDs that were skipped (e.g., closed).
        message: Human-readable status message.
    """

    success: bool
    tabs_cleared: tuple[str, ...]
    skipped_tabs: tuple[str, ...]
    message: str


@dataclass(slots=True, frozen=True)
class RejectResult:
    """Result of rejecting a review.

    Attributes:
        success: Whether the reject operation succeeded.
        tabs_restored: List of tab IDs where documents were restored.
        skipped_tabs: List of tab IDs that were skipped (e.g., closed).
        blocked_tabs: List of tab IDs that couldn't be restored.
        message: Human-readable status message.
    """

    success: bool
    tabs_restored: tuple[str, ...]
    skipped_tabs: tuple[str, ...]
    blocked_tabs: tuple[str, ...]
    message: str


class AcceptReviewUseCase:
    """Use case for accepting pending AI edits.

    Orchestrates accepting a pending review:
    1. Validate review is ready for acceptance
    2. Clear overlays for all affected tabs
    3. Mark review as accepted
    4. Sync workspace state to disk

    Events Emitted (via ReviewManager):
        - ReviewAccepted: After edits are accepted
        - ReviewStateChanged: After review is cleared
    """

    __slots__ = (
        "_review_manager",
        "_overlay_manager",
        "_workspace_syncer",
        "_event_bus",
    )

    def __init__(
        self,
        review_manager: ReviewManager,
        overlay_manager: OverlayManager,
        event_bus: EventBus,
        *,
        workspace_syncer: WorkspaceSyncer | None = None,
    ) -> None:
        """Initialize the use case.

        Args:
            review_manager: Manager for pending reviews.
            overlay_manager: Manager for diff overlays.
            event_bus: Event bus for publishing events.
            workspace_syncer: Optional syncer for persisting changes.
        """
        self._review_manager = review_manager
        self._overlay_manager = overlay_manager
        self._workspace_syncer = workspace_syncer
        self._event_bus = event_bus

    def execute(self) -> AcceptResult:
        """Accept the pending review.

        Returns:
            AcceptResult with details about the operation.
        """
        pending = self._review_manager.pending_review
        if pending is None:
            LOGGER.debug("AcceptReviewUseCase: no pending review")
            return AcceptResult(
                success=False,
                tabs_cleared=(),
                skipped_tabs=(),
                message="No AI edits pending review",
            )

        if not pending.ready:
            LOGGER.debug("AcceptReviewUseCase: review not ready")
            return AcceptResult(
                success=False,
                tabs_cleared=(),
                skipped_tabs=(),
                message="AI turn still running – review not ready",
            )

        # Clear overlays for affected tabs
        tabs_cleared: list[str] = []
        skipped_tabs: list[str] = []

        for tab_id in pending.sessions.keys():
            if self._overlay_manager.has_overlay(tab_id):
                if self._overlay_manager.clear_overlay(tab_id):
                    tabs_cleared.append(tab_id)
                else:
                    skipped_tabs.append(tab_id)
            else:
                # Tab had no overlay (may have been closed)
                skipped_tabs.append(tab_id)

        # Accept the review (emits ReviewAccepted event)
        affected_tabs = self._review_manager.accept()

        # Format summary message
        summary = self._review_manager.format_summary()
        message = f"Accepted {summary}"
        if skipped_tabs:
            suffix = "tab" if len(skipped_tabs) == 1 else "tabs"
            message = f"{message} (skipped {len(skipped_tabs)} closed {suffix})"

        # Sync workspace to persist changes
        if self._workspace_syncer is not None:
            try:
                self._workspace_syncer.sync_workspace()
            except Exception:  # pragma: no cover
                LOGGER.debug(
                    "AcceptReviewUseCase: workspace sync failed",
                    exc_info=True,
                )

        LOGGER.debug(
            "AcceptReviewUseCase: accepted, cleared=%d, skipped=%d",
            len(tabs_cleared),
            len(skipped_tabs),
        )

        return AcceptResult(
            success=True,
            tabs_cleared=tuple(tabs_cleared),
            skipped_tabs=tuple(skipped_tabs),
            message=message,
        )


class RejectReviewUseCase:
    """Use case for rejecting pending AI edits.

    Orchestrates rejecting a pending review:
    1. Validate review is ready for rejection
    2. Verify documents haven't changed since the AI turn
    3. Restore document snapshots for all affected tabs
    4. Restore previous overlay states
    5. Restore chat panel snapshot
    6. Mark review as rejected
    7. Sync workspace state to disk

    Events Emitted (via ReviewManager):
        - ReviewRejected: After edits are rejected
        - ReviewStateChanged: After review is cleared
    """

    __slots__ = (
        "_review_manager",
        "_overlay_manager",
        "_document_restorer",
        "_overlay_restorer",
        "_chat_restorer",
        "_workspace_syncer",
        "_event_bus",
    )

    def __init__(
        self,
        review_manager: ReviewManager,
        overlay_manager: OverlayManager,
        document_restorer: DocumentRestorer,
        event_bus: EventBus,
        *,
        overlay_restorer: OverlayRestorer | None = None,
        chat_restorer: ChatRestorer | None = None,
        workspace_syncer: WorkspaceSyncer | None = None,
    ) -> None:
        """Initialize the use case.

        Args:
            review_manager: Manager for pending reviews.
            overlay_manager: Manager for diff overlays.
            document_restorer: Restorer for document snapshots.
            event_bus: Event bus for publishing events.
            overlay_restorer: Optional restorer for overlay states.
            chat_restorer: Optional restorer for chat states.
            workspace_syncer: Optional syncer for persisting changes.
        """
        self._review_manager = review_manager
        self._overlay_manager = overlay_manager
        self._document_restorer = document_restorer
        self._overlay_restorer = overlay_restorer
        self._chat_restorer = chat_restorer
        self._workspace_syncer = workspace_syncer
        self._event_bus = event_bus

    def execute(self) -> RejectResult:
        """Reject the pending review and restore documents.

        Returns:
            RejectResult with details about the operation.
        """
        pending = self._review_manager.pending_review
        if pending is None:
            LOGGER.debug("RejectReviewUseCase: no pending review")
            return RejectResult(
                success=False,
                tabs_restored=(),
                skipped_tabs=(),
                blocked_tabs=(),
                message="No AI edits pending review",
            )

        if not pending.ready:
            LOGGER.debug("RejectReviewUseCase: review not ready")
            return RejectResult(
                success=False,
                tabs_restored=(),
                skipped_tabs=(),
                blocked_tabs=(),
                message="AI turn still running – review not ready",
            )

        if not pending.sessions:
            LOGGER.debug("RejectReviewUseCase: no sessions to restore")
            self._review_manager.drop("empty-review")
            return RejectResult(
                success=False,
                tabs_restored=(),
                skipped_tabs=(),
                blocked_tabs=(),
                message="No AI edits pending review",
            )

        # Restore documents for affected tabs
        tabs_restored: list[str] = []
        skipped_tabs: list[str] = []
        blocked_tabs: list[str] = []

        for tab_id, session in pending.sessions.items():
            # Try to restore the document
            snapshot_copy = deepcopy(session.snapshot)

            if not self._document_restorer.restore_document(tab_id, snapshot_copy):
                # Document couldn't be restored (tab closed or document changed)
                skipped_tabs.append(tab_id)
                continue

            tabs_restored.append(tab_id)

            # Clear the current overlay
            self._overlay_manager.discard_overlay(tab_id)

            # Restore previous overlay if available
            if session.previous_overlay is not None and self._overlay_restorer is not None:
                prior = session.previous_overlay
                try:
                    self._overlay_restorer.show_overlay(
                        tab_id=tab_id,
                        diff=prior.diff,
                        spans=prior.spans,
                        summary=prior.summary,
                        source=prior.source,
                    )
                except Exception:  # pragma: no cover
                    LOGGER.debug(
                        "RejectReviewUseCase: failed to restore overlay for %s",
                        tab_id,
                        exc_info=True,
                    )

        # Restore chat snapshot if available
        # Note: We need to access the chat snapshot from the pending review
        # This would need to be stored in the ReviewManager/PendingReview
        if self._chat_restorer is not None:
            # Chat snapshot restoration would go here
            # The actual snapshot needs to come from somewhere
            pass

        # Reject the review (emits ReviewRejected event, returns snapshots)
        self._review_manager.reject()

        # Format summary message
        message = f"Rejected {len(tabs_restored)} edit(s)"
        if skipped_tabs:
            suffix = "tab" if len(skipped_tabs) == 1 else "tabs"
            message = f"{message} (skipped {len(skipped_tabs)} closed {suffix})"

        # Sync workspace to persist changes
        if self._workspace_syncer is not None:
            try:
                self._workspace_syncer.sync_workspace()
            except Exception:  # pragma: no cover
                LOGGER.debug(
                    "RejectReviewUseCase: workspace sync failed",
                    exc_info=True,
                )

        LOGGER.debug(
            "RejectReviewUseCase: rejected, restored=%d, skipped=%d, blocked=%d",
            len(tabs_restored),
            len(skipped_tabs),
            len(blocked_tabs),
        )

        return RejectResult(
            success=True,
            tabs_restored=tuple(tabs_restored),
            skipped_tabs=tuple(skipped_tabs),
            blocked_tabs=tuple(blocked_tabs),
            message=message,
        )


__all__ = [
    "DocumentRestorer",
    "OverlayRestorer",
    "ChatRestorer",
    "WorkspaceSyncer",
    "AcceptResult",
    "RejectResult",
    "AcceptReviewUseCase",
    "RejectReviewUseCase",
]
