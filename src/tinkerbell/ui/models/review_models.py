"""Review-related data models for AI edit review workflow.

These dataclasses represent the state of pending AI edits awaiting user review.
They are used by the domain layer (ReviewManager) and emitted via events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ...editor.document_model import DocumentState
    from ...editor.editor_widget import DiffOverlayState


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class PendingEdit:
    """A single edit proposed by the AI, awaiting user review.

    Attributes:
        edit_id: Unique identifier for this edit.
        tab_id: The tab where this edit applies.
        action: The edit action type (e.g., "replace", "insert", "delete").
        range: Character range (start, end) in the original document.
        diff: The diff text showing the change.
        spans: Tuple of (start, end) spans affected by this edit.
        created_at: When this edit was recorded.
    """

    edit_id: str
    tab_id: str
    action: str
    range: tuple[int, int]
    diff: str
    spans: tuple[tuple[int, int], ...] = ()
    created_at: datetime = field(default_factory=_utcnow)


@dataclass(slots=True)
class ReviewSession:
    """Per-tab session data for a pending AI turn review.

    Tracks the document state and all edits for a single tab within
    an AI turn review.

    Attributes:
        tab_id: The tab identifier.
        document_id: The document identifier.
        snapshot: Document state snapshot taken before edits.
        edits: List of pending edits for this tab.
        previous_overlay: The diff overlay state before this session.
        merged_spans: Merged spans of all edits for overlay display.
    """

    tab_id: str
    document_id: str
    snapshot: DocumentState
    edits: list[PendingEdit] = field(default_factory=list)
    previous_overlay: DiffOverlayState | None = None
    merged_spans: tuple[tuple[int, int], ...] = ()


@dataclass(slots=True)
class PendingReview:
    """Envelope capturing the full state of an AI turn awaiting review.

    Contains all sessions (one per affected tab) for a completed AI turn
    that is awaiting user acceptance or rejection.

    Attributes:
        turn_id: Unique identifier for the AI turn.
        prompt: The user prompt that triggered the turn.
        sessions: Mapping of tab_id to ReviewSession.
        ready: Whether the review is ready for user decision.
        created_at: When the review was initiated.
        total_edit_count: Total number of edits across all sessions.
    """

    turn_id: str
    prompt: str
    sessions: dict[str, ReviewSession] = field(default_factory=dict)
    ready: bool = False
    created_at: datetime = field(default_factory=_utcnow)
    total_edit_count: int = 0

    def add_session(self, session: ReviewSession) -> None:
        """Add or replace a session for the given tab."""
        self.sessions[session.tab_id] = session

    def get_session(self, tab_id: str) -> ReviewSession | None:
        """Get the session for a specific tab."""
        return self.sessions.get(tab_id)

    @property
    def affected_tab_count(self) -> int:
        """Return the number of tabs affected by this review."""
        return len(self.sessions)


__all__ = [
    "PendingEdit",
    "ReviewSession",
    "PendingReview",
]
