"""AI turn state models for tracking AI execution lifecycle.

These dataclasses and enums represent the state of AI turns during
execution. They are used by the domain layer (AITurnManager) and
emitted via events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


class AITurnStatus(Enum):
    """Status of an AI turn in its lifecycle.

    Values:
        PENDING: Turn is queued but not yet started.
        RUNNING: Turn is actively executing.
        COMPLETED: Turn finished successfully.
        FAILED: Turn encountered an error.
        CANCELED: Turn was canceled by the user.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


@dataclass(slots=True)
class AITurnState:
    """State of an AI turn during execution.

    Tracks the full lifecycle of an AI turn from start to completion,
    including any edits made and errors encountered.

    Attributes:
        turn_id: Unique identifier for this turn.
        prompt: The user prompt that initiated this turn.
        status: Current status of the turn.
        edit_count: Number of edits applied during this turn.
        error: Error message if the turn failed.
        created_at: When the turn was initiated.
        completed_at: When the turn finished (success, failure, or cancel).
        metadata: Additional metadata about the turn.
    """

    turn_id: str
    prompt: str
    status: AITurnStatus = AITurnStatus.PENDING
    edit_count: int = 0
    error: str | None = None
    created_at: datetime = field(default_factory=_utcnow)
    completed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_running(self) -> bool:
        """Check if the turn is currently running."""
        return self.status == AITurnStatus.RUNNING

    @property
    def is_finished(self) -> bool:
        """Check if the turn has finished (any terminal state)."""
        return self.status in (
            AITurnStatus.COMPLETED,
            AITurnStatus.FAILED,
            AITurnStatus.CANCELED,
        )

    @property
    def is_successful(self) -> bool:
        """Check if the turn completed successfully."""
        return self.status == AITurnStatus.COMPLETED

    def mark_running(self) -> None:
        """Transition to running state."""
        self.status = AITurnStatus.RUNNING

    def mark_completed(self, edit_count: int = 0) -> None:
        """Transition to completed state."""
        self.status = AITurnStatus.COMPLETED
        self.edit_count = edit_count
        self.completed_at = _utcnow()

    def mark_failed(self, error: str) -> None:
        """Transition to failed state with error message."""
        self.status = AITurnStatus.FAILED
        self.error = error
        self.completed_at = _utcnow()

    def mark_canceled(self) -> None:
        """Transition to canceled state."""
        self.status = AITurnStatus.CANCELED
        self.completed_at = _utcnow()


__all__ = [
    "AITurnStatus",
    "AITurnState",
]
