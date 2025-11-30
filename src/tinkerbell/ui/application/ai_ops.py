"""AI operation use cases.

This module provides use cases for AI turn operations:
- RunAITurnUseCase: Execute an AI turn with the orchestrator
- CancelAITurnUseCase: Cancel a running AI turn
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol, Sequence

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ..domain.ai_turn_manager import AITurnManager
    from ..domain.review_manager import ReviewManager
    from ..events import EventBus
    from ..models.ai_models import AITurnState

LOGGER = logging.getLogger(__name__)


class SnapshotProvider(Protocol):
    """Protocol for generating document snapshots."""

    def generate_snapshot(
        self,
        *,
        tab_id: str | None = None,
        delta_only: bool = False,
    ) -> dict[str, Any]:
        """Generate a snapshot of the workspace documents.

        Args:
            tab_id: Optional specific tab to snapshot.
            delta_only: If True, only include changes since last snapshot.

        Returns:
            Snapshot dictionary with document content and metadata.
            If no tabs are open, returns an empty snapshot with no_document=True.
        """
        ...


class RunAITurnUseCase:
    """Use case for executing an AI turn.

    Orchestrates the full AI turn workflow:
    1. Auto-accept any pending review
    2. Begin a new review session
    3. Generate document snapshot
    4. Execute the AI turn via AITurnManager
    5. Finalize the review on completion

    Events Emitted (via AITurnManager):
        - AITurnStarted: When the turn begins
        - AITurnStreamChunk: For each streaming content chunk
        - AITurnCompleted: When the turn finishes successfully
        - AITurnFailed: When the turn encounters an error
        - AITurnCanceled: When the turn is canceled
    """

    __slots__ = (
        "_ai_turn_manager",
        "_review_manager",
        "_snapshot_provider",
        "_event_bus",
        "_metadata_enricher",
        "_history_provider",
        "_stream_handler",
    )

    def __init__(
        self,
        ai_turn_manager: AITurnManager,
        review_manager: ReviewManager,
        snapshot_provider: SnapshotProvider,
        event_bus: EventBus,
        *,
        metadata_enricher: Callable[[dict[str, Any]], None] | None = None,
        history_provider: Callable[[], Sequence[Mapping[str, str]] | None] | None = None,
        stream_handler: Callable[[Any], None] | None = None,
    ) -> None:
        """Initialize the use case.

        Args:
            ai_turn_manager: Manager for AI turn execution.
            review_manager: Manager for pending reviews.
            snapshot_provider: Provider for document snapshots.
            event_bus: Event bus for publishing events.
            metadata_enricher: Optional callback to add metadata to snapshot.
            history_provider: Optional callback to get chat history.
            stream_handler: Optional callback for streaming events.
        """
        self._ai_turn_manager = ai_turn_manager
        self._review_manager = review_manager
        self._snapshot_provider = snapshot_provider
        self._event_bus = event_bus
        self._metadata_enricher = metadata_enricher
        self._history_provider = history_provider
        self._stream_handler = stream_handler

    async def execute(
        self,
        prompt: str,
        metadata: Mapping[str, Any] | None = None,
        *,
        chat_snapshot: Mapping[str, Any] | None = None,
    ) -> AITurnState:
        """Execute an AI turn.

        Args:
            prompt: The user prompt to send to the AI.
            metadata: Optional metadata for the turn.
            chat_snapshot: Optional snapshot of chat state for restoration.

        Returns:
            The AITurnState representing this turn.

        Raises:
            RuntimeError: If AI is unavailable or turn already running.
        """
        LOGGER.debug(
            "RunAITurnUseCase.execute: prompt_length=%d",
            len(prompt),
        )

        # Auto-accept any pending review before starting new turn
        pending = self._review_manager.pending_review
        if pending is not None and pending.ready:
            LOGGER.debug(
                "RunAITurnUseCase: auto-accepting pending review %s",
                pending.turn_id,
            )
            self._review_manager.accept()

        # Begin new review session
        # Generate a turn_id that matches what AITurnManager will create
        # (We could also get this from AITurnManager after start_turn)
        import uuid
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"

        self._review_manager.begin_review(
            turn_id=turn_id,
            prompt=prompt,
            metadata=dict(metadata) if metadata else None,
        )

        # Generate document snapshot
        snapshot = self._snapshot_provider.generate_snapshot()

        # Enrich snapshot with additional metadata (e.g., embedding info)
        if self._metadata_enricher is not None:
            try:
                self._metadata_enricher(snapshot)
            except Exception:  # pragma: no cover
                LOGGER.debug(
                    "RunAITurnUseCase: metadata enricher failed",
                    exc_info=True,
                )

        # Get chat history if provider available
        history: Sequence[Mapping[str, str]] | None = None
        if self._history_provider is not None:
            try:
                history = self._history_provider()
            except Exception:  # pragma: no cover
                LOGGER.debug(
                    "RunAITurnUseCase: history provider failed",
                    exc_info=True,
                )

        # Execute the AI turn
        try:
            turn_state = await self._ai_turn_manager.start_turn(
                prompt=prompt,
                snapshot=snapshot,
                metadata=dict(metadata) if metadata else {},
                history=history,
                on_stream_event=self._stream_handler,
                turn_id=turn_id,
            )

            # Finalize review based on turn success
            success = turn_state.status.value == "completed"
            self._review_manager.finalize(success)

            LOGGER.debug(
                "RunAITurnUseCase: turn completed, turn_id=%s, success=%s",
                turn_state.turn_id,
                success,
            )

            return turn_state

        except Exception as exc:
            # Turn failed - drop the review
            LOGGER.debug(
                "RunAITurnUseCase: turn failed with %s, dropping review",
                type(exc).__name__,
            )
            self._review_manager.drop("turn-failed")
            raise


class CancelAITurnUseCase:
    """Use case for canceling a running AI turn.

    Cancels the current AI turn and drops any associated pending review.

    Events Emitted (via AITurnManager):
        - AITurnCanceled: After the turn is canceled
    """

    __slots__ = ("_ai_turn_manager", "_review_manager", "_event_bus")

    def __init__(
        self,
        ai_turn_manager: AITurnManager,
        review_manager: ReviewManager,
        event_bus: EventBus,
    ) -> None:
        """Initialize the use case.

        Args:
            ai_turn_manager: Manager for AI turn execution.
            review_manager: Manager for pending reviews.
            event_bus: Event bus for publishing events.
        """
        self._ai_turn_manager = ai_turn_manager
        self._review_manager = review_manager
        self._event_bus = event_bus

    def execute(self) -> bool:
        """Cancel the current AI turn.

        Returns:
            True if a turn was canceled, False if no turn was running.
        """
        if not self._ai_turn_manager.is_running():
            LOGGER.debug("CancelAITurnUseCase: no turn running")
            return False

        current = self._ai_turn_manager.current_turn
        turn_id = current.turn_id if current else "unknown"

        LOGGER.debug("CancelAITurnUseCase: canceling turn %s", turn_id)

        # Cancel the AI turn
        self._ai_turn_manager.cancel()

        # Drop any pending review associated with this turn
        pending = self._review_manager.pending_review
        if pending is not None:
            self._review_manager.drop("turn-canceled")

        return True


__all__ = [
    "SnapshotProvider",
    "RunAITurnUseCase",
    "CancelAITurnUseCase",
]
