"""AI turn manager domain service.

Manages AI turn execution state and lifecycle, emitting events for
each state transition. This is the single source of truth for AI turn state.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from ..events import (
    AITurnCanceled,
    AITurnCompleted,
    AITurnFailed,
    AITurnStarted,
    AITurnStreamChunk,
    EventBus,
)
from ..models.ai_models import AITurnState, AITurnStatus

if TYPE_CHECKING:  # pragma: no cover
    from ...ai.orchestration import AIOrchestrator

LOGGER = logging.getLogger(__name__)


class AITurnManager:
    """Domain manager for AI turn execution.

    Manages the lifecycle of AI turns, including starting, streaming,
    completion, failure, and cancellation. All state transitions emit
    events through the event bus.

    Events Emitted:
        - AITurnStarted: When a turn begins
        - AITurnStreamChunk: For each streaming content chunk
        - AITurnCompleted: When a turn finishes successfully
        - AITurnFailed: When a turn encounters an error
        - AITurnCanceled: When a turn is canceled
    """

    def __init__(
        self,
        orchestrator_provider: Callable[[], AIOrchestrator | None],
        event_bus: EventBus,
    ) -> None:
        """Initialize the AI turn manager.

        Args:
            orchestrator_provider: Callable that returns the AI orchestrator
                or None if unavailable.
            event_bus: The event bus for publishing events.
        """
        self._get_orchestrator = orchestrator_provider
        self._bus = event_bus
        self._current_turn: AITurnState | None = None
        self._task: asyncio.Task[Any] | None = None
        self._response_text: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_turn(self) -> AITurnState | None:
        """Get the current turn state, if any."""
        return self._current_turn

    def is_running(self) -> bool:
        """Check if an AI turn is currently running."""
        return self._current_turn is not None and self._current_turn.is_running

    # ------------------------------------------------------------------
    # Turn Lifecycle
    # ------------------------------------------------------------------

    async def start_turn(
        self,
        prompt: str,
        snapshot: Mapping[str, Any],
        metadata: Mapping[str, Any],
        history: Sequence[Mapping[str, str]] | None,
        on_stream_event: Callable[[Any], None] | None = None,
    ) -> AITurnState:
        """Start a new AI turn.

        Args:
            prompt: The user prompt to send to the AI.
            snapshot: Document snapshot data.
            metadata: Additional metadata for the turn.
            history: Chat history for context.
            on_stream_event: Optional callback for raw streaming events.

        Returns:
            The AITurnState for this turn.

        Raises:
            RuntimeError: If a turn is already in progress or orchestrator
                is unavailable.

        Emits:
            AITurnStarted: When the turn begins.
            AITurnStreamChunk: For each content chunk received.
            AITurnCompleted: On successful completion.
            AITurnFailed: On error.
        """
        if self.is_running():
            raise RuntimeError("AI turn already in progress")

        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            raise RuntimeError("AI orchestrator unavailable")

        # Generate unique turn ID
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"

        # Create turn state
        self._current_turn = AITurnState(
            turn_id=turn_id,
            prompt=prompt,
            status=AITurnStatus.RUNNING,
            metadata=dict(metadata) if metadata else {},
        )
        self._response_text = ""

        LOGGER.debug(
            "AITurnManager.start_turn: turn_id=%s, prompt_length=%d",
            turn_id,
            len(prompt),
        )

        # Emit started event
        self._bus.publish(AITurnStarted(turn_id=turn_id, prompt=prompt))

        try:
            # Run the orchestrator
            result = await orchestrator.run_chat(
                prompt,
                snapshot,
                metadata=metadata,
                history=history,
                on_event=lambda e: self._handle_stream_event(e, on_stream_event),
            )

            # Extract response text
            response_text = self._extract_response_text(result)
            self._response_text = response_text

            # Mark completed
            self._current_turn.mark_completed(edit_count=self._current_turn.edit_count)

            LOGGER.debug(
                "AITurnManager: turn completed, turn_id=%s, edit_count=%d",
                turn_id,
                self._current_turn.edit_count,
            )

            self._bus.publish(AITurnCompleted(
                turn_id=turn_id,
                success=True,
                edit_count=self._current_turn.edit_count,
                response_text=response_text,
            ))

            return self._current_turn

        except asyncio.CancelledError:
            # Turn was canceled
            if self._current_turn:
                self._current_turn.mark_canceled()
            LOGGER.debug("AITurnManager: turn canceled, turn_id=%s", turn_id)
            self._bus.publish(AITurnCanceled(turn_id=turn_id))
            raise

        except Exception as exc:
            # Turn failed
            error_msg = str(exc)
            if self._current_turn:
                self._current_turn.mark_failed(error_msg)

            LOGGER.warning(
                "AITurnManager: turn failed, turn_id=%s, error=%s",
                turn_id,
                error_msg,
            )

            self._bus.publish(AITurnFailed(turn_id=turn_id, error=error_msg))
            raise

    def cancel(self) -> None:
        """Cancel the current AI turn if running.

        Emits:
            AITurnCanceled: After the turn is canceled.
        """
        if not self.is_running():
            LOGGER.debug("AITurnManager.cancel: no turn running")
            return

        turn_id = self._current_turn.turn_id if self._current_turn else "unknown"
        LOGGER.debug("AITurnManager.cancel: canceling turn_id=%s", turn_id)

        # Try to cancel via orchestrator
        orchestrator = self._get_orchestrator()
        if orchestrator is not None:
            cancel_method = getattr(orchestrator, "cancel", None)
            if callable(cancel_method):
                try:
                    cancel_method()
                except Exception:  # pragma: no cover
                    LOGGER.debug("Error calling orchestrator.cancel", exc_info=True)

        # Cancel the task if we have one
        if self._task is not None and not self._task.done():
            self._task.cancel()

        # Update state
        if self._current_turn:
            self._current_turn.mark_canceled()
            self._bus.publish(AITurnCanceled(turn_id=self._current_turn.turn_id))

    def increment_edit_count(self) -> None:
        """Increment the edit count for the current turn.

        Call this when an edit is applied during the turn.
        """
        if self._current_turn is not None:
            self._current_turn.edit_count += 1

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _handle_stream_event(
        self,
        event: Any,
        external_handler: Callable[[Any], None] | None,
    ) -> None:
        """Process a streaming event from the orchestrator.

        Extracts content chunks and emits AITurnStreamChunk events.
        Also forwards to external handler if provided.
        """
        # Forward to external handler first
        if external_handler is not None:
            try:
                external_handler(event)
            except Exception:  # pragma: no cover
                LOGGER.debug("Error in external stream handler", exc_info=True)

        # Extract content from event
        event_type = getattr(event, "type", "") or ""

        if event_type in {"content.delta", "refusal.delta"}:
            content = getattr(event, "content", None)
            if content and self._current_turn:
                self._bus.publish(AITurnStreamChunk(
                    turn_id=self._current_turn.turn_id,
                    content=str(content),
                ))

    def _extract_response_text(self, result: Any) -> str:
        """Extract response text from orchestrator result.

        Handles both ChatResult objects and legacy dict responses.
        """
        if result is None:
            return ""

        # ChatResult object
        if hasattr(result, "response"):
            return (getattr(result, "response", "") or "").strip()

        # Legacy dict response
        if isinstance(result, dict):
            return (result.get("response", "") or "").strip()

        return ""


__all__ = ["AITurnManager"]
