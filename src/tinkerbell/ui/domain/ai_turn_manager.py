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
    AITurnToolExecuted,
    EditorLockChanged,
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
        turn_id: str | None = None,
    ) -> AITurnState:
        """Start a new AI turn.

        Args:
            prompt: The user prompt to send to the AI.
            snapshot: Document snapshot data.
            metadata: Additional metadata for the turn.
            history: Chat history for context.
            on_stream_event: Optional callback for raw streaming events.
            turn_id: Optional turn ID. If not provided, a unique ID is generated.

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

        # Use provided turn ID or generate a unique one
        if turn_id is None:
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

        # Lock editor during AI turn
        self._bus.publish(EditorLockChanged(locked=True, reason="AI_TURN"))

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

            # Unlock editor
            self._bus.publish(EditorLockChanged(locked=False, reason=""))

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
            # Unlock editor
            self._bus.publish(EditorLockChanged(locked=False, reason=""))
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

            # Unlock editor
            self._bus.publish(EditorLockChanged(locked=False, reason=""))
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

    async def suggest_followups(
        self,
        history: Sequence[Mapping[str, str]],
        *,
        max_suggestions: int = 4,
    ) -> list[str]:
        """Generate follow-up suggestions based on chat history.

        Args:
            history: Conversation history as role/content mappings.
            max_suggestions: Maximum number of suggestions to generate.

        Returns:
            List of suggested follow-up prompts, or empty list if
            orchestrator is unavailable or history is empty.
        """
        if not history:
            return []

        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            LOGGER.debug("AITurnManager.suggest_followups: no orchestrator")
            return []

        suggest_method = getattr(orchestrator, "suggest_followups", None)
        if not callable(suggest_method):
            LOGGER.debug("AITurnManager.suggest_followups: orchestrator has no suggest_followups")
            return []

        try:
            suggestions = await suggest_method(history, max_suggestions=max_suggestions)
            LOGGER.debug(
                "AITurnManager.suggest_followups: generated %d suggestions",
                len(suggestions),
            )
            return suggestions
        except Exception:
            LOGGER.debug("AITurnManager.suggest_followups: error", exc_info=True)
            return []

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
        Also handles tool execution events.
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

        # Handle tool call arguments done (tool is about to execute)
        elif event_type == "tool_calls.function.arguments.done":
            if self._current_turn:
                tool_name = getattr(event, "tool_name", "") or ""
                tool_call_id = getattr(event, "tool_call_id", "") or ""
                arguments = getattr(event, "tool_arguments", "") or ""
                self._bus.publish(AITurnToolExecuted(
                    turn_id=self._current_turn.turn_id,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                    result="(running…)",
                    success=True,
                    duration_ms=0.0,
                ))

        # Handle tool result (tool execution completed)
        elif event_type == "tool_calls.result":
            if self._current_turn:
                tool_name = getattr(event, "tool_name", None) or ""
                tool_call_id = getattr(event, "tool_call_id", "") or ""
                content = getattr(event, "content", "") or ""
                duration_ms = getattr(event, "duration_ms", 0.0) or 0.0
                # Try to detect success from parsed result
                parsed = getattr(event, "parsed", None)
                success = True
                if parsed is not None:
                    success = self._detect_tool_success(parsed, content)
                self._bus.publish(AITurnToolExecuted(
                    turn_id=self._current_turn.turn_id,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                    arguments="",
                    result=content[:500] if content else "",  # Truncate for display
                    success=success,
                    duration_ms=duration_ms,
                ))

    def _detect_tool_success(self, parsed: Any, content: str) -> bool:
        """Detect if a tool execution was successful from result."""
        # Check parsed result for status field
        if isinstance(parsed, dict):
            status = parsed.get("status", "")
            if isinstance(status, str):
                status_lower = status.lower()
                if status_lower in {"error", "failed", "failure"}:
                    return False
            # Check for error field
            if parsed.get("error") or parsed.get("exception"):
                return False
        # Check content for error indicators
        content_lower = (content or "").lower()
        if content_lower.startswith("error:"):
            return False
        return True

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
