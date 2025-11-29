"""Status updater components for the presentation layer.

This module provides reactive updater components that subscribe to
domain events and update UI widgets accordingly.

Classes:
    StatusBarUpdater: Updates status bar widgets based on events
    ChatPanelUpdater: Updates chat panel based on AI events
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ..events import (
        AITurnCanceled,
        AITurnCompleted,
        AITurnFailed,
        AITurnStarted,
        AITurnStreamChunk,
        DocumentModified,
        EditorLockChanged,
        EmbeddingStateChanged,
        EventBus,
        NoticePosted,
        ReviewStateChanged,
        StatusMessage,
    )

LOGGER = logging.getLogger(__name__)


class StatusBarProtocol(Protocol):
    """Protocol for status bar widgets."""

    def set_message(self, message: str, *, timeout_ms: int | None = None) -> None:
        """Show a primary status message."""
        ...

    def set_autosave_state(self, state: str, *, detail: str | None = None) -> None:
        """Update the autosave indicator."""
        ...

    def set_embedding_status(self, status: str, *, detail: str | None = None) -> None:
        """Update the embedding status indicator."""
        ...

    def set_lock_state(self, locked: bool, *, reason: str = "") -> None:
        """Update the editor lock indicator."""
        ...

    def show_review_controls(self, visible: bool) -> None:
        """Show or hide the review accept/reject controls."""
        ...

    def set_review_summary(self, summary: str) -> None:
        """Update the review summary text."""
        ...


class ChatPanelProtocol(Protocol):
    """Protocol for chat panel widgets."""

    def append_ai_message(
        self,
        message: Any,
        *,
        streaming: bool = False,
    ) -> Any:
        """Add an AI-authored message, optionally streaming."""
        ...

    def set_guardrail_state(
        self,
        status: str,
        *,
        detail: str | None = None,
    ) -> None:
        """Update the guardrail indicator state."""
        ...


class StatusBarUpdater:
    """Reactive updater that subscribes to events and updates the status bar.

    This component provides a clean separation between domain events and
    UI updates. It subscribes to relevant events and translates them
    into status bar widget updates.

    Events Handled:
        - StatusMessage: Updates the main status message
        - EditorLockChanged: Updates the lock indicator
        - ReviewStateChanged: Shows/hides review controls
        - EmbeddingStateChanged: Updates embedding status
        - DocumentModified: Updates autosave indicator

    Example:
        updater = StatusBarUpdater(status_bar, event_bus)
        # Status bar now automatically updates when events are published
        # ...
        updater.dispose()  # Clean up subscriptions
    """

    __slots__ = ("_status_bar", "_event_bus", "_subscribed")

    def __init__(
        self,
        status_bar: StatusBarProtocol,
        event_bus: "EventBus",
    ) -> None:
        """Initialize the updater and subscribe to events.

        Args:
            status_bar: The status bar widget to update.
            event_bus: The event bus to subscribe to.
        """
        self._status_bar = status_bar
        self._event_bus = event_bus
        self._subscribed = True

        # Subscribe to events
        self._subscribe()

    def _subscribe(self) -> None:
        """Subscribe to all relevant events."""
        from ..events import (
            DocumentModified,
            EditorLockChanged,
            EmbeddingStateChanged,
            ReviewStateChanged,
            StatusMessage,
        )

        self._event_bus.subscribe(StatusMessage, self._on_status_message)
        self._event_bus.subscribe(EditorLockChanged, self._on_editor_lock_changed)
        self._event_bus.subscribe(ReviewStateChanged, self._on_review_state_changed)
        self._event_bus.subscribe(EmbeddingStateChanged, self._on_embedding_state_changed)
        self._event_bus.subscribe(DocumentModified, self._on_document_modified)

        LOGGER.debug("StatusBarUpdater: subscribed to events")

    def dispose(self) -> None:
        """Unsubscribe from all events.

        Call this method to clean up when the updater is no longer needed.
        """
        if not self._subscribed:
            return

        from ..events import (
            DocumentModified,
            EditorLockChanged,
            EmbeddingStateChanged,
            ReviewStateChanged,
            StatusMessage,
        )

        self._event_bus.unsubscribe(StatusMessage, self._on_status_message)
        self._event_bus.unsubscribe(EditorLockChanged, self._on_editor_lock_changed)
        self._event_bus.unsubscribe(ReviewStateChanged, self._on_review_state_changed)
        self._event_bus.unsubscribe(EmbeddingStateChanged, self._on_embedding_state_changed)
        self._event_bus.unsubscribe(DocumentModified, self._on_document_modified)

        self._subscribed = False
        LOGGER.debug("StatusBarUpdater: disposed")

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    def _on_status_message(self, event: "StatusMessage") -> None:
        """Handle StatusMessage events."""
        timeout = event.timeout_ms if event.timeout_ms > 0 else None
        self._status_bar.set_message(event.message, timeout_ms=timeout)

    def _on_editor_lock_changed(self, event: "EditorLockChanged") -> None:
        """Handle EditorLockChanged events."""
        self._status_bar.set_lock_state(event.locked, reason=event.reason)

    def _on_review_state_changed(self, event: "ReviewStateChanged") -> None:
        """Handle ReviewStateChanged events."""
        if event.ready and event.edit_count > 0:
            # Show review controls with summary
            self._status_bar.show_review_controls(True)
            edit_label = "edit" if event.edit_count == 1 else "edits"
            tab_label = "tab" if event.tabs_affected == 1 else "tabs"
            summary = f"{event.edit_count} {edit_label} across {event.tabs_affected} {tab_label}"
            self._status_bar.set_review_summary(summary)
        else:
            # Hide review controls
            self._status_bar.show_review_controls(False)

    def _on_embedding_state_changed(self, event: "EmbeddingStateChanged") -> None:
        """Handle EmbeddingStateChanged events."""
        # Map status to display text
        if event.status == "ready":
            status_text = "Ready"
        elif event.status == "initializing":
            status_text = "Initializing..."
        elif event.status == "error":
            status_text = "Error"
        elif event.status == "disabled":
            status_text = "Disabled"
        else:
            status_text = event.status.capitalize()

        detail = event.detail
        if event.backend and event.backend != "disabled":
            detail = f"{event.backend}: {detail}" if detail else event.backend

        self._status_bar.set_embedding_status(status_text, detail=detail)

    def _on_document_modified(self, event: "DocumentModified") -> None:
        """Handle DocumentModified events."""
        # Mark document as having unsaved changes
        self._status_bar.set_autosave_state("Unsaved", detail="Modified")


class ChatPanelUpdater:
    """Reactive updater that subscribes to events and updates the chat panel.

    This component provides a clean separation between AI events and
    chat panel UI updates. It handles streaming, completion, and notices.

    Events Handled:
        - AITurnStarted: Sets running state
        - AITurnStreamChunk: Appends streaming content
        - AITurnCompleted: Finalizes the AI message
        - AITurnFailed: Shows error message
        - AITurnCanceled: Shows cancellation notice
        - NoticePosted: Shows notice message
        - ReviewStateChanged: Updates guardrail state

    Example:
        updater = ChatPanelUpdater(chat_panel, event_bus)
        # Chat panel now automatically updates when events are published
        # ...
        updater.dispose()  # Clean up subscriptions
    """

    __slots__ = ("_chat_panel", "_event_bus", "_message_factory", "_subscribed")

    def __init__(
        self,
        chat_panel: ChatPanelProtocol,
        event_bus: "EventBus",
        *,
        message_factory: Any | None = None,
    ) -> None:
        """Initialize the updater and subscribe to events.

        Args:
            chat_panel: The chat panel widget to update.
            event_bus: The event bus to subscribe to.
            message_factory: Optional factory for creating ChatMessage objects.
                           If not provided, imports from chat module.
        """
        self._chat_panel = chat_panel
        self._event_bus = event_bus
        self._message_factory = message_factory
        self._subscribed = True

        # Subscribe to events
        self._subscribe()

    def _subscribe(self) -> None:
        """Subscribe to all relevant events."""
        from ..events import (
            AITurnCanceled,
            AITurnCompleted,
            AITurnFailed,
            AITurnStarted,
            AITurnStreamChunk,
            NoticePosted,
            ReviewStateChanged,
        )

        self._event_bus.subscribe(AITurnStarted, self._on_ai_turn_started)
        self._event_bus.subscribe(AITurnStreamChunk, self._on_ai_turn_stream_chunk)
        self._event_bus.subscribe(AITurnCompleted, self._on_ai_turn_completed)
        self._event_bus.subscribe(AITurnFailed, self._on_ai_turn_failed)
        self._event_bus.subscribe(AITurnCanceled, self._on_ai_turn_canceled)
        self._event_bus.subscribe(NoticePosted, self._on_notice_posted)
        self._event_bus.subscribe(ReviewStateChanged, self._on_review_state_changed)

        LOGGER.debug("ChatPanelUpdater: subscribed to events")

    def dispose(self) -> None:
        """Unsubscribe from all events.

        Call this method to clean up when the updater is no longer needed.
        """
        if not self._subscribed:
            return

        from ..events import (
            AITurnCanceled,
            AITurnCompleted,
            AITurnFailed,
            AITurnStarted,
            AITurnStreamChunk,
            NoticePosted,
            ReviewStateChanged,
        )

        self._event_bus.unsubscribe(AITurnStarted, self._on_ai_turn_started)
        self._event_bus.unsubscribe(AITurnStreamChunk, self._on_ai_turn_stream_chunk)
        self._event_bus.unsubscribe(AITurnCompleted, self._on_ai_turn_completed)
        self._event_bus.unsubscribe(AITurnFailed, self._on_ai_turn_failed)
        self._event_bus.unsubscribe(AITurnCanceled, self._on_ai_turn_canceled)
        self._event_bus.unsubscribe(NoticePosted, self._on_notice_posted)
        self._event_bus.unsubscribe(ReviewStateChanged, self._on_review_state_changed)

        self._subscribed = False
        LOGGER.debug("ChatPanelUpdater: disposed")

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    def _create_message(self, role: str, content: str) -> Any:
        """Create a ChatMessage using the factory or import."""
        if self._message_factory is not None:
            return self._message_factory(role=role, content=content)

        from ...chat.chat_panel import ChatMessage

        return ChatMessage(role=role, content=content)

    def _on_ai_turn_started(self, event: "AITurnStarted") -> None:
        """Handle AITurnStarted events."""
        # The chat panel's composer should already show the prompt
        # This handler can be used to set additional state if needed
        LOGGER.debug(
            "ChatPanelUpdater: AI turn started, turn_id=%s",
            event.turn_id,
        )

    def _on_ai_turn_stream_chunk(self, event: "AITurnStreamChunk") -> None:
        """Handle AITurnStreamChunk events."""
        if not event.content:
            return

        message = self._create_message(role="assistant", content=event.content)
        self._chat_panel.append_ai_message(message, streaming=True)

    def _on_ai_turn_completed(self, event: "AITurnCompleted") -> None:
        """Handle AITurnCompleted events."""
        # Finalize the streaming message (pass empty content to close stream)
        message = self._create_message(role="assistant", content="")
        self._chat_panel.append_ai_message(message, streaming=False)

        LOGGER.debug(
            "ChatPanelUpdater: AI turn completed, turn_id=%s, edit_count=%d",
            event.turn_id,
            event.edit_count,
        )

    def _on_ai_turn_failed(self, event: "AITurnFailed") -> None:
        """Handle AITurnFailed events."""
        error_message = f"AI turn failed: {event.error}"
        message = self._create_message(role="assistant", content=error_message)
        self._chat_panel.append_ai_message(message)

        LOGGER.debug(
            "ChatPanelUpdater: AI turn failed, turn_id=%s, error=%s",
            event.turn_id,
            event.error,
        )

    def _on_ai_turn_canceled(self, event: "AITurnCanceled") -> None:
        """Handle AITurnCanceled events."""
        message = self._create_message(
            role="assistant",
            content="AI turn was canceled.",
        )
        self._chat_panel.append_ai_message(message)

        LOGGER.debug(
            "ChatPanelUpdater: AI turn canceled, turn_id=%s",
            event.turn_id,
        )

    def _on_notice_posted(self, event: "NoticePosted") -> None:
        """Handle NoticePosted events."""
        if not event.message.strip():
            return

        message = self._create_message(role="assistant", content=event.message)
        self._chat_panel.append_ai_message(message)

    def _on_review_state_changed(self, event: "ReviewStateChanged") -> None:
        """Handle ReviewStateChanged events for guardrail state."""
        if event.ready and event.edit_count > 0:
            # Review ready - show pending state
            edit_label = "edit" if event.edit_count == 1 else "edits"
            detail = f"{event.edit_count} {edit_label} pending review"
            self._chat_panel.set_guardrail_state("pending", detail=detail)
        else:
            # No pending review
            self._chat_panel.set_guardrail_state("idle")


__all__ = [
    "StatusBarProtocol",
    "ChatPanelProtocol",
    "StatusBarUpdater",
    "ChatPanelUpdater",
]
