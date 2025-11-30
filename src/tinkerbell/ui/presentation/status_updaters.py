"""Status updater components for the presentation layer.

This module provides reactive updater components that subscribe to
domain events and update UI widgets accordingly.

Classes:
    StatusBarUpdater: Updates status bar widgets based on events
    ChatPanelUpdater: Updates chat panel based on AI events
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Protocol

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ..events import (
        AITurnCanceled,
        AITurnCompleted,
        AITurnFailed,
        AITurnStarted,
        AITurnStreamChunk,
        AITurnToolExecuted,
        DocumentCreated,
        DocumentModified,
        DocumentSaved,
        EditApplied,
        EditorLockChanged,
        EmbeddingStateChanged,
        EventBus,
        NoticePosted,
        OutlineUpdated,
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

    def set_ai_state(self, state: str) -> None:
        """Update the AI state indicator (Idle, Thinking, Streaming)."""
        ...

    def set_subagent_status(self, status: str | None, *, detail: str | None = None) -> None:
        """Update the subagent status indicator."""
        ...

    def set_outline_status(self, status: str | None, *, tooltip: str | None = None) -> None:
        """Update the outline status indicator."""
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

    def set_ai_running(self, active: bool) -> None:
        """Toggle AI running state (affects submit/stop button)."""
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
        - DocumentModified: Updates autosave indicator to "Unsaved"
        - DocumentSaved: Updates autosave indicator to "Saved"
        - OutlineUpdated: Updates outline status indicator
        - AITurnStarted: Sets AI state to "Thinking"
        - AITurnStreamChunk: Sets AI state to "Streaming"
        - AITurnToolExecuted: Updates subagent status for analyze/transform tools
        - AITurnCompleted: Sets AI state to "Idle"
        - AITurnFailed: Sets AI state to "Idle"
        - AITurnCanceled: Sets AI state to "Idle"

    Example:
        updater = StatusBarUpdater(status_bar, event_bus)
        # Status bar now automatically updates when events are published
        # ...
        updater.dispose()  # Clean up subscriptions
    """

    __slots__ = (
        "_status_bar",
        "_event_bus",
        "_subscribed",
        "_streaming",
        "_accept_callback",
        "_reject_callback",
    )

    def __init__(
        self,
        status_bar: StatusBarProtocol,
        event_bus: "EventBus",
        *,
        accept_callback: Callable[[], None] | None = None,
        reject_callback: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the updater and subscribe to events.

        Args:
            status_bar: The status bar widget to update.
            event_bus: The event bus to subscribe to.
            accept_callback: Optional callback for accept button clicks.
            reject_callback: Optional callback for reject button clicks.
        """
        self._status_bar = status_bar
        self._event_bus = event_bus
        self._subscribed = True
        self._streaming = False  # Track if we're streaming to avoid repeated updates
        self._accept_callback = accept_callback
        self._reject_callback = reject_callback

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
            AITurnToolExecuted,
            DocumentModified,
            DocumentSaved,
            EditorLockChanged,
            EmbeddingStateChanged,
            OutlineUpdated,
            ReviewStateChanged,
            StatusMessage,
        )

        self._event_bus.subscribe(StatusMessage, self._on_status_message)
        self._event_bus.subscribe(EditorLockChanged, self._on_editor_lock_changed)
        self._event_bus.subscribe(ReviewStateChanged, self._on_review_state_changed)
        self._event_bus.subscribe(EmbeddingStateChanged, self._on_embedding_state_changed)
        self._event_bus.subscribe(DocumentModified, self._on_document_modified)
        self._event_bus.subscribe(DocumentSaved, self._on_document_saved)
        self._event_bus.subscribe(OutlineUpdated, self._on_outline_updated)
        # AI turn events for AI state indicator
        self._event_bus.subscribe(AITurnStarted, self._on_ai_turn_started)
        self._event_bus.subscribe(AITurnStreamChunk, self._on_ai_turn_stream_chunk)
        self._event_bus.subscribe(AITurnToolExecuted, self._on_ai_turn_tool_executed)
        self._event_bus.subscribe(AITurnCompleted, self._on_ai_turn_completed)
        self._event_bus.subscribe(AITurnFailed, self._on_ai_turn_failed)
        self._event_bus.subscribe(AITurnCanceled, self._on_ai_turn_canceled)

        LOGGER.debug("StatusBarUpdater: subscribed to events")

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
            AITurnToolExecuted,
            DocumentModified,
            DocumentSaved,
            EditorLockChanged,
            EmbeddingStateChanged,
            OutlineUpdated,
            ReviewStateChanged,
            StatusMessage,
        )

        self._event_bus.unsubscribe(StatusMessage, self._on_status_message)
        self._event_bus.unsubscribe(EditorLockChanged, self._on_editor_lock_changed)
        self._event_bus.unsubscribe(ReviewStateChanged, self._on_review_state_changed)
        self._event_bus.unsubscribe(EmbeddingStateChanged, self._on_embedding_state_changed)
        self._event_bus.unsubscribe(DocumentModified, self._on_document_modified)
        self._event_bus.unsubscribe(DocumentSaved, self._on_document_saved)
        self._event_bus.unsubscribe(OutlineUpdated, self._on_outline_updated)
        # AI turn events
        self._event_bus.unsubscribe(AITurnStarted, self._on_ai_turn_started)
        self._event_bus.unsubscribe(AITurnStreamChunk, self._on_ai_turn_stream_chunk)
        self._event_bus.unsubscribe(AITurnToolExecuted, self._on_ai_turn_tool_executed)
        self._event_bus.unsubscribe(AITurnCompleted, self._on_ai_turn_completed)
        self._event_bus.unsubscribe(AITurnFailed, self._on_ai_turn_failed)
        self._event_bus.unsubscribe(AITurnCanceled, self._on_ai_turn_canceled)

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
            # Show review controls with summary and callbacks
            edit_label = "edit" if event.edit_count == 1 else "edits"
            tab_label = "tab" if event.tabs_affected == 1 else "tabs"
            summary = f"{event.edit_count} {edit_label} across {event.tabs_affected} {tab_label}"
            self._status_bar.set_review_state(
                summary,
                accept_callback=self._accept_callback,
                reject_callback=self._reject_callback,
            )
        else:
            # Hide review controls
            self._status_bar.clear_review_state()

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

    def _on_ai_turn_started(self, event: "AITurnStarted") -> None:
        """Handle AITurnStarted events."""
        self._streaming = False
        self._status_bar.set_ai_state("Thinking")

    def _on_ai_turn_stream_chunk(self, event: "AITurnStreamChunk") -> None:
        """Handle AITurnStreamChunk events."""
        # Only update once when streaming starts to avoid repeated updates
        if not self._streaming:
            self._streaming = True
            self._status_bar.set_ai_state("Streaming")

    def _on_ai_turn_tool_executed(self, event: "AITurnToolExecuted") -> None:
        """Handle AITurnToolExecuted events for subagent status."""
        # Subagent tools that should show status
        subagent_tools = {"analyze_document", "transform_document"}
        
        if event.tool_name in subagent_tools:
            # Check if this is the start (result is "(running…)") or completion
            if event.result == "(running…)":
                # Tool is starting
                tool_label = event.tool_name.replace("_", " ").title()
                self._status_bar.set_subagent_status("Running", detail=tool_label)
            else:
                # Tool completed - show briefly then reset
                status = "Done" if event.success else "Failed"
                self._status_bar.set_subagent_status(status)

    def _on_ai_turn_completed(self, event: "AITurnCompleted") -> None:
        """Handle AITurnCompleted events."""
        self._streaming = False
        self._status_bar.set_ai_state("Idle")
        self._status_bar.set_subagent_status(None)  # Reset to idle

    def _on_ai_turn_failed(self, event: "AITurnFailed") -> None:
        """Handle AITurnFailed events."""
        self._streaming = False
        self._status_bar.set_ai_state("Idle")
        self._status_bar.set_subagent_status(None)  # Reset to idle

    def _on_ai_turn_canceled(self, event: "AITurnCanceled") -> None:
        """Handle AITurnCanceled events."""
        self._streaming = False
        self._status_bar.set_ai_state("Idle")
        self._status_bar.set_subagent_status(None)  # Reset to idle

    def _on_document_saved(self, event: "DocumentSaved") -> None:
        """Handle DocumentSaved events."""
        # Mark document as saved
        self._status_bar.set_autosave_state("Saved", detail="")

    def _on_outline_updated(self, event: "OutlineUpdated") -> None:
        """Handle OutlineUpdated events."""
        # Update outline status with node count
        node_label = "node" if event.node_count == 1 else "nodes"
        tooltip = f"{event.node_count} {node_label}"
        self._status_bar.set_outline_status("Ready", tooltip=tooltip)


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
        )

        self._event_bus.subscribe(AITurnStarted, self._on_ai_turn_started)
        self._event_bus.subscribe(AITurnStreamChunk, self._on_ai_turn_stream_chunk)
        self._event_bus.subscribe(AITurnCompleted, self._on_ai_turn_completed)
        self._event_bus.subscribe(AITurnFailed, self._on_ai_turn_failed)
        self._event_bus.subscribe(AITurnCanceled, self._on_ai_turn_canceled)
        self._event_bus.subscribe(NoticePosted, self._on_notice_posted)

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
        )

        self._event_bus.unsubscribe(AITurnStarted, self._on_ai_turn_started)
        self._event_bus.unsubscribe(AITurnStreamChunk, self._on_ai_turn_stream_chunk)
        self._event_bus.unsubscribe(AITurnCompleted, self._on_ai_turn_completed)
        self._event_bus.unsubscribe(AITurnFailed, self._on_ai_turn_failed)
        self._event_bus.unsubscribe(AITurnCanceled, self._on_ai_turn_canceled)
        self._event_bus.unsubscribe(NoticePosted, self._on_notice_posted)

        self._subscribed = False
        LOGGER.debug("ChatPanelUpdater: disposed")

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    def _create_message(self, role: str, content: str) -> Any:
        """Create a ChatMessage using the factory or import."""
        if self._message_factory is not None:
            return self._message_factory(role=role, content=content)

        from .chat.chat_panel import ChatMessage

        return ChatMessage(role=role, content=content)

    def _on_ai_turn_started(self, event: "AITurnStarted") -> None:
        """Handle AITurnStarted events."""
        # Set chat panel to running state (changes submit to stop button)
        self._chat_panel.set_ai_running(True)
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
        # Clear running state (changes stop button back to submit)
        self._chat_panel.set_ai_running(False)
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
        # Clear running state
        self._chat_panel.set_ai_running(False)
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
        # Clear running state
        self._chat_panel.set_ai_running(False)
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


class EditTracker:
    """Tracks edits during AI turns and updates managers accordingly.

    This component subscribes to EditApplied and DocumentCreated events
    and coordinates edit tracking between AITurnManager and ReviewManager.

    Events Handled:
        - EditApplied: Increments edit count and records edit for review
        - DocumentCreated: Increments edit count when AI creates a document

    Example:
        tracker = EditTracker(ai_turn_manager, review_manager, event_bus)
        # Tracker now automatically updates managers when edits are applied
        # ...
        tracker.dispose()  # Clean up subscriptions
    """

    __slots__ = (
        "_ai_turn_manager",
        "_review_manager",
        "_document_provider",
        "_event_bus",
        "_subscribed",
    )

    def __init__(
        self,
        ai_turn_manager: Any,
        review_manager: Any,
        event_bus: "EventBus",
        document_provider: Any = None,
    ) -> None:
        """Initialize the edit tracker and subscribe to events.

        Args:
            ai_turn_manager: The AI turn manager to update edit counts.
            review_manager: The review manager to record edits.
            event_bus: The event bus to subscribe to.
            document_provider: Optional provider for document snapshots.
        """
        self._ai_turn_manager = ai_turn_manager
        self._review_manager = review_manager
        self._document_provider = document_provider
        self._event_bus = event_bus
        self._subscribed = True

        # Subscribe to edit events
        self._subscribe()

    def _subscribe(self) -> None:
        """Subscribe to edit events."""
        from ..events import DocumentCreated, EditApplied

        self._event_bus.subscribe(EditApplied, self._on_edit_applied)
        self._event_bus.subscribe(DocumentCreated, self._on_document_created)
        LOGGER.debug("EditTracker: subscribed to EditApplied and DocumentCreated events")

    def dispose(self) -> None:
        """Unsubscribe from all events.

        Call this method to clean up when the tracker is no longer needed.
        """
        if not self._subscribed:
            return

        from ..events import DocumentCreated, EditApplied

        self._event_bus.unsubscribe(EditApplied, self._on_edit_applied)
        self._event_bus.unsubscribe(DocumentCreated, self._on_document_created)

        self._subscribed = False
        LOGGER.debug("EditTracker: disposed")

    def _on_edit_applied(self, event: "EditApplied") -> None:
        """Handle EditApplied events.

        Increments the AI turn manager's edit count and records
        the edit in the review manager.
        """
        # Increment AI turn manager edit count
        if self._ai_turn_manager is not None:
            self._ai_turn_manager.increment_edit_count()
            LOGGER.debug(
                "EditTracker: incremented edit count for tab_id=%s, edit_id=%s",
                event.tab_id,
                event.edit_id,
            )

        # Record edit in review manager for accept/reject workflow
        if self._review_manager is not None:
            from ..models.review_models import PendingEdit

            pending_edit = PendingEdit(
                edit_id=event.edit_id,
                tab_id=event.tab_id,
                action=event.action,
                range=event.range,
                diff=event.diff,
            )

            # Get document snapshot if provider available
            # Note: This is the post-edit state; future work may need pre-edit snapshots
            document_snapshot = None
            if self._document_provider is not None:
                try:
                    document_snapshot = self._document_provider(event.tab_id)
                except Exception:
                    LOGGER.debug(
                        "EditTracker: failed to get document snapshot",
                        exc_info=True,
                    )

            # Record the edit (will be no-op if no pending review)
            if document_snapshot is not None:
                self._review_manager.record_edit(
                    event.tab_id,
                    document_snapshot,
                    pending_edit,
                )
            else:
                # Still need to increment the review manager's count
                # even without full snapshot support
                pending = self._review_manager.pending_review
                if pending is not None:
                    pending.total_edit_count += 1
                    LOGGER.debug(
                        "EditTracker: incremented review total_edit_count to %d",
                        pending.total_edit_count,
                    )

    def _on_document_created(self, event: "DocumentCreated") -> None:
        """Handle DocumentCreated events.

        When the AI creates a new document, it counts as an edit
        for the accept/reject workflow. We also ensure a review session
        is created for the new tab so that accept/reject workflows
        properly track the affected tabs.
        """
        # Increment AI turn manager edit count
        if self._ai_turn_manager is not None:
            self._ai_turn_manager.increment_edit_count()
            LOGGER.debug(
                "EditTracker: incremented edit count for created document tab_id=%s",
                event.tab_id,
            )

        # Record the document creation as an edit in review manager
        if self._review_manager is not None:
            pending = self._review_manager.pending_review
            if pending is not None:
                from ..models.review_models import PendingEdit

                # Create a PendingEdit representing the document creation
                pending_edit = PendingEdit(
                    edit_id=f"create-{event.tab_id}",
                    tab_id=event.tab_id,
                    action="create_document",
                    range=(0, 0),
                    diff=f"Created new document (tab_id={event.tab_id})",
                )

                # Get document snapshot if provider available
                document_snapshot = None
                if self._document_provider is not None:
                    try:
                        document_snapshot = self._document_provider(event.tab_id)
                    except Exception:
                        LOGGER.debug(
                            "EditTracker: failed to get document snapshot for created document",
                            exc_info=True,
                        )

                # Record the edit - this ensures a session is created for the tab
                if document_snapshot is not None:
                    self._review_manager.record_edit(
                        event.tab_id,
                        document_snapshot,
                        pending_edit,
                    )
                    LOGGER.debug(
                        "EditTracker: recorded create_document edit for tab_id=%s",
                        event.tab_id,
                    )
                else:
                    # Fallback: increment count even without full session support
                    pending.total_edit_count += 1
                    LOGGER.debug(
                        "EditTracker: incremented review total_edit_count to %d for created document (no snapshot)",
                        pending.total_edit_count,
                    )


__all__ = [
    "StatusBarProtocol",
    "ChatPanelProtocol",
    "StatusBarUpdater",
    "ChatPanelUpdater",
    "EditTracker",
]
