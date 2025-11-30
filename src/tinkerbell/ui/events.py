"""Event bus infrastructure for decoupled UI component communication.

This module provides the foundation for the event-driven UI architecture,
allowing components to communicate without direct dependencies.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
    TYPE_CHECKING,
)
from weakref import WeakMethod, ref

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from typing import DefaultDict

logger = logging.getLogger(__name__)

# Type variable for event types
E = TypeVar("E", bound="Event")

# Handler type: a callable that takes an event and returns None
Handler = Callable[[E], None]


@dataclass(slots=True)
class Event:
    """Base class for all events in the system.

    All event classes should inherit from this base class and use
    the @dataclass decorator with slots=True for memory efficiency.

    Example::

        @dataclass(slots=True)
        class DocumentOpened(Event):
            tab_id: str
            document_id: str
            path: Path | None = None
    """

    pass


# High-frequency event types that should not log each publish
_QUIET_EVENT_TYPES: set[type] = set()


# =============================================================================
# Document Events (WS1.2)
# =============================================================================


@dataclass(slots=True)
class DocumentCreated(Event):
    """Emitted when a new document tab is created.

    Attributes:
        tab_id: The unique identifier of the tab containing the document.
        document_id: The unique identifier of the document itself.
    """

    tab_id: str
    document_id: str


@dataclass(slots=True)
class DocumentOpened(Event):
    """Emitted when a document is opened from disk.

    Attributes:
        tab_id: The unique identifier of the tab containing the document.
        document_id: The unique identifier of the document itself.
        path: The filesystem path the document was opened from, if any.
    """

    tab_id: str
    document_id: str
    path: str | None = None


@dataclass(slots=True)
class DocumentClosed(Event):
    """Emitted when a document tab is closed.

    Attributes:
        tab_id: The unique identifier of the closed tab.
        document_id: The unique identifier of the closed document.
    """

    tab_id: str
    document_id: str


@dataclass(slots=True)
class DocumentSaved(Event):
    """Emitted when a document is saved to disk.

    Attributes:
        tab_id: The unique identifier of the tab containing the document.
        document_id: The unique identifier of the document itself.
        path: The filesystem path the document was saved to.
    """

    tab_id: str
    document_id: str
    path: str


@dataclass(slots=True)
class DocumentModified(Event):
    """Emitted when a document's content changes.

    This event is fired whenever the document text is modified,
    providing version tracking information for change detection.

    Attributes:
        tab_id: The unique identifier of the tab containing the document.
        document_id: The unique identifier of the document itself.
        version_id: The incremented version number after the modification.
        content_hash: A hash of the document content for change detection.
    """

    tab_id: str
    document_id: str
    version_id: int
    content_hash: str


@dataclass(slots=True)
class ActiveTabChanged(Event):
    """Emitted when the active (focused) tab changes.

    Attributes:
        tab_id: The unique identifier of the newly active tab.
        document_id: The unique identifier of the document in the active tab.
    """

    tab_id: str
    document_id: str


# =============================================================================
# AI Turn Events (WS1.3)
# =============================================================================


@dataclass(slots=True)
class AITurnStarted(Event):
    """Emitted when an AI turn begins processing.

    Attributes:
        turn_id: The unique identifier for this AI turn (e.g., "turn-1").
        prompt: The user prompt that initiated the AI turn.
    """

    turn_id: str
    prompt: str


@dataclass(slots=True)
class AITurnStreamChunk(Event):
    """Emitted when a streaming chunk is received from the AI.

    This event is fired for each piece of content received during
    streaming, allowing real-time UI updates.

    Attributes:
        turn_id: The unique identifier of the AI turn.
        content: The text content of this streaming chunk.
    """

    turn_id: str
    content: str


# Register high-frequency event types (done after class definition)
_QUIET_EVENT_TYPES.add(AITurnStreamChunk)


@dataclass(slots=True)
class AITurnToolExecuted(Event):
    """Emitted when a tool is executed during an AI turn.

    Attributes:
        turn_id: The unique identifier of the AI turn.
        tool_name: The name of the tool that was executed.
        tool_call_id: The unique identifier for this tool call.
        arguments: JSON string of the tool arguments.
        result: The result content from the tool (if completed).
        success: Whether the tool execution succeeded.
        duration_ms: Execution time in milliseconds.
    """

    turn_id: str
    tool_name: str
    tool_call_id: str
    arguments: str = ""
    result: str = ""
    success: bool = True
    duration_ms: float = 0.0


@dataclass(slots=True)
class AITurnCompleted(Event):
    """Emitted when an AI turn completes successfully.

    Attributes:
        turn_id: The unique identifier of the completed AI turn.
        success: Whether the turn completed without errors.
        edit_count: The number of edits applied during this turn.
        response_text: The final response text from the AI.
    """

    turn_id: str
    success: bool
    edit_count: int
    response_text: str


@dataclass(slots=True)
class AITurnFailed(Event):
    """Emitted when an AI turn fails due to an error.

    Attributes:
        turn_id: The unique identifier of the failed AI turn.
        error: A description of the error that occurred.
    """

    turn_id: str
    error: str


@dataclass(slots=True)
class AITurnCanceled(Event):
    """Emitted when an AI turn is canceled by the user.

    Attributes:
        turn_id: The unique identifier of the canceled AI turn.
    """

    turn_id: str


# =============================================================================
# Edit & Review Events (WS1.4)
# =============================================================================


@dataclass(slots=True)
class EditApplied(Event):
    """Emitted when an edit is successfully applied to a document.

    Attributes:
        tab_id: The unique identifier of the tab containing the document.
        document_id: The unique identifier of the document.
        edit_id: A unique identifier for this specific edit.
        action: The type of edit action (e.g., "patch", "replace", "insert").
        range: The affected text range as a tuple (start, end).
        diff: A unified diff representation of the change.
    """

    tab_id: str
    document_id: str
    edit_id: str
    action: str
    range: tuple[int, int]
    diff: str


@dataclass(slots=True)
class EditFailed(Event):
    """Emitted when an edit fails to apply to a document.

    Attributes:
        tab_id: The unique identifier of the tab containing the document.
        document_id: The unique identifier of the document.
        action: The type of edit action that was attempted.
        reason: A description of why the edit failed.
    """

    tab_id: str
    document_id: str
    action: str
    reason: str


@dataclass(slots=True)
class ReviewStateChanged(Event):
    """Emitted when the review state changes for an AI turn.

    This event is fired when edits from an AI turn are ready for
    review, or when the review state otherwise changes.

    Attributes:
        turn_id: The unique identifier of the AI turn being reviewed.
        ready: Whether the review is ready for user acceptance/rejection.
        edit_count: The total number of edits pending review.
        tabs_affected: The number of tabs affected by the pending edits.
    """

    turn_id: str
    ready: bool
    edit_count: int
    tabs_affected: int


@dataclass(slots=True)
class ReviewAccepted(Event):
    """Emitted when the user accepts pending review edits.

    Attributes:
        turn_id: The unique identifier of the accepted AI turn.
        tabs: A tuple of tab IDs that were affected by the accepted edits.
    """

    turn_id: str
    tabs: tuple[str, ...]


@dataclass(slots=True)
class ReviewRejected(Event):
    """Emitted when the user rejects pending review edits.

    Attributes:
        turn_id: The unique identifier of the rejected AI turn.
        tabs: A tuple of tab IDs that were affected by the rejected edits.
    """

    turn_id: str
    tabs: tuple[str, ...]


# =============================================================================
# UI Events (WS1.5)
# =============================================================================


@dataclass(slots=True)
class StatusMessage(Event):
    """Emitted to display a message in the status bar.

    Attributes:
        message: The text message to display in the status bar.
        timeout_ms: Duration in milliseconds to show the message.
                   Use 0 for persistent messages, or a positive value
                   for auto-dismissing messages.
    """

    message: str
    timeout_ms: int = 0


@dataclass(slots=True)
class NoticePosted(Event):
    """Emitted when a notice should be shown to the user.

    Notices are typically shown in the chat panel as system messages
    or as non-modal notifications to provide feedback on operations.

    Attributes:
        message: The notice text to display to the user.
    """

    message: str


@dataclass(slots=True)
class WindowTitleChanged(Event):
    """Emitted when the window title should be updated.

    Attributes:
        title: The new title for the application window.
    """

    title: str


@dataclass(slots=True)
class EditorLockChanged(Event):
    """Emitted when the editor lock state changes.

    This event is fired when the editor is locked (e.g., during AI turns)
    or unlocked to allow user editing.

    Attributes:
        locked: Whether the editor is currently locked.
        reason: A human-readable reason for the lock state.
                Common values: "AI_TURN", "REVIEW_PENDING", "TRANSACTION", or empty for unlocked.
    """

    locked: bool
    reason: str


# =============================================================================
# Infrastructure Events (WS1.6)
# =============================================================================


@dataclass(slots=True)
class SettingsChanged(Event):
    """Emitted when application settings are modified.

    This event is fired when settings are updated, allowing components
    to react to configuration changes without polling.

    Attributes:
        settings: A mapping of the changed settings. The exact structure
                  depends on the settings implementation.
    """

    settings: dict[str, Any]


@dataclass(slots=True)
class EmbeddingStateChanged(Event):
    """Emitted when the embedding system state changes.

    This event is fired when the embedding backend initializes, becomes
    ready, encounters an error, or is disabled.

    Attributes:
        backend: The name of the embedding backend (e.g., "langchain",
                "sentence-transformers", "openai", "disabled").
        status: The current status (e.g., "ready", "error", "disabled",
               "unavailable", "initializing").
        detail: Additional detail about the state (e.g., error message
               or model name).
    """

    backend: str
    status: str
    detail: str | None = None


@dataclass(slots=True)
class OutlineUpdated(Event):
    """Emitted when a document outline is updated.

    This event is fired when the outline builder finishes processing
    a document, making outline data available for navigation and context.

    Attributes:
        document_id: The unique identifier of the document.
        outline_hash: A hash of the outline content for change detection.
        node_count: The number of nodes in the outline tree.
    """

    document_id: str
    outline_hash: str
    node_count: int


@dataclass(slots=True)
class TelemetryEvent(Event):
    """Emitted to record telemetry data for monitoring and debugging.

    This event is used to forward telemetry from various subsystems
    to centralized logging/analytics.

    Attributes:
        name: The name/type of the telemetry event (e.g., "embedding.cache.hit",
             "ai.turn.completed").
        payload: A mapping of telemetry data associated with the event.
    """

    name: str
    payload: dict[str, Any]


@dataclass(slots=True)
class WorkspaceRestored(Event):
    """Emitted when a workspace session is restored from saved state.

    This event is fired by RestoreWorkspaceUseCase after successfully
    restoring tabs and document state from a previous session.

    Attributes:
        tab_count: The number of tabs that were restored.
        active_tab_id: The unique identifier of the active tab after restore,
                      or None if no tabs were restored.
    """

    tab_count: int
    active_tab_id: str | None


class EventBus(Generic[E]):
    """A typed publish-subscribe event bus for decoupled communication.

    The EventBus allows components to subscribe to specific event types
    and receive notifications when those events are published. Handlers
    are stored as weak references where possible to prevent memory leaks.

    Example::

        bus = EventBus()

        def on_document_opened(event: DocumentOpened) -> None:
            print(f"Opened: {event.path}")

        bus.subscribe(DocumentOpened, on_document_opened)
        bus.publish(DocumentOpened(tab_id="1", document_id="doc1", path=Path("test.txt")))
        bus.unsubscribe(DocumentOpened, on_document_opened)

    Thread Safety:
        This implementation is NOT thread-safe. All operations should be
        performed from the main thread (Qt's event loop thread).

    Attributes:
        _handlers: Mapping from event type to list of handler references.
    """

    __slots__ = ("_handlers",)

    def __init__(self) -> None:
        """Initialize an empty event bus."""
        self._handlers: DefaultDict[type[Event], list[_HandlerRef]] = defaultdict(list)

    def subscribe(self, event_type: type[E], handler: Handler[E]) -> None:
        """Register a handler to receive events of the specified type.

        The handler will be called whenever an event of the given type
        (or a subclass) is published. Handlers are stored as weak references
        where possible (for bound methods), allowing automatic cleanup when
        the handler's owner is garbage collected.

        Args:
            event_type: The class of events to subscribe to.
            handler: A callable that will be invoked with the event.

        Note:
            Subscribing the same handler multiple times will result in
            multiple invocations when an event is published.
        """
        handler_ref = _HandlerRef.create(handler)
        self._handlers[event_type].append(handler_ref)
        logger.debug(
            "Subscribed handler %s to event type %s",
            _handler_name(handler),
            event_type.__name__,
        )

    def unsubscribe(self, event_type: type[E], handler: Handler[E]) -> None:
        """Remove a previously registered handler.

        If the handler was registered multiple times, only the first
        occurrence is removed.

        Args:
            event_type: The event type the handler was registered for.
            handler: The handler to remove.

        Note:
            This method is safe to call even if the handler was never
            subscribed or has already been unsubscribed.
        """
        handlers = self._handlers.get(event_type)
        if handlers is None:
            return

        # Find and remove the handler by identity comparison
        for i, handler_ref in enumerate(handlers):
            if handler_ref.matches(handler):
                handlers.pop(i)
                logger.debug(
                    "Unsubscribed handler %s from event type %s",
                    _handler_name(handler),
                    event_type.__name__,
                )
                return

    def publish(self, event: E) -> None:
        """Broadcast an event to all registered handlers.

        Handlers are invoked synchronously in the order they were
        registered. If a handler raises an exception, it is logged
        and remaining handlers continue to be invoked.

        Args:
            event: The event instance to publish.
        """
        event_type = type(event)
        handlers = self._handlers.get(event_type)
        is_quiet = event_type in _QUIET_EVENT_TYPES

        if handlers is None:
            # Only log for non-quiet events
            if not is_quiet:
                logger.debug("No handlers for event type %s", event_type.__name__)
            return

        # Only log for non-quiet events (skip high-frequency streaming events)
        if not is_quiet:
            logger.debug(
                "Publishing %s to %d handler(s)",
                event_type.__name__,
                len(handlers),
            )

        # Collect dead references for cleanup
        dead_indices: list[int] = []

        for i, handler_ref in enumerate(handlers):
            handler = handler_ref.resolve()
            if handler is None:
                # Handler was garbage collected
                dead_indices.append(i)
                continue

            try:
                handler(event)
            except Exception:
                logger.exception(
                    "Handler %s raised exception for event %s",
                    _handler_name(handler),
                    event_type.__name__,
                )

        # Clean up dead references (in reverse order to preserve indices)
        for i in reversed(dead_indices):
            handlers.pop(i)

    def clear(self) -> None:
        """Remove all registered handlers.

        Useful for cleanup during application shutdown or testing.
        """
        self._handlers.clear()
        logger.debug("Cleared all event handlers")

    def handler_count(self, event_type: type[E] | None = None) -> int:
        """Return the number of registered handlers.

        Args:
            event_type: If provided, return count for that event type only.
                       If None, return total count across all event types.

        Returns:
            The number of registered handlers.
        """
        if event_type is not None:
            return len(self._handlers.get(event_type, []))
        return sum(len(handlers) for handlers in self._handlers.values())


class _HandlerRef:
    """Wrapper for handler references supporting both weak and strong refs.

    For bound methods, we use WeakMethod to allow automatic cleanup when
    the object is garbage collected. For regular functions and lambdas,
    we use a strong reference since they typically have module-level
    lifetime or are explicitly managed.

    Attributes:
        _ref: Either a WeakMethod, weak ref, or the handler itself.
        _is_weak: True if using weak reference semantics.
    """

    __slots__ = ("_ref", "_is_weak")

    def __init__(self, handler_ref: WeakMethod | ref | Handler, is_weak: bool) -> None:
        """Initialize a handler reference.

        Args:
            handler_ref: The reference to the handler.
            is_weak: Whether this is a weak reference.
        """
        self._ref = handler_ref
        self._is_weak = is_weak

    @classmethod
    def create(cls, handler: Handler) -> _HandlerRef:
        """Create a handler reference, using weak refs where possible.

        Args:
            handler: The handler callable.

        Returns:
            A new _HandlerRef instance.
        """
        # Check if this is a bound method
        if hasattr(handler, "__self__") and hasattr(handler, "__func__"):
            # Bound method - use WeakMethod
            try:
                return cls(WeakMethod(handler), is_weak=True)
            except TypeError:
                # Some callables can't be weakly referenced
                pass

        # Regular function or lambda - use strong reference
        return cls(handler, is_weak=False)

    def resolve(self) -> Handler | None:
        """Resolve the reference to the actual handler.

        Returns:
            The handler callable, or None if it was garbage collected.
        """
        if not self._is_weak:
            return self._ref  # type: ignore[return-value]

        # Weak reference - resolve it
        handler = self._ref()  # type: ignore[operator]
        return handler

    def matches(self, handler: Handler) -> bool:
        """Check if this reference points to the given handler.

        Args:
            handler: The handler to compare against.

        Returns:
            True if this reference points to the same handler.
        """
        resolved = self.resolve()
        if resolved is None:
            return False
        return resolved == handler


def _handler_name(handler: Handler) -> str:
    """Get a human-readable name for a handler for logging purposes."""
    if hasattr(handler, "__self__") and hasattr(handler, "__func__"):
        # Bound method
        cls_name = type(handler.__self__).__name__
        return f"{cls_name}.{handler.__func__.__name__}"
    if hasattr(handler, "__name__"):
        return handler.__name__
    return repr(handler)


__all__ = [
    # Core infrastructure
    "Event",
    "EventBus",
    "Handler",
    # Document events (WS1.2)
    "DocumentCreated",
    "DocumentOpened",
    "DocumentClosed",
    "DocumentSaved",
    "DocumentModified",
    "ActiveTabChanged",
    # AI turn events (WS1.3)
    "AITurnStarted",
    "AITurnStreamChunk",
    "AITurnToolExecuted",
    "AITurnCompleted",
    "AITurnFailed",
    "AITurnCanceled",
    # Edit & review events (WS1.4)
    "EditApplied",
    "EditFailed",
    "ReviewStateChanged",
    "ReviewAccepted",
    "ReviewRejected",
    # UI events (WS1.5)
    "StatusMessage",
    "NoticePosted",
    "WindowTitleChanged",
    "EditorLockChanged",
    # Infrastructure events (WS1.6)
    "SettingsChanged",
    "EmbeddingStateChanged",
    "OutlineUpdated",
    "TelemetryEvent",
    "WorkspaceRestored",
]
