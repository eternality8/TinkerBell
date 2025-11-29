"""Unit tests for :mod:`tinkerbell.ui.events`."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import pytest

from tinkerbell.ui.events import Event, EventBus, Handler


@dataclass(slots=True)
class SampleEvent(Event):
    """A sample event for testing."""

    message: str
    value: int = 0


@dataclass(slots=True)
class AnotherEvent(Event):
    """Another event type for testing isolation."""

    data: str


class TestEvent:
    """Tests for the Event base class."""

    def test_event_is_dataclass(self) -> None:
        """Event base class can be used as a dataclass."""
        event = SampleEvent(message="hello", value=42)
        assert event.message == "hello"
        assert event.value == 42

    def test_event_uses_slots(self) -> None:
        """Event dataclasses should use slots for memory efficiency."""
        event = SampleEvent(message="test")
        assert hasattr(event, "__slots__")


class TestEventBusSubscription:
    """Tests for EventBus subscription functionality."""

    def test_subscribe_adds_handler(self) -> None:
        """subscribe adds a handler for the event type."""
        bus: EventBus[Event] = EventBus()
        received: list[SampleEvent] = []

        def handler(event: SampleEvent) -> None:
            received.append(event)

        bus.subscribe(SampleEvent, handler)
        assert bus.handler_count(SampleEvent) == 1

    def test_subscribe_multiple_handlers(self) -> None:
        """Multiple handlers can subscribe to the same event type."""
        bus: EventBus[Event] = EventBus()
        received1: list[SampleEvent] = []
        received2: list[SampleEvent] = []

        bus.subscribe(SampleEvent, lambda e: received1.append(e))
        bus.subscribe(SampleEvent, lambda e: received2.append(e))

        assert bus.handler_count(SampleEvent) == 2

    def test_subscribe_same_handler_twice(self) -> None:
        """Subscribing the same handler twice results in two registrations."""
        bus: EventBus[Event] = EventBus()
        received: list[SampleEvent] = []

        def handler(event: SampleEvent) -> None:
            received.append(event)

        bus.subscribe(SampleEvent, handler)
        bus.subscribe(SampleEvent, handler)
        assert bus.handler_count(SampleEvent) == 2

    def test_subscribe_different_event_types(self) -> None:
        """Handlers for different event types are tracked separately."""
        bus: EventBus[Event] = EventBus()

        bus.subscribe(SampleEvent, lambda e: None)
        bus.subscribe(AnotherEvent, lambda e: None)

        assert bus.handler_count(SampleEvent) == 1
        assert bus.handler_count(AnotherEvent) == 1
        assert bus.handler_count() == 2


class TestEventBusUnsubscription:
    """Tests for EventBus unsubscription functionality."""

    def test_unsubscribe_removes_handler(self) -> None:
        """unsubscribe removes the handler from the event type."""
        bus: EventBus[Event] = EventBus()

        def handler(event: SampleEvent) -> None:
            pass

        bus.subscribe(SampleEvent, handler)
        assert bus.handler_count(SampleEvent) == 1

        bus.unsubscribe(SampleEvent, handler)
        assert bus.handler_count(SampleEvent) == 0

    def test_unsubscribe_nonexistent_handler_is_safe(self) -> None:
        """unsubscribe is safe when handler was never subscribed."""
        bus: EventBus[Event] = EventBus()

        def handler(event: SampleEvent) -> None:
            pass

        # Should not raise
        bus.unsubscribe(SampleEvent, handler)

    def test_unsubscribe_wrong_event_type_is_safe(self) -> None:
        """unsubscribe is safe when using wrong event type."""
        bus: EventBus[Event] = EventBus()

        def handler(event: SampleEvent) -> None:
            pass

        bus.subscribe(SampleEvent, handler)

        # Should not raise - wrong event type
        bus.unsubscribe(AnotherEvent, handler)  # type: ignore[arg-type]
        assert bus.handler_count(SampleEvent) == 1

    def test_unsubscribe_one_of_duplicate_handlers(self) -> None:
        """unsubscribe removes only one occurrence of duplicate handlers."""
        bus: EventBus[Event] = EventBus()
        received: list[SampleEvent] = []

        def handler(event: SampleEvent) -> None:
            received.append(event)

        bus.subscribe(SampleEvent, handler)
        bus.subscribe(SampleEvent, handler)
        assert bus.handler_count(SampleEvent) == 2

        bus.unsubscribe(SampleEvent, handler)
        assert bus.handler_count(SampleEvent) == 1


class TestEventBusPublish:
    """Tests for EventBus publish functionality."""

    def test_publish_invokes_handlers(self) -> None:
        """publish invokes all subscribed handlers."""
        bus: EventBus[Event] = EventBus()
        received: list[SampleEvent] = []

        def handler(event: SampleEvent) -> None:
            received.append(event)

        bus.subscribe(SampleEvent, handler)

        event = SampleEvent(message="hello", value=42)
        bus.publish(event)

        assert len(received) == 1
        assert received[0] is event

    def test_publish_invokes_handlers_in_order(self) -> None:
        """publish invokes handlers in subscription order."""
        bus: EventBus[Event] = EventBus()
        order: list[int] = []

        bus.subscribe(SampleEvent, lambda e: order.append(1))
        bus.subscribe(SampleEvent, lambda e: order.append(2))
        bus.subscribe(SampleEvent, lambda e: order.append(3))

        bus.publish(SampleEvent(message="test"))

        assert order == [1, 2, 3]

    def test_publish_only_invokes_matching_handlers(self) -> None:
        """publish only invokes handlers for the matching event type."""
        bus: EventBus[Event] = EventBus()
        sample_received: list[SampleEvent] = []
        another_received: list[AnotherEvent] = []

        bus.subscribe(SampleEvent, lambda e: sample_received.append(e))
        bus.subscribe(AnotherEvent, lambda e: another_received.append(e))

        bus.publish(SampleEvent(message="sample"))

        assert len(sample_received) == 1
        assert len(another_received) == 0

    def test_publish_no_handlers_is_safe(self) -> None:
        """publish is safe when no handlers are subscribed."""
        bus: EventBus[Event] = EventBus()

        # Should not raise
        bus.publish(SampleEvent(message="orphan"))

    def test_publish_continues_after_handler_exception(self) -> None:
        """publish continues invoking handlers even after one raises."""
        bus: EventBus[Event] = EventBus()
        received: list[int] = []

        def handler1(event: SampleEvent) -> None:
            received.append(1)

        def handler2(event: SampleEvent) -> None:
            raise ValueError("boom")

        def handler3(event: SampleEvent) -> None:
            received.append(3)

        bus.subscribe(SampleEvent, handler1)
        bus.subscribe(SampleEvent, handler2)
        bus.subscribe(SampleEvent, handler3)

        # Should not raise
        bus.publish(SampleEvent(message="test"))

        # Handler 1 and 3 should still run
        assert received == [1, 3]


class TestEventBusWeakReferences:
    """Tests for EventBus weak reference handling."""

    def test_bound_method_handler_cleaned_up_on_gc(self) -> None:
        """Bound method handlers are cleaned up when owner is garbage collected."""
        bus: EventBus[Event] = EventBus()
        received: list[SampleEvent] = []

        class Subscriber:
            def handle(self, event: SampleEvent) -> None:
                received.append(event)

        subscriber = Subscriber()
        bus.subscribe(SampleEvent, subscriber.handle)
        assert bus.handler_count(SampleEvent) == 1

        # Publish should work
        bus.publish(SampleEvent(message="before gc"))
        assert len(received) == 1

        # Delete the subscriber and force garbage collection
        del subscriber
        gc.collect()

        # The dead handler should be cleaned up on next publish
        bus.publish(SampleEvent(message="after gc"))

        # Only the first message should have been received
        assert len(received) == 1

    def test_function_handler_not_gc_collected(self) -> None:
        """Function handlers are kept via strong reference."""
        bus: EventBus[Event] = EventBus()
        received: list[SampleEvent] = []

        def handler(event: SampleEvent) -> None:
            received.append(event)

        bus.subscribe(SampleEvent, handler)

        # Even after gc, function should remain
        gc.collect()
        bus.publish(SampleEvent(message="test"))

        assert len(received) == 1

    def test_unsubscribe_bound_method(self) -> None:
        """Bound methods can be properly unsubscribed."""
        bus: EventBus[Event] = EventBus()

        class Subscriber:
            def handle(self, event: SampleEvent) -> None:
                pass

        subscriber = Subscriber()
        bus.subscribe(SampleEvent, subscriber.handle)
        assert bus.handler_count(SampleEvent) == 1

        bus.unsubscribe(SampleEvent, subscriber.handle)
        assert bus.handler_count(SampleEvent) == 0


class TestEventBusClear:
    """Tests for EventBus clear functionality."""

    def test_clear_removes_all_handlers(self) -> None:
        """clear removes all handlers for all event types."""
        bus: EventBus[Event] = EventBus()

        bus.subscribe(SampleEvent, lambda e: None)
        bus.subscribe(SampleEvent, lambda e: None)
        bus.subscribe(AnotherEvent, lambda e: None)

        assert bus.handler_count() == 3

        bus.clear()

        assert bus.handler_count() == 0
        assert bus.handler_count(SampleEvent) == 0
        assert bus.handler_count(AnotherEvent) == 0


class TestEventBusHandlerCount:
    """Tests for EventBus handler_count functionality."""

    def test_handler_count_empty_bus(self) -> None:
        """handler_count returns 0 for empty bus."""
        bus: EventBus[Event] = EventBus()
        assert bus.handler_count() == 0
        assert bus.handler_count(SampleEvent) == 0

    def test_handler_count_specific_event_type(self) -> None:
        """handler_count returns count for specific event type."""
        bus: EventBus[Event] = EventBus()

        bus.subscribe(SampleEvent, lambda e: None)
        bus.subscribe(SampleEvent, lambda e: None)
        bus.subscribe(AnotherEvent, lambda e: None)

        assert bus.handler_count(SampleEvent) == 2
        assert bus.handler_count(AnotherEvent) == 1

    def test_handler_count_total(self) -> None:
        """handler_count with no argument returns total across all types."""
        bus: EventBus[Event] = EventBus()

        bus.subscribe(SampleEvent, lambda e: None)
        bus.subscribe(AnotherEvent, lambda e: None)
        bus.subscribe(AnotherEvent, lambda e: None)

        assert bus.handler_count() == 3


class TestIntegration:
    """Integration tests for complete event workflows."""

    def test_full_subscribe_publish_unsubscribe_cycle(self) -> None:
        """Full lifecycle: subscribe, publish, unsubscribe, publish again."""
        bus: EventBus[Event] = EventBus()
        received: list[str] = []

        def handler(event: SampleEvent) -> None:
            received.append(event.message)

        bus.subscribe(SampleEvent, handler)
        bus.publish(SampleEvent(message="first"))
        assert received == ["first"]

        bus.unsubscribe(SampleEvent, handler)
        bus.publish(SampleEvent(message="second"))
        assert received == ["first"]  # Should not have received second

    def test_multiple_event_types_isolation(self) -> None:
        """Different event types are completely isolated."""
        bus: EventBus[Event] = EventBus()
        sample_messages: list[str] = []
        another_data: list[str] = []

        bus.subscribe(SampleEvent, lambda e: sample_messages.append(e.message))
        bus.subscribe(AnotherEvent, lambda e: another_data.append(e.data))

        bus.publish(SampleEvent(message="sample1"))
        bus.publish(AnotherEvent(data="another1"))
        bus.publish(SampleEvent(message="sample2"))

        assert sample_messages == ["sample1", "sample2"]
        assert another_data == ["another1"]


# =============================================================================
# Document Event Tests (WS1.2)
# =============================================================================


from tinkerbell.ui.events import (
    DocumentCreated,
    DocumentOpened,
    DocumentClosed,
    DocumentSaved,
    DocumentModified,
    ActiveTabChanged,
)


class TestDocumentCreated:
    """Tests for the DocumentCreated event."""

    def test_document_created_fields(self) -> None:
        """DocumentCreated stores tab_id and document_id."""
        event = DocumentCreated(tab_id="t1", document_id="doc-abc")
        assert event.tab_id == "t1"
        assert event.document_id == "doc-abc"

    def test_document_created_publish_subscribe(self) -> None:
        """DocumentCreated can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[DocumentCreated] = []

        bus.subscribe(DocumentCreated, lambda e: received.append(e))
        bus.publish(DocumentCreated(tab_id="t1", document_id="doc-123"))

        assert len(received) == 1
        assert received[0].tab_id == "t1"
        assert received[0].document_id == "doc-123"


class TestDocumentOpened:
    """Tests for the DocumentOpened event."""

    def test_document_opened_with_path(self) -> None:
        """DocumentOpened stores tab_id, document_id, and optional path."""
        event = DocumentOpened(
            tab_id="t2",
            document_id="doc-xyz",
            path="/home/user/test.txt",
        )
        assert event.tab_id == "t2"
        assert event.document_id == "doc-xyz"
        assert event.path == "/home/user/test.txt"

    def test_document_opened_without_path(self) -> None:
        """DocumentOpened can be created without a path."""
        event = DocumentOpened(tab_id="t2", document_id="doc-xyz")
        assert event.path is None

    def test_document_opened_publish_subscribe(self) -> None:
        """DocumentOpened can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[DocumentOpened] = []

        bus.subscribe(DocumentOpened, lambda e: received.append(e))
        bus.publish(DocumentOpened(tab_id="t1", document_id="doc-1", path="/test.md"))

        assert len(received) == 1
        assert received[0].path == "/test.md"


class TestDocumentClosed:
    """Tests for the DocumentClosed event."""

    def test_document_closed_fields(self) -> None:
        """DocumentClosed stores tab_id and document_id."""
        event = DocumentClosed(tab_id="t3", document_id="doc-closed")
        assert event.tab_id == "t3"
        assert event.document_id == "doc-closed"

    def test_document_closed_publish_subscribe(self) -> None:
        """DocumentClosed can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[DocumentClosed] = []

        bus.subscribe(DocumentClosed, lambda e: received.append(e))
        bus.publish(DocumentClosed(tab_id="t1", document_id="doc-1"))

        assert len(received) == 1
        assert received[0].tab_id == "t1"


class TestDocumentSaved:
    """Tests for the DocumentSaved event."""

    def test_document_saved_fields(self) -> None:
        """DocumentSaved stores tab_id, document_id, and path."""
        event = DocumentSaved(
            tab_id="t4",
            document_id="doc-saved",
            path="/home/user/saved.md",
        )
        assert event.tab_id == "t4"
        assert event.document_id == "doc-saved"
        assert event.path == "/home/user/saved.md"

    def test_document_saved_publish_subscribe(self) -> None:
        """DocumentSaved can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[DocumentSaved] = []

        bus.subscribe(DocumentSaved, lambda e: received.append(e))
        bus.publish(DocumentSaved(tab_id="t1", document_id="doc-1", path="/out.txt"))

        assert len(received) == 1
        assert received[0].path == "/out.txt"


class TestDocumentModified:
    """Tests for the DocumentModified event."""

    def test_document_modified_fields(self) -> None:
        """DocumentModified stores version tracking info."""
        event = DocumentModified(
            tab_id="t5",
            document_id="doc-mod",
            version_id=42,
            content_hash="abc123def456",
        )
        assert event.tab_id == "t5"
        assert event.document_id == "doc-mod"
        assert event.version_id == 42
        assert event.content_hash == "abc123def456"

    def test_document_modified_publish_subscribe(self) -> None:
        """DocumentModified can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[DocumentModified] = []

        bus.subscribe(DocumentModified, lambda e: received.append(e))
        bus.publish(DocumentModified(
            tab_id="t1",
            document_id="doc-1",
            version_id=1,
            content_hash="hash1",
        ))

        assert len(received) == 1
        assert received[0].version_id == 1


class TestActiveTabChanged:
    """Tests for the ActiveTabChanged event."""

    def test_active_tab_changed_fields(self) -> None:
        """ActiveTabChanged stores tab_id and document_id."""
        event = ActiveTabChanged(tab_id="t6", document_id="doc-active")
        assert event.tab_id == "t6"
        assert event.document_id == "doc-active"

    def test_active_tab_changed_publish_subscribe(self) -> None:
        """ActiveTabChanged can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[ActiveTabChanged] = []

        bus.subscribe(ActiveTabChanged, lambda e: received.append(e))
        bus.publish(ActiveTabChanged(tab_id="t2", document_id="doc-2"))

        assert len(received) == 1
        assert received[0].tab_id == "t2"


class TestDocumentEventIsolation:
    """Tests for isolation between document event types."""

    def test_document_events_are_isolated(self) -> None:
        """Each document event type is independent."""
        bus: EventBus[Event] = EventBus()
        created: list[DocumentCreated] = []
        opened: list[DocumentOpened] = []
        closed: list[DocumentClosed] = []

        bus.subscribe(DocumentCreated, lambda e: created.append(e))
        bus.subscribe(DocumentOpened, lambda e: opened.append(e))
        bus.subscribe(DocumentClosed, lambda e: closed.append(e))

        bus.publish(DocumentCreated(tab_id="t1", document_id="doc-1"))
        bus.publish(DocumentOpened(tab_id="t2", document_id="doc-2", path="/test.md"))
        bus.publish(DocumentClosed(tab_id="t3", document_id="doc-3"))

        assert len(created) == 1
        assert len(opened) == 1
        assert len(closed) == 1
        assert created[0].tab_id == "t1"
        assert opened[0].tab_id == "t2"
        assert closed[0].tab_id == "t3"


# =============================================================================
# AI Turn Event Tests (WS1.3)
# =============================================================================


from tinkerbell.ui.events import (
    AITurnStarted,
    AITurnStreamChunk,
    AITurnCompleted,
    AITurnFailed,
    AITurnCanceled,
)


class TestAITurnStarted:
    """Tests for the AITurnStarted event."""

    def test_ai_turn_started_fields(self) -> None:
        """AITurnStarted stores turn_id and prompt."""
        event = AITurnStarted(turn_id="turn-1", prompt="Summarize this document")
        assert event.turn_id == "turn-1"
        assert event.prompt == "Summarize this document"

    def test_ai_turn_started_publish_subscribe(self) -> None:
        """AITurnStarted can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[AITurnStarted] = []

        bus.subscribe(AITurnStarted, lambda e: received.append(e))
        bus.publish(AITurnStarted(turn_id="turn-42", prompt="Fix the bug"))

        assert len(received) == 1
        assert received[0].turn_id == "turn-42"
        assert received[0].prompt == "Fix the bug"


class TestAITurnStreamChunk:
    """Tests for the AITurnStreamChunk event."""

    def test_ai_turn_stream_chunk_fields(self) -> None:
        """AITurnStreamChunk stores turn_id and content."""
        event = AITurnStreamChunk(turn_id="turn-1", content="Here is the")
        assert event.turn_id == "turn-1"
        assert event.content == "Here is the"

    def test_ai_turn_stream_chunk_publish_subscribe(self) -> None:
        """AITurnStreamChunk can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[AITurnStreamChunk] = []

        bus.subscribe(AITurnStreamChunk, lambda e: received.append(e))
        bus.publish(AITurnStreamChunk(turn_id="turn-1", content="chunk 1"))
        bus.publish(AITurnStreamChunk(turn_id="turn-1", content="chunk 2"))

        assert len(received) == 2
        assert received[0].content == "chunk 1"
        assert received[1].content == "chunk 2"


class TestAITurnCompleted:
    """Tests for the AITurnCompleted event."""

    def test_ai_turn_completed_fields(self) -> None:
        """AITurnCompleted stores completion details."""
        event = AITurnCompleted(
            turn_id="turn-5",
            success=True,
            edit_count=3,
            response_text="I made 3 edits to improve clarity.",
        )
        assert event.turn_id == "turn-5"
        assert event.success is True
        assert event.edit_count == 3
        assert event.response_text == "I made 3 edits to improve clarity."

    def test_ai_turn_completed_failure(self) -> None:
        """AITurnCompleted can represent failed turns."""
        event = AITurnCompleted(
            turn_id="turn-6",
            success=False,
            edit_count=0,
            response_text="",
        )
        assert event.success is False
        assert event.edit_count == 0

    def test_ai_turn_completed_publish_subscribe(self) -> None:
        """AITurnCompleted can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[AITurnCompleted] = []

        bus.subscribe(AITurnCompleted, lambda e: received.append(e))
        bus.publish(AITurnCompleted(
            turn_id="turn-1",
            success=True,
            edit_count=2,
            response_text="Done!",
        ))

        assert len(received) == 1
        assert received[0].edit_count == 2


class TestAITurnFailed:
    """Tests for the AITurnFailed event."""

    def test_ai_turn_failed_fields(self) -> None:
        """AITurnFailed stores turn_id and error."""
        event = AITurnFailed(turn_id="turn-7", error="API rate limit exceeded")
        assert event.turn_id == "turn-7"
        assert event.error == "API rate limit exceeded"

    def test_ai_turn_failed_publish_subscribe(self) -> None:
        """AITurnFailed can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[AITurnFailed] = []

        bus.subscribe(AITurnFailed, lambda e: received.append(e))
        bus.publish(AITurnFailed(turn_id="turn-1", error="Connection timeout"))

        assert len(received) == 1
        assert received[0].error == "Connection timeout"


class TestAITurnCanceled:
    """Tests for the AITurnCanceled event."""

    def test_ai_turn_canceled_fields(self) -> None:
        """AITurnCanceled stores turn_id."""
        event = AITurnCanceled(turn_id="turn-8")
        assert event.turn_id == "turn-8"

    def test_ai_turn_canceled_publish_subscribe(self) -> None:
        """AITurnCanceled can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[AITurnCanceled] = []

        bus.subscribe(AITurnCanceled, lambda e: received.append(e))
        bus.publish(AITurnCanceled(turn_id="turn-99"))

        assert len(received) == 1
        assert received[0].turn_id == "turn-99"


class TestAITurnEventIsolation:
    """Tests for isolation between AI turn event types."""

    def test_ai_turn_events_are_isolated(self) -> None:
        """Each AI turn event type is independent."""
        bus: EventBus[Event] = EventBus()
        started: list[AITurnStarted] = []
        chunks: list[AITurnStreamChunk] = []
        completed: list[AITurnCompleted] = []
        failed: list[AITurnFailed] = []
        canceled: list[AITurnCanceled] = []

        bus.subscribe(AITurnStarted, lambda e: started.append(e))
        bus.subscribe(AITurnStreamChunk, lambda e: chunks.append(e))
        bus.subscribe(AITurnCompleted, lambda e: completed.append(e))
        bus.subscribe(AITurnFailed, lambda e: failed.append(e))
        bus.subscribe(AITurnCanceled, lambda e: canceled.append(e))

        bus.publish(AITurnStarted(turn_id="turn-1", prompt="test"))
        bus.publish(AITurnStreamChunk(turn_id="turn-1", content="chunk"))
        bus.publish(AITurnCompleted(turn_id="turn-1", success=True, edit_count=1, response_text="ok"))
        bus.publish(AITurnFailed(turn_id="turn-2", error="oops"))
        bus.publish(AITurnCanceled(turn_id="turn-3"))

        assert len(started) == 1
        assert len(chunks) == 1
        assert len(completed) == 1
        assert len(failed) == 1
        assert len(canceled) == 1


# =============================================================================
# Edit & Review Event Tests (WS1.4)
# =============================================================================


from tinkerbell.ui.events import (
    EditApplied,
    EditFailed,
    ReviewStateChanged,
    ReviewAccepted,
    ReviewRejected,
)


class TestEditApplied:
    """Tests for the EditApplied event."""

    def test_edit_applied_fields(self) -> None:
        """EditApplied stores all edit details."""
        event = EditApplied(
            tab_id="t1",
            document_id="doc-1",
            edit_id="edit-abc",
            action="patch",
            range=(100, 150),
            diff="@@ -1,3 +1,4 @@\n old\n+new",
        )
        assert event.tab_id == "t1"
        assert event.document_id == "doc-1"
        assert event.edit_id == "edit-abc"
        assert event.action == "patch"
        assert event.range == (100, 150)
        assert event.diff == "@@ -1,3 +1,4 @@\n old\n+new"

    def test_edit_applied_publish_subscribe(self) -> None:
        """EditApplied can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[EditApplied] = []

        bus.subscribe(EditApplied, lambda e: received.append(e))
        bus.publish(EditApplied(
            tab_id="t1",
            document_id="doc-1",
            edit_id="edit-1",
            action="replace",
            range=(0, 10),
            diff="-old\n+new",
        ))

        assert len(received) == 1
        assert received[0].action == "replace"


class TestEditFailed:
    """Tests for the EditFailed event."""

    def test_edit_failed_fields(self) -> None:
        """EditFailed stores failure details."""
        event = EditFailed(
            tab_id="t2",
            document_id="doc-2",
            action="patch",
            reason="Target text not found",
        )
        assert event.tab_id == "t2"
        assert event.document_id == "doc-2"
        assert event.action == "patch"
        assert event.reason == "Target text not found"

    def test_edit_failed_publish_subscribe(self) -> None:
        """EditFailed can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[EditFailed] = []

        bus.subscribe(EditFailed, lambda e: received.append(e))
        bus.publish(EditFailed(
            tab_id="t1",
            document_id="doc-1",
            action="insert",
            reason="Invalid range",
        ))

        assert len(received) == 1
        assert received[0].reason == "Invalid range"


class TestReviewStateChanged:
    """Tests for the ReviewStateChanged event."""

    def test_review_state_changed_fields(self) -> None:
        """ReviewStateChanged stores review state details."""
        event = ReviewStateChanged(
            turn_id="turn-5",
            ready=True,
            edit_count=3,
            tabs_affected=2,
        )
        assert event.turn_id == "turn-5"
        assert event.ready is True
        assert event.edit_count == 3
        assert event.tabs_affected == 2

    def test_review_state_changed_not_ready(self) -> None:
        """ReviewStateChanged can indicate not ready state."""
        event = ReviewStateChanged(
            turn_id="turn-6",
            ready=False,
            edit_count=0,
            tabs_affected=0,
        )
        assert event.ready is False

    def test_review_state_changed_publish_subscribe(self) -> None:
        """ReviewStateChanged can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[ReviewStateChanged] = []

        bus.subscribe(ReviewStateChanged, lambda e: received.append(e))
        bus.publish(ReviewStateChanged(
            turn_id="turn-1",
            ready=True,
            edit_count=5,
            tabs_affected=1,
        ))

        assert len(received) == 1
        assert received[0].edit_count == 5


class TestReviewAccepted:
    """Tests for the ReviewAccepted event."""

    def test_review_accepted_fields(self) -> None:
        """ReviewAccepted stores turn_id and affected tabs."""
        event = ReviewAccepted(
            turn_id="turn-7",
            tabs=("t1", "t2", "t3"),
        )
        assert event.turn_id == "turn-7"
        assert event.tabs == ("t1", "t2", "t3")

    def test_review_accepted_single_tab(self) -> None:
        """ReviewAccepted can have a single affected tab."""
        event = ReviewAccepted(turn_id="turn-8", tabs=("t1",))
        assert len(event.tabs) == 1

    def test_review_accepted_publish_subscribe(self) -> None:
        """ReviewAccepted can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[ReviewAccepted] = []

        bus.subscribe(ReviewAccepted, lambda e: received.append(e))
        bus.publish(ReviewAccepted(turn_id="turn-1", tabs=("t1", "t2")))

        assert len(received) == 1
        assert received[0].tabs == ("t1", "t2")


class TestReviewRejected:
    """Tests for the ReviewRejected event."""

    def test_review_rejected_fields(self) -> None:
        """ReviewRejected stores turn_id and affected tabs."""
        event = ReviewRejected(
            turn_id="turn-9",
            tabs=("t4", "t5"),
        )
        assert event.turn_id == "turn-9"
        assert event.tabs == ("t4", "t5")

    def test_review_rejected_publish_subscribe(self) -> None:
        """ReviewRejected can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[ReviewRejected] = []

        bus.subscribe(ReviewRejected, lambda e: received.append(e))
        bus.publish(ReviewRejected(turn_id="turn-1", tabs=("t1",)))

        assert len(received) == 1
        assert received[0].turn_id == "turn-1"


class TestEditReviewEventIsolation:
    """Tests for isolation between edit and review event types."""

    def test_edit_review_events_are_isolated(self) -> None:
        """Each edit/review event type is independent."""
        bus: EventBus[Event] = EventBus()
        applied: list[EditApplied] = []
        failed: list[EditFailed] = []
        state_changed: list[ReviewStateChanged] = []
        accepted: list[ReviewAccepted] = []
        rejected: list[ReviewRejected] = []

        bus.subscribe(EditApplied, lambda e: applied.append(e))
        bus.subscribe(EditFailed, lambda e: failed.append(e))
        bus.subscribe(ReviewStateChanged, lambda e: state_changed.append(e))
        bus.subscribe(ReviewAccepted, lambda e: accepted.append(e))
        bus.subscribe(ReviewRejected, lambda e: rejected.append(e))

        bus.publish(EditApplied(
            tab_id="t1", document_id="d1", edit_id="e1",
            action="patch", range=(0, 10), diff="+new"
        ))
        bus.publish(EditFailed(
            tab_id="t2", document_id="d2", action="replace", reason="error"
        ))
        bus.publish(ReviewStateChanged(
            turn_id="turn-1", ready=True, edit_count=1, tabs_affected=1
        ))
        bus.publish(ReviewAccepted(turn_id="turn-1", tabs=("t1",)))
        bus.publish(ReviewRejected(turn_id="turn-2", tabs=("t2",)))

        assert len(applied) == 1
        assert len(failed) == 1
        assert len(state_changed) == 1
        assert len(accepted) == 1
        assert len(rejected) == 1


# =============================================================================
# UI Event Tests (WS1.5)
# =============================================================================


from tinkerbell.ui.events import (
    StatusMessage,
    NoticePosted,
    WindowTitleChanged,
    EditorLockChanged,
)


class TestStatusMessage:
    """Tests for the StatusMessage event."""

    def test_status_message_fields(self) -> None:
        """StatusMessage stores message and timeout."""
        event = StatusMessage(message="Ready", timeout_ms=5000)
        assert event.message == "Ready"
        assert event.timeout_ms == 5000

    def test_status_message_default_timeout(self) -> None:
        """StatusMessage has default timeout of 0 (persistent)."""
        event = StatusMessage(message="Saving...")
        assert event.timeout_ms == 0

    def test_status_message_publish_subscribe(self) -> None:
        """StatusMessage can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[StatusMessage] = []

        bus.subscribe(StatusMessage, lambda e: received.append(e))
        bus.publish(StatusMessage(message="Document saved", timeout_ms=3000))

        assert len(received) == 1
        assert received[0].message == "Document saved"
        assert received[0].timeout_ms == 3000


class TestNoticePosted:
    """Tests for the NoticePosted event."""

    def test_notice_posted_fields(self) -> None:
        """NoticePosted stores the message."""
        event = NoticePosted(message="AI completed 3 edits")
        assert event.message == "AI completed 3 edits"

    def test_notice_posted_publish_subscribe(self) -> None:
        """NoticePosted can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[NoticePosted] = []

        bus.subscribe(NoticePosted, lambda e: received.append(e))
        bus.publish(NoticePosted(message="Operation completed"))

        assert len(received) == 1
        assert received[0].message == "Operation completed"


class TestWindowTitleChanged:
    """Tests for the WindowTitleChanged event."""

    def test_window_title_changed_fields(self) -> None:
        """WindowTitleChanged stores the title."""
        event = WindowTitleChanged(title="Document.md - TinkerBell")
        assert event.title == "Document.md - TinkerBell"

    def test_window_title_changed_publish_subscribe(self) -> None:
        """WindowTitleChanged can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[WindowTitleChanged] = []

        bus.subscribe(WindowTitleChanged, lambda e: received.append(e))
        bus.publish(WindowTitleChanged(title="*Untitled - TinkerBell"))

        assert len(received) == 1
        assert received[0].title == "*Untitled - TinkerBell"


class TestEditorLockChanged:
    """Tests for the EditorLockChanged event."""

    def test_editor_lock_changed_locked(self) -> None:
        """EditorLockChanged stores locked state and reason."""
        event = EditorLockChanged(locked=True, reason="AI_TURN")
        assert event.locked is True
        assert event.reason == "AI_TURN"

    def test_editor_lock_changed_unlocked(self) -> None:
        """EditorLockChanged can indicate unlocked state."""
        event = EditorLockChanged(locked=False, reason="")
        assert event.locked is False
        assert event.reason == ""

    def test_editor_lock_changed_publish_subscribe(self) -> None:
        """EditorLockChanged can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[EditorLockChanged] = []

        bus.subscribe(EditorLockChanged, lambda e: received.append(e))
        bus.publish(EditorLockChanged(locked=True, reason="REVIEW_PENDING"))

        assert len(received) == 1
        assert received[0].locked is True
        assert received[0].reason == "REVIEW_PENDING"


class TestUIEventIsolation:
    """Tests for isolation between UI event types."""

    def test_ui_events_are_isolated(self) -> None:
        """Each UI event type is independent."""
        bus: EventBus[Event] = EventBus()
        status: list[StatusMessage] = []
        notices: list[NoticePosted] = []
        titles: list[WindowTitleChanged] = []
        locks: list[EditorLockChanged] = []

        bus.subscribe(StatusMessage, lambda e: status.append(e))
        bus.subscribe(NoticePosted, lambda e: notices.append(e))
        bus.subscribe(WindowTitleChanged, lambda e: titles.append(e))
        bus.subscribe(EditorLockChanged, lambda e: locks.append(e))

        bus.publish(StatusMessage(message="status", timeout_ms=1000))
        bus.publish(NoticePosted(message="notice"))
        bus.publish(WindowTitleChanged(title="title"))
        bus.publish(EditorLockChanged(locked=True, reason="AI_TURN"))

        assert len(status) == 1
        assert len(notices) == 1
        assert len(titles) == 1
        assert len(locks) == 1


# =============================================================================
# Infrastructure Event Tests (WS1.6)
# =============================================================================


from tinkerbell.ui.events import (
    SettingsChanged,
    EmbeddingStateChanged,
    OutlineUpdated,
    TelemetryEvent,
)


class TestSettingsChanged:
    """Tests for the SettingsChanged event."""

    def test_settings_changed_fields(self) -> None:
        """SettingsChanged stores settings dict."""
        settings = {"theme": "dark", "font_size": 14}
        event = SettingsChanged(settings=settings)
        assert event.settings == {"theme": "dark", "font_size": 14}

    def test_settings_changed_publish_subscribe(self) -> None:
        """SettingsChanged can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[SettingsChanged] = []

        bus.subscribe(SettingsChanged, lambda e: received.append(e))
        bus.publish(SettingsChanged(settings={"api_key": "***"}))

        assert len(received) == 1
        assert received[0].settings == {"api_key": "***"}


class TestEmbeddingStateChanged:
    """Tests for the EmbeddingStateChanged event."""

    def test_embedding_state_changed_fields(self) -> None:
        """EmbeddingStateChanged stores backend, status, and detail."""
        event = EmbeddingStateChanged(
            backend="langchain",
            status="ready",
            detail="text-embedding-ada-002",
        )
        assert event.backend == "langchain"
        assert event.status == "ready"
        assert event.detail == "text-embedding-ada-002"

    def test_embedding_state_changed_default_detail(self) -> None:
        """EmbeddingStateChanged has default detail of None."""
        event = EmbeddingStateChanged(backend="disabled", status="disabled")
        assert event.detail is None

    def test_embedding_state_changed_error(self) -> None:
        """EmbeddingStateChanged can represent error state."""
        event = EmbeddingStateChanged(
            backend="openai",
            status="error",
            detail="API key invalid",
        )
        assert event.status == "error"
        assert event.detail == "API key invalid"

    def test_embedding_state_changed_publish_subscribe(self) -> None:
        """EmbeddingStateChanged can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[EmbeddingStateChanged] = []

        bus.subscribe(EmbeddingStateChanged, lambda e: received.append(e))
        bus.publish(EmbeddingStateChanged(
            backend="sentence-transformers",
            status="initializing",
        ))

        assert len(received) == 1
        assert received[0].backend == "sentence-transformers"


class TestOutlineUpdated:
    """Tests for the OutlineUpdated event."""

    def test_outline_updated_fields(self) -> None:
        """OutlineUpdated stores document_id, outline_hash, and node_count."""
        event = OutlineUpdated(
            document_id="doc-abc",
            outline_hash="sha256:abc123",
            node_count=42,
        )
        assert event.document_id == "doc-abc"
        assert event.outline_hash == "sha256:abc123"
        assert event.node_count == 42

    def test_outline_updated_publish_subscribe(self) -> None:
        """OutlineUpdated can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[OutlineUpdated] = []

        bus.subscribe(OutlineUpdated, lambda e: received.append(e))
        bus.publish(OutlineUpdated(
            document_id="doc-1",
            outline_hash="hash123",
            node_count=10,
        ))

        assert len(received) == 1
        assert received[0].node_count == 10


class TestTelemetryEvent:
    """Tests for the TelemetryEvent event."""

    def test_telemetry_event_fields(self) -> None:
        """TelemetryEvent stores name and payload."""
        event = TelemetryEvent(
            name="embedding.cache.hit",
            payload={"document_id": "doc-1", "latency_ms": 5},
        )
        assert event.name == "embedding.cache.hit"
        assert event.payload == {"document_id": "doc-1", "latency_ms": 5}

    def test_telemetry_event_empty_payload(self) -> None:
        """TelemetryEvent can have empty payload."""
        event = TelemetryEvent(name="app.startup", payload={})
        assert event.payload == {}

    def test_telemetry_event_publish_subscribe(self) -> None:
        """TelemetryEvent can be published and received."""
        bus: EventBus[Event] = EventBus()
        received: list[TelemetryEvent] = []

        bus.subscribe(TelemetryEvent, lambda e: received.append(e))
        bus.publish(TelemetryEvent(
            name="ai.turn.completed",
            payload={"turn_id": "turn-1", "edit_count": 3},
        ))

        assert len(received) == 1
        assert received[0].name == "ai.turn.completed"


class TestInfrastructureEventIsolation:
    """Tests for isolation between infrastructure event types."""

    def test_infrastructure_events_are_isolated(self) -> None:
        """Each infrastructure event type is independent."""
        bus: EventBus[Event] = EventBus()
        settings: list[SettingsChanged] = []
        embeddings: list[EmbeddingStateChanged] = []
        outlines: list[OutlineUpdated] = []
        telemetry: list[TelemetryEvent] = []

        bus.subscribe(SettingsChanged, lambda e: settings.append(e))
        bus.subscribe(EmbeddingStateChanged, lambda e: embeddings.append(e))
        bus.subscribe(OutlineUpdated, lambda e: outlines.append(e))
        bus.subscribe(TelemetryEvent, lambda e: telemetry.append(e))

        bus.publish(SettingsChanged(settings={"key": "value"}))
        bus.publish(EmbeddingStateChanged(backend="test", status="ready"))
        bus.publish(OutlineUpdated(document_id="d1", outline_hash="h1", node_count=5))
        bus.publish(TelemetryEvent(name="test.event", payload={}))

        assert len(settings) == 1
        assert len(embeddings) == 1
        assert len(outlines) == 1
        assert len(telemetry) == 1
