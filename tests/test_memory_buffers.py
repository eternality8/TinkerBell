"""Tests for the memory buffer helpers."""

from __future__ import annotations

from pathlib import Path

import gc
import weakref

from tinkerbell.ai.memory import buffers
from tinkerbell.ai.memory.cache_bus import (
    ChunkCacheSubscriber,
    DocumentCacheBus,
    DocumentCacheEvent,
    DocumentChangedEvent,
    DocumentClosedEvent,
)


def test_conversation_memory_trims_messages_and_tokens() -> None:
    memory = buffers.ConversationMemory(max_messages=3, max_tokens=6, token_counter=len)

    memory.add("user", "aaaa")
    memory.add("assistant", "bb")
    memory.add("user", "ccc")  # pushes token budget over the limit
    memory.add("assistant", "dd")

    messages = [message.content for message in memory.get_messages()]

    assert messages == ["ccc", "dd"]
    assert memory.total_tokens <= 6


def test_document_summary_memory_uses_custom_summarizer_and_prunes() -> None:
    def summarizer(previous: str | None, new_text: str) -> str:
        prefix = previous or ""
        return f"{prefix}|{new_text.lower()}".strip("|")

    summary_memory = buffers.DocumentSummaryMemory(max_entries=2, summarizer=summarizer)

    summary_memory.update("doc-1", text="Alpha")
    summary_memory.update("doc-2", text="Beta")
    summary_memory.update("doc-3", summary="Manual summary")

    assert summary_memory.get("doc-1") is None  # pruned because only 2 entries allowed
    doc3 = summary_memory.get("doc-3")
    assert doc3 is not None and doc3.summary == "Manual summary"
    doc2 = summary_memory.get("doc-2")
    assert doc2 is not None and "beta" in doc2.summary


def test_memory_store_roundtrip(tmp_path: Path) -> None:
    store = buffers.MemoryStore(tmp_path)
    convo = buffers.ConversationMemory(token_counter=len)
    convo.add("user", "Hello world", metadata={"name": "operator"})

    store.save_conversation("My Notes.md", convo)
    loaded = store.load_conversation(
        "My Notes.md",
        conversation_factory=lambda: buffers.ConversationMemory(token_counter=len),
    )

    assert [message.content for message in loaded.get_messages()] == ["Hello world"]

    summaries = buffers.DocumentSummaryMemory()
    summaries.update("doc-42", summary="Short summary", highlights=["Intro", "Conclusion"])

    store.save_document_summaries(summaries)
    restored = buffers.DocumentSummaryMemory()
    store.load_document_summaries(restored)

    record = restored.get("doc-42")
    assert record is not None
    assert record.summary == "Short summary"
    assert record.highlights == ["Intro", "Conclusion"]


def test_document_cache_bus_notifies_subscribers_in_order() -> None:
    bus = DocumentCacheBus()
    received: list[tuple[str, str]] = []

    def on_changed(event: DocumentCacheEvent) -> None:
        assert isinstance(event, DocumentChangedEvent)
        received.append(("changed", event.document_id))

    def on_closed(event: DocumentCacheEvent) -> None:
        assert isinstance(event, DocumentClosedEvent)
        received.append(("closed", event.document_id))

    bus.subscribe(DocumentChangedEvent, on_changed)
    bus.subscribe(DocumentClosedEvent, on_closed)

    bus.publish(
        DocumentChangedEvent(document_id="doc-1", version_id=2, content_hash="abc", edited_ranges=((0, 5),)),
    )
    bus.publish(DocumentClosedEvent(document_id="doc-1", version_id=2, content_hash="abc"))

    assert received == [("changed", "doc-1"), ("closed", "doc-1")]

    bus.unsubscribe(DocumentChangedEvent, on_changed)
    bus.publish(
        DocumentChangedEvent(document_id="doc-1", version_id=3, content_hash="def", edited_ranges=((0, 1),)),
    )
    assert received == [("changed", "doc-1"), ("closed", "doc-1")]


def test_document_cache_bus_supports_weak_subscribers() -> None:
    bus = DocumentCacheBus()
    events: list[str] = []

    class Observer:
        def handle(self, event: DocumentCacheEvent) -> None:
            assert isinstance(event, DocumentChangedEvent)
            events.append(event.document_id)

    observer = Observer()
    bus.subscribe(DocumentChangedEvent, observer.handle, weak=True)
    bus.publish(DocumentChangedEvent(document_id="doc-weak", version_id=1, content_hash="x"))

    assert events == ["doc-weak"]

    observer_ref = weakref.ref(observer)
    del observer
    gc.collect()

    bus.publish(DocumentChangedEvent(document_id="doc-weak-2", version_id=2, content_hash="y"))

    assert observer_ref() is None
    assert events == ["doc-weak"]


def test_cache_stub_subscribers_record_events() -> None:
    bus = DocumentCacheBus()
    chunk = ChunkCacheSubscriber(bus=bus)

    bus.publish(DocumentChangedEvent(document_id="doc-stub", version_id=9, content_hash="zzz"))
    bus.publish(DocumentClosedEvent(document_id="doc-stub", version_id=9, content_hash="zzz"))

    assert len(chunk.events) == 2
    assert isinstance(chunk.events[0], DocumentChangedEvent)
