"""Tests for the memory buffer helpers."""

from __future__ import annotations

from pathlib import Path

from tinkerbell.ai.memory import buffers


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
