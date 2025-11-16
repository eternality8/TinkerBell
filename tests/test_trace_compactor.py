"""Tests for the TraceCompactor service."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Mapping

from tinkerbell.ai.services.trace_compactor import TraceCompactor
from tinkerbell.chat.message_model import ToolPointerMessage


def _token_counter(message: Mapping[str, object]) -> int:
    return len(str(message.get("content") or ""))


def _pointer_factory(prefix: str = "ptr"):
    counter = {"value": 0}

    def _builder(record: Mapping[str, object], content: str) -> ToolPointerMessage:
        counter["value"] += 1
        pointer_id = f"{prefix}-{counter['value']}"
        return ToolPointerMessage(
            pointer_id=pointer_id,
            kind="text",
            display_text="summary",
            rehydrate_instructions="rehydrate",
        )

    return _builder


def test_compact_entry_updates_stats_and_records() -> None:
    compactor = TraceCompactor(
        pointer_builder=_pointer_factory(),
        estimate_message_tokens=_token_counter,
    )
    content = "X" * 200
    message = {"role": "tool", "content": content}
    record = {"id": "call-1", "summarizable": True}
    entry = compactor.new_entry(message, record, raw_content=content, summarizable=True)

    saved = compactor.compact_entry(entry)

    assert saved > 0
    assert entry.compacted is True
    assert "pointer" in record
    stats = compactor.stats_snapshot()
    assert stats.total_compactions == 1
    assert stats.tokens_saved >= saved


def test_compact_history_prefers_oldest_entries() -> None:
    compactor = TraceCompactor(
        pointer_builder=_pointer_factory(),
        estimate_message_tokens=_token_counter,
    )
    entries = []
    for idx in range(2):
        content = "Y" * 400
        message = {"role": "tool", "content": content}
        record = {"id": f"call-{idx}", "summarizable": True}
        entry = compactor.new_entry(message, record, raw_content=content, summarizable=True)
        compactor.commit_entry(entry, current_tokens=len(content))
        entries.append(entry)

    def _evaluate(prompt_tokens: int, pending_tokens: int) -> SimpleNamespace:
        total = prompt_tokens + pending_tokens
        verdict = "needs_summary" if total > 900 else "ok"
        return SimpleNamespace(verdict=verdict, dry_run=False)

    new_tokens, decision = compactor.compact_history(
        evaluate=_evaluate,
        conversation_tokens=800,
        pending_tokens=200,
    )

    assert decision.verdict == "ok"
    assert new_tokens < 800
    assert entries[0].compacted is True
    assert entries[1].compacted is False