"""Trace compaction helpers coordinating pointer swaps for oversized tool output."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Mapping, MutableMapping

from tinkerbell.ui.presentation.chat.message_model import ToolPointerMessage


EstimateMessageTokens = Callable[[Mapping[str, Any]], int]
PointerBuilder = Callable[[Mapping[str, Any], str], ToolPointerMessage]
BudgetEvaluator = Callable[[int, int], Any]


@dataclass(slots=True)
class TraceCompactionStats:
    """Aggregated metrics describing the current compaction session."""

    entries_tracked: int = 0
    total_compactions: int = 0
    tokens_saved: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "entries_tracked": self.entries_tracked,
            "total_compactions": self.total_compactions,
            "tokens_saved": self.tokens_saved,
        }


@dataclass(slots=True)
class TraceLedgerEntry:
    """Ledger item referencing the conversation message for a tool output."""

    message: MutableMapping[str, Any]
    record: MutableMapping[str, Any]
    raw_content: str
    summarizable: bool
    current_tokens: int = 0
    pointer: ToolPointerMessage | None = None
    compacted: bool = False
    tool_call_id: str | None = None


class TraceCompactor:
    """Stateful helper that compacts tool traces when budgets are exceeded."""

    def __init__(
        self,
        *,
        pointer_builder: PointerBuilder,
        estimate_message_tokens: EstimateMessageTokens,
    ) -> None:
        self._pointer_builder = pointer_builder
        self._estimate_message_tokens = estimate_message_tokens
        self._ledger: Deque[TraceLedgerEntry] = deque()
        self._stats = TraceCompactionStats()

    def reset(self) -> None:
        """Clear the ledger/stats between chat runs."""

        self._ledger.clear()
        self._stats = TraceCompactionStats()

    def stats_snapshot(self) -> TraceCompactionStats:
        """Return a shallow copy of the current compaction stats."""

        return TraceCompactionStats(
            entries_tracked=self._stats.entries_tracked,
            total_compactions=self._stats.total_compactions,
            tokens_saved=self._stats.tokens_saved,
        )

    def new_entry(
        self,
        message: MutableMapping[str, Any],
        record: MutableMapping[str, Any],
        *,
        raw_content: str,
        summarizable: bool,
    ) -> TraceLedgerEntry:
        """Create (but do not persist) a ledger entry for ``message``."""

        tool_call_id = None
        record_id = record.get("id") if isinstance(record, Mapping) else None
        if record_id is not None:
            tool_call_id = str(record_id)
        return TraceLedgerEntry(
            message=message,
            record=record,
            raw_content=raw_content,
            summarizable=summarizable,
            tool_call_id=tool_call_id,
        )

    def commit_entry(self, entry: TraceLedgerEntry, *, current_tokens: int) -> None:
        """Persist ``entry`` to the ledger with its current token count."""

        entry.current_tokens = current_tokens
        self._ledger.append(entry)
        self._stats.entries_tracked += 1

    def compact_entry(self, entry: TraceLedgerEntry) -> int:
        """Convert ``entry`` to a pointer, returning the tokens saved."""

        if entry.compacted or not entry.summarizable:
            return 0
        if not entry.raw_content.strip():
            return 0
        before_tokens = entry.current_tokens or self._estimate_message_tokens(entry.message)
        pointer = self._pointer_builder(entry.record, entry.raw_content)
        entry.pointer = pointer
        entry.message["content"] = pointer.as_chat_content()
        entry.record["pointer"] = pointer.as_dict()
        entry.compacted = True
        after_tokens = self._estimate_message_tokens(entry.message)
        entry.current_tokens = after_tokens
        saved = max(0, before_tokens - after_tokens)
        if saved > 0:
            self._stats.tokens_saved += saved
        self._stats.total_compactions += 1
        return saved

    def compact_history(
        self,
        *,
        evaluate: BudgetEvaluator,
        conversation_tokens: int,
        pending_tokens: int,
    ) -> tuple[int, Any]:
        """Compact existing ledger entries until the evaluator reports OK."""

        decision = evaluate(conversation_tokens, pending_tokens)
        if decision is None or getattr(decision, "verdict", None) != "needs_summary":
            return conversation_tokens, decision
        for entry in self._ledger:
            if entry.compacted or not entry.summarizable:
                continue
            if not entry.raw_content.strip():
                continue
            saved = self.compact_entry(entry)
            if saved <= 0:
                continue
            conversation_tokens = max(0, conversation_tokens - saved)
            decision = evaluate(conversation_tokens, pending_tokens)
            if decision is None or getattr(decision, "verdict", None) != "needs_summary":
                break
        return conversation_tokens, decision

    def ledger_snapshot(self) -> tuple[TraceLedgerEntry, ...]:
        """Expose the current ledger for tests/inspection."""

        return tuple(self._ledger)


__all__ = [
    "TraceCompactor",
    "TraceCompactionStats",
    "TraceLedgerEntry",
]
