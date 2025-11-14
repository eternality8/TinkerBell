"""Conversation and summary memory helpers used by the agent graph."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, cast

from openai.types.chat import ChatCompletionMessageParam

ChatRole = Literal["system", "user", "assistant", "tool"]
TokenCounter = Callable[[str], int]
SummaryReducer = Callable[[str | None, str], str]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _default_token_estimator(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def _safe_document_id(document_id: str) -> str:
    safe = document_id.strip().replace(" ", "_")
    return "".join(char for char in safe if char.isalnum() or char in {"_", "-", "."}) or "default"


@dataclass(slots=True)
class ConversationMessage:
    """Individual chat turn stored inside :class:`ConversationMemory`."""

    role: ChatRole
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_utcnow)
    token_count: int = 0

    def to_chat_param(self) -> ChatCompletionMessageParam:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if name := self.metadata.get("name"):
            payload["name"] = name
        if tool_call_id := self.metadata.get("tool_call_id"):
            payload["tool_call_id"] = tool_call_id
        return cast(ChatCompletionMessageParam, payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ConversationMessage:
        created_at = payload.get("created_at")
        timestamp = (
            datetime.fromisoformat(created_at)
            if isinstance(created_at, str)
            else _utcnow()
        )
        metadata = dict(payload.get("metadata", {}))
        token_count = int(payload.get("token_count", 0))
        return cls(
            role=cast(ChatRole, payload.get("role", "user")),
            content=str(payload.get("content", "")),
            metadata=metadata,
            created_at=timestamp,
            token_count=token_count,
        )


class ConversationMemory:
    """Token-aware rolling buffer of chat turns."""

    def __init__(
        self,
        *,
        max_messages: int = 30,
        max_tokens: int = 2048,
        token_counter: TokenCounter | None = None,
        initial_messages: Iterable[ConversationMessage | Mapping[str, Any]] | None = None,
    ) -> None:
        self._max_messages = max(1, max_messages)
        self._max_tokens = max_tokens if max_tokens > 0 else 0
        self._token_counter = token_counter or _default_token_estimator
        self._messages: list[ConversationMessage] = []
        self._total_tokens = 0
        if initial_messages:
            self.extend(initial_messages)

    def add(
        self,
        role: ChatRole,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ConversationMessage:
        message = ConversationMessage(
            role=role,
            content=content,
            metadata=dict(metadata or {}),
        )
        message.token_count = self._token_counter(content)
        self._messages.append(message)
        self._total_tokens += message.token_count
        self._trim()
        return message

    def extend(
        self, messages: Iterable[ConversationMessage | Mapping[str, Any]]
    ) -> None:
        for message in messages:
            if isinstance(message, ConversationMessage):
                self._messages.append(message)
            else:
                self._messages.append(ConversationMessage.from_dict(message))
            self._total_tokens += self._messages[-1].token_count
        self._trim()

    def clear(self) -> None:
        self._messages.clear()
        self._total_tokens = 0

    def _trim(self) -> None:
        while len(self._messages) > self._max_messages:
            removed = self._messages.pop(0)
            self._total_tokens -= removed.token_count

        if self._max_tokens:
            while self._total_tokens > self._max_tokens and len(self._messages) > 1:
                removed = self._messages.pop(0)
                self._total_tokens -= removed.token_count

    def as_chat_params(self) -> list[ChatCompletionMessageParam]:
        return [message.to_chat_param() for message in self._messages]

    def get_messages(self) -> list[ConversationMessage]:
        return list(self._messages)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def snapshot(self) -> dict[str, Any]:
        return {
            "message_count": len(self._messages),
            "total_tokens": self._total_tokens,
            "max_messages": self._max_messages,
            "max_tokens": self._max_tokens,
        }


@dataclass(slots=True)
class SummaryRecord:
    """Represents the rolling summary for a document."""

    document_id: str
    summary: str
    highlights: list[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "summary": self.summary,
            "highlights": list(self.highlights),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SummaryRecord:
        updated = payload.get("updated_at")
        timestamp = (
            datetime.fromisoformat(updated)
            if isinstance(updated, str)
            else _utcnow()
        )
        return cls(
            document_id=str(payload.get("document_id", "unknown")),
            summary=str(payload.get("summary", "")),
            highlights=list(payload.get("highlights", [])),
            updated_at=timestamp,
        )


class DocumentSummaryMemory:
    """Stores rolling summaries keyed by document identifier."""

    def __init__(
        self,
        *,
        max_entries: int = 8,
        max_summary_chars: int = 1200,
        summarizer: SummaryReducer | None = None,
    ) -> None:
        self._max_entries = max(1, max_entries)
        self._max_summary_chars = max_summary_chars
        self._summarizer = summarizer or self._default_summarizer
        self._records: dict[str, SummaryRecord] = {}

    def _default_summarizer(self, previous: str | None, text: str) -> str:
        snippet = (text or "").strip()
        if not snippet:
            return previous or ""
        if previous:
            combined = f"{previous}\n\nRecent excerpt:\n{snippet}"
        else:
            combined = snippet
        if self._max_summary_chars:
            return combined[: self._max_summary_chars]
        return combined

    def update(
        self,
        document_id: str,
        *,
        summary: str | None = None,
        text: str | None = None,
        highlights: Sequence[str] | None = None,
    ) -> SummaryRecord:
        if not summary and not text:
            raise ValueError("Either summary or text must be provided to update memory")
        previous = self._records.get(document_id)
        computed_summary = summary or self._summarizer(previous.summary if previous else None, text or "")
        if self._max_summary_chars:
            computed_summary = computed_summary[: self._max_summary_chars]
        record = SummaryRecord(
            document_id=document_id,
            summary=computed_summary,
            highlights=list(highlights or (previous.highlights if previous else [])),
        )
        self._records[document_id] = record
        self._prune_if_needed()
        return record

    def get(self, document_id: str) -> SummaryRecord | None:
        return self._records.get(document_id)

    def _prune_if_needed(self) -> None:
        if len(self._records) <= self._max_entries:
            return
        sorted_items = sorted(self._records.items(), key=lambda item: item[1].updated_at)
        while len(sorted_items) > self._max_entries:
            document_id, _ = sorted_items.pop(0)
            self._records.pop(document_id, None)

    def as_dict(self) -> dict[str, Any]:
        return {doc_id: record.to_dict() for doc_id, record in self._records.items()}

    def load_dict(self, payload: Mapping[str, Any]) -> None:
        for document_id, record in payload.items():
            if isinstance(record, Mapping):
                self._records[document_id] = SummaryRecord.from_dict(record)
        self._prune_if_needed()

    def __len__(self) -> int:
        return len(self._records)


class MemoryStore:
    """File-based persistence for memory buffers."""

    def __init__(self, storage_dir: Path) -> None:
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def save_conversation(self, document_id: str, memory: ConversationMemory) -> Path:
        path = self._conversation_path(document_id)
        payload = {"messages": [message.to_dict() for message in memory.get_messages()]}
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def load_conversation(
        self,
        document_id: str,
        *,
        conversation_factory: Callable[[], ConversationMemory] | None = None,
    ) -> ConversationMemory:
        path = self._conversation_path(document_id)
        memory = conversation_factory() if conversation_factory else ConversationMemory()
        if not path.exists():
            return memory
        payload = json.loads(path.read_text(encoding="utf-8"))
        memory.extend(payload.get("messages", []))
        return memory

    def save_document_summaries(self, memory: DocumentSummaryMemory) -> Path:
        path = self._storage_dir / "document_summaries.json"
        path.write_text(
            json.dumps(memory.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def load_document_summaries(self, memory: DocumentSummaryMemory) -> DocumentSummaryMemory:
        path = self._storage_dir / "document_summaries.json"
        if not path.exists():
            return memory
        payload = json.loads(path.read_text(encoding="utf-8"))
        memory.load_dict(payload)
        return memory

    def _conversation_path(self, document_id: str) -> Path:
        return self._storage_dir / f"{_safe_document_id(document_id)}.conversation.json"

