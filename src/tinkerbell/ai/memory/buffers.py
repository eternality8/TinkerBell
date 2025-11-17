"""Conversation and summary memory helpers used by the agent graph."""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, cast

from openai.types.chat import ChatCompletionMessageParam

@dataclass(slots=True)
class OutlineNode:
    """Represents a single outline node within a document hierarchy."""

    id: str
    parent_id: str | None
    level: int
    text: str
    char_range: tuple[int, int]
    chunk_id: str | None = None
    blurb: str = ""
    token_estimate: int = 0
    truncated: bool = False
    children: list["OutlineNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "parent_id": self.parent_id,
            "level": self.level,
            "text": self.text,
            "char_range": list(self.char_range),
            "chunk_id": self.chunk_id,
            "blurb": self.blurb,
            "token_estimate": self.token_estimate,
            "truncated": self.truncated,
            "children": [child.to_dict() for child in self.children],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OutlineNode":
        children_payload = payload.get("children", [])
        char_values = list(payload.get("char_range", (0, 0)))
        start = int(char_values[0]) if char_values else 0
        end = int(char_values[1]) if len(char_values) > 1 else start
        node = cls(
            id=str(payload.get("id", "")),
            parent_id=payload.get("parent_id"),
            level=int(payload.get("level", 0)),
            text=str(payload.get("text", "")),
            char_range=(start, end),
            chunk_id=payload.get("chunk_id"),
            blurb=str(payload.get("blurb", "")),
            token_estimate=int(payload.get("token_estimate", 0)),
            truncated=bool(payload.get("truncated", False)),
        )
        node.children = [OutlineNode.from_dict(child) for child in children_payload if isinstance(child, Mapping)]
        return node

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
    version_id: int | None = None
    outline_hash: str | None = None
    nodes: list[OutlineNode] = field(default_factory=list)
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "summary": self.summary,
            "highlights": list(self.highlights),
            "updated_at": self.updated_at.isoformat(),
            "version_id": self.version_id,
            "outline_hash": self.outline_hash,
            "content_hash": self.content_hash,
            "nodes": [node.to_dict() for node in self.nodes],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SummaryRecord:
        updated = payload.get("updated_at")
        timestamp = (
            datetime.fromisoformat(updated)
            if isinstance(updated, str)
            else _utcnow()
        )
        nodes_payload = payload.get("nodes", [])
        return cls(
            document_id=str(payload.get("document_id", "unknown")),
            summary=str(payload.get("summary", "")),
            highlights=list(payload.get("highlights", [])),
            updated_at=timestamp,
            version_id=payload.get("version_id"),
            outline_hash=payload.get("outline_hash"),
            nodes=[OutlineNode.from_dict(node) for node in nodes_payload if isinstance(node, Mapping)],
            content_hash=payload.get("content_hash"),
            metadata=dict(payload.get("metadata", {})),
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
        version_id: int | None = None,
        outline_hash: str | None = None,
        nodes: Sequence[OutlineNode | Mapping[str, Any]] | None = None,
        content_hash: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SummaryRecord:
        if not summary and not text:
            raise ValueError("Either summary or text must be provided to update memory")
        previous = self._records.get(document_id)
        computed_summary = summary or self._summarizer(previous.summary if previous else None, text or "")
        if self._max_summary_chars:
            computed_summary = computed_summary[: self._max_summary_chars]
        outline_nodes: list[OutlineNode] | None = None
        if nodes is not None:
            outline_nodes = []
            for node in nodes:
                if isinstance(node, OutlineNode):
                    outline_nodes.append(node)
                elif isinstance(node, Mapping):
                    outline_nodes.append(OutlineNode.from_dict(node))
        fallback_nodes = list(previous.nodes) if previous and previous.nodes else []
        merged_metadata: dict[str, Any] = {}
        if previous and previous.metadata:
            merged_metadata.update(previous.metadata)
        if metadata:
            for key, value in metadata.items():
                merged_metadata[str(key)] = value
        if highlights is not None:
            highlight_values = list(highlights)
        elif previous:
            highlight_values = list(previous.highlights)
        else:
            highlight_values = []

        record = SummaryRecord(
            document_id=document_id,
            summary=computed_summary,
            highlights=highlight_values,
            version_id=version_id if version_id is not None else getattr(previous, "version_id", None),
            outline_hash=outline_hash if outline_hash is not None else getattr(previous, "outline_hash", None),
            nodes=outline_nodes if outline_nodes is not None else fallback_nodes,
            content_hash=content_hash if content_hash is not None else getattr(previous, "content_hash", None),
            metadata=merged_metadata,
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


class OutlineCacheStore:
    """Persist outline payloads per document for fast cold-start hydration."""

    def __init__(self, cache_dir: Path) -> None:
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: SummaryRecord) -> Path:
        payload = record.to_dict()
        path = self._path_for(record.document_id)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
        return path

    def load(self, document_id: str) -> SummaryRecord | None:
        path = self._path_for(document_id)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return SummaryRecord.from_dict(payload)

    def load_all(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for path in sorted(self._cache_dir.glob("*.outline.json")):
            try:
                record_payload = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            document_id = str(record_payload.get("document_id")) or path.stem
            payload[document_id] = record_payload
        return payload

    def delete(self, document_id: str) -> None:
        path = self._path_for(document_id)
        with contextlib.suppress(FileNotFoundError):
            path.unlink()

    def _path_for(self, document_id: str) -> Path:
        safe_id = _safe_document_id(document_id)
        return self._cache_dir / f"{safe_id}.outline.json"

