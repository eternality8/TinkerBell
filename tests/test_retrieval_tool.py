"""Tests for the DocumentFindSectionsTool."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from tinkerbell.ai.memory.buffers import DocumentSummaryMemory, OutlineNode
from tinkerbell.ai.memory.embeddings import (
    ChunkEmbeddingRecord,
    DocumentEmbeddingIndex,
    EmbeddingMatch,
)
from tinkerbell.ai.tools.document_find_sections import DocumentFindSectionsTool
from tinkerbell.editor.document_model import DocumentMetadata, DocumentState
from tinkerbell.services import telemetry as telemetry_service


class _StubEmbeddingIndex:
    def __init__(self, matches: list[EmbeddingMatch]) -> None:
        self.matches = matches
        self.calls: list[tuple[str, str, int, float]] = []
        self.provider_name = "stub"

    async def similarity_search(
        self,
        document_id: str,
        *,
        query_text: str | None = None,
        query_vector=None,
        top_k: int = 6,
        min_score: float = 0.0,
    ) -> list[EmbeddingMatch]:
        self.calls.append((document_id, query_text or "", top_k, min_score))
        return [match for match in self.matches if match.score >= min_score]


def test_retrieval_tool_returns_embedding_matches() -> None:
    document = _build_document("doc-embed", "Alpha beta gamma delta epsilon")
    nodes = _build_outline_nodes()
    memory = DocumentSummaryMemory()
    memory.update(
        document.document_id,
        summary="Summary",
        nodes=nodes,
        version_id=document.version_id,
        outline_hash="outline-hash",
    )
    record = _build_chunk_record(document.document_id, chunk_id="chunk-1", start=0, end=25, outline_node_id=nodes[0].id)
    matches = [EmbeddingMatch(record=record, score=0.92)]
    stub = _StubEmbeddingIndex(matches)
    index = cast(DocumentEmbeddingIndex, stub)

    tool = DocumentFindSectionsTool(
        embedding_index=index,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
        outline_memory=memory,
    )

    response = tool.run(document_id=document.document_id, query="alpha beta", top_k=4, min_confidence=0.2)

    assert response["status"] == "ok"
    assert response["strategy"] == "embedding"
    assert response["pointers"]
    pointer = response["pointers"][0]
    assert pointer["chunk_id"] == "chunk-1"
    assert pointer["outline_context"]["node_id"] == nodes[0].id
    assert stub.calls == [(document.document_id, "alpha beta", 4, 0.2)]


def test_retrieval_tool_falls_back_without_embeddings() -> None:
    document = _build_document("doc-fallback", "Heading\nAlpha beta body text.")
    tool = DocumentFindSectionsTool(
        embedding_index=None,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
    )

    response = tool.run(document_id=document.document_id, query="alpha", top_k=2)

    assert response["strategy"] == "fallback"
    assert response["status"] == "offline_fallback"
    assert response["pointers"]
    assert response["pointers"][0]["match_type"] in {"regex", "document"}
    assert response["offline_mode"] is True


def test_retrieval_tool_fallback_when_no_match() -> None:
    document = _build_document("doc-low", "Alpha beta gamma delta")
    record = _build_chunk_record(document.document_id, chunk_id="chunk-1", start=0, end=15)
    matches = [EmbeddingMatch(record=record, score=0.2)]
    stub = _StubEmbeddingIndex(matches)
    index = cast(DocumentEmbeddingIndex, stub)
    tool = DocumentFindSectionsTool(
        embedding_index=index,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
    )

    response = tool.run(document_id=document.document_id, query="alpha", min_confidence=0.9)

    assert response["strategy"] == "fallback"
    assert response["fallback_reason"] == "no_embedding_matches"
    assert response["pointers"]


def test_retrieval_tool_handles_missing_query() -> None:
    document = _build_document("doc-missing", "Alpha")
    tool = DocumentFindSectionsTool(document_lookup=lambda doc_id: document)

    response = tool.run(document_id=document.document_id, query="   ")

    assert response["status"] == "invalid_request"


def test_retrieval_tool_handles_missing_document() -> None:
    tool = DocumentFindSectionsTool(document_lookup=lambda _: None)

    response = tool.run(document_id="missing", query="alpha")

    assert response["status"] == "no_document"


def test_retrieval_tool_emits_retrieval_event() -> None:
    document = _build_document("doc-telemetry", "Alpha beta gamma delta")
    record = _build_chunk_record(document.document_id, chunk_id="chunk-telemetry", start=0, end=20)
    matches = [EmbeddingMatch(record=record, score=0.91)]
    index = cast(DocumentEmbeddingIndex, _StubEmbeddingIndex(matches))
    events: list[dict[str, object]] = []
    telemetry_service.register_event_listener("retrieval.query", lambda payload: events.append(payload))

    tool = DocumentFindSectionsTool(
        embedding_index=index,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
    )

    response = tool.run(document_id=document.document_id, query="alpha", top_k=3)

    assert response["status"] == "ok"
    assert events
    payload = events[-1]
    assert payload["document_id"] == document.document_id
    assert payload["pointer_count"] == len(response["pointers"])
    saved = payload.get("tokens_saved")
    assert isinstance(saved, (int, float))
    assert saved >= 0


def test_retrieval_tool_reports_provider_error_and_falls_back() -> None:
    document = _build_document("doc-error", "Alpha beta gamma delta")
    events: list[dict[str, object]] = []
    telemetry_service.register_event_listener("retrieval.provider.error", lambda payload: events.append(payload))
    index = cast(DocumentEmbeddingIndex, _ErrorEmbeddingIndex())
    tool = DocumentFindSectionsTool(
        embedding_index=index,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
    )

    response = tool.run(document_id=document.document_id, query="alpha")

    assert response["strategy"] == "fallback"
    assert response["fallback_reason"] == "provider_error"
    assert events
    payload = events[-1]
    assert payload["document_id"] == document.document_id
    assert payload["provider"] == "error-stub"


def test_retrieval_tool_detects_unsupported_documents() -> None:
    metadata = DocumentMetadata()
    metadata.path = Path("manual.pdf")
    document = DocumentState(text="", metadata=metadata, document_id="doc-unsupported")
    tool = DocumentFindSectionsTool(
        embedding_index=None,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
    )

    response = tool.run(document_id=document.document_id, query="anything")

    assert response["status"] == "unsupported_format"
    assert response["document_id"] == document.document_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_document(document_id: str, text: str) -> DocumentState:
    return DocumentState(text=text, document_id=document_id)


def _build_outline_nodes() -> list[OutlineNode]:
    node = OutlineNode(
        id="node-0",
        parent_id=None,
        level=1,
        text="Heading",
        char_range=(0, 30),
        chunk_id="chunk-1",
        blurb="Heading blurb",
        token_estimate=10,
    )
    return [node]


def _build_chunk_record(
    document_id: str,
    *,
    chunk_id: str,
    start: int,
    end: int,
    outline_node_id: str | None = None,
) -> ChunkEmbeddingRecord:
    return ChunkEmbeddingRecord(
        document_id=document_id,
        chunk_id=chunk_id,
        version_id=1,
        content_hash="hash",
        chunk_hash="chunk-hash",
        start_offset=start,
        end_offset=end,
        outline_node_id=outline_node_id,
        token_count=12,
        outline_hash="outline-hash",
        provider="stub",
        dims=3,
        vector=(0.1, 0.2, 0.3),
    )


class _ErrorEmbeddingIndex:
    provider_name = "error-stub"

    async def similarity_search(
        self,
        document_id: str,
        *,
        query_text: str | None = None,
        query_vector=None,
        top_k: int = 6,
        min_score: float = 0.0,
    ) -> list[EmbeddingMatch]:
        raise RuntimeError("provider boom")
