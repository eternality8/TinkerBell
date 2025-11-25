"""Tests for the DocumentFindTextTool."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest

from tinkerbell.ai.memory.buffers import DocumentSummaryMemory, OutlineNode
from tinkerbell.ai.memory.embeddings import (
    ChunkEmbeddingRecord,
    DocumentEmbeddingIndex,
    EmbeddingMatch,
    LocalEmbeddingProvider,
)
from tinkerbell.ai.tools.document_find_text import DocumentFindTextTool
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

    tool = DocumentFindTextTool(
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
    tool = DocumentFindTextTool(
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
    tool = DocumentFindTextTool(
        embedding_index=index,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
    )

    response = tool.run(document_id=document.document_id, query="alpha", min_confidence=0.9)

    assert response["strategy"] == "fallback"
    assert response["fallback_reason"] == "no_embedding_matches"
    assert response["pointers"]


def test_retrieval_tool_handles_missing_query() -> None:
    document = _build_document("doc-missing", "Alpha")
    tool = DocumentFindTextTool(document_lookup=lambda doc_id: document)

    response = tool.run(document_id=document.document_id, query="   ")

    assert response["status"] == "invalid_request"


def test_retrieval_tool_handles_missing_document() -> None:
    tool = DocumentFindTextTool(document_lookup=lambda _: None)

    response = tool.run(document_id="missing", query="alpha")

    assert response["status"] == "no_document"


def test_retrieval_tool_emits_retrieval_event() -> None:
    document = _build_document("doc-telemetry", "Alpha beta gamma delta")
    record = _build_chunk_record(document.document_id, chunk_id="chunk-telemetry", start=0, end=20)
    matches = [EmbeddingMatch(record=record, score=0.91)]
    index = cast(DocumentEmbeddingIndex, _StubEmbeddingIndex(matches))
    events: list[dict[str, object]] = []
    telemetry_service.register_event_listener("retrieval.query", lambda payload: events.append(payload))

    tool = DocumentFindTextTool(
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
    tool = DocumentFindTextTool(
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


@pytest.mark.asyncio
async def test_retrieval_tool_ingests_with_local_embedding_provider(tmp_path: Path) -> None:
    document = DocumentState(
        text="""## Intro\nAlpha beta text\n## Details\nGamma delta epsilon""",
        document_id="doc-local",
    )
    nodes = _build_local_outline_nodes(document)
    memory = DocumentSummaryMemory()
    memory.update(
        document.document_id,
        summary="Summary",
        nodes=nodes,
        version_id=document.version_id,
        outline_hash="outline-local",
    )

    def _vectorize(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if "gamma" in lowered else 0.0, float(len(text))])
        return vectors

    provider = LocalEmbeddingProvider(embed_batch=_vectorize, name="local:dummy", max_batch_size=4)
    index = DocumentEmbeddingIndex(
        storage_dir=tmp_path,
        provider=provider,
        loop=asyncio.get_running_loop(),
        mode="local",
        provider_label="SentenceTransformers",
    )

    ingest = await index.ingest_outline(document=document, nodes=nodes, outline_hash="outline-local")
    assert ingest.embedded == len(nodes)

    tool = DocumentFindTextTool(
        embedding_index=index,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
        outline_memory=memory,
    )

    response = await asyncio.to_thread(tool.run, document_id=document.document_id, query="Gamma delta")

    assert response["status"] == "ok"
    assert response["strategy"] == "embedding"
    assert response["offline_mode"] is False
    assert response["pointers"]
    pointer = response["pointers"][0]
    assert pointer["outline_node_id"] == nodes[1].id
    outline_context = pointer.get("outline_context", {})
    assert outline_context.get("node_id") == nodes[1].id

    await index.aclose()


def test_retrieval_tool_detects_unsupported_documents() -> None:
    metadata = DocumentMetadata()
    metadata.path = Path("manual.pdf")
    document = DocumentState(text="", metadata=metadata, document_id="doc-unsupported")
    tool = DocumentFindTextTool(
        embedding_index=None,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
    )

    response = tool.run(document_id=document.document_id, query="anything")

    assert response["status"] == "unsupported_format"
    assert response["document_id"] == document.document_id


# ---------------------------------------------------------------------------
# WS3 4.1.x: Confidence field tests
# ---------------------------------------------------------------------------

def test_retrieval_tool_confidence_high_with_embedding_match() -> None:
    """WS3 4.1.3: High confidence when embedding strategy returns matches."""
    document = _build_document("doc-conf-high", "Alpha beta gamma delta epsilon")
    record = _build_chunk_record(document.document_id, chunk_id="c1", start=0, end=20)
    matches = [EmbeddingMatch(record=record, score=0.9)]
    index = cast(DocumentEmbeddingIndex, _StubEmbeddingIndex(matches))
    tool = DocumentFindTextTool(
        embedding_index=index,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
    )

    response = tool.run(document_id=document.document_id, query="alpha", min_confidence=0.3)

    assert response["confidence"] == "high"
    assert response["warning"] is None


def test_retrieval_tool_confidence_low_with_fallback() -> None:
    """WS3 4.1.3: Low confidence when using fallback strategy."""
    document = _build_document("doc-conf-low", "Alpha beta gamma")
    # No embedding matches above threshold -> fallback
    record = _build_chunk_record(document.document_id, chunk_id="c1", start=0, end=10)
    matches = [EmbeddingMatch(record=record, score=0.1)]  # Below default threshold
    index = cast(DocumentEmbeddingIndex, _StubEmbeddingIndex(matches))
    tool = DocumentFindTextTool(
        embedding_index=index,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
    )

    response = tool.run(document_id=document.document_id, query="alpha", min_confidence=0.5)

    assert response["confidence"] == "low"
    assert response["warning"] is not None
    assert "fallback" in response["warning"].lower() or "semantic" in response["warning"].lower()


def test_retrieval_tool_confidence_low_offline_mode() -> None:
    """WS3 4.1.3: Low confidence in offline mode (no embedding index)."""
    document = _build_document("doc-offline", "Alpha beta gamma")
    tool = DocumentFindTextTool(
        embedding_index=None,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
    )

    response = tool.run(document_id=document.document_id, query="alpha")

    assert response["confidence"] == "low"
    assert response["offline_mode"] is True
    assert response["warning"] is not None
    assert "unavailable" in response["warning"].lower()


# ---------------------------------------------------------------------------
# WS3 4.2.x: Line span tests
# ---------------------------------------------------------------------------

def test_retrieval_tool_pointers_include_line_span() -> None:
    """WS3 4.2.3: Embedding pointers have line_span with start_line/end_line."""
    document = _build_document("doc-lines", "Line 0\nLine 1\nLine 2\nLine 3\n")
    # Chunk covers "Line 1\nLine 2" (offsets 7-21)
    record = _build_chunk_record(document.document_id, chunk_id="c1", start=7, end=21)
    matches = [EmbeddingMatch(record=record, score=0.9)]
    index = cast(DocumentEmbeddingIndex, _StubEmbeddingIndex(matches))
    tool = DocumentFindTextTool(
        embedding_index=index,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
    )

    response = tool.run(document_id=document.document_id, query="line", min_confidence=0.3)

    assert response["pointers"]
    pointer = response["pointers"][0]
    assert "line_span" in pointer
    assert pointer["line_span"]["start_line"] == 1
    assert pointer["line_span"]["end_line"] == 2


def test_retrieval_tool_fallback_pointers_include_line_span() -> None:
    """WS3 4.2.3: Fallback pointers also have line_span."""
    document = _build_document("doc-lines-fb", "First line\nSecond line\nThird line\n")
    tool = DocumentFindTextTool(
        embedding_index=None,
        document_lookup=lambda doc_id: document if doc_id == document.document_id else None,
    )

    response = tool.run(document_id=document.document_id, query="Second")

    assert response["pointers"]
    pointer = response["pointers"][0]
    assert "line_span" in pointer
    # Fallback extends preview window, so line_span starts at 0 (whole text fits in preview)
    assert "start_line" in pointer["line_span"]
    assert "end_line" in pointer["line_span"]
    # line_span values should be consistent with char_range
    assert pointer["line_span"]["start_line"] >= 0
    assert pointer["line_span"]["end_line"] >= pointer["line_span"]["start_line"]


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


def _build_local_outline_nodes(document: DocumentState) -> list[OutlineNode]:
    text = document.text
    divider = text.index("## Details")
    intro_range = (0, divider)
    details_range = (divider, len(text))
    return [
        OutlineNode(
            id="local-node-0",
            parent_id=None,
            level=1,
            text="Intro",
            char_range=intro_range,
            chunk_id=f"{document.document_id}-intro",
            blurb="Intro section",
            token_estimate=12,
        ),
        OutlineNode(
            id="local-node-1",
            parent_id=None,
            level=1,
            text="Details",
            char_range=details_range,
            chunk_id=f"{document.document_id}-details",
            blurb="Details section",
            token_estimate=16,
        ),
    ]


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
