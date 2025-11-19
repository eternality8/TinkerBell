"""Unit tests for DocumentChunkTool."""

from __future__ import annotations

from typing import Any, Mapping

from tinkerbell.ai.memory.chunk_index import ChunkIndex
from tinkerbell.ai.tools.document_chunk import DocumentChunkTool


class _BridgeStub:
    def __init__(self, text: str) -> None:
        self.text = text
        self.requests: list[Mapping[str, Any]] = []

    def generate_snapshot(  # type: ignore[override]
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_open_documents: bool = False,
        window: Mapping[str, Any] | str | None = None,
        chunk_profile: str | None = None,
        max_tokens: int | None = None,
        include_text: bool = True,
    ) -> Mapping[str, Any]:
        del delta_only, tab_id, include_open_documents, chunk_profile, max_tokens, include_text
        start = 0
        end = len(self.text)
        if isinstance(window, Mapping):
            start = int(window.get("start", 0))
            end = int(window.get("end", end))
        self.requests.append({"start": start, "end": end})
        return {"text": self.text[start:end]}

    def get_last_diff_summary(self, tab_id: str | None = None) -> str | None:  # noqa: ARG002
        return None

    def get_last_snapshot_version(self, tab_id: str | None = None) -> str | None:  # noqa: ARG002
        return None


def _manifest(document_id: str, chunks: list[tuple[int, int]], *, cache_key: str = "cache-1") -> dict[str, Any]:
    entries = []
    for idx, (start, end) in enumerate(chunks):
        entries.append(
            {
                "id": f"chunk:{document_id}:{start}:{end}",
                "start": start,
                "end": end,
                "length": end - start,
                "hash": f"hash-{idx}",
                "selection_overlap": True,
                "outline_pointer_id": None,
            }
        )
    return {
        "document_id": document_id,
        "chunk_profile": "auto",
        "chunk_chars": 2_048,
        "chunk_overlap": 256,
        "window": {"start": 0, "end": chunks[-1][1], "length": chunks[-1][1]},
        "chunks": entries,
        "cache_key": cache_key,
        "version": f"{document_id}:1:hash",
        "content_hash": "hash",
        "generated_at": 123.0,
    }


def test_document_chunk_tool_returns_inline_text():
    text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    bridge = _BridgeStub(text)
    index = ChunkIndex()
    manifest = _manifest("doc-1", [(0, 20), (20, 44)])
    entries = index.ingest_manifest(manifest)
    tool = DocumentChunkTool(bridge=bridge, chunk_index=index)

    result = tool.run(chunk_id=entries[0].chunk_id, cache_key=manifest["cache_key"], document_id="doc-1")

    assert result["status"] == "ok"
    chunk = result["chunk"]
    assert chunk["text"] == text[:20]
    assert chunk["pointer"] is None


def test_document_chunk_tool_pointerizes_when_over_cap():
    text = "alpha " * 400
    bridge = _BridgeStub(text)
    index = ChunkIndex()
    manifest = _manifest("doc-2", [(0, len(text))])
    entry = index.ingest_manifest(manifest)[0]
    tool = DocumentChunkTool(
        bridge=bridge,
        chunk_index=index,
        chunk_config_resolver=lambda: {
            "default_profile": "auto",
            "overlap_chars": 128,
            "max_inline_tokens": 50,
            "iterator_limit": 2,
        },
    )

    result = tool.run(chunk_id=entry.chunk_id, cache_key=manifest["cache_key"], document_id="doc-2")

    assert result["status"] == "pointer_only"
    chunk = result["chunk"]
    assert chunk["text"] == ""
    assert chunk["pointer"]["reason"] == "inline_cap_exceeded"


def test_document_chunk_tool_iterator_fetches_multiple_chunks():
    text = "The quick brown fox jumps over the lazy dog"
    bridge = _BridgeStub(text)
    index = ChunkIndex()
    manifest = _manifest("doc-3", [(0, 9), (9, 19), (19, len(text))], cache_key="cursor-1")
    index.ingest_manifest(manifest)
    tool = DocumentChunkTool(bridge=bridge, chunk_index=index)

    result = tool.run(iterator={"cache_key": "cursor-1", "limit": 2})

    assert result["status"] == "iterator"
    iterator = result["iterator"]
    assert iterator["count"] == 2
    assert iterator["has_more"] is True
    chunk_texts = [chunk["text"] for chunk in iterator["chunks"]]
    assert chunk_texts[0] == text[:9]
    assert chunk_texts[1] == text[9:19]
    assert iterator["next_chunk_id"].startswith("chunk:doc-3:")