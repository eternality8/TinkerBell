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
                "span_overlap": True,
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


# ---------------------------------------------------------------------------
# WS3 4.3.x: Chunk cache recovery tests
# ---------------------------------------------------------------------------

def test_document_chunk_tool_retry_hint_on_not_found():
    """WS3 4.3.1: Cache miss returns retry_hint."""
    text = "Lorem ipsum"
    bridge = _BridgeStub(text)
    index = ChunkIndex()
    tool = DocumentChunkTool(bridge=bridge, chunk_index=index)

    result = tool.run(chunk_id="missing-chunk", document_id="doc-unknown")

    assert result["status"] == "not_found"
    assert "retry_hint" in result
    assert "snapshot" in result["retry_hint"].lower()


def test_document_chunk_tool_auto_refresh_recovers_on_miss():
    """WS3 4.3.2: Auto-refresh can recover from cache miss."""

    class _RefreshingBridgeStub(_BridgeStub):
        def __init__(self, text: str, index: ChunkIndex) -> None:
            super().__init__(text)
            self._index = index
            self._manifest_created = False

        def generate_snapshot(
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
            # On first call, create the manifest so chunk can be found
            if not self._manifest_created:
                manifest = _manifest("doc-refresh", [(0, len(self.text))], cache_key="refreshed-cache")
                self._index.ingest_manifest(manifest)
                self._manifest_created = True
                return {
                    "text": self.text,
                    "chunk_manifest": {"cache_key": "refreshed-cache"},
                }
            return super().generate_snapshot(
                delta_only=delta_only,
                tab_id=tab_id,
                include_open_documents=include_open_documents,
                window=window,
                chunk_profile=chunk_profile,
                max_tokens=max_tokens,
                include_text=include_text,
            )

    text = "Recovery test content"
    index = ChunkIndex()
    bridge = _RefreshingBridgeStub(text, index)
    tool = DocumentChunkTool(bridge=bridge, chunk_index=index, auto_refresh_on_miss=True)

    # Request a chunk that doesn't exist yet - should trigger auto-refresh
    result = tool.run(chunk_id="chunk:doc-refresh:0:21", document_id="doc-refresh")

    assert result.get("recovered") is True
    assert result["status"] == "recovered"
    assert result["chunk"]["text"] == text


def test_document_chunk_tool_no_auto_refresh_when_disabled():
    """WS3 4.3.2: Auto-refresh is disabled by default."""
    text = "Lorem ipsum"
    bridge = _BridgeStub(text)
    index = ChunkIndex()
    tool = DocumentChunkTool(bridge=bridge, chunk_index=index, auto_refresh_on_miss=False)

    result = tool.run(chunk_id="missing-chunk", document_id="doc-missing")

    assert result["status"] == "not_found"
    assert result.get("recovered") is None