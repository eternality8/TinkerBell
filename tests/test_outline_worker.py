"""Tests for the outline builder worker and helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from tinkerbell.ai.memory.cache_bus import DocumentCacheBus, DocumentChangedEvent
from tinkerbell.ai.services import OutlineBuilderConfig, OutlineBuilderWorker, build_outline_nodes
from tinkerbell.editor.document_model import DocumentMetadata, DocumentState


def test_build_outline_nodes_handles_markdown() -> None:
    text = """# Title\n\n## Intro\nSome details\n\n## Usage\nSteps"""
    nodes = build_outline_nodes("doc-md", text, language="markdown")

    assert nodes
    assert nodes[0].text == "Title"
    assert nodes[0].children and nodes[0].children[0].text == "Intro"


def test_outline_worker_processes_cache_events(tmp_path: Path) -> None:
    async def _run() -> None:
        bus = DocumentCacheBus()
        doc = DocumentState(text="# Title\nBody", metadata=DocumentMetadata(language="markdown"))
        doc.document_id = "doc-worker"
        documents = {doc.document_id: doc}

        worker = OutlineBuilderWorker(
            document_provider=documents.get,
            storage_dir=tmp_path,
            cache_bus=bus,
            loop=asyncio.get_running_loop(),
            config=OutlineBuilderConfig(debounce_seconds=0.01, min_rebuild_interval=0.01),
        )

        bus.publish(
            DocumentChangedEvent(
                document_id=doc.document_id,
                version_id=doc.version_id,
                content_hash=doc.content_hash,
                edited_ranges=((0, len(doc.text)),),
            )
        )

        await asyncio.sleep(0.05)
        record = worker.memory.get(doc.document_id)
        assert record is not None
        assert record.nodes

        await worker.aclose()

        restarted = OutlineBuilderWorker(
            document_provider=documents.get,
            storage_dir=tmp_path,
            cache_bus=bus,
            loop=asyncio.get_running_loop(),
            config=OutlineBuilderConfig(debounce_seconds=0.01, min_rebuild_interval=0.01),
        )
        cached = restarted.memory.get(doc.document_id)
        assert cached is not None and cached.nodes
        await restarted.aclose()

    asyncio.run(_run())


def test_outline_worker_applies_huge_document_guardrail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def _run() -> None:
        bus = DocumentCacheBus()
        doc = DocumentState(text="# Heading\nBody", metadata=DocumentMetadata(language="markdown"))
        doc.document_id = "doc-huge"
        documents = {doc.document_id: doc}

        worker = OutlineBuilderWorker(
            document_provider=documents.get,
            storage_dir=tmp_path,
            cache_bus=bus,
            loop=asyncio.get_running_loop(),
            config=OutlineBuilderConfig(debounce_seconds=0.01, min_rebuild_interval=0.01),
        )
        monkeypatch.setattr("tinkerbell.ai.services.outline_worker.is_huge_document", lambda *_: True)
        monkeypatch.setattr("tinkerbell.ai.services.outline_worker._limit_outline_depth", lambda *_a, **_k: True)

        bus.publish(
            DocumentChangedEvent(
                document_id=doc.document_id,
                version_id=doc.version_id,
                content_hash=doc.content_hash,
                edited_ranges=((0, len(doc.text)),),
            )
        )

        await asyncio.sleep(0.05)
        record = worker.memory.get(doc.document_id)
        assert record is not None
        assert record.metadata.get("huge_document_guardrail") is True
        assert record.metadata.get("document_bytes")
        assert all(not node.children for node in record.nodes)

        await worker.aclose()

    asyncio.run(_run())


def test_outline_worker_records_unsupported_documents(tmp_path: Path) -> None:
    async def _run() -> None:
        bus = DocumentCacheBus()
        metadata = DocumentMetadata(language="markdown")
        metadata.path = Path("binary.pdf")
        doc = DocumentState(text="", metadata=metadata)
        doc.document_id = "doc-unsupported"
        documents = {doc.document_id: doc}

        worker = OutlineBuilderWorker(
            document_provider=documents.get,
            storage_dir=tmp_path,
            cache_bus=bus,
            loop=asyncio.get_running_loop(),
            config=OutlineBuilderConfig(debounce_seconds=0.01, min_rebuild_interval=0.01),
        )

        bus.publish(
            DocumentChangedEvent(
                document_id=doc.document_id,
                version_id=doc.version_id,
                content_hash=doc.content_hash,
                edited_ranges=((0, 1),),
            )
        )

        await asyncio.sleep(0.05)
        record = worker.memory.get(doc.document_id)
        assert record is not None
        assert record.summary == "Unsupported document format"
        reason = record.metadata.get("unsupported_format")
        assert isinstance(reason, str) and "binary" in reason
        assert record.nodes == []

        await worker.aclose()

    asyncio.run(_run())


def test_outline_worker_rebuilds_when_cache_hash_mismatches(tmp_path: Path) -> None:
    async def _run() -> None:
        bus = DocumentCacheBus()
        doc = DocumentState(text="# Title\nBody", metadata=DocumentMetadata(language="markdown"))
        doc.document_id = "doc-cache"
        documents = {doc.document_id: doc}

        cache_dir = tmp_path / "outline_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        nodes = build_outline_nodes(doc.document_id, doc.text, language="markdown")
        payload = {
            "document_id": doc.document_id,
            "summary": "Cached",
            "highlights": ["Title"],
            "updated_at": doc.metadata.updated_at.isoformat(),
            "version_id": doc.version_id,
            "outline_hash": "mismatch",
            "content_hash": doc.content_hash,
            "nodes": [node.to_dict() for node in nodes],
            "metadata": {},
        }
        path = cache_dir / f"{doc.document_id}.outline.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        worker = OutlineBuilderWorker(
            document_provider=documents.get,
            storage_dir=tmp_path,
            cache_bus=bus,
            loop=asyncio.get_running_loop(),
            config=OutlineBuilderConfig(debounce_seconds=0.01, min_rebuild_interval=0.01),
        )

        assert worker.memory.get(doc.document_id) is None
        pending = getattr(worker, "_pending")
        assert doc.document_id in pending

        await worker.aclose()

    asyncio.run(_run())
