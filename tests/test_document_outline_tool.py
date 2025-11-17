"""Unit tests for :mod:`tinkerbell.ai.tools.document_outline`."""

from __future__ import annotations

from pathlib import Path
from typing import Callable
from tinkerbell.ai.memory.buffers import DocumentSummaryMemory, OutlineNode
from tinkerbell.ai.tools.document_outline import DocumentOutlineTool
from tinkerbell.editor.document_model import DocumentMetadata, DocumentState
from tinkerbell.services import telemetry as telemetry_service


def _build_document(document_id: str = "doc-1", version: int = 2) -> DocumentState:
    metadata = DocumentMetadata()
    return DocumentState(text="Sample document", metadata=metadata, document_id=document_id, version_id=version)


def _build_nodes(*token_estimates: int) -> list[OutlineNode]:
    estimates = token_estimates or (6, 4)
    root = OutlineNode(
        id="node-root",
        parent_id=None,
        level=1,
        text="Intro",
        char_range=(0, 12),
        token_estimate=estimates[0],
        blurb="Intro section",
    )
    child = OutlineNode(
        id="node-child",
        parent_id="node-root",
        level=2,
        text="Details",
        char_range=(12, 32),
        token_estimate=estimates[1] if len(estimates) > 1 else 4,
        blurb="Detail paragraph",
    )
    root.children.append(child)
    return [root]


def _build_memory(nodes: list[OutlineNode], *, document_id: str = "doc-1", version: int = 2, outline_hash: str = "digest-1") -> DocumentSummaryMemory:
    memory = DocumentSummaryMemory()
    memory.update(
        document_id,
        summary="Outline summary",
        nodes=nodes,
        version_id=version,
        outline_hash=outline_hash,
    )
    return memory


def _lookup_factory(document: DocumentState) -> Callable[[str], DocumentState | None]:
    def lookup(doc_id: str) -> DocumentState | None:
        return document if doc_id == document.document_id else None

    return lookup


def test_document_outline_tool_returns_nodes_with_digest_and_pointers() -> None:
    document = _build_document()
    nodes = _build_nodes(5, 3)
    memory = _build_memory(nodes)
    tool = DocumentOutlineTool(
        memory_resolver=lambda: memory,
        document_lookup=_lookup_factory(document),
        active_document_provider=lambda: document,
    )

    result = tool.run(document_id=document.document_id, include_blurbs=True)

    assert result["status"] == "ok"
    assert result["outline_digest"] == "digest-1"
    assert result["node_count"] >= 1
    pointer_id = result["nodes"][0]["pointer_id"]
    assert pointer_id.startswith(f"outline:{document.document_id}:")


def test_document_outline_tool_marks_stale_when_version_differs() -> None:
    document = _build_document(version=5)
    nodes = _build_nodes()
    memory = _build_memory(nodes, version=2)
    tool = DocumentOutlineTool(
        memory_resolver=lambda: memory,
        document_lookup=_lookup_factory(document),
        active_document_provider=lambda: document,
    )

    result = tool.run(document_id=document.document_id)

    assert result["status"] == "stale"
    assert result["is_stale"] is True
    assert result["stale_delta"] == 3


def test_document_outline_tool_trims_outline_to_token_budget() -> None:
    document = _build_document()
    nodes = _build_nodes(6, 6)
    memory = _build_memory(nodes)
    tool = DocumentOutlineTool(
        memory_resolver=lambda: memory,
        document_lookup=_lookup_factory(document),
        active_document_provider=lambda: document,
        max_outline_tokens=8,
    )

    result = tool.run(document_id=document.document_id)

    assert result["trimmed"] is True
    assert result["token_count"] <= 8
    assert result["nodes"][0]["children"] == []


def test_document_outline_tool_emits_hit_event() -> None:
    document = _build_document()
    nodes = _build_nodes(4, 2)
    memory = _build_memory(nodes)
    tool = DocumentOutlineTool(
        memory_resolver=lambda: memory,
        document_lookup=_lookup_factory(document),
        active_document_provider=lambda: document,
    )
    events: list[dict[str, object]] = []
    telemetry_service.register_event_listener("outline.tool.hit", lambda payload: events.append(payload))

    result = tool.run(document_id=document.document_id)

    assert result["status"] == "ok"
    assert events
    payload = events[-1]
    assert payload["document_id"] == document.document_id
    assert payload["token_count"] == result["token_count"]


def test_document_outline_tool_emits_stale_event_when_versions_differ() -> None:
    document = _build_document(version=4)
    nodes = _build_nodes()
    memory = _build_memory(nodes, version=1)
    tool = DocumentOutlineTool(
        memory_resolver=lambda: memory,
        document_lookup=_lookup_factory(document),
        active_document_provider=lambda: document,
    )
    events: list[dict[str, object]] = []
    telemetry_service.register_event_listener("outline.stale", lambda payload: events.append(payload))

    result = tool.run(document_id=document.document_id)

    assert result["status"] == "stale"
    assert events
    payload = events[-1]
    assert payload["document_id"] == document.document_id
    record = memory.get(document.document_id)
    assert record is not None
    assert payload["outline_version_id"] == record.version_id


def test_document_outline_tool_emits_miss_event_when_outline_missing() -> None:
    document = _build_document()
    memory = DocumentSummaryMemory()
    tool = DocumentOutlineTool(
        memory_resolver=lambda: memory,
        document_lookup=_lookup_factory(document),
        active_document_provider=lambda: document,
    )
    events: list[dict[str, object]] = []
    telemetry_service.register_event_listener("outline.tool.miss", lambda payload: events.append(payload))

    result = tool.run(document_id=document.document_id)

    assert result["status"] == "outline_missing"
    assert events
    payload = events[-1]
    assert payload["status"] == "outline_missing"


def test_document_outline_tool_surfaces_guardrail_metadata() -> None:
    document = _build_document()
    nodes = _build_nodes()
    memory = _build_memory(nodes)
    record = memory.get(document.document_id)
    assert record is not None
    record.metadata["huge_document_guardrail"] = True
    record.metadata["document_bytes"] = 6_500_000
    tool = DocumentOutlineTool(
        memory_resolver=lambda: memory,
        document_lookup=_lookup_factory(document),
        active_document_provider=lambda: document,
    )

    result = tool.run(document_id=document.document_id)

    assert result["guardrails"][0]["type"] == "huge_document"
    assert result["outline_metadata"]["huge_document_guardrail"] is True
    assert result["document_bytes"] == 6_500_000


def test_document_outline_tool_returns_pending_status() -> None:
    document = _build_document(version=6)
    nodes = _build_nodes()
    memory = _build_memory(nodes, version=2)
    tool = DocumentOutlineTool(
        memory_resolver=lambda: memory,
        document_lookup=_lookup_factory(document),
        active_document_provider=lambda: document,
        pending_outline_checker=lambda _doc_id: True,
    )

    result = tool.run(document_id=document.document_id)

    assert result["status"] == "pending"
    assert result["retry_after_ms"] >= 250


def test_document_outline_tool_detects_unsupported_documents() -> None:
    metadata = DocumentMetadata()
    metadata.path = Path("manual.pdf")
    document = DocumentState(text="", metadata=metadata, document_id="doc-pdf")
    memory = DocumentSummaryMemory()
    tool = DocumentOutlineTool(
        memory_resolver=lambda: memory,
        document_lookup=_lookup_factory(document),
        active_document_provider=lambda: document,
    )

    result = tool.run(document_id=document.document_id)

    assert result["status"] == "unsupported_format"
    assert result["outline_available"] is False
