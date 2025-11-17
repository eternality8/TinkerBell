from __future__ import annotations

from tinkerbell.ai.memory.cache_bus import DocumentCacheBus
from tinkerbell.ai.memory.plot_state import DocumentPlotStateStore
from tinkerbell.ai.tools.document_plot_state import DocumentPlotStateTool
from tinkerbell.editor.document_model import DocumentState


def _build_store_with_doc(document_id: str) -> DocumentPlotStateStore:
    bus = DocumentCacheBus()
    store = DocumentPlotStateStore(bus=bus)
    store.ingest_chunk_summary(
        document_id,
        "Alice and Bob debate continuity risks while Dana observes.",
        version_id="v9",
        chunk_hash="chunk-plot",
        pointer_id="pointer-plot",
        metadata={"source_job_id": "job-plot"},
    )
    return store


def test_document_plot_state_tool_returns_payload_for_active_document() -> None:
    document_id = "doc-active"
    store = _build_store_with_doc(document_id)
    doc = DocumentState(text="Sample", document_id=document_id)
    tool = DocumentPlotStateTool(
        plot_state_resolver=lambda: store,
        active_document_provider=lambda: doc,
        feature_enabled=lambda: True,
    )

    payload = tool.run(include_arcs=False, max_entities=1)

    assert payload["status"] == "ok"
    assert payload["document_id"] == document_id
    assert payload["entities"] == payload["entities"][:1]
    assert payload["arcs"] == []


def test_document_plot_state_tool_accepts_explicit_document_id() -> None:
    store = _build_store_with_doc("doc-explicit")
    tool = DocumentPlotStateTool(
        plot_state_resolver=lambda: store,
        active_document_provider=lambda: None,
        feature_enabled=lambda: True,
    )

    payload = tool.run(document_id="doc-explicit", include_entities=False)

    assert payload["status"] == "ok"
    assert payload["entities"] == []
    assert payload["plot_state_available"] is True


def test_document_plot_state_tool_handles_disabled_flag() -> None:
    tool = DocumentPlotStateTool(
        plot_state_resolver=lambda: None,
        feature_enabled=lambda: False,
    )

    result = tool.run()

    assert result["status"] == "plot_state_disabled"


def test_document_plot_state_tool_reports_missing_store() -> None:
    tool = DocumentPlotStateTool(
        plot_state_resolver=lambda: None,
        active_document_provider=lambda: DocumentState(text="", document_id="doc-missing"),
        feature_enabled=lambda: True,
    )

    result = tool.run()

    assert result["status"] == "plot_state_unavailable"


def test_document_plot_state_tool_reports_missing_document_id() -> None:
    store = _build_store_with_doc("doc-has-store")
    tool = DocumentPlotStateTool(
        plot_state_resolver=lambda: store,
        active_document_provider=lambda: None,
        feature_enabled=lambda: True,
    )

    result = tool.run()

    assert result["status"] == "no_document"
