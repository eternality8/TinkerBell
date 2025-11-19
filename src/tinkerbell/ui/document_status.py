"""Helper models + formatting for the document status console."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(slots=True)
class DocumentDescriptor:
    """Lightweight metadata describing an open document tab."""

    document_id: str
    label: str
    tab_id: str


def _coerce_label(payload: Mapping[str, Any] | None) -> str:
    if not payload:
        return "document"
    for key in ("label", "title", "path", "document_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "document"


def _format_planner_line(planner: Mapping[str, Any] | None) -> str:
    if not isinstance(planner, Mapping):
        return "Planner: unavailable"
    pending = int(planner.get("pending", 0) or 0)
    completed = int(planner.get("completed", 0) or 0)
    total = completed + pending
    if total == 0:
        return "Planner: no tracked tasks"
    if pending:
        return f"Planner: {pending} pending / {completed} completed"
    return f"Planner: {completed} completed"


def _format_chunk_line(chunks: Mapping[str, Any] | None) -> str:
    if not isinstance(chunks, Mapping):
        return "Chunks: manifest unavailable"
    manifest = chunks.get("chunk_manifest")
    chunk_count = 0
    profile = chunks.get("chunk_profile") or "auto"
    window = chunks.get("window") or {}
    if isinstance(manifest, Mapping):
        chunks_payload = manifest.get("chunks")
        if isinstance(chunks_payload, Sequence):
            chunk_count = len(chunks_payload)
    selection = window.get("selection") if isinstance(window, Mapping) else None
    if isinstance(selection, Mapping):
        selection_range = (selection.get("start"), selection.get("end"))
    else:
        selection_range = None
    if selection_range and all(isinstance(value, int) for value in selection_range):
        return (
            f"Chunks: {chunk_count} ({profile}) covering selection "
            f"{selection_range[0]}–{selection_range[1]}"
        )
    return f"Chunks: {chunk_count} ({profile})"


def _format_outline_line(outline: Mapping[str, Any] | None) -> str:
    if not isinstance(outline, Mapping):
        return "Outline: unavailable"
    status = (outline.get("status") or "unknown").strip()
    if status != "ok":
        return f"Outline: {status}"
    node_count = int(outline.get("node_count", 0) or 0)
    outline_hash = outline.get("outline_hash") or "n/a"
    return f"Outline: {node_count} nodes (hash {outline_hash})"


def _format_chunk_flow_line(telemetry: Mapping[str, Any] | None) -> str:
    chunk_flow = telemetry.get("chunk_flow") if isinstance(telemetry, Mapping) else None
    if not isinstance(chunk_flow, Mapping):
        return "Chunk flow: n/a"
    status = (chunk_flow.get("status") or "Idle").strip()
    detail = chunk_flow.get("detail")
    return f"Chunk flow: {status}{f' — {detail}' if detail else ''}"


def format_document_status_summary(payload: Mapping[str, Any]) -> str:
    """Return a condensed multi-line summary for chat/manual commands."""

    document = payload.get("document") if isinstance(payload, Mapping) else None
    label = _coerce_label(document)
    lines = [f"Document status for {label}:"]

    chunks = payload.get("chunks") if isinstance(payload, Mapping) else None
    lines.append(f"- {_format_chunk_line(chunks)}")

    outline = payload.get("outline") if isinstance(payload, Mapping) else None
    lines.append(f"- {_format_outline_line(outline)}")

    planner = payload.get("planner") if isinstance(payload, Mapping) else None
    lines.append(f"- {_format_planner_line(planner)}")

    telemetry = payload.get("telemetry") if isinstance(payload, Mapping) else None
    if isinstance(telemetry, Mapping):
        lines.append(f"- {_format_chunk_flow_line(telemetry)}")
        analysis_state = telemetry.get("analysis")
        if isinstance(analysis_state, Mapping) and analysis_state.get("status"):
            detail = analysis_state.get("detail")
            suffix = f" — {detail}" if isinstance(detail, str) and detail.strip() else ""
            lines.append(f"- Analysis: {analysis_state['status']}{suffix}")

    return "\n".join(lines)


__all__ = ["DocumentDescriptor", "format_document_status_summary"]
