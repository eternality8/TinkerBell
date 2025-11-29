"""Guardrail hint generation for tool results.

This module provides functions for generating contextual guardrail hints
based on tool execution results. These hints help guide the model's
subsequent actions based on tool status and payload content.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "format_guardrail_hint",
    "outline_guardrail_hints",
    "retrieval_guardrail_hints",
]


def format_guardrail_hint(source: str, lines: Sequence[str]) -> str:
    """Format a guardrail hint with source attribution.

    Args:
        source: The source of the hint (e.g., "get_outline", "search_document").
        lines: The hint lines to format.

    Returns:
        Formatted hint string, or empty string if no valid lines.
    """
    filtered = [str(line).strip() for line in lines if str(line).strip()]
    if not filtered:
        return ""
    body = "\n".join(f"- {line}" for line in filtered)
    return f"Guardrail hint ({source}):\n{body}"


def outline_guardrail_hints(payload: Mapping[str, Any]) -> list[str]:
    """Generate guardrail hints for document outline tool results.

    Args:
        payload: The deserialized tool result payload.

    Returns:
        List of formatted guardrail hint strings.
    """
    hints: list[str] = []
    guardrails = payload.get("guardrails")
    if isinstance(guardrails, Sequence):
        for entry in guardrails:
            if not isinstance(entry, Mapping):
                continue
            guardrail_type = str(entry.get("type") or "guardrail")
            message = str(entry.get("message") or "").strip()
            action = str(entry.get("action") or "").strip()
            lines: list[str] = []
            if message:
                lines.append(message)
            if action:
                lines.append(f"Action: {action}")
            hint = format_guardrail_hint(f"get_outline • {guardrail_type}", lines)
            if hint:
                hints.append(hint)
    status = str(payload.get("status") or "").lower()
    document_id = str(payload.get("document_id") or "this document")
    reason = str(payload.get("reason") or "").strip()
    retry_after_ms = payload.get("retry_after_ms")
    is_stale = bool(payload.get("is_stale")) or status == "stale"
    trimmed_reason = str(payload.get("trimmed_reason") or "").lower()
    outline_available = payload.get("outline_available")
    if status == "pending":
        retry_hint = None
        if isinstance(retry_after_ms, (int, float)) and retry_after_ms > 0:
            retry_seconds = retry_after_ms / 1000.0
            retry_hint = f"Retry after ~{retry_seconds:.1f}s or continue with read_document while the worker rebuilds."
        lines = [f"Outline for {document_id} is still building; treat existing nodes as stale hints only."]
        if retry_hint:
            lines.append(retry_hint)
        hints.append(format_guardrail_hint("get_outline", lines))
    elif status == "unsupported_format":
        detail = reason or "unsupported format"
        lines = [
            f"Outline unavailable for {document_id}: {detail}.",
            "Navigate manually with read_document or other tools.",
        ]
        hints.append(format_guardrail_hint("get_outline", lines))
    elif status in {"outline_missing", "outline_unavailable", "no_document"}:
        detail = reason or "outline not cached yet"
        lines = [
            f"Outline missing for {document_id} ({detail}).",
            "Queue the worker or rely on selection-scoped snapshots until it exists.",
        ]
        hints.append(format_guardrail_hint("get_outline", lines))
    if is_stale:
        lines = [
            f"Outline for {document_id} is stale compared to the latest read_document.",
            "Refresh the outline or treat headings as hints only before editing.",
        ]
        hints.append(format_guardrail_hint("get_outline", lines))
    if trimmed_reason == "token_budget":
        lines = [
            "Outline was trimmed by the token budget.",
            "Request fewer levels or hydrate specific pointers before editing deeper sections.",
        ]
        hints.append(format_guardrail_hint("get_outline", lines))
    if outline_available is False and status not in {"pending", "unsupported_format", "outline_missing", "outline_unavailable"}:
        lines = [
            f"Outline payload for {document_id} indicated no nodes were returned.",
            "Avoid planning edits that rely on missing structure until the worker succeeds.",
        ]
        hints.append(format_guardrail_hint("get_outline", lines))
    return [hint for hint in hints if hint]


def retrieval_guardrail_hints(payload: Mapping[str, Any]) -> list[str]:
    """Generate guardrail hints for document retrieval/search tool results.

    Args:
        payload: The deserialized tool result payload.

    Returns:
        List of formatted guardrail hint strings.
    """
    hints: list[str] = []
    status = str(payload.get("status") or "").lower()
    document_id = str(payload.get("document_id") or "this document")
    reason = str(payload.get("reason") or "").strip()
    fallback_reason = str(payload.get("fallback_reason") or "").strip()
    offline_mode = bool(payload.get("offline_mode"))
    if status == "unsupported_format":
        detail = reason or "unsupported format"
        lines = [
            f"Retrieval disabled for {document_id}: {detail}.",
            "Use read_document, manual navigation, or outline pointers instead.",
        ]
        hints.append(format_guardrail_hint("search_document", lines))
    if offline_mode or status in {"offline_fallback", "offline_no_results"}:
        label_reason = fallback_reason or ("offline mode" if offline_mode else "fallback strategy")
        lines = [
            f"Retrieval is running without embeddings ({label_reason}).",
            "Treat matches as low-confidence hints and rehydrate via read_document before editing.",
        ]
        hints.append(format_guardrail_hint("search_document", lines))
    if status == "offline_no_results":
        lines = [
            f"Offline fallback could not find matches for {document_id}.",
            "Try a different query or scan the outline/snapshot manually.",
        ]
        hints.append(format_guardrail_hint("search_document", lines))
    return [hint for hint in hints if hint]
