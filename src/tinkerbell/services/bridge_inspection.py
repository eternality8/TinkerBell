"""Post-edit inspection helpers for the document bridge."""

from __future__ import annotations

from typing import Any, Mapping


def auto_revert_remediation(reason: str | None) -> str:
    """Return remediation guidance based on rejection reason."""
    mapping = {
        "duplicate_paragraphs": (
            "Retry with replace_all=true or delete the original text before inserting "
            "the rewrite to avoid duplicated passages."
        ),
        "duplicate_windows": (
            "Retry with replace_all=true or narrow the edit range so the rewrite "
            "replaces the previous text instead of appending it."
        ),
        "boundary_dropped": (
            "Ensure the edit preserves blank lines between paragraphs or include "
            "surrounding context in the diff."
        ),
        "split_tokens": (
            "Insert whitespace around the edited span so tokens are not merged "
            "across boundaries before retrying."
        ),
        "split_token_regex": (
            "Add a newline or space before markdown tokens (e.g., '#') so headings "
            "are not fused with preceding words."
        ),
    }
    return mapping.get(
        reason or "",
        (
            "Refresh document_snapshot and rebuild the diff before retrying to "
            "ensure the edit targets the latest content."
        ),
    )


def format_auto_revert_message(details: Mapping[str, Any] | None) -> str:
    """Format a human-readable auto-revert message from rejection details."""
    if not isinstance(details, Mapping):
        return (
            "Auto-revert triggered. Document was restored to the previous snapshot; refresh "
            "document_snapshot and retry with an updated diff."
        )
    reason = str(details.get("reason") or "inspection_failure")
    detail = str(details.get("detail") or "Edit rejected")
    remediation = details.get("remediation")
    message = f"Auto-revert triggered ({reason}). {detail}"
    if remediation:
        message = f"{message} {remediation}".strip()
    return message


def build_failure_metadata(
    *,
    version: Any | None,  # DocumentVersion
    status: str | None = None,
    reason: str | None = None,
    cause: str | None = None,
    range_count: int | None = None,
    streamed: bool | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    scope_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build structured failure metadata for telemetry and error reporting."""
    metadata: dict[str, Any] = {}
    if version is not None:
        metadata["document_id"] = version.document_id
        metadata["version_id"] = version.version_id
        metadata["content_hash"] = version.content_hash
    if status:
        metadata["status"] = status
    if reason:
        metadata["reason"] = reason
    if cause:
        metadata["cause"] = cause
    if range_count is not None:
        metadata["range_count"] = range_count
    if streamed is not None:
        metadata["streamed"] = bool(streamed)
    if diagnostics:
        metadata["diagnostics"] = dict(diagnostics)
    attach_scope_metadata(metadata, scope_summary)
    return metadata


def attach_scope_metadata(target: dict[str, Any], scope_summary: Mapping[str, Any] | None) -> None:
    """Attach scope metadata fields to a telemetry payload."""
    if not isinstance(scope_summary, Mapping):
        return
    origin = scope_summary.get("origin")
    if isinstance(origin, str) and origin.strip():
        target["scope_origin"] = origin.strip()
    length = scope_summary.get("length")
    try:
        if length is not None:
            target["scope_length"] = max(0, int(length))
    except (TypeError, ValueError):
        pass
    range_payload = scope_summary.get("range")
    if isinstance(range_payload, Mapping):
        start = range_payload.get("start")
        end = range_payload.get("end")
        try:
            if start is not None and end is not None:
                target["scope_range"] = {"start": int(start), "end": int(end)}
        except (TypeError, ValueError):
            pass
