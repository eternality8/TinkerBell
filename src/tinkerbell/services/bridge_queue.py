"""Edit queue management and directive normalization for the document bridge."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ..chat.commands import ActionType, parse_agent_payload, validate_directive
from ..chat.message_model import EditDirective
from ..documents.ranges import TextRange
from .bridge_types import PatchRangePayload, QueuedEdit


def normalize_directive(
    directive: EditDirective | Mapping[str, Any],
    document_text: str,
    *,
    extract_context_version_fn: Any,
    extract_content_hash_fn: Any,
) -> QueuedEdit:
    """Validate, normalize, and prepare a directive for queuing.
    
    Args:
        directive: The directive to normalize (EditDirective or dict)
        document_text: Current document text for validation
        extract_context_version_fn: Function to extract version from payload
        extract_content_hash_fn: Function to extract content hash from payload
        
    Returns:
        A QueuedEdit ready for execution
        
    Raises:
        TypeError: If directive is not an EditDirective or mapping
        ValueError: If directive fails validation
    """
    if isinstance(directive, EditDirective):
        payload: dict[str, Any] = {
            "action": directive.action,
            "content": directive.content,
            "target_range": directive.target_range.to_dict(),
        }
        if directive.rationale is not None:
            payload["rationale"] = directive.rationale
    elif isinstance(directive, Mapping):
        payload = dict(directive)
    else:  # pragma: no cover - guard against unsupported payloads
        raise TypeError("Directive must be an EditDirective or mapping.")

    payload = parse_agent_payload(payload)
    validation = validate_directive(payload)
    if not validation.ok:
        raise ValueError(validation.message)

    action = str(payload.get("action", "")).lower()
    rationale = payload.get("rationale")
    context_version = extract_context_version_fn(payload)

    match_text_value = payload.get("match_text")
    expected_text_value = payload.get("expected_text")

    if action == ActionType.PATCH.value:
        diff_text = str(payload.get("diff", ""))
        ranges = normalize_patch_ranges(payload.get("ranges"))
        if not diff_text.strip() and not ranges:
            raise ValueError("Patch directives must include a diff string or ranges payload")
        if not context_version:
            raise ValueError("Patch directives must include the originating document version")
        content_hash = extract_content_hash_fn(payload)
        if not content_hash:
            raise ValueError("Patch directives must include the originating content_hash")
        scope_summary = summarize_patch_scopes(ranges)
        range_hint = range_hint_from_payload(ranges, payload)
        edit_directive = EditDirective(
            action=action,
            target_range=range_hint,
            content="",
            rationale=str(rationale) if rationale is not None else None,
            diff=diff_text if diff_text.strip() else None,
            match_text=str(match_text_value) if isinstance(match_text_value, str) else None,
            expected_text=str(expected_text_value) if isinstance(expected_text_value, str) else None,
        )
        return QueuedEdit(
            directive=edit_directive,
            context_version=context_version,
            content_hash=content_hash,
            payload=payload,
            diff=diff_text if diff_text.strip() else None,
            ranges=ranges,
            scope_summary=scope_summary,
        )
    raise ValueError(
        "DocumentBridge only accepts patch directives; convert inline edits into patches before queueing."
    )


def normalize_patch_ranges(ranges: Any) -> tuple[PatchRangePayload, ...]:
    """Normalize and validate patch range payloads.
    
    Args:
        ranges: Raw ranges from the directive payload
        
    Returns:
        Tuple of normalized PatchRangePayload objects
        
    Raises:
        ValueError: If ranges are malformed
    """
    if ranges in (None, (), []):
        return ()
    if not isinstance(ranges, Sequence):
        raise ValueError("Patch ranges must be provided as an array")
    normalized: list[PatchRangePayload] = []
    for entry in ranges:
        if not isinstance(entry, Mapping):
            raise ValueError("Patch ranges must be objects")
        if "start" not in entry or "end" not in entry:
            raise ValueError("Patch ranges require 'start' and 'end' keys")
        replacement = entry.get("replacement") or entry.get("content") or entry.get("text")
        if replacement is None:
            raise ValueError("Patch ranges must include replacement text")
        match_text = entry.get("match_text")
        if match_text is None:
            raise ValueError("Patch ranges must include match_text")
        start = int(entry.get("start", 0))
        end = int(entry.get("end", 0))
        if end < start:
            start, end = end, start
        scope_payload = entry.get("scope")
        scope_mapping = dict(scope_payload) if isinstance(scope_payload, Mapping) else None
        scope_origin = entry.get("scope_origin")
        if not isinstance(scope_origin, str) and scope_mapping is not None:
            scope_origin = scope_mapping.get("origin")
        normalized_scope_origin = normalize_scope_origin(scope_origin)
        scope_length = entry.get("scope_length")
        if scope_length is None and scope_mapping is not None:
            scope_length = scope_mapping.get("length")
        scope_range_payload = entry.get("scope_range")
        if scope_range_payload is None and scope_mapping is not None:
            scope_range_payload = scope_mapping.get("range")
        normalized_scope_range = coerce_scope_range(scope_range_payload)
        normalized_scope_length = coerce_scope_length(scope_length)
        if normalized_scope_length is None and normalized_scope_range is not None:
            normalized_scope_length = max(0, normalized_scope_range[1] - normalized_scope_range[0])
        chunk_id = str(entry.get("chunk_id")) if isinstance(entry.get("chunk_id"), str) else None
        chunk_hash = str(entry.get("chunk_hash")) if isinstance(entry.get("chunk_hash"), str) else None
        scope_mapping = normalize_scope_mapping(
            scope_mapping,
            origin=normalized_scope_origin,
            scope_range=normalized_scope_range,
            scope_length=normalized_scope_length,
        )
        validate_scope_requirements(
            origin=normalized_scope_origin,
            scope_range=normalized_scope_range,
            scope_length=normalized_scope_length,
            chunk_id=chunk_id,
            chunk_hash=chunk_hash,
        )
        normalized.append(
            PatchRangePayload(
                start=start,
                end=end,
                replacement=str(replacement),
                match_text=str(match_text),
                chunk_id=chunk_id,
                chunk_hash=chunk_hash,
                scope_origin=normalized_scope_origin,
                scope_length=normalized_scope_length,
                scope_range=normalized_scope_range,
                scope=scope_mapping,
            )
        )
    return tuple(normalized)


def normalize_scope_origin(origin: Any) -> str | None:
    """Normalize a scope origin value to lowercase or None."""
    if not isinstance(origin, str):
        return None
    token = origin.strip().lower()
    return token or None


def coerce_scope_length(value: Any) -> int | None:
    """Coerce a scope length value to a non-negative integer or None."""
    if value is None:
        return None
    try:
        length = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, length)


def coerce_scope_range(value: Any) -> tuple[int, int] | None:
    """Coerce a scope range to a (start, end) tuple or None."""
    if value is None:
        return None
    try:
        text_range = TextRange.from_value(value)
    except (TypeError, ValueError):
        return None
    start, end = text_range.to_tuple()
    if end < start:
        start, end = end, start
    return (start, end)


def normalize_scope_mapping(
    scope: Mapping[str, Any] | None,
    *,
    origin: str | None,
    scope_range: tuple[int, int] | None,
    scope_length: int | None,
) -> Mapping[str, Any] | None:
    """Build a normalized scope mapping from components."""
    has_data = any(value is not None for value in (origin, scope_range, scope_length))
    if not isinstance(scope, Mapping) and not has_data:
        return None
    mapping = dict(scope) if isinstance(scope, Mapping) else {}
    if origin:
        mapping.setdefault("origin", origin)
    if scope_range is not None:
        mapping.setdefault("range", {"start": scope_range[0], "end": scope_range[1]})
    if scope_length is not None:
        mapping.setdefault("length", scope_length)
    return mapping or None


def validate_scope_requirements(
    *,
    origin: str | None,
    scope_range: tuple[int, int] | None,
    scope_length: int | None,
    chunk_id: str | None,
    chunk_hash: str | None,
) -> None:
    """Validate that scope metadata meets requirements.
    
    Raises:
        ValueError: If required scope metadata is missing
    """
    if origin is None:
        raise ValueError("Patch ranges must include scope metadata (scope.origin)")
    if origin != "document" and scope_range is None:
        raise ValueError("Patch ranges must include scope_range metadata for non-document scopes")
    if origin == "chunk" and not chunk_id and not chunk_hash:
        raise ValueError("Chunk-scoped ranges must include chunk_id or chunk_hash metadata")
    if scope_range is not None and scope_length is not None:
        expected = max(0, scope_range[1] - scope_range[0])
        if expected != scope_length:
            raise ValueError("scope_length must match the provided scope_range span")


def summarize_patch_scopes(ranges: Sequence[PatchRangePayload]) -> Mapping[str, Any] | None:
    """Build a summary of scope metadata across patch ranges."""
    if not ranges:
        return None
    origins: set[str] = set()
    lengths: list[int] = []
    for entry in ranges:
        origin = entry.scope_origin
        if not origin and isinstance(entry.scope, Mapping):
            origin = normalize_scope_origin(entry.scope.get("origin"))
        if origin:
            origins.add(origin)
        length = entry.scope_length
        if length is None and entry.scope_range is not None:
            length = max(0, entry.scope_range[1] - entry.scope_range[0])
        if length is None:
            length = max(0, entry.end - entry.start)
        lengths.append(length)
    if not origins:
        origins.add("unknown")
    origin_summary = origins.pop() if len(origins) == 1 else "mixed"
    summary: dict[str, Any] = {"origin": origin_summary}
    if lengths:
        summary["length"] = sum(lengths)
    range_starts = [entry.scope_range[0] if entry.scope_range else entry.start for entry in ranges]
    range_ends = [entry.scope_range[1] if entry.scope_range else entry.end for entry in ranges]
    if range_starts and range_ends:
        summary["range"] = {
            "start": min(range_starts),
            "end": max(range_ends),
        }
    return summary


def range_hint_from_payload(
    ranges: Sequence[PatchRangePayload],
    payload: Mapping[str, Any],
) -> tuple[int, int]:
    """Extract a range hint from ranges or payload metadata."""
    if ranges:
        start = min(entry.start for entry in ranges)
        end = max(entry.end for entry in ranges)
        return (start, end)
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        hint = coerce_text_range_hint(metadata.get("target_range"))
        if hint is not None:
            return hint
    hint = coerce_text_range_hint(payload.get("target_range"))
    if hint is not None:
        return hint
    return (0, 0)


def coerce_text_range_hint(value: Any) -> tuple[int, int] | None:
    """Coerce a value to a (start, end) tuple or None."""
    if value is None:
        return None
    try:
        text_range = TextRange.from_value(value)
    except (TypeError, ValueError):
        return None
    return text_range.to_tuple()


def refresh_scope_span(
    scope: Mapping[str, Any] | None,
    start: int,
    end: int,
) -> Mapping[str, Any] | None:
    """Update a scope mapping with new range bounds."""
    if not isinstance(scope, Mapping):
        return None
    updated = dict(scope)
    updated["range"] = {"start": start, "end": end}
    updated["length"] = max(0, end - start)
    return updated
