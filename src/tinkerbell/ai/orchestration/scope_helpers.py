"""Scope and range helper functions for tool argument processing.

This module provides utilities for extracting, normalizing, and building
scope summaries from tool arguments. These functions help understand the
target range and origin of tool operations.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .controller_utils import coerce_optional_int, normalize_scope_origin

__all__ = [
    "scope_summary_from_arguments",
    "scope_summary_from_metadata",
    "scope_summary_from_ranges",
    "scope_summary_from_target_range",
    "scope_summary_from_chunk_arguments",
    "scope_fields_from_summary",
    "range_bounds_from_entry",
    "range_bounds_from_mapping",
    "extract_chunk_id",
    "parse_chunk_bounds",
]


def scope_summary_from_arguments(arguments: Any) -> dict[str, Any] | None:
    """Extract scope summary from tool arguments.

    Tries multiple strategies in order:
    1. metadata.scope
    2. ranges array
    3. target_range
    4. chunk_id based bounds

    Args:
        arguments: Tool arguments mapping.

    Returns:
        Scope summary dict or None if no scope info found.
    """
    if not isinstance(arguments, Mapping):
        return None
    metadata_summary = scope_summary_from_metadata(arguments.get("metadata"))
    if metadata_summary:
        return metadata_summary
    ranges_summary = scope_summary_from_ranges(arguments.get("ranges"))
    if ranges_summary:
        return ranges_summary
    target_summary = scope_summary_from_target_range(arguments.get("target_range"))
    if target_summary:
        return target_summary
    chunk_summary = scope_summary_from_chunk_arguments(arguments)
    if chunk_summary:
        return chunk_summary
    return None


def scope_summary_from_metadata(metadata: Any) -> dict[str, Any] | None:
    """Extract scope summary from metadata object."""
    if not isinstance(metadata, Mapping):
        return None
    scope_payload = metadata.get("scope") if isinstance(metadata.get("scope"), Mapping) else None
    origin = metadata.get("scope_origin")
    if (not isinstance(origin, str) or not origin.strip()) and isinstance(scope_payload, Mapping):
        origin = scope_payload.get("origin")
    normalized_origin = normalize_scope_origin(origin)
    summary: dict[str, Any] = {}
    if normalized_origin:
        summary["origin"] = normalized_origin
    length_value = metadata.get("scope_length")
    length = coerce_optional_int(length_value)
    if length is None and isinstance(scope_payload, Mapping):
        length = coerce_optional_int(scope_payload.get("length"))
    if length is not None:
        summary["length"] = max(0, length)
    range_payload = metadata.get("scope_range")
    if not isinstance(range_payload, Mapping) and isinstance(scope_payload, Mapping):
        candidate = scope_payload.get("range")
        if isinstance(candidate, Mapping):
            range_payload = candidate
    bounds = range_bounds_from_mapping(range_payload)
    if bounds is not None:
        start, end = bounds
        summary["range"] = {"start": start, "end": end}
    return summary or None


def scope_summary_from_ranges(ranges: Any) -> dict[str, Any] | None:
    """Extract scope summary from ranges array."""
    if not isinstance(ranges, Sequence) or isinstance(ranges, (str, bytes)):
        return None
    origins: set[str] = set()
    total_length = 0
    range_starts: list[int] = []
    range_ends: list[int] = []
    found = False
    for entry in ranges:
        if not isinstance(entry, Mapping):
            continue
        found = True
        origin = normalize_scope_origin(entry.get("scope_origin"))
        scope_payload = entry.get("scope") if isinstance(entry.get("scope"), Mapping) else None
        if origin is None and isinstance(scope_payload, Mapping):
            origin = normalize_scope_origin(scope_payload.get("origin"))
        if origin:
            origins.add(origin)
        length = coerce_optional_int(entry.get("scope_length"))
        if length is None and isinstance(scope_payload, Mapping):
            length = coerce_optional_int(scope_payload.get("length"))
        bounds = range_bounds_from_entry(entry)
        if bounds is not None:
            start, end = bounds
            range_starts.append(start)
            range_ends.append(end)
            if length is None:
                length = max(0, end - start)
        if length is not None:
            total_length += max(0, length)
    if not found:
        return None
    origin_summary = "mixed"
    if len(origins) == 1:
        origin_summary = origins.pop()
    elif not origins:
        origin_summary = "explicit_span"
    summary: dict[str, Any] = {"origin": origin_summary}
    if total_length > 0:
        summary["length"] = total_length
    if range_starts and range_ends:
        summary["range"] = {"start": min(range_starts), "end": max(range_ends)}
    return summary


def scope_summary_from_target_range(target_range: Any) -> dict[str, Any] | None:
    """Extract scope summary from target_range argument."""
    if target_range is None:
        return None
    if isinstance(target_range, str):
        token = target_range.strip().lower()
        if token in {"document"}:
            return {"origin": "document"}
        return None
    start: int | None = None
    end: int | None = None
    origin: str | None = None
    if isinstance(target_range, Mapping):
        scope_token = str(target_range.get("scope") or target_range.get("origin") or "").strip().lower()
        if scope_token in {"document"}:
            origin = "document"
        start = coerce_optional_int(target_range.get("start"))
        end = coerce_optional_int(target_range.get("end"))
    elif isinstance(target_range, Sequence) and len(target_range) == 2 and not isinstance(target_range, (str, bytes)):
        start = coerce_optional_int(target_range[0])
        end = coerce_optional_int(target_range[1])
    if start is None or end is None:
        if origin == "document":
            return {"origin": "document"}
        return None
    if end < start:
        start, end = end, start
    summary = {
        "origin": origin or "explicit_span",
        "range": {"start": start, "end": end},
        "length": max(0, end - start),
    }
    return summary


def scope_summary_from_chunk_arguments(arguments: Mapping[str, Any]) -> dict[str, Any] | None:
    """Extract scope summary from chunk_id in arguments."""
    chunk_id = extract_chunk_id(arguments)
    if not chunk_id:
        return None
    bounds = parse_chunk_bounds(chunk_id)
    if not bounds:
        return None
    start, end = bounds
    return {
        "origin": "chunk",
        "range": {"start": start, "end": end},
        "length": max(0, end - start),
        "chunk_id": chunk_id,
    }


def scope_fields_from_summary(summary: Mapping[str, Any] | None) -> dict[str, Any]:
    """Convert scope summary to flat field dict for records."""
    if not isinstance(summary, Mapping):
        return {}
    payload: dict[str, Any] = {}
    origin = summary.get("origin")
    if isinstance(origin, str) and origin.strip():
        payload["scope_origin"] = origin.strip()
    length = summary.get("length")
    try:
        if length is not None:
            payload["scope_length"] = max(0, int(length))
    except (TypeError, ValueError):
        pass
    range_payload = summary.get("range")
    if isinstance(range_payload, Mapping):
        try:
            start = int(range_payload.get("start"))
            end = int(range_payload.get("end"))
        except (TypeError, ValueError):
            start = end = None
        if start is not None and end is not None:
            payload["scope_range"] = {"start": start, "end": end}
    return payload


def range_bounds_from_entry(entry: Mapping[str, Any]) -> tuple[int, int] | None:
    """Extract range bounds from a ranges array entry."""
    range_payload = entry.get("scope_range") if isinstance(entry.get("scope_range"), Mapping) else None
    if range_payload is None:
        scope_payload = entry.get("scope") if isinstance(entry.get("scope"), Mapping) else None
        if isinstance(scope_payload, Mapping):
            candidate = scope_payload.get("range")
            if isinstance(candidate, Mapping):
                range_payload = candidate
    bounds = range_bounds_from_mapping(range_payload)
    if bounds is not None:
        return bounds
    start = coerce_optional_int(entry.get("start"))
    end = coerce_optional_int(entry.get("end"))
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return start, end


def range_bounds_from_mapping(payload: Mapping[str, Any] | None) -> tuple[int, int] | None:
    """Extract (start, end) tuple from a range mapping."""
    if not isinstance(payload, Mapping):
        return None
    start = coerce_optional_int(payload.get("start"))
    end = coerce_optional_int(payload.get("end"))
    if start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return start, end


def extract_chunk_id(arguments: Mapping[str, Any]) -> str | None:
    """Extract chunk_id from various locations in arguments."""
    def _coerce(value: Any) -> str | None:
        if isinstance(value, str):
            text = value.strip()
            return text or None
        return None

    chunk_id = _coerce(arguments.get("chunk_id"))
    if chunk_id:
        return chunk_id
    metadata = arguments.get("metadata")
    if isinstance(metadata, Mapping):
        chunk_id = _coerce(metadata.get("chunk_id"))
        if chunk_id:
            return chunk_id
    for key in ("patches", "ranges"):
        entries = arguments.get(key)
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            chunk_id = _coerce(entry.get("chunk_id"))
            if chunk_id:
                return chunk_id
    return None


def parse_chunk_bounds(chunk_id: str | None) -> tuple[int, int] | None:
    """Parse character range bounds from a chunk_id string.

    Chunk IDs have format like: "doc:hash:start:end"
    """
    if not chunk_id:
        return None
    parts = chunk_id.split(":")
    if len(parts) < 4:
        return None
    try:
        start = int(parts[-2])
        end = int(parts[-1])
    except ValueError:
        return None
    if end < start:
        start, end = end, start
    return start, end
