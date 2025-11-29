"""Static utility functions for AI controller operations.

This module provides pure functions for normalizing, coercing, and sanitizing
values used throughout the AI controller. These are stateless helpers that
don't depend on controller instance state.
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = [
    "normalize_iterations",
    "normalize_scope_origin",
    "normalize_context_tokens",
    "normalize_response_reserve",
    "normalize_temperature",
    "coerce_optional_int",
    "coerce_optional_float",
    "coerce_optional_str",
    "sanitize_suggestions",
]

# Canonical scope origin values for telemetry normalization
_SCOPE_ORIGIN_MAPPING: dict[str, str] = {
    "doc": "document",
    "full_document": "document",
    "whole_document": "document",
    "entire_document": "document",
    "sel": "selection",
    "selected": "selection",
    "user_selection": "selection",
    "explicit": "explicit_span",
    "explicit_range": "explicit_span",
    "manual_span": "explicit_span",
    "chunk": "chunk",
    "chunk_manifest": "chunk",
    "manifest_chunk": "chunk",
}


def normalize_iterations(value: int | None) -> int:
    """Normalize iteration count to valid range [1, 200]."""
    try:
        candidate = int(value) if value is not None else 8
    except (TypeError, ValueError):
        candidate = 8
    return max(1, min(candidate, 200))


def normalize_scope_origin(value: Any) -> str | None:
    """Normalize scope origin strings to canonical values.

    Args:
        value: Raw scope origin value.

    Returns:
        Canonical scope origin string or None if invalid.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    if normalized in _SCOPE_ORIGIN_MAPPING:
        return _SCOPE_ORIGIN_MAPPING[normalized]
    # Allow through known canonical values
    if normalized in {"document", "selection", "explicit_span", "chunk", "mixed"}:
        return normalized
    return normalized


def normalize_context_tokens(value: int | None) -> int:
    """Normalize max context tokens to valid range [8_000, 4_000_000]."""
    try:
        candidate = int(value) if value is not None else 128_000
    except (TypeError, ValueError):
        candidate = 128_000
    return max(8_000, min(candidate, 4_000_000))


def normalize_response_reserve(value: int | None) -> int | None:
    """Normalize response reserve tokens to valid range or None."""
    if value is None:
        return None
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return None
    if candidate <= 0:
        return None
    return max(256, min(candidate, 64_000))


def normalize_temperature(value: float | None) -> float:
    """Normalize temperature to valid range [0.0, 2.0]."""
    if value is None:
        return 0.7
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return 0.7
    return max(0.0, min(candidate, 2.0))


def coerce_optional_int(value: Any) -> int | None:
    """Coerce a value to int, returning None if conversion fails."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_optional_float(value: Any) -> float | None:
    """Coerce a value to float, returning None if conversion fails."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_optional_str(value: Any) -> str | None:
    """Coerce a value to non-empty string, returning None if empty."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def sanitize_suggestions(items: Sequence[Any], max_count: int) -> list[str]:
    """Sanitize and limit follow-up suggestions.

    Args:
        items: Raw suggestion items.
        max_count: Maximum number of suggestions to return.

    Returns:
        List of cleaned, non-empty suggestion strings.
    """
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned:
            continue
        # Remove common list prefixes
        if cleaned[0].isdigit():
            # Strip leading "1." or "1)" style prefixes
            parts = cleaned.split(".", 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                cleaned = parts[1].strip()
            else:
                parts = cleaned.split(")", 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    cleaned = parts[1].strip()
        if not cleaned:
            continue
        lower = cleaned.lower()
        if lower in seen:
            continue
        seen.add(lower)
        result.append(cleaned)
        if len(result) >= max_count:
            break
    return result
