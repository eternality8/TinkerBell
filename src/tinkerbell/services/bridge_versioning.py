"""Version token formatting and validation helpers for the document bridge."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Mapping, Optional

from ..editor.document_model import DocumentState, DocumentVersion


_LOGGER = logging.getLogger(__name__)


def format_version_token(version: DocumentVersion, tab_id: str | None) -> str:
    """Format version token with tab_id prefix.
    
    Format: 'tab_id:document_id:version_id:content_hash'
    If no tab_id is set, uses 'default' as placeholder.
    """
    effective_tab_id = tab_id or "default"
    return f"{effective_tab_id}:{version.document_id}:{version.version_id}:{version.content_hash}"


def is_version_current(document: DocumentState, expected: str) -> bool:
    """Check if document version matches expected token.
    
    Handles both 3-part (document_id:version_id:hash) and 
    4-part (tab_id:document_id:version_id:hash) tokens.
    """
    signature = document.version_signature()
    # If expected has 4 parts (with tab_id prefix), strip the prefix
    parts = expected.split(":")
    if len(parts) >= 4:
        # 4-part format: tab_id:document_id:version_id:hash
        expected = ":".join(parts[1:])  # Remove tab_id prefix
    return signature == expected


def hash_text(text: str) -> str:
    """Compute SHA-1 hash of text."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def extract_context_version(payload: Mapping[str, Any]) -> Optional[str]:
    """Extract the document version token from a payload."""
    for key in ("document_version", "snapshot_version", "version", "document_digest"):
        token = payload.get(key)
        if token is None:
            continue
        token_str = str(token).strip()
        if token_str:
            return token_str
    return None


def extract_content_hash(payload: Mapping[str, Any]) -> Optional[str]:
    """Extract the content hash from a payload."""
    token = payload.get("content_hash")
    if token is None:
        return None
    token_str = str(token).strip()
    return token_str or None


def parse_chunk_bounds(chunk_id: str | None) -> tuple[int, int] | None:
    """Parse chunk bounds from a chunk ID string.
    
    Expected format: 'chunk:{document_id}:{start}:{end}'
    Returns (start, end) tuple or None if parsing fails.
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
    return (start, end)


def compute_line_start_offsets(text: str) -> list[int]:
    """Compute line start offsets for a text string.
    
    Returns a list of offsets where each line starts (0-indexed).
    The first offset is always 0.
    """
    offsets = [0]
    if not text:
        return offsets
    cursor = 0
    for segment in text.splitlines(keepends=True):
        cursor += len(segment)
        offsets.append(cursor)
    return offsets


def clamp_range(start: int, end: int, length: int) -> tuple[int, int]:
    """Clamp a range to valid bounds within a document.
    
    Ensures start <= end and both are within [0, length].
    """
    start = max(0, min(start, length))
    end = max(0, min(end, length))
    if end < start:
        start, end = end, start
    return start, end


def summarize_diff(before: str, after: str) -> str:
    """Generate a lightweight summary of a diff.
    
    Returns a string like '+100 chars' or '-50 chars' or 'Δ0'.
    """
    delta = len(after) - len(before)
    if delta == 0:
        return "Δ0"
    sign = "+" if delta > 0 else "-"
    return f"{sign}{abs(delta)} chars"
