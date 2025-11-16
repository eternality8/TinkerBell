"""Deterministic summarizer + pointer helpers for oversized tool payloads."""

from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from ...chat.message_model import ToolPointerMessage

_SUMMARY_TOKEN_BUDGET = 512
_DIFF_SAMPLE_LINES = 40
_PLAIN_SAMPLE_CHARS = 4 * _SUMMARY_TOKEN_BUDGET
_DIFF_HUNK_PATTERN = re.compile(r"^@@.+@@", re.MULTILINE)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text.encode("utf-8", errors="ignore")) / 4))


def _looks_like_diff(text: str) -> bool:
    if not text:
        return False
    if text.lstrip().startswith("--- ") or text.lstrip().startswith("@@"):
        return True
    return bool(_DIFF_HUNK_PATTERN.search(text))


def _summarize_plaintext(text: str, *, max_chars: int) -> str:
    stripped = text.strip()
    if not stripped:
        return "(pointer placeholder: empty)"
    condensed = " ".join(stripped.split())
    if len(condensed) <= max_chars:
        return condensed
    # Preserve sentence-ish chunks where possible.
    sentences = re.split(r"(?<=[.!?])\s+", condensed)
    summary_parts: list[str] = []
    total = 0
    for sentence in sentences:
        if not sentence:
            continue
        fragment = sentence.strip()
        if not fragment:
            continue
        summary_parts.append(fragment)
        total += len(fragment) + 1
        if total >= max_chars:
            break
    summary = " ".join(summary_parts) or condensed
    if len(summary) > max_chars:
        summary = f"{summary[: max_chars - 1].rstrip()}…"
    return summary


def _summarize_diff(text: str, *, sample_lines: int) -> str:
    if not text:
        return "(pointer placeholder: empty diff)"
    additions = deletions = 0
    hunk_count = 0
    preview: list[str] = []
    for line in text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@"):
            hunk_count += 1
            preview.append(line.strip())
        elif line.startswith("+"):
            additions += 1
            preview.append(line[:120])
        elif line.startswith("-"):
            deletions += 1
            preview.append(line[:120])
        else:
            if len(preview) < sample_lines:
                preview.append(line[:120])
        if len(preview) >= sample_lines:
            break
    if hunk_count == 0 and preview:
        hunk_count = 1
    summary = [f"Diff stats: +{additions}/-{deletions} across {hunk_count} hunk(s)."]
    if preview:
        summary.append("Sample:")
        summary.extend(preview[:sample_lines])
    return "\n".join(summary)


@dataclass(slots=True)
class ToolPayload:
    """Normalized representation of a tool output to be summarized."""

    name: str
    content: str
    arguments: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(slots=True)
class SummaryResult:
    """Summary metadata emitted after compressing a tool payload."""

    kind: str
    summary_text: str
    original_tokens: int
    summary_tokens: int
    truncated: bool
    metadata: dict[str, Any] = field(default_factory=dict)


def summarize_tool_content(
    payload: ToolPayload,
    *,
    schema_hint: str | None = None,
    budget_tokens: int = _SUMMARY_TOKEN_BUDGET,
) -> SummaryResult:
    """Summarize ``payload.content`` using heuristics for diffs vs plaintext."""

    text = payload.content or ""
    kind = (schema_hint or "").strip().lower() or ("diff" if _looks_like_diff(text) else "text")
    original_tokens = _estimate_tokens(text)
    if kind == "diff":
        summary = _summarize_diff(text, sample_lines=_DIFF_SAMPLE_LINES)
    else:
        summary = _summarize_plaintext(text, max_chars=max(80, budget_tokens * 4))
    summary_tokens = _estimate_tokens(summary)
    truncated = summary_tokens < original_tokens
    metadata = {
        "schema_hint": schema_hint or kind,
        "original_tokens": original_tokens,
        "summary_tokens": summary_tokens,
        "truncated": truncated,
    }
    if payload.arguments:
        metadata["arguments"] = dict(payload.arguments)
    if payload.metadata:
        metadata["metadata"] = dict(payload.metadata)
    return SummaryResult(
        kind=kind,
        summary_text=summary,
        original_tokens=original_tokens,
        summary_tokens=summary_tokens,
        truncated=truncated,
        metadata=metadata,
    )


def build_pointer(
    summary: SummaryResult,
    *,
    tool_name: str,
    pointer_id: str | None = None,
    rehydrate_instructions: str | None = None,
) -> ToolPointerMessage:
    """Create a :class:`ToolPointerMessage` for downstream serialization."""

    pointer_identifier = pointer_id or uuid.uuid4().hex
    default_instructions = (
        rehydrate_instructions
        or f"Pointer {pointer_identifier} stores the raw output. Re-run {tool_name} (optionally with narrower scope) to rehydrate it."
    )
    pointer = ToolPointerMessage(
        pointer_id=pointer_identifier,
        kind=summary.kind or tool_name,
        display_text=summary.summary_text,
        rehydrate_instructions=default_instructions.strip(),
        metadata=dict(summary.metadata),
    )
    pointer.metadata.setdefault("tool_name", tool_name)
    return pointer


__all__ = [
    "ToolPayload",
    "SummaryResult",
    "summarize_tool_content",
    "build_pointer",
]
