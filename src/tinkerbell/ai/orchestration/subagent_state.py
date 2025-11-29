"""Subagent document state tracking for the AI controller."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SubagentDocumentState:
    """Tracks helper scheduling metadata per document."""

    last_job_hashes: dict[str, str] = field(default_factory=dict)
    edit_churn: int = 0
    last_edit_ts: float = 0.0
    last_job_ts: float = 0.0


__all__ = [
    "SubagentDocumentState",
]
