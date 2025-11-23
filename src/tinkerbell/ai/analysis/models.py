"""Dataclasses shared across the analysis package."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence


def _default_timestamp() -> float:
    return time.time()


@dataclass(slots=True)
class AnalysisWarning:
    """Structured warning surfaced to the operator and telemetry."""

    code: str
    message: str
    severity: str = "warning"

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "severity": self.severity}


@dataclass(slots=True)
class ToolRecommendation:
    """Represents a tool the analyzer believes should run."""

    name: str
    reason: str
    required: bool = True
    priority: int = 0


@dataclass(slots=True)
class AnalysisInput:
    """Snapshot of document/runtime state evaluated by the analyzer."""

    document_id: str
    document_version: str | None
    document_path: str | None = None
    span_start: int = 0
    span_end: int = 0
    document_chars: int | None = None
    chunk_profile_hint: str | None = None
    chunk_index_ready: bool = False
    chunk_manifest_profile: str | None = None
    chunk_manifest_cache_key: str | None = None
    outline_digest: str | None = None
    outline_age_seconds: float | None = None
    outline_version_id: int | None = None
    plot_state_status: str | None = None
    plot_override_version: int | None = None
    concordance_status: str | None = None
    concordance_age_seconds: float | None = None
    retrieval_enabled: bool = True
    extra_metadata: Mapping[str, object] | None = None
    chunk_flow_warnings: Sequence[str] | None = None
    plot_loop_flags: Sequence[str] | None = None

    def span_length(self) -> int:
        return max(0, self.span_end - self.span_start)

    def cache_key(self) -> tuple[object, ...]:
        return (
            self.document_id,
            self.document_version,
            self.span_start,
            self.span_end,
            self.outline_digest,
            self.plot_override_version,
            self.chunk_manifest_cache_key,
        )


@dataclass(slots=True)
class AnalysisFinding:
    """Intermediate result emitted by a rule."""

    trace: str
    chunk_profile: str | None = None
    required_tools: tuple[str, ...] = field(default_factory=tuple)
    optional_tools: tuple[str, ...] = field(default_factory=tuple)
    must_refresh_outline: bool | None = None
    plot_state_status: str | None = None
    concordance_status: str | None = None
    warnings: tuple[AnalysisWarning, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class AnalysisAdvice:
    """Final structured payload shared with prompts, UI, and telemetry."""

    document_id: str
    document_version: str | None
    chunk_profile: str = "auto"
    required_tools: tuple[str, ...] = field(default_factory=tuple)
    optional_tools: tuple[str, ...] = field(default_factory=tuple)
    must_refresh_outline: bool = False
    plot_state_status: str | None = None
    concordance_status: str | None = None
    warnings: tuple[AnalysisWarning, ...] = field(default_factory=tuple)
    rule_trace: tuple[str, ...] = field(default_factory=tuple)
    cache_state: str = "miss"
    generated_at: float = field(default_factory=_default_timestamp)

    def with_cache_state(self, cache_state: str) -> "AnalysisAdvice":
        if cache_state == self.cache_state:
            return self
        return replace(self, cache_state=cache_state)

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "document_version": self.document_version,
            "chunk_profile": self.chunk_profile,
            "required_tools": list(self.required_tools),
            "optional_tools": list(self.optional_tools),
            "must_refresh_outline": self.must_refresh_outline,
            "plot_state_status": self.plot_state_status,
            "concordance_status": self.concordance_status,
            "warnings": [warning.as_dict() for warning in self.warnings],
            "rule_trace": list(self.rule_trace),
            "cache_state": self.cache_state,
            "generated_at": self.generated_at,
        }
