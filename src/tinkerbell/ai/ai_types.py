"""Shared typing contracts for AI infrastructure."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, Sequence


class TokenCounterProtocol(Protocol):
    """Protocol describing tokenizer implementations."""

    model_name: str | None

    def count(self, text: str) -> int:
        """Return the precise token count for *text*."""
        ...

    def estimate(self, text: str) -> int:
        """Return a deterministic fallback estimate when precise counts fail."""
        ...


@dataclass(slots=True)
class AgentRetryPolicy:
    """Deterministic retry/backoff configuration for LangGraph nodes."""

    planner_max_retries: int = 2
    tool_retry_limit: int = 2
    validation_retry_limit: int = 1
    backoff_min_seconds: float = 0.75
    backoff_max_seconds: float = 6.0
    tool_timeout_seconds: float = 35.0

    def as_metadata(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(slots=True)
class AgentConfig:
    """Tunable parameters that shape the declarative agent graph."""

    max_iterations: int = 8
    allow_parallel_tools: bool = False
    diff_required_for_edits: bool = True
    retry_policy: AgentRetryPolicy = field(default_factory=AgentRetryPolicy)

    def clamp(self) -> AgentConfig:
        """Clamp values into safe operating ranges and return ``self``."""

        self.max_iterations = max(1, min(int(self.max_iterations or 1), 50))
        self.retry_policy.planner_max_retries = max(0, int(self.retry_policy.planner_max_retries))
        self.retry_policy.tool_retry_limit = max(0, int(self.retry_policy.tool_retry_limit))
        self.retry_policy.validation_retry_limit = max(0, int(self.retry_policy.validation_retry_limit))
        self.retry_policy.backoff_min_seconds = max(0.05, float(self.retry_policy.backoff_min_seconds))
        self.retry_policy.backoff_max_seconds = max(
            self.retry_policy.backoff_min_seconds,
            float(self.retry_policy.backoff_max_seconds),
        )
        self.retry_policy.tool_timeout_seconds = max(1.0, float(self.retry_policy.tool_timeout_seconds))
        return self


@dataclass(slots=True)
class SubagentBudget:
    """Quota configuration applied to individual subagent jobs."""

    max_prompt_tokens: int = 2_000
    max_completion_tokens: int = 800
    max_runtime_seconds: float = 30.0
    max_tool_iterations: int = 0

    def clamp(self) -> SubagentBudget:
        self.max_prompt_tokens = max(256, int(self.max_prompt_tokens or 256))
        self.max_completion_tokens = max(128, int(self.max_completion_tokens or 128))
        self.max_runtime_seconds = max(1.0, float(self.max_runtime_seconds or 1.0))
        self.max_tool_iterations = max(0, int(self.max_tool_iterations or 0))
        return self


@dataclass(slots=True)
class ChunkReference:
    """Lightweight pointer that identifies the document span a subagent analyzes."""

    document_id: str
    chunk_id: str
    version_id: str | None = None
    pointer_id: str | None = None
    char_range: tuple[int, int] | None = None
    outline_node_id: str | None = None
    token_estimate: int | None = None
    chunk_hash: str | None = None
    preview: str | None = None

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "version_id": self.version_id,
            "pointer_id": self.pointer_id,
            "char_range": self.char_range,
            "outline_node_id": self.outline_node_id,
            "token_estimate": self.token_estimate,
            "chunk_hash": self.chunk_hash,
            "preview": self.preview,
        }
        return {key: value for key, value in payload.items() if value not in (None, "")}


class SubagentJobState(str, Enum):
    """State machine for subagent jobs."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class SubagentJobResult:
    """Outcome captured after a subagent finishes (or fails)."""

    status: str
    summary: str
    details: str | None = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    tool_calls: Sequence[dict[str, Any]] = ()

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "summary": self.summary,
            "details": self.details,
            "tokens_used": self.tokens_used,
            "latency_ms": round(self.latency_ms, 3),
            "tool_calls": list(self.tool_calls or ()),
        }
        return {key: value for key, value in payload.items() if value not in (None, "")}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class SubagentJob:
    """Work item that scopes a chunk-level analysis task."""

    job_id: str
    parent_run_id: str
    instructions: str
    chunk_ref: ChunkReference
    allowed_tools: tuple[str, ...]
    budget: SubagentBudget
    state: SubagentJobState = SubagentJobState.QUEUED
    created_at: datetime = field(default_factory=_utcnow)
    result: SubagentJobResult | None = None
    dedup_hash: str | None = None

    def as_payload(self) -> dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "parent_run_id": self.parent_run_id,
            "state": self.state.value,
            "created_at": self.created_at.isoformat(),
            "allowed_tools": list(self.allowed_tools),
            "instructions": self.instructions,
            "chunk": self.chunk_ref.as_payload(),
            "budget": asdict(self.budget),
            "result": self.result.as_payload() if self.result else None,
            "dedup_hash": self.dedup_hash,
        }
        return {key: value for key, value in payload.items() if value not in (None, "", [])}


@dataclass(slots=True)
class SubagentRuntimeConfig:
    """Feature flag + tuning knobs for the subagent sandbox."""

    enabled: bool = False
    max_jobs_per_turn: int = 2
    selection_min_chars: int = 400
    chunk_preview_chars: int = 1_200
    plot_outline_min_chars: int = 400
    allowed_tools: tuple[str, ...] = (
        "document_snapshot",
        "document_outline",
        "document_find_sections",
    )
    instructions_template: str = (
        "You are a focused editing subagent. Analyze only the provided chunk, summarize its intent, "
        "list any risks or continuity issues, and return concrete follow-up suggestions for the main controller."
    )
    plot_scaffolding_enabled: bool = False

    def clamp(self) -> SubagentRuntimeConfig:
        self.max_jobs_per_turn = max(0, int(self.max_jobs_per_turn or 0))
        self.selection_min_chars = max(0, int(self.selection_min_chars or 0))
        self.chunk_preview_chars = max(200, int(self.chunk_preview_chars or 200))
        self.plot_outline_min_chars = max(0, int(self.plot_outline_min_chars or 0))
        allowed = tuple(tool.strip() for tool in self.allowed_tools if tool)
        self.allowed_tools = allowed or (
            "document_snapshot",
            "document_outline",
            "document_find_sections",
        )
        self.plot_scaffolding_enabled = bool(self.plot_scaffolding_enabled)
        return self


__all__ = [
    "AgentConfig",
    "AgentRetryPolicy",
    "TokenCounterProtocol",
    "SubagentBudget",
    "ChunkReference",
    "SubagentJob",
    "SubagentJobResult",
    "SubagentJobState",
    "SubagentRuntimeConfig",
]
