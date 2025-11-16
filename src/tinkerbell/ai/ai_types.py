"""Shared typing contracts for AI infrastructure."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol


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


__all__ = ["AgentConfig", "AgentRetryPolicy", "TokenCounterProtocol"]
