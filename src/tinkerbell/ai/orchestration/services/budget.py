"""Budget service for the orchestration pipeline.

This module provides a service layer for context budget management,
wrapping the existing ContextBudgetPolicy with a cleaner interface
for the turn pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ...services.context_policy import BudgetDecision, BudgetVerdict, ContextBudgetPolicy

__all__ = [
    "BudgetService",
    "BudgetConfig",
    "BudgetEvaluation",
    "BudgetExceededError",
]

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class BudgetExceededError(RuntimeError):
    """Raised when the context budget policy rejects a prompt.
    
    Attributes:
        evaluation: The evaluation that caused the rejection.
    """

    def __init__(self, evaluation: "BudgetEvaluation") -> None:
        super().__init__(f"Context budget exceeded: {evaluation.reason}")
        self.evaluation = evaluation


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class BudgetConfig:
    """Configuration for the budget service.

    Attributes:
        enabled: Whether budget enforcement is enabled.
        dry_run: If True, log violations but don't reject.
        prompt_budget: Maximum tokens for prompts.
        response_reserve: Tokens reserved for response.
        emergency_buffer: Extra buffer before hard rejection.
        model_name: The model name for context sizing.
        max_context_tokens: Maximum context window size.
    """

    enabled: bool = True
    dry_run: bool = False
    prompt_budget: int = 100_000
    response_reserve: int = 4_000
    emergency_buffer: int = 2_000
    model_name: str | None = None
    max_context_tokens: int | None = None

    @classmethod
    def disabled(cls) -> "BudgetConfig":
        """Create a disabled budget configuration."""
        return cls(enabled=False, dry_run=True)


# -----------------------------------------------------------------------------
# Evaluation Result
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class BudgetEvaluation:
    """Result of evaluating a prompt against the budget.

    Attributes:
        verdict: The budget decision (ok, needs_summary, reject).
        reason: Human-readable explanation.
        prompt_tokens: Number of tokens in the prompt.
        budget_tokens: The configured budget limit.
        response_reserve: Tokens reserved for response.
        pending_tool_tokens: Additional tokens for pending tools.
        deficit: How many tokens over budget (0 if within).
        dry_run: Whether this is a dry-run evaluation.
        document_id: Optional document context.
        timestamp: When the evaluation occurred.
    """

    verdict: BudgetVerdict
    reason: str
    prompt_tokens: int
    budget_tokens: int
    response_reserve: int
    pending_tool_tokens: int = 0
    deficit: int = 0
    dry_run: bool = False
    document_id: str | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def is_ok(self) -> bool:
        """Whether the prompt is within budget."""
        return self.verdict == "ok"

    @property
    def needs_summary(self) -> bool:
        """Whether summarization is recommended."""
        return self.verdict == "needs_summary"

    @property
    def is_rejected(self) -> bool:
        """Whether the prompt was rejected."""
        return self.verdict == "reject"

    @property
    def headroom(self) -> int:
        """Tokens remaining before hitting budget (can be negative)."""
        return self.budget_tokens - self.prompt_tokens

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for logging/telemetry."""
        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "prompt_tokens": self.prompt_tokens,
            "budget_tokens": self.budget_tokens,
            "response_reserve": self.response_reserve,
            "pending_tool_tokens": self.pending_tool_tokens,
            "deficit": self.deficit,
            "dry_run": self.dry_run,
            "document_id": self.document_id,
            "timestamp": self.timestamp,
            "headroom": self.headroom,
        }

    @classmethod
    def from_decision(cls, decision: BudgetDecision) -> "BudgetEvaluation":
        """Create from a BudgetDecision."""
        return cls(
            verdict=decision.verdict,
            reason=decision.reason,
            prompt_tokens=decision.prompt_tokens,
            budget_tokens=decision.prompt_budget,
            response_reserve=decision.response_reserve,
            pending_tool_tokens=decision.pending_tool_tokens,
            deficit=decision.deficit,
            dry_run=decision.dry_run,
            document_id=decision.document_id,
            timestamp=decision.timestamp,
        )


# -----------------------------------------------------------------------------
# Budget Service
# -----------------------------------------------------------------------------


class BudgetService:
    """Service for managing context window budget.

    Provides budget evaluation and tracking for the turn pipeline.
    Wraps ContextBudgetPolicy with a cleaner interface.

    Example:
        >>> service = BudgetService(BudgetConfig(prompt_budget=50000))
        >>> eval = service.evaluate(prompt_tokens=30000)
        >>> if eval.is_ok:
        ...     print("Prompt fits within budget!")
        >>> elif eval.needs_summary:
        ...     print("Consider summarizing to fit budget")
    """

    def __init__(
        self,
        config: BudgetConfig | None = None,
        *,
        telemetry_emitter: Callable[[str, Mapping[str, Any]], Any] | None = None,
    ) -> None:
        """Initialize the budget service.

        Args:
            config: Budget configuration. Uses defaults if not provided.
            telemetry_emitter: Optional callback for telemetry events.
        """
        self._config = config or BudgetConfig()
        self._telemetry_emitter = telemetry_emitter
        self._policy = self._create_policy()
        self._last_evaluation: BudgetEvaluation | None = None

    @property
    def config(self) -> BudgetConfig:
        """The service configuration."""
        return self._config

    @property
    def enabled(self) -> bool:
        """Whether budget enforcement is enabled."""
        return self._config.enabled

    @property
    def last_evaluation(self) -> BudgetEvaluation | None:
        """The most recent evaluation result."""
        return self._last_evaluation

    def evaluate(
        self,
        *,
        prompt_tokens: int,
        response_reserve: int | None = None,
        pending_tool_tokens: int = 0,
        document_id: str | None = None,
        emit_telemetry: bool = True,
        raise_on_reject: bool = False,
    ) -> BudgetEvaluation:
        """Evaluate whether a prompt fits within the budget.

        Args:
            prompt_tokens: Number of tokens in the prompt.
            response_reserve: Override for response token reserve.
            pending_tool_tokens: Additional tokens for pending tool calls.
            document_id: Optional document identifier for context.
            emit_telemetry: Whether to emit telemetry for this evaluation.
            raise_on_reject: Whether to raise BudgetExceededError on rejection.

        Returns:
            The evaluation result.

        Raises:
            BudgetExceededError: If raise_on_reject is True and prompt is rejected.
        """
        decision = self._policy.tokens_available(
            prompt_tokens=prompt_tokens,
            response_reserve=response_reserve,
            pending_tool_tokens=pending_tool_tokens,
            document_id=document_id,
        )

        evaluation = BudgetEvaluation.from_decision(decision)
        self._last_evaluation = evaluation

        # Emit telemetry
        if emit_telemetry and self._telemetry_emitter is not None:
            try:
                self._telemetry_emitter("budget_evaluation", evaluation.to_dict())
            except Exception:
                LOGGER.debug("Failed to emit budget telemetry", exc_info=True)

        # Log significant decisions
        if evaluation.is_rejected:
            LOGGER.warning(
                "Budget rejected: %d tokens exceeds budget of %d (deficit: %d)",
                evaluation.prompt_tokens,
                evaluation.budget_tokens,
                evaluation.deficit,
            )
        elif evaluation.needs_summary:
            LOGGER.info(
                "Budget warning: %d tokens exceeds soft limit of %d",
                evaluation.prompt_tokens,
                evaluation.budget_tokens,
            )

        # Raise if requested and rejected (and not dry-run)
        if raise_on_reject and evaluation.is_rejected and not evaluation.dry_run:
            raise BudgetExceededError(evaluation)

        return evaluation

    def record_usage(
        self,
        turn_id: str,
        *,
        prompt_tokens: int,
        response_reserve: int | None = None,
    ) -> None:
        """Record actual token usage for a turn.

        Args:
            turn_id: Identifier for the turn.
            prompt_tokens: Actual tokens used in prompt.
            response_reserve: Response reserve used.
        """
        self._policy.record_usage(
            turn_id,
            prompt_tokens=prompt_tokens,
            response_reserve=response_reserve,
        )

    def status_snapshot(self) -> dict[str, Any]:
        """Get a status snapshot for UI display.

        Returns:
            Dictionary with current budget status.
        """
        snapshot = self._policy.status_snapshot()
        
        # Add evaluation info if available
        if self._last_evaluation is not None:
            snapshot["last_evaluation"] = self._last_evaluation.to_dict()
        
        return snapshot

    def reconfigure(self, config: BudgetConfig) -> None:
        """Reconfigure the service with new settings.

        Args:
            config: New configuration to apply.
        """
        self._config = config
        self._policy = self._create_policy()
        LOGGER.debug(
            "Budget service reconfigured: enabled=%s, budget=%d",
            config.enabled,
            config.prompt_budget,
        )

    def _create_policy(self) -> ContextBudgetPolicy:
        """Create a ContextBudgetPolicy from config."""
        if not self._config.enabled:
            return ContextBudgetPolicy.disabled(
                model_name=self._config.model_name,
                max_context_tokens=self._config.max_context_tokens,
            )

        return ContextBudgetPolicy(
            model_name=self._config.model_name,
            enabled=self._config.enabled,
            dry_run=self._config.dry_run,
            prompt_budget=self._config.prompt_budget,
            response_reserve=self._config.response_reserve,
            emergency_buffer=self._config.emergency_buffer,
            max_context_tokens=self._config.max_context_tokens,
        )
