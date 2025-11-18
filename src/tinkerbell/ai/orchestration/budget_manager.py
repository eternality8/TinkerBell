"""Budget supervision utilities extracted from the AI controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..services import telemetry as telemetry_service
from ..services.context_policy import BudgetDecision, ContextBudgetPolicy


class ContextBudgetExceeded(RuntimeError):
    """Raised when the context budget policy rejects a prompt."""

    def __init__(self, decision: BudgetDecision):
        super().__init__(f"Context budget exceeded: {decision.reason}")
        self.decision = decision


@dataclass(slots=True)
class BudgetManager:
    """Coordinates context-window policies and telemetry emission."""

    policy: ContextBudgetPolicy | None = None
    telemetry_emitter: Callable[[str, Mapping[str, Any]], Any] | None = field(
        default_factory=lambda: getattr(telemetry_service, "emit", None)
    )
    last_decision: BudgetDecision | None = None

    def configure_policy(
        self,
        policy: ContextBudgetPolicy | None,
        *,
        model_name: str | None,
        max_context_tokens: int,
    ) -> ContextBudgetPolicy:
        """Normalize and store the active budget policy."""

        if policy is None:
            policy = ContextBudgetPolicy.disabled(
                model_name=model_name,
                max_context_tokens=max_context_tokens,
            )
        self.policy = policy
        return policy

    def evaluate(
        self,
        *,
        prompt_tokens: int,
        response_reserve: int | None,
        document_id: str | None,
        pending_tool_tokens: int = 0,
        suppress_telemetry: bool = False,
        raise_on_reject: bool = True,
    ) -> BudgetDecision | None:
        """Consult the active policy and optionally raise on rejection."""

        policy = self.policy
        if policy is None:
            return None
        decision = policy.tokens_available(
            prompt_tokens=prompt_tokens,
            response_reserve=response_reserve,
            pending_tool_tokens=pending_tool_tokens,
            document_id=document_id,
        )
        self.last_decision = decision
        emitter = self.telemetry_emitter
        if not suppress_telemetry and callable(emitter):
            emitter("context_budget_decision", decision.as_payload())
        if raise_on_reject and decision.verdict == "reject" and not decision.dry_run:
            raise ContextBudgetExceeded(decision)
        return decision

    def record_usage(
        self,
        *,
        run_id: str,
        prompt_tokens: int,
        response_reserve: int | None,
    ) -> None:
        policy = self.policy
        if policy is None:
            return
        policy.record_usage(
            run_id,
            prompt_tokens=prompt_tokens,
            response_reserve=response_reserve,
        )

    def status_snapshot(self) -> dict[str, object] | None:
        policy = self.policy
        if policy is None:
            return None
        return policy.status_snapshot()
