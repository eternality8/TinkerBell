"""Context budget policy primitives for dry-run enforcement."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Literal

from ...services.settings import ContextPolicySettings

BudgetVerdict = Literal["ok", "needs_summary", "reject"]


@dataclass(slots=True)
class BudgetDecision:
    """Outcome from evaluating a prompt against the active budget policy."""

    verdict: BudgetVerdict
    reason: str
    prompt_tokens: int
    prompt_budget: int
    response_reserve: int
    pending_tool_tokens: int = 0
    deficit: int = 0
    dry_run: bool = True
    model: str | None = None
    document_id: str | None = None
    timestamp: float = field(default_factory=lambda: time.time())

    def as_payload(self) -> dict[str, object]:
        """Return a telemetry-friendly dictionary for this decision."""

        return {
            "verdict": self.verdict,
            "reason": self.reason,
            "prompt_tokens": int(self.prompt_tokens),
            "prompt_budget": int(self.prompt_budget),
            "response_reserve": int(self.response_reserve),
            "pending_tool_tokens": int(self.pending_tool_tokens),
            "deficit": int(max(0, self.deficit)),
            "dry_run": bool(self.dry_run),
            "model": self.model,
            "document_id": self.document_id,
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class ContextBudgetPolicy:
    """Executes prompt budgeting decisions ahead of controller turns."""

    model_name: str | None
    enabled: bool
    dry_run: bool
    prompt_budget: int
    response_reserve: int
    emergency_buffer: int
    history_limit: int = 50
    max_context_tokens: int | None = None
    _recent_decisions: Deque[BudgetDecision] = field(default_factory=deque, init=False, repr=False)
    _recent_usage: Deque[dict[str, int]] = field(default_factory=deque, init=False, repr=False)

    @classmethod
    def disabled(cls, model_name: str | None = None, *, max_context_tokens: int | None = None) -> "ContextBudgetPolicy":
        return cls(
            model_name=model_name,
            enabled=False,
            dry_run=True,
            prompt_budget=max(0, int(max_context_tokens or 0)),
            response_reserve=0,
            emergency_buffer=0,
        )

    @classmethod
    def from_settings(
        cls,
        settings: ContextPolicySettings | None,
        *,
        model_name: str | None,
        max_context_tokens: int,
        response_token_reserve: int,
    ) -> "ContextBudgetPolicy":
        policy_settings = settings or ContextPolicySettings()
        prompt_budget = policy_settings.prompt_budget_override
        if prompt_budget is None:
            prompt_budget = max(0, max_context_tokens - response_token_reserve)
        prompt_budget = max(1, min(int(prompt_budget), max_context_tokens))
        response_reserve = policy_settings.response_reserve_override
        if response_reserve is None:
            response_reserve = response_token_reserve
        response_reserve = max(0, min(int(response_reserve), max_context_tokens))
        emergency_buffer = max(512, int(getattr(policy_settings, "emergency_buffer", 2_000)))
        return cls(
            model_name=model_name,
            enabled=bool(policy_settings.enabled),
            dry_run=bool(policy_settings.dry_run),
            prompt_budget=prompt_budget,
            response_reserve=response_reserve,
            emergency_buffer=emergency_buffer,
            max_context_tokens=max_context_tokens,
        )

    def tokens_available(
        self,
        *,
        prompt_tokens: int,
        response_reserve: int | None = None,
        pending_tool_tokens: int = 0,
        document_id: str | None = None,
        **legacy_kwargs: Any,
    ) -> BudgetDecision:
        """Return a decision describing whether the prompt fits the active budget."""

        total_prompt = max(0, int(prompt_tokens)) + max(0, int(pending_tool_tokens))
        if response_reserve is None:
            response_reserve = legacy_kwargs.pop("response_tokens_reserved", None)
        reserve = response_reserve if response_reserve is not None else self.response_reserve
        reserve = max(0, int(reserve))
        budget = max(0, int(self.prompt_budget))
        emergency_limit = budget + max(0, int(self.emergency_buffer))

        if not self.enabled:
            decision = BudgetDecision(
                verdict="ok",
                reason="policy-disabled",
                prompt_tokens=total_prompt,
                prompt_budget=budget,
                response_reserve=reserve,
                pending_tool_tokens=max(0, int(pending_tool_tokens)),
                dry_run=True,
                model=self.model_name,
                document_id=document_id,
            )
            self._record_decision(decision)
            return decision

        headroom = budget - total_prompt
        if headroom >= 0:
            verdict: BudgetVerdict = "ok"
            reason = "within-budget"
            deficit = 0
        elif total_prompt <= emergency_limit:
            verdict = "needs_summary"
            reason = "exceeds-budget"
            deficit = -headroom
        else:
            verdict = "reject"
            reason = "exceeds-emergency"
            deficit = total_prompt - emergency_limit

        decision = BudgetDecision(
            verdict=verdict,
            reason=reason,
            prompt_tokens=total_prompt,
            prompt_budget=budget,
            response_reserve=reserve,
            pending_tool_tokens=max(0, int(pending_tool_tokens)),
            deficit=max(0, deficit),
            dry_run=self.dry_run,
            model=self.model_name,
            document_id=document_id,
        )
        self._record_decision(decision)
        return decision

    def record_usage(
        self,
        turn_id: str,
        *,
        prompt_tokens: int,
        response_reserve: int | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        """Persist the raw inputs used for a controller turn."""

        if response_reserve is None:
            response_reserve = legacy_kwargs.pop("response_tokens_reserved", None)
        entry = {
            "turn_id": str(turn_id or ""),
            "prompt_tokens": max(0, int(prompt_tokens)),
            "response_reserve": max(0, int(response_reserve or 0)),
            "timestamp": time.time(),
        }
        self._recent_usage.append(entry)
        while len(self._recent_usage) > self.history_limit:
            self._recent_usage.popleft()

    def status_snapshot(self) -> dict[str, object]:
        """Return a lightweight snapshot for UI widgets."""

        latest = self._recent_decisions[-1] if self._recent_decisions else None
        summary = self._format_summary(latest)
        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "prompt_budget": self.prompt_budget,
            "response_reserve": self.response_reserve,
            "emergency_buffer": self.emergency_buffer,
            "summary_text": summary,
            "verdict": latest.verdict if latest else None,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_decision(self, decision: BudgetDecision) -> None:
        self._recent_decisions.append(decision)
        while len(self._recent_decisions) > self.history_limit:
            self._recent_decisions.popleft()

    def _format_summary(self, decision: BudgetDecision | None) -> str:
        if not self.enabled:
            return "Budget: legacy (disabled)"
        if decision is None:
            return f"Budget: 0/{self.prompt_budget:,} tokens"
        verdict = decision.verdict.upper()
        used = decision.prompt_tokens
        total = decision.prompt_budget
        suffix = " (dry-run)" if self.dry_run else ""
        return f"Budget: {used:,}/{total:,} ({verdict}){suffix}"