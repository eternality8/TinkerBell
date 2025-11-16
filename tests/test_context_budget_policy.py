"""Tests for the context budget policy helpers."""

from __future__ import annotations

from tinkerbell.ai.services.context_policy import ContextBudgetPolicy
from tinkerbell.services.settings import ContextPolicySettings


def _policy(**overrides: int | bool | None) -> ContextBudgetPolicy:
    settings = ContextPolicySettings(**overrides)
    return ContextBudgetPolicy.from_settings(
        settings,
        model_name="test-model",
        max_context_tokens=120_000,
        response_token_reserve=16_000,
    )


def test_policy_from_settings_applies_overrides() -> None:
    policy = _policy(enabled=True, dry_run=False, prompt_budget_override=60_000, response_reserve_override=12_000)

    assert policy.enabled is True
    assert policy.dry_run is False
    assert policy.prompt_budget == 60_000
    assert policy.response_reserve == 12_000


def test_policy_tokens_available_flags_needs_summary() -> None:
    settings = ContextPolicySettings(enabled=True, dry_run=True, prompt_budget_override=10_000, emergency_buffer=2_000)
    policy = ContextBudgetPolicy.from_settings(
        settings,
        model_name="test",
        max_context_tokens=16_000,
        response_token_reserve=2_000,
    )

    decision = policy.tokens_available(prompt_tokens=11_500, response_reserve=2_000, pending_tool_tokens=0, document_id="doc-1")
    assert decision.verdict == "needs_summary"
    assert decision.deficit > 0

    reject = policy.tokens_available(prompt_tokens=15_000, response_reserve=2_000, pending_tool_tokens=0, document_id="doc-1")
    assert reject.verdict == "reject"


def test_policy_status_snapshot_shows_latest_decision() -> None:
    policy = _policy(enabled=True, dry_run=True, prompt_budget_override=8_000)
    policy.tokens_available(prompt_tokens=7_000, response_reserve=2_000, pending_tool_tokens=0, document_id="sample")
    snapshot = policy.status_snapshot()
    assert snapshot["enabled"] is True
    assert "Budget:" in snapshot["summary_text"]
