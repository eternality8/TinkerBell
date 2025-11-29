"""Tests for the budget service.

This module tests the BudgetService class that provides context
budget management for the turn pipeline.
"""

from __future__ import annotations

import pytest
from typing import Any, Mapping

from tinkerbell.ai.orchestration.services.budget import (
    BudgetService,
    BudgetConfig,
    BudgetEvaluation,
    BudgetExceededError,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config() -> BudgetConfig:
    """Create a test budget configuration."""
    return BudgetConfig(
        enabled=True,
        dry_run=False,
        prompt_budget=10_000,
        response_reserve=1_000,
        emergency_buffer=2_000,
    )


@pytest.fixture
def service(config: BudgetConfig) -> BudgetService:
    """Create a budget service with test config."""
    return BudgetService(config)


@pytest.fixture
def disabled_service() -> BudgetService:
    """Create a disabled budget service."""
    return BudgetService(BudgetConfig.disabled())


# =============================================================================
# BudgetConfig Tests
# =============================================================================


class TestBudgetConfig:
    """Tests for BudgetConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = BudgetConfig()
        
        assert config.enabled is True
        assert config.dry_run is False
        assert config.prompt_budget == 100_000
        assert config.response_reserve == 4_000
        assert config.emergency_buffer == 2_000

    def test_custom_values(self):
        """Should accept custom values."""
        config = BudgetConfig(
            enabled=False,
            dry_run=True,
            prompt_budget=50_000,
            response_reserve=2_000,
            emergency_buffer=1_000,
            model_name="test-model",
            max_context_tokens=128_000,
        )
        
        assert config.enabled is False
        assert config.dry_run is True
        assert config.prompt_budget == 50_000
        assert config.response_reserve == 2_000
        assert config.emergency_buffer == 1_000
        assert config.model_name == "test-model"
        assert config.max_context_tokens == 128_000

    def test_frozen(self):
        """Config should be frozen."""
        config = BudgetConfig()
        
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore

    def test_disabled_factory(self):
        """disabled() should create disabled config."""
        config = BudgetConfig.disabled()
        
        assert config.enabled is False
        assert config.dry_run is True


# =============================================================================
# BudgetEvaluation Tests
# =============================================================================


class TestBudgetEvaluation:
    """Tests for BudgetEvaluation dataclass."""

    def test_is_ok(self):
        """is_ok should return True for ok verdict."""
        evaluation = BudgetEvaluation(
            verdict="ok",
            reason="within-budget",
            prompt_tokens=5000,
            budget_tokens=10000,
            response_reserve=1000,
        )
        
        assert evaluation.is_ok is True
        assert evaluation.needs_summary is False
        assert evaluation.is_rejected is False

    def test_needs_summary(self):
        """needs_summary should return True for needs_summary verdict."""
        evaluation = BudgetEvaluation(
            verdict="needs_summary",
            reason="exceeds-budget",
            prompt_tokens=11000,
            budget_tokens=10000,
            response_reserve=1000,
            deficit=1000,
        )
        
        assert evaluation.is_ok is False
        assert evaluation.needs_summary is True
        assert evaluation.is_rejected is False

    def test_is_rejected(self):
        """is_rejected should return True for reject verdict."""
        evaluation = BudgetEvaluation(
            verdict="reject",
            reason="exceeds-emergency",
            prompt_tokens=15000,
            budget_tokens=10000,
            response_reserve=1000,
            deficit=3000,
        )
        
        assert evaluation.is_ok is False
        assert evaluation.needs_summary is False
        assert evaluation.is_rejected is True

    def test_headroom_positive(self):
        """headroom should be positive when under budget."""
        evaluation = BudgetEvaluation(
            verdict="ok",
            reason="within-budget",
            prompt_tokens=5000,
            budget_tokens=10000,
            response_reserve=1000,
        )
        
        assert evaluation.headroom == 5000

    def test_headroom_negative(self):
        """headroom should be negative when over budget."""
        evaluation = BudgetEvaluation(
            verdict="needs_summary",
            reason="exceeds-budget",
            prompt_tokens=12000,
            budget_tokens=10000,
            response_reserve=1000,
        )
        
        assert evaluation.headroom == -2000

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        evaluation = BudgetEvaluation(
            verdict="ok",
            reason="within-budget",
            prompt_tokens=5000,
            budget_tokens=10000,
            response_reserve=1000,
            pending_tool_tokens=500,
            deficit=0,
            dry_run=False,
            document_id="doc-1",
            timestamp=1234567890.0,
        )
        
        result = evaluation.to_dict()
        
        assert result["verdict"] == "ok"
        assert result["reason"] == "within-budget"
        assert result["prompt_tokens"] == 5000
        assert result["budget_tokens"] == 10000
        assert result["response_reserve"] == 1000
        assert result["pending_tool_tokens"] == 500
        assert result["deficit"] == 0
        assert result["dry_run"] is False
        assert result["document_id"] == "doc-1"
        assert result["timestamp"] == 1234567890.0
        assert result["headroom"] == 5000

    def test_frozen(self):
        """Evaluation should be frozen."""
        evaluation = BudgetEvaluation(
            verdict="ok",
            reason="test",
            prompt_tokens=5000,
            budget_tokens=10000,
            response_reserve=1000,
        )
        
        with pytest.raises(AttributeError):
            evaluation.verdict = "reject"  # type: ignore


# =============================================================================
# BudgetExceededError Tests
# =============================================================================


class TestBudgetExceededError:
    """Tests for BudgetExceededError exception."""

    def test_stores_evaluation(self):
        """Should store the evaluation that caused rejection."""
        evaluation = BudgetEvaluation(
            verdict="reject",
            reason="exceeds-emergency",
            prompt_tokens=15000,
            budget_tokens=10000,
            response_reserve=1000,
        )
        
        error = BudgetExceededError(evaluation)
        
        assert error.evaluation is evaluation
        assert "exceeds-emergency" in str(error)


# =============================================================================
# BudgetService Basic Tests
# =============================================================================


class TestBudgetServiceBasic:
    """Tests for basic BudgetService operations."""

    def test_init_with_default_config(self):
        """Should initialize with default config."""
        service = BudgetService()
        
        assert service.config.enabled is True
        assert service.enabled is True
        assert service.last_evaluation is None

    def test_init_with_custom_config(self, config: BudgetConfig):
        """Should initialize with custom config."""
        service = BudgetService(config)
        
        assert service.config is config
        assert service.enabled is True

    def test_init_disabled(self, disabled_service: BudgetService):
        """Should initialize in disabled state."""
        assert disabled_service.enabled is False


# =============================================================================
# BudgetService.evaluate Tests
# =============================================================================


class TestBudgetServiceEvaluate:
    """Tests for BudgetService.evaluate method."""

    def test_within_budget_returns_ok(self, service: BudgetService):
        """Should return ok for prompt within budget."""
        result = service.evaluate(prompt_tokens=5000)
        
        assert result.is_ok is True
        assert result.verdict == "ok"
        assert result.prompt_tokens == 5000

    def test_exceeds_budget_returns_needs_summary(self, service: BudgetService):
        """Should return needs_summary when exceeding budget but not emergency."""
        # Budget is 10000, emergency buffer is 2000, so 11000 should need summary
        result = service.evaluate(prompt_tokens=11000)
        
        assert result.needs_summary is True
        assert result.verdict == "needs_summary"

    def test_exceeds_emergency_returns_reject(self, service: BudgetService):
        """Should return reject when exceeding emergency buffer."""
        # Budget is 10000, emergency buffer is 2000, so >12000 should reject
        result = service.evaluate(prompt_tokens=15000)
        
        assert result.is_rejected is True
        assert result.verdict == "reject"

    def test_stores_last_evaluation(self, service: BudgetService):
        """Should store the last evaluation."""
        result = service.evaluate(prompt_tokens=5000)
        
        assert service.last_evaluation is result

    def test_pending_tool_tokens_included(self, service: BudgetService):
        """Should include pending tool tokens in evaluation."""
        # 8000 prompt + 3000 pending = 11000 total, exceeds 10000 budget
        result = service.evaluate(
            prompt_tokens=8000,
            pending_tool_tokens=3000,
        )
        
        assert result.needs_summary is True
        assert result.pending_tool_tokens == 3000

    def test_document_id_included(self, service: BudgetService):
        """Should include document ID in evaluation."""
        result = service.evaluate(prompt_tokens=5000, document_id="doc-1")
        
        assert result.document_id == "doc-1"

    def test_disabled_always_ok(self, disabled_service: BudgetService):
        """Disabled service should always return ok."""
        result = disabled_service.evaluate(prompt_tokens=1_000_000)
        
        assert result.is_ok is True
        assert result.dry_run is True

    def test_raise_on_reject(self, service: BudgetService):
        """Should raise when raise_on_reject is True and rejected."""
        with pytest.raises(BudgetExceededError) as exc_info:
            service.evaluate(prompt_tokens=15000, raise_on_reject=True)
        
        assert exc_info.value.evaluation.is_rejected is True

    def test_no_raise_on_dry_run(self):
        """Should not raise on dry-run even if rejected."""
        config = BudgetConfig(
            enabled=True,
            dry_run=True,  # Dry-run mode
            prompt_budget=10_000,
            emergency_buffer=2_000,
        )
        service = BudgetService(config)
        
        # Should not raise because dry_run is True
        result = service.evaluate(prompt_tokens=15000, raise_on_reject=True)
        
        assert result.is_rejected is True
        assert result.dry_run is True

    def test_telemetry_emitter_called(self, config: BudgetConfig):
        """Should call telemetry emitter on evaluation."""
        emissions: list[tuple[str, Mapping[str, Any]]] = []
        
        def emitter(event: str, payload: Mapping[str, Any]) -> None:
            emissions.append((event, payload))
        
        service = BudgetService(config, telemetry_emitter=emitter)
        service.evaluate(prompt_tokens=5000)
        
        assert len(emissions) == 1
        assert emissions[0][0] == "budget_evaluation"
        assert emissions[0][1]["prompt_tokens"] == 5000

    def test_telemetry_can_be_suppressed(self, config: BudgetConfig):
        """Should suppress telemetry when requested."""
        emissions: list[tuple[str, Mapping[str, Any]]] = []
        
        def emitter(event: str, payload: Mapping[str, Any]) -> None:
            emissions.append((event, payload))
        
        service = BudgetService(config, telemetry_emitter=emitter)
        service.evaluate(prompt_tokens=5000, emit_telemetry=False)
        
        assert len(emissions) == 0


# =============================================================================
# BudgetService.record_usage Tests
# =============================================================================


class TestBudgetServiceRecordUsage:
    """Tests for BudgetService.record_usage method."""

    def test_records_usage(self, service: BudgetService):
        """Should record usage without error."""
        # Just verify it doesn't raise
        service.record_usage(
            turn_id="turn-1",
            prompt_tokens=5000,
            response_reserve=1000,
        )


# =============================================================================
# BudgetService.status_snapshot Tests
# =============================================================================


class TestBudgetServiceStatusSnapshot:
    """Tests for BudgetService.status_snapshot method."""

    def test_returns_snapshot(self, service: BudgetService):
        """Should return status snapshot."""
        snapshot = service.status_snapshot()
        
        assert "enabled" in snapshot
        assert "prompt_budget" in snapshot
        assert snapshot["enabled"] is True

    def test_includes_last_evaluation(self, service: BudgetService):
        """Should include last evaluation if available."""
        service.evaluate(prompt_tokens=5000)
        
        snapshot = service.status_snapshot()
        
        assert "last_evaluation" in snapshot
        assert snapshot["last_evaluation"]["prompt_tokens"] == 5000


# =============================================================================
# BudgetService.reconfigure Tests
# =============================================================================


class TestBudgetServiceReconfigure:
    """Tests for BudgetService.reconfigure method."""

    def test_reconfigure_changes_config(self, service: BudgetService):
        """Should update configuration."""
        new_config = BudgetConfig(prompt_budget=50_000)
        
        service.reconfigure(new_config)
        
        assert service.config is new_config
        assert service.config.prompt_budget == 50_000

    def test_reconfigure_affects_evaluations(self, service: BudgetService):
        """Should affect subsequent evaluations."""
        # Original budget is 10000, should need summary at 11000
        result1 = service.evaluate(prompt_tokens=11000)
        assert result1.needs_summary is True
        
        # Reconfigure to larger budget
        service.reconfigure(BudgetConfig(prompt_budget=50_000))
        
        # Same tokens should now be ok
        result2 = service.evaluate(prompt_tokens=11000)
        assert result2.is_ok is True

    def test_reconfigure_to_disabled(self, service: BudgetService):
        """Should be able to disable via reconfigure."""
        service.reconfigure(BudgetConfig.disabled())
        
        assert service.enabled is False
        
        # Should always return ok when disabled
        result = service.evaluate(prompt_tokens=1_000_000)
        assert result.is_ok is True
