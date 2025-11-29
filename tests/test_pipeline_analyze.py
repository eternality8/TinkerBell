"""Unit tests for the analyze pipeline stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pytest

from tinkerbell.ai.orchestration.pipeline.analyze import (
    analyze_turn,
    generate_hints,
    format_hints_block,
    AnalysisProvider,
)
from tinkerbell.ai.orchestration.transaction import DocumentSnapshot
from tinkerbell.ai.orchestration.types import (
    AnalyzedTurn,
    BudgetEstimate,
    Message,
    PreparedTurn,
    TurnConfig,
)


# -----------------------------------------------------------------------------
# Mock Analysis Types
# -----------------------------------------------------------------------------


@dataclass
class MockWarning:
    """Mock warning for testing."""
    code: str = "test_warning"
    message: str = "Test warning message"
    severity: str = "warning"


@dataclass
class MockAnalysisAdvice:
    """Mock analysis advice for testing."""
    document_id: str = "test-doc"
    chunk_profile: str = "auto"
    required_tools: tuple[str, ...] = ()
    optional_tools: tuple[str, ...] = ()
    must_refresh_outline: bool = False
    plot_state_status: str | None = None
    concordance_status: str | None = None
    warnings: tuple[MockWarning, ...] = ()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "chunk_profile": self.chunk_profile,
            "required_tools": list(self.required_tools),
            "optional_tools": list(self.optional_tools),
            "must_refresh_outline": self.must_refresh_outline,
            "plot_state_status": self.plot_state_status,
            "concordance_status": self.concordance_status,
        }


class MockAnalysisProvider:
    """Mock analysis provider for testing."""
    
    def __init__(
        self,
        advice: MockAnalysisAdvice | None = None,
        *,
        should_fail: bool = False,
    ) -> None:
        self.advice = advice
        self.should_fail = should_fail
        self.call_count = 0
        self.last_snapshot: Mapping[str, Any] | None = None
        self.last_source: str | None = None
        self.last_force_refresh: bool = False
    
    def run_analysis(
        self,
        snapshot: Mapping[str, Any],
        *,
        source: str = "pipeline",
        force_refresh: bool = False,
    ) -> MockAnalysisAdvice | None:
        self.call_count += 1
        self.last_snapshot = snapshot
        self.last_source = source
        self.last_force_refresh = force_refresh
        
        if self.should_fail:
            raise RuntimeError("Analysis failed")
        
        return self.advice


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def sample_snapshot() -> DocumentSnapshot:
    """Create a sample document snapshot."""
    return DocumentSnapshot(
        tab_id="test-doc",
        content="Hello, world!\nThis is a test document.",
        version_token="v123",
    )


@pytest.fixture
def sample_config() -> TurnConfig:
    """Create a sample turn configuration with analysis enabled."""
    return TurnConfig(
        max_iterations=5,
        analysis_enabled=True,
        max_context_tokens=100_000,
    )


@pytest.fixture
def config_no_analysis() -> TurnConfig:
    """Create a configuration with analysis disabled."""
    return TurnConfig(
        analysis_enabled=False,
    )


@pytest.fixture
def sample_prepared() -> PreparedTurn:
    """Create a sample prepared turn."""
    messages = (
        Message.system("System prompt"),
        Message.user("User request"),
    )
    budget = BudgetEstimate(
        prompt_tokens=100,
        completion_budget=4000,
        total_budget=100_000,
    )
    return PreparedTurn(
        messages=messages,
        budget=budget,
        system_prompt="System prompt",
        document_context="Test content",
    )


@pytest.fixture
def sample_advice() -> MockAnalysisAdvice:
    """Create sample analysis advice."""
    return MockAnalysisAdvice(
        document_id="test-doc",
        chunk_profile="large_doc",
        required_tools=("get_outline", "read_document"),
        optional_tools=("search_document",),
        must_refresh_outline=True,
    )


# -----------------------------------------------------------------------------
# generate_hints Tests
# -----------------------------------------------------------------------------


class TestGenerateHints:
    """Tests for generate_hints function."""

    def test_none_advice_returns_empty(self, sample_snapshot: DocumentSnapshot) -> None:
        """None advice should return empty tuple."""
        result = generate_hints(None, sample_snapshot)
        assert result == ()

    def test_chunk_profile_hint(self, sample_snapshot: DocumentSnapshot) -> None:
        """Should include chunk profile hint if not auto."""
        advice = MockAnalysisAdvice(chunk_profile="large_doc")
        result = generate_hints(advice, sample_snapshot)
        assert any("large_doc" in hint for hint in result)

    def test_no_chunk_profile_hint_for_auto(self, sample_snapshot: DocumentSnapshot) -> None:
        """Should not include chunk profile hint if auto."""
        advice = MockAnalysisAdvice(chunk_profile="auto")
        result = generate_hints(advice, sample_snapshot)
        assert not any("chunking profile" in hint.lower() for hint in result)

    def test_required_tools_hint(self, sample_snapshot: DocumentSnapshot) -> None:
        """Should include required tools hint."""
        advice = MockAnalysisAdvice(required_tools=("get_outline", "read_document"))
        result = generate_hints(advice, sample_snapshot)
        assert any("get_outline" in hint and "read_document" in hint for hint in result)

    def test_optional_tools_hint(self, sample_snapshot: DocumentSnapshot) -> None:
        """Should include optional tools hint."""
        advice = MockAnalysisAdvice(optional_tools=("search_document",))
        result = generate_hints(advice, sample_snapshot)
        assert any("search_document" in hint for hint in result)

    def test_outline_refresh_hint(self, sample_snapshot: DocumentSnapshot) -> None:
        """Should include outline refresh hint when needed."""
        advice = MockAnalysisAdvice(must_refresh_outline=True)
        result = generate_hints(advice, sample_snapshot)
        assert any("outline" in hint.lower() and "stale" in hint.lower() for hint in result)

    def test_no_outline_refresh_hint(self, sample_snapshot: DocumentSnapshot) -> None:
        """Should not include outline hint when not needed."""
        advice = MockAnalysisAdvice(must_refresh_outline=False)
        result = generate_hints(advice, sample_snapshot)
        assert not any("stale" in hint.lower() for hint in result)

    def test_plot_state_pending_hint(self, sample_snapshot: DocumentSnapshot) -> None:
        """Should include plot state hint for pending updates."""
        advice = MockAnalysisAdvice(plot_state_status="pending_update")
        result = generate_hints(advice, sample_snapshot)
        assert any("pending" in hint.lower() for hint in result)

    def test_plot_state_stale_hint(self, sample_snapshot: DocumentSnapshot) -> None:
        """Should include plot state hint for stale status."""
        advice = MockAnalysisAdvice(plot_state_status="stale")
        result = generate_hints(advice, sample_snapshot)
        assert any("outdated" in hint.lower() or "stale" in hint.lower() for hint in result)

    def test_concordance_stale_hint(self, sample_snapshot: DocumentSnapshot) -> None:
        """Should include concordance hint when stale."""
        advice = MockAnalysisAdvice(concordance_status="stale")
        result = generate_hints(advice, sample_snapshot)
        assert any("concordance" in hint.lower() for hint in result)

    def test_warnings_included(self, sample_snapshot: DocumentSnapshot) -> None:
        """Should include warning messages."""
        advice = MockAnalysisAdvice(
            warnings=(MockWarning(message="Something needs attention"),)
        )
        result = generate_hints(advice, sample_snapshot)
        assert any("attention" in hint.lower() for hint in result)

    def test_multiple_hints(self, sample_snapshot: DocumentSnapshot) -> None:
        """Should generate multiple hints when applicable."""
        advice = MockAnalysisAdvice(
            chunk_profile="large_doc",
            required_tools=("get_outline",),
            must_refresh_outline=True,
            warnings=(MockWarning(message="Warning 1"),),
        )
        result = generate_hints(advice, sample_snapshot)
        assert len(result) >= 3

    def test_returns_tuple(self, sample_snapshot: DocumentSnapshot) -> None:
        """Should return immutable tuple."""
        advice = MockAnalysisAdvice(required_tools=("test",))
        result = generate_hints(advice, sample_snapshot)
        assert isinstance(result, tuple)


# -----------------------------------------------------------------------------
# format_hints_block Tests
# -----------------------------------------------------------------------------


class TestFormatHintsBlock:
    """Tests for format_hints_block function."""

    def test_empty_hints_returns_empty(self) -> None:
        """Empty hints should return empty string."""
        result = format_hints_block([])
        assert result == ""

    def test_formats_single_hint(self) -> None:
        """Should format single hint with header."""
        result = format_hints_block(["Use get_outline first"])
        assert "Analysis hints:" in result
        assert "- Use get_outline first" in result

    def test_formats_multiple_hints(self) -> None:
        """Should format multiple hints as list."""
        result = format_hints_block(["Hint 1", "Hint 2", "Hint 3"])
        assert "- Hint 1" in result
        assert "- Hint 2" in result
        assert "- Hint 3" in result

    def test_handles_tuple_input(self) -> None:
        """Should handle tuple input."""
        result = format_hints_block(("Hint A", "Hint B"))
        assert "- Hint A" in result
        assert "- Hint B" in result


# -----------------------------------------------------------------------------
# analyze_turn Tests
# -----------------------------------------------------------------------------


class TestAnalyzeTurn:
    """Tests for analyze_turn function."""

    def test_returns_analyzed_turn(
        self,
        sample_prepared: PreparedTurn,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """Should return an AnalyzedTurn object."""
        result = analyze_turn(sample_prepared, sample_snapshot, sample_config)
        assert isinstance(result, AnalyzedTurn)

    def test_skips_when_disabled(
        self,
        sample_prepared: PreparedTurn,
        sample_snapshot: DocumentSnapshot,
        config_no_analysis: TurnConfig,
    ) -> None:
        """Should skip analysis when disabled in config."""
        provider = MockAnalysisProvider(advice=MockAnalysisAdvice())
        
        result = analyze_turn(
            sample_prepared,
            sample_snapshot,
            config_no_analysis,
            analysis_provider=provider,
        )
        
        assert result.analysis_ran is False
        assert result.hints == ()
        assert result.advice is None
        assert provider.call_count == 0  # Provider not called

    def test_skips_when_no_provider(
        self,
        sample_prepared: PreparedTurn,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """Should skip analysis when no provider."""
        result = analyze_turn(
            sample_prepared,
            sample_snapshot,
            sample_config,
            analysis_provider=None,
        )
        
        assert result.analysis_ran is False
        assert result.hints == ()

    def test_runs_analysis_with_provider(
        self,
        sample_prepared: PreparedTurn,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
        sample_advice: MockAnalysisAdvice,
    ) -> None:
        """Should run analysis when provider available."""
        provider = MockAnalysisProvider(advice=sample_advice)
        
        result = analyze_turn(
            sample_prepared,
            sample_snapshot,
            sample_config,
            analysis_provider=provider,
        )
        
        assert result.analysis_ran is True
        assert provider.call_count == 1
        assert len(result.hints) > 0

    def test_passes_snapshot_to_provider(
        self,
        sample_prepared: PreparedTurn,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """Should pass snapshot data to provider."""
        provider = MockAnalysisProvider(advice=MockAnalysisAdvice())
        
        analyze_turn(
            sample_prepared,
            sample_snapshot,
            sample_config,
            analysis_provider=provider,
        )
        
        assert provider.last_snapshot is not None
        assert provider.last_snapshot["document_id"] == sample_snapshot.tab_id
        assert provider.last_snapshot["text"] == sample_snapshot.content

    def test_passes_force_refresh(
        self,
        sample_prepared: PreparedTurn,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """Should pass force_refresh to provider."""
        provider = MockAnalysisProvider(advice=MockAnalysisAdvice())
        
        analyze_turn(
            sample_prepared,
            sample_snapshot,
            sample_config,
            analysis_provider=provider,
            force_refresh=True,
        )
        
        assert provider.last_force_refresh is True

    def test_handles_provider_failure(
        self,
        sample_prepared: PreparedTurn,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """Should continue without hints if provider fails."""
        provider = MockAnalysisProvider(should_fail=True)
        
        result = analyze_turn(
            sample_prepared,
            sample_snapshot,
            sample_config,
            analysis_provider=provider,
        )
        
        assert result.analysis_ran is True  # Analysis was attempted
        assert result.hints == ()
        assert result.advice is None

    def test_stores_advice_dict(
        self,
        sample_prepared: PreparedTurn,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
        sample_advice: MockAnalysisAdvice,
    ) -> None:
        """Should store advice as dict."""
        provider = MockAnalysisProvider(advice=sample_advice)
        
        result = analyze_turn(
            sample_prepared,
            sample_snapshot,
            sample_config,
            analysis_provider=provider,
        )
        
        assert result.advice is not None
        assert result.advice["document_id"] == sample_advice.document_id
        assert result.advice["chunk_profile"] == sample_advice.chunk_profile

    def test_preserves_prepared_turn(
        self,
        sample_prepared: PreparedTurn,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """Should preserve the prepared turn in output."""
        result = analyze_turn(sample_prepared, sample_snapshot, sample_config)
        
        assert result.prepared is sample_prepared
        assert result.prepared.messages == sample_prepared.messages

    def test_generates_hints_from_advice(
        self,
        sample_prepared: PreparedTurn,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """Should generate hints from analysis advice."""
        advice = MockAnalysisAdvice(
            chunk_profile="large_doc",
            required_tools=("get_outline",),
            must_refresh_outline=True,
        )
        provider = MockAnalysisProvider(advice=advice)
        
        result = analyze_turn(
            sample_prepared,
            sample_snapshot,
            sample_config,
            analysis_provider=provider,
        )
        
        assert len(result.hints) > 0
        # Should include tool recommendation
        assert any("get_outline" in hint for hint in result.hints)


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------


class TestAnalyzeIntegration:
    """Integration tests for the analyze stage."""

    def test_full_analysis_flow(
        self,
        sample_snapshot: DocumentSnapshot,
    ) -> None:
        """Test complete analysis flow."""
        # Set up prepared turn
        messages = (
            Message.system("You are a helpful assistant."),
            Message.user("Edit the document"),
        )
        budget = BudgetEstimate(
            prompt_tokens=500,
            completion_budget=4000,
            total_budget=128_000,
        )
        prepared = PreparedTurn(
            messages=messages,
            budget=budget,
            system_prompt="You are a helpful assistant.",
            document_context=sample_snapshot.content,
        )
        
        # Set up advice with multiple recommendations
        advice = MockAnalysisAdvice(
            document_id=sample_snapshot.tab_id,
            chunk_profile="large_doc",
            required_tools=("get_outline", "read_document"),
            optional_tools=("search_document",),
            must_refresh_outline=True,
            warnings=(MockWarning(message="Document is large"),),
        )
        provider = MockAnalysisProvider(advice=advice)
        
        # Run analysis
        config = TurnConfig(analysis_enabled=True)
        result = analyze_turn(
            prepared,
            sample_snapshot,
            config,
            analysis_provider=provider,
        )
        
        # Verify structure
        assert result.analysis_ran is True
        assert result.prepared is prepared
        assert len(result.hints) >= 3  # Multiple recommendations
        assert result.advice is not None

    def test_messages_with_hints_integration(
        self,
        sample_snapshot: DocumentSnapshot,
    ) -> None:
        """Test that hints can be injected into messages."""
        messages = (
            Message.system("You are a helpful assistant."),
            Message.user("Test request"),
        )
        budget = BudgetEstimate(
            prompt_tokens=100,
            completion_budget=4000,
            total_budget=128_000,
        )
        prepared = PreparedTurn(
            messages=messages,
            budget=budget,
        )
        
        advice = MockAnalysisAdvice(
            required_tools=("get_outline",),
        )
        provider = MockAnalysisProvider(advice=advice)
        
        config = TurnConfig(analysis_enabled=True)
        result = analyze_turn(
            prepared,
            sample_snapshot,
            config,
            analysis_provider=provider,
        )
        
        # Use messages_with_hints method
        messages_with_hints = result.messages_with_hints()
        
        # Should have hints injected
        assert len(messages_with_hints) >= 2
        # System message should contain hints
        system_msg = messages_with_hints[0]
        assert "get_outline" in system_msg.content

    def test_immutability(
        self,
        sample_prepared: PreparedTurn,
        sample_snapshot: DocumentSnapshot,
        sample_config: TurnConfig,
    ) -> None:
        """Analyzed turn should be immutable."""
        result = analyze_turn(sample_prepared, sample_snapshot, sample_config)
        
        # Should be frozen
        with pytest.raises(AttributeError):
            result.analysis_ran = False  # type: ignore[misc]
        
        # Hints should be tuple
        assert isinstance(result.hints, tuple)
