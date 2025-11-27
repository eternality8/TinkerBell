from __future__ import annotations

import pytest

from tinkerbell.ai.analysis.agent import AnalysisAgent
from tinkerbell.ai.analysis.models import AnalysisInput


def _make_input(**overrides: object) -> AnalysisInput:
    base = dict(
        document_id="doc-123",
        document_version="v1",
        span_start=0,
        span_end=0,
        document_chars=70_000,
        outline_age_seconds=1200.0,
        plot_state_status="stale",
    )
    base.update(overrides)
    return AnalysisInput(**base)


def test_large_document_prefers_prose_chunk_profile() -> None:
    agent = AnalysisAgent()
    advice = agent.analyze(_make_input())
    assert advice.chunk_profile == "prose"
    assert "chunk_profile:prose" in "".join(advice.rule_trace)
    # analyze_document replaces document_chunk
    assert "analyze_document" in advice.required_tools


def test_outline_staleness_sets_flag_and_warning() -> None:
    agent = AnalysisAgent()
    advice = agent.analyze(_make_input(outline_age_seconds=999.0))
    assert advice.must_refresh_outline is True
    assert advice.warnings
    warning_codes = {warning.code for warning in advice.warnings}
    assert "outline.stale" in warning_codes or "outline.missing" in warning_codes


def test_outline_digest_avoids_missing_warning() -> None:
    agent = AnalysisAgent()
    advice = agent.analyze(_make_input(outline_age_seconds=None, outline_digest="abc123"))
    assert advice.must_refresh_outline is False
    warning_codes = {warning.code for warning in advice.warnings}
    assert "outline.missing" not in warning_codes


def test_cache_hit_short_circuits_analysis() -> None:
    agent = AnalysisAgent()
    analysis_input = _make_input(document_chars=10_000, span_end=500)
    first = agent.analyze(analysis_input)
    second = agent.analyze(analysis_input)
    assert first.cache_state == "miss"
    assert second.cache_state == "hit"
    assert first.chunk_profile == second.chunk_profile


def test_force_refresh_bypasses_cache() -> None:
    agent = AnalysisAgent()
    analysis_input = _make_input(document_chars=5_000)
    agent.analyze(analysis_input)
    refreshed = agent.analyze(analysis_input, force_refresh=True)
    assert refreshed.cache_state == "miss"


def test_plot_state_rule_recommends_update_tool() -> None:
    agent = AnalysisAgent()
    advice = agent.analyze(_make_input(plot_state_status="stale"))
    # transform_document replaces plot_state_update
    assert "transform_document" in advice.required_tools


def test_small_document_skips_retrieval_rule() -> None:
    agent = AnalysisAgent()
    advice = agent.analyze(_make_input(document_chars=1_000, span_end=1000))
    # read_document replaces document_snapshot
    assert "read_document" not in advice.required_tools
