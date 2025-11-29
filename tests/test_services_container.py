"""Tests for the services container.

This module tests the Services container dataclass and factory function.
"""

from __future__ import annotations

import pytest

from tinkerbell.ai.orchestration.services import (
    Services,
    create_services,
    DocumentCache,
    AnalysisCache,
    BudgetService,
    TelemetryService,
    CacheConfig,
    AnalysisCacheConfig,
    BudgetConfig,
    TelemetryConfig,
)


# =============================================================================
# Services Container Tests
# =============================================================================


class TestServices:
    """Tests for the Services container."""

    def test_default_all_none(self):
        """Should initialize with all services as None."""
        services = Services()
        
        assert services.document_cache is None
        assert services.analysis_cache is None
        assert services.budget is None
        assert services.telemetry is None

    def test_with_all_services(self):
        """Should accept all service instances."""
        doc_cache = DocumentCache()
        analysis_cache = AnalysisCache()
        budget = BudgetService()
        telemetry = TelemetryService()
        
        services = Services(
            document_cache=doc_cache,
            analysis_cache=analysis_cache,
            budget=budget,
            telemetry=telemetry,
        )
        
        assert services.document_cache is doc_cache
        assert services.analysis_cache is analysis_cache
        assert services.budget is budget
        assert services.telemetry is telemetry

    def test_has_document_cache(self):
        """has_document_cache should reflect availability."""
        services_with = Services(document_cache=DocumentCache())
        services_without = Services()
        
        assert services_with.has_document_cache is True
        assert services_without.has_document_cache is False

    def test_has_analysis_cache(self):
        """has_analysis_cache should reflect availability."""
        services_with = Services(analysis_cache=AnalysisCache())
        services_without = Services()
        
        assert services_with.has_analysis_cache is True
        assert services_without.has_analysis_cache is False

    def test_has_budget(self):
        """has_budget should reflect availability."""
        services_with = Services(budget=BudgetService())
        services_without = Services()
        
        assert services_with.has_budget is True
        assert services_without.has_budget is False

    def test_has_telemetry(self):
        """has_telemetry should reflect availability."""
        services_with = Services(telemetry=TelemetryService())
        services_without = Services()
        
        assert services_with.has_telemetry is True
        assert services_without.has_telemetry is False

    def test_summary(self):
        """summary() should return service availability."""
        services = Services(
            document_cache=DocumentCache(),
            budget=BudgetService(),
        )
        
        summary = services.summary()
        
        assert summary == {
            "document_cache": True,
            "analysis_cache": False,
            "budget": True,
            "telemetry": False,
        }

    def test_summary_all_enabled(self):
        """summary() with all services."""
        services = Services(
            document_cache=DocumentCache(),
            analysis_cache=AnalysisCache(),
            budget=BudgetService(),
            telemetry=TelemetryService(),
        )
        
        summary = services.summary()
        
        assert all(summary.values())

    def test_summary_none_enabled(self):
        """summary() with no services."""
        services = Services()
        
        summary = services.summary()
        
        assert not any(summary.values())


# =============================================================================
# create_services Factory Tests
# =============================================================================


class TestCreateServices:
    """Tests for the create_services factory function."""

    def test_creates_all_by_default(self):
        """Should create all services by default."""
        services = create_services()
        
        assert services.document_cache is not None
        assert services.analysis_cache is not None
        assert services.budget is not None
        assert services.telemetry is not None

    def test_disable_document_cache(self):
        """Should skip document cache when disabled."""
        services = create_services(enable_document_cache=False)
        
        assert services.document_cache is None
        assert services.analysis_cache is not None

    def test_disable_analysis_cache(self):
        """Should skip analysis cache when disabled."""
        services = create_services(enable_analysis_cache=False)
        
        assert services.document_cache is not None
        assert services.analysis_cache is None

    def test_disable_budget(self):
        """Should skip budget when disabled."""
        services = create_services(enable_budget=False)
        
        assert services.budget is None
        assert services.telemetry is not None

    def test_disable_telemetry(self):
        """Should skip telemetry when disabled."""
        services = create_services(enable_telemetry=False)
        
        assert services.budget is not None
        assert services.telemetry is None

    def test_disable_all(self):
        """Should create empty container when all disabled."""
        services = create_services(
            enable_document_cache=False,
            enable_analysis_cache=False,
            enable_budget=False,
            enable_telemetry=False,
        )
        
        assert services.document_cache is None
        assert services.analysis_cache is None
        assert services.budget is None
        assert services.telemetry is None

    def test_custom_document_cache(self):
        """Should use provided document cache."""
        custom_cache = DocumentCache(CacheConfig(max_entries=50))
        
        services = create_services(document_cache=custom_cache)
        
        assert services.document_cache is custom_cache

    def test_custom_analysis_cache(self):
        """Should use provided analysis cache."""
        custom_cache = AnalysisCache(AnalysisCacheConfig(max_entries=25))
        
        services = create_services(analysis_cache=custom_cache)
        
        assert services.analysis_cache is custom_cache

    def test_custom_budget(self):
        """Should use provided budget service."""
        custom_budget = BudgetService(BudgetConfig(prompt_budget=50_000))
        
        services = create_services(budget=custom_budget)
        
        assert services.budget is custom_budget

    def test_custom_telemetry(self):
        """Should use provided telemetry service."""
        custom_telemetry = TelemetryService(TelemetryConfig(sink_capacity=100))
        
        services = create_services(telemetry=custom_telemetry)
        
        assert services.telemetry is custom_telemetry

    def test_custom_overrides_enable_flag(self):
        """Custom instance should be used even if enable flag is False."""
        custom_cache = DocumentCache()
        
        services = create_services(
            enable_document_cache=False,
            document_cache=custom_cache,
        )
        
        # Custom instance wins over enable flag
        assert services.document_cache is custom_cache
