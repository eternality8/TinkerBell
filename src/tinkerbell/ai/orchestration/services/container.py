"""Services container for the orchestration pipeline.

This module defines the Services container that holds all service instances
needed during turn execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis_cache import AnalysisCache
    from .budget import BudgetService
    from .document_cache import DocumentCache
    from .telemetry import TelemetryService

__all__ = [
    "Services",
    "create_services",
]


@dataclass(slots=True)
class Services:
    """Container holding all orchestration services.

    This dataclass groups together all service instances that the turn
    pipeline needs during execution. It provides a single point of access
    to caching, budget management, and telemetry services.

    All services are optional - the pipeline will work without them,
    but with reduced functionality (no caching, no budget enforcement,
    no telemetry recording).

    Attributes:
        document_cache: Cache for document snapshots.
        analysis_cache: Cache for analysis results.
        budget: Budget management service.
        telemetry: Telemetry recording service.

    Example:
        >>> from orchestration.services import Services, create_services
        >>> services = create_services()  # Creates with defaults
        >>> # Or create with custom configuration
        >>> services = Services(
        ...     document_cache=DocumentCache(CacheConfig(max_entries=50)),
        ...     telemetry=TelemetryService(TelemetryConfig(enabled=True)),
        ... )
    """

    document_cache: "DocumentCache | None" = None
    analysis_cache: "AnalysisCache | None" = None
    budget: "BudgetService | None" = None
    telemetry: "TelemetryService | None" = None

    @property
    def has_document_cache(self) -> bool:
        """Whether document caching is available."""
        return self.document_cache is not None

    @property
    def has_analysis_cache(self) -> bool:
        """Whether analysis caching is available."""
        return self.analysis_cache is not None

    @property
    def has_budget(self) -> bool:
        """Whether budget management is available."""
        return self.budget is not None

    @property
    def has_telemetry(self) -> bool:
        """Whether telemetry recording is available."""
        return self.telemetry is not None

    def summary(self) -> dict[str, bool]:
        """Get a summary of which services are available.

        Returns:
            Dictionary mapping service names to availability.
        """
        return {
            "document_cache": self.has_document_cache,
            "analysis_cache": self.has_analysis_cache,
            "budget": self.has_budget,
            "telemetry": self.has_telemetry,
        }


def create_services(
    *,
    enable_document_cache: bool = True,
    enable_analysis_cache: bool = True,
    enable_budget: bool = True,
    enable_telemetry: bool = True,
    document_cache: "DocumentCache | None" = None,
    analysis_cache: "AnalysisCache | None" = None,
    budget: "BudgetService | None" = None,
    telemetry: "TelemetryService | None" = None,
) -> Services:
    """Create a Services container with default instances.

    This factory function creates service instances with default
    configurations unless custom instances are provided.

    Args:
        enable_document_cache: Whether to create document cache.
        enable_analysis_cache: Whether to create analysis cache.
        enable_budget: Whether to create budget service.
        enable_telemetry: Whether to create telemetry service.
        document_cache: Custom document cache instance.
        analysis_cache: Custom analysis cache instance.
        budget: Custom budget service instance.
        telemetry: Custom telemetry service instance.

    Returns:
        Configured Services container.

    Example:
        >>> services = create_services()  # All services with defaults
        >>> services = create_services(enable_budget=False)  # No budget
        >>> services = create_services(telemetry=my_telemetry)  # Custom telemetry
    """
    # Import here to avoid circular imports
    from .analysis_cache import AnalysisCache
    from .budget import BudgetService
    from .document_cache import DocumentCache
    from .telemetry import TelemetryService

    return Services(
        document_cache=(
            document_cache
            if document_cache is not None
            else (DocumentCache() if enable_document_cache else None)
        ),
        analysis_cache=(
            analysis_cache
            if analysis_cache is not None
            else (AnalysisCache() if enable_analysis_cache else None)
        ),
        budget=(
            budget
            if budget is not None
            else (BudgetService() if enable_budget else None)
        ),
        telemetry=(
            telemetry
            if telemetry is not None
            else (TelemetryService() if enable_telemetry else None)
        ),
    )
