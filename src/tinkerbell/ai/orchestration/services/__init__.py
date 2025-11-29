"""Services for the orchestration pipeline.

This package contains service classes that provide caching, telemetry,
and other cross-cutting concerns for the turn pipeline.

Services:
    - DocumentCache: LRU cache for document snapshots
    - AnalysisCache: LRU cache for analysis results
    - BudgetService: Context window budget management
    - TelemetryService: Turn and tool call telemetry recording
    - Services: Container holding all service instances
"""

from .analysis_cache import (
    AnalysisCache,
    AnalysisCacheConfig,
    AnalysisCacheEntry,
    AnalysisCacheStats,
    compute_snapshot_hash,
)
from .budget import (
    BudgetConfig,
    BudgetEvaluation,
    BudgetExceededError,
    BudgetService,
)
from .container import (
    Services,
    create_services,
)
from .document_cache import (
    CacheConfig,
    CacheEntry,
    CacheStats,
    DocumentCache,
)
from .telemetry import (
    InMemoryTelemetrySink,
    TelemetryConfig,
    TelemetryEvent,
    TelemetryService,
    TelemetrySink,
)

__all__ = [
    # Services Container
    "Services",
    "create_services",
    # Document Cache
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "DocumentCache",
    # Analysis Cache
    "AnalysisCache",
    "AnalysisCacheConfig",
    "AnalysisCacheEntry",
    "AnalysisCacheStats",
    "compute_snapshot_hash",
    # Budget Service
    "BudgetConfig",
    "BudgetEvaluation",
    "BudgetExceededError",
    "BudgetService",
    # Telemetry Service
    "InMemoryTelemetrySink",
    "TelemetryConfig",
    "TelemetryEvent",
    "TelemetryService",
    "TelemetrySink",
]
