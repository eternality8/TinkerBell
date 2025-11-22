"""Window context and outline metadata models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ...ai.orchestration import AIController
    from ...services.settings import Settings, SettingsStore
    from ...services.unsaved_cache import UnsavedCache, UnsavedCacheStore


@dataclass(slots=True)
class WindowContext:
    """Shared context passed to the main window when constructing the UI."""

    settings: Optional[Settings] = None
    ai_controller: Optional[AIController] = None
    settings_store: Optional[SettingsStore] = None
    unsaved_cache: Optional[UnsavedCache] = None
    unsaved_cache_store: Optional[UnsavedCacheStore] = None


@dataclass(slots=True)
class OutlineStatusInfo:
    """Tracks outline freshness/latency metadata per document."""

    status_label: str = ""
    status_code: str = ""
    tooltip: str = ""
    version_id: int | None = None
    outline_hash: str | None = None
    latency_ms: float | None = None
    completed_at: float | None = None
    node_count: int | None = None
    token_count: int | None = None
    stale: bool = False
    stale_since: float | None = None


__all__ = ["WindowContext", "OutlineStatusInfo"]
