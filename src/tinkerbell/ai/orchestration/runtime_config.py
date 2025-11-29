"""Runtime configuration classes for the AI controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ChunkingRuntimeConfig:
    """Settings governing chunk manifest preferences and tool caps."""

    default_profile: str = "auto"
    overlap_chars: int = 256
    max_inline_tokens: int = 1_800
    iterator_limit: int = 4


@dataclass(slots=True)
class AnalysisRuntimeConfig:
    """Toggles for the preflight analysis agent."""

    enabled: bool = True
    ttl_seconds: float = 120.0


__all__ = [
    "ChunkingRuntimeConfig",
    "AnalysisRuntimeConfig",
]
