"""Analyze stage: Run preflight analysis and generate hints.

This module implements the second stage of the turn pipeline, which:
1. Optionally runs preflight analysis on the document
2. Generates hints based on analysis advice
3. Produces an AnalyzedTurn ready for model execution

All functions are stateless and operate on immutable data.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..types import (
    AnalyzedTurn,
    DocumentSnapshot,
    PreparedTurn,
    TurnConfig,
)

LOGGER = logging.getLogger(__name__)


__all__ = [
    "analyze_turn",
    "generate_hints",
    "AnalysisProvider",
    "AnalysisAdvice",
]


# -----------------------------------------------------------------------------
# Analysis Types
# -----------------------------------------------------------------------------


class AnalysisAdvice(Protocol):
    """Protocol for analysis advice objects.
    
    This protocol allows any advice implementation that has these attributes.
    """
    
    @property
    def document_id(self) -> str:
        """Document identifier."""
        ...
    
    @property
    def chunk_profile(self) -> str:
        """Recommended chunking profile."""
        ...
    
    @property
    def required_tools(self) -> Sequence[str]:
        """Tools that must be called."""
        ...
    
    @property
    def optional_tools(self) -> Sequence[str]:
        """Tools that may be helpful."""
        ...
    
    @property
    def must_refresh_outline(self) -> bool:
        """Whether outline needs refreshing."""
        ...
    
    @property
    def plot_state_status(self) -> str | None:
        """Current plot state status."""
        ...
    
    @property
    def concordance_status(self) -> str | None:
        """Current concordance status."""
        ...
    
    @property
    def warnings(self) -> Sequence[Any]:
        """Analysis warnings."""
        ...


class AnalysisProvider(Protocol):
    """Protocol for analysis providers.
    
    Implementations can provide analysis advice for documents.
    This allows injection of different analysis implementations.
    """
    
    def run_analysis(
        self,
        snapshot: Mapping[str, Any],
        *,
        source: str = "pipeline",
        force_refresh: bool = False,
    ) -> AnalysisAdvice | None:
        """Run analysis on a document snapshot.
        
        Args:
            snapshot: Document snapshot as a mapping.
            source: Source identifier for telemetry.
            force_refresh: Force refresh even if cached.
            
        Returns:
            AnalysisAdvice or None if analysis unavailable.
        """
        ...


# -----------------------------------------------------------------------------
# Hint Generation
# -----------------------------------------------------------------------------


def generate_hints(
    advice: AnalysisAdvice | None,
    snapshot: DocumentSnapshot,
    *,
    config: TurnConfig | None = None,
) -> tuple[str, ...]:
    """Generate analysis hints from advice.
    
    Transforms analysis advice into actionable hints that can be
    injected into the system prompt.
    
    Args:
        advice: Analysis advice (or None if no analysis).
        snapshot: Document snapshot for context.
        config: Turn configuration.
        
    Returns:
        Tuple of hint strings.
    """
    if advice is None:
        return ()
    
    hints: list[str] = []
    
    # Chunk profile hint
    if advice.chunk_profile and advice.chunk_profile != "auto":
        hints.append(f"Document chunking profile: {advice.chunk_profile}")
    
    # Required tools hint
    required_tools = list(advice.required_tools) if advice.required_tools else []
    if required_tools:
        tools_str = ", ".join(required_tools)
        hints.append(f"Recommended tools to call: {tools_str}")
    
    # Optional tools hint
    optional_tools = list(advice.optional_tools) if advice.optional_tools else []
    if optional_tools:
        tools_str = ", ".join(optional_tools)
        hints.append(f"Optional tools if needed: {tools_str}")
    
    # Outline refresh hint
    if advice.must_refresh_outline:
        hints.append("The document outline may be stale. Consider calling get_outline to refresh.")
    
    # Plot state hint
    if advice.plot_state_status:
        status = advice.plot_state_status
        if status == "pending_update":
            hints.append("Plot state has pending updates. Verify with get_outline before making structural changes.")
        elif status == "stale":
            hints.append("Plot state may be outdated. Refresh outline data before major edits.")
    
    # Concordance hint
    if advice.concordance_status:
        status = advice.concordance_status
        if status in ("stale", "outdated"):
            hints.append("Document concordance needs updating. Review consistency before edits.")
    
    # Warnings
    warnings = list(advice.warnings) if advice.warnings else []
    for warning in warnings:
        message = getattr(warning, "message", str(warning))
        if message:
            hints.append(f"⚠️ {message}")
    
    return tuple(hints)


def format_hints_block(hints: Sequence[str]) -> str:
    """Format hints into a text block for prompt injection.
    
    Args:
        hints: Sequence of hint strings.
        
    Returns:
        Formatted hint block string.
    """
    if not hints:
        return ""
    
    lines = ["Analysis hints:"]
    for hint in hints:
        lines.append(f"- {hint}")
    
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def analyze_turn(
    prepared: PreparedTurn,
    snapshot: DocumentSnapshot,
    config: TurnConfig,
    *,
    analysis_provider: AnalysisProvider | None = None,
    force_refresh: bool = False,
) -> AnalyzedTurn:
    """Analyze a prepared turn and generate hints.
    
    This is the main entry point for the analyze stage. It optionally
    runs preflight analysis and produces an AnalyzedTurn ready for
    model execution.
    
    Args:
        prepared: The prepared turn from the previous stage.
        snapshot: Document snapshot.
        config: Turn configuration.
        analysis_provider: Optional provider for analysis.
        force_refresh: Force analysis cache refresh.
        
    Returns:
        AnalyzedTurn with analysis hints injected.
    """
    # Skip analysis if disabled in config
    if not config.analysis_enabled:
        return AnalyzedTurn(
            prepared=prepared,
            hints=(),
            advice=None,
            analysis_ran=False,
        )
    
    # Skip analysis if no provider
    if analysis_provider is None:
        LOGGER.debug("No analysis provider available, skipping analysis")
        return AnalyzedTurn(
            prepared=prepared,
            hints=(),
            advice=None,
            analysis_ran=False,
        )
    
    # Run analysis
    advice: AnalysisAdvice | None = None
    try:
        snapshot_dict = _snapshot_to_dict(snapshot)
        advice = analysis_provider.run_analysis(
            snapshot_dict,
            source="pipeline",
            force_refresh=force_refresh,
        )
    except Exception:
        LOGGER.warning("Analysis failed, continuing without hints", exc_info=True)
    
    # Generate hints from advice
    hints = generate_hints(advice, snapshot, config=config)
    
    # Convert advice to dict for storage (if it exists)
    advice_dict: Mapping[str, Any] | None = None
    if advice is not None:
        try:
            to_dict = getattr(advice, "to_dict", None)
            if callable(to_dict):
                advice_dict = to_dict()
            else:
                # Build dict from protocol attributes
                advice_dict = {
                    "document_id": advice.document_id,
                    "chunk_profile": advice.chunk_profile,
                    "required_tools": list(advice.required_tools),
                    "optional_tools": list(advice.optional_tools),
                    "must_refresh_outline": advice.must_refresh_outline,
                    "plot_state_status": advice.plot_state_status,
                    "concordance_status": advice.concordance_status,
                }
        except Exception:
            LOGGER.debug("Failed to serialize advice to dict", exc_info=True)
    
    return AnalyzedTurn(
        prepared=prepared,
        hints=hints,
        advice=advice_dict,
        analysis_ran=True,
    )


# -----------------------------------------------------------------------------
# Private Helpers
# -----------------------------------------------------------------------------


def _snapshot_to_dict(snapshot: DocumentSnapshot) -> dict[str, Any]:
    """Convert DocumentSnapshot to dict for analysis provider.
    
    Args:
        snapshot: Document snapshot.
        
    Returns:
        Dictionary with snapshot data.
    """
    return {
        "document_id": snapshot.tab_id,
        "tab_id": snapshot.tab_id,
        "text": snapshot.content,
        "version": snapshot.version_token,
        "length": len(snapshot.content),
    }
