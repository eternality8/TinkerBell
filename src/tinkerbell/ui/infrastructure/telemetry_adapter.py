"""Telemetry adapter for forwarding telemetry events through the EventBus.

This module extracts telemetry event forwarding from TelemetryController into
a clean adapter that emits TelemetryEvent through the EventBus.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, TYPE_CHECKING

from ..events import EventBus, TelemetryEvent

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ...ai.analysis import AnalysisAdvice
    from ...ai.orchestration import AIOrchestrator
    from ...services.settings import Settings
    from ..models.window_state import WindowContext

_LOGGER = logging.getLogger(__name__)


class TelemetryAdapter:
    """Adapter for forwarding telemetry events through the EventBus.

    This adapter registers listeners with the telemetry service and forwards
    events as TelemetryEvent instances through the EventBus. It also provides
    helper methods for refreshing context usage and retrieving telemetry
    snapshots.

    The adapter decouples telemetry collection from UI components by using
    the event bus for communication.

    Attributes:
        _event_bus: The event bus for publishing TelemetryEvent instances.
        _subagent_telemetry_registered: Whether subagent listeners are registered.
        _chunk_flow_registered: Whether chunk flow listeners are registered.
        _last_compaction_stats: Most recent compaction statistics.
        _chunk_flow_snapshot: Current chunk flow state snapshot.
        _analysis_snapshot: Current analysis state snapshot.
    """

    __slots__ = (
        "_event_bus",
        "_subagent_telemetry_registered",
        "_chunk_flow_registered",
        "_last_compaction_stats",
        "_chunk_flow_snapshot",
        "_analysis_snapshot",
    )

    def __init__(self, event_bus: EventBus) -> None:
        """Initialize the telemetry adapter.

        Args:
            event_bus: The event bus for publishing telemetry events.
        """
        self._event_bus = event_bus
        self._subagent_telemetry_registered = False
        self._chunk_flow_registered = False
        self._last_compaction_stats: Mapping[str, int] | None = None
        self._chunk_flow_snapshot: dict[str, str] | None = None
        self._analysis_snapshot: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_telemetry_listeners(self) -> None:
        """Register all telemetry service listeners.

        This registers listeners for subagent and chunk flow events,
        forwarding them through the event bus.
        """
        self._register_subagent_listeners()
        self._register_chunk_flow_listeners()

    def set_compaction_stats(self, stats: Mapping[str, int] | None) -> None:
        """Update the cached compaction statistics.

        Args:
            stats: The compaction statistics mapping, or None to clear.
        """
        self._last_compaction_stats = dict(stats) if isinstance(stats, Mapping) else None

    def refresh_context_usage(
        self,
        orchestrator: AIOrchestrator | None,
        settings: Settings | None,
    ) -> dict[str, Any] | None:
        """Refresh and return context usage information.

        This method gathers context usage data from the orchestrator and
        returns it as a dictionary suitable for display.

        Args:
            orchestrator: The AI orchestrator to query for context events.
            settings: The settings containing debug configuration.

        Returns:
            A dictionary with usage information, or None if unavailable.
        """
        if orchestrator is None or settings is None:
            return None

        debug_settings = getattr(settings, "debug", None)
        if not getattr(debug_settings, "token_logging_enabled", False):
            return None

        limit = getattr(debug_settings, "token_log_limit", 1)
        try:
            limit_value = max(1, int(limit))
        except (TypeError, ValueError):
            limit_value = 1

        try:
            from ...services import telemetry as telemetry_service
        except ImportError:  # pragma: no cover - dependency guard
            _LOGGER.debug("Telemetry service not available")
            return None

        events = orchestrator.get_recent_context_events(limit=limit_value)
        dashboard = telemetry_service.build_usage_dashboard(events)
        if dashboard is None:
            return None

        summary_text = dashboard.summary_text

        # Add compaction stats if available
        compaction_stats = self._last_compaction_stats
        if isinstance(compaction_stats, Mapping):
            compactions = int(compaction_stats.get("total_compactions", 0))
            tokens_saved = int(compaction_stats.get("tokens_saved", 0))
            if compactions or tokens_saved:
                stats_bits = f"Compactions {compactions}"
                if tokens_saved:
                    stats_bits = f"{stats_bits} (saved {tokens_saved:,})"
                summary_text = f"{summary_text} · {stats_bits}" if summary_text else stats_bits

        # Add budget status if available
        budget_snapshot = None
        getter = getattr(orchestrator, "get_budget_status", None)
        if callable(getter):
            budget_snapshot = getter()
        if isinstance(budget_snapshot, Mapping):
            budget_text = str(budget_snapshot.get("summary_text") or "").strip()
            if budget_text:
                summary_text = f"{summary_text} · {budget_text}" if summary_text else budget_text

        return {
            "summary_text": summary_text,
            "totals_text": dashboard.totals_text,
            "last_tool": dashboard.summary.last_tool,
        }

    def get_chunk_flow_snapshot(self) -> Mapping[str, str] | None:
        """Return the current chunk flow state snapshot.

        Returns:
            A mapping with 'status' and 'detail' keys, or None if no state.
        """
        return self._chunk_flow_snapshot

    def get_analysis_snapshot(self) -> Mapping[str, str] | None:
        """Return the current analysis state snapshot.

        Returns:
            A mapping with analysis indicator information, or None if no state.
        """
        return self._analysis_snapshot

    def describe_analysis_indicator(
        self,
        advice: AnalysisAdvice,
        *,
        document_label: str | None = None,
    ) -> dict[str, str]:
        """Format analysis advice into display-ready strings.

        Args:
            advice: The analysis advice to format.
            document_label: Optional label for the document.

        Returns:
            A dictionary with 'status', 'badge', and 'detail' keys.
        """
        status_text, badge_text, detail_text = self._format_analysis_indicator(
            advice, document_label
        )
        return {"status": status_text, "badge": badge_text, "detail": detail_text}

    def update_analysis_state(
        self,
        advice: AnalysisAdvice | None,
        *,
        document_label: str | None = None,
    ) -> None:
        """Update the cached analysis state.

        Args:
            advice: The analysis advice, or None to clear.
            document_label: Optional label for the document.
        """
        if advice is None:
            self._analysis_snapshot = None
        else:
            overview = self.describe_analysis_indicator(advice, document_label=document_label)
            self._analysis_snapshot = dict(overview)

    def reset_chunk_flow_state(self) -> None:
        """Reset the chunk flow state to inactive."""
        self._chunk_flow_snapshot = None

    # ------------------------------------------------------------------
    # Listener registration
    # ------------------------------------------------------------------

    def _register_subagent_listeners(self) -> None:
        """Register listeners for subagent telemetry events."""
        if self._subagent_telemetry_registered:
            return

        try:
            from ...services import telemetry as telemetry_service
        except ImportError:  # pragma: no cover - dependency guard
            _LOGGER.debug("Telemetry service not available")
            return

        for event_name in (
            "subagent.job_started",
            "subagent.job_completed",
            "subagent.job_failed",
            "subagent.job_skipped",
            "subagent.jobs_queued",
        ):
            telemetry_service.register_event_listener(
                event_name, self._handle_subagent_telemetry
            )
        self._subagent_telemetry_registered = True

    def _register_chunk_flow_listeners(self) -> None:
        """Register listeners for chunk flow telemetry events."""
        if self._chunk_flow_registered:
            return

        try:
            from ...services import telemetry as telemetry_service
        except ImportError:  # pragma: no cover - dependency guard
            _LOGGER.debug("Telemetry service not available")
            return

        for event_name in (
            "chunk_flow.requested",
            "chunk_flow.escaped_full_snapshot",
            "chunk_flow.retry_success",
        ):
            telemetry_service.register_event_listener(
                event_name, self._handle_chunk_flow_event
            )
        self._chunk_flow_registered = True

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_subagent_telemetry(self, payload: Mapping[str, Any] | None) -> None:
        """Handle subagent telemetry events and forward to event bus.

        Args:
            payload: The telemetry event payload.
        """
        if not isinstance(payload, Mapping):
            return

        event_name = str(payload.get("event") or "")
        if not event_name.startswith("subagent."):
            return

        # Forward to event bus as TelemetryEvent
        self._event_bus.publish(TelemetryEvent(name=event_name, payload=dict(payload)))

    def _handle_chunk_flow_event(self, payload: Mapping[str, Any] | None) -> None:
        """Handle chunk flow telemetry events and update state.

        Args:
            payload: The telemetry event payload.
        """
        if not isinstance(payload, Mapping):
            return

        event_name = str(payload.get("event") or "")

        # Forward to event bus as TelemetryEvent
        self._event_bus.publish(TelemetryEvent(name=event_name, payload=dict(payload)))

        # Update internal state based on event type
        if event_name == "chunk_flow.escaped_full_snapshot":
            detail = self._format_chunk_flow_warning(payload)
            self._chunk_flow_snapshot = {"status": "Chunk Flow Warning", "detail": detail}
        elif event_name == "chunk_flow.retry_success":
            detail = self._format_chunk_flow_recovery(payload)
            self._chunk_flow_snapshot = {"status": "Chunk Flow Recovered", "detail": detail}
        elif event_name == "chunk_flow.requested":
            if self._chunk_flow_snapshot is None or "Warning" not in self._chunk_flow_snapshot.get(
                "status", ""
            ):
                self._chunk_flow_snapshot = None

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_analysis_indicator(
        self,
        advice: AnalysisAdvice,
        document_label: str | None,
    ) -> tuple[str, str, str]:
        """Format analysis advice into status, badge, and detail strings.

        Args:
            advice: The analysis advice to format.
            document_label: Optional label for the document.

        Returns:
            A tuple of (status_text, badge_text, detail_text).
        """
        doc_label = document_label or advice.document_id or "document"
        required = ", ".join(advice.required_tools) if advice.required_tools else "none"
        optional = ", ".join(advice.optional_tools) if advice.optional_tools else "none"
        warning_count = len(advice.warnings)

        summary_bits: list[str] = []
        if advice.must_refresh_outline:
            summary_bits.append("Refresh outline")
        if advice.required_tools:
            summary_bits.append(f"Tools {', '.join(advice.required_tools)}")
        elif advice.optional_tools:
            summary_bits.append(f"Optional {optional}")
        if warning_count:
            summary_bits.append(f"{warning_count} warning{'s' if warning_count != 1 else ''}")
        summary_bits.append(f"Profile {advice.chunk_profile}")

        summary = " · ".join(summary_bits[:3])
        status_text = summary or "Ready"
        badge_text = f"Preflight: {status_text}" if summary else "Preflight: Ready"

        detail_lines = [
            f"Document: {doc_label}",
            f"Chunk profile: {advice.chunk_profile}",
            f"Required tools: {required}",
        ]
        if advice.optional_tools:
            detail_lines.append(f"Optional tools: {optional}")
        detail_lines.append(
            f"Outline refresh required: {'yes' if advice.must_refresh_outline else 'no'}"
        )
        if advice.plot_state_status:
            detail_lines.append(f"Plot state: {advice.plot_state_status}")
        if advice.concordance_status:
            detail_lines.append(f"Concordance: {advice.concordance_status}")
        if warning_count:
            detail_lines.append("Warnings:")
            for warning in advice.warnings:
                detail_lines.append(f"- ({warning.severity}) {warning.code}: {warning.message}")

        detail_text = "\n".join(detail_lines)
        return status_text, badge_text, detail_text

    @staticmethod
    def _format_chunk_flow_warning(payload: Mapping[str, Any]) -> str:
        """Format a chunk flow warning message.

        Args:
            payload: The warning event payload.

        Returns:
            A formatted warning string.
        """
        document_id = str(payload.get("document_id") or "this document")
        reason = str(payload.get("reason") or "full snapshot").replace("_", " ")
        doc_length = _coerce_int(payload.get("document_length"))
        approx = f" (~{doc_length:,} chars)" if doc_length else ""
        window_kind = str(payload.get("window_kind") or "").strip()
        window_hint = f" via {window_kind}" if window_kind else ""
        return f"{document_id}: {reason}{window_hint}{approx}".strip()

    @staticmethod
    def _format_chunk_flow_recovery(payload: Mapping[str, Any]) -> str:
        """Format a chunk flow recovery message.

        Args:
            payload: The recovery event payload.

        Returns:
            A formatted recovery string.
        """
        document_id = str(payload.get("document_id") or "this document")
        recovered_via = str(
            payload.get("recovered_via") or payload.get("source") or "DocumentChunkTool"
        )
        return f"{document_id}: recovered via {recovered_via}"


def _coerce_int(value: Any) -> int | None:
    """Coerce a value to int or return None.

    Args:
        value: The value to coerce.

    Returns:
        The integer value, or None if conversion fails.
    """
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = ["TelemetryAdapter"]
