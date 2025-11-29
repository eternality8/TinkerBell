"""Preflight analysis coordination for chat turns."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable, Mapping, TYPE_CHECKING

from ...services import telemetry as telemetry_service
from ..analysis.agent import AnalysisAgent
from ..analysis.models import AnalysisAdvice, AnalysisInput
from .runtime_config import AnalysisRuntimeConfig, ChunkingRuntimeConfig
from .controller_utils import (
    coerce_optional_int,
    coerce_optional_float,
    coerce_optional_str,
)

if TYPE_CHECKING:
    from .chunk_flow import ChunkFlowTracker
    from .turn_tracking import PlotLoopTracker

LOGGER = logging.getLogger(__name__)


class AnalysisCoordinator:
    """Coordinates preflight analysis for chat turns.
    
    Manages the AnalysisAgent lifecycle and provides:
    - Preflight analysis execution and caching
    - Analysis hint message generation
    - Document snapshot caching
    - Cache invalidation on document changes
    """

    def __init__(
        self,
        config: AnalysisRuntimeConfig,
        chunking_config: ChunkingRuntimeConfig | None = None,
        document_id_resolver: Callable[[Mapping[str, Any]], str | None] | None = None,
        snapshot_span_resolver: Callable[[Mapping[str, Any]], tuple[int, int] | None] | None = None,
        chunk_flow_tracker: Callable[[], "ChunkFlowTracker | None"] | None = None,
        plot_loop_tracker: Callable[[], "PlotLoopTracker | None"] | None = None,
        chunk_index_ready: Callable[[], bool] | None = None,
    ) -> None:
        """Initialize the analysis coordinator.
        
        Args:
            config: Analysis runtime configuration.
            chunking_config: Chunking runtime configuration.
            document_id_resolver: Callable to resolve document ID from snapshot.
            snapshot_span_resolver: Callable to resolve snapshot span bounds.
            chunk_flow_tracker: Callable to get current chunk flow tracker.
            plot_loop_tracker: Callable to get current plot loop tracker.
            chunk_index_ready: Callable to check if chunk index is available.
        """
        self._config = config
        self._chunking_config = chunking_config or ChunkingRuntimeConfig()
        self._resolve_document_id = document_id_resolver or (lambda s: s.get("document_id"))
        self._resolve_snapshot_span = snapshot_span_resolver
        self._get_chunk_flow_tracker = chunk_flow_tracker
        self._get_plot_loop_tracker = plot_loop_tracker
        self._is_chunk_index_ready = chunk_index_ready or (lambda: False)
        self._agent: AnalysisAgent | None = None
        self._advice_cache: dict[str, AnalysisAdvice] = {}
        self._snapshot_cache: dict[str, Mapping[str, Any]] = {}

    @property
    def config(self) -> AnalysisRuntimeConfig:
        """Return the current analysis configuration."""
        return self._config

    def update_config(self, config: AnalysisRuntimeConfig) -> None:
        """Update the analysis configuration.
        
        Args:
            config: New configuration to apply.
        """
        self._config = config

    @property
    def enabled(self) -> bool:
        """Return whether analysis is enabled."""
        return bool(getattr(self._config, "enabled", False))

    def run_analysis(
        self,
        snapshot: Mapping[str, Any],
        *,
        source: str = "controller",
        force_refresh: bool = False,
    ) -> AnalysisAdvice | None:
        """Run preflight analysis on a document snapshot.
        
        Args:
            snapshot: Document snapshot.
            source: Source identifier for telemetry.
            force_refresh: Force refresh even if cached.
            
        Returns:
            AnalysisAdvice or None if analysis disabled/failed.
        """
        if not self.enabled:
            return None
        document_id = self._resolve_document_id(snapshot) if self._resolve_document_id else None
        if not document_id:
            return None
        agent = self._ensure_agent()
        analysis_input = self.build_analysis_input(snapshot)
        advice = agent.analyze(analysis_input, force_refresh=force_refresh, source=source)
        self._advice_cache[document_id] = advice
        return advice

    def _ensure_agent(self) -> AnalysisAgent:
        """Ensure analysis agent is initialized.
        
        Returns:
            AnalysisAgent instance.
        """
        agent = self._agent
        if agent is not None:
            return agent
        telemetry_emitter = getattr(telemetry_service, "emit", None)
        if not callable(telemetry_emitter):
            telemetry_emitter = None
        agent = AnalysisAgent(
            ttl_seconds=self._config.ttl_seconds,
            telemetry_emitter=telemetry_emitter,
        )
        self._agent = agent
        return agent

    def build_analysis_input(self, snapshot: Mapping[str, Any]) -> AnalysisInput:
        """Build analysis input from document snapshot.
        
        Args:
            snapshot: Document snapshot.
            
        Returns:
            AnalysisInput for the analysis agent.
        """
        span_start, span_end = self._snapshot_span_bounds(snapshot)
        manifest = snapshot.get("chunk_manifest") if isinstance(snapshot.get("chunk_manifest"), Mapping) else None
        chunk_profile = manifest.get("chunk_profile") if isinstance(manifest, Mapping) else None
        chunk_cache_key = manifest.get("cache_key") if isinstance(manifest, Mapping) else None
        
        outline_age = coerce_optional_float(snapshot.get("outline_age_seconds"))
        if outline_age is None:
            completed = snapshot.get("outline_completed_at")
            outline_age = self._outline_age_from_timestamp(completed)
            
        document_chars = coerce_optional_int(snapshot.get("length"))
        if document_chars is None:
            text = snapshot.get("text")
            if isinstance(text, str):
                document_chars = len(text)
                
        chunk_flow_flags: tuple[str, ...] = ()
        if self._get_chunk_flow_tracker is not None:
            tracker = self._get_chunk_flow_tracker()
            if tracker and tracker.warning_active:
                chunk_flow_flags = (tracker.last_reason or "chunk_flow_warning",)
                
        plot_loop_flags: tuple[str, ...] = ()
        plot_state_status = coerce_optional_str(snapshot.get("plot_state_status"))
        if self._get_plot_loop_tracker is not None:
            loop_tracker = self._get_plot_loop_tracker()
            if loop_tracker is not None:
                if loop_tracker.pending_update:
                    plot_state_status = "pending_update"
                    plot_loop_flags = ("pending_update",)
                elif loop_tracker.outline_called and not plot_state_status:
                    plot_state_status = "ok"
                    
        concordance_status = coerce_optional_str(snapshot.get("concordance_status"))
        concordance_age = coerce_optional_float(snapshot.get("concordance_age_seconds"))
        
        extras: dict[str, object] = {}
        if chunk_flow_flags:
            extras["chunk_flow"] = chunk_flow_flags
        if plot_loop_flags:
            extras["plot_loop"] = plot_loop_flags
        if manifest and manifest.get("generated_at") is not None:
            extras["chunk_manifest_generated_at"] = manifest.get("generated_at")
            
        document_id = (
            self._resolve_document_id(snapshot) if self._resolve_document_id else None
        ) or "document"
        version = snapshot.get("version") or snapshot.get("version_id") or snapshot.get("document_version")
        
        return AnalysisInput(
            document_id=document_id,
            document_version=str(version) if version else None,
            document_path=coerce_optional_str(snapshot.get("path")),
            span_start=span_start,
            span_end=span_end,
            document_chars=document_chars,
            chunk_profile_hint=self._chunking_config.default_profile,
            chunk_index_ready=self._is_chunk_index_ready(),
            chunk_manifest_profile=str(chunk_profile) if chunk_profile else None,
            chunk_manifest_cache_key=str(chunk_cache_key) if chunk_cache_key else None,
            outline_digest=coerce_optional_str(snapshot.get("outline_digest")),
            outline_age_seconds=outline_age,
            outline_version_id=coerce_optional_int(snapshot.get("outline_version_id")),
            plot_state_status=plot_state_status,
            plot_override_version=coerce_optional_int(snapshot.get("plot_override_version")),
            concordance_status=concordance_status,
            concordance_age_seconds=concordance_age,
            retrieval_enabled=True,
            extra_metadata=extras or None,
            chunk_flow_warnings=chunk_flow_flags or None,
            plot_loop_flags=plot_loop_flags or None,
        )

    def _snapshot_span_bounds(self, snapshot: Mapping[str, Any]) -> tuple[int, int]:
        """Get span bounds from snapshot.
        
        Args:
            snapshot: Document snapshot.
            
        Returns:
            Tuple of (start, end) indices.
        """
        if self._resolve_snapshot_span is not None:
            span = self._resolve_snapshot_span(snapshot)
            if span is not None:
                return span
                
        start = 0
        end = 0
        text_range = snapshot.get("text_range")
        if isinstance(text_range, Mapping):
            start = self._coerce_index(text_range.get("start"), 0)
            end = self._coerce_index(text_range.get("end"), start)
        if (start, end) == (0, 0):
            window = snapshot.get("window")
            if isinstance(window, Mapping):
                start = self._coerce_index(window.get("start"), 0)
                end = self._coerce_index(window.get("end"), start)
        document_length = coerce_optional_int(snapshot.get("length"))
        if document_length is not None:
            start = max(0, min(start, document_length))
            end = max(start, min(end, document_length))
        return (start, end)

    def format_hint(self, advice: AnalysisAdvice) -> str:
        """Format analysis advice as hint text.
        
        Args:
            advice: Analysis advice to format.
            
        Returns:
            Formatted hint string.
        """
        lines = [
            f"- Chunk profile: {advice.chunk_profile}",
            f"- Required tools: {', '.join(advice.required_tools) if advice.required_tools else 'none'}",
        ]
        if advice.optional_tools:
            lines.append(f"- Optional tools: {', '.join(advice.optional_tools)}")
        lines.append(f"- Outline refresh required: {'yes' if advice.must_refresh_outline else 'no'}")
        if advice.plot_state_status:
            lines.append(f"- Plot state status: {advice.plot_state_status}")
        if advice.concordance_status:
            lines.append(f"- Concordance status: {advice.concordance_status}")
        if advice.warnings:
            warning_lines = "\n".join(f"  - {warning.message}" for warning in advice.warnings)
            lines.append(f"- Warnings:\n{warning_lines}")
        return "Preflight analysis summary:\n" + "\n".join(lines)

    def hint_message(self, snapshot: Mapping[str, Any]) -> dict[str, str] | None:
        """Generate analysis hint message for a snapshot.
        
        Args:
            snapshot: Document snapshot.
            
        Returns:
            System message dict or None if no hint.
        """
        advice = self.run_analysis(snapshot, source="controller")
        if advice is None:
            return None
        hint_text = self.format_hint(advice)
        if not hint_text:
            return None
        return {"role": "system", "content": hint_text}

    def get_latest_advice(self, document_id: str | None) -> AnalysisAdvice | None:
        """Get cached analysis advice for a document.
        
        Args:
            document_id: Document identifier.
            
        Returns:
            Cached AnalysisAdvice or None.
        """
        if not document_id:
            return None
        return self._advice_cache.get(document_id)

    def get_latest_snapshot(self, document_id: str | None) -> Mapping[str, Any] | None:
        """Get cached snapshot for a document.
        
        Args:
            document_id: Document identifier.
            
        Returns:
            Cached snapshot or None.
        """
        if not document_id:
            return None
        cached = self._snapshot_cache.get(document_id)
        return dict(cached) if cached else None

    def remember_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        """Cache a document snapshot.
        
        Args:
            snapshot: Snapshot to cache.
        """
        document_id = self._resolve_document_id(snapshot) if self._resolve_document_id else None
        if not document_id:
            return
        self._snapshot_cache[document_id] = dict(snapshot)

    def invalidate_document(self, document_id: str) -> None:
        """Invalidate caches for a document.
        
        Args:
            document_id: Document identifier.
        """
        self._advice_cache.pop(document_id, None)
        self._snapshot_cache.pop(document_id, None)
        if self._agent:
            try:
                self._agent.invalidate_document(document_id)
            except Exception:
                LOGGER.debug("Failed to invalidate analysis cache for %s", document_id, exc_info=True)

    def request_analysis(
        self,
        *,
        document_id: str | None = None,
        snapshot: Mapping[str, Any] | None = None,
        span_start: int | None = None,
        span_end: int | None = None,
        force_refresh: bool = False,
        reason: str | None = None,
        source: str = "tool",
    ) -> AnalysisAdvice | None:
        """Request analysis via public entry point.
        
        This is the main entry point for UI code and tools to run analysis.
        
        Args:
            document_id: Target document ID.
            snapshot: Document snapshot (or use cached if None).
            span_start: Optional span start override.
            span_end: Optional span end override.
            force_refresh: Force cache refresh.
            reason: Reason for request (telemetry).
            source: Invocation source identifier.
            
        Returns:
            AnalysisAdvice or None.
            
        Raises:
            ValueError: If neither snapshot nor document_id provided.
        """
        if not self.enabled:
            return None
            
        candidate_snapshot: Mapping[str, Any] | None = snapshot
        target_id = document_id
        snapshot_origin = "provided"
        
        if candidate_snapshot is None and target_id:
            cached = self._snapshot_cache.get(target_id)
            if cached is not None:
                candidate_snapshot = dict(cached)
                snapshot_origin = "cache"
            else:
                snapshot_origin = "missing"
                
        if candidate_snapshot is None:
            raise ValueError("snapshot or document_id is required for analysis")
            
        snapshot_payload = dict(candidate_snapshot)
        resolved_id = target_id or (
            self._resolve_document_id(snapshot_payload) if self._resolve_document_id else None
        )
        if resolved_id:
            snapshot_payload.setdefault("document_id", resolved_id)
            
        # Apply span overrides
        if span_start is None or span_end is None:
            fallback_start, fallback_end = self._snapshot_span_bounds(snapshot_payload)
            if span_start is None:
                span_start = fallback_start
            if span_end is None:
                span_end = fallback_end
                
        # Emit telemetry for tool invocations
        if source == "tool":
            self._emit_invocation_event(
                event_name="analysis.advisor_tool.invoked",
                document_id=resolved_id,
                span_start=span_start,
                span_end=span_end,
                force_refresh=force_refresh,
                reason=reason,
                snapshot_origin=snapshot_origin,
            )
            
        return self.run_analysis(snapshot_payload, source=source, force_refresh=force_refresh)

    def _emit_invocation_event(
        self,
        *,
        event_name: str,
        document_id: str | None,
        span_start: int | None,
        span_end: int | None,
        force_refresh: bool,
        reason: str | None,
        snapshot_origin: str,
    ) -> None:
        """Emit telemetry for analysis invocation.
        
        Args:
            event_name: Telemetry event name.
            document_id: Document identifier.
            span_start: Span start index.
            span_end: Span end index.
            force_refresh: Whether refresh was forced.
            reason: Invocation reason.
            snapshot_origin: Source of snapshot.
        """
        emitter = getattr(telemetry_service, "emit", None)
        if not callable(emitter):
            return
        payload: dict[str, object] = {
            "document_id": document_id,
            "span_start": span_start if span_start is not None else None,
            "span_end": span_end if span_end is not None else None,
            "force_refresh": bool(force_refresh),
            "snapshot_origin": snapshot_origin,
            "source": "tool",
        }
        if reason:
            payload["reason"] = reason
        payload["has_span_override"] = span_start is not None and span_end is not None
        emitter(event_name, payload)

    @staticmethod
    def _outline_age_from_timestamp(completed: Any) -> float | None:
        """Calculate outline age from completion timestamp.
        
        Args:
            completed: Completion timestamp.
            
        Returns:
            Age in seconds or None.
        """
        if completed is None:
            return None
        if isinstance(completed, (int, float)):
            return max(0.0, time.time() - float(completed))
        if isinstance(completed, str):
            try:
                ts = datetime.fromisoformat(completed.replace("Z", "+00:00"))
                return max(0.0, (datetime.now(ts.tzinfo) - ts).total_seconds())
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _coerce_index(value: Any, default: int) -> int:
        """Coerce value to int index.
        
        Args:
            value: Value to coerce.
            default: Default if coercion fails.
            
        Returns:
            Integer index.
        """
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
