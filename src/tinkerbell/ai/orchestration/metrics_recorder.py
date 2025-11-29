"""Telemetry and metrics recording for chat turns."""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Mapping, Sequence, TYPE_CHECKING

from ...services import telemetry as telemetry_service
from .controller_utils import (
    coerce_optional_int,
    coerce_optional_float,
    coerce_optional_str,
    normalize_scope_origin,
)
from .scope_helpers import (
    scope_fields_from_summary,
    range_bounds_from_mapping,
)

if TYPE_CHECKING:
    from ..analysis.models import AnalysisAdvice


class MetricsRecorder:
    """Records metrics and telemetry for chat turn execution.
    
    Centralizes turn context creation, tool metrics recording, and
    telemetry emission for the chat orchestration loop.
    """

    def __init__(self, telemetry_enabled: bool = False) -> None:
        """Initialize the metrics recorder.
        
        Args:
            telemetry_enabled: Whether telemetry emission is active.
        """
        self._telemetry_enabled = telemetry_enabled

    @property
    def telemetry_enabled(self) -> bool:
        """Return whether telemetry is enabled."""
        return self._telemetry_enabled

    @telemetry_enabled.setter
    def telemetry_enabled(self, value: bool) -> None:
        """Update the telemetry enabled flag."""
        self._telemetry_enabled = bool(value)

    def new_turn_context(
        self,
        *,
        snapshot: Mapping[str, Any],
        prompt_tokens: int,
        conversation_length: int,
        response_reserve: int | None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        """Create a new turn context for metrics collection.
        
        Initializes a context dictionary to accumulate metrics throughout
        a chat turn's execution.
        
        Args:
            snapshot: Document snapshot for the turn.
            prompt_tokens: Initial prompt token count.
            conversation_length: Number of messages in conversation.
            response_reserve: Token budget reserved for response.
            model_name: Model identifier for the turn.
            
        Returns:
            Turn context dictionary for metrics accumulation.
        """
        document_id = _resolve_document_id(snapshot)
        model = model_name or "unknown"
        context: dict[str, Any] = {
            "document_id": document_id,
            "model": model,
            "prompt_tokens": int(prompt_tokens),
            "tool_tokens": 0,
            "response_reserve": response_reserve,
            "timestamp": time.time(),
            "conversation_length": conversation_length,
            "tool_names": set(),
            "run_id": uuid.uuid4().hex,
        }
        self.copy_snapshot_outline_metrics(context, snapshot)
        self.copy_snapshot_embedding_metadata(context, snapshot)
        return context

    def record_tool_names(
        self,
        context: dict[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        """Record tool names used during a turn.
        
        Args:
            context: Turn metrics context.
            records: Tool execution records.
        """
        names = context.get("tool_names")
        if not isinstance(names, set):
            return
        for record in records:
            name = str(record.get("name") or "").strip()
            if name:
                names.add(name)

    def record_tool_metrics(
        self,
        context: dict[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        """Record detailed tool execution metrics.
        
        Extracts outline and retrieval metrics from tool records.
        
        Args:
            context: Turn metrics context.
            records: Tool execution records.
        """
        if not records:
            return
        for record in records:
            name = str(record.get("name") or "").strip().lower()
            if name == "document_outline":
                self.capture_outline_tool_metrics(context, record)
            elif name == "document_find_text":
                self.capture_retrieval_tool_metrics(context, record)
        self.record_scope_metrics(context, records)

    def record_scope_metrics(
        self,
        context: dict[str, Any],
        records: Sequence[Mapping[str, Any]],
    ) -> None:
        """Record scope origin statistics from tool records.
        
        Counts scope origins (selection, chunk, document) and accumulates
        total scope length across tool calls.
        
        Args:
            context: Turn metrics context.
            records: Tool execution records.
        """
        if not records:
            return
        counts: dict[str, int]
        existing_counts = context.get("scope_origin_counts")
        if isinstance(existing_counts, dict):
            counts = existing_counts
        else:
            counts = {}
            context["scope_origin_counts"] = counts
        missing = context.get("scope_missing_count")
        if not isinstance(missing, int):
            missing = 0
        total_length = context.get("scope_total_length")
        if not isinstance(total_length, int):
            total_length = 0
        for record in records:
            summary = record.get("scope_summary") if isinstance(record.get("scope_summary"), Mapping) else None
            origin = record.get("scope_origin")
            if not isinstance(origin, str) or not origin.strip():
                if isinstance(summary, Mapping):
                    summary_origin = summary.get("origin")
                    if isinstance(summary_origin, str):
                        origin = summary_origin
            normalized_origin = normalize_scope_origin(origin)
            length_value = record.get("scope_length")
            if length_value is None and isinstance(summary, Mapping):
                length_value = summary.get("length")
            length = coerce_optional_int(length_value)
            range_payload = record.get("scope_range") if isinstance(record.get("scope_range"), Mapping) else None
            if range_payload is None and isinstance(summary, Mapping):
                candidate = summary.get("range")
                if isinstance(candidate, Mapping):
                    range_payload = candidate
            if length is None and range_payload is not None:
                bounds = range_bounds_from_mapping(range_payload)
                if bounds is not None:
                    length = max(0, bounds[1] - bounds[0])
            if normalized_origin is None:
                missing += 1
                continue
            counts[normalized_origin] = counts.get(normalized_origin, 0) + 1
            if length is not None:
                total_length += max(0, length)
        context["scope_missing_count"] = missing
        context["scope_total_length"] = total_length

    def capture_outline_tool_metrics(
        self,
        context: dict[str, Any],
        record: Mapping[str, Any],
    ) -> None:
        """Extract outline tool metrics from a tool record.
        
        Args:
            context: Turn metrics context.
            record: Tool execution record with result payload.
        """
        payload = _deserialize_tool_result(record)
        if not isinstance(payload, Mapping):
            return
        digest = payload.get("outline_digest") or payload.get("outline_hash")
        if digest:
            context["outline_digest"] = str(digest)
        version_id = coerce_optional_int(payload.get("version_id"))
        if version_id is not None:
            context["outline_version_id"] = version_id
        status = payload.get("status")
        if status:
            context["outline_status"] = str(status)
        node_count = coerce_optional_int(payload.get("node_count"))
        if node_count is not None:
            context["outline_node_count"] = node_count
        token_count = coerce_optional_int(payload.get("token_count"))
        if token_count is not None:
            context["outline_token_count"] = token_count
        trimmed = payload.get("trimmed")
        if isinstance(trimmed, bool):
            context["outline_trimmed"] = trimmed
        is_stale = payload.get("is_stale")
        if isinstance(is_stale, bool):
            context["outline_is_stale"] = is_stale
            if status is None:
                context["outline_status"] = "stale" if is_stale else "ok"
        latency = coerce_optional_float(record.get("duration_ms"))
        if latency is not None:
            context["outline_latency_ms"] = latency
        generated_at = payload.get("generated_at")
        age = _outline_age_from_timestamp(generated_at)
        if age is not None:
            context["outline_age_seconds"] = age

    def capture_retrieval_tool_metrics(
        self,
        context: dict[str, Any],
        record: Mapping[str, Any],
    ) -> None:
        """Extract retrieval tool metrics from a tool record.
        
        Args:
            context: Turn metrics context.
            record: Tool execution record with result payload.
        """
        payload = _deserialize_tool_result(record)
        if not isinstance(payload, Mapping):
            return
        strategy = payload.get("strategy")
        if strategy:
            context["retrieval_strategy"] = str(strategy)
        status = payload.get("status")
        if status:
            context["retrieval_status"] = str(status)
        pointer_count = payload.get("pointers")
        if isinstance(pointer_count, Sequence) and not isinstance(pointer_count, (str, bytes)):
            context["retrieval_pointer_count"] = len(pointer_count)
        latency = payload.get("latency_ms")
        latency_value = coerce_optional_float(latency)
        if latency_value is None:
            latency_value = coerce_optional_float(record.get("duration_ms"))
        if latency_value is not None:
            context["retrieval_latency_ms"] = latency_value

    def copy_snapshot_outline_metrics(
        self,
        context: dict[str, Any],
        snapshot: Mapping[str, Any],
    ) -> None:
        """Copy outline metrics from snapshot to turn context.
        
        Args:
            context: Turn metrics context.
            snapshot: Document snapshot with outline metadata.
        """
        if not isinstance(snapshot, Mapping):
            return
        digest = snapshot.get("outline_digest")
        if digest:
            context["outline_digest"] = str(digest)
        token_count = coerce_optional_int(snapshot.get("outline_token_count"))
        if token_count is not None:
            context["outline_token_count"] = token_count
        trimmed = snapshot.get("outline_trimmed")
        if isinstance(trimmed, bool):
            context["outline_trimmed"] = trimmed
        is_stale = snapshot.get("outline_is_stale")
        if isinstance(is_stale, bool):
            context["outline_is_stale"] = is_stale
        age_seconds = coerce_optional_float(snapshot.get("outline_age_seconds"))
        if age_seconds is not None:
            context["outline_age_seconds"] = age_seconds
        else:
            completed = snapshot.get("outline_completed_at")
            age_from_completed = _outline_age_from_timestamp(completed)
            if age_from_completed is not None:
                context["outline_age_seconds"] = age_from_completed

    def copy_snapshot_embedding_metadata(
        self,
        context: dict[str, Any],
        snapshot: Mapping[str, Any],
    ) -> None:
        """Copy embedding metadata from snapshot to turn context.
        
        Args:
            context: Turn metrics context.
            snapshot: Document snapshot with embedding metadata.
        """
        if not isinstance(snapshot, Mapping):
            return
        backend = snapshot.get("embedding_backend")
        if backend:
            context["embedding_backend"] = str(backend)
        model = snapshot.get("embedding_model")
        if model:
            context["embedding_model"] = str(model)
        status = snapshot.get("embedding_status")
        if status:
            context["embedding_status"] = str(status)
        detail = snapshot.get("embedding_detail")
        if detail:
            context["embedding_detail"] = str(detail)

    def record_analysis_summary(
        self,
        context: dict[str, Any],
        document_id: str | None,
        advice: "AnalysisAdvice | None",
    ) -> None:
        """Record analysis summary in turn context.
        
        Args:
            context: Turn metrics context.
            document_id: Document identifier.
            advice: Analysis advice result (optional).
        """
        if not document_id or advice is None:
            return
        context["analysis_chunk_profile"] = advice.chunk_profile
        context["analysis_required_tools"] = tuple(advice.required_tools)
        context["analysis_optional_tools"] = tuple(advice.optional_tools)
        context["analysis_must_refresh_outline"] = bool(advice.must_refresh_outline)
        if advice.plot_state_status:
            context["analysis_plot_state_status"] = advice.plot_state_status
        if advice.concordance_status:
            context["analysis_concordance_status"] = advice.concordance_status
        warning_codes = tuple(warning.code for warning in advice.warnings if warning.code)
        if warning_codes:
            context["analysis_warning_codes"] = warning_codes
        if advice.cache_state:
            context["analysis_cache_state"] = advice.cache_state
        context["analysis_generated_at"] = advice.generated_at
        if advice.rule_trace:
            context["analysis_rule_trace"] = tuple(advice.rule_trace)

    def emit_context_usage(self, context: dict[str, Any]) -> None:
        """Emit context usage telemetry event.
        
        Args:
            context: Turn metrics context to emit.
        """
        if not self._telemetry_enabled:
            return
        context.setdefault("timestamp", context.get("timestamp", time.time()))
        context.setdefault("run_id", context.get("run_id") or uuid.uuid4().hex)
        # Actual emission handled by TelemetryManager in controller
        # This method is a coordination point


def _resolve_document_id(snapshot: Mapping[str, Any]) -> str | None:
    """Extract document ID from a snapshot.
    
    Args:
        snapshot: Document snapshot mapping.
        
    Returns:
        Document ID string or None if not found.
    """
    for key in ("document_id", "tab_id", "id"):
        value = snapshot.get(key)
        if value:
            return str(value)
    path = snapshot.get("path")
    if path:
        return str(path)
    version = snapshot.get("version")
    if version:
        return str(version)
    return None


def _deserialize_tool_result(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Deserialize tool result JSON from a record.
    
    Args:
        record: Tool execution record with 'result' field.
        
    Returns:
        Parsed result mapping or None if invalid.
    """
    import json
    result_text = record.get("result")
    if not isinstance(result_text, str) or not result_text.strip():
        return None
    try:
        parsed = json.loads(result_text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, Mapping):
        return parsed
    return None


def _outline_age_from_timestamp(value: object) -> float | None:
    """Calculate outline age from a timestamp value.
    
    Args:
        value: Timestamp as float, int, or ISO string.
        
    Returns:
        Age in seconds or None if invalid.
    """
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, time.time() - float(value))
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return max(0.0, time.time() - parsed.timestamp())
    return None
