"""Telemetry helpers extracted from the AI controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..services.telemetry import ContextUsageEvent, PersistentTelemetrySink, TelemetrySink, default_telemetry_path

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TelemetryManager:
    """Owns telemetry sink lifecycle and event emission."""

    enabled: bool = False
    limit: int = 200
    sink: TelemetrySink | None = None

    def configure(
        self,
        *,
        enabled: bool,
        limit: int,
        sink: TelemetrySink | None,
    ) -> None:
        self.enabled = bool(enabled)
        self.limit = self._normalize_limit(limit)
        if sink is None:
            self.sink = PersistentTelemetrySink(path=default_telemetry_path(), capacity=self.limit)
        else:
            self.sink = sink

    def emit_context_usage(self, context: Mapping[str, Any]) -> None:
        if not self.enabled or self.sink is None:
            return
        event = ContextUsageEvent(
            document_id=context.get("document_id"),
            model=str(context.get("model") or "unknown"),
            prompt_tokens=int(context.get("prompt_tokens", 0)),
            tool_tokens=int(context.get("tool_tokens", 0)),
            response_reserve=context.get("response_reserve"),
            timestamp=float(context.get("timestamp", 0.0) or 0.0),
            conversation_length=int(context.get("conversation_length", 0)),
            tool_names=self._normalize_tool_names(context.get("tool_names")),
            run_id=str(context.get("run_id") or ""),
            embedding_backend=self._coerce_optional_str(context.get("embedding_backend")),
            embedding_model=self._coerce_optional_str(context.get("embedding_model")),
            embedding_status=self._coerce_optional_str(context.get("embedding_status")),
            embedding_detail=self._coerce_optional_str(context.get("embedding_detail")),
            outline_digest=self._coerce_optional_str(context.get("outline_digest")),
            outline_status=self._coerce_optional_str(context.get("outline_status")),
            outline_version_id=self._coerce_optional_int(context.get("outline_version_id")),
            outline_latency_ms=self._coerce_optional_float(context.get("outline_latency_ms")),
            outline_node_count=self._coerce_optional_int(context.get("outline_node_count")),
            outline_token_count=self._coerce_optional_int(context.get("outline_token_count")),
            outline_trimmed=context.get("outline_trimmed"),
            outline_is_stale=context.get("outline_is_stale"),
            outline_age_seconds=self._coerce_optional_float(context.get("outline_age_seconds")),
            retrieval_status=self._coerce_optional_str(context.get("retrieval_status")),
            retrieval_strategy=self._coerce_optional_str(context.get("retrieval_strategy")),
            retrieval_latency_ms=self._coerce_optional_float(context.get("retrieval_latency_ms")),
            retrieval_pointer_count=self._coerce_optional_int(context.get("retrieval_pointer_count")),
            analysis_chunk_profile=self._coerce_optional_str(context.get("analysis_chunk_profile")),
            analysis_required_tools=self._normalize_string_tuple(context.get("analysis_required_tools")),
            analysis_optional_tools=self._normalize_string_tuple(context.get("analysis_optional_tools")),
            analysis_must_refresh_outline=self._coerce_optional_bool(context.get("analysis_must_refresh_outline")),
            analysis_plot_state_status=self._coerce_optional_str(context.get("analysis_plot_state_status")),
            analysis_concordance_status=self._coerce_optional_str(context.get("analysis_concordance_status")),
            analysis_warning_codes=self._normalize_string_tuple(context.get("analysis_warning_codes")),
            analysis_cache_state=self._coerce_optional_str(context.get("analysis_cache_state")),
            analysis_generated_at=self._coerce_optional_float(context.get("analysis_generated_at")),
            analysis_rule_trace=self._normalize_string_tuple(context.get("analysis_rule_trace")),
            scope_origin_counts=self._normalize_scope_counts(context.get("scope_origin_counts")),
            scope_missing_count=self._coerce_optional_int(context.get("scope_missing_count")),
            scope_total_length=self._coerce_optional_int(context.get("scope_total_length")),
        )
        try:
            self.sink.record(event)
        except Exception:  # pragma: no cover - sink errors must not break chat
            LOGGER.debug("Telemetry sink rejected event", exc_info=True)

    def get_recent_events(self, limit: int | None = None) -> list[ContextUsageEvent]:
        sink = self.sink
        if sink is None:
            return []
        if hasattr(sink, "tail"):
            tail = getattr(sink, "tail")
            try:
                return list(tail(limit))  # type: ignore[misc]
            except TypeError:
                return list(tail())  # type: ignore[misc]
        return []

    @staticmethod
    def _normalize_limit(limit: int) -> int:
        try:
            normalized = int(limit)
        except (TypeError, ValueError):  # pragma: no cover - defensive coercion
            normalized = 200
        return max(20, min(normalized, 10_000))

    @staticmethod
    def _normalize_tool_names(tool_names: Any) -> tuple[str, ...]:
        return TelemetryManager._normalize_string_tuple(tool_names)

    @staticmethod
    def _normalize_string_tuple(value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, set):
            items = sorted(value)
        elif isinstance(value, (list, tuple)):
            items = value
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return ()
            return (stripped,)
        else:
            return ()
        normalized: list[str] = []
        for item in items:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return tuple(normalized)

    @staticmethod
    def _normalize_scope_counts(value: Any) -> tuple[tuple[str, int], ...]:
        if value in (None, ""):
            return ()
        items: Sequence[tuple[Any, Any]] | None = None
        if isinstance(value, Mapping):
            items = tuple(value.items())
        elif isinstance(value, Sequence):
            candidate: list[tuple[Any, Any]] = []
            for entry in value:
                if isinstance(entry, Mapping):
                    candidate.append((entry.get("origin"), entry.get("count")))
                elif isinstance(entry, Sequence) and len(entry) == 2:
                    candidate.append((entry[0], entry[1]))
            items = tuple(candidate)
        if not items:
            return ()
        normalized: list[tuple[str, int]] = []
        for origin_raw, count_raw in items:
            origin = str(origin_raw).strip()
            if not origin:
                continue
            try:
                count = int(count_raw)
            except (TypeError, ValueError):
                continue
            if count < 0:
                continue
            normalized.append((origin, count))
        normalized.sort(key=lambda pair: pair[0])
        return tuple(normalized)

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_optional_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _coerce_optional_bool(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y"}:
                return True
            if lowered in {"false", "0", "no", "n"}:
                return False
        return None
