"""Status and telemetry controller extracted from the main window."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

from ..services import telemetry as telemetry_service
from ..widgets.status_bar import StatusBar
from .models.window_state import WindowContext

LOGGER = logging.getLogger(__name__)


class TelemetryController:
    """Owns memory usage status updates and subagent telemetry indicators."""

    def __init__(
        self,
        *,
        status_bar: StatusBar | None,
        context: WindowContext,
        initial_subagent_enabled: bool = False,
    ) -> None:
        self._status_bar = status_bar
        self._context = context
        self._last_compaction_stats: Mapping[str, int] | None = None
        self._subagent_enabled = initial_subagent_enabled
        self._subagent_active_jobs: set[str] = set()
        self._subagent_job_totals: dict[str, int] = {"completed": 0, "failed": 0, "skipped": 0}
        self._subagent_last_event = ""
        self._subagent_telemetry_registered = False

    # ------------------------------------------------------------------
    # Context usage dashboard helpers
    # ------------------------------------------------------------------
    def set_compaction_stats(self, stats: Mapping[str, int] | None) -> None:
        self._last_compaction_stats = dict(stats) if isinstance(stats, Mapping) else None

    def refresh_context_usage_status(self) -> None:
        controller = getattr(self._context, "ai_controller", None)
        settings = getattr(self._context, "settings", None)
        status_bar = self._status_bar
        if controller is None or settings is None or status_bar is None:
            return
        debug_settings = getattr(settings, "debug", None)
        if not getattr(debug_settings, "token_logging_enabled", False):
            return
        limit = getattr(debug_settings, "token_log_limit", 1)
        try:
            limit_value = max(1, int(limit))
        except (TypeError, ValueError):
            limit_value = 1
        events = controller.get_recent_context_events(limit=limit_value)
        dashboard = telemetry_service.build_usage_dashboard(events)
        if dashboard is None:
            return
        summary_text = dashboard.summary_text
        compaction_stats = self._last_compaction_stats
        if isinstance(compaction_stats, Mapping):
            compactions = int(compaction_stats.get("total_compactions", 0))
            tokens_saved = int(compaction_stats.get("tokens_saved", 0))
            if compactions or tokens_saved:
                stats_bits = f"Compactions {compactions}"
                if tokens_saved:
                    stats_bits = f"{stats_bits} (saved {tokens_saved:,})"
                summary_text = f"{summary_text} · {stats_bits}" if summary_text else stats_bits
        budget_snapshot = None
        getter = getattr(controller, "get_budget_status", None)
        if callable(getter):
            budget_snapshot = getter()
        if isinstance(budget_snapshot, Mapping):
            budget_text = str(budget_snapshot.get("summary_text") or "").strip()
            if budget_text:
                summary_text = f"{summary_text} · {budget_text}" if summary_text else budget_text
        try:
            status_bar.set_memory_usage(
                summary_text,
                totals=dashboard.totals_text,
                last_tool=dashboard.summary.last_tool,
            )
        except Exception:  # pragma: no cover - defensive UI call
            LOGGER.debug("Failed to update memory usage status", exc_info=True)

    # ------------------------------------------------------------------
    # Subagent telemetry helpers
    # ------------------------------------------------------------------
    def register_subagent_listeners(self) -> None:
        if self._subagent_telemetry_registered:
            return
        for event_name in (
            "subagent.job_started",
            "subagent.job_completed",
            "subagent.job_failed",
            "subagent.job_skipped",
        ):
            telemetry_service.register_event_listener(event_name, self.handle_subagent_telemetry)
        self._subagent_telemetry_registered = True

    def handle_subagent_telemetry(self, payload: Mapping[str, Any] | None) -> None:
        if not isinstance(payload, Mapping):
            return
        event_name = str(payload.get("event") or "")
        if not event_name.startswith("subagent."):
            return
        handler_map = {
            "subagent.job_started": self._record_subagent_job_started,
            "subagent.job_completed": self._record_subagent_job_completed,
            "subagent.job_failed": self._record_subagent_job_failed,
            "subagent.job_skipped": self._record_subagent_job_skipped,
        }
        handler = handler_map.get(event_name)
        if handler is None:
            return
        handler(payload)
        self.update_subagent_indicator()

    def set_subagent_enabled(self, enabled: bool) -> None:
        normalized = bool(enabled)
        if self._subagent_enabled == normalized:
            if not normalized:
                self._subagent_active_jobs.clear()
            self.update_subagent_indicator()
            return
        self._subagent_enabled = normalized
        if not normalized:
            self._subagent_active_jobs.clear()
        self.update_subagent_indicator()

    def update_subagent_indicator(self) -> None:
        status_bar = self._status_bar
        if status_bar is None:
            return
        enabled = bool(self._subagent_enabled)
        if not enabled:
            detail = "Enable Phase 4 subagents in Settings to capture chunk-level scouts."
            try:
                status_bar.set_subagent_status("Off", detail=detail)
            except Exception:
                LOGGER.debug("Failed to update subagent status", exc_info=True)
            return

        active_jobs = len(self._subagent_active_jobs)
        status = f"Running ({active_jobs})" if active_jobs else "Idle"
        detail_parts: list[str] = []
        if active_jobs:
            detail_parts.append(f"{active_jobs} active job{'s' if active_jobs != 1 else ''}")
        totals_text = self._format_subagent_totals()
        if totals_text:
            detail_parts.append(totals_text)
        event_text = (self._subagent_last_event or "").strip()
        if event_text:
            detail_parts.append(event_text)
        detail = " · ".join(detail_parts) or "No subagent telemetry yet."
        try:
            status_bar.set_subagent_status(status, detail=detail)
        except Exception:
            LOGGER.debug("Failed to update subagent indicator", exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_subagent_job_started(self, payload: Mapping[str, Any]) -> None:
        job_id = self._coerce_subagent_job_id(payload)
        if job_id:
            self._subagent_active_jobs.add(job_id)
        chunk_id = payload.get("chunk_id") or payload.get("document_id")
        estimate = payload.get("token_estimate") or payload.get("prompt_tokens")
        bits = []
        if chunk_id:
            bits.append(f"chunk {chunk_id}")
        if estimate:
            bits.append(f"~{estimate:,} tokens")
        self._subagent_last_event = self._format_subagent_event_detail("Job started", job_id, bits)

    def _record_subagent_job_completed(self, payload: Mapping[str, Any]) -> None:
        job_id = self._coerce_subagent_job_id(payload)
        if job_id:
            self._subagent_active_jobs.discard(job_id)
        self._increment_subagent_total("completed")
        chunk_id = payload.get("chunk_id") or payload.get("document_id")
        latency = payload.get("latency_ms")
        tokens = payload.get("tokens_used")
        bits = []
        if chunk_id:
            bits.append(f"chunk {chunk_id}")
        if latency is not None:
            bits.append(f"{float(latency):.0f} ms")
        if tokens is not None:
            bits.append(f"{int(tokens):,} tokens")
        self._subagent_last_event = self._format_subagent_event_detail("Job completed", job_id, bits)

    def _record_subagent_job_failed(self, payload: Mapping[str, Any]) -> None:
        job_id = self._coerce_subagent_job_id(payload)
        if job_id:
            self._subagent_active_jobs.discard(job_id)
        self._increment_subagent_total("failed")
        reason = str(payload.get("error") or payload.get("details") or "Failed")
        chunk_id = payload.get("chunk_id") or payload.get("document_id")
        bits = [reason[:160]]
        if chunk_id:
            bits.insert(0, f"chunk {chunk_id}")
        self._subagent_last_event = self._format_subagent_event_detail("Job failed", job_id, bits)

    def _record_subagent_job_skipped(self, payload: Mapping[str, Any]) -> None:
        job_id = self._coerce_subagent_job_id(payload)
        if job_id:
            self._subagent_active_jobs.discard(job_id)
        self._increment_subagent_total("skipped")
        reason = str(payload.get("reason") or payload.get("details") or "Skipped")
        chunk_id = payload.get("chunk_id") or payload.get("document_id")
        bits = [reason[:160]]
        if chunk_id:
            bits.insert(0, f"chunk {chunk_id}")
        self._subagent_last_event = self._format_subagent_event_detail("Job skipped", job_id, bits)

    def _increment_subagent_total(self, key: str) -> None:
        current = self._subagent_job_totals.get(key, 0)
        self._subagent_job_totals[key] = current + 1

    @staticmethod
    def _coerce_subagent_job_id(payload: Mapping[str, Any]) -> str:
        job_id = payload.get("job_id") or payload.get("subagent_job_id") or payload.get("id")
        if not job_id:
            return ""
        return str(job_id)

    def _format_subagent_event_detail(
        self,
        prefix: str,
        job_id: str | None,
        segments: Iterable[str] | None = None,
    ) -> str:
        bits: list[str] = [prefix]
        if job_id:
            bits.append(f"#{job_id}")
        if segments:
            for segment in segments:
                if segment is None:
                    continue
                text = str(segment).strip()
                if text:
                    bits.append(text)
        return " – ".join(bits)

    def _format_subagent_totals(self) -> str:
        labels = {
            "completed": "Done",
            "failed": "Failed",
            "skipped": "Skipped",
        }
        parts = []
        for key, label in labels.items():
            count = int(self._subagent_job_totals.get(key, 0))
            if count:
                parts.append(f"{label} {count}")
        return " · ".join(parts)


__all__ = ["TelemetryController"]
