"""Status and telemetry controller extracted from the main window."""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

from ..ai.analysis import AnalysisAdvice
from ..services import telemetry as telemetry_service
from ..widgets.status_bar import StatusBar
from .models.window_state import WindowContext
from ..chat.chat_panel import ChatPanel

LOGGER = logging.getLogger(__name__)


class TelemetryController:
    """Owns memory usage status updates and subagent telemetry indicators."""

    def __init__(
        self,
        *,
        status_bar: StatusBar | None,
        context: WindowContext,
        initial_subagent_enabled: bool = False,
        chat_panel: ChatPanel | None = None,
    ) -> None:
        self._status_bar = status_bar
        self._context = context
        self._last_compaction_stats: Mapping[str, int] | None = None
        self._subagent_enabled = initial_subagent_enabled
        self._subagent_active_jobs: set[str] = set()
        self._subagent_pending_jobs: set[str] = set()
        self._subagent_job_totals: dict[str, int] = {"completed": 0, "failed": 0, "skipped": 0}
        self._subagent_last_event = ""
        self._subagent_last_queue_detail = ""
        self._subagent_telemetry_registered = False
        self._chat_panel = chat_panel
        self._chunk_flow_registered = False
        self._chunk_flow_warning_active = False
        self._chunk_flow_status_text: str = ""
        self._chunk_flow_detail_text: str = ""
        self._analysis_indicator: dict[str, str] | None = None

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
            "subagent.jobs_queued",
        ):
            telemetry_service.register_event_listener(event_name, self.handle_subagent_telemetry)
        self._subagent_telemetry_registered = True

    def register_chunk_flow_listeners(self) -> None:
        if self._chunk_flow_registered:
            return
        for event_name in (
            "chunk_flow.requested",
            "chunk_flow.escaped_full_snapshot",
            "chunk_flow.retry_success",
        ):
            telemetry_service.register_event_listener(event_name, self.handle_chunk_flow_event)
        self._chunk_flow_registered = True

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
            "subagent.jobs_queued": self._record_subagent_jobs_queued,
        }
        handler = handler_map.get(event_name)
        if handler is None:
            return
        handler(payload)
        self.update_subagent_indicator()

    def handle_chunk_flow_event(self, payload: Mapping[str, Any] | None) -> None:
        if not isinstance(payload, Mapping):
            return
        event_name = str(payload.get("event") or "")
        if event_name == "chunk_flow.escaped_full_snapshot":
            detail = self._format_chunk_flow_warning(payload)
            self._chunk_flow_warning_active = True
            self._set_chunk_flow_indicator("Chunk Flow Warning", detail)
            return
        if event_name == "chunk_flow.retry_success":
            detail = self._format_chunk_flow_recovery(payload)
            self._chunk_flow_warning_active = False
            self._set_chunk_flow_indicator("Chunk Flow Recovered", detail)
            return
        if event_name == "chunk_flow.requested" and not self._chunk_flow_warning_active:
            self.reset_chunk_flow_state()

    def reset_chunk_flow_state(self) -> None:
        self._chunk_flow_warning_active = False
        self._set_chunk_flow_indicator(None)

    def refresh_analysis_state(self, document_id: str | None, *, document_label: str | None = None) -> None:
        controller = getattr(self._context, "ai_controller", None)
        if controller is None or not document_id:
            self._set_analysis_indicator(None)
            return
        getter = getattr(controller, "get_latest_analysis_advice", None)
        if not callable(getter):
            self._set_analysis_indicator(None)
            return
        advice = getter(document_id)
        if advice is None:
            self._set_analysis_indicator(None)
            return
        self._set_analysis_indicator(advice, document_label=document_label or document_id)

    def chunk_flow_snapshot(self) -> Mapping[str, str] | None:
        if not self._chunk_flow_status_text:
            return None
        return {"status": self._chunk_flow_status_text, "detail": self._chunk_flow_detail_text}

    def analysis_snapshot(self) -> Mapping[str, str] | None:
        if self._analysis_indicator is None:
            return None
        return dict(self._analysis_indicator)

    def describe_analysis_indicator(
        self,
        advice: AnalysisAdvice,
        *,
        document_label: str | None = None,
    ) -> dict[str, str]:
        status_text, badge_text, detail_text = self._format_analysis_indicator(advice, document_label)
        return {"status": status_text, "badge": badge_text, "detail": detail_text}

    def set_subagent_enabled(self, enabled: bool) -> None:
        normalized = bool(enabled)
        if self._subagent_enabled == normalized:
            if not normalized:
                self._subagent_active_jobs.clear()
                self._subagent_pending_jobs.clear()
                self._subagent_last_queue_detail = ""
            self.update_subagent_indicator()
            return
        self._subagent_enabled = normalized
        if not normalized:
            self._subagent_active_jobs.clear()
            self._subagent_pending_jobs.clear()
            self._subagent_last_queue_detail = ""
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
        pending_jobs = len(self._subagent_pending_jobs)
        status = f"Running ({active_jobs})" if active_jobs else "Idle"
        if active_jobs == 0 and pending_jobs:
            status = f"Queued ({pending_jobs})"
        detail_parts: list[str] = []
        if active_jobs:
            detail_parts.append(f"{active_jobs} active job{'s' if active_jobs != 1 else ''}")
        if pending_jobs:
            detail_parts.append(f"{pending_jobs} queued job{'s' if pending_jobs != 1 else ''}")
            if self._subagent_last_queue_detail:
                detail_parts.append(self._subagent_last_queue_detail)
        totals_text = self._format_subagent_totals()
        if totals_text:
            detail_parts.append(totals_text)
        event_text = (self._subagent_last_event or "").strip()
        if event_text:
            detail_parts.append(event_text)
        if not pending_jobs:
            self._subagent_last_queue_detail = ""
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
            self._subagent_pending_jobs.discard(job_id)
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
            self._subagent_pending_jobs.discard(job_id)
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
            self._subagent_pending_jobs.discard(job_id)
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
            self._subagent_pending_jobs.discard(job_id)
            self._subagent_active_jobs.discard(job_id)

        self._increment_subagent_total("skipped")
        reason = str(payload.get("reason") or payload.get("details") or "Skipped")
        chunk_id = payload.get("chunk_id") or payload.get("document_id")
        bits = [reason[:160]]
        if chunk_id:
            bits.insert(0, f"chunk {chunk_id}")
        self._subagent_last_event = self._format_subagent_event_detail("Job skipped", job_id, bits)

    def _record_subagent_jobs_queued(self, payload: Mapping[str, Any]) -> None:
        job_ids = payload.get("job_ids")
        added = 0
        if isinstance(job_ids, Iterable) and not isinstance(job_ids, (str, bytes)):
            for job_id in job_ids:
                normalized = str(job_id).strip()
                if normalized:
                    if normalized not in self._subagent_active_jobs:
                        self._subagent_pending_jobs.add(normalized)
                    added += 1
        if added == 0:
            try:
                added = int(payload.get("job_count") or 0)
            except (TypeError, ValueError):
                added = 0
        raw_chunk_ids = payload.get("chunk_ids")
        chunk_ids = (
            raw_chunk_ids
            if isinstance(raw_chunk_ids, Iterable) and not isinstance(raw_chunk_ids, (str, bytes))
            else None
        )
        raw_reasons = payload.get("reasons")
        reasons = (
            raw_reasons
            if isinstance(raw_reasons, Iterable) and not isinstance(raw_reasons, (str, bytes))
            else None
        )
        detail = self._format_subagent_queue_detail(chunk_ids, reasons)
        self._subagent_last_queue_detail = detail
        segments: list[str] = []
        if added:
            segments.append(f"{added} job{'s' if added != 1 else ''}")
        if detail:
            segments.append(detail)
        self._subagent_last_event = self._format_subagent_event_detail("Jobs queued", None, segments)

    def _format_subagent_queue_detail(
        self,
        chunk_ids: Iterable[Any] | None,
        reasons: Iterable[Any] | None,
    ) -> str:
        bits: list[str] = []
        if chunk_ids:
            labels = [str(value).strip() for value in chunk_ids if str(value).strip()]
            if labels:
                preview = ", ".join(labels[:3])
                if len(labels) > 3:
                    preview = f"{preview}, +{len(labels) - 3} more"
                bits.append(f"chunks {preview}")
        if reasons:
            reason_labels = [str(value).strip() for value in reasons if str(value).strip()]
            if reason_labels:
                bits.append(f"reasons: {', '.join(reason_labels)}")
        return " · ".join(bits)

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

    def _set_chunk_flow_indicator(self, status: str | None, detail: str | None = None) -> None:
        status_bar = self._status_bar
        chat_panel = self._chat_panel
        if status:
            self._chunk_flow_status_text = status.strip()
            self._chunk_flow_detail_text = (detail or "").strip()
        else:
            self._chunk_flow_status_text = ""
            self._chunk_flow_detail_text = ""
        if status_bar is not None:
            status_bar.set_chunk_flow_state(status, detail=detail)
        if chat_panel is not None:
            setter = getattr(chat_panel, "set_guardrail_state", None)
            if callable(setter):
                setter(status, detail=detail, category="chunk_flow")

    def _set_analysis_indicator(self, advice: AnalysisAdvice | None, *, document_label: str | None = None) -> None:
        status_bar = self._status_bar
        chat_panel = self._chat_panel
        if advice is None:
            self._analysis_indicator = None
            if status_bar is not None:
                status_bar.set_analysis_state(None)
            if chat_panel is not None:
                badge_setter = getattr(chat_panel, "set_analysis_badge", None)
                if callable(badge_setter):
                    badge_setter(None)
            return
        overview = self.describe_analysis_indicator(advice, document_label=document_label)
        self._analysis_indicator = dict(overview)
        if status_bar is not None:
            status_bar.set_analysis_state(overview["status"], detail=overview["detail"])
        if chat_panel is not None:
            badge_setter = getattr(chat_panel, "set_analysis_badge", None)
            if callable(badge_setter):
                badge_setter(overview["badge"], detail=overview["detail"])

    def _format_analysis_indicator(
        self,
        advice: AnalysisAdvice,
        document_label: str | None,
    ) -> tuple[str, str, str]:
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
        detail_lines.append(f"Outline refresh required: {'yes' if advice.must_refresh_outline else 'no'}")
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
        document_id = str(payload.get("document_id") or "this document")
        reason = str(payload.get("reason") or "full snapshot").replace("_", " ")
        doc_length = TelemetryController._coerce_int(payload.get("document_length"))
        approx = f" (~{doc_length:,} chars)" if doc_length else ""
        window_kind = str(payload.get("window_kind") or "").strip()
        window_hint = f" via {window_kind}" if window_kind else ""
        return f"{document_id}: {reason}{window_hint}{approx}".strip()

    @staticmethod
    def _format_chunk_flow_recovery(payload: Mapping[str, Any]) -> str:
        document_id = str(payload.get("document_id") or "this document")
        recovered_via = str(payload.get("recovered_via") or payload.get("source") or "DocumentChunkTool")
        return f"{document_id}: recovered via {recovered_via}"

    @staticmethod
    def _coerce_int(value: Any) -> int | None:  # pragma: no cover - helper mirrors controller logic
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


__all__ = ["TelemetryController"]
