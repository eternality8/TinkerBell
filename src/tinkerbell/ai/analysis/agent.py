"""Rule-based analyzer that emits tool recommendations."""

from __future__ import annotations

import logging
from typing import Callable, Mapping, Sequence

from .cache import AnalysisCache
from .models import AnalysisAdvice, AnalysisFinding, AnalysisInput, AnalysisWarning
from .rules import RuleContext, default_rules, iterate_findings

LOGGER = logging.getLogger(__name__)
TelemetryEmitter = Callable[[str, Mapping[str, object]], None]


class AnalysisAgent:
    """Evaluates document metadata and produces actionable advice."""

    def __init__(
        self,
        *,
        cache: AnalysisCache | None = None,
        rules: Sequence[object] | None = None,
        telemetry_emitter: TelemetryEmitter | None = None,
        ttl_seconds: float = 120.0,
    ) -> None:
        self._cache = cache or AnalysisCache(ttl_seconds=ttl_seconds)
        self._rules = tuple(rules) if rules else default_rules()
        self._telemetry_emitter = telemetry_emitter

    def analyze(
        self,
        analysis_input: AnalysisInput,
        *,
        force_refresh: bool = False,
        source: str = "controller",
    ) -> AnalysisAdvice:
        cache_key = analysis_input.cache_key()
        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached is not None:
                advice = cached.with_cache_state("hit")
                self._emit("analysis.preflight.cache_hit", self._base_payload(analysis_input, source, advice))
                return advice
        self._emit("analysis.preflight.requested", self._base_payload(analysis_input, source))
        context = RuleContext(default_profile=analysis_input.chunk_profile_hint or "auto")
        try:
            findings = iterate_findings(analysis_input, context, self._rules)
            advice = self._build_advice(analysis_input, findings)
            self._cache.set(cache_key, advice)
            self._emit("analysis.preflight.completed", self._base_payload(analysis_input, source, advice))
            return advice
        except Exception:
            LOGGER.exception("Analysis agent failed")
            warning = AnalysisWarning(code="analysis.error", message="Preflight analysis failed", severity="error")
            advice = AnalysisAdvice(
                document_id=analysis_input.document_id,
                document_version=analysis_input.document_version,
                warnings=(warning,),
            )
            self._emit("analysis.preflight.failed", self._base_payload(analysis_input, source, advice))
            return advice

    def invalidate_document(self, document_id: str) -> None:
        self._cache.invalidate_document(document_id)

    def clear_cache(self) -> None:
        self._cache.clear()

    def _build_advice(self, analysis_input: AnalysisInput, findings: Sequence[AnalysisFinding]) -> AnalysisAdvice:
        chunk_profile = analysis_input.chunk_profile_hint or "auto"
        required_tools: list[str] = []
        optional_tools: list[str] = []
        warnings: list[AnalysisWarning] = []
        rule_trace: list[str] = []
        must_refresh_outline = False
        plot_state_status = analysis_input.plot_state_status
        concordance_status = analysis_input.concordance_status

        for finding in findings:
            rule_trace.append(finding.trace)
            if finding.chunk_profile:
                chunk_profile = finding.chunk_profile
            for name in finding.required_tools:
                if name not in required_tools:
                    required_tools.append(name)
            for name in finding.optional_tools:
                if name not in optional_tools and name not in required_tools:
                    optional_tools.append(name)
            if finding.must_refresh_outline:
                must_refresh_outline = True
            if finding.plot_state_status:
                plot_state_status = finding.plot_state_status
            if finding.concordance_status:
                concordance_status = finding.concordance_status
            if finding.warnings:
                warnings.extend(finding.warnings)

        advice = AnalysisAdvice(
            document_id=analysis_input.document_id,
            document_version=analysis_input.document_version,
            chunk_profile=chunk_profile,
            required_tools=tuple(required_tools),
            optional_tools=tuple(optional_tools),
            must_refresh_outline=must_refresh_outline,
            plot_state_status=plot_state_status,
            concordance_status=concordance_status,
            warnings=tuple(warnings),
            rule_trace=tuple(rule_trace),
        )
        return advice

    def _base_payload(
        self,
        analysis_input: AnalysisInput,
        source: str,
        advice: AnalysisAdvice | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "document_id": analysis_input.document_id,
            "document_version": analysis_input.document_version or "",
            "selection_len": analysis_input.selection_length(),
            "source": source,
        }
        if advice is not None:
            payload.update(
                {
                    "chunk_profile": advice.chunk_profile,
                    "required_tools": list(advice.required_tools),
                    "warnings": [warning.code for warning in advice.warnings],
                    "cache_state": advice.cache_state,
                }
            )
        return payload

    def _emit(self, event_name: str, payload: Mapping[str, object]) -> None:
        if not self._telemetry_emitter:
            return
        try:
            self._telemetry_emitter(event_name, payload)
        except Exception:  # pragma: no cover - telemetry must never break callers
            LOGGER.debug("Analysis telemetry emit failed for %s", event_name, exc_info=True)
