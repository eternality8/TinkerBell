"""Rule engine for generating analysis findings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

from .models import AnalysisFinding, AnalysisInput, AnalysisWarning
from .sources import infer_profile_from_path, normalize_chunk_profile


@dataclass(slots=True)
class RuleContext:
    """Shared metadata available to every rule."""

    default_profile: str = "auto"
    outline_stale_after_seconds: float = 600.0
    concordance_stale_after_seconds: float = 600.0


class AnalysisRule(Protocol):
    """Interface implemented by concrete rule classes."""

    name: str

    def evaluate(self, analysis_input: AnalysisInput, context: RuleContext) -> AnalysisFinding | None:
        ...


def _warning(code: str, message: str, *, severity: str = "warning") -> AnalysisWarning:
    return AnalysisWarning(code=code, message=message, severity=severity)


class ChunkProfileRule:
    name = "chunk_profile"

    def evaluate(self, analysis_input: AnalysisInput, context: RuleContext) -> AnalysisFinding | None:
        profile = normalize_chunk_profile(analysis_input.chunk_profile_hint)
        if profile is None:
            profile = normalize_chunk_profile(analysis_input.chunk_manifest_profile)
        if profile is None:
            profile = infer_profile_from_path(analysis_input.document_path)
        char_count = analysis_input.document_chars or 0
        if profile is None:
            if char_count >= 60_000:
                profile = "prose"
            else:
                profile = context.default_profile
        required_tools: tuple[str, ...] = ("document_chunk",)
        trace = f"chunk_profile:{profile}:{char_count}"
        return AnalysisFinding(trace=trace, chunk_profile=profile, required_tools=required_tools)


class OutlineFreshnessRule:
    name = "outline_freshness"

    def evaluate(self, analysis_input: AnalysisInput, context: RuleContext) -> AnalysisFinding | None:
        age = analysis_input.outline_age_seconds
        if age is None:
            warning = _warning("outline.missing", "Outline has never been generated for this document")
            return AnalysisFinding(trace="outline:missing", warnings=(warning,), must_refresh_outline=True)
        if age > context.outline_stale_after_seconds:
            warning = _warning("outline.stale", f"Outline is {int(age)}s old; refresh recommended")
            return AnalysisFinding(
                trace=f"outline:stale:{int(age)}",
                warnings=(warning,),
                must_refresh_outline=True,
            )
        return AnalysisFinding(trace="outline:fresh", must_refresh_outline=False)


class PlotStateRule:
    name = "plot_state"

    def evaluate(self, analysis_input: AnalysisInput, context: RuleContext) -> AnalysisFinding | None:
        status = analysis_input.plot_state_status
        if not status:
            return None
        status_lower = status.lower()
        if status_lower in {"missing", "stale"}:
            warning = _warning("plot_state." + status_lower, "Plot state requires an update before editing")
            required = ("plot_state_update",)
            return AnalysisFinding(
                trace=f"plot_state:{status_lower}",
                plot_state_status=status_lower,
                warnings=(warning,),
                required_tools=required,
            )
        return AnalysisFinding(trace=f"plot_state:{status_lower}", plot_state_status=status_lower)


class ConcordanceRule:
    name = "concordance"

    def evaluate(self, analysis_input: AnalysisInput, context: RuleContext) -> AnalysisFinding | None:
        status = (analysis_input.concordance_status or "").lower()
        age = analysis_input.concordance_age_seconds
        if not status and age is None:
            return None
        if status in {"missing", "stale"} or (age and age > context.concordance_stale_after_seconds):
            warning = _warning("concordance.stale", "Character/concordance data is missing or stale")
            optional = ("character_map",)
            return AnalysisFinding(
                trace="concordance:stale",
                concordance_status=status or "stale",
                warnings=(warning,),
                optional_tools=optional,
            )
        return AnalysisFinding(trace=f"concordance:{status or 'fresh'}", concordance_status=status or "fresh")


class RetrievalRule:
    name = "retrieval"

    def evaluate(self, analysis_input: AnalysisInput, context: RuleContext) -> AnalysisFinding | None:
        char_count = analysis_input.document_chars or 0
        selection_len = analysis_input.selection_length()
        if char_count < 5_000 and selection_len < 2_000:
            return None
        required = ("document_snapshot",)
        optional = ("document_chunk",) if char_count < 20_000 else ()
        return AnalysisFinding(
            trace=f"retrieval:{char_count}:{selection_len}",
            required_tools=required,
            optional_tools=optional,
        )


def default_rules() -> tuple[AnalysisRule, ...]:
    return (
        ChunkProfileRule(),
        OutlineFreshnessRule(),
        PlotStateRule(),
        ConcordanceRule(),
        RetrievalRule(),
    )


def iterate_findings(analysis_input: AnalysisInput, context: RuleContext, rules: Iterable[AnalysisRule]) -> list[AnalysisFinding]:
    findings: list[AnalysisFinding] = []
    for rule in rules:
        finding = rule.evaluate(analysis_input, context)
        if finding is None:
            continue
        findings.append(finding)
    return findings
