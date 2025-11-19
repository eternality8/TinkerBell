"""Tool that re-runs the preflight analysis on demand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping

from ..analysis import AnalysisAdvice

AnalysisInvoker = Callable[..., AnalysisAdvice | None]


@dataclass(slots=True)
class ToolUsageAdvisorTool:
    """Expose the analysis agent to the model as a callable tool."""

    advisor: AnalysisInvoker
    summarizable: ClassVar[bool] = False

    def run(
        self,
        *,
        document_id: str | None = None,
        selection_start: int | None = None,
        selection_end: int | None = None,
        snapshot: Mapping[str, Any] | None = None,
        force_refresh: bool = False,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if not callable(self.advisor):
            raise RuntimeError("analysis advisor is not configured")
        advice = self.advisor(
            document_id=document_id,
            selection_start=selection_start,
            selection_end=selection_end,
            snapshot=snapshot,
            force_refresh=force_refresh,
            reason=reason,
        )
        if advice is None:
            return {
                "status": "disabled",
                "message": "Analysis agent is disabled or unavailable",
            }
        return {
            "status": "ok",
            "advice": advice.to_dict(),
        }
