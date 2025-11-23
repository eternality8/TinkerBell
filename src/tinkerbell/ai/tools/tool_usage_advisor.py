"""Tool that re-runs the preflight analysis on demand."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Mapping, Sequence

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
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None = None,
        snapshot: Mapping[str, Any] | None = None,
        force_refresh: bool = False,
        reason: str | None = None,
    ) -> dict[str, Any]:
        if not callable(self.advisor):
            raise RuntimeError("analysis advisor is not configured")
        span_start, span_end = self._coerce_range(target_range)
        advice = self.advisor(
            document_id=document_id,
            span_start=span_start,
            span_end=span_end,
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

    @staticmethod
    def _coerce_range(
        target_range: Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
    ) -> tuple[int | None, int | None]:
        if target_range is None:
            return None, None
        if isinstance(target_range, Mapping):
            start = target_range.get("start")
            end = target_range.get("end")
        elif isinstance(target_range, Sequence) and len(target_range) == 2 and not isinstance(target_range, (str, bytes)):
            start, end = target_range
        else:  # pragma: no cover - schema enforces the shape
            raise ValueError("target_range must be a mapping or [start, end] sequence")
        try:
            start_i = int(start)
            end_i = int(end)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_range must contain numeric start/end values") from exc
        if start_i < 0 or end_i < 0:
            raise ValueError("target_range values must be positive integers")
        if end_i < start_i:
            start_i, end_i = end_i, start_i
        return start_i, end_i
