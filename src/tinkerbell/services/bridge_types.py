"""Type definitions and dataclasses for the document bridge module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, TypeVar

from ..core.ranges import TextRange


TResult = TypeVar("TResult")
Executor = Callable[[Callable[[], TResult]], TResult]


class EditorAdapter(Protocol):
    """Minimal interface consumed by the bridge."""

    def load_document(self, document: "DocumentState") -> None:
        ...

    def to_document(self) -> "DocumentState":
        ...

    def apply_ai_edit(
        self, directive: "EditDirective", *, preserve_selection: bool = False
    ) -> "DocumentState":
        ...

    def apply_patch_result(
        self,
        result: "PatchResult",
        selection_hint: tuple[int, int] | None = None,
        *,
        preserve_selection: bool = False,
    ) -> "DocumentState":
        ...

    def restore_document(self, document: "DocumentState") -> "DocumentState":
        ...


# Avoid circular imports by using string forward references
# Actual types are imported at runtime by bridge.py
if False:  # TYPE_CHECKING equivalent that doesn't require typing import
    from ..editor.document_model import DocumentState
    from ..editor.patches import PatchResult
    from tinkerbell.ui.presentation.chat.message_model import EditDirective


@dataclass(slots=True)
class PatchRangePayload:
    """Normalized representation of a streamed patch range payload."""

    start: int
    end: int
    replacement: str
    match_text: str
    chunk_id: str | None = None
    chunk_hash: str | None = None
    scope_origin: str | None = None
    scope_length: int | None = None
    scope_range: tuple[int, int] | None = None
    scope: Mapping[str, Any] | None = None


@dataclass(slots=True)
class EditContext:
    """Details about the most recently applied directive."""

    action: str
    target_range: TextRange
    replaced_text: str
    content: str
    diff: str | None = None
    spans: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_range", TextRange.from_value(self.target_range, fallback=(0, 0))
        )


@dataclass(slots=True)
class PatchMetrics:
    """Lightweight telemetry captured for diff-based edits."""

    total: int = 0
    conflicts: int = 0
    avg_latency_ms: float = 0.0

    def record_success(self, duration_seconds: float) -> None:
        self.total += 1
        duration_ms = max(0.0, duration_seconds * 1000.0)
        if self.avg_latency_ms == 0.0:
            self.avg_latency_ms = duration_ms
        else:
            self.avg_latency_ms = (self.avg_latency_ms * 0.8) + (duration_ms * 0.2)

    def record_conflict(self) -> None:
        self.conflicts += 1


@dataclass(slots=True)
class SafeEditSettings:
    """Runtime configuration for post-edit inspections."""

    enabled: bool = False
    duplicate_threshold: int = 2
    token_drift: float = 0.05


class DocumentVersionMismatchError(RuntimeError):
    """Raised when an edit references a stale document version."""

    def __init__(
        self,
        message: str,
        *,
        cause: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.cause = cause
        self.details = dict(details) if isinstance(details, Mapping) else None


@dataclass(slots=True)
class QueuedEdit:
    """Internal representation of a validated directive awaiting execution."""

    directive: Any  # EditDirective - using Any to avoid circular import
    context_version: str | None = None
    content_hash: str | None = None
    payload: Mapping[str, Any] | None = None
    diff: str | None = None
    ranges: tuple[PatchRangePayload, ...] = ()
    scope_summary: Mapping[str, Any] | None = None
