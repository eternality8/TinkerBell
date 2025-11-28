"""Document bridge connecting AI directives to the editor widget."""

from __future__ import annotations

import hashlib
import inspect
import logging
import time
from copy import deepcopy
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Mapping, Optional, Protocol, Sequence, TypeVar

from ..ai.memory.cache_bus import (
    DocumentCacheBus,
    DocumentChangedEvent,
    DocumentClosedEvent,
    get_document_cache_bus,
)
from ..ai.tools.version import VersionManager, get_version_manager
from ..chat.commands import ActionType, parse_agent_payload, validate_directive
from ..chat.message_model import EditDirective
from ..documents.ranges import TextRange
from ..documents.range_normalizer import normalize_text_range
from ..editor.document_model import DocumentState, DocumentVersion
from ..editor.patches import PatchApplyError, PatchResult, RangePatch, apply_streamed_ranges, apply_unified_diff
from ..editor.post_edit_inspector import InspectionResult, PostEditInspector
from .telemetry import emit as telemetry_emit


_LOGGER = logging.getLogger(__name__)


TResult = TypeVar("TResult")
Executor = Callable[[Callable[[], TResult]], TResult]
EditAppliedListener = Callable[[EditDirective, DocumentState, str], None]


class EditorAdapter(Protocol):
    """Minimal interface consumed by the bridge."""

    def load_document(self, document: DocumentState) -> None:
        ...

    def to_document(self) -> DocumentState:
        ...

    def apply_ai_edit(self, directive: EditDirective, *, preserve_selection: bool = False) -> DocumentState:
        ...

    def apply_patch_result(
        self,
        result: PatchResult,
        selection_hint: tuple[int, int] | None = None,
        *,
        preserve_selection: bool = False,
    ) -> DocumentState:
        ...

    def restore_document(self, document: DocumentState) -> DocumentState:
        ...


@dataclass(slots=True)
class _QueuedEdit:
    """Internal representation of a validated directive awaiting execution."""

    directive: EditDirective
    context_version: Optional[str] = None
    content_hash: Optional[str] = None
    payload: Optional[Mapping[str, Any]] = None
    diff: Optional[str] = None
    ranges: tuple["PatchRangePayload", ...] = ()
    scope_summary: Mapping[str, Any] | None = None


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
    diff: Optional[str] = None
    spans: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_range", TextRange.from_value(self.target_range, fallback=(0, 0)))


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


class DocumentBridge:
    """Orchestrates safe document snapshots, conflict detection, and queued edits."""

    PATCH_EVENT_NAME = "patch.apply"
    EDIT_REJECTED_EVENT_NAME = "edit_rejected"
    AUTO_REVERT_EVENT_NAME = "auto_revert"
    DUPLICATE_DETECTED_EVENT_NAME = "duplicate_detected"
    HASH_MISMATCH_EVENT_NAME = "hash_mismatch"
    CAUSE_HASH_MISMATCH = "hash_mismatch"
    CAUSE_CHUNK_HASH_MISMATCH = "chunk_hash_mismatch"
    CAUSE_INSPECTOR_FAILURE = "inspector_failure"
    _HASH_FAILURE_REASONS = {
        "context_mismatch",
        "context_ambiguous",
        "context_overflow",
        "range_mismatch",
        "range_overlap",
        "range_overflow",
        "unexpected_eof",
    }

    def __init__(
        self,
        *,
        editor: EditorAdapter,
        main_thread_executor: Optional[Executor] = None,
        cache_bus: DocumentCacheBus | None = None,
    ) -> None:
        self.editor = editor
        self._pending_edits: Deque[_QueuedEdit] = deque()
        self._draining = False
        self._last_diff: Optional[str] = None
        self._last_snapshot_token: Optional[str] = None
        self._last_document_version: Optional[DocumentVersion] = None
        self._last_edit_context: Optional[EditContext] = None
        self._main_thread_executor = main_thread_executor
        self._edit_listeners: list[EditAppliedListener] = []
        self._failure_listeners: list[Callable[[EditDirective, str], None]] = []
        self._failure_listener_capabilities: dict[Callable[..., Any], bool] = {}
        self._patch_metrics = PatchMetrics()
        self._cache_bus = cache_bus or get_document_cache_bus()
        self._chunk_manifest_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._tab_id: Optional[str] = None
        self._last_failure_metadata: Optional[dict[str, Any]] = None
        self._safe_edit_settings = SafeEditSettings()
        self._post_edit_inspector = PostEditInspector(
            duplicate_threshold=self._safe_edit_settings.duplicate_threshold,
            token_drift=self._safe_edit_settings.token_drift,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_snapshot(
        self,
        *,
        delta_only: bool = False,
        window: Mapping[str, Any] | str | None = None,
        max_tokens: int | None = None,
        chunk_profile: str | None = None,
        include_text: bool = True,
    ) -> dict:
        """Return a document snapshot enriched with version metadata."""

        document = self.editor.to_document()
        snapshot = document.snapshot(delta_only=delta_only)
        snapshot["length"] = len(document.text)
        snapshot["line_start_offsets"] = self._compute_line_start_offsets(document.text)
        window_range = self._resolve_window(
            window,
            length=len(document.text),
            max_tokens=max_tokens,
        )
        focus_span = TextRange(window_range["start"], window_range["end"])
        chunk_profile_name = self._normalize_chunk_profile(chunk_profile)
        snapshot["window"] = dict(window_range)
        snapshot["text_range"] = {"start": window_range["start"], "end": window_range["end"]}
        if not include_text:
            snapshot["text"] = ""
        elif not delta_only and not window_range["includes_full_document"]:
            snapshot["text"] = document.text[window_range["start"] : window_range["end"]]
        chunk_manifest = self._resolve_chunk_manifest(
            document,
            window_range,
            chunk_profile_name,
            focus_span=focus_span,
        )
        if chunk_manifest is not None:
            snapshot["chunk_manifest"] = chunk_manifest
        version = document.version_info()
        snapshot.setdefault("document_id", version.document_id)
        snapshot.setdefault("version_id", version.version_id)
        snapshot.setdefault("content_hash", version.content_hash)
        token = self._format_version_token(version)
        snapshot["version"] = token
        self._last_snapshot_token = token
        self._last_document_version = version
        
        # Register tab with version manager so write tools can validate tokens
        self._register_with_version_manager(version)
        
        return snapshot

    @property
    def last_diff_summary(self) -> Optional[str]:
        """Return a lightweight description of the most recent edit diff."""

        return self._last_diff

    @property
    def last_snapshot_version(self) -> Optional[str]:
        """Expose the digest associated with the latest document snapshot."""

        return self._last_snapshot_token

    @property
    def last_document_version(self) -> Optional[DocumentVersion]:
        """Expose the most recent :class:`DocumentVersion`."""

        return self._last_document_version

    @property
    def last_edit_context(self) -> Optional[EditContext]:
        """Expose metadata about the most recently applied edit."""

        return self._last_edit_context

    @property
    def patch_metrics(self) -> PatchMetrics:
        """Return aggregated telemetry for patch directives."""

        return self._patch_metrics

    @property
    def last_failure_metadata(self) -> Optional[Mapping[str, Any]]:
        """Expose structured details about the most recent failure if any."""

        if self._last_failure_metadata is None:
            return None
        return dict(self._last_failure_metadata)

    def add_edit_listener(self, listener: EditAppliedListener) -> None:
        """Register a callback fired after each successful directive."""

        self._edit_listeners.append(listener)

    def remove_edit_listener(self, listener: EditAppliedListener) -> None:
        """Remove a previously registered edit listener if present."""

        try:
            self._edit_listeners.remove(listener)
        except ValueError:  # pragma: no cover - defensive guard
            pass

    def add_failure_listener(self, listener: Callable[[EditDirective, str], None]) -> None:
        """Attach a callback invoked when directive application fails."""

        self._failure_listeners.append(listener)
        self._failure_listener_capabilities[listener] = self._supports_failure_metadata(listener)

    def remove_failure_listener(self, listener: Callable[[EditDirective, str], None]) -> None:
        """Detach a previously registered failure listener if present."""

        try:
            self._failure_listeners.remove(listener)
            self._failure_listener_capabilities.pop(listener, None)
        except ValueError:  # pragma: no cover - defensive guard
            pass

    def set_main_thread_executor(self, executor: Optional[Executor]) -> None:
        """Configure a callable used to marshal edits onto the UI thread."""

        self._main_thread_executor = executor

    def set_tab_context(self, *, tab_id: str | None) -> None:
        """Annotate telemetry emitted by this bridge with the owning tab identifier."""

        if isinstance(tab_id, str):
            value = tab_id.strip()
            self._tab_id = value or None
            return
        self._tab_id = None

    # ------------------------------------------------------------------
    # DocumentProvider Protocol Implementation
    # ------------------------------------------------------------------
    def get_document_content(self, tab_id: str) -> str | None:
        """Get the content of a specific tab.
        
        For single-document bridges, returns content if tab_id matches
        the current tab context, otherwise None.
        """
        # Single-document bridge - check if this is our tab
        if self._tab_id is not None and tab_id != self._tab_id:
            return None
        return self.editor.to_document().text

    def get_document_text(self, tab_id: str | None = None) -> str:
        """Get the text content of the document.
        
        Args:
            tab_id: Optional tab ID. If None, returns active document.
        """
        return self.editor.to_document().text

    def set_document_content(self, tab_id: str, content: str) -> None:
        """Set the content of a specific tab.
        
        Creates a new DocumentState with the given content and loads it.
        """
        current_doc = self.editor.to_document()
        new_doc = DocumentState(
            document_id=current_doc.document_id,
            text=content,
        )
        self.editor.load_document(new_doc)
        
        # Update version tracking
        version = new_doc.version_info()
        self._register_with_version_manager(version)

    def get_active_tab_id(self) -> str | None:
        """Get the ID of the currently active tab."""
        return self._tab_id

    def get_document_metadata(self, tab_id: str) -> dict[str, Any] | None:
        """Get metadata for a specific tab.
        
        Returns basic metadata about the document.
        """
        if self._tab_id is not None and tab_id != self._tab_id:
            return None
        doc = self.editor.to_document()
        return {
            "tab_id": tab_id,
            "document_id": doc.document_id,
            "length": len(doc.text),
            "line_count": doc.text.count("\n") + 1 if doc.text else 0,
        }

    def configure_safe_editing(
        self,
        *,
        enabled: bool,
        duplicate_threshold: int | None = None,
        token_drift: float | None = None,
    ) -> None:
        """Toggle the post-edit inspector and update thresholds."""

        self._safe_edit_settings.enabled = bool(enabled)
        if duplicate_threshold is not None:
            value = max(2, int(duplicate_threshold))
            self._safe_edit_settings.duplicate_threshold = value
        if token_drift is not None:
            value = max(0.0, float(token_drift))
            self._safe_edit_settings.token_drift = value
        self._post_edit_inspector.configure(
            duplicate_threshold=self._safe_edit_settings.duplicate_threshold,
            token_drift=self._safe_edit_settings.token_drift,
        )

    def queue_edit(self, directive: EditDirective | Mapping[str, Any]) -> None:
        """Validate, normalize, and enqueue a directive for application."""

        normalized = self._normalize_directive(directive)
        self._pending_edits.append(normalized)
        self._drain_queue()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _drain_queue(self) -> None:
        if self._draining:
            return
        self._draining = True
        try:
            while self._pending_edits:
                queued = self._pending_edits.popleft()
                self._apply_edit(queued)
        finally:
            self._draining = False

    def _apply_edit(self, queued: _QueuedEdit) -> None:
        document_before = self.editor.to_document()
        pre_edit_snapshot = deepcopy(document_before)
        version_metadata = document_before.version_info()
        if queued.directive.action != ActionType.PATCH.value:
            raise ValueError("DocumentBridge only processes patch directives")

        try:
            self._validate_patch_context(document_before, queued)
        except DocumentVersionMismatchError as exc:
            self._record_patch_rejection(
                queued,
                version_metadata,
                status="stale",
                reason=str(exc),
                cause=exc.cause or self.CAUSE_HASH_MISMATCH,
                range_count=len(queued.ranges) if queued.ranges else None,
                streamed=bool(queued.ranges),
                record_conflict=True,
            )
            raise

        self._apply_patch_directive(queued, document_before)

    def _execute_on_main_thread(self, func: Callable[[], DocumentState]) -> DocumentState:
        if self._main_thread_executor is not None:
            return self._main_thread_executor(func)
        return func()

    def _notify_listeners(self, directive: EditDirective, state: DocumentState) -> None:
        if self._last_diff is None:
            return
        for listener in list(self._edit_listeners):
            try:
                listener(directive, state, self._last_diff)
            except Exception:  # pragma: no cover - defensive logging
                _LOGGER.exception("Edit listener failed")

    def _record_failure(
        self,
        directive: EditDirective,
        message: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] | None = dict(metadata) if metadata else {}
        range_payload = directive.target_range.to_dict()
        payload.setdefault("target_range", range_payload)
        if self._tab_id:
            payload.setdefault("tab_id", self._tab_id)
        if _LOGGER.isEnabledFor(logging.DEBUG):
            tab_hint = payload.get("tab_id") if payload else None
            _LOGGER.debug(
                "DocumentBridge failure (action=%s, tab_id=%s): %s | metadata=%s",
                directive.action,
                tab_hint,
                message,
                payload,
            )
        dispatch_payload = dict(payload) if payload is not None else None
        self._last_failure_metadata = dict(dispatch_payload) if dispatch_payload is not None else None
        for listener in list(self._failure_listeners):
            accepts_metadata = self._failure_listener_capabilities.get(listener)
            if accepts_metadata is None:
                accepts_metadata = self._supports_failure_metadata(listener)
                self._failure_listener_capabilities[listener] = accepts_metadata
            try:
                if accepts_metadata:
                    listener(directive, message, dict(dispatch_payload) if dispatch_payload is not None else None)
                else:
                    listener(directive, message)
            except Exception:  # pragma: no cover - defensive guard
                _LOGGER.exception("Edit failure listener raised")

    def _record_patch_rejection(
        self,
        queued: _QueuedEdit,
        version: DocumentVersion,
        *,
        status: str,
        reason: str,
        cause: str | None,
        range_count: int | None,
        streamed: bool | None,
        record_conflict: bool,
        diagnostics: Mapping[str, Any] | None = None,
        scope_summary: Mapping[str, Any] | None = None,
    ) -> None:
        summary = reason or "edit rejected"
        self._last_diff = f"failed: {summary}"
        scope_summary = scope_summary or queued.scope_summary
        failure_metadata = self._build_failure_metadata(
            version=version,
            status=status,
            reason=summary,
            cause=cause,
            range_count=range_count,
            streamed=streamed,
            diagnostics=diagnostics,
            scope_summary=scope_summary,
        )
        self._record_failure(queued.directive, self._last_diff, failure_metadata)
        if record_conflict:
            self._patch_metrics.record_conflict()
        self._emit_patch_event(
            status=status,
            version=version,
            diff_summary=None,
            reason=summary,
            range_count=range_count,
            streamed=streamed,
            cause=cause,
            scope_summary=scope_summary,
        )
        self._emit_edit_rejected_event(
            version=version,
            action=queued.directive.action,
            reason=summary,
            cause=cause,
            range_count=range_count,
            streamed=streamed,
            diagnostics=diagnostics,
            scope_summary=scope_summary,
        )
        if cause in {self.CAUSE_HASH_MISMATCH, self.CAUSE_CHUNK_HASH_MISMATCH}:
            self._emit_hash_mismatch_event(
                version=version,
                status=status,
                reason=summary,
                cause=cause,
                range_count=range_count,
                streamed=streamed,
                diagnostics=diagnostics,
                scope_summary=scope_summary,
            )

    def _validate_patch_context(self, document: DocumentState, queued: _QueuedEdit) -> None:
        if not queued.context_version:
            raise DocumentVersionMismatchError(
                "Patch directives must include document_version metadata",
                cause=self.CAUSE_HASH_MISMATCH,
            )
        if not self._is_version_current(document, queued.context_version):
            raise DocumentVersionMismatchError(
                "Patch directive references a stale document snapshot",
                cause=self.CAUSE_HASH_MISMATCH,
            )
        expected_hash = queued.content_hash
        if expected_hash is None and queued.payload is not None:
            expected_hash = self._extract_content_hash(queued.payload)
        if not expected_hash:
            raise DocumentVersionMismatchError(
                "Patch directives must include content_hash metadata",
                cause=self.CAUSE_HASH_MISMATCH,
            )
        if document.content_hash != expected_hash:
            raise DocumentVersionMismatchError(
                "Patch directive content_hash no longer matches the live document",
                cause=self.CAUSE_HASH_MISMATCH,
            )

    def _apply_patch_directive(self, queued: _QueuedEdit, document_before: DocumentState) -> None:
        diff_text = queued.diff or queued.directive.diff or ""
        use_ranges = bool(queued.ranges)
        if not diff_text and not use_ranges:
            raise RuntimeError("Patch directive missing diff payload")

        start_time = time.perf_counter()
        pre_edit_snapshot = deepcopy(document_before)
        pre_version = document_before.version_info()
        range_count = len(queued.ranges) if use_ranges else None
        range_hint: tuple[int, int] | None = None
        scope_summary = queued.scope_summary
        try:
            if use_ranges:
                expanded_ranges = self._expand_patch_ranges(document_before, queued.ranges)
                range_count = len(expanded_ranges)
                range_hint = self._range_hint_from_expanded_ranges(document_before.text, expanded_ranges)
                if range_hint is not None:
                    queued.directive.target_range = TextRange.from_value(range_hint)
                self._validate_range_chunk_hashes(document_before, expanded_ranges)
                scope_summary = self._summarize_patch_scopes(expanded_ranges)
                queued.scope_summary = scope_summary
                patch_result = self._apply_range_payloads(expanded_ranges, document_before)
            else:
                patch_result = apply_unified_diff(document_before.text, diff_text)
        except DocumentVersionMismatchError as exc:
            self._record_patch_rejection(
                queued,
                pre_version,
                status="stale",
                reason=str(exc),
                cause=exc.cause or self.CAUSE_HASH_MISMATCH,
                range_count=range_count,
                streamed=True,
                record_conflict=True,
            )
            raise
        except PatchApplyError as exc:
            reason = getattr(exc, "reason", str(exc))
            cause = self.CAUSE_HASH_MISMATCH if reason in self._HASH_FAILURE_REASONS else None
            self._record_patch_rejection(
                queued,
                pre_version,
                status="conflict",
                reason=reason,
                cause=cause,
                range_count=range_count,
                streamed=use_ranges,
                record_conflict=True,
            )
            if use_ranges and cause == self.CAUSE_HASH_MISMATCH:
                raise DocumentVersionMismatchError("Streamed patch validation failed", cause=cause) from exc
            raise RuntimeError(f"Patch application failed: {exc}") from exc

        selection_hint = range_hint
        updated_state = self._execute_on_main_thread(
            lambda: self.editor.apply_patch_result(
                patch_result,
                selection_hint=selection_hint,
                preserve_selection=True,
            )
        )
        diff_summary = patch_result.summary
        inspection = self._run_post_edit_inspection(
            document_before=document_before,
            patch_result=patch_result,
            range_hint=range_hint,
        )
        if inspection is not None and not inspection.ok:
            rejection_details = self._handle_post_edit_rejection(
                queued=queued,
                version=pre_version,
                pre_edit_snapshot=pre_edit_snapshot,
                range_count=range_count,
                streamed=use_ranges,
                inspection=inspection,
                diff_summary=diff_summary,
            )
            raise DocumentVersionMismatchError(
                self._format_auto_revert_message(rejection_details),
                cause=self.CAUSE_INSPECTOR_FAILURE,
                details=rejection_details,
            )

        target_range = range_hint or self._derive_patch_range(patch_result.spans, len(updated_state.text))
        self._last_edit_context = EditContext(
            action=queued.directive.action,
            target_range=target_range,
            replaced_text="",
            content="",
            diff=diff_text if not use_ranges else None,
            spans=patch_result.spans,
        )
        self._last_diff = diff_summary
        version = updated_state.version_info(edited_ranges=patch_result.spans)
        self._last_snapshot_token = self._format_version_token(version)
        self._last_document_version = version
        elapsed = time.perf_counter() - start_time
        self._patch_metrics.record_success(elapsed)
        self._emit_patch_event(
            status="success",
            version=version,
            diff_summary=diff_summary,
            duration_ms=elapsed * 1000.0,
            range_count=len(patch_result.spans),
            streamed=use_ranges,
            cause=None,
            scope_summary=scope_summary,
        )

        _LOGGER.debug("Applied patch directive diff=%s spans=%s", patch_result.summary, patch_result.spans)
        self._notify_listeners(queued.directive, pre_edit_snapshot)
        self._publish_document_changed(
            version,
            spans=patch_result.spans,
            source=queued.directive.action,
        )
        self._last_failure_metadata = None

    def _emit_patch_event(
        self,
        *,
        status: str,
        version: DocumentVersion,
        diff_summary: str | None,
        duration_ms: float | None = None,
        range_count: int | None = None,
        streamed: bool | None = None,
        reason: str | None = None,
        cause: str | None = None,
        scope_summary: Mapping[str, Any] | None = None,
    ) -> None:
        event_name = self.PATCH_EVENT_NAME
        if not event_name:
            return
        payload: dict[str, Any] = {
            "document_id": version.document_id,
            "version_id": version.version_id,
            "content_hash": version.content_hash,
            "status": status,
        }
        if self._tab_id:
            payload["tab_id"] = self._tab_id
        if diff_summary:
            payload["diff_summary"] = diff_summary
        if duration_ms is not None:
            payload["duration_ms"] = max(0.0, round(duration_ms, 3))
        if range_count is not None:
            payload["range_count"] = range_count
        if streamed is not None:
            payload["streamed"] = bool(streamed)
        if reason:
            payload["reason"] = reason
        if cause:
            payload["cause"] = cause
        self._attach_scope_metadata(payload, scope_summary)
        telemetry_emit(event_name, payload)

    def _format_auto_revert_message(self, details: Mapping[str, Any] | None) -> str:
        if not isinstance(details, Mapping):
            return (
                "Auto-revert triggered. Document was restored to the previous snapshot; refresh "
                "document_snapshot and retry with an updated diff."
            )
        reason = str(details.get("reason") or "inspection_failure")
        detail = str(details.get("detail") or "Edit rejected")
        remediation = details.get("remediation")
        message = f"Auto-revert triggered ({reason}). {detail}"
        if remediation:
            message = f"{message} {remediation}".strip()
        return message

    def _emit_auto_revert_event(
        self,
        *,
        version: DocumentVersion,
        diff_summary: str | None,
        reason: str,
        detail: str,
        diagnostics: Mapping[str, Any] | None,
        scope_summary: Mapping[str, Any] | None = None,
    ) -> None:
        event_name = self.AUTO_REVERT_EVENT_NAME
        if not event_name:
            return
        payload: dict[str, Any] = {
            "document_id": version.document_id,
            "version_id": version.version_id,
            "content_hash": version.content_hash,
            "reason": reason,
            "detail": detail,
        }
        if diff_summary:
            payload["diff_summary"] = diff_summary
        if self._tab_id:
            payload["tab_id"] = self._tab_id
        if diagnostics:
            payload["diagnostics"] = dict(diagnostics)
        self._attach_scope_metadata(payload, scope_summary)
        telemetry_emit(event_name, payload)

    def _emit_duplicate_detected_event(
        self,
        *,
        version: DocumentVersion,
        reason: str,
        diagnostics: Mapping[str, Any] | None,
    ) -> None:
        event_name = self.DUPLICATE_DETECTED_EVENT_NAME
        if not event_name or not isinstance(diagnostics, Mapping):
            return
        duplicate_payload = diagnostics.get("duplicate")
        if not isinstance(duplicate_payload, Mapping):
            return
        payload: dict[str, Any] = {
            "document_id": version.document_id,
            "version_id": version.version_id,
            "content_hash": version.content_hash,
            "reason": reason,
            "duplicate": dict(duplicate_payload),
        }
        if self._tab_id:
            payload["tab_id"] = self._tab_id
        telemetry_emit(event_name, payload)

    def _auto_revert_remediation(self, reason: str | None) -> str:
        mapping = {
            "duplicate_paragraphs": "Retry with replace_all=true or delete the original text before inserting the rewrite to avoid duplicated passages.",
            "duplicate_windows": "Retry with replace_all=true or narrow the edit range so the rewrite replaces the previous text instead of appending it.",
            "boundary_dropped": "Ensure the edit preserves blank lines between paragraphs or include surrounding context in the diff.",
            "split_tokens": "Insert whitespace around the edited span so tokens are not merged across boundaries before retrying.",
            "split_token_regex": "Add a newline or space before markdown tokens (e.g., '#') so headings are not fused with preceding words.",
        }
        return mapping.get(
            reason or "",
            "Refresh document_snapshot and rebuild the diff before retrying to ensure the edit targets the latest content.",
        )

    def _run_post_edit_inspection(
        self,
        *,
        document_before: DocumentState,
        patch_result: PatchResult,
        range_hint: tuple[int, int] | None,
    ) -> InspectionResult | None:
        if not self._safe_edit_settings.enabled:
            return None
        try:
            return self._post_edit_inspector.inspect(
                before_text=document_before.text,
                after_text=patch_result.text,
                spans=patch_result.spans,
                range_hint=range_hint,
            )
        except Exception:  # pragma: no cover - inspector failures should not block edits
            _LOGGER.debug("Post-edit inspector failed", exc_info=True)
            return None

    def _handle_post_edit_rejection(
        self,
        *,
        queued: _QueuedEdit,
        version: DocumentVersion,
        pre_edit_snapshot: DocumentState,
        range_count: int | None,
        streamed: bool | None,
        inspection: InspectionResult,
        diff_summary: str | None,
    ) -> dict[str, Any]:
        self._restore_document_snapshot(pre_edit_snapshot)
        diagnostics = dict(inspection.diagnostics or {})
        reason_code = inspection.reason or "inspector_failure"
        detail = inspection.detail or reason_code or "Edit rejected"
        remediation = self._auto_revert_remediation(reason_code)
        self._emit_auto_revert_event(
            version=version,
            diff_summary=diff_summary,
            reason=reason_code,
            detail=detail,
            diagnostics=diagnostics,
            scope_summary=queued.scope_summary,
        )
        if diagnostics.get("duplicate"):
            self._emit_duplicate_detected_event(
                version=version,
                reason=reason_code,
                diagnostics=diagnostics,
            )
        self._record_patch_rejection(
            queued,
            version,
            status="rejected",
            reason=detail,
            cause=self.CAUSE_INSPECTOR_FAILURE,
            range_count=range_count,
            streamed=streamed,
            record_conflict=False,
            diagnostics=diagnostics,
            scope_summary=queued.scope_summary,
        )
        return {
            "code": "auto_revert",
            "reason": reason_code,
            "detail": detail,
            "remediation": remediation,
            "diagnostics": diagnostics,
        }

    def _restore_document_snapshot(self, snapshot: DocumentState) -> DocumentState:
        snapshot_copy = deepcopy(snapshot)
        restorer = getattr(self.editor, "restore_document", None)
        if callable(restorer):
            restored = self._execute_on_main_thread(lambda: restorer(deepcopy(snapshot_copy)))
        else:
            def _load() -> DocumentState:
                self.editor.load_document(deepcopy(snapshot_copy))
                return self.editor.to_document()

            restored = self._execute_on_main_thread(_load)
        version = restored.version_info()
        self._last_document_version = version
        self._last_snapshot_token = self._format_version_token(version)
        return restored

    def _build_failure_metadata(
        self,
        *,
        version: DocumentVersion | None,
        status: str | None = None,
        reason: str | None = None,
        cause: str | None = None,
        range_count: int | None = None,
        streamed: bool | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        scope_summary: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if version is not None:
            metadata["document_id"] = version.document_id
            metadata["version_id"] = version.version_id
            metadata["content_hash"] = version.content_hash
        if status:
            metadata["status"] = status
        if reason:
            metadata["reason"] = reason
        if cause:
            metadata["cause"] = cause
        if range_count is not None:
            metadata["range_count"] = range_count
        if streamed is not None:
            metadata["streamed"] = bool(streamed)
        if diagnostics:
            metadata["diagnostics"] = dict(diagnostics)
        self._attach_scope_metadata(metadata, scope_summary)
        return metadata

    @staticmethod
    def _supports_failure_metadata(listener: Callable[..., Any]) -> bool:
        try:
            signature = inspect.signature(listener)
        except (TypeError, ValueError):  # pragma: no cover - assume flexible
            return True
        params = list(signature.parameters.values())
        if not params:
            return False
        if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params):
            return True
        positional = [
            param
            for param in params
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional) >= 3

    def _emit_edit_rejected_event(
        self,
        *,
        version: DocumentVersion,
        action: str,
        reason: str,
        cause: str | None,
        range_count: int | None,
        streamed: bool | None,
        diagnostics: Mapping[str, Any] | None = None,
        scope_summary: Mapping[str, Any] | None = None,
    ) -> None:
        event_name = self.EDIT_REJECTED_EVENT_NAME
        if not event_name:
            return
        payload: dict[str, Any] = {
            "document_id": version.document_id,
            "version_id": version.version_id,
            "content_hash": version.content_hash,
            "action": action,
            "reason": reason,
        }
        if cause:
            payload["cause"] = cause
        if self._tab_id:
            payload["tab_id"] = self._tab_id
        if range_count is not None:
            payload["range_count"] = range_count
        if streamed is not None:
            payload["streamed"] = bool(streamed)
        if diagnostics:
            payload["diagnostics"] = dict(diagnostics)
        self._attach_scope_metadata(payload, scope_summary)
        telemetry_emit(event_name, payload)

    def _emit_hash_mismatch_event(
        self,
        *,
        version: DocumentVersion,
        status: str,
        reason: str,
        cause: str,
        range_count: int | None,
        streamed: bool | None,
        diagnostics: Mapping[str, Any] | None = None,
        scope_summary: Mapping[str, Any] | None = None,
    ) -> None:
        event_name = self.HASH_MISMATCH_EVENT_NAME
        if not event_name or not cause:
            return
        stage = "chunk_hash" if cause == self.CAUSE_CHUNK_HASH_MISMATCH else "bridge"
        payload: dict[str, Any] = {
            "document_id": version.document_id,
            "version_id": version.version_id,
            "content_hash": version.content_hash,
            "status": status,
            "cause": cause,
            "reason": reason,
            "stage": stage,
            "source": "document_bridge",
        }
        if self._tab_id:
            payload["tab_id"] = self._tab_id
        if range_count is not None:
            payload["range_count"] = range_count
        if streamed is not None:
            payload["streamed"] = bool(streamed)
        if diagnostics:
            payload["diagnostics"] = dict(diagnostics)
        self._attach_scope_metadata(payload, scope_summary)
        telemetry_emit(event_name, payload)

    def _apply_range_payloads(
        self,
        ranges: Sequence[PatchRangePayload],
        document_before: DocumentState,
    ) -> PatchResult:
        range_specs = [
            RangePatch(
                start=entry.start,
                end=entry.end,
                replacement=entry.replacement,
                match_text=entry.match_text,
                chunk_id=entry.chunk_id,
                chunk_hash=entry.chunk_hash,
            )
            for entry in ranges
        ]
        return apply_streamed_ranges(document_before.text, range_specs)

    def _validate_range_chunk_hashes(
        self,
        document: DocumentState,
        ranges: Sequence[PatchRangePayload],
    ) -> None:
        if not ranges:
            return
        text = document.text
        length = len(text)
        for entry in ranges:
            if not entry.chunk_hash:
                continue
            bounds = self._parse_chunk_bounds(entry.chunk_id)
            if bounds is None:
                continue
            start, end = bounds
            start = max(0, min(start, length))
            end = max(start, min(end, length))
            actual_slice = text[start:end]
            actual_hash = self._hash_text(actual_slice)
            if actual_hash != entry.chunk_hash:
                raise DocumentVersionMismatchError(
                    "Chunk hash mismatch detected before applying streamed patch",
                    cause=self.CAUSE_CHUNK_HASH_MISMATCH,
                )

    def notify_document_closed(self, *, reason: str | None = None) -> None:
        self._chunk_manifest_cache.clear()
        document = self.editor.to_document()
        version = document.version_info()
        self._last_document_version = version
        self._last_snapshot_token = self._format_version_token(version)
        if self._cache_bus is None:
            return
        event = DocumentClosedEvent(
            document_id=version.document_id,
            version_id=version.version_id,
            content_hash=version.content_hash,
            reason=reason,
            source="document-bridge",
        )
        self._cache_bus.publish(event)

    @staticmethod
    def _derive_patch_range(spans: Sequence[tuple[int, int]], length: int) -> tuple[int, int]:
        if not spans:
            return (0, 0)
        start = min(span[0] for span in spans)
        end = min(length, max(span[1] for span in spans))
        return (start, end)

    def _expand_patch_ranges(
        self,
        document: DocumentState,
        ranges: Sequence["PatchRangePayload"],
    ) -> tuple["PatchRangePayload", ...]:
        if not ranges:
            return ()
        text = document.text or ""
        length = len(text)
        expanded: list[PatchRangePayload] = []
        for entry in ranges:
            expanded.append(self._expand_single_range(text, length, entry))
        return tuple(expanded)

    def _range_hint_from_expanded_ranges(
        self,
        text: str,
        ranges: Sequence["PatchRangePayload"],
    ) -> tuple[int, int] | None:
        if not ranges:
            return None
        length = len(text or "")
        starts = [max(0, min(entry.start, length)) for entry in ranges]
        if not starts:
            return None
        start = min(starts)
        end_candidates = [self._logical_range_end(text, entry) for entry in ranges]
        end = max(end_candidates) if end_candidates else start
        if end < start:
            end = start
        return (start, end)

    def _logical_range_end(self, text: str, entry: "PatchRangePayload") -> int:
        if not text:
            return entry.end
        length = len(text)
        end = max(0, min(entry.end, length))
        if end == 0:
            return 0
        idx = end - 1
        last_char = text[idx]
        if last_char not in {"\n", "\r"}:
            return end
        cursor = end
        while cursor < length and text[cursor] in {" ", "\t", "\r"}:
            cursor += 1
        if cursor < length and text[cursor] == "\n":
            return idx
        return end

    def _expand_single_range(
        self,
        text: str,
        length: int,
        entry: "PatchRangePayload",
    ) -> "PatchRangePayload":
        if not text or entry.chunk_id or entry.chunk_hash:
            return entry
        normalized = normalize_text_range(text, entry.start, entry.end, replacement=entry.replacement)
        if normalized.start == entry.start and normalized.end == entry.end:
            return entry
        scope_mapping = self._refresh_scope_span(entry.scope, normalized.start, normalized.end)
        scope_origin = scope_mapping.get("origin") if isinstance(scope_mapping, Mapping) else entry.scope_origin
        scope_length = scope_mapping.get("length") if isinstance(scope_mapping, Mapping) else entry.scope_length
        scope_range = (normalized.start, normalized.end) if scope_mapping is not None else entry.scope_range
        return PatchRangePayload(
            start=normalized.start,
            end=normalized.end,
            replacement=entry.replacement,
            match_text=normalized.slice_text,
            chunk_id=entry.chunk_id,
            chunk_hash=entry.chunk_hash,
            scope_origin=self._normalize_scope_origin(scope_origin) if scope_origin else entry.scope_origin,
            scope_length=self._coerce_scope_length(scope_length) if scope_length is not None else entry.scope_length,
            scope_range=scope_range,
            scope=scope_mapping if scope_mapping is not None else entry.scope,
        )

    def _range_hint_from_payload(
        self,
        ranges: Sequence["PatchRangePayload"],
        payload: Mapping[str, Any],
    ) -> tuple[int, int]:
        if ranges:
            start = min(entry.start for entry in ranges)
            end = max(entry.end for entry in ranges)
            return (start, end)
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            hint = self._coerce_text_range_hint(metadata.get("target_range"))
            if hint is not None:
                return hint
        hint = self._coerce_text_range_hint(payload.get("target_range"))
        if hint is not None:
            return hint
        return (0, 0)

    @staticmethod
    def _coerce_text_range_hint(value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        try:
            text_range = TextRange.from_value(value)
        except (TypeError, ValueError):
            return None
        return text_range.to_tuple()

    def _normalize_directive(self, directive: EditDirective | Mapping[str, Any]) -> _QueuedEdit:
        if isinstance(directive, EditDirective):
            payload: dict[str, Any] = {
                "action": directive.action,
                "content": directive.content,
                "target_range": directive.target_range.to_dict(),
            }
            if directive.rationale is not None:
                payload["rationale"] = directive.rationale
        elif isinstance(directive, Mapping):
            payload = dict(directive)
        else:  # pragma: no cover - guard against unsupported payloads
            raise TypeError("Directive must be an EditDirective or mapping.")

        payload = parse_agent_payload(payload)
        validation = validate_directive(payload)
        if not validation.ok:
            raise ValueError(validation.message)

        document = self.editor.to_document()
        action = str(payload.get("action", "")).lower()
        rationale = payload.get("rationale")
        context_version = self._extract_context_version(payload)

        match_text_value = payload.get("match_text")
        expected_text_value = payload.get("expected_text")

        if action == ActionType.PATCH.value:
            diff_text = str(payload.get("diff", ""))
            ranges = self._normalize_patch_ranges(payload.get("ranges"))
            if not diff_text.strip() and not ranges:
                raise ValueError("Patch directives must include a diff string or ranges payload")
            if not context_version:
                raise ValueError("Patch directives must include the originating document version")
            content_hash = self._extract_content_hash(payload)
            if not content_hash:
                raise ValueError("Patch directives must include the originating content_hash")
            scope_summary = self._summarize_patch_scopes(ranges)
            range_hint = self._range_hint_from_payload(ranges, payload)
            directive = EditDirective(
                action=action,
                target_range=range_hint,
                content="",
                rationale=str(rationale) if rationale is not None else None,
                diff=diff_text if diff_text.strip() else None,
                match_text=str(match_text_value) if isinstance(match_text_value, str) else None,
                expected_text=str(expected_text_value) if isinstance(expected_text_value, str) else None,
            )
            return _QueuedEdit(
                directive=directive,
                context_version=context_version,
                content_hash=content_hash,
                payload=payload,
                diff=diff_text if diff_text.strip() else None,
                ranges=ranges,
                scope_summary=scope_summary,
            )
        raise ValueError(
            "DocumentBridge only accepts patch directives; convert inline edits into patches before queueing."
        )

    def _normalize_target_range(self, target_range: Any, document: DocumentState) -> tuple[int, int]:
        if target_range is None:
            resolved: Sequence[Any] = (0, 0)
        elif isinstance(target_range, Mapping):
            resolved = (
                target_range.get("start", 0),
                target_range.get("end", 0),
            )
        elif isinstance(target_range, Sequence) and len(target_range) == 2:
            resolved = target_range
        else:
            raise ValueError("target_range must specify start and end positions")

        try:
            start = int(resolved[0])
            end = int(resolved[1])
        except (TypeError, ValueError) as exc:
            raise ValueError("target_range must contain numeric bounds") from exc

        return self._clamp_range(start, end, len(document.text))

    def _normalize_patch_ranges(self, ranges: Any) -> tuple[PatchRangePayload, ...]:
        if ranges in (None, (), []):
            return ()
        if not isinstance(ranges, Sequence):
            raise ValueError("Patch ranges must be provided as an array")
        normalized: list[PatchRangePayload] = []
        for entry in ranges:
            if not isinstance(entry, Mapping):
                raise ValueError("Patch ranges must be objects")
            if "start" not in entry or "end" not in entry:
                raise ValueError("Patch ranges require 'start' and 'end' keys")
            replacement = entry.get("replacement") or entry.get("content") or entry.get("text")
            if replacement is None:
                raise ValueError("Patch ranges must include replacement text")
            match_text = entry.get("match_text")
            if match_text is None:
                raise ValueError("Patch ranges must include match_text")
            start = int(entry.get("start", 0))
            end = int(entry.get("end", 0))
            if end < start:
                start, end = end, start
            scope_payload = entry.get("scope")
            scope_mapping = dict(scope_payload) if isinstance(scope_payload, Mapping) else None
            scope_origin = entry.get("scope_origin")
            if not isinstance(scope_origin, str) and scope_mapping is not None:
                scope_origin = scope_mapping.get("origin")
            normalized_scope_origin = self._normalize_scope_origin(scope_origin)
            scope_length = entry.get("scope_length")
            if scope_length is None and scope_mapping is not None:
                scope_length = scope_mapping.get("length")
            scope_range_payload = entry.get("scope_range")
            if scope_range_payload is None and scope_mapping is not None:
                scope_range_payload = scope_mapping.get("range")
            normalized_scope_range = self._coerce_scope_range(scope_range_payload)
            normalized_scope_length = self._coerce_scope_length(scope_length)
            if normalized_scope_length is None and normalized_scope_range is not None:
                normalized_scope_length = max(0, normalized_scope_range[1] - normalized_scope_range[0])
            chunk_id = str(entry.get("chunk_id")) if isinstance(entry.get("chunk_id"), str) else None
            chunk_hash = str(entry.get("chunk_hash")) if isinstance(entry.get("chunk_hash"), str) else None
            scope_mapping = self._normalize_scope_mapping(
                scope_mapping,
                origin=normalized_scope_origin,
                scope_range=normalized_scope_range,
                scope_length=normalized_scope_length,
            )
            self._validate_scope_requirements(
                origin=normalized_scope_origin,
                scope_range=normalized_scope_range,
                scope_length=normalized_scope_length,
                chunk_id=chunk_id,
                chunk_hash=chunk_hash,
            )
            normalized.append(
                PatchRangePayload(
                    start=start,
                    end=end,
                    replacement=str(replacement),
                    match_text=str(match_text),
                    chunk_id=chunk_id,
                    chunk_hash=chunk_hash,
                    scope_origin=normalized_scope_origin,
                    scope_length=normalized_scope_length,
                    scope_range=normalized_scope_range,
                    scope=scope_mapping,
                )
            )
        return tuple(normalized)

    @staticmethod
    def _normalize_scope_origin(origin: Any) -> str | None:
        if not isinstance(origin, str):
            return None
        token = origin.strip().lower()
        return token or None

    @staticmethod
    def _coerce_scope_length(value: Any) -> int | None:
        if value is None:
            return None
        try:
            length = int(value)
        except (TypeError, ValueError):
            return None
        return max(0, length)

    @staticmethod
    def _refresh_scope_span(
        scope: Mapping[str, Any] | None,
        start: int,
        end: int,
    ) -> Mapping[str, Any] | None:
        if not isinstance(scope, Mapping):
            return None
        updated = dict(scope)
        updated["range"] = {"start": start, "end": end}
        updated["length"] = max(0, end - start)
        return updated

    @staticmethod
    def _coerce_scope_range(value: Any) -> tuple[int, int] | None:
        if value is None:
            return None
        try:
            text_range = TextRange.from_value(value)
        except (TypeError, ValueError):
            return None
        start, end = text_range.to_tuple()
        if end < start:
            start, end = end, start
        return (start, end)

    @staticmethod
    def _normalize_scope_mapping(
        scope: Mapping[str, Any] | None,
        *,
        origin: str | None,
        scope_range: tuple[int, int] | None,
        scope_length: int | None,
    ) -> Mapping[str, Any] | None:
        has_data = any(value is not None for value in (origin, scope_range, scope_length))
        if not isinstance(scope, Mapping) and not has_data:
            return None
        mapping = dict(scope) if isinstance(scope, Mapping) else {}
        if origin:
            mapping.setdefault("origin", origin)
        if scope_range is not None:
            mapping.setdefault("range", {"start": scope_range[0], "end": scope_range[1]})
        if scope_length is not None:
            mapping.setdefault("length", scope_length)
        return mapping or None

    def _validate_scope_requirements(
        self,
        *,
        origin: str | None,
        scope_range: tuple[int, int] | None,
        scope_length: int | None,
        chunk_id: str | None,
        chunk_hash: str | None,
    ) -> None:
        if origin is None:
            raise ValueError("Patch ranges must include scope metadata (scope.origin)")
        if origin != "document" and scope_range is None:
            raise ValueError("Patch ranges must include scope_range metadata for non-document scopes")
        if origin == "chunk" and not chunk_id and not chunk_hash:
            raise ValueError("Chunk-scoped ranges must include chunk_id or chunk_hash metadata")
        if scope_range is not None and scope_length is not None:
            expected = max(0, scope_range[1] - scope_range[0])
            if expected != scope_length:
                raise ValueError("scope_length must match the provided scope_range span")

    def _summarize_patch_scopes(self, ranges: Sequence[PatchRangePayload]) -> Mapping[str, Any] | None:
        if not ranges:
            return None
        origins: set[str] = set()
        lengths: list[int] = []
        for entry in ranges:
            origin = entry.scope_origin
            if not origin and isinstance(entry.scope, Mapping):
                origin = self._normalize_scope_origin(entry.scope.get("origin"))
            if origin:
                origins.add(origin)
            length = entry.scope_length
            if length is None and entry.scope_range is not None:
                length = max(0, entry.scope_range[1] - entry.scope_range[0])
            if length is None:
                length = max(0, entry.end - entry.start)
            lengths.append(length)
        if not origins:
            origins.add("unknown")
        origin_summary = origins.pop() if len(origins) == 1 else "mixed"
        summary: dict[str, Any] = {"origin": origin_summary}
        if lengths:
            summary["length"] = sum(lengths)
        range_starts = [entry.scope_range[0] if entry.scope_range else entry.start for entry in ranges]
        range_ends = [entry.scope_range[1] if entry.scope_range else entry.end for entry in ranges]
        if range_starts and range_ends:
            summary["range"] = {
                "start": min(range_starts),
                "end": max(range_ends),
            }
        return summary

    @staticmethod
    def _attach_scope_metadata(target: dict[str, Any], scope_summary: Mapping[str, Any] | None) -> None:
        if not isinstance(scope_summary, Mapping):
            return
        origin = scope_summary.get("origin")
        if isinstance(origin, str) and origin.strip():
            target["scope_origin"] = origin.strip()
        length = scope_summary.get("length")
        try:
            if length is not None:
                target["scope_length"] = max(0, int(length))
        except (TypeError, ValueError):
            pass
        range_payload = scope_summary.get("range")
        if isinstance(range_payload, Mapping):
            start = range_payload.get("start")
            end = range_payload.get("end")
            try:
                if start is not None and end is not None:
                    target["scope_range"] = {"start": int(start), "end": int(end)}
            except (TypeError, ValueError):
                pass

    def _extract_context_version(self, payload: Mapping[str, Any]) -> Optional[str]:
        for key in ("document_version", "snapshot_version", "version", "document_digest"):
            token = payload.get(key)
            if token is None:
                continue
            token_str = str(token).strip()
            if token_str:
                return token_str
        return None

    @staticmethod
    def _extract_content_hash(payload: Mapping[str, Any]) -> Optional[str]:
        token = payload.get("content_hash")
        if token is None:
            return None
        token_str = str(token).strip()
        return token_str or None

    def _format_version_token(self, version: DocumentVersion) -> str:
        """Format version token with tab_id prefix.
        
        Format: 'tab_id:document_id:version_id:content_hash'
        If no tab_id is set, uses 'default' as placeholder.
        """
        tab_id = self._tab_id or "default"
        return f"{tab_id}:{version.document_id}:{version.version_id}:{version.content_hash}"

    def _register_with_version_manager(self, version: DocumentVersion) -> None:
        """Register the current tab/document with the global version manager.
        
        This ensures that version tokens generated by the bridge can be
        validated by write tools that check the version manager.
        """
        tab_id = self._tab_id or "default"
        try:
            vm = get_version_manager()
            vm.register_tab(tab_id, version.document_id, version.content_hash)
        except Exception:
            # Don't fail snapshot generation if version manager has issues
            _LOGGER.debug("Failed to register tab with version manager", exc_info=True)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_version_current(document: DocumentState, expected: str) -> bool:
        """Check if document version matches expected token.
        
        Handles both 3-part (document_id:version_id:hash) and 
        4-part (tab_id:document_id:version_id:hash) tokens.
        """
        signature = document.version_signature()
        # If expected has 4 parts (with tab_id prefix), strip the prefix
        parts = expected.split(":")
        if len(parts) >= 4:
            # 4-part format: tab_id:document_id:version_id:hash
            expected = ":".join(parts[1:])  # Remove tab_id prefix
        return signature == expected

    @staticmethod
    def _clamp_range(start: int, end: int, length: int) -> tuple[int, int]:
        start = max(0, min(start, length))
        end = max(0, min(end, length))
        if end < start:
            start, end = end, start
        return start, end

    @staticmethod
    def _summarize_diff(before: str, after: str) -> str:
        delta = len(after) - len(before)
        if delta == 0:
            return "Δ0"
        sign = "+" if delta > 0 else "-"
        return f"{sign}{abs(delta)} chars"

    @staticmethod
    def _parse_chunk_bounds(chunk_id: str | None) -> tuple[int, int] | None:
        if not chunk_id:
            return None
        parts = chunk_id.split(":")
        if len(parts) < 4:
            return None
        try:
            start = int(parts[-2])
            end = int(parts[-1])
        except ValueError:
            return None
        return (start, end)

    @staticmethod
    def _compute_line_start_offsets(text: str) -> list[int]:
        offsets = [0]
        if not text:
            return offsets
        cursor = 0
        for segment in text.splitlines(keepends=True):
            cursor += len(segment)
            offsets.append(cursor)
        return offsets

    def _publish_document_changed(
        self,
        version: DocumentVersion,
        *,
        spans: Sequence[tuple[int, int]] | tuple[tuple[int, int], ...] | None,
        source: str,
    ) -> None:
        if self._cache_bus is None:
            return
        edited = tuple(spans or version.edited_ranges)
        event = DocumentChangedEvent(
            document_id=version.document_id,
            version_id=version.version_id,
            content_hash=version.content_hash,
            edited_ranges=edited,
            source=source,
        )
        self._cache_bus.publish(event)

    def _resolve_chunk_manifest(
        self,
        document: DocumentState,
        window: Mapping[str, Any],
        chunk_profile: str,
        *,
        focus_span: TextRange | None,
    ) -> dict[str, Any] | None:
        cache_key = self._manifest_cache_key(window, chunk_profile, document.content_hash)
        cached = self._chunk_manifest_cache.get(cache_key)
        if cached is not None and cached.get("content_hash") == document.content_hash:
            manifest = deepcopy(cached)
            manifest["cache_hit"] = True
            return manifest

        bounded_focus = (focus_span or TextRange.zero()).clamp(lower=0, upper=len(document.text))
        manifest = self._build_chunk_manifest(document, window, chunk_profile, focus_span=bounded_focus)
        if manifest is None:
            return None
        self._store_manifest_cache_entry(cache_key, manifest)
        result = deepcopy(manifest)
        result["cache_hit"] = False
        return result

    def _store_manifest_cache_entry(self, key: str, manifest: dict[str, Any]) -> None:
        limit = 6
        self._chunk_manifest_cache[key] = manifest
        self._chunk_manifest_cache.move_to_end(key)
        while len(self._chunk_manifest_cache) > limit:
            self._chunk_manifest_cache.popitem(last=False)

    def _build_chunk_manifest(
        self,
        document: DocumentState,
        window: Mapping[str, Any],
        chunk_profile: str,
        *,
        focus_span: TextRange,
    ) -> dict[str, Any] | None:
        start = int(window.get("start", 0))
        end = int(window.get("end", start))
        length = max(0, end - start)
        if length <= 0:
            return None
        config = self._chunk_profile_config(chunk_profile)
        chunk_chars = max(256, config["chunk_chars"])
        overlap = max(0, min(config["chunk_overlap"], chunk_chars // 2))
        text = document.text
        cursor = start
        chunks: list[dict[str, Any]] = []
        focus_start = focus_span.start
        focus_end = focus_span.end
        focus_length = focus_span.length
        version = document.version_info()
        while cursor < end:
            chunk_end = min(end, cursor + chunk_chars)
            if chunk_end <= cursor:
                break
            segment = text[cursor:chunk_end]
            if not segment:
                break
            overlap_flag = focus_length > 0 and not (chunk_end <= focus_start or cursor >= focus_end)
            chunks.append(
                {
                    "id": f"chunk:{document.document_id}:{cursor}:{chunk_end}",
                    "hash": self._hash_text(segment),
                    "start": cursor,
                    "end": chunk_end,
                    "length": chunk_end - cursor,
                    "span_overlap": overlap_flag,
                    "outline_pointer_id": None,
                }
            )
            if chunk_end >= end:
                break
            next_cursor = chunk_end - overlap if overlap else chunk_end
            if next_cursor <= cursor:
                next_cursor = chunk_end
            cursor = next_cursor

        manifest = {
            "document_id": document.document_id,
            "chunk_profile": chunk_profile,
            "chunk_chars": chunk_chars,
            "chunk_overlap": overlap,
            "window": {
                "start": start,
                "end": end,
                "length": length,
            },
            "chunks": chunks,
            "content_hash": document.content_hash,
            "version": self._format_version_token(version),
            "cache_key": self._manifest_cache_key(window, chunk_profile, document.content_hash),
            "generated_at": time.time(),
        }
        return manifest

    def _manifest_cache_key(self, window: Mapping[str, Any], profile: str, content_hash: str) -> str:
        start = int(window.get("start", 0))
        end = int(window.get("end", start))
        return f"{content_hash}:{profile}:{start}:{end}"

    def _chunk_profile_config(self, profile: str) -> dict[str, int]:
        presets = {
            "auto": {"chunk_chars": 2048, "chunk_overlap": 256},
            "prose": {"chunk_chars": 1792, "chunk_overlap": 192},
            "code": {"chunk_chars": 1536, "chunk_overlap": 160},
            "notes": {"chunk_chars": 1024, "chunk_overlap": 128},
        }
        return presets.get(profile, presets["auto"])

    def _normalize_chunk_profile(self, profile: str | None) -> str:
        if not profile:
            return "auto"
        normalized = str(profile).strip().lower()
        if normalized in {"auto", "prose", "code", "notes"}:
            return normalized
        return "auto"

    def _resolve_window(
        self,
        window: Mapping[str, Any] | str | None,
        *,
        length: int,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        default_kind = "document"
        kind = default_kind
        padding = 2048
        max_chars = 8192
        start_override: int | None = None
        end_override: int | None = None
        focus_override: TextRange | None = None
        if isinstance(window, str):
            kind = window.strip().lower() or kind
        elif isinstance(window, Mapping):
            raw_kind = window.get("kind")
            if isinstance(raw_kind, str) and raw_kind.strip():
                kind = raw_kind.strip().lower()
            if "padding" in window:
                padding = max(0, int(self._safe_int(window.get("padding"), fallback=padding)))
            if "max_chars" in window:
                max_chars = max(0, int(self._safe_int(window.get("max_chars"), fallback=max_chars)))
            if "start" in window:
                start_override = int(self._safe_int(window.get("start"), fallback=0))
            if "end" in window:
                end_override = int(self._safe_int(window.get("end"), fallback=length))
            token_override = window.get("max_tokens")
            if token_override is not None:
                max_tokens = int(self._safe_int(token_override, fallback=max_tokens or 0)) or max_tokens
            focus_payload = window.get("target_span") or window.get("focus_span") or window.get("span")
            if focus_payload is not None:
                try:
                    focus_override = TextRange.from_value(focus_payload).clamp(lower=0, upper=length)
                except Exception:
                    focus_override = None

        token_cap = self._estimate_chars_from_tokens(max_tokens)
        span_cap = max_chars if max_chars else None
        if token_cap is not None:
            span_cap = min(span_cap, token_cap) if span_cap is not None else token_cap

        focus_hint = focus_override or self._focus_span_hint(length)
        requested_kind = kind
        if kind in {"document", "full", "entire"}:
            start = 0
            end = length
            if span_cap is not None and span_cap < (end - start):
                end = min(length, start + span_cap)
        elif start_override is not None or end_override is not None:
            start, end = self._clamp_range(start_override or 0, end_override or length, length)
            kind = "range"
        else:
            span = focus_hint.length if focus_hint.length > 0 else padding * 2 or 1024
            if span_cap is not None:
                span = min(span, span_cap)
            if span <= 0:
                span = span_cap or length or 0
            if span <= 0:
                span = length
            half = max(1, span // 2)
            center = focus_hint.end if focus_hint.length > 0 else focus_hint.start
            start = max(0, min(center - half, length))
            end = min(length, start + span)
            if end - start < span and start > 0:
                start = max(0, end - span)
            kind = "focus"

        includes_full = start == 0 and end == length
        return {
            "kind": kind,
            "requested_kind": requested_kind,
            "defaulted": window is None,
            "start": start,
            "end": end,
            "length": max(0, end - start),
            "padding": padding,
            "max_chars": span_cap,
            "max_tokens": max_tokens,
            "source_length": length,
            "includes_full_document": includes_full,
        }

    def _focus_span_hint(self, length: int) -> TextRange:
        context = self._last_edit_context
        if context is None:
            return TextRange.zero()
        try:
            return context.target_range.clamp(lower=0, upper=length)
        except Exception:
            return TextRange.zero()

    @staticmethod
    def _estimate_chars_from_tokens(max_tokens: int | None) -> int | None:
        if max_tokens is None:
            return None
        try:
            tokens = int(max_tokens)
        except (TypeError, ValueError):
            return None
        if tokens <= 0:
            return None
        return max(512, tokens * 4)

    @staticmethod
    def _safe_int(value: Any, *, fallback: int | None) -> int | None:
        try:
            return int(value) if value is not None else fallback
        except (TypeError, ValueError):
            return fallback

