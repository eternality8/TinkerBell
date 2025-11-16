"""Document bridge connecting AI directives to the editor widget."""

from __future__ import annotations

import hashlib
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Mapping, Optional, Protocol, Sequence, TypeVar

from ..ai.memory.cache_bus import (
    DocumentCacheBus,
    DocumentChangedEvent,
    DocumentClosedEvent,
    get_document_cache_bus,
)
from ..chat.commands import ActionType, parse_agent_payload, validate_directive
from ..chat.message_model import EditDirective
from ..editor.document_model import DocumentState, DocumentVersion
from ..editor.patches import PatchApplyError, PatchResult, apply_unified_diff


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

    def apply_ai_edit(self, directive: EditDirective) -> DocumentState:
        ...

    def apply_patch_result(self, result: PatchResult) -> DocumentState:
        ...


@dataclass(slots=True)
class _QueuedEdit:
    """Internal representation of a validated directive awaiting execution."""

    directive: EditDirective
    context_version: Optional[str] = None
    payload: Optional[Mapping[str, Any]] = None
    diff: Optional[str] = None


@dataclass(slots=True)
class EditContext:
    """Details about the most recently applied directive."""

    action: str
    target_range: tuple[int, int]
    replaced_text: str
    content: str
    diff: Optional[str] = None
    spans: tuple[tuple[int, int], ...] = ()


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


class DocumentVersionMismatchError(RuntimeError):
    """Raised when an edit references a stale document version."""


class DocumentBridge:
    """Orchestrates safe document snapshots, conflict detection, and queued edits."""

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
        self._patch_metrics = PatchMetrics()
        self._cache_bus = cache_bus or get_document_cache_bus()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_snapshot(self, *, delta_only: bool = False) -> dict:
        """Return a document snapshot enriched with version metadata."""

        document = self.editor.to_document()
        snapshot = document.snapshot(delta_only=delta_only)
        start, end = self._clamp_range(*document.selection.as_tuple(), len(document.text))
        snapshot["selection_text"] = document.text[start:end]
        snapshot["length"] = len(document.text)
        snapshot["line_offsets"] = self._compute_line_offsets(document.text)
        if snapshot["selection_text"]:
            snapshot["selection_hash"] = self._hash_text(snapshot["selection_text"])
        version = document.version_info()
        snapshot.setdefault("document_id", version.document_id)
        snapshot.setdefault("version_id", version.version_id)
        snapshot.setdefault("content_hash", version.content_hash)
        token = self._format_version_token(version)
        snapshot["version"] = token
        self._last_snapshot_token = token
        self._last_document_version = version
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

    def remove_failure_listener(self, listener: Callable[[EditDirective, str], None]) -> None:
        """Detach a previously registered failure listener if present."""

        try:
            self._failure_listeners.remove(listener)
        except ValueError:  # pragma: no cover - defensive guard
            pass

    def set_main_thread_executor(self, executor: Optional[Executor]) -> None:
        """Configure a callable used to marshal edits onto the UI thread."""

        self._main_thread_executor = executor

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
        if queued.context_version and not self._is_version_current(document_before, queued.context_version):
            message = "Directive is stale relative to the current document state"
            if queued.directive.action == ActionType.PATCH.value:
                self._patch_metrics.record_conflict()
            self._record_failure(queued.directive, message)
            raise DocumentVersionMismatchError(message)

        if queued.directive.action == ActionType.PATCH.value:
            self._apply_patch_directive(queued, document_before)
            return

        before_text = document_before.text
        try:
            updated_state = self._execute_on_main_thread(lambda: self.editor.apply_ai_edit(queued.directive))
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.exception("Failed to apply directive: action=%s", queued.directive.action)
            self._record_failure(queued.directive, str(exc))
            raise RuntimeError(f"Failed to apply directive: {queued.directive.action}") from exc

        start, end = queued.directive.target_range
        replaced_segment = before_text[start:end]
        self._last_edit_context = EditContext(
            action=queued.directive.action,
            target_range=(start, end),
            replaced_text=replaced_segment,
            content=queued.directive.content,
        )
        self._last_diff = self._summarize_diff(before_text, updated_state.text)
        version = updated_state.version_info()
        self._last_snapshot_token = self._format_version_token(version)
        self._last_document_version = version
        _LOGGER.debug(
            "Applied directive action=%s range=%s diff=%s",
            queued.directive.action,
            queued.directive.target_range,
            self._last_diff,
        )
        self._notify_listeners(queued.directive, updated_state)
        self._publish_document_changed(
            version,
            spans=(queued.directive.target_range,),
            source=queued.directive.action,
        )

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

    def _record_failure(self, directive: EditDirective, message: str) -> None:
        for listener in list(self._failure_listeners):
            try:
                listener(directive, message)
            except Exception:  # pragma: no cover - defensive guard
                _LOGGER.exception("Edit failure listener raised")

    def _apply_patch_directive(self, queued: _QueuedEdit, document_before: DocumentState) -> None:
        diff_text = queued.diff or queued.directive.diff or ""
        if not diff_text:
            raise RuntimeError("Patch directive missing diff payload")

        start_time = time.perf_counter()
        try:
            patch_result = apply_unified_diff(document_before.text, diff_text)
        except PatchApplyError as exc:
            self._patch_metrics.record_conflict()
            summary = f"failed: {exc.reason}"
            self._last_diff = summary
            self._record_failure(queued.directive, summary)
            raise RuntimeError(f"Patch application failed: {exc}") from exc

        updated_state = self._execute_on_main_thread(lambda: self.editor.apply_patch_result(patch_result))
        target_range = self._derive_patch_range(patch_result.spans, len(updated_state.text))
        self._last_edit_context = EditContext(
            action=queued.directive.action,
            target_range=target_range,
            replaced_text="",
            content="",
            diff=diff_text,
            spans=patch_result.spans,
        )
        self._last_diff = patch_result.summary
        version = updated_state.version_info(edited_ranges=patch_result.spans)
        self._last_snapshot_token = self._format_version_token(version)
        self._last_document_version = version
        elapsed = time.perf_counter() - start_time
        self._patch_metrics.record_success(elapsed)

        _LOGGER.debug("Applied patch directive diff=%s spans=%s", patch_result.summary, patch_result.spans)
        self._notify_listeners(queued.directive, updated_state)
        self._publish_document_changed(
            version,
            spans=patch_result.spans,
            source=queued.directive.action,
        )

    def notify_document_closed(self, *, reason: str | None = None) -> None:
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

    def _normalize_directive(self, directive: EditDirective | Mapping[str, Any]) -> _QueuedEdit:
        if isinstance(directive, EditDirective):
            payload: dict[str, Any] = {
                "action": directive.action,
                "content": directive.content,
                "target_range": directive.target_range,
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
        action = str(payload.get("action", ""))
        rationale = payload.get("rationale")
        context_version = self._extract_context_version(payload)

        if action == ActionType.PATCH.value:
            diff_text = str(payload.get("diff", ""))
            if not diff_text.strip():
                raise ValueError("Patch directives must include a non-empty diff payload")
            if not context_version:
                raise ValueError("Patch directives must include the originating document version")
            directive = EditDirective(
                action=action,
                target_range=(0, 0),
                content="",
                rationale=str(rationale) if rationale is not None else None,
                diff=diff_text,
            )
            return _QueuedEdit(directive=directive, context_version=context_version, payload=payload, diff=diff_text)

        start, end = self._normalize_target_range(payload.get("target_range"), document)
        fingerprint = payload.get("selection_fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            actual = self._hash_text(document.text[start:end])
            if actual != fingerprint:
                raise RuntimeError("Selection fingerprint mismatch; refresh the snapshot before editing")

        normalized = EditDirective(
            action=action,
            target_range=(start, end),
            content=str(payload["content"]),
            rationale=str(rationale) if rationale is not None else None,
            selection_fingerprint=str(fingerprint) if fingerprint else None,
        )
        return _QueuedEdit(directive=normalized, context_version=context_version, payload=payload)

    def _normalize_target_range(self, target_range: Any, document: DocumentState) -> tuple[int, int]:
        selection = document.selection.as_tuple()
        resolved: Sequence[Any]
        if target_range is None:
            resolved = selection
        elif isinstance(target_range, Mapping):
            resolved = (
                target_range.get("start", selection[0]),
                target_range.get("end", selection[1]),
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
    def _format_version_token(version: DocumentVersion) -> str:
        return f"{version.document_id}:{version.version_id}:{version.content_hash}"

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_version_current(document: DocumentState, expected: str) -> bool:
        return document.version_signature() == expected

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
    def _compute_line_offsets(text: str) -> list[int]:
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

