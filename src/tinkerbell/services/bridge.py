"""Document bridge connecting AI directives to the editor widget."""

from __future__ import annotations

import hashlib
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
from ..chat.commands import ActionType, parse_agent_payload, validate_directive
from ..chat.message_model import EditDirective
from ..editor.document_model import DocumentState, DocumentVersion
from ..editor.patches import PatchApplyError, PatchResult, RangePatch, apply_streamed_ranges, apply_unified_diff
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
    ranges: tuple["PatchRangePayload", ...] = ()


@dataclass(slots=True)
class PatchRangePayload:
    """Normalized representation of a streamed patch range payload."""

    start: int
    end: int
    replacement: str
    match_text: str
    chunk_id: str | None = None
    chunk_hash: str | None = None


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

    PATCH_EVENT_NAME = "patch.apply"

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
        self._chunk_manifest_cache: "OrderedDict[str, dict[str, Any]]" = OrderedDict()

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
        start, end = self._clamp_range(*document.selection.as_tuple(), len(document.text))
        snapshot["selection_text"] = document.text[start:end]
        snapshot["length"] = len(document.text)
        snapshot["line_offsets"] = self._compute_line_offsets(document.text)
        if snapshot["selection_text"]:
            snapshot["selection_hash"] = self._hash_text(snapshot["selection_text"])
        window_range = self._resolve_window(
            window,
            selection=(start, end),
            length=len(document.text),
            max_tokens=max_tokens,
        )
        chunk_profile_name = self._normalize_chunk_profile(chunk_profile)
        snapshot["window"] = dict(window_range)
        snapshot["text_range"] = {"start": window_range["start"], "end": window_range["end"]}
        if not include_text:
            snapshot["text"] = ""
        elif not delta_only and not window_range["includes_full_document"]:
            snapshot["text"] = document.text[window_range["start"] : window_range["end"]]
        chunk_manifest = self._resolve_chunk_manifest(document, window_range, chunk_profile_name)
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
        pre_edit_snapshot = deepcopy(document_before)
        if queued.context_version and not self._is_version_current(document_before, queued.context_version):
            message = "Directive is stale relative to the current document state"
            if queued.directive.action == ActionType.PATCH.value:
                self._patch_metrics.record_conflict()
                self._emit_patch_event(
                    status="stale",
                    version=document_before.version_info(),
                    diff_summary=None,
                    reason=message,
                    range_count=len(queued.ranges) if queued.ranges else None,
                    streamed=bool(queued.ranges),
                )
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
        self._notify_listeners(queued.directive, pre_edit_snapshot)
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
        use_ranges = bool(queued.ranges)
        if not diff_text and not use_ranges:
            raise RuntimeError("Patch directive missing diff payload")

        start_time = time.perf_counter()
        pre_edit_snapshot = deepcopy(document_before)
        pre_version = document_before.version_info()
        try:
            if use_ranges:
                patch_result = self._apply_range_payloads(queued.ranges, document_before)
            else:
                patch_result = apply_unified_diff(document_before.text, diff_text)
        except PatchApplyError as exc:
            self._patch_metrics.record_conflict()
            reason = getattr(exc, "reason", str(exc))
            summary = f"failed: {reason}"
            self._last_diff = summary
            self._record_failure(queued.directive, summary)
            self._emit_patch_event(
                status="conflict",
                version=pre_version,
                diff_summary=None,
                reason=reason,
                range_count=len(queued.ranges) if use_ranges else None,
                streamed=use_ranges,
            )
            raise RuntimeError(f"Patch application failed: {exc}") from exc

        updated_state = self._execute_on_main_thread(lambda: self.editor.apply_patch_result(patch_result))
        target_range = self._derive_patch_range(patch_result.spans, len(updated_state.text))
        self._last_edit_context = EditContext(
            action=queued.directive.action,
            target_range=target_range,
            replaced_text="",
            content="",
            diff=diff_text if not use_ranges else None,
            spans=patch_result.spans,
        )
        self._last_diff = patch_result.summary
        version = updated_state.version_info(edited_ranges=patch_result.spans)
        self._last_snapshot_token = self._format_version_token(version)
        self._last_document_version = version
        elapsed = time.perf_counter() - start_time
        self._patch_metrics.record_success(elapsed)
        self._emit_patch_event(
            status="success",
            version=version,
            diff_summary=patch_result.summary,
            duration_ms=elapsed * 1000.0,
            range_count=len(patch_result.spans),
            streamed=use_ranges,
        )

        _LOGGER.debug("Applied patch directive diff=%s spans=%s", patch_result.summary, patch_result.spans)
        self._notify_listeners(queued.directive, pre_edit_snapshot)
        self._publish_document_changed(
            version,
            spans=patch_result.spans,
            source=queued.directive.action,
        )

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
            )
            for entry in ranges
        ]
        return apply_streamed_ranges(document_before.text, range_specs)

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

        match_text_value = payload.get("match_text")
        expected_text_value = payload.get("expected_text")

        if action == ActionType.PATCH.value:
            diff_text = str(payload.get("diff", ""))
            ranges = self._normalize_patch_ranges(payload.get("ranges"))
            if not diff_text.strip() and not ranges:
                raise ValueError("Patch directives must include a diff string or ranges payload")
            if not context_version:
                raise ValueError("Patch directives must include the originating document version")
            directive = EditDirective(
                action=action,
                target_range=(0, 0),
                content="",
                rationale=str(rationale) if rationale is not None else None,
                diff=diff_text if diff_text.strip() else None,
                match_text=str(match_text_value) if isinstance(match_text_value, str) else None,
                expected_text=str(expected_text_value) if isinstance(expected_text_value, str) else None,
            )
            return _QueuedEdit(
                directive=directive,
                context_version=context_version,
                payload=payload,
                diff=diff_text if diff_text.strip() else None,
                ranges=ranges,
            )

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
            match_text=str(match_text_value) if isinstance(match_text_value, str) else None,
            expected_text=str(expected_text_value) if isinstance(expected_text_value, str) else None,
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
            normalized.append(
                PatchRangePayload(
                    start=start,
                    end=end,
                    replacement=str(replacement),
                    match_text=str(match_text),
                    chunk_id=str(entry.get("chunk_id")) if isinstance(entry.get("chunk_id"), str) else None,
                    chunk_hash=str(entry.get("chunk_hash")) if isinstance(entry.get("chunk_hash"), str) else None,
                )
            )
        return tuple(normalized)

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

    def _resolve_chunk_manifest(
        self,
        document: DocumentState,
        window: Mapping[str, Any],
        chunk_profile: str,
    ) -> dict[str, Any] | None:
        cache_key = self._manifest_cache_key(window, chunk_profile, document.content_hash)
        cached = self._chunk_manifest_cache.get(cache_key)
        if cached is not None and cached.get("content_hash") == document.content_hash:
            manifest = deepcopy(cached)
            manifest["cache_hit"] = True
            return manifest

        manifest = self._build_chunk_manifest(document, window, chunk_profile)
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
        selection = window.get("selection") or {}
        selection_start = int(selection.get("start", start))
        selection_end = int(selection.get("end", selection_start))
        version = document.version_info()
        while cursor < end:
            chunk_end = min(end, cursor + chunk_chars)
            if chunk_end <= cursor:
                break
            segment = text[cursor:chunk_end]
            if not segment:
                break
            overlap_flag = not (chunk_end <= selection_start or cursor >= selection_end)
            chunks.append(
                {
                    "id": f"chunk:{document.document_id}:{cursor}:{chunk_end}",
                    "hash": self._hash_text(segment),
                    "start": cursor,
                    "end": chunk_end,
                    "length": chunk_end - cursor,
                    "selection_overlap": overlap_flag,
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
                "selection": {
                    "start": selection_start,
                    "end": selection_end,
                },
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
        selection: tuple[int, int],
        length: int,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        default_kind = "selection"
        kind = default_kind
        padding = 2048
        max_chars = 8192
        start_override: int | None = None
        end_override: int | None = None
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

        token_cap = self._estimate_chars_from_tokens(max_tokens)
        span_cap = max_chars if max_chars else None
        if token_cap is not None:
            span_cap = min(span_cap, token_cap) if span_cap is not None else token_cap

        selection_start, selection_end = selection
        selection_start, selection_end = self._clamp_range(selection_start, selection_end, length)
        requested_kind = kind
        if kind in {"document", "full", "entire"}:
            start = 0
            end = length
            kind = "document"
        elif start_override is not None or end_override is not None:
            start, end = self._clamp_range(start_override or 0, end_override or length, length)
            kind = "range"
        else:
            center = selection_end if selection_end > selection_start else selection_start
            span = selection_end - selection_start
            if span <= 0:
                span = padding * 2 or 1024
            start = max(0, selection_start - padding)
            end = min(length, selection_end + padding if selection_end > selection_start else center + padding)
            if end <= start:
                end = min(length, start + span)
            if span_cap is not None and (end - start) > span_cap:
                half = span_cap // 2 or 1
                start = max(0, center - half)
                end = min(length, start + span_cap)
                if end - start < span_cap and start > 0:
                    start = max(0, end - span_cap)

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
            "selection": {"start": selection_start, "end": selection_end},
            "source_length": length,
            "includes_full_document": includes_full,
        }

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

