"""Document bridge connecting AI directives to the editor widget."""

from __future__ import annotations

import hashlib
import logging
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Mapping, Optional, Protocol, Sequence, TypeVar

from ..chat.commands import parse_agent_payload, validate_directive
from ..chat.message_model import EditDirective
from ..editor.document_model import DocumentState


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


@dataclass(slots=True)
class _QueuedEdit:
    """Internal representation of a validated directive awaiting execution."""

    directive: EditDirective
    context_version: Optional[str] = None
    payload: Optional[Mapping[str, Any]] = None


@dataclass(slots=True)
class EditContext:
    """Details about the most recently applied directive."""

    action: str
    target_range: tuple[int, int]
    replaced_text: str
    content: str


class DocumentBridge:
    """Orchestrates safe document snapshots, conflict detection, and queued edits."""

    def __init__(self, *, editor: EditorAdapter, main_thread_executor: Optional[Executor] = None) -> None:
        self.editor = editor
        self._pending_edits: Deque[_QueuedEdit] = deque()
        self._draining = False
        self._last_diff: Optional[str] = None
        self._last_snapshot_token: Optional[str] = None
        self._last_edit_context: Optional[EditContext] = None
        self._main_thread_executor = main_thread_executor
        self._edit_listeners: list[EditAppliedListener] = []

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
        token = self._compute_document_token(document)
        snapshot["version"] = token
        self._last_snapshot_token = token
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
    def last_edit_context(self) -> Optional[EditContext]:
        """Expose metadata about the most recently applied edit."""

        return self._last_edit_context

    def add_edit_listener(self, listener: EditAppliedListener) -> None:
        """Register a callback fired after each successful directive."""

        self._edit_listeners.append(listener)

    def remove_edit_listener(self, listener: EditAppliedListener) -> None:
        """Remove a previously registered edit listener if present."""

        try:
            self._edit_listeners.remove(listener)
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
            raise RuntimeError("Directive is stale relative to the current document state")

        before_text = document_before.text
        try:
            updated_state = self._execute_on_main_thread(lambda: self.editor.apply_ai_edit(queued.directive))
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.exception("Failed to apply directive: action=%s", queued.directive.action)
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
        self._last_snapshot_token = self._compute_document_token(updated_state)
        _LOGGER.debug(
            "Applied directive action=%s range=%s diff=%s",
            queued.directive.action,
            queued.directive.target_range,
            self._last_diff,
        )
        self._notify_listeners(queued.directive, updated_state)

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
        start, end = self._normalize_target_range(payload.get("target_range"), document)
        rationale = payload.get("rationale")
        context_version = self._extract_context_version(payload)

        normalized = EditDirective(
            action=str(payload["action"]).lower(),
            target_range=(start, end),
            content=str(payload["content"]),
            rationale=str(rationale) if rationale is not None else None,
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
    def _compute_document_token(document: DocumentState) -> str:
        return hashlib.sha1(document.text.encode("utf-8")).hexdigest()

    @staticmethod
    def _is_version_current(document: DocumentState, expected: str) -> bool:
        return DocumentBridge._compute_document_token(document) == expected

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

