"""Coordinates AI turn execution and streaming events for MainWindow."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..ai.orchestration.editor_lock import (
    EditorLockManager,
    LockReason,
    LockSession,
)
from ..chat.message_model import ChatMessage
from .tool_trace_presenter import ToolTracePresenter

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AITurnCoordinator:
    """Runs AI turns and fans out streaming events to the chat panel."""

    controller_resolver: Callable[[], Any | None]
    chat_panel: Any
    review_controller: Any
    telemetry_controller: Any
    status_updater: Callable[[str], None]
    failure_handler: Callable[[Exception], None]
    response_finalizer: Callable[[str], None]
    tool_trace_presenter: ToolTracePresenter
    stream_state_setter: Callable[[bool], None]
    stream_text_coercer: Callable[[Any], str]
    lock_manager: EditorLockManager | None = None

    _ai_stream_active: bool = False
    _active_lock_session: LockSession | None = field(default=None, repr=False)

    async def run_ai_turn(
        self,
        *,
        prompt: str,
        snapshot: Mapping[str, Any],
        metadata: Mapping[str, Any],
        history: Sequence[Mapping[str, str]] | None,
    ) -> None:
        controller = self.controller_resolver()
        if controller is None:
            return
        reset_chunk_flow = getattr(self.telemetry_controller, "reset_chunk_flow_state", None)
        if callable(reset_chunk_flow):
            reset_chunk_flow()
        self._ai_stream_active = False
        self.stream_state_setter(False)
        self.tool_trace_presenter.reset()
        normalized_metadata = self._normalize_metadata(metadata)
        
        # Acquire editor lock to prevent user edits during AI turn
        self._acquire_editor_lock()
        
        self.status_updater("AI thinking…")
        try:
            result = await controller.run_chat(
                prompt,
                snapshot,
                metadata=normalized_metadata,
                history=history,
                on_event=self._handle_stream_event,
            )
        except Exception as exc:  # pragma: no cover - runtime path
            self._release_editor_lock()
            self.review_controller.finalize_pending_turn_review(success=False)
            self.failure_handler(exc)
            return

        # Handle both ChatResult objects and legacy dict responses
        if hasattr(result, 'response'):
            # ChatResult object
            tool_records = getattr(result, 'tool_calls', None)
            trace_compaction = None  # ChatResult doesn't have trace_compaction
            response_text = (getattr(result, 'response', '') or '').strip() or "The AI did not return any content."
        else:
            # Legacy dict response
            payload = result or {}
            tool_records = payload.get("tool_calls")
            trace_compaction = payload.get("trace_compaction")
            response_text = payload.get("response", "").strip() or "The AI did not return any content."
        self.tool_trace_presenter.annotate_compaction(tool_records)
        self.telemetry_controller.set_compaction_stats(trace_compaction)
        self.response_finalizer(response_text)
        self.telemetry_controller.refresh_context_usage_status()
        document_id = snapshot.get("document_id") if isinstance(snapshot, Mapping) else None
        document_label = None
        if isinstance(snapshot, Mapping):
            label_source = snapshot.get("path") or snapshot.get("tab_title")
            if label_source:
                document_label = str(label_source)
        refresh_analysis = getattr(self.telemetry_controller, "refresh_analysis_state", None)
        if callable(refresh_analysis):
            refresh_analysis(
                document_id=str(document_id) if document_id else None,
                document_label=document_label,
            )
        
        # Release editor lock after AI turn completes
        self._release_editor_lock()
        
        _LOGGER.debug("AI turn completed, calling finalize_pending_turn_review")
        self.status_updater("AI response ready")
        self.review_controller.finalize_pending_turn_review(success=True)

    def _handle_stream_event(self, event: Any) -> None:
        self._process_stream_event(event)

    def _process_stream_event(self, event: Any) -> None:
        event_type = getattr(event, "type", "") or ""
        if event_type in {"content.delta", "refusal.delta"}:
            chunk = self.stream_text_coercer(getattr(event, "content", None))
            if chunk:
                self._chat_append(chunk, streaming=True)
                self._update_stream_state(True)
            return

        if event_type in {"content.done", "refusal.done"}:
            if self._ai_stream_active:
                return
            chunk = self.stream_text_coercer(getattr(event, "content", None))
            if chunk:
                self._chat_append(chunk, streaming=True)
                self._update_stream_state(True)
            return

        if event_type == "tool_calls.function.arguments.delta":
            self.tool_trace_presenter.record_arguments_delta(event)
            return

        if event_type == "tool_calls.function.arguments.done":
            self.tool_trace_presenter.finalize_arguments(event)
            return

        if event_type == "tool_calls.result":
            self.tool_trace_presenter.record_result(event)
            return

    def _chat_append(self, text: str, *, streaming: bool) -> None:
        self.chat_panel.append_ai_message(ChatMessage("assistant", text), streaming=streaming)

    def _update_stream_state(self, active: bool) -> None:
        self._ai_stream_active = active
        self.stream_state_setter(active)

    @staticmethod
    def _normalize_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            return dict(metadata)
        except Exception:
            return metadata

    # ------------------------------------------------------------------
    # Editor Lock Management
    # ------------------------------------------------------------------
    def _acquire_editor_lock(self) -> None:
        """Acquire the editor lock to prevent user edits during AI turn."""
        if self.lock_manager is None:
            return
        try:
            session = self.lock_manager.acquire(LockReason.AI_TURN)
            if session is not None:
                self._active_lock_session = session
                _LOGGER.debug("Editor lock acquired: session=%s", session.session_id)
            else:
                _LOGGER.debug("Failed to acquire editor lock (already locked)")
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.debug("Error acquiring editor lock", exc_info=True)

    def _release_editor_lock(self) -> None:
        """Release the editor lock after AI turn completes."""
        if self.lock_manager is None:
            return
        session = self._active_lock_session
        if session is None:
            return
        try:
            session_id = session.session_id
            released = self.lock_manager.release(session_id)
            if released:
                _LOGGER.debug("Editor lock released: session=%s", session_id)
            else:
                _LOGGER.debug("Failed to release editor lock: session=%s", session_id)
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.debug("Error releasing editor lock", exc_info=True)
        finally:
            self._active_lock_session = None

    def force_release_lock(self) -> None:
        """Force release the editor lock (for cancellation)."""
        if self.lock_manager is None:
            return
        try:
            self.lock_manager.force_release()
            _LOGGER.debug("Editor lock force-released")
        except Exception:  # pragma: no cover - defensive guard
            _LOGGER.debug("Error force-releasing editor lock", exc_info=True)
        finally:
            self._active_lock_session = None


__all__ = ["AITurnCoordinator"]
