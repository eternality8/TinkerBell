"""Coordinates AI turn execution and streaming events for MainWindow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..chat.message_model import ChatMessage
from .tool_trace_presenter import ToolTracePresenter


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

    _ai_stream_active: bool = False

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
        self._ai_stream_active = False
        self.stream_state_setter(False)
        self.tool_trace_presenter.reset()
        normalized_metadata = self._normalize_metadata(metadata)
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
            self.review_controller.finalize_pending_turn_review(success=False)
            self.failure_handler(exc)
            return

        payload = result or {}
        tool_records = payload.get("tool_calls")
        self.tool_trace_presenter.annotate_compaction(tool_records)
        self.telemetry_controller.set_compaction_stats(payload.get("trace_compaction"))
        response_text = payload.get("response", "").strip() or "The AI did not return any content."
        self.response_finalizer(response_text)
        self.telemetry_controller.refresh_context_usage_status()
        self.status_updater("AI response ready")
        self.review_controller.finalize_pending_turn_review(success=True)

    async def _handle_stream_event(self, event: Any) -> None:
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


__all__ = ["AITurnCoordinator"]
