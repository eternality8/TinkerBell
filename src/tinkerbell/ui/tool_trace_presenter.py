"""Utility that tracks streaming tool traces and applies compaction metadata."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from ..chat.message_model import ToolTrace
from .models.tool_traces import PendingToolTrace

Summarizer = Callable[[Any], str]


@dataclass(slots=True)
class ToolTracePresenter:
    """Aggregates streaming tool call events into ToolTrace payloads."""

    chat_panel: Any
    tool_trace_index: MutableMapping[str, ToolTrace]
    pending_tool_traces: MutableMapping[str, PendingToolTrace] = field(default_factory=dict)
    summarize_io: Summarizer | None = None
    clock: Callable[[], float] = time.perf_counter

    def reset(self) -> None:
        self.pending_tool_traces.clear()
        self.tool_trace_index.clear()

    def record_arguments_delta(self, event: Any) -> None:
        key = self._tool_call_key(event)
        if not key:
            return
        state = self._pending_state(key, event)
        delta = getattr(event, "arguments_delta", None) or getattr(event, "tool_arguments", None)
        if delta:
            state.arguments_chunks.append(str(delta))

    def finalize_arguments(self, event: Any) -> None:
        key = self._tool_call_key(event)
        if not key:
            return
        state = self._pending_state(key, event)
        arguments_text = getattr(event, "tool_arguments", None)
        if not arguments_text:
            arguments_text = "".join(state.arguments_chunks)
        state.arguments_chunks.clear()
        state.raw_input = str(arguments_text or "")
        metadata: dict[str, Any] = {"raw_input": state.raw_input}
        metadata["tool_call_id"] = state.tool_call_id or key
        trace = ToolTrace(
            name=state.name,
            input_summary=self._summarize(state.raw_input),
            output_summary="(running…)",
            metadata=metadata,
        )
        state.trace = trace
        state.started_at = self.clock()
        self._show_trace(trace)
        if state.pending_output is not None:
            self._apply_tool_result_to_trace(key, state)

    def record_result(self, event: Any) -> None:
        key = self._tool_call_key(event)
        if not key:
            return
        state = self._pending_state(key, event)
        content = getattr(event, "content", None) or getattr(event, "tool_arguments", None) or ""
        state.pending_output = str(content)
        state.pending_parsed = getattr(event, "parsed", None)
        self._apply_tool_result_to_trace(key, state)

    def annotate_compaction(self, records: Sequence[Mapping[str, Any]] | None) -> None:
        if not records:
            return
        for record in records:
            if not isinstance(record, Mapping):
                continue
            pointer = record.get("pointer")
            if not pointer:
                continue
            call_id = record.get("id") or record.get("tool_call_id")
            if not call_id:
                continue
            trace = self.tool_trace_index.get(str(call_id))
            if trace is None:
                continue
            metadata = dict(trace.metadata or {})
            metadata["compacted"] = True
            metadata["pointer"] = pointer
            instructions = pointer.get("rehydrate_instructions") if isinstance(pointer, Mapping) else None
            if instructions:
                metadata["pointer_instructions"] = instructions
            summary = pointer.get("display_text") if isinstance(pointer, Mapping) else None
            if summary:
                metadata["pointer_summary"] = summary
                trace.output_summary = summary
            trace.metadata = metadata
            self._update_trace(trace)

    def _pending_state(self, key: str, event: Any) -> PendingToolTrace:
        state = self.pending_tool_traces.get(key)
        if state is None:
            state = PendingToolTrace(name=self._normalize_tool_name(event))
            self.pending_tool_traces[key] = state
        state.tool_call_id = key
        return state

    def _apply_tool_result_to_trace(self, key: str, state: PendingToolTrace) -> None:
        trace = state.trace
        if trace is None or state.pending_output is None:
            return
        trace.output_summary = self._summarize(state.pending_output)
        metadata = dict(trace.metadata)
        if state.raw_input is not None:
            metadata.setdefault("raw_input", state.raw_input)
        metadata["raw_output"] = state.pending_output
        if state.pending_parsed is not None:
            metadata["parsed_output"] = state.pending_parsed
        trace.metadata = metadata
        tool_call_id = metadata.get("tool_call_id")
        if isinstance(tool_call_id, str):
            self.tool_trace_index[tool_call_id] = trace
        if state.started_at is not None:
            elapsed = max(0.0, self.clock() - state.started_at)
            trace.duration_ms = int(elapsed * 1000)
        self._update_trace(trace)
        self.pending_tool_traces.pop(key, None)

    def _show_trace(self, trace: ToolTrace) -> None:
        show_trace = getattr(self.chat_panel, "show_tool_trace", None)
        if callable(show_trace):
            show_trace(trace)

    def _update_trace(self, trace: ToolTrace) -> None:
        updater = getattr(self.chat_panel, "update_tool_trace", None)
        if callable(updater):
            updater(trace)

    def _summarize(self, payload: Any) -> str:
        if self.summarize_io is not None:
            return self.summarize_io(payload)
        text = str(payload).strip()
        if not text:
            return "(empty)"
        condensed = " ".join(text.split())
        if not condensed:
            return "(empty)"
        if len(condensed) <= 140:
            return condensed
        return f"{condensed[:139].rstrip()}…"

    @staticmethod
    def _tool_call_key(event: Any) -> str | None:
        identifier = getattr(event, "tool_call_id", None) or getattr(event, "id", None)
        if identifier:
            return str(identifier)
        index = getattr(event, "tool_index", None)
        name = ToolTracePresenter._normalize_tool_name(event)
        return f"{name}:{index if index is not None else 0}"

    @staticmethod
    def _normalize_tool_name(event: Any) -> str:
        name = getattr(event, "tool_name", None) or getattr(event, "name", None) or "tool"
        text = str(name).strip()
        return text or "tool"


__all__ = ["ToolTracePresenter"]
