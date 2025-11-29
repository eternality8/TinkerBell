"""Chunk flow tracking for the AI controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ...services import telemetry as telemetry_service
from .. import prompts


@dataclass(slots=True)
class ChunkContext:
    """Resolved chunk metadata pulled from a manifest for subagent planning."""

    chunk_id: str
    document_id: str
    char_range: tuple[int, int]
    chunk_hash: str | None = None
    pointer_id: str | None = None
    text: str | None = None


@dataclass(slots=True)
class ChunkFlowTracker:
    """Tracks whether the agent stays on the chunk-first path during a run.
    
    Recognizes both legacy and new tool names:
    - document_snapshot / read_document
    - document_chunk / analyze_document
    """

    document_id: str | None
    warning_active: bool = False
    last_reason: str | None = None

    # Tool name sets for matching (legacy + new names)
    _SNAPSHOT_TOOLS: frozenset[str] = frozenset({"document_snapshot", "read_document"})
    _CHUNK_TOOLS: frozenset[str] = frozenset({"document_chunk", "analyze_document"})

    def observe_tool(self, record: Mapping[str, Any], payload: Mapping[str, Any] | None) -> list[str] | None:
        name = str(record.get("name") or "").lower()
        if name in self._SNAPSHOT_TOOLS:
            return self._handle_snapshot(record, payload, source=name)
        if name in self._CHUNK_TOOLS:
            self._handle_chunk_tool(payload, source=name)
        return None

    # ------------------------------------------------------------------
    # Snapshot handling
    # ------------------------------------------------------------------
    def _handle_snapshot(self, record: Mapping[str, Any], payload: Mapping[str, Any] | None, *, source: str = "read_document") -> list[str] | None:
        if not isinstance(payload, Mapping):
            return None
        window = payload.get("window") if isinstance(payload.get("window"), Mapping) else None
        if window is None:
            window = {}
        includes_full = bool(window.get("includes_full_document"))
        doc_length = self._coerce_int(payload.get("length"))
        window_span = self._window_span(window, payload)
        span_length = self._span_length(window, payload)
        manifest = payload.get("chunk_manifest") if isinstance(payload.get("chunk_manifest"), Mapping) else None
        metadata: dict[str, Any] = {
            "document_id": payload.get("document_id") or self.document_id,
            "document_length": doc_length,
            "window_span": window_span,
            "span_length": span_length,
            "window_kind": str(window.get("kind") or ""),
            "requested_window": str(window.get("requested_kind") or ""),
            "defaulted": bool(window.get("defaulted")),
            "source": source,
        }
        if manifest:
            chunks = manifest.get("chunks")
            if isinstance(chunks, Sequence):
                metadata["chunk_count"] = len(chunks)
            cache_hit = manifest.get("cache_hit")
            if cache_hit is not None:
                metadata["cache_hit"] = bool(cache_hit)
        threshold_triggered = self._is_large_window(doc_length, window_span)
        if includes_full and threshold_triggered:
            metadata["reason"] = self._snapshot_reason(record, metadata)
            return self._emit_warning(metadata)
        self._emit_request(metadata)
        if self.warning_active:
            recovery = dict(metadata)
            recovery["recovered_via"] = source
            self._emit_recovery(recovery)
        return None

    def _snapshot_reason(self, record: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
        resolved = record.get("resolved_arguments") if isinstance(record.get("resolved_arguments"), Mapping) else None
        if resolved:
            window_arg = resolved.get("window")
            if isinstance(window_arg, str) and window_arg.strip():
                return f"window:{window_arg.strip().lower()}"
            if isinstance(window_arg, Mapping):
                kind = window_arg.get("kind") or window_arg.get("requested_kind")
                if isinstance(kind, str) and kind.strip():
                    return f"window:{kind.strip().lower()}"
        if metadata.get("defaulted"):
            return "default_window_full"
        return "full_window_range"

    def _window_span(self, window: Mapping[str, Any], payload: Mapping[str, Any]) -> int | None:
        start = self._coerce_int(window.get("start"))
        end = self._coerce_int(window.get("end"))
        if start is not None and end is not None:
            return max(0, end - start)
        text_range = payload.get("text_range")
        if isinstance(text_range, Mapping):
            range_start = self._coerce_int(text_range.get("start"))
            range_end = self._coerce_int(text_range.get("end"))
            if range_start is not None and range_end is not None:
                return max(0, range_end - range_start)
        return None

    @staticmethod
    def _span_length(window: Mapping[str, Any], payload: Mapping[str, Any]) -> int | None:
        text_range = payload.get("text_range") if isinstance(payload.get("text_range"), Mapping) else None
        if isinstance(text_range, Mapping):
            start = ChunkFlowTracker._coerce_int(text_range.get("start"))
            end = ChunkFlowTracker._coerce_int(text_range.get("end"))
            if start is not None and end is not None:
                return max(0, end - start)
        start = ChunkFlowTracker._coerce_int(window.get("start"))
        end = ChunkFlowTracker._coerce_int(window.get("end"))
        if start is None or end is None:
            return None
        return max(0, end - start)

    @staticmethod
    def _is_large_window(doc_length: int | None, window_span: int | None) -> bool:
        span = window_span if window_span is not None else doc_length
        if span is None:
            return False
        return span >= prompts.LARGE_DOC_CHAR_THRESHOLD

    # ------------------------------------------------------------------
    # Chunk tool handling
    # ------------------------------------------------------------------
    def _handle_chunk_tool(self, payload: Mapping[str, Any] | None, *, source: str = "analyze_document") -> None:
        if not isinstance(payload, Mapping):
            return
        chunk = payload.get("chunk")
        if not isinstance(chunk, Mapping):
            return
        start = self._coerce_int(chunk.get("start"))
        end = self._coerce_int(chunk.get("end"))
        length = chunk.get("length")
        if not isinstance(length, int) and start is not None and end is not None:
            length = max(0, end - start)
        metadata = {
            "document_id": chunk.get("document_id") or self.document_id,
            "chunk_id": chunk.get("chunk_id"),
            "chunk_length": length,
            "window_start": start,
            "window_end": end,
            "pointerized": bool(chunk.get("pointer")),
            "source": source,
        }
        self._emit_request(metadata)
        if self.warning_active:
            recovery = dict(metadata)
            recovery["recovered_via"] = source
            self._emit_recovery(recovery)

    # ------------------------------------------------------------------
    # Telemetry helpers
    # ------------------------------------------------------------------
    def _emit_request(self, metadata: Mapping[str, Any]) -> None:
        telemetry_service.emit("chunk_flow.requested", self._clean_payload(metadata))

    def _emit_warning(self, metadata: Mapping[str, Any]) -> list[str]:
        payload = dict(metadata)
        if self.warning_active:
            payload["repeat"] = True
        telemetry_service.emit("chunk_flow.escaped_full_snapshot", self._clean_payload(payload))
        self.warning_active = True
        self.last_reason = str(metadata.get("reason") or "full_snapshot")
        doc_length = self._coerce_int(metadata.get("document_length"))
        approx = f" (~{doc_length:,} chars)" if doc_length else ""
        return [
            f"read_document fetched the entire document{approx}.",
            "Request a selection-scoped read or analyze a chunk via analyze_document before editing.",
            "If a full read is unavoidable, explain the fallback to the user and immediately return to chunked context.",
        ]

    def _emit_recovery(self, metadata: Mapping[str, Any]) -> None:
        if not self.warning_active:
            return
        telemetry_service.emit("chunk_flow.retry_success", self._clean_payload(metadata))
        self.warning_active = False
        self.last_reason = None

    @staticmethod
    def _clean_payload(metadata: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in metadata.items() if value not in (None, "")}

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


__all__ = [
    "ChunkContext",
    "ChunkFlowTracker",
]
