"""Handler for NeedsRangeError in tool execution."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Mapping, TYPE_CHECKING

from ..tools.errors import NeedsRangeError
from .scope_helpers import (
    scope_summary_from_arguments,
    scope_fields_from_summary,
    extract_chunk_id,
    parse_chunk_bounds,
)

if TYPE_CHECKING:
    from .model_types import ToolCallRequest

LOGGER = logging.getLogger(__name__)


class NeedsRangeHandler:
    """Handles NeedsRangeError by building informative payloads with span hints.
    
    When a tool requires explicit range information but none was provided,
    this handler builds a detailed error payload with hints about how to
    resolve the issue.
    """

    def __init__(
        self,
        snapshot_span_resolver: Callable[[Mapping[str, Any] | None], tuple[int, int] | None] | None = None,
        selection_tool_invoker: Callable[[str | None], Mapping[str, Any] | None] | None = None,
        tools_registry: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize the NeedsRange handler.
        
        Args:
            snapshot_span_resolver: Optional callable to resolve span from snapshot.
            selection_tool_invoker: Optional callable to invoke selection_range tool.
            tools_registry: Optional tool registry for looking up selection_range.
        """
        self._resolve_snapshot_span = snapshot_span_resolver
        self._invoke_selection_tool = selection_tool_invoker
        self._tools = tools_registry or {}

    def format_payload(
        self,
        call: "ToolCallRequest",
        resolved_arguments: Any,
        error: NeedsRangeError,
    ) -> dict[str, Any]:
        """Format a NeedsRangeError into an informative payload.
        
        Args:
            call: The tool call request that failed.
            resolved_arguments: The resolved tool arguments.
            error: The NeedsRangeError that was raised.
            
        Returns:
            Dictionary with error details, hints, and span information.
        """
        tab_id = self._extract_tab_id(resolved_arguments)
        message = str(error) or "needs_range: Provide target_range or replace_all=true before retrying."
        
        payload: dict[str, Any] = {
            "error": getattr(error, "code", "needs_range"),
            "needs_range": True,
            "message": message,
            "tab_id": tab_id,
            "hint": (
                "Call document_snapshot to capture the intended span and retry document_apply_patch with target_range"
                " or replace_all=true."
            ),
        }
        
        scope_summary = scope_summary_from_arguments(resolved_arguments)
        if scope_summary:
            payload["scope_summary"] = scope_summary
            scope_fields = scope_fields_from_summary(scope_summary)
            if scope_fields:
                payload.update(scope_fields)
                
        span_hint = self.resolve_span_hint(resolved_arguments, tab_id)
        if span_hint:
            payload["span_hint"] = span_hint
            
        content_length = getattr(error, "content_length", None)
        if content_length is not None:
            payload["content_length"] = content_length
        threshold = getattr(error, "threshold", None)
        if threshold is not None:
            payload["threshold"] = threshold
            
        return payload

    @staticmethod
    def format_message(tool_name: str | None, payload: Mapping[str, Any]) -> str:
        """Format an error message from the payload.
        
        Args:
            tool_name: Name of the tool that failed.
            payload: Error payload with message.
            
        Returns:
            Formatted error message string.
        """
        label = tool_name or "document_apply_patch"
        message = str(payload.get("message") or "needs_range: Provide explicit range information before retrying.")
        return f"Tool '{label}' failed: {message}"

    def resolve_span_hint(
        self,
        resolved_arguments: Any,
        tab_id: str | None,
    ) -> dict[str, Any] | None:
        """Resolve a span hint from various sources.
        
        Attempts to build a span hint from:
        1. Scope summary in arguments
        2. Chunk arguments
        3. Snapshot metadata
        4. Selection tool (if available)
        
        Args:
            resolved_arguments: Tool arguments to extract hints from.
            tab_id: Tab ID for selection tool fallback.
            
        Returns:
            Span hint dictionary or None if no hint available.
        """
        scope_summary = scope_summary_from_arguments(resolved_arguments)
        scope_hint = self._span_hint_from_scope_summary(scope_summary)
        if scope_hint is not None:
            return scope_hint
            
        chunk_hint = self._span_hint_from_chunk_arguments(resolved_arguments)
        if chunk_hint is not None:
            return chunk_hint
            
        snapshot_hint = self._span_hint_from_snapshot(resolved_arguments)
        if snapshot_hint is not None:
            return snapshot_hint
            
        if tab_id:
            selection_hint = self._span_hint_from_selection_tool(tab_id)
            if selection_hint is not None:
                return selection_hint
                
        return None

    def _span_hint_from_snapshot(self, arguments: Any) -> dict[str, Any] | None:
        """Build span hint from snapshot metadata in arguments.
        
        Args:
            arguments: Tool arguments that may contain snapshot.
            
        Returns:
            Span hint or None.
        """
        if not isinstance(arguments, Mapping):
            return None
        snapshot = arguments.get("snapshot")
        if not isinstance(snapshot, Mapping):
            return None
        if self._resolve_snapshot_span is None:
            return None
        span = self._resolve_snapshot_span(snapshot)
        if span is None:
            return None
        start, end = span
        return {
            "source": "snapshot_span",
            "target_range": {"start": start, "end": end},
        }

    def _span_hint_from_chunk_arguments(self, arguments: Any) -> dict[str, Any] | None:
        """Build span hint from chunk_id in arguments.
        
        Args:
            arguments: Tool arguments that may contain chunk_id.
            
        Returns:
            Span hint or None.
        """
        if not isinstance(arguments, Mapping):
            return None
        chunk_id = extract_chunk_id(arguments)
        if not chunk_id:
            return None
        bounds = parse_chunk_bounds(chunk_id)
        if not bounds:
            return None
        start, end = bounds
        return {
            "source": "chunk_manifest",
            "chunk_id": chunk_id,
            "target_range": {"start": start, "end": end},
        }

    def _span_hint_from_selection_tool(self, tab_id: str | None) -> dict[str, Any] | None:
        """Build span hint by invoking selection_range tool.
        
        Args:
            tab_id: Tab ID to get selection for.
            
        Returns:
            Span hint or None.
        """
        if self._invoke_selection_tool is not None:
            # Use custom invoker if provided
            result = self._invoke_selection_tool(tab_id)
            if not isinstance(result, Mapping):
                return None
        else:
            # Fall back to direct tool registry access
            registration = self._tools.get("selection_range")
            if registration is None:
                return None
            runner = getattr(registration, "impl", registration)
            runner = getattr(runner, "run", runner)
            if not callable(runner):
                return None
            try:
                result = runner(tab_id=tab_id)
            except TypeError:
                try:
                    result = runner()
                except Exception:
                    LOGGER.debug("SelectionRangeTool failed while building needs_range span hint", exc_info=True)
                    return None
            except Exception:
                LOGGER.debug("SelectionRangeTool failed while building needs_range span hint", exc_info=True)
                return None
            if inspect.isawaitable(result):
                LOGGER.debug("SelectionRangeTool returned awaitable; skipping needs_range hint")
                return None
            if not isinstance(result, Mapping):
                return None
                
        start_line = result.get("start_line")
        end_line = result.get("end_line")
        if not isinstance(start_line, int) or not isinstance(end_line, int):
            return None
        normalized_start = max(0, start_line)
        normalized_end = max(0, end_line)
        return {
            "source": "selection_range_tool",
            "target_span": {"start_line": normalized_start, "end_line": normalized_end},
            "content_hash": result.get("content_hash"),
        }

    def _span_hint_from_scope_summary(self, scope_summary: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """Build span hint from scope summary metadata.
        
        Args:
            scope_summary: Scope summary mapping.
            
        Returns:
            Span hint or None.
        """
        if not isinstance(scope_summary, Mapping):
            return None
        scope_fields = scope_fields_from_summary(scope_summary)
        if not scope_fields:
            return None
        range_payload = scope_fields.get("scope_range")
        origin = scope_fields.get("scope_origin")
        hint: dict[str, Any] = {"source": "tool_scope_metadata"}
        if isinstance(range_payload, Mapping):
            hint["target_range"] = {"start": range_payload.get("start"), "end": range_payload.get("end")}
        elif origin == "document":
            hint["target_range"] = {"scope": "document"}
        else:
            return None
        if origin:
            hint["scope_origin"] = origin
        scope_length = scope_fields.get("scope_length")
        if isinstance(scope_length, int):
            hint["scope_length"] = scope_length
        chunk_id = scope_summary.get("chunk_id") if isinstance(scope_summary, Mapping) else None
        if isinstance(chunk_id, str) and chunk_id.strip():
            hint["chunk_id"] = chunk_id.strip()
        return hint

    @staticmethod
    def _extract_tab_id(arguments: Any) -> str | None:
        """Extract tab_id from arguments.
        
        Args:
            arguments: Tool arguments mapping.
            
        Returns:
            Tab ID string or None.
        """
        if not isinstance(arguments, Mapping):
            return None
        candidate = arguments.get("tab_id")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        metadata = arguments.get("metadata")
        if isinstance(metadata, Mapping):
            tab_candidate = metadata.get("tab_id")
            if isinstance(tab_candidate, str) and tab_candidate.strip():
                return tab_candidate.strip()
        return None
