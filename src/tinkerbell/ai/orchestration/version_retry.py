"""Version mismatch retry handling for tool execution."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Mapping, MutableMapping, Awaitable, TYPE_CHECKING

from ...services import telemetry as telemetry_service
from ...services.bridge import DocumentVersionMismatchError
from .scope_helpers import scope_fields_from_summary

if TYPE_CHECKING:
    from .model_types import ToolCallRequest
    from .controller import OpenAIToolSpec

LOGGER = logging.getLogger(__name__)

# Tools that support automatic retry on version mismatch
_RETRYABLE_VERSION_TOOLS: frozenset[str] = frozenset({"document_apply_patch", "search_replace"})


class VersionRetryHandler:
    """Handles version mismatch errors with automatic retry.
    
    Provides version retry logic for edit tools that may fail due to
    document changes occurring between snapshot and edit attempt.
    """

    def __init__(
        self,
        snapshot_refresher: Callable[[str | None], Awaitable[Mapping[str, Any] | None]],
        tool_invoker: Callable[[Any, Any], Awaitable[Any]],
        scope_summary_extractor: Callable[[Any], Mapping[str, Any] | None] | None = None,
    ) -> None:
        """Initialize the version retry handler.
        
        Args:
            snapshot_refresher: Async callable to refresh document snapshot.
            tool_invoker: Async callable to invoke tool implementation.
            scope_summary_extractor: Optional callable to extract scope from arguments.
        """
        self._refresh_snapshot = snapshot_refresher
        self._invoke_tool = tool_invoker
        self._extract_scope_summary = scope_summary_extractor

    @staticmethod
    def supports_retry(tool_name: str | None) -> bool:
        """Check if a tool supports version retry.
        
        Args:
            tool_name: Name of the tool to check.
            
        Returns:
            True if the tool supports automatic version retry.
        """
        if not tool_name:
            return False
        return tool_name.strip().lower() in _RETRYABLE_VERSION_TOOLS

    async def handle_retry(
        self,
        call: "ToolCallRequest",
        registration: "OpenAIToolSpec",
        resolved_arguments: Any,
        error: DocumentVersionMismatchError,
    ) -> tuple[Any, dict[str, Any]]:
        """Handle a version mismatch by refreshing snapshot and retrying.
        
        Args:
            call: Original tool call request.
            registration: Tool registration with implementation.
            resolved_arguments: Coerced and normalized arguments.
            error: The version mismatch error that triggered retry.
            
        Returns:
            Tuple of (result, retry_context) where retry_context contains
            metadata about the retry attempt.
        """
        tab_id = self.extract_tab_id(resolved_arguments)
        snapshot = await self._refresh_snapshot(tab_id)
        document_id = self._snapshot_document_id(snapshot)
        self.inject_snapshot_metadata(resolved_arguments, snapshot)
        
        scope_summary = None
        if self._extract_scope_summary:
            scope_summary = self._extract_scope_summary(resolved_arguments)
        scope_fields = scope_fields_from_summary(scope_summary)
        
        base_context: dict[str, Any] = {
            "tool": call.name or "unknown",
            "tab_id": tab_id,
            "document_id": document_id,
            "cause": error.cause or "hash_mismatch",
            "attempts": 2,
        }
        if scope_summary:
            base_context["scope_summary"] = scope_summary
        if scope_fields:
            base_context.update(scope_fields)
            
        LOGGER.warning(
            "Retrying %s after DocumentVersionMismatchError (cause=%s)",
            call.name,
            error.cause,
        )
        
        try:
            result = await self._invoke_tool(registration.impl, resolved_arguments)
        except DocumentVersionMismatchError as retry_exc:
            failure_context = dict(base_context)
            failure_context["status"] = "failed"
            failure_context["reason"] = "retry_exhausted"
            failure_context["cause"] = retry_exc.cause or failure_context.get("cause")
            self._emit_retry_event(failure_context)
            message = self.format_failure_message(call.name, retry_exc)
            return message, failure_context

        success_context = dict(base_context)
        success_context["status"] = "success"
        self._emit_retry_event(success_context)
        return result, success_context

    @staticmethod
    def inject_snapshot_metadata(arguments: Any, snapshot: Mapping[str, Any] | None) -> None:
        """Inject snapshot metadata into tool arguments.
        
        Updates the arguments mapping in-place with version information
        from the refreshed document snapshot.
        
        Args:
            arguments: Tool arguments mapping to update.
            snapshot: Document snapshot with version metadata.
        """
        if not isinstance(arguments, MutableMapping) or not isinstance(snapshot, Mapping):
            return
        version = snapshot.get("version")
        if isinstance(version, str) and version.strip():
            arguments["document_version"] = version.strip()
        version_id = snapshot.get("version_id")
        if version_id is not None:
            arguments["version_id"] = version_id
        content_hash = snapshot.get("content_hash")
        if isinstance(content_hash, str) and content_hash.strip():
            arguments["content_hash"] = content_hash.strip()

    @staticmethod
    def extract_tab_id(arguments: Any) -> str | None:
        """Extract tab_id from tool arguments.
        
        Args:
            arguments: Tool arguments that may contain tab_id.
            
        Returns:
            The tab_id string or None if not found.
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

    def _emit_retry_event(self, payload: Mapping[str, Any]) -> None:
        """Emit a telemetry event for version retry.
        
        Args:
            payload: Event payload with retry context.
        """
        if not payload:
            return
        event_payload = dict(payload)
        event_payload.setdefault("event_source", "version_retry")
        scope_summary = (
            event_payload.get("scope_summary")
            if isinstance(event_payload.get("scope_summary"), Mapping)
            else None
        )
        scope_fields = scope_fields_from_summary(scope_summary)
        for key, value in scope_fields.items():
            event_payload.setdefault(key, value)
        telemetry_service.emit("document_edit.retry", event_payload)

    @staticmethod
    def format_failure_message(
        tool_name: str | None,
        error: DocumentVersionMismatchError,
    ) -> str:
        """Format a user-friendly retry failure message.
        
        Args:
            tool_name: Name of the tool that failed.
            error: The version mismatch error.
            
        Returns:
            Formatted error message string.
        """
        label = tool_name or "document_edit"
        cause = f" (cause={error.cause})" if getattr(error, "cause", None) else ""
        return (
            f"Tool '{label}' failed: document snapshot was stale even after an automatic retry{cause}. "
            "Call document_snapshot again and rebuild your diff before retrying."
        )

    @staticmethod
    def _snapshot_document_id(snapshot: Mapping[str, Any] | None) -> str | None:
        """Extract document_id from a snapshot.
        
        Args:
            snapshot: Document snapshot mapping.
            
        Returns:
            The document_id or None.
        """
        if not isinstance(snapshot, Mapping):
            return None
        document_id = snapshot.get("document_id")
        if isinstance(document_id, str):
            text = document_id.strip()
            return text or None
        return None
