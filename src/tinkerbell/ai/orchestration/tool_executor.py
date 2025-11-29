"""Tool execution and result handling."""

from __future__ import annotations

import ast
import inspect
import json
import logging
import time
from typing import Any, Callable, Mapping, Sequence, Awaitable, TYPE_CHECKING

from ..client import AIStreamEvent
from ..tools.errors import NeedsRangeError
from ...services.bridge import DocumentVersionMismatchError

if TYPE_CHECKING:
    from .model_types import ToolCallRequest
    from .controller import OpenAIToolSpec
    from .version_retry import VersionRetryHandler
    from .needs_range_handler import NeedsRangeHandler
    from .tool_dispatcher import ToolDispatcher, DispatchResult

LOGGER = logging.getLogger(__name__)

# Type alias for event callbacks
ToolCallback = Callable[[AIStreamEvent], Awaitable[None] | None] | None


class ToolExecutor:
    """Executes tool calls and handles results.
    
    Orchestrates tool execution through either the new ToolDispatcher system
    or legacy direct invocation, handling version retry and NeedsRange errors.
    """

    def __init__(
        self,
        tool_registry: Mapping[str, "OpenAIToolSpec"],
        dispatcher: "ToolDispatcher | None" = None,
        version_retry_handler: "VersionRetryHandler | None" = None,
        needs_range_handler: "NeedsRangeHandler | None" = None,
        token_estimator: Callable[[str], int] | None = None,
        new_registry_checker: Callable[[str], bool] | None = None,
        plot_loop_blocker: Callable[[str | None], str | None] | None = None,
    ) -> None:
        """Initialize the tool executor.
        
        Args:
            tool_registry: Mapping of tool names to OpenAIToolSpec registrations.
            dispatcher: Optional ToolDispatcher for new-style tool execution.
            version_retry_handler: Handler for version mismatch retries.
            needs_range_handler: Handler for NeedsRangeError formatting.
            token_estimator: Callable to estimate tokens in text.
            new_registry_checker: Callable to check if tool is in new registry.
            plot_loop_blocker: Callable to check for plot loop blocking.
        """
        self._tools = tool_registry
        self._dispatcher = dispatcher
        self._version_retry = version_retry_handler
        self._needs_range = needs_range_handler
        self._estimate_tokens = token_estimator or (lambda x: len(x) // 4)
        self._is_new_registry_tool = new_registry_checker or (lambda _: False)
        self._plot_loop_block = plot_loop_blocker

    def update_dispatcher(self, dispatcher: "ToolDispatcher | None") -> None:
        """Update the tool dispatcher.
        
        Args:
            dispatcher: New dispatcher instance or None.
        """
        self._dispatcher = dispatcher

    async def handle_tool_calls(
        self,
        tool_calls: Sequence["ToolCallRequest"],
        on_event: ToolCallback,
        *,
        build_record_callback: Callable[..., dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        """Handle a batch of tool calls.
        
        Args:
            tool_calls: Sequence of tool call requests to execute.
            on_event: Callback for streaming events.
            build_record_callback: Optional callback to build tool records.
            
        Returns:
            Tuple of (messages, records, token_cost).
        """
        messages: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        token_cost = 0
        
        for call in tool_calls:
            registration = self._tools.get(call.name) if call.name else None
            summarizable = (
                getattr(registration, "summarizable", True)
                if registration is not None
                else True
            )
            started_at = time.time()
            start_perf = time.perf_counter()
            
            content, resolved_arguments, raw_result, retry_context = await self.execute_tool_call(
                call,
                registration,
                on_event,
            )
            
            duration_ms = max(0.0, (time.perf_counter() - start_perf) * 1000.0)
            argument_tokens = self._estimate_tokens(call.arguments or "")
            result_tokens = self._estimate_tokens(content)
            call_token_cost = argument_tokens + result_tokens
            
            tool_message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": content,
            }
            if call.name:
                tool_message["name"] = call.name
            messages.append(tool_message)
            
            if build_record_callback is not None:
                record = build_record_callback(
                    call,
                    resolved_arguments,
                    content,
                    call_token_cost,
                    duration_ms,
                    raw_result,
                    summarizable=summarizable,
                    started_at=started_at,
                )
            else:
                record = self._build_basic_record(
                    call,
                    resolved_arguments,
                    content,
                    call_token_cost,
                    duration_ms,
                    raw_result,
                    summarizable=summarizable,
                    started_at=started_at,
                )
            if retry_context:
                record["retry"] = dict(retry_context)
            records.append(record)
            token_cost += call_token_cost
            
        return messages, records, token_cost

    async def execute_tool_call(
        self,
        call: "ToolCallRequest",
        registration: "OpenAIToolSpec | None",
        on_event: ToolCallback,
    ) -> tuple[str, Any, Any, dict[str, Any] | None]:
        """Execute a single tool call.
        
        Args:
            call: Tool call request.
            registration: Tool registration or None if not found.
            on_event: Callback for streaming events.
            
        Returns:
            Tuple of (serialized_result, resolved_arguments, raw_result, retry_context).
        """
        resolved_arguments = self.coerce_arguments(call.arguments, call.parsed)
        resolved_arguments = self.normalize_arguments(call, resolved_arguments)
        
        if registration is None:
            message = f"Tool '{call.name or 'unknown'}' is not registered."
            await self._emit_tool_result_event(call, message, None, on_event)
            return message, resolved_arguments, None, None

        # Check for plot loop blocking
        if self._plot_loop_block is not None:
            block_reason = self._plot_loop_block(call.name)
            if block_reason:
                payload = {"status": "plot_loop_blocked", "reason": block_reason}
                await self._emit_tool_result_event(call, block_reason, payload, on_event)
                return block_reason, resolved_arguments, payload, None

        # Check if tool is in new registry - dispatch through ToolDispatcher if so
        tool_name = call.name or ""
        if self._dispatcher is not None and self._is_new_registry_tool(tool_name):
            return await self._execute_via_dispatcher(call, resolved_arguments, on_event)

        # Legacy path: use invoke_tool_impl directly
        return await self._execute_legacy(call, registration, resolved_arguments, on_event)

    async def _execute_legacy(
        self,
        call: "ToolCallRequest",
        registration: "OpenAIToolSpec",
        resolved_arguments: Any,
        on_event: ToolCallback,
    ) -> tuple[str, Any, Any, dict[str, Any] | None]:
        """Execute tool through legacy direct invocation path.
        
        Args:
            call: Tool call request.
            registration: Tool registration with implementation.
            resolved_arguments: Coerced and normalized arguments.
            on_event: Callback for streaming events.
            
        Returns:
            Tuple of (serialized_result, resolved_arguments, raw_result, retry_context).
        """
        retry_context: dict[str, Any] | None = None
        serialized_override: str | None = None
        
        try:
            result = await self.invoke_tool_impl(registration.impl, resolved_arguments)
        except DocumentVersionMismatchError as exc:
            if self._version_retry is None or not self._version_retry.supports_retry(call.name):
                raise
            result, retry_context = await self._version_retry.handle_retry(
                call,
                registration,
                resolved_arguments,
                exc,
            )
        except NeedsRangeError as exc:
            if self._needs_range is not None:
                result = self._needs_range.format_payload(call, resolved_arguments, exc)
                serialized_override = self._needs_range.format_message(call.name, result)
            else:
                result = {"error": "needs_range", "message": str(exc)}
                serialized_override = f"Tool '{call.name}' failed: {exc}"
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Tool %s failed", call.name)
            result = f"Tool '{call.name}' failed: {exc}"

        serialized = serialized_override or self.serialize_result(result)
        await self._emit_tool_result_event(call, serialized, result, on_event)
        return serialized, resolved_arguments, result, retry_context

    async def _execute_via_dispatcher(
        self,
        call: "ToolCallRequest",
        resolved_arguments: Any,
        on_event: ToolCallback,
    ) -> tuple[str, Any, Any, dict[str, Any] | None]:
        """Execute a tool call through the ToolDispatcher.
        
        Args:
            call: Tool call request.
            resolved_arguments: Coerced and normalized arguments.
            on_event: Callback for streaming events.
            
        Returns:
            Tuple of (serialized_result, resolved_arguments, raw_result, retry_context).
        """
        tool_name = call.name or ""
        arguments = dict(resolved_arguments) if isinstance(resolved_arguments, dict) else {}
        
        # Dispatch through the new system
        dispatch_result = await self._dispatcher.dispatch(tool_name, arguments)
        
        if dispatch_result.success:
            result = dispatch_result.result
            serialized = self.serialize_result(result)
            await self._emit_tool_result_event(call, serialized, result, on_event)
            return serialized, resolved_arguments, result, None
        else:
            # Handle error from dispatcher
            error = dispatch_result.error
            error_message = error.message if error else "Unknown error"
            
            # Check for version mismatch errors that should trigger retry
            from ..tools.errors import VersionMismatchToolError
            if (
                isinstance(error, VersionMismatchToolError)
                and self._version_retry is not None
                and self._version_retry.supports_retry(tool_name)
            ):
                # Convert to legacy exception and handle retry
                exc = DocumentVersionMismatchError(cause="version_mismatch")
                try:
                    # Get registration from legacy registry for retry
                    registration = self._tools.get(tool_name)
                    if registration:
                        result, retry_context = await self._version_retry.handle_retry(
                            call,
                            registration,
                            resolved_arguments,
                            exc,
                        )
                        serialized = self.serialize_result(result)
                        await self._emit_tool_result_event(call, serialized, result, on_event)
                        return serialized, resolved_arguments, result, retry_context
                except Exception:
                    pass  # Fall through to error handling
            
            # Format error as result
            result = f"Tool '{tool_name}' failed: {error_message}"
            await self._emit_tool_result_event(call, result, dispatch_result.to_dict(), on_event)
            return result, resolved_arguments, dispatch_result.to_dict(), None

    async def invoke_tool_impl(self, tool_impl: Any, arguments: Any) -> Any:
        """Invoke a tool implementation with arguments.
        
        Args:
            tool_impl: Tool implementation (callable or object with run method).
            arguments: Arguments to pass to the tool.
            
        Returns:
            Result from tool execution.
            
        Raises:
            TypeError: If tool is not callable.
        """
        target = getattr(tool_impl, "run", tool_impl)
        if not callable(target):  # pragma: no cover - safety
            raise TypeError(f"Registered tool {tool_impl!r} is not callable")

        if isinstance(arguments, Mapping):
            result = target(**arguments)
        elif arguments in (None, {}):
            result = target()
        else:
            # Only use positional invocation for non-mapping arguments
            result = target(arguments)

        if inspect.isawaitable(result):
            result = await result
        return result

    def serialize_result(self, result: Any) -> str:
        """Serialize a tool result to string.
        
        Args:
            result: Tool execution result.
            
        Returns:
            Serialized string representation.
        """
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(result)

    def coerce_arguments(self, raw_arguments: str | None, parsed: Any | None) -> Any:
        """Coerce raw argument string to typed value.
        
        Args:
            raw_arguments: Raw JSON arguments string.
            parsed: Pre-parsed arguments if available.
            
        Returns:
            Coerced argument value (dict, string, or original).
        """
        if parsed is not None:
            return parsed
        if raw_arguments is None:
            return {}
        text = raw_arguments.strip()
        if not text:
            return {}
        try:
            return json.loads(text, strict=False)
        except (ValueError, TypeError):
            fallback = self._literal_arguments(text)
            if fallback is not None:
                return fallback
            return text

    def normalize_arguments(self, call: "ToolCallRequest", arguments: Any) -> Any:
        """Normalize tool arguments based on tool type.
        
        Args:
            call: Tool call request with name.
            arguments: Coerced arguments to normalize.
            
        Returns:
            Normalized arguments mapping.
        """
        if not isinstance(arguments, Mapping):
            return arguments
        name = (call.name or "").strip().lower()
        normalized = dict(arguments)
        
        if name == "document_apply_patch":
            replacement = normalized.get("replacement_text")
            content = normalized.get("content")
            if isinstance(replacement, str) and not isinstance(content, str):
                normalized["content"] = replacement
            if "replacement_text" in normalized:
                normalized.pop("replacement_text", None)
        elif name == "document_edit":
            text_value = normalized.get("text")
            content_value = normalized.get("content")
            if isinstance(text_value, str) and not isinstance(content_value, str):
                normalized["content"] = text_value
                normalized.pop("text", None)
                
        return normalized

    def _literal_arguments(self, text: str) -> Any | None:
        """Try to parse arguments as Python literal.
        
        Args:
            text: Text to parse.
            
        Returns:
            Parsed value or None.
        """
        if not text or text[0] not in "{[":
            return None
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None

    def _build_basic_record(
        self,
        call: "ToolCallRequest",
        resolved_arguments: Any,
        serialized_result: str,
        tokens_used: int,
        duration_ms: float,
        raw_result: Any,
        *,
        started_at: float,
        summarizable: bool,
    ) -> dict[str, Any]:
        """Build a basic tool execution record.
        
        Args:
            call: Tool call request.
            resolved_arguments: Resolved arguments.
            serialized_result: Serialized result string.
            tokens_used: Token cost of this call.
            duration_ms: Execution duration in milliseconds.
            raw_result: Raw result value.
            started_at: Unix timestamp of execution start.
            summarizable: Whether result can be summarized.
            
        Returns:
            Tool execution record dictionary.
        """
        return {
            "id": call.call_id,
            "name": call.name,
            "index": call.index,
            "arguments": call.arguments,
            "parsed": call.parsed,
            "resolved_arguments": resolved_arguments,
            "result": serialized_result,
            "raw_result": raw_result,
            "status": "ok",
            "tokens_used": tokens_used,
            "duration_ms": round(duration_ms, 3),
            "started_at": started_at,
            "summarizable": summarizable,
        }

    async def _emit_tool_result_event(
        self,
        call: "ToolCallRequest",
        content: str,
        raw_result: Any,
        on_event: ToolCallback,
    ) -> None:
        """Emit a tool result event.
        
        Args:
            call: Tool call request.
            content: Serialized result content.
            raw_result: Raw result value.
            on_event: Event callback.
        """
        if on_event is None:
            return
        event = AIStreamEvent(
            type="tool_calls.result",
            content=content,
            tool_name=call.name or None,
            tool_arguments=content,
            tool_index=call.index,
            parsed=raw_result,
            tool_call_id=call.call_id,
        )
        result = on_event(event)
        if inspect.isawaitable(result):
            await result
