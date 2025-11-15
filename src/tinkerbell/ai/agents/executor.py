"""Agent executor façade wrapping LangChain/LangGraph interactions."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, MutableMapping, Optional, Sequence, cast

from openai.types.chat import ChatCompletionToolParam

from .. import prompts
from ..client import AIClient, AIStreamEvent
from .graph import build_agent_graph

LOGGER = logging.getLogger(__name__)

ToolCallback = Callable[[AIStreamEvent], Awaitable[None] | None]


@dataclass(slots=True)
class _ToolCallRequest:
    """Internal representation of tool call directives emitted by the model."""

    call_id: str
    name: str
    index: int
    arguments: str | None
    parsed: Any | None


@dataclass(slots=True)
class _ModelTurnResult:
    """Aggregate of a single model turn (stream) including tool metadata."""

    assistant_message: Dict[str, Any]
    response_text: str
    tool_calls: list[_ToolCallRequest]


@dataclass(slots=True)
class ToolRegistration:
    """Metadata stored for each registered tool."""

    name: str
    impl: Any
    description: str | None = None
    parameters: Mapping[str, Any] | None = None
    strict: bool = True

    def as_openai_tool(self) -> ChatCompletionToolParam:
        """Return an OpenAI-compatible tool spec for the AI client."""

        parameters = self.parameters or getattr(self.impl, "args_schema", None)
        if callable(parameters):
            try:  # pragma: no cover - defensive, args_schema can be pydantic.BaseModel
                parameters = parameters()
            except TypeError:
                parameters = None
        if parameters is None:
            parameters = {"type": "object", "properties": {}}

        description = self.description or getattr(self.impl, "description", None) or (
            inspect.getdoc(self.impl) or f"Tool {self.name}"
        )

        return cast(
            ChatCompletionToolParam,
            {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": description,
                    "parameters": parameters,
                    "strict": bool(self.strict),
                },
            },
        )


@dataclass(slots=True)
class AIController:
    """High-level interface invoked by the chat panel."""

    client: AIClient
    tools: MutableMapping[str, ToolRegistration] = field(default_factory=dict)
    _graph: Dict[str, Any] = field(init=False, repr=False)
    _active_task: asyncio.Task[dict] | None = field(default=None, init=False, repr=False)
    _task_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tools:
            normalized: Dict[str, ToolRegistration] = {}
            for name, value in self.tools.items():
                if isinstance(value, ToolRegistration):
                    normalized[name] = value
                else:
                    normalized[name] = ToolRegistration(name=name, impl=value)
            self.tools = normalized
        self._rebuild_graph()

    @property
    def graph(self) -> Dict[str, Any]:
        """Return the current compiled graph representation."""

        return dict(self._graph)

    def register_tool(
        self,
        name: str,
        tool: Any,
        *,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        strict: bool | None = None,
    ) -> None:
        """Register (or replace) a tool available to the agent."""

        registration = ToolRegistration(
            name=name,
            impl=tool,
            description=description,
            parameters=parameters,
            strict=True if strict is None else bool(strict),
        )
        self.tools[name] = registration
        LOGGER.debug("Registered tool: %s", name)
        self._rebuild_graph()

    def unregister_tool(self, name: str) -> None:
        """Remove a tool and rebuild the agent graph."""

        if name in self.tools:
            self.tools.pop(name)
            LOGGER.debug("Unregistered tool: %s", name)
            self._rebuild_graph()

    def update_client(self, client: AIClient) -> None:
        """Swap the underlying AI client (e.g., when settings change)."""

        self.client = client

    async def run_chat(
        self,
        prompt: str,
        doc_snapshot: Mapping[str, Any] | None,
        *,
        metadata: Mapping[str, str] | None = None,
        on_event: ToolCallback | None = None,
    ) -> dict:
        """Execute a chat turn against the compiled agent graph."""

        snapshot = dict(doc_snapshot or {})
        base_messages = self._build_messages(prompt, snapshot)
        merged_metadata = self._build_metadata(snapshot, metadata)
        tool_specs = [registration.as_openai_tool() for registration in self.tools.values()]
        max_iterations = self._graph.get("metadata", {}).get("max_iterations")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            max_iterations = 8

        LOGGER.debug(
            "Starting chat turn (prompt length=%s, tools=%s)",
            len(prompt),
            list(self.tools.keys()),
        )

        async def _runner() -> dict:
            conversation: list[dict[str, Any]] = list(base_messages)
            response_text = ""
            turn_count = 0
            tool_iterations = 0
            executed_tool_calls: list[dict[str, Any]] = []

            while True:
                turn_count += 1
                turn = await self._invoke_model_turn(
                    conversation,
                    tool_specs=tool_specs if tool_specs else None,
                    metadata=merged_metadata,
                    on_event=on_event,
                )
                conversation.append(turn.assistant_message)
                response_text = turn.response_text

                if not turn.tool_calls:
                    break

                tool_iterations += 1
                if tool_iterations > max_iterations:
                    LOGGER.warning(
                        "Max tool iterations (%s) reached; returning partial response.",
                        max_iterations,
                    )
                    break

                tool_messages, tool_records = await self._handle_tool_calls(turn.tool_calls, on_event)
                executed_tool_calls.extend(tool_records)
                conversation.extend(tool_messages)

            self._log_response_text(response_text)
            LOGGER.debug(
                "Chat turn complete (chars=%s, tool calls=%s)",
                len(response_text),
                len(executed_tool_calls),
            )
            return {
                "prompt": prompt,
                "response": response_text,
                "doc_snapshot": snapshot,
                "tool_calls": executed_tool_calls,
                "graph": self.graph,
            }

        async with self._task_lock:
            task = asyncio.create_task(_runner())
            self._active_task = task

        try:
            return await task
        finally:
            if self._active_task is task:
                self._active_task = None

    def cancel(self) -> None:
        """Cancel the active chat turn, if any."""

        if self._active_task and not self._active_task.done():
            LOGGER.debug("Cancelling active chat task")
            self._active_task.cancel()

    def available_tools(self) -> tuple[str, ...]:
        """Return the names of the registered tools."""

        return tuple(self.tools.keys())

    def _rebuild_graph(self) -> None:
        self._graph = build_agent_graph(tools={name: registration.impl for name, registration in self.tools.items()})

    def _build_messages(self, prompt: str, snapshot: Mapping[str, Any]) -> list[dict[str, str]]:
        user_prompt = prompts.format_user_prompt(prompt, dict(snapshot))
        return [
            {"role": "system", "content": prompts.base_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]

    def _build_metadata(
        self,
        snapshot: Mapping[str, Any],
        runtime_metadata: Mapping[str, str] | None,
    ) -> Optional[Dict[str, str]]:
        metadata: Dict[str, str] = {}
        path = snapshot.get("path")
        if path:
            metadata["doc_path"] = str(path)
        selection = snapshot.get("selection")
        if isinstance(selection, Mapping):
            start = selection.get("start")
            end = selection.get("end")
            metadata["selection"] = f"{start}:{end}"
        if runtime_metadata:
            metadata.update(runtime_metadata)
        return metadata or None

    def _log_response_text(self, response_text: str) -> None:
        settings = getattr(self.client, "settings", None)
        debug_logging_enabled = bool(getattr(settings, "debug_logging", False)) if settings else False
        if not debug_logging_enabled:
            return

        if response_text:
            LOGGER.debug("AI response text:\n%s", response_text)
        else:
            LOGGER.debug("AI response text: <empty>")

    async def _dispatch_event(self, event: AIStreamEvent, handler: ToolCallback | None) -> None:
        if handler is None:
            return
        result = handler(event)
        if inspect.isawaitable(result):
            await result

    async def _invoke_model_turn(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        tool_specs: Sequence[ChatCompletionToolParam] | None,
        metadata: Mapping[str, str] | None,
        on_event: ToolCallback | None,
    ) -> _ModelTurnResult:
        deltas: list[str] = []
        final_chunk: str | None = None
        tool_calls: list[_ToolCallRequest] = []
        async for event in self.client.stream_chat(
            messages=list(conversation),
            tools=list(tool_specs) if tool_specs else None,
            metadata=metadata,
        ):
            await self._dispatch_event(event, on_event)
            if event.type == "content.delta" and event.content:
                deltas.append(event.content)
            elif event.type == "content.done" and event.content:
                final_chunk = str(event.content)
            elif event.type.endswith("tool_calls.function.arguments.done"):
                call_id = self._normalize_tool_call_id(event, len(tool_calls))
                tool_calls.append(
                    _ToolCallRequest(
                        call_id=call_id,
                        name=event.tool_name or "",
                        index=event.tool_index if event.tool_index is not None else len(tool_calls),
                        arguments=event.tool_arguments,
                        parsed=event.parsed,
                    )
                )

        response_text = "".join(deltas)
        if final_chunk:
            if not response_text:
                response_text = final_chunk
            elif not response_text.endswith(final_chunk):
                response_text += final_chunk
        response_text = response_text.strip()
        assistant_message: Dict[str, Any] = {"role": "assistant", "content": response_text or None}
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments or "{}",
                    },
                }
                for call in tool_calls
            ]
        return _ModelTurnResult(assistant_message=assistant_message, response_text=response_text, tool_calls=tool_calls)

    async def _handle_tool_calls(
        self,
        tool_calls: Sequence[_ToolCallRequest],
        on_event: ToolCallback | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        messages: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        for call in tool_calls:
            content, record = await self._execute_tool_call(call, on_event)
            tool_message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": content,
            }
            if call.name:
                tool_message["name"] = call.name
            messages.append(tool_message)
            records.append(record)
        return messages, records

    async def _execute_tool_call(
        self,
        call: _ToolCallRequest,
        on_event: ToolCallback | None,
    ) -> tuple[str, dict[str, Any]]:
        registration = self.tools.get(call.name) if call.name else None
        resolved_arguments = self._coerce_tool_arguments(call.arguments, call.parsed)
        if registration is None:
            message = f"Tool '{call.name or 'unknown'}' is not registered."
            await self._emit_tool_result_event(call, message, None, on_event)
            return message, self._build_tool_record(call, resolved_arguments, message)

        try:
            result = await self._invoke_tool_impl(registration.impl, resolved_arguments)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Tool %s failed", call.name)
            result = f"Tool '{call.name}' failed: {exc}"

        serialized = self._serialize_tool_result(result)
        await self._emit_tool_result_event(call, serialized, result, on_event)
        return serialized, self._build_tool_record(call, resolved_arguments, serialized)

    async def _emit_tool_result_event(
        self,
        call: _ToolCallRequest,
        content: str,
        raw_result: Any,
        on_event: ToolCallback | None,
    ) -> None:
        await self._dispatch_event(
            AIStreamEvent(
                type="tool_calls.result",
                content=content,
                tool_name=call.name or None,
                tool_arguments=content,
                tool_index=call.index,
                parsed=raw_result,
                tool_call_id=call.call_id,
            ),
            on_event,
        )

    def _build_tool_record(
        self,
        call: _ToolCallRequest,
        resolved_arguments: Any,
        serialized_result: str,
    ) -> dict[str, Any]:
        return {
            "id": call.call_id,
            "name": call.name,
            "index": call.index,
            "arguments": call.arguments,
            "parsed": call.parsed,
            "resolved_arguments": resolved_arguments,
            "result": serialized_result,
        }

    def _coerce_tool_arguments(self, raw_arguments: str | None, parsed: Any | None) -> Any:
        if parsed is not None:
            return parsed
        if raw_arguments is None:
            return {}
        text = raw_arguments.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return text

    async def _invoke_tool_impl(self, tool_impl: Any, arguments: Any) -> Any:
        target = getattr(tool_impl, "run", tool_impl)
        if not callable(target):  # pragma: no cover - safety
            raise TypeError(f"Registered tool {tool_impl!r} is not callable")

        try:
            if isinstance(arguments, Mapping):
                result = target(**arguments)
            elif arguments in (None, {}):
                result = target()
            else:
                result = target(arguments)
        except TypeError:
            # Fallback to positional invocation when kwargs mismatch.
            result = target(arguments)

        if inspect.isawaitable(result):
            result = await result
        return result

    def _serialize_tool_result(self, result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(result)

    def _normalize_tool_call_id(self, event: AIStreamEvent, ordinal: int) -> str:
        candidate = (event.tool_call_id or "").strip() if getattr(event, "tool_call_id", None) else ""
        if candidate:
            return candidate
        name = (event.tool_name or "tool").strip() or "tool"
        return f"{name}-{ordinal}-{uuid.uuid4().hex[:8]}"


