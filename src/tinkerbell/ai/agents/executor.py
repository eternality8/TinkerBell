"""Agent executor façade wrapping LangChain/LangGraph interactions."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, cast

try:  # pragma: no cover - optional dependency used when available
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - tiktoken is optional in dev/test envs
    tiktoken = None

from openai.types.chat import ChatCompletionToolParam

from .. import prompts
from ..client import AIClient, AIStreamEvent
from .graph import build_agent_graph

LOGGER = logging.getLogger(__name__)
_PROMPT_HEADROOM = 4_096
_SUGGESTION_SYSTEM_PROMPT = (
    "You are a proactive writing assistant that proposes the next helpful user prompts after reviewing the prior "
    "conversation. Respond ONLY with a JSON array of concise suggestion strings (each under 120 characters). "
    "Return at most {max_suggestions} unique ideas tailored to the conversation."
)

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
    max_tool_iterations: int = 8
    diff_builder_reminder_threshold: int = 3
    max_pending_patch_reminders: int = 2
    max_context_tokens: int = 128_000
    response_token_reserve: int = 16_000
    _graph: Dict[str, Any] = field(init=False, repr=False)
    _max_tool_iterations: int = field(init=False, repr=False)
    _active_task: asyncio.Task[dict] | None = field(default=None, init=False, repr=False)
    _task_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _token_encoder: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tools:
            normalized: Dict[str, ToolRegistration] = {}
            for name, value in self.tools.items():
                if isinstance(value, ToolRegistration):
                    normalized[name] = value
                else:
                    normalized[name] = ToolRegistration(name=name, impl=value)
            self.tools = normalized
        self._max_tool_iterations = self._normalize_iterations(self.max_tool_iterations)
        self.max_tool_iterations = self._max_tool_iterations
        self._rebuild_graph()
        self.configure_context_window(
            max_context_tokens=self.max_context_tokens,
            response_token_reserve=self.response_token_reserve,
        )

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

    def set_max_tool_iterations(self, iterations: int) -> None:
        """Update the maximum allowed tool iterations and rebuild the graph if needed."""

        normalized = self._normalize_iterations(iterations)
        if normalized == self._max_tool_iterations:
            return
        self._max_tool_iterations = normalized
        self.max_tool_iterations = normalized
        self._rebuild_graph()

    def configure_context_window(
        self,
        *,
        max_context_tokens: int | None = None,
        response_token_reserve: int | None = None,
    ) -> None:
        """Normalize and apply context window constraints for prompt planning."""

        if max_context_tokens is not None:
            self.max_context_tokens = self._normalize_context_tokens(max_context_tokens)
        else:
            self.max_context_tokens = self._normalize_context_tokens(self.max_context_tokens)
        if response_token_reserve is not None:
            self.response_token_reserve = self._normalize_response_reserve(response_token_reserve)
        else:
            self.response_token_reserve = self._normalize_response_reserve(self.response_token_reserve)

    def unregister_tool(self, name: str) -> None:
        """Remove a tool and rebuild the agent graph."""

        if name in self.tools:
            self.tools.pop(name)
            LOGGER.debug("Unregistered tool: %s", name)
            self._rebuild_graph()

    def update_client(self, client: AIClient) -> None:
        """Swap the underlying AI client (e.g., when settings change)."""

        self.client = client
        self._token_encoder = None

    async def run_chat(
        self,
        prompt: str,
        doc_snapshot: Mapping[str, Any] | None,
        *,
        metadata: Mapping[str, str] | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
        on_event: ToolCallback | None = None,
    ) -> dict:
        """Execute a chat turn against the compiled agent graph."""

        snapshot = dict(doc_snapshot or {})
        base_messages, completion_budget = self._build_messages(prompt, snapshot, history)
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
            pending_patch_application = False
            diff_builders_since_edit = 0
            patch_reminders_sent = 0

            while True:
                turn_count += 1
                turn = await self._invoke_model_turn(
                    conversation,
                    tool_specs=tool_specs if tool_specs else None,
                    metadata=merged_metadata,
                    on_event=on_event,
                    max_completion_tokens=completion_budget,
                )
                conversation.append(turn.assistant_message)
                response_text = turn.response_text

                if not turn.tool_calls:
                    if pending_patch_application and patch_reminders_sent < self.max_pending_patch_reminders:
                        reminder = self._pending_patch_prompt()
                        LOGGER.debug("Injected patch reminder (pending diff)")
                        conversation.append({"role": "system", "content": reminder})
                        patch_reminders_sent += 1
                        continue
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

                for record in tool_records:
                    name = str(record.get("name") or "").lower()
                    if name == "diff_builder" and not self._tool_call_failed(record):
                        pending_patch_application = True
                        diff_builders_since_edit += 1
                    elif name == "document_edit":
                        if self._tool_call_failed(record):
                            pending_patch_application = True
                        else:
                            pending_patch_application = False
                            diff_builders_since_edit = 0
                            patch_reminders_sent = 0

                if (
                    pending_patch_application
                    and diff_builders_since_edit >= self.diff_builder_reminder_threshold
                ):
                    reminder_text = self._diff_accumulation_prompt(diff_builders_since_edit)
                    LOGGER.debug(
                        "Injected diff_builder consolidation reminder (count=%s)", diff_builders_since_edit
                    )
                    conversation.append({"role": "system", "content": reminder_text})
                    diff_builders_since_edit = 0
                    continue

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

    async def suggest_followups(
        self,
        history: Sequence[Mapping[str, str]],
        *,
        max_suggestions: int = 4,
    ) -> list[str]:
        """Generate contextual follow-up suggestions based on chat history."""

        if not history:
            return []

        messages = self._build_suggestion_messages(history, max_suggestions)
        response_text = await self._complete_simple_chat(messages)
        return self._parse_suggestion_response(response_text, max_suggestions)

    def cancel(self) -> None:
        """Cancel the active chat turn, if any."""

        if self._active_task and not self._active_task.done():
            LOGGER.debug("Cancelling active chat task")
            self._active_task.cancel()

    def available_tools(self) -> tuple[str, ...]:
        """Return the names of the registered tools."""

        return tuple(self.tools.keys())

    async def aclose(self) -> None:
        """Cancel any active work and close the underlying AI client."""

        task = self._active_task
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            if self._active_task is task:
                self._active_task = None
        close = getattr(self.client, "aclose", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    def _rebuild_graph(self) -> None:
        self._graph = build_agent_graph(
            tools={name: registration.impl for name, registration in self.tools.items()},
            max_iterations=self._max_tool_iterations,
        )

    def _build_suggestion_messages(
        self,
        history: Sequence[Mapping[str, str]],
        max_suggestions: int,
    ) -> list[dict[str, str]]:
        transcript_lines: list[str] = []
        for entry in history[-10:]:
            role = (entry.get("role") or "user").lower()
            label = "User" if role == "user" else "Assistant"
            content = str(entry.get("content", "")).strip()
            if not content:
                continue
            if len(content) > 400:
                content = f"{content[:397].rstrip()}…"
            transcript_lines.append(f"{label}: {content}")

        transcript = "\n".join(transcript_lines) or "(no content)"
        system_prompt = _SUGGESTION_SYSTEM_PROMPT.format(max_suggestions=max(1, max_suggestions))
        user_prompt = (
            "Here is the recent conversation between the user and assistant:\n"
            f"{transcript}\n\n"
            "Suggest the next helpful questions or commands the user could ask to keep making progress."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    async def _complete_simple_chat(self, messages: Sequence[Mapping[str, Any]]) -> str:
        chunks: list[str] = []
        final_chunk: str | None = None
        async for event in self.client.stream_chat(messages=messages, temperature=0.3, max_tokens=300):
            if event.type == "content.delta" and event.content:
                chunks.append(str(event.content))
            elif event.type == "content.done" and event.content:
                final_chunk = str(event.content)
        response = "".join(chunks)
        if final_chunk and final_chunk not in response:
            response += final_chunk
        return response.strip()

    def _parse_suggestion_response(self, text: str, max_suggestions: int) -> list[str]:
        if not text:
            return []

        suggestions = self._try_parse_json_suggestions(text, max_suggestions)
        if suggestions:
            return suggestions

        lines = [line.strip(" -*\t") for line in text.splitlines() if line.strip()]
        return self._sanitize_suggestions(lines, max_suggestions)

    def _try_parse_json_suggestions(self, text: str, max_suggestions: int) -> list[str]:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []

        if isinstance(parsed, dict):
            if "suggestions" in parsed and isinstance(parsed["suggestions"], list):
                parsed = parsed["suggestions"]
            else:
                parsed = list(parsed.values())

        if isinstance(parsed, list):
            return self._sanitize_suggestions(parsed, max_suggestions)
        return []

    @staticmethod
    def _normalize_iterations(value: int | None) -> int:
        try:
            candidate = int(value) if value is not None else 8
        except (TypeError, ValueError):
            candidate = 8
        return max(1, min(candidate, 50))

    @staticmethod
    def _normalize_context_tokens(value: int | None) -> int:
        default = 128_000
        try:
            candidate = int(value) if value is not None else default
        except (TypeError, ValueError):
            candidate = default
        return max(32_000, min(candidate, 512_000))

    @staticmethod
    def _normalize_response_reserve(value: int | None) -> int:
        default = 16_000
        try:
            candidate = int(value) if value is not None else default
        except (TypeError, ValueError):
            candidate = default
        return max(4_000, min(candidate, 64_000))

    def _effective_response_reserve(self, context_limit: int) -> int:
        limit = max(0, int(context_limit))
        max_reserve = max(0, limit - _PROMPT_HEADROOM)
        return max(0, min(self.response_token_reserve, max_reserve))

    def _sanitize_suggestions(self, raw_items: Iterable[Any], max_suggestions: int) -> list[str]:
        sanitized: list[str] = []
        seen: set[str] = set()
        limit = max(1, max_suggestions)
        for item in raw_items:
            text = str(item).strip()
            if not text or text in seen:
                continue
            sanitized.append(text)
            seen.add(text)
            if len(sanitized) >= limit:
                break
        return sanitized

    @staticmethod
    def _tool_call_failed(record: Mapping[str, Any]) -> bool:
        result = record.get("result")
        if not isinstance(result, str):
            return False
        name = record.get("name") or ""
        failure_prefix = f"Tool '{name}' failed:"
        return result.startswith(failure_prefix)

    def _pending_patch_prompt(self) -> str:
        return (
            "You generated a diff via diff_builder but did not call document_edit to apply it yet. "
            "Use document_edit with action=\"patch\", include the diff text, and pass the latest document_version before responding."
        )

    def _diff_accumulation_prompt(self, diff_count: int) -> str:
        return (
            f"You've produced {diff_count} diff_builder results without applying them. "
            "Consolidate the change into a single diff and immediately call document_edit (action=\"patch\")."
        )

    def _build_messages(
        self,
        prompt: str,
        snapshot: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[list[dict[str, str]], int | None]:
        user_prompt = prompts.format_user_prompt(prompt, dict(snapshot))
        system_message = {"role": "system", "content": prompts.base_system_prompt()}
        user_message = {"role": "user", "content": user_prompt}

        context_limit = self.max_context_tokens
        reserve = self._effective_response_reserve(context_limit)
        prompt_budget = max(0, context_limit - reserve)

        system_tokens = self._estimate_message_tokens(system_message)
        user_tokens = self._estimate_message_tokens(user_message)
        required_prompt_tokens = system_tokens + user_tokens

        if prompt_budget < required_prompt_tokens and reserve > 0:
            shortfall = required_prompt_tokens - prompt_budget
            reclaimed = min(reserve, shortfall)
            reserve -= reclaimed
            prompt_budget += reclaimed

        history_budget = max(0, prompt_budget - required_prompt_tokens)
        trimmed_history: list[dict[str, str]] = []
        if history:
            trimmed_history = self._sanitize_history(history, limit=60, token_budget=history_budget)

        messages: list[dict[str, str]] = [system_message]
        messages.extend(trimmed_history)
        messages.append(user_message)

        completion_budget = reserve if reserve > 0 else None
        return messages, completion_budget

    def _sanitize_history(
        self,
        history: Sequence[Mapping[str, Any]],
        limit: int = 20,
        *,
        token_budget: int | None = None,
    ) -> list[dict[str, str]]:
        allowed_roles = {"user", "assistant", "system", "tool"}
        window = list(history)[-limit:] if limit else list(history)
        sanitized: list[dict[str, str]] = []
        for entry in window:
            role = str(entry.get("role", "user")).lower()
            if role not in allowed_roles:
                continue
            text = str(entry.get("content", "")).strip()
            if not text:
                continue
            sanitized.append({"role": role, "content": text})
        if token_budget is None:
            return sanitized
        budget = max(0, token_budget)
        if budget == 0 or not sanitized:
            return []
        trimmed: list[dict[str, str]] = []
        remaining = budget
        for entry in reversed(sanitized):
            tokens = self._estimate_message_tokens(entry)
            if trimmed and tokens > remaining:
                break
            if not trimmed and tokens > remaining:
                trimmed.append(entry)
                break
            trimmed.append(entry)
            remaining -= tokens
            if remaining <= 0:
                break
        trimmed.reverse()
        return trimmed

    def _estimate_message_tokens(self, message: Mapping[str, Any]) -> int:
        content = str(message.get("content", "") or "")
        tokens = self._estimate_text_tokens(content)
        if message.get("role"):
            tokens += 4  # allowance for role + message boundary tokens
        return tokens

    def _estimate_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        encoder = self._get_token_encoder()
        if encoder is not None:
            try:
                return len(encoder.encode(text))
            except Exception:  # pragma: no cover - defensive when encoder misbehaves
                self._token_encoder = None
        return max(1, math.ceil(len(text) / 4))

    def _get_token_encoder(self) -> Any | None:
        if tiktoken is None:
            return None
        if self._token_encoder is not None:
            return self._token_encoder
        model_name: str | None = None
        client_settings = getattr(self.client, "settings", None)
        if client_settings is not None:
            model_name = getattr(client_settings, "model", None)
        try:
            encoding = (
                tiktoken.encoding_for_model(model_name)
                if model_name
                else tiktoken.get_encoding("cl100k_base")
            )
        except Exception:  # pragma: no cover - fall back to default encoding
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                encoding = None
        self._token_encoder = encoding
        return encoding

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
        max_completion_tokens: int | None = None,
    ) -> _ModelTurnResult:
        deltas: list[str] = []
        final_chunk: str | None = None
        tool_calls: list[_ToolCallRequest] = []
        stream_kwargs: Dict[str, Any] = {
            "messages": list(conversation),
            "tools": list(tool_specs) if tool_specs else None,
            "metadata": metadata,
        }
        if max_completion_tokens is not None:
            stream_kwargs["max_completion_tokens"] = max_completion_tokens

        async for event in self.client.stream_chat(**stream_kwargs):
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
        index = getattr(event, "tool_index", None)
        if index is None:
            index = ordinal
        try:
            index_text = str(int(index))
        except (TypeError, ValueError):
            index_text = str(ordinal)
        return f"{name}:{index_text}"


