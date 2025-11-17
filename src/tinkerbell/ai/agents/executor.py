"""Agent executor façade wrapping LangChain/LangGraph interactions."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, cast

from openai.types.chat import ChatCompletionToolParam

from .. import prompts
from ..ai_types import AgentConfig
from ..client import AIClient, AIStreamEvent
from ..services import ToolPayload, build_pointer, summarize_tool_content, telemetry as telemetry_service
from ..services.trace_compactor import TraceCompactor
from ...chat.message_model import ToolPointerMessage
from ..services.context_policy import BudgetDecision, ContextBudgetPolicy
from ..services.telemetry import (
    ContextUsageEvent,
    PersistentTelemetrySink,
    TelemetrySink,
    default_telemetry_path,
)
from .graph import build_agent_graph

LOGGER = logging.getLogger(__name__)
_PROMPT_HEADROOM = 4_096
_POINTER_SUMMARY_TOKENS = 512
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
class _MessagePlan:
    """Normalized plan for the prompt and completion budgeting."""

    messages: list[dict[str, str]]
    completion_budget: int | None
    prompt_tokens: int


class ContextBudgetExceeded(RuntimeError):
    """Raised when the context budget policy rejects a prompt."""

    def __init__(self, decision: BudgetDecision):
        super().__init__(f"Context budget exceeded: {decision.reason}")
        self.decision = decision


@dataclass(slots=True)
class ToolRegistration:
    """Metadata stored for each registered tool."""

    name: str
    impl: Any
    description: str | None = None
    parameters: Mapping[str, Any] | None = None
    strict: bool = True
    summarizable: bool = True

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
    telemetry_enabled: bool = False
    telemetry_limit: int = 200
    telemetry_sink: TelemetrySink | None = None
    agent_config: AgentConfig | None = None
    budget_policy: ContextBudgetPolicy | None = None
    _graph: Dict[str, Any] = field(init=False, repr=False)
    _active_task: asyncio.Task[dict] | None = field(default=None, init=False, repr=False)
    _task_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _telemetry_sink: TelemetrySink | None = field(default=None, init=False, repr=False)
    _last_budget_decision: BudgetDecision | None = field(default=None, init=False, repr=False)
    _trace_compactor: TraceCompactor | None = field(default=None, init=False, repr=False)
    _outline_digest_cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.tools:
            normalized: Dict[str, ToolRegistration] = {}
            for name, value in self.tools.items():
                if isinstance(value, ToolRegistration):
                    normalized[name] = value
                else:
                    normalized[name] = ToolRegistration(name=name, impl=value)
            self.tools = normalized
        config = self.agent_config or AgentConfig(max_iterations=self.max_tool_iterations)
        config.max_iterations = self._normalize_iterations(config.max_iterations)
        self.agent_config = config.clamp()
        self.max_tool_iterations = self.agent_config.max_iterations
        self._rebuild_graph()
        self.configure_context_window(
            max_context_tokens=self.max_context_tokens,
            response_token_reserve=self.response_token_reserve,
        )
        self._configure_telemetry_sink()
        self.configure_budget_policy(self.budget_policy)
        self._trace_compactor = TraceCompactor(
            pointer_builder=self._build_tool_pointer,
            estimate_message_tokens=self._estimate_message_tokens,
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
        summarizable: bool | None = None,
    ) -> None:
        """Register (or replace) a tool available to the agent."""

        if summarizable is None:
            attr_flag = getattr(tool, "summarizable", None)
            summarizable_flag = True if attr_flag is None else bool(attr_flag)
        else:
            summarizable_flag = bool(summarizable)
        registration = ToolRegistration(
            name=name,
            impl=tool,
            description=description,
            parameters=parameters,
            strict=True if strict is None else bool(strict),
            summarizable=summarizable_flag,
        )
        self.tools[name] = registration
        LOGGER.debug("Registered tool: %s", name)
        self._rebuild_graph()

    def set_max_tool_iterations(self, iterations: int) -> None:
        """Update the maximum allowed tool iterations and rebuild the graph if needed."""

        normalized = self._normalize_iterations(iterations)
        config = self.agent_config or AgentConfig(max_iterations=self.max_tool_iterations)
        if normalized == config.max_iterations:
            return
        config.max_iterations = normalized
        self.agent_config = config
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

    def configure_budget_policy(self, policy: ContextBudgetPolicy | None) -> None:
        """Update (or initialize) the active budget policy."""

        model_name = getattr(getattr(self.client, "settings", None), "model", None)
        if policy is None:
            policy = ContextBudgetPolicy.disabled(
                model_name=model_name,
                max_context_tokens=self.max_context_tokens,
            )
        self.budget_policy = policy

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
        history: Sequence[Mapping[str, str]] | None = None,
        on_event: ToolCallback | None = None,
    ) -> dict:
        """Execute a chat turn against the compiled agent graph."""

        snapshot = dict(doc_snapshot or {})
        message_plan = self._build_messages(prompt, snapshot, history)
        outline_hint = self._outline_routing_hint(prompt, snapshot)
        base_messages = list(message_plan.messages)
        if outline_hint:
            hint_message = {"role": "system", "content": outline_hint}
            base_messages.insert(1, hint_message)
            message_plan.prompt_tokens += self._estimate_message_tokens(hint_message)
        completion_budget = message_plan.completion_budget
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

        self._evaluate_budget(
            prompt_tokens=message_plan.prompt_tokens,
            response_reserve=completion_budget,
            snapshot=snapshot,
        )

        async def _runner() -> dict:
            if self._trace_compactor is not None:
                self._trace_compactor.reset()
            conversation: list[dict[str, Any]] = list(base_messages)
            conversation_tokens = message_plan.prompt_tokens
            response_text = ""
            turn_count = 0
            tool_iterations = 0
            executed_tool_calls: list[dict[str, Any]] = []
            pending_patch_application = False
            diff_builders_since_edit = 0
            patch_reminders_sent = 0

            turn_metrics = self._new_turn_context(
                snapshot=snapshot,
                prompt_tokens=message_plan.prompt_tokens,
                conversation_length=len(base_messages),
                response_reserve=completion_budget,
            )
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
                conversation_tokens += self._estimate_message_tokens(turn.assistant_message)
                response_text = turn.response_text

                if not turn.tool_calls:
                    if pending_patch_application and patch_reminders_sent < self.max_pending_patch_reminders:
                        reminder = self._pending_patch_prompt()
                        LOGGER.debug("Injected patch reminder (pending diff)")
                        reminder_message = {"role": "system", "content": reminder}
                        conversation.append(reminder_message)
                        conversation_tokens += self._estimate_message_tokens(reminder_message)
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

                tool_messages, tool_records, tool_token_cost = await self._handle_tool_calls(
                    turn.tool_calls,
                    on_event,
                )
                executed_tool_calls.extend(tool_records)
                tool_messages, conversation_tokens = self._compact_tool_messages(
                    tool_messages,
                    tool_records,
                    conversation_tokens=conversation_tokens,
                    response_reserve=completion_budget,
                    snapshot=snapshot,
                )
                conversation.extend(tool_messages)
                turn_metrics["tool_tokens"] += tool_token_cost
                turn_metrics["conversation_length"] = len(conversation)
                self._record_tool_names(turn_metrics, tool_records)
                self._record_tool_metrics(turn_metrics, tool_records)
                guardrail_hints = self._guardrail_hints_from_records(tool_records)
                for hint in guardrail_hints:
                    hint_message = {"role": "system", "content": hint}
                    conversation.append(hint_message)
                    conversation_tokens += self._estimate_message_tokens(hint_message)

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
                    reminder_message = {"role": "system", "content": reminder_text}
                    conversation.append(reminder_message)
                    conversation_tokens += self._estimate_message_tokens(reminder_message)
                    diff_builders_since_edit = 0
                    continue

            turn_metrics["conversation_length"] = len(conversation)
            self._log_response_text(response_text)
            self._emit_context_usage(turn_metrics)
            LOGGER.debug(
                "Chat turn complete (chars=%s, tool calls=%s)",
                len(response_text),
                len(executed_tool_calls),
            )
            compaction_stats = None
            if self._trace_compactor is not None:
                trace_stats = self._trace_compactor.stats_snapshot().as_dict()
                compaction_stats = trace_stats
                if trace_stats.get("total_compactions") or trace_stats.get("tokens_saved"):
                    telemetry_service.emit(
                        "trace_compaction",
                        {
                            "run_id": turn_metrics.get("run_id"),
                            "document_id": turn_metrics.get("document_id"),
                            **trace_stats,
                        },
                    )
            return {
                "prompt": prompt,
                "response": response_text,
                "doc_snapshot": snapshot,
                "tool_calls": executed_tool_calls,
                "graph": self.graph,
                "trace_compaction": compaction_stats,
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

    def configure_telemetry(
        self,
        *,
        enabled: bool | None = None,
        sink: TelemetrySink | None = None,
        limit: int | None = None,
    ) -> None:
        """Update telemetry settings without reconstructing the controller."""

        if enabled is not None:
            self.telemetry_enabled = bool(enabled)
        if limit is not None:
            try:
                self.telemetry_limit = max(20, min(int(limit), 10_000))
            except (TypeError, ValueError):
                pass
        if sink is not None:
            self.telemetry_sink = sink
        self._configure_telemetry_sink()

    def get_recent_context_events(self, limit: int | None = None) -> list[ContextUsageEvent]:
        """Return the most recent context usage events from the active sink."""

        sink = self._telemetry_sink
        if sink is None:
            return []
        if hasattr(sink, "tail"):
            tail = getattr(sink, "tail")
            try:
                return list(tail(limit))  # type: ignore[misc]
            except TypeError:
                return list(tail())  # type: ignore[misc]
        return []

    def get_budget_status(self) -> dict[str, object] | None:
        """Expose the most recent context budget policy decision."""

        if self.budget_policy is None:
            return None
        return self.budget_policy.status_snapshot()

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
        config = self.agent_config or AgentConfig(max_iterations=self.max_tool_iterations)
        config.max_iterations = self._normalize_iterations(config.max_iterations)
        self.agent_config = config.clamp()
        self.max_tool_iterations = self.agent_config.max_iterations
        self._graph = build_agent_graph(
            tools={name: registration.impl for name, registration in self.tools.items()},
            config=self.agent_config,
        )

    def _configure_telemetry_sink(self) -> None:
        limit = self.telemetry_limit
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 200
        limit = max(20, min(limit, 10_000))
        if self.telemetry_sink is None:
            self._telemetry_sink = PersistentTelemetrySink(path=default_telemetry_path(), capacity=limit)
        else:
            self._telemetry_sink = self.telemetry_sink

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
        if result.startswith(failure_prefix):
            return True
        if result.startswith("Tool '") and result.endswith("is not registered."):
            return True
        return False

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

    def _new_turn_context(
        self,
        *,
        snapshot: Mapping[str, Any],
        prompt_tokens: int,
        conversation_length: int,
        response_reserve: int | None,
    ) -> dict[str, Any]:
        document_id = self._resolve_document_id(snapshot)
        model_name = getattr(getattr(self.client, "settings", None), "model", None) or "unknown"
        context = {
            "document_id": document_id,
            "model": model_name,
            "prompt_tokens": int(prompt_tokens),
            "tool_tokens": 0,
            "response_reserve": response_reserve,
            "timestamp": time.time(),
            "conversation_length": conversation_length,
            "tool_names": set(),
            "run_id": uuid.uuid4().hex,
        }
        self._record_budget_usage(
            run_id=context["run_id"],
            prompt_tokens=int(prompt_tokens),
            response_reserve=response_reserve,
        )
        self._copy_snapshot_outline_metrics(context, snapshot)
        self._copy_snapshot_embedding_metadata(context, snapshot)
        return context

    def _record_tool_names(self, context: dict[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
        names = context.get("tool_names")
        if not isinstance(names, set):
            return
        for record in records:
            name = str(record.get("name") or "").strip()
            if name:
                names.add(name)

    def _record_tool_metrics(self, context: dict[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            return
        for record in records:
            name = str(record.get("name") or "").strip().lower()
            if name == "document_outline":
                self._capture_outline_tool_metrics(context, record)
            elif name == "document_find_sections":
                self._capture_retrieval_tool_metrics(context, record)

    def _guardrail_hints_from_records(self, records: Sequence[Mapping[str, Any]]) -> list[str]:
        if not records:
            return []
        hints: list[str] = []
        seen: set[str] = set()
        for record in records:
            payload = self._deserialize_tool_result(record)
            if not isinstance(payload, Mapping):
                continue
            name = str(record.get("name") or "").lower()
            candidate: Sequence[str] | None = None
            if name == "document_outline":
                candidate = self._outline_guardrail_hints(payload)
            elif name == "document_find_sections":
                candidate = self._retrieval_guardrail_hints(payload)
            if not candidate:
                continue
            for entry in candidate:
                text = entry.strip()
                if not text or text in seen:
                    continue
                hints.append(text)
                seen.add(text)
        return hints

    def _outline_guardrail_hints(self, payload: Mapping[str, Any]) -> list[str]:
        hints: list[str] = []
        guardrails = payload.get("guardrails")
        if isinstance(guardrails, Sequence):
            for entry in guardrails:
                if not isinstance(entry, Mapping):
                    continue
                guardrail_type = str(entry.get("type") or "guardrail")
                message = str(entry.get("message") or "").strip()
                action = str(entry.get("action") or "").strip()
                lines: list[str] = []
                if message:
                    lines.append(message)
                if action:
                    lines.append(f"Action: {action}")
                hint = self._format_guardrail_hint(f"DocumentOutlineTool • {guardrail_type}", lines)
                if hint:
                    hints.append(hint)
        status = str(payload.get("status") or "").lower()
        document_id = str(payload.get("document_id") or "this document")
        reason = str(payload.get("reason") or "").strip()
        retry_after_ms = payload.get("retry_after_ms")
        is_stale = bool(payload.get("is_stale")) or status == "stale"
        trimmed_reason = str(payload.get("trimmed_reason") or "").lower()
        outline_available = payload.get("outline_available")
        if status == "pending":
            retry_hint = None
            if isinstance(retry_after_ms, (int, float)) and retry_after_ms > 0:
                retry_seconds = retry_after_ms / 1000.0
                retry_hint = f"Retry after ~{retry_seconds:.1f}s or continue with DocumentSnapshot while the worker rebuilds."
            lines = [f"Outline for {document_id} is still building; treat existing nodes as stale hints only."]
            if retry_hint:
                lines.append(retry_hint)
            hints.append(self._format_guardrail_hint("DocumentOutlineTool", lines))
        elif status == "unsupported_format":
            detail = reason or "unsupported format"
            lines = [
                f"Outline unavailable for {document_id}: {detail}.",
                "Navigate manually with DocumentSnapshot or other tools.",
            ]
            hints.append(self._format_guardrail_hint("DocumentOutlineTool", lines))
        elif status in {"outline_missing", "outline_unavailable", "no_document"}:
            detail = reason or "outline not cached yet"
            lines = [
                f"Outline missing for {document_id} ({detail}).",
                "Queue the worker or rely on selection-scoped snapshots until it exists.",
            ]
            hints.append(self._format_guardrail_hint("DocumentOutlineTool", lines))
        if is_stale:
            lines = [
                f"Outline for {document_id} is stale compared to the latest DocumentSnapshot.",
                "Refresh the outline or treat headings as hints only before editing.",
            ]
            hints.append(self._format_guardrail_hint("DocumentOutlineTool", lines))
        if trimmed_reason == "token_budget":
            lines = [
                "Outline was trimmed by the token budget.",
                "Request fewer levels or hydrate specific pointers before editing deeper sections.",
            ]
            hints.append(self._format_guardrail_hint("DocumentOutlineTool", lines))
        if outline_available is False and status not in {"pending", "unsupported_format", "outline_missing", "outline_unavailable"}:
            lines = [
                f"Outline payload for {document_id} indicated no nodes were returned.",
                "Avoid planning edits that rely on missing structure until the worker succeeds.",
            ]
            hints.append(self._format_guardrail_hint("DocumentOutlineTool", lines))
        return [hint for hint in hints if hint]

    def _retrieval_guardrail_hints(self, payload: Mapping[str, Any]) -> list[str]:
        hints: list[str] = []
        status = str(payload.get("status") or "").lower()
        document_id = str(payload.get("document_id") or "this document")
        reason = str(payload.get("reason") or "").strip()
        fallback_reason = str(payload.get("fallback_reason") or "").strip()
        offline_mode = bool(payload.get("offline_mode"))
        if status == "unsupported_format":
            detail = reason or "unsupported format"
            lines = [
                f"Retrieval disabled for {document_id}: {detail}.",
                "Use DocumentSnapshot, manual navigation, or outline pointers instead.",
            ]
            hints.append(self._format_guardrail_hint("DocumentFindSectionsTool", lines))
        if offline_mode or status in {"offline_fallback", "offline_no_results"}:
            label_reason = fallback_reason or ("offline mode" if offline_mode else "fallback strategy")
            lines = [
                f"Retrieval is running without embeddings ({label_reason}).",
                "Treat matches as low-confidence hints and rehydrate via DocumentSnapshot before editing.",
            ]
            hints.append(self._format_guardrail_hint("DocumentFindSectionsTool", lines))
        if status == "offline_no_results":
            lines = [
                f"Offline fallback could not find matches for {document_id}.",
                "Try a different query or scan the outline/snapshot manually.",
            ]
            hints.append(self._format_guardrail_hint("DocumentFindSectionsTool", lines))
        return [hint for hint in hints if hint]

    def _format_guardrail_hint(self, source: str, lines: Sequence[str]) -> str:
        filtered = [str(line).strip() for line in lines if str(line).strip()]
        if not filtered:
            return ""
        body = "\n".join(f"- {line}" for line in filtered)
        return f"Guardrail hint ({source}):\n{body}"

    def _capture_outline_tool_metrics(self, context: dict[str, Any], record: Mapping[str, Any]) -> None:
        payload = self._deserialize_tool_result(record)
        if not isinstance(payload, Mapping):
            return
        digest = payload.get("outline_digest") or payload.get("outline_hash")
        if digest:
            context["outline_digest"] = str(digest)
        version_id = self._coerce_optional_int(payload.get("version_id"))
        if version_id is not None:
            context["outline_version_id"] = version_id
        status = payload.get("status")
        if status:
            context["outline_status"] = str(status)
        node_count = self._coerce_optional_int(payload.get("node_count"))
        if node_count is not None:
            context["outline_node_count"] = node_count
        token_count = self._coerce_optional_int(payload.get("token_count"))
        if token_count is not None:
            context["outline_token_count"] = token_count
        trimmed = payload.get("trimmed")
        if isinstance(trimmed, bool):
            context["outline_trimmed"] = trimmed
        is_stale = payload.get("is_stale")
        if isinstance(is_stale, bool):
            context["outline_is_stale"] = is_stale
            if status is None:
                context["outline_status"] = "stale" if is_stale else "ok"
        latency = self._coerce_optional_float(record.get("duration_ms"))
        if latency is not None:
            context["outline_latency_ms"] = latency
        generated_at = payload.get("generated_at")
        age = self._outline_age_from_timestamp(generated_at)
        if age is not None:
            context["outline_age_seconds"] = age

    def _capture_retrieval_tool_metrics(self, context: dict[str, Any], record: Mapping[str, Any]) -> None:
        payload = self._deserialize_tool_result(record)
        if not isinstance(payload, Mapping):
            return
        strategy = payload.get("strategy")
        if strategy:
            context["retrieval_strategy"] = str(strategy)
        status = payload.get("status")
        if status:
            context["retrieval_status"] = str(status)
        pointer_count = payload.get("pointers")
        if isinstance(pointer_count, Sequence) and not isinstance(pointer_count, (str, bytes)):
            context["retrieval_pointer_count"] = len(pointer_count)
        latency = payload.get("latency_ms")
        latency_value = self._coerce_optional_float(latency)
        if latency_value is None:
            latency_value = self._coerce_optional_float(record.get("duration_ms"))
        if latency_value is not None:
            context["retrieval_latency_ms"] = latency_value

    def _deserialize_tool_result(self, record: Mapping[str, Any]) -> Mapping[str, Any] | None:
        result_text = record.get("result")
        if not isinstance(result_text, str) or not result_text.strip():
            return None
        try:
            parsed = json.loads(result_text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, Mapping):
            return parsed
        return None

    def _copy_snapshot_outline_metrics(self, context: dict[str, Any], snapshot: Mapping[str, Any]) -> None:
        if not isinstance(snapshot, Mapping):
            return
        digest = snapshot.get("outline_digest")
        if digest:
            context["outline_digest"] = str(digest)
        status = snapshot.get("outline_status")
        if status:
            context["outline_status"] = str(status)
        version_id = self._coerce_optional_int(snapshot.get("outline_version_id"))
        if version_id is not None:
            context["outline_version_id"] = version_id
        latency = self._coerce_optional_float(snapshot.get("outline_latency_ms"))
        if latency is not None:
            context["outline_latency_ms"] = latency
        node_count = self._coerce_optional_int(snapshot.get("outline_node_count"))
        if node_count is not None:
            context["outline_node_count"] = node_count
        token_count = self._coerce_optional_int(snapshot.get("outline_token_count"))
        if token_count is not None:
            context["outline_token_count"] = token_count
        trimmed = snapshot.get("outline_trimmed")
        if isinstance(trimmed, bool):
            context["outline_trimmed"] = trimmed
        is_stale = snapshot.get("outline_is_stale")
        if isinstance(is_stale, bool):
            context["outline_is_stale"] = is_stale
        age_seconds = self._coerce_optional_float(snapshot.get("outline_age_seconds"))
        if age_seconds is not None:
            context["outline_age_seconds"] = age_seconds
        else:
            completed = snapshot.get("outline_completed_at")
            age_from_completed = self._outline_age_from_timestamp(completed)
            if age_from_completed is not None:
                context["outline_age_seconds"] = age_from_completed

    def _copy_snapshot_embedding_metadata(self, context: dict[str, Any], snapshot: Mapping[str, Any]) -> None:
        if not isinstance(snapshot, Mapping):
            return
        backend = snapshot.get("embedding_backend")
        if backend:
            context["embedding_backend"] = str(backend)
        model = snapshot.get("embedding_model")
        if model:
            context["embedding_model"] = str(model)
        status = snapshot.get("embedding_status")
        if status:
            context["embedding_status"] = str(status)
        detail = snapshot.get("embedding_detail")
        if detail:
            context["embedding_detail"] = str(detail)

    def _outline_age_from_timestamp(self, value: object) -> float | None:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            return max(0.0, time.time() - float(value))
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return None
            return max(0.0, time.time() - parsed.timestamp())
        return None

    @staticmethod
    def _coerce_optional_int(value: object) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_optional_float(value: object) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_optional_str(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _evaluate_budget(
        self,
        *,
        prompt_tokens: int,
        response_reserve: int | None,
        snapshot: Mapping[str, Any],
        pending_tool_tokens: int = 0,
        suppress_telemetry: bool = False,
    ) -> BudgetDecision | None:
        policy = self.budget_policy
        if policy is None:
            return None
        decision = policy.tokens_available(
            prompt_tokens=prompt_tokens,
            response_reserve=response_reserve,
            pending_tool_tokens=pending_tool_tokens,
            document_id=self._resolve_document_id(snapshot),
        )
        self._last_budget_decision = decision
        emitter = getattr(telemetry_service, "emit", None)
        if not suppress_telemetry and callable(emitter):
            emitter("context_budget_decision", decision.as_payload())
        if decision.verdict == "reject" and not decision.dry_run:
            raise ContextBudgetExceeded(decision)
        return decision

    def _record_budget_usage(
        self,
        *,
        run_id: str,
        prompt_tokens: int,
        response_reserve: int | None,
    ) -> None:
        policy = self.budget_policy
        if policy is None:
            return
        policy.record_usage(
            run_id,
            prompt_tokens=prompt_tokens,
            response_reserve=response_reserve,
        )

    def _emit_context_usage(self, context: dict[str, Any]) -> None:
        if not self.telemetry_enabled or self._telemetry_sink is None:
            return
        tool_names = context.get("tool_names")
        if isinstance(tool_names, set):
            tool_names_tuple = tuple(sorted(tool_names))
        else:
            tool_names_tuple = ()
        event = ContextUsageEvent(
            document_id=context.get("document_id"),
            model=str(context.get("model") or "unknown"),
            prompt_tokens=int(context.get("prompt_tokens", 0)),
            tool_tokens=int(context.get("tool_tokens", 0)),
            response_reserve=context.get("response_reserve"),
            timestamp=float(context.get("timestamp", time.time())),
            conversation_length=int(context.get("conversation_length", 0)),
            tool_names=tool_names_tuple,
            run_id=str(context.get("run_id") or uuid.uuid4().hex),
            embedding_backend=self._coerce_optional_str(context.get("embedding_backend")),
            embedding_model=self._coerce_optional_str(context.get("embedding_model")),
            embedding_status=self._coerce_optional_str(context.get("embedding_status")),
            embedding_detail=self._coerce_optional_str(context.get("embedding_detail")),
            outline_digest=self._coerce_optional_str(context.get("outline_digest")),
            outline_status=self._coerce_optional_str(context.get("outline_status")),
            outline_version_id=self._coerce_optional_int(context.get("outline_version_id")),
            outline_latency_ms=self._coerce_optional_float(context.get("outline_latency_ms")),
            outline_node_count=self._coerce_optional_int(context.get("outline_node_count")),
            outline_token_count=self._coerce_optional_int(context.get("outline_token_count")),
            outline_trimmed=context.get("outline_trimmed"),
            outline_is_stale=context.get("outline_is_stale"),
            outline_age_seconds=self._coerce_optional_float(context.get("outline_age_seconds")),
            retrieval_status=self._coerce_optional_str(context.get("retrieval_status")),
            retrieval_strategy=self._coerce_optional_str(context.get("retrieval_strategy")),
            retrieval_latency_ms=self._coerce_optional_float(context.get("retrieval_latency_ms")),
            retrieval_pointer_count=self._coerce_optional_int(context.get("retrieval_pointer_count")),
        )
        try:
            self._telemetry_sink.record(event)
        except Exception:  # pragma: no cover - sink errors should not break chat
            LOGGER.debug("Telemetry sink rejected event", exc_info=True)
        else:
            LOGGER.debug(
                "Telemetry event recorded (document=%s, prompt=%s tokens, tools=%s tokens)",
                event.document_id,
                event.prompt_tokens,
                event.tool_tokens,
            )

    @staticmethod
    def _resolve_document_id(snapshot: Mapping[str, Any]) -> str | None:
        for key in ("document_id", "tab_id", "id"):
            value = snapshot.get(key)
            if value:
                return str(value)
        path = snapshot.get("path")
        if path:
            return str(path)
        version = snapshot.get("version")
        if version:
            return str(version)
        return None

    def _build_messages(
        self,
        prompt: str,
        snapshot: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> _MessagePlan:
        model_name = getattr(getattr(self.client, "settings", None), "model", None)
        user_prompt = prompts.format_user_prompt(prompt, dict(snapshot), model_name=model_name)
        system_message = {"role": "system", "content": prompts.base_system_prompt(model_name=model_name)}
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
        prompt_tokens = sum(self._estimate_message_tokens(message) for message in messages)
        return _MessagePlan(messages=messages, completion_budget=completion_budget, prompt_tokens=prompt_tokens)

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
        counter_fn = getattr(self.client, "count_tokens", None)
        if callable(counter_fn):
            try:
                value: Any = counter_fn(text)
                return int(value)
            except Exception:  # pragma: no cover - defensive fallback
                LOGGER.debug("AI client token counter failed; using heuristic", exc_info=True)
        byte_length = len(text.encode("utf-8", errors="ignore"))
        return max(1, math.ceil(byte_length / 4))

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

    def _outline_routing_hint(self, prompt: str, snapshot: Mapping[str, Any]) -> str | None:
        prompt_text = (prompt or "").strip().lower()
        if not prompt_text and not snapshot:
            return None
        doc_id = self._resolve_document_id(snapshot)
        raw_text = snapshot.get("text")
        if isinstance(raw_text, str):
            doc_chars = len(raw_text)
        else:
            try:
                doc_chars = int(snapshot.get("char_count") or 0)
            except (TypeError, ValueError):
                doc_chars = 0

        outline_keywords = (
            "outline",
            "heading",
            "headings",
            "section",
            "sections",
            "toc",
            "table of contents",
            "chapter",
        )
        retrieval_keywords = (
            "find section",
            "find heading",
            "find chapter",
            "locate",
            "where is",
            "which section",
            "quote",
            "passage",
            "excerpt",
        )

        def _contains(text: str, keywords: tuple[str, ...]) -> bool:
            return any(keyword in text for keyword in keywords)

        mentions_outline = _contains(prompt_text, outline_keywords)
        mentions_retrieval = _contains(prompt_text, retrieval_keywords)
        large_doc = doc_chars >= prompts.LARGE_DOC_CHAR_THRESHOLD if doc_chars else False

        guidance: list[str] = []
        if mentions_outline or large_doc:
            reasons: list[str] = []
            if large_doc:
                reasons.append(f"the document is large (~{doc_chars:,} chars)")
            if mentions_outline:
                reasons.append("the user referenced headings/sections")
            reason_text = " and ".join(reasons) if reasons else "document context"
            guidance.append(
                f"Call DocumentOutlineTool first because {reason_text}. Compare outline_digest values to avoid redundant calls."
            )
        if mentions_retrieval:
            guidance.append(
                "After reviewing the outline, call DocumentFindSectionsTool to pull the passages requested before drafting edits."
            )

        digest_hint: str | None = None
        digest = str(snapshot.get("outline_digest") or "").strip()
        if digest and doc_id:
            previous = self._outline_digest_cache.get(doc_id)
            self._outline_digest_cache[doc_id] = digest
            digest_prefix = digest[:8]
            if previous and previous == digest:
                digest_hint = (
                    f"Outline digest {digest_prefix}… matches your prior fetch this session; reuse the previous outline data unless it was marked stale."
                )
            else:
                digest_hint = (
                    f"Outline digest updated to {digest_prefix}…; use this version when reasoning about structure."
                )

        if not guidance and not digest_hint:
            return None

        lines = ["Controller hint:"]
        for entry in guidance:
            lines.append(f"- {entry}")
        if digest_hint:
            lines.append(f"- {digest_hint}")
        if not guidance and digest_hint:
            lines.append("- Only re-run DocumentOutlineTool if the digest changes or the tool reports stale data.")
        return "\n".join(lines)

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
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        messages: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        token_cost = 0
        for call in tool_calls:
            registration = self.tools.get(call.name) if call.name else None
            summarizable = getattr(registration, "summarizable", True) if registration is not None else True
            started_at = time.time()
            start_perf = time.perf_counter()
            content, resolved_arguments, raw_result = await self._execute_tool_call(call, registration, on_event)
            duration_ms = max(0.0, (time.perf_counter() - start_perf) * 1000.0)
            argument_tokens = self._estimate_text_tokens(call.arguments or "")
            result_tokens = self._estimate_text_tokens(content)
            call_token_cost = argument_tokens + result_tokens
            tool_message: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": content,
            }
            if call.name:
                tool_message["name"] = call.name
            messages.append(tool_message)
            record = self._build_tool_record(
                call,
                resolved_arguments,
                content,
                call_token_cost,
                duration_ms,
                raw_result,
                summarizable=summarizable,
                started_at=started_at,
            )
            records.append(record)
            token_cost += call_token_cost
        return messages, records, token_cost

    def _compact_tool_messages(
        self,
        messages: list[dict[str, Any]],
        records: list[dict[str, Any]],
        *,
        conversation_tokens: int,
        response_reserve: int | None,
        snapshot: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        if not messages:
            return messages, conversation_tokens
        updated_tokens = conversation_tokens
        compacted: list[dict[str, Any]] = []
        compactor = self._trace_compactor

        def _evaluate(prompt_tokens: int, pending_tokens: int) -> BudgetDecision | None:
            return self._evaluate_budget(
                prompt_tokens=prompt_tokens,
                response_reserve=response_reserve,
                snapshot=snapshot,
                pending_tool_tokens=pending_tokens,
                suppress_telemetry=True,
            )

        for idx, message in enumerate(messages):
            record = records[idx] if idx < len(records) else {}
            if not isinstance(record, dict):
                record = dict(record)
                if idx < len(records):
                    records[idx] = record
            summarizable = bool(record.get("summarizable", True))
            content_text = str(message.get("content") or "")
            entry = None
            if compactor is not None:
                entry = compactor.new_entry(
                    message,
                    record,
                    raw_content=content_text,
                    summarizable=summarizable,
                )
            pending_tokens = self._estimate_text_tokens(content_text)
            decision = _evaluate(updated_tokens, pending_tokens)
            if compactor is not None and decision is not None and decision.verdict == "needs_summary":
                updated_tokens, decision = compactor.compact_history(
                    evaluate=_evaluate,
                    conversation_tokens=updated_tokens,
                    pending_tokens=pending_tokens,
                )
            if (
                decision is not None
                and decision.verdict == "needs_summary"
                and summarizable
                and content_text
            ):
                if compactor is not None and entry is not None:
                    compactor.compact_entry(entry)
                else:
                    pointer = self._build_tool_pointer(record, content_text)
                    message["content"] = pointer.as_chat_content()
                    record["pointer"] = pointer.as_dict()
                pending_tokens = self._estimate_text_tokens(message.get("content") or "")
                decision = _evaluate(updated_tokens, pending_tokens)
            if decision is not None and decision.verdict == "reject" and not decision.dry_run:
                raise ContextBudgetExceeded(decision)
            message_tokens = self._estimate_message_tokens(message)
            updated_tokens += message_tokens
            if compactor is not None and entry is not None:
                compactor.commit_entry(entry, current_tokens=message_tokens)
            compacted.append(message)
        return compacted, updated_tokens

    def _build_tool_pointer(self, record: Mapping[str, Any], content_text: str) -> ToolPointerMessage:
        tool_name = str(record.get("name") or "tool")
        arguments = record.get("resolved_arguments")
        if not isinstance(arguments, Mapping):
            arguments = None
        payload = ToolPayload(
            name=tool_name,
            content=content_text,
            arguments=arguments,
            metadata={
                "tool_call_id": record.get("id"),
                "status": record.get("status"),
            },
        )
        summary = summarize_tool_content(payload, budget_tokens=_POINTER_SUMMARY_TOKENS)
        instructions = self._pointer_rehydrate_instructions(tool_name, arguments)
        pointer = build_pointer(summary, tool_name=tool_name, rehydrate_instructions=instructions)
        pointer.metadata.setdefault("source", "context_budget")
        return pointer

    @staticmethod
    def _pointer_rehydrate_instructions(tool_name: str, arguments: Mapping[str, Any] | None) -> str:
        if arguments:
            try:
                encoded = json.dumps(arguments, ensure_ascii=False)
            except (TypeError, ValueError):
                encoded = str(arguments)
            if len(encoded) > 180:
                encoded = f"{encoded[:177].rstrip()}…"
            return (
                f"Re-run {tool_name} with arguments similar to {encoded} to recover the full payload pointed to above."
            )
        return f"Re-run {tool_name} to recover the full payload referenced by this pointer."

    async def _execute_tool_call(
        self,
        call: _ToolCallRequest,
        registration: ToolRegistration | None,
        on_event: ToolCallback | None,
    ) -> tuple[str, Any, Any]:
        resolved_arguments = self._coerce_tool_arguments(call.arguments, call.parsed)
        if registration is None:
            message = f"Tool '{call.name or 'unknown'}' is not registered."
            await self._emit_tool_result_event(call, message, None, on_event)
            return message, resolved_arguments, None

        try:
            result = await self._invoke_tool_impl(registration.impl, resolved_arguments)
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Tool %s failed", call.name)
            result = f"Tool '{call.name}' failed: {exc}"

        serialized = self._serialize_tool_result(result)
        await self._emit_tool_result_event(call, serialized, result, on_event)
        return serialized, resolved_arguments, result

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
        tokens_used: int,
        duration_ms: float,
        raw_result: Any,
        *,
        started_at: float,
        summarizable: bool,
    ) -> dict[str, Any]:
        diff_summary = self._summarize_tool_result(call.name, resolved_arguments, raw_result, serialized_result)
        status = self._derive_tool_status(call.name, serialized_result)
        return {
            "id": call.call_id,
            "name": call.name,
            "index": call.index,
            "arguments": call.arguments,
            "parsed": call.parsed,
            "resolved_arguments": resolved_arguments,
            "result": serialized_result,
            "status": status,
            "tokens_used": tokens_used,
            "duration_ms": round(duration_ms, 3),
            "started_at": started_at,
            "diff_summary": diff_summary,
            "summarizable": summarizable,
        }

    def _derive_tool_status(self, tool_name: str | None, serialized_result: str) -> str:
        probe = {"name": tool_name, "result": serialized_result}
        return "failed" if self._tool_call_failed(probe) else "ok"

    def _summarize_tool_result(
        self,
        tool_name: str | None,
        resolved_arguments: Any,
        raw_result: Any,
        serialized_result: str,
    ) -> str | None:
        diff_source: str | None = None
        name = (tool_name or "").lower()
        if name == "diff_builder":
            if isinstance(raw_result, str) and raw_result.strip():
                diff_source = raw_result
            elif serialized_result.strip():
                diff_source = serialized_result
        elif name == "document_edit" and isinstance(resolved_arguments, Mapping):
            diff_value = resolved_arguments.get("diff")
            if isinstance(diff_value, str):
                diff_source = diff_value
        if not diff_source:
            return None
        return self._summarize_diff_text(diff_source)

    @staticmethod
    def _summarize_diff_text(diff_text: str) -> str:
        lines = diff_text.splitlines()
        additions = sum(1 for line in lines if line.startswith("+") and not line.startswith("+++"))
        deletions = sum(1 for line in lines if line.startswith("-") and not line.startswith("---"))
        hunk_count = sum(1 for line in lines if line.startswith("@@")) or (1 if lines else 0)
        return f"+{additions}/-{deletions} lines across {hunk_count} hunk(s)"

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


