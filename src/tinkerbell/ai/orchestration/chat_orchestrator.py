"""Main chat turn orchestration loop."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import uuid
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
)

from openai.types.chat import ChatCompletionToolParam

from ...chat.message_model import ToolPointerMessage
from ...services import telemetry as telemetry_service
from ..ai_types import SubagentJob
from ..client import AIStreamEvent
from ..services.summarizer import ToolPayload, build_pointer, summarize_tool_content
from .budget_manager import BudgetDecision, ContextBudgetExceeded
from .chunk_flow import ChunkFlowTracker
from .event_log import ChatEventLogger
from .guardrail_hints import (
    format_guardrail_hint,
    outline_guardrail_hints,
    retrieval_guardrail_hints,
)
from .model_types import MessagePlan, ModelTurnResult, ToolCallRequest
from .tool_call_parser import (
    TOOL_MARKER_TRANSLATION,
    TOOL_CALLS_BLOCK_RE,
    TOOL_CALL_ENTRY_RE,
    parsed_tool_call_id,
    try_parse_json_block,
)
from .turn_context import TurnContext
from .turn_tracking import PlotLoopTracker, SnapshotRefreshTracker

if TYPE_CHECKING:
    from .controller import AIController

LOGGER = logging.getLogger(__name__)

_POINTER_SUMMARY_TOKENS = 512

ToolCallback = Callable[[AIStreamEvent], Awaitable[None] | None]


class ChatOrchestrator:
    """Orchestrates the main chat turn execution loop.
    
    Manages:
    - Turn iteration with tool call handling
    - Subagent pipeline integration
    - Guardrail hint injection
    - Response follow-up prompting
    - Trace compaction
    - Event logging
    """

    def __init__(
        self,
        controller: "AIController",
    ) -> None:
        """Initialize the chat orchestrator.
        
        Args:
            controller: Parent AIController for state access.
        """
        self._controller = controller

    @property
    def _client(self):
        return self._controller.client

    @property
    def _tools(self):
        return self._controller.tools

    @property
    def _max_tool_iterations(self) -> int:
        return self._controller.max_tool_iterations

    @property
    def _max_edits_without_snapshot(self) -> int:
        return self._controller.max_edits_without_snapshot

    @property
    def _temperature(self) -> float:
        return self._controller.temperature

    @property
    def _diff_builder_reminder_threshold(self) -> int:
        return self._controller.diff_builder_reminder_threshold

    @property
    def _max_pending_patch_reminders(self) -> int:
        return self._controller.max_pending_patch_reminders

    @property
    def _max_tool_followup_prompts(self) -> int:
        return self._controller.max_tool_followup_prompts

    @property
    def _max_tool_followup_user_prompts(self) -> int:
        return self._controller.max_tool_followup_user_prompts

    async def run(
        self,
        prompt: str,
        snapshot: Mapping[str, Any],
        *,
        metadata: Mapping[str, str] | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
        on_event: ToolCallback | None = None,
    ) -> dict:
        """Execute a complete chat turn.
        
        Args:
            prompt: User prompt text.
            snapshot: Document snapshot.
            metadata: Optional metadata to include.
            history: Conversation history.
            on_event: Callback for stream events.
            
        Returns:
            Result dictionary with response, tool calls, etc.
        """
        ctrl = self._controller
        
        # Build messages and hints
        message_plan = ctrl._build_messages(prompt, snapshot, history)
        outline_hint = ctrl._outline_routing_hint(prompt, snapshot)
        analysis_hint = ctrl._analysis_hint_message(snapshot)
        
        base_messages = list(message_plan.messages)
        insert_index = 1
        if analysis_hint:
            base_messages.insert(insert_index, analysis_hint)
            message_plan.prompt_tokens += ctrl._estimate_message_tokens(analysis_hint)
            insert_index += 1
        if outline_hint:
            hint_message = {"role": "system", "content": outline_hint}
            base_messages.insert(insert_index, hint_message)
            message_plan.prompt_tokens += ctrl._estimate_message_tokens(hint_message)
            insert_index += 1
            
        completion_budget = message_plan.completion_budget
        merged_metadata = self._build_metadata(snapshot, metadata)
        tool_specs = [registration.as_openai_tool() for registration in ctrl.tools.values()]
        max_iterations = ctrl._graph.get("metadata", {}).get("max_iterations")
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            max_iterations = 8

        LOGGER.debug(
            "Starting chat turn (prompt length=%s, tools=%s)",
            len(prompt),
            list(ctrl.tools.keys()),
        )

        ctrl._evaluate_budget(
            prompt_tokens=message_plan.prompt_tokens,
            response_reserve=completion_budget,
            snapshot=snapshot,
        )

        async def _runner() -> dict:
            return await self._execute_turn_loop(
                prompt=prompt,
                snapshot=snapshot,
                base_messages=base_messages,
                message_plan=message_plan,
                completion_budget=completion_budget,
                merged_metadata=merged_metadata,
                tool_specs=tool_specs,
                max_iterations=max_iterations,
                on_event=on_event,
                history=history,
            )

        async with ctrl._task_lock:
            task = asyncio.create_task(_runner())
            ctrl._active_task = task

        try:
            return await task
        finally:
            if ctrl._active_task is task:
                ctrl._active_task = None

    async def _execute_turn_loop(
        self,
        *,
        prompt: str,
        snapshot: Mapping[str, Any],
        base_messages: list[dict[str, Any]],
        message_plan: MessagePlan,
        completion_budget: int | None,
        merged_metadata: dict[str, str] | None,
        tool_specs: list[ChatCompletionToolParam],
        max_iterations: int,
        on_event: ToolCallback | None,
        history: Sequence[Mapping[str, str]] | None,
    ) -> dict:
        """Execute the main turn loop with tool iterations."""
        ctrl = self._controller
        
        if ctrl._trace_compactor is not None:
            ctrl._trace_compactor.reset()
        
        # Create and set turn context to pin tab_id for this turn
        turn_context = TurnContext.from_snapshot(snapshot)
        if ctrl._tool_dispatcher is not None:
            ctrl._tool_dispatcher.set_turn_context(turn_context)
            LOGGER.debug(
                "Turn context created: turn_id=%s, pinned_tab_id=%s",
                turn_context.turn_id,
                turn_context.pinned_tab_id,
            )
        
        # Create trackers
        chunk_tracker = ChunkFlowTracker(document_id=ctrl._resolve_document_id(snapshot))
        ctrl._chunk_flow_tracker = chunk_tracker
        
        refresh_tracker: SnapshotRefreshTracker | None = None
        if self._max_edits_without_snapshot > 0:
            refresh_tracker = SnapshotRefreshTracker(
                document_id=ctrl._resolve_document_id(snapshot),
                threshold=self._max_edits_without_snapshot,
            )
        ctrl._snapshot_refresh_tracker = refresh_tracker
        
        plot_tracker: PlotLoopTracker | None = None
        if ctrl._should_enforce_plot_loop(snapshot):
            plot_tracker = PlotLoopTracker(document_id=ctrl._resolve_document_id(snapshot))
        ctrl._plot_loop_tracker = plot_tracker
        
        turn_metrics = ctrl._new_turn_context(
            snapshot=snapshot,
            prompt_tokens=message_plan.prompt_tokens,
            conversation_length=len(base_messages),
            response_reserve=completion_budget,
        )
        
        document_id = ctrl._resolve_document_id(snapshot)
        document_path = snapshot.get("path") or snapshot.get("tab_name")
        event_logger = ctrl._event_logger or ChatEventLogger(enabled=False)
        
        log_run = event_logger.start_run(
            run_id=str(turn_metrics.get("run_id")),
            prompt=prompt,
            document_id=document_id,
            document_path=str(document_path) if document_path else None,
            snapshot=snapshot,
            metadata=merged_metadata,
            history=history or None,
        )
        log_context = log_run
        active_log = log_context.__enter__()
        
        try:
            conversation: list[dict[str, Any]] = list(base_messages)
            conversation_tokens = message_plan.prompt_tokens
            response_text = ""
            turn_count = 0
            tool_iterations = 0
            executed_tool_calls: list[dict[str, Any]] = []
            pending_patch_application = False
            diff_builders_since_edit = 0
            patch_reminders_sent = 0
            tool_followup_prompts_sent = 0
            tool_followup_user_prompts_sent = 0
            plot_loop_reminders_sent = 0
            subagent_jobs: list[SubagentJob] = []
            subagent_messages: list[dict[str, str]] = []
            
            # Run subagent pipeline if enabled
            runtime = ctrl._subagent_runtime
            if runtime is not None and runtime.manager is not None and ctrl.subagent_config.enabled:
                try:
                    subagent_jobs, subagent_messages = await ctrl._run_subagent_pipeline(
                        prompt=prompt,
                        snapshot=snapshot,
                        turn_context=turn_metrics,
                    )
                except Exception:  # pragma: no cover - defensive guard
                    LOGGER.debug("Subagent pipeline failed; continuing without helper context", exc_info=True)
                    subagent_jobs = []
                    subagent_messages = []
                    
                job_pointer_ids = tuple(job.job_id for job in subagent_jobs if job.job_id)
                document_id_hint = ctrl._resolve_document_id(snapshot)
                
                for message in subagent_messages:
                    pending_tokens = ctrl._estimate_message_tokens(message)
                    decision = ctrl._evaluate_budget(
                        prompt_tokens=conversation_tokens,
                        response_reserve=completion_budget,
                        snapshot=snapshot,
                        pending_tool_tokens=pending_tokens,
                        suppress_telemetry=True,
                    )
                    if decision is not None and decision.verdict == "reject" and not decision.dry_run:
                        LOGGER.debug("Skipping subagent summary due to budget limits")
                        break
                    conversation.append(message)
                    conversation_tokens += pending_tokens
                    turn_metrics["prompt_tokens"] += pending_tokens
                    ctrl._register_subagent_trace_entry(
                        message,
                        token_count=pending_tokens,
                        job_ids=job_pointer_ids,
                        document_id=document_id_hint,
                    )

            # Main turn loop
            while True:
                turn_count += 1
                turn = await self._invoke_model_turn(
                    conversation,
                    tool_specs=tool_specs if tool_specs else None,
                    metadata=merged_metadata,
                    on_event=on_event,
                    max_completion_tokens=completion_budget,
                )
                assistant_message = turn.assistant_message
                assistant_tokens = ctrl._estimate_message_tokens(assistant_message)
                conversation.append(assistant_message)
                conversation_tokens += assistant_tokens
                response_text = turn.response_text
                active_log.log_assistant_message(
                    turn_index=turn_count,
                    message=assistant_message,
                    response_text=response_text,
                    tool_calls=[
                        {
                            "id": call.call_id,
                            "name": call.name,
                            "index": call.index,
                            "arguments": call.arguments,
                            "parsed": call.parsed,
                        }
                        for call in turn.tool_calls
                    ],
                )

                if not turn.tool_calls:
                    fallback_applied = False
                    if not response_text and executed_tool_calls:
                        if tool_followup_prompts_sent < self._max_tool_followup_prompts:
                            reminder_text = self._tool_only_response_prompt()
                            LOGGER.debug(
                                "Injected tool follow-up reminder (empty assistant response, attempt=%s)",
                                tool_followup_prompts_sent + 1,
                            )
                            reminder_message = {"role": "system", "content": reminder_text}
                            conversation.append(reminder_message)
                            conversation_tokens += ctrl._estimate_message_tokens(reminder_message)
                            tool_followup_prompts_sent += 1
                            continue
                        if tool_followup_user_prompts_sent < self._max_tool_followup_user_prompts:
                            user_followup = self._tool_only_response_user_prompt(executed_tool_calls)
                            LOGGER.debug(
                                "Injected tool follow-up user prompt (empty assistant response, attempt=%s)",
                                tool_followup_user_prompts_sent + 1,
                            )
                            user_message = {"role": "user", "content": user_followup}
                            conversation.append(user_message)
                            conversation_tokens += ctrl._estimate_message_tokens(user_message)
                            tool_followup_user_prompts_sent += 1
                            continue
                        response_text = self._tool_only_response_fallback(executed_tool_calls)
                        assistant_message["content"] = response_text
                        new_tokens = ctrl._estimate_message_tokens(assistant_message)
                        conversation_tokens += new_tokens - assistant_tokens
                        assistant_tokens = new_tokens
                        fallback_applied = True
                        LOGGER.warning(
                            "No assistant response after %s follow-up prompt(s); returning fallback summary.",
                            tool_followup_prompts_sent,
                        )

                    if (
                        pending_patch_application
                        and not fallback_applied
                        and patch_reminders_sent < self._max_pending_patch_reminders
                    ):
                        reminder = self._pending_patch_prompt()
                        LOGGER.debug("Injected patch reminder (pending diff)")
                        reminder_message = {"role": "system", "content": reminder}
                        conversation.append(reminder_message)
                        conversation_tokens += ctrl._estimate_message_tokens(reminder_message)
                        patch_reminders_sent += 1
                        continue
                        
                    plot_tracker = ctrl._plot_loop_tracker
                    if (
                        plot_tracker is not None
                        and plot_tracker.needs_update_prompt()
                        and plot_loop_reminders_sent < self._max_pending_patch_reminders
                    ):
                        reminder_message = {
                            "role": "system",
                            "content": plot_tracker.update_prompt(),
                        }
                        LOGGER.debug("Injected plot-state update reminder")
                        conversation.append(reminder_message)
                        conversation_tokens += ctrl._estimate_message_tokens(reminder_message)
                        plot_loop_reminders_sent += 1
                        continue
                    break

                tool_followup_prompts_sent = 0
                tool_followup_user_prompts_sent = 0
                tool_iterations += 1
                if tool_iterations > max_iterations:
                    LOGGER.warning(
                        "Max tool iterations (%s) reached; returning partial response.",
                        max_iterations,
                    )
                    break

                tool_messages, tool_records, tool_token_cost = await ctrl._handle_tool_calls(
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
                active_log.log_tool_batch(
                    turn_index=turn_count,
                    records=tool_records,
                    messages=tool_messages,
                )
                turn_metrics["tool_tokens"] += tool_token_cost
                turn_metrics["conversation_length"] = len(conversation)
                ctrl._record_tool_names(turn_metrics, tool_records)
                ctrl._record_tool_metrics(turn_metrics, tool_records)
                guardrail_hints = self._guardrail_hints_from_records(tool_records)
                for hint in guardrail_hints:
                    hint_message = {"role": "system", "content": hint}
                    conversation.append(hint_message)
                    conversation_tokens += ctrl._estimate_message_tokens(hint_message)

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
                    elif name == "plot_state_update" and not self._tool_call_failed(record):
                        plot_loop_reminders_sent = 0

                if (
                    pending_patch_application
                    and diff_builders_since_edit >= self._diff_builder_reminder_threshold
                ):
                    reminder_text = self._diff_accumulation_prompt(diff_builders_since_edit)
                    LOGGER.debug(
                        "Injected diff_builder consolidation reminder (count=%s)", diff_builders_since_edit
                    )
                    reminder_message = {"role": "system", "content": reminder_text}
                    conversation.append(reminder_message)
                    conversation_tokens += ctrl._estimate_message_tokens(reminder_message)
                    diff_builders_since_edit = 0
                    continue

            turn_metrics["conversation_length"] = len(conversation)
            ctrl._log_response_text(response_text)
            ctrl._emit_context_usage(turn_metrics)
            LOGGER.debug(
                "Chat turn complete (chars=%s, tool calls=%s)",
                len(response_text),
                len(executed_tool_calls),
            )
            
            compaction_stats = None
            if ctrl._trace_compactor is not None:
                trace_stats = ctrl._trace_compactor.stats_snapshot().as_dict()
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
                    
            completion_warnings: list[str] = []
            if pending_patch_application:
                completion_warnings.append("pending_patch_pending")
            if chunk_tracker.warning_active:
                completion_warnings.append("chunk_flow_warning")
            if refresh_tracker is not None and refresh_tracker.warning_active:
                completion_warnings.append("snapshot_drift_warning")
            if plot_tracker is not None and plot_tracker.pending_update:
                completion_warnings.append("plot_state_pending_update")
                
            active_log.log_completion(
                response_text=response_text,
                tool_call_count=len(executed_tool_calls),
                warnings=completion_warnings,
                trace_compaction=compaction_stats or {},
            )
            log_context.__exit__(None, None, None)
            log_path = getattr(log_context, "path", None)
            event_log_path = str(log_path) if log_path else None
            
            return {
                "prompt": prompt,
                "response": response_text,
                "doc_snapshot": snapshot,
                "tool_calls": executed_tool_calls,
                "graph": ctrl.graph,
                "trace_compaction": compaction_stats,
                "subagent_jobs": [job.as_payload() for job in subagent_jobs],
                "event_log_path": event_log_path,
            }
        except Exception as exc:
            log_context.__exit__(exc.__class__, exc, exc.__traceback__)
            raise
        finally:
            # Clear turn context to prevent stale pinned tab_id
            if ctrl._tool_dispatcher is not None:
                ctrl._tool_dispatcher.clear_turn_context()
            if ctrl._chunk_flow_tracker is chunk_tracker:
                ctrl._chunk_flow_tracker = None
            if ctrl._snapshot_refresh_tracker is refresh_tracker:
                ctrl._snapshot_refresh_tracker = None
            if ctrl._plot_loop_tracker is plot_tracker:
                ctrl._plot_loop_tracker = None

    async def _invoke_model_turn(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        tool_specs: Sequence[ChatCompletionToolParam] | None,
        metadata: Mapping[str, str] | None,
        on_event: ToolCallback | None,
        max_completion_tokens: int | None = None,
    ) -> ModelTurnResult:
        """Invoke a single model turn and parse the response.
        
        Args:
            conversation: Current conversation messages.
            tool_specs: Available tool specifications.
            metadata: Request metadata.
            on_event: Event callback.
            max_completion_tokens: Token limit for completion.
            
        Returns:
            ModelTurnResult with assistant message and any tool calls.
        """
        deltas: list[str] = []
        final_chunk: str | None = None
        tool_calls: list[ToolCallRequest] = []
        stream_kwargs: Dict[str, Any] = {
            "messages": list(conversation),
            "tools": list(tool_specs) if tool_specs else None,
            "metadata": metadata,
        }
        if max_completion_tokens is not None:
            stream_kwargs["max_completion_tokens"] = max_completion_tokens
        stream_kwargs["temperature"] = self._temperature

        async for event in self._client.stream_chat(**stream_kwargs):
            await self._dispatch_event(event, on_event)
            if event.type == "content.delta" and event.content:
                deltas.append(event.content)
            elif event.type == "content.done" and event.content:
                final_chunk = str(event.content)
            elif event.type.endswith("tool_calls.function.arguments.done"):
                call_id = self._normalize_tool_call_id(event, len(tool_calls))
                tool_calls.append(
                    ToolCallRequest(
                        call_id=call_id,
                        name=event.tool_name or "",
                        index=event.tool_index if event.tool_index is not None else len(tool_calls),
                        arguments=event.tool_arguments,
                        parsed=event.parsed,
                    )
                )

        raw_response_text = "".join(deltas)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("Raw assistant text before parsing:\n%s", raw_response_text)
        if final_chunk:
            if not raw_response_text:
                raw_response_text = final_chunk
            elif not raw_response_text.endswith(final_chunk):
                raw_response_text += final_chunk
        response_text = raw_response_text.strip()
        if response_text:
            response_text, embedded_calls = self._parse_embedded_tool_calls(
                response_text,
                start_index=len(tool_calls),
            )
            if embedded_calls:
                tool_calls.extend(embedded_calls)
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
        return ModelTurnResult(assistant_message=assistant_message, response_text=response_text, tool_calls=tool_calls)

    async def _dispatch_event(self, event: AIStreamEvent, handler: ToolCallback | None) -> None:
        """Dispatch a stream event to the handler if provided."""
        if handler is None:
            return
        result = handler(event)
        if inspect.isawaitable(result):
            await result

    def _normalize_tool_call_id(self, event: AIStreamEvent, fallback_index: int) -> str:
        """Normalize tool call ID from event or generate fallback."""
        candidate = (event.tool_call_id or "").strip() if getattr(event, "tool_call_id", None) else ""
        if candidate:
            return candidate
        name = (event.tool_name or "tool").strip() or "tool"
        index = getattr(event, "tool_index", None)
        if index is None:
            index = fallback_index
        try:
            index_text = str(int(index))
        except (TypeError, ValueError):
            index_text = str(fallback_index)
        return f"{name}:{index_text}"

    def _parse_embedded_tool_calls(
        self, text: str, *, start_index: int = 0
    ) -> tuple[str, list[ToolCallRequest]]:
        """Parse embedded tool calls from response text.
        
        Args:
            text: Response text that may contain embedded tool markers.
            start_index: Starting index for tool call numbering.
            
        Returns:
            Tuple of (cleaned_text, parsed_tool_calls).
        """
        if not text or "<" not in text:
            return text, []
        normalized, index_map = self._normalize_tool_marker_text(text)
        matches = list(TOOL_CALLS_BLOCK_RE.finditer(normalized))
        if not matches:
            return text, []

        cleaned_parts: list[str] = []
        cursor = 0
        parsed_calls: list[ToolCallRequest] = []
        for block in matches:
            block_start, block_end = block.span()
            orig_block_start = index_map[block_start]
            orig_block_end = index_map[block_end]
            cleaned_parts.append(text[cursor:orig_block_start])
            try:
                body_start, body_end = block.span("body")
            except IndexError:  # pragma: no cover - defensive guard
                body_start = body_end = -1
            block_calls: list[ToolCallRequest] = []
            if body_start >= 0 and body_end >= body_start:
                start_offset = start_index + len(parsed_calls)
                block_calls = self._parse_tool_call_entries(
                    text,
                    normalized,
                    body_start=index_map[body_start],
                    body_end=index_map[body_end],
                    starting_index=start_offset,
                )
            if block_calls:
                parsed_calls.extend(block_calls)
            else:
                LOGGER.debug("Dropping embedded tool_calls block that could not be parsed")
            cursor = orig_block_end
        cleaned_parts.append(text[cursor:])
        cleaned_text = "".join(cleaned_parts)
        return cleaned_text, parsed_calls

    def _parse_tool_call_entries(
        self,
        original_text: str,
        normalized_text: str,
        *,
        body_start: int,
        body_end: int,
        starting_index: int,
    ) -> list[ToolCallRequest]:
        """Parse individual tool call entries from a tool_calls block."""
        body_original = original_text[body_start:body_end]
        body_normalized, body_index_map = self._normalize_tool_marker_text(body_original)
        parsed: list[ToolCallRequest] = []
        for match in TOOL_CALL_ENTRY_RE.finditer(body_normalized):
            try:
                name_start, name_end = match.span("name")
                args_start, args_end = match.span("args")
            except IndexError:  # pragma: no cover - defensive guard
                continue
            orig_name_start = body_index_map[name_start]
            orig_name_end = body_index_map[name_end]
            orig_args_start = body_index_map[args_start]
            orig_args_end = body_index_map[args_end]
            name_text = (
                body_original[orig_name_start:orig_name_end]
                .translate(TOOL_MARKER_TRANSLATION)
                .strip()
                .strip("<>|")
            )
            args_text = body_original[orig_args_start:orig_args_end].strip()
            if not name_text:
                continue
            parsed_args = try_parse_json_block(args_text)
            ordinal = starting_index + len(parsed)
            call_id = parsed_tool_call_id(name_text, ordinal)
            parsed.append(
                ToolCallRequest(
                    call_id=call_id,
                    name=name_text,
                    index=ordinal,
                    arguments=args_text,
                    parsed=parsed_args,
                )
            )
        return parsed

    def _build_metadata(
        self,
        snapshot: Mapping[str, Any],
        runtime_metadata: Mapping[str, str] | None,
    ) -> Optional[Dict[str, str]]:
        """Build request metadata from snapshot and runtime metadata."""
        metadata: Dict[str, str] = {}
        path = snapshot.get("path")
        if path:
            metadata["doc_path"] = str(path)
        text_range = snapshot.get("text_range")
        if isinstance(text_range, Mapping):
            start = text_range.get("start")
            end = text_range.get("end")
            metadata["window_range"] = f"{start}:{end}"
        if runtime_metadata:
            metadata.update(runtime_metadata)
        return metadata or None

    def _guardrail_hints_from_records(
        self, records: Sequence[Mapping[str, Any]]
    ) -> list[str]:
        """Generate guardrail hints from tool execution records.
        
        Args:
            records: Tool execution records.
            
        Returns:
            List of guardrail hint strings.
        """
        if not records:
            return []
        ctrl = self._controller
        hints: list[str] = []
        seen: set[str] = set()
        chunk_tracker = ctrl._chunk_flow_tracker
        plot_tracker = ctrl._plot_loop_tracker
        refresh_tracker = ctrl._snapshot_refresh_tracker
        
        for record in records:
            payload = ctrl._deserialize_tool_result(record)
            if chunk_tracker is not None:
                chunk_lines = chunk_tracker.observe_tool(
                    record, payload if isinstance(payload, Mapping) else None
                )
                if chunk_lines:
                    chunk_hint = format_guardrail_hint("Chunk Flow", chunk_lines)
                    if chunk_hint and chunk_hint not in seen:
                        hints.append(chunk_hint)
                        seen.add(chunk_hint)
            if refresh_tracker is not None:
                drift_lines = refresh_tracker.observe_tool(
                    record, payload if isinstance(payload, Mapping) else None
                )
                if drift_lines:
                    drift_hint = format_guardrail_hint("Snapshot Drift", drift_lines)
                    if drift_hint and drift_hint not in seen:
                        hints.append(drift_hint)
                        seen.add(drift_hint)
            if plot_tracker is not None:
                plot_lines = plot_tracker.observe_tool(
                    record, payload if isinstance(payload, Mapping) else None
                )
                if plot_lines:
                    plot_hint = format_guardrail_hint("Plot Loop", plot_lines)
                    if plot_hint and plot_hint not in seen:
                        hints.append(plot_hint)
                        seen.add(plot_hint)
            if not isinstance(payload, Mapping):
                continue
            name = str(record.get("name") or "").lower()
            candidate: Sequence[str] | None = None
            if name == "document_outline":
                candidate = outline_guardrail_hints(payload)
            elif name == "document_find_text":
                candidate = retrieval_guardrail_hints(payload)
            if not candidate:
                continue
            for entry in candidate:
                text = entry.strip()
                if not text or text in seen:
                    continue
                hints.append(text)
                seen.add(text)
        return hints

    def _compact_tool_messages(
        self,
        messages: list[dict[str, Any]],
        records: list[dict[str, Any]],
        *,
        conversation_tokens: int,
        response_reserve: int | None,
        snapshot: Mapping[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        """Compact tool messages to fit within token budget.
        
        Args:
            messages: Tool result messages.
            records: Tool execution records.
            conversation_tokens: Current conversation token count.
            response_reserve: Token reserve for response.
            snapshot: Document snapshot.
            
        Returns:
            Tuple of (compacted_messages, updated_token_count).
        """
        if not messages:
            return messages, conversation_tokens
        
        ctrl = self._controller
        updated_tokens = conversation_tokens
        compacted: list[dict[str, Any]] = []
        compactor = ctrl._trace_compactor

        def _evaluate(prompt_tokens: int, pending_tokens: int) -> BudgetDecision | None:
            return ctrl._evaluate_budget(
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
            pending_tokens = ctrl._estimate_text_tokens(content_text)
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
                pending_tokens = ctrl._estimate_text_tokens(message.get("content") or "")
                decision = _evaluate(updated_tokens, pending_tokens)
            if decision is not None and decision.verdict == "reject" and not decision.dry_run:
                raise ContextBudgetExceeded(decision)
            message_tokens = ctrl._estimate_message_tokens(message)
            updated_tokens += message_tokens
            if compactor is not None and entry is not None:
                compactor.commit_entry(entry, current_tokens=message_tokens)
            compacted.append(message)
        return compacted, updated_tokens

    def _build_tool_pointer(
        self, record: Mapping[str, Any], content_text: str
    ) -> ToolPointerMessage:
        """Build a tool pointer for summarized tool output.
        
        Args:
            record: Tool execution record.
            content_text: Original tool output content.
            
        Returns:
            ToolPointerMessage with summary and rehydration instructions.
        """
        pointer_kind = str(record.get("pointer_kind") or "")
        if pointer_kind == "subagent_summary":
            return self._build_subagent_pointer(record, content_text)
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

    def _build_subagent_pointer(
        self, record: Mapping[str, Any], content_text: str
    ) -> ToolPointerMessage:
        """Build a pointer for subagent summary content."""
        job_ids = tuple(str(job_id) for job_id in record.get("subagent_jobs", ()) if job_id)
        arguments: Mapping[str, Any] | None = {"job_ids": list(job_ids)} if job_ids else None
        payload = ToolPayload(
            name="SubagentScoutingReport",
            content=content_text,
            arguments=arguments,
            metadata={"source": "subagent_summary"},
        )
        summary = summarize_tool_content(payload, budget_tokens=_POINTER_SUMMARY_TOKENS)
        instructions = (
            "Re-run the AI helper with Subagents enabled on the same document selection or call "
            "PlotOutlineTool to rebuild the scouting report referenced by this pointer."
        )
        pointer = build_pointer(
            summary, tool_name="SubagentScoutingReport", rehydrate_instructions=instructions
        )
        pointer.metadata.setdefault("source", "context_budget")
        pointer.metadata["pointer_kind"] = "subagent_summary"
        if job_ids:
            pointer.metadata["job_ids"] = list(job_ids)
        document_id = record.get("document_id")
        if document_id:
            pointer.metadata["document_id"] = str(document_id)
        return pointer

    @staticmethod
    def _pointer_rehydrate_instructions(
        tool_name: str, arguments: Mapping[str, Any] | None
    ) -> str:
        """Generate rehydration instructions for a tool pointer."""
        if arguments:
            try:
                encoded = json.dumps(arguments, ensure_ascii=False)
            except (TypeError, ValueError):
                encoded = str(arguments)
            if len(encoded) > 180:
                encoded = f"{encoded[:177].rstrip()}…"
            return (
                f"Re-run {tool_name} with arguments similar to {encoded} "
                "to recover the full payload pointed to above."
            )
        return f"Re-run {tool_name} to recover the full payload referenced by this pointer."

    @staticmethod
    def _tool_call_failed(record: Mapping[str, Any]) -> bool:
        """Check if a tool call record indicates failure."""
        status = str(record.get("status") or "").lower()
        if status in ("error", "failed", "failure"):
            return True
        error = record.get("error")
        if error:
            return True
        return False

    @staticmethod
    def _normalize_tool_marker_text(text: str) -> tuple[str, list[int]]:
        """Normalize tool marker text and build an index map.
        
        Args:
            text: The text to normalize.
            
        Returns:
            Tuple of (normalized_text, index_map) where index_map maps
            positions in the normalized text back to the original text.
        """
        if not text:
            return "", [0]
        translation = TOOL_MARKER_TRANSLATION
        normalized_chars: list[str] = []
        index_map: list[int] = []
        idx = 0
        length = len(text)
        while idx < length:
            char = text[idx]
            if char == "\r":
                next_idx = idx + 1
                if next_idx < length and text[next_idx] == "\n":
                    normalized_chars.append("\n")
                    index_map.append(idx)
                    idx += 2
                    continue
                normalized_chars.append("\n")
                index_map.append(idx)
                idx += 1
                continue
            mapped = translation.get(ord(char))
            if mapped is None:
                normalized_chars.append(char)
            else:
                normalized_chars.append(mapped)
            index_map.append(idx)
            idx += 1
        normalized_text = "".join(normalized_chars)
        index_map.append(len(text))
        return normalized_text, index_map

    @staticmethod
    def _pending_patch_prompt() -> str:
        """Generate reminder prompt for pending diff application."""
        return (
            "You generated a diff via diff_builder but did not call document_edit to apply it yet. "
            "Use document_edit with action=\"patch\", include the diff text, and pass the latest "
            "document_version before responding."
        )

    @staticmethod
    def _tool_only_response_prompt() -> str:
        """Generate reminder prompt for missing assistant response."""
        return (
            "You executed tools but did not provide any assistant response. "
            "Summarize the tool results for the user and describe next actions before ending the turn."
        )

    @staticmethod
    def _tool_only_response_user_prompt(records: Sequence[Mapping[str, Any]]) -> str:
        """Generate user-style prompt for missing assistant response."""
        latest = records[-1] if records else {}
        tool_name = str(latest.get("name") or "document tool")
        status = str(latest.get("status") or "ok")
        diff_summary = str(latest.get("diff_summary") or "").strip()
        result_text = str(latest.get("result") or "").strip()
        summary = diff_summary if diff_summary else result_text
        if summary and len(summary) > 400:
            summary = f"{summary[:397].rstrip()}…"
        if not summary:
            summary = "(Tool result contained structured data)"
        return (
            "You already executed {tool} (status={status}) but did not send a reply. "
            "Draft the assistant response now summarizing the tool output before taking any new tools.\n"
            "Summary:\n{summary}"
        ).format(tool=tool_name, status=status, summary=summary)

    @staticmethod
    def _tool_only_response_fallback(records: Sequence[Mapping[str, Any]]) -> str:
        """Generate fallback response when assistant provides none."""
        if not records:
            return (
                "Tool execution completed, but the assistant returned an empty response. "
                "Describe the latest tool results manually before continuing."
            )
        latest = records[-1]
        name = str(latest.get("name") or "tool")
        status = str(latest.get("status") or "ok")
        diff_summary = str(latest.get("diff_summary") or "").strip()
        result_text = str(latest.get("result") or "").strip()
        summary = diff_summary or result_text
        if summary and len(summary) > 400:
            summary = f"{summary[:397].rstrip()}…"
        if not summary:
            summary = "Tool completed without returning additional details."
        return (
            "Tool execution completed, but the assistant did not send a response. "
            f"Latest tool: {name} (status={status}).\n{summary}"
        )

    @staticmethod
    def _diff_accumulation_prompt(diff_count: int) -> str:
        """Generate reminder prompt for accumulated diffs."""
        return (
            f"You've produced {diff_count} diff_builder results without applying them. "
            "Consolidate the change into a single diff and immediately call document_edit (action=\"patch\")."
        )
