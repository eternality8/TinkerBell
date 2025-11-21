"""Agent executor façade wrapping LangChain/LangGraph interactions."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import json
import logging
import math
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, ClassVar, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, cast

from openai.types.chat import ChatCompletionToolParam

from .. import prompts
from ..analysis import AnalysisAgent, AnalysisAdvice, AnalysisInput
from ..ai_types import (
    AgentConfig,
    ChunkReference,
    SubagentBudget,
    SubagentJob,
    SubagentJobState,
    SubagentRuntimeConfig,
)
from ..client import AIClient, AIStreamEvent
from ..memory.cache_bus import (
    DocumentCacheBus,
    DocumentCacheEvent,
    DocumentChangedEvent,
    DocumentClosedEvent,
    get_document_cache_bus,
)
from ..memory.chunk_index import ChunkIndex
from ..memory.plot_state import DocumentPlotStateStore
from ..memory.character_map import CharacterMapStore
from ..services import ToolPayload, build_pointer, summarize_tool_content, telemetry as telemetry_service
from ...services.bridge import DocumentVersionMismatchError
from ..services.trace_compactor import TraceCompactor
from ...chat.message_model import ToolPointerMessage
from ..services.context_policy import BudgetDecision, ContextBudgetPolicy
from ..services.telemetry import ContextUsageEvent, TelemetrySink
from ..agents.graph import build_agent_graph
from .budget_manager import BudgetManager, ContextBudgetExceeded
from .event_log import ChatEventLogger
from .subagent_runtime import SubagentRuntimeManager
from .telemetry_manager import TelemetryManager

LOGGER = logging.getLogger(__name__)
_PROMPT_HEADROOM = 4_096
_POINTER_SUMMARY_TOKENS = 512
_SUGGESTION_SYSTEM_PROMPT = (
    "You are a proactive writing assistant that proposes the next helpful user prompts after reviewing the prior "
    "conversation. Respond ONLY with a JSON array of concise suggestion strings (each under 120 characters). "
    "Return at most {max_suggestions} unique ideas tailored to the conversation."
)

_TOOL_MARKER_TRANSLATION = str.maketrans(
    {
        ord("｜"): "|",
        ord("￨"): "|",
        ord("∣"): "|",
        ord("▕"): "|",
        ord("＜"): "<",
        ord("﹤"): "<",
        ord("〈"): "<",
        ord("⟨"): "<",
        ord("《"): "<",
        ord("＞"): ">",
        ord("﹥"): ">",
        ord("〉"): ">",
        ord("⟩"): ">",
        ord("》"): ">",
        ord("▁"): "_",
        ord("＿"): "_",
        ord("﹍"): "_",
        ord("﹎"): "_",
        ord("﹏"): "_",
        ord("　"): " ",
        ord("\u200b"): " ",
        ord("\u200c"): " ",
        ord("\u200d"): " ",
        ord("\ufeff"): " ",
    }
)
_TOOL_CALLS_BLOCK_RE = re.compile(
    r"<\s*\|?\s*tool[\s_]*calls[\s_]*begin\s*\|?\s*>(?P<body>.*?)<\s*\|?\s*tool[\s_]*calls[\s_]*end\s*\|?\s*>",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_CALL_ENTRY_RE = re.compile(
    r"<\s*\|?\s*tool[\s_]*call[\s_]*begin\s*\|?\s*>(?P<name>.*?)<\s*\|?\s*tool[\s_]*sep\s*\|?\s*>(?P<args>.*?)<\s*\|?\s*tool[\s_]*call[\s_]*end\s*\|?\s*>",
    re.IGNORECASE | re.DOTALL,
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


@dataclass(slots=True)
class _ChunkContext:
    """Resolved chunk metadata pulled from a manifest for subagent planning."""

    chunk_id: str
    document_id: str
    char_range: tuple[int, int]
    chunk_hash: str | None = None
    pointer_id: str | None = None
    text: str | None = None


@dataclass(slots=True)
class _ChunkFlowTracker:
    """Tracks whether the agent stays on the chunk-first path during a run."""

    document_id: str | None
    warning_active: bool = False
    last_reason: str | None = None

    def observe_tool(self, record: Mapping[str, Any], payload: Mapping[str, Any] | None) -> list[str] | None:
        name = str(record.get("name") or "").lower()
        if name == "document_snapshot":
            return self._handle_snapshot(record, payload)
        if name == "document_chunk":
            self._handle_chunk_tool(payload)
        return None

    # ------------------------------------------------------------------
    # Snapshot handling
    # ------------------------------------------------------------------
    def _handle_snapshot(self, record: Mapping[str, Any], payload: Mapping[str, Any] | None) -> list[str] | None:
        if not isinstance(payload, Mapping):
            return None
        window = payload.get("window") if isinstance(payload.get("window"), Mapping) else None
        if window is None:
            window = {}
        includes_full = bool(window.get("includes_full_document"))
        doc_length = self._coerce_int(payload.get("length"))
        window_span = self._window_span(window, payload)
        selection_span = self._selection_span(window, payload)
        manifest = payload.get("chunk_manifest") if isinstance(payload.get("chunk_manifest"), Mapping) else None
        metadata: dict[str, Any] = {
            "document_id": payload.get("document_id") or self.document_id,
            "document_length": doc_length,
            "window_span": window_span,
            "selection_span": selection_span,
            "window_kind": str(window.get("kind") or ""),
            "requested_window": str(window.get("requested_kind") or ""),
            "defaulted": bool(window.get("defaulted")),
            "source": "document_snapshot",
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
            recovery["recovered_via"] = "document_snapshot"
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
    def _selection_span(window: Mapping[str, Any], payload: Mapping[str, Any]) -> int | None:
        selection = window.get("selection")
        if not isinstance(selection, Mapping):
            selection = payload.get("selection") if isinstance(payload.get("selection"), Mapping) else None
        if not isinstance(selection, Mapping):
            return None
        start = _ChunkFlowTracker._coerce_int(selection.get("start"))
        end = _ChunkFlowTracker._coerce_int(selection.get("end"))
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
    def _handle_chunk_tool(self, payload: Mapping[str, Any] | None) -> None:
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
            "source": "document_chunk",
        }
        self._emit_request(metadata)
        if self.warning_active:
            recovery = dict(metadata)
            recovery["recovered_via"] = "document_chunk"
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
            f"DocumentSnapshot fetched the entire document{approx}.",
            "Request a selection-scoped snapshot or hydrate a chunk via DocumentChunkTool before editing.",
            "If a full snapshot is unavoidable, explain the fallback to the user and immediately return to chunked context.",
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


@dataclass(slots=True)
class _PlotLoopTracker:
    """Ensures the agent follows the plot-outline → edit → update contract."""

    document_id: str | None
    outline_called: bool = False
    pending_update: bool = False
    blocked_edits: int = 0

    def before_tool(self, tool_name: str | None) -> str | None:
        name = (tool_name or "").strip().lower()
        if name in {"document_apply_patch", "document_edit"} and not self.outline_called:
            self.blocked_edits += 1
            return (
                "Plot loop guardrail: call PlotOutlineTool for continuity context before applying edits."
            )
        return None

    def observe_tool(self, record: Mapping[str, Any], payload: Mapping[str, Any] | None) -> list[str] | None:
        name = str(record.get("name") or "").lower()
        status = str(record.get("status") or "ok").lower()
        succeeded = status == "ok"
        if name in {"plot_outline", "document_plot_state"} and succeeded:
            self.outline_called = True
            return None
        if name == "plot_state_update":
            if succeeded and self.pending_update:
                self.pending_update = False
                return [
                    "PlotStateUpdateTool received your changes. You may proceed to the next edit after reading the outline if needed.",
                ]
            if succeeded:
                self.pending_update = False
            return None
        if name in {"document_apply_patch", "document_edit"} and succeeded:
            self.pending_update = True
            return [
                "Plot loop reminder: call PlotStateUpdateTool to log the changes you just applied so downstream agents stay in sync.",
            ]
        return None

    def needs_update_prompt(self) -> bool:
        return self.pending_update

    def update_prompt(self) -> str:
        target = self.document_id or "this document"
        return (
            f"Plot loop requirement: run PlotStateUpdateTool for {target} before finishing this turn so plot scaffolding reflects your edits."
        )


@dataclass(slots=True)
class ChunkingRuntimeConfig:
    """Settings governing chunk manifest preferences and tool caps."""

    default_profile: str = "auto"
    overlap_chars: int = 256
    max_inline_tokens: int = 1_800
    iterator_limit: int = 4


@dataclass(slots=True)
class AnalysisRuntimeConfig:
    """Toggles for the preflight analysis agent."""

    enabled: bool = True
    ttl_seconds: float = 120.0

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
    max_tool_followup_prompts: int = 2
    max_tool_followup_user_prompts: int = 1
    max_context_tokens: int = 128_000
    response_token_reserve: int = 16_000
    temperature: float = 0.2
    telemetry_enabled: bool = False
    telemetry_limit: int = 200
    debug_event_logging: bool = False
    event_log_dir: Path | str | None = None
    telemetry_sink: TelemetrySink | None = None
    agent_config: AgentConfig | None = None
    budget_policy: ContextBudgetPolicy | None = None
    subagent_config: SubagentRuntimeConfig = field(default_factory=SubagentRuntimeConfig)
    analysis_config: AnalysisRuntimeConfig = field(default_factory=AnalysisRuntimeConfig)
    _graph: Dict[str, Any] = field(init=False, repr=False)
    _active_task: asyncio.Task[dict] | None = field(default=None, init=False, repr=False)
    _task_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _budget_manager: BudgetManager = field(default_factory=BudgetManager, init=False, repr=False)
    _telemetry_manager: TelemetryManager = field(default_factory=TelemetryManager, init=False, repr=False)
    _trace_compactor: TraceCompactor | None = field(default=None, init=False, repr=False)
    _outline_digest_cache: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _subagent_runtime: SubagentRuntimeManager | None = field(default=None, init=False, repr=False)
    _graph_rebuild_suspensions: int = field(default=0, init=False, repr=False)
    _graph_dirty: bool = field(default=False, init=False, repr=False)
    _chunk_config: ChunkingRuntimeConfig = field(default_factory=ChunkingRuntimeConfig, init=False, repr=False)
    _chunk_index: ChunkIndex | None = field(default=None, init=False, repr=False)
    _chunk_flow_tracker: _ChunkFlowTracker | None = field(default=None, init=False, repr=False)
    _plot_loop_tracker: _PlotLoopTracker | None = field(default=None, init=False, repr=False)
    _analysis_agent: AnalysisAgent | None = field(default=None, init=False, repr=False)
    _analysis_advice_cache: dict[str, AnalysisAdvice] = field(default_factory=dict, init=False, repr=False)
    _latest_snapshot_cache: dict[str, Mapping[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _cache_bus: DocumentCacheBus | None = field(default=None, init=False, repr=False)
    _event_logger: ChatEventLogger | None = field(default=None, init=False, repr=False)
    _RETRYABLE_VERSION_TOOLS: ClassVar[frozenset[str]] = frozenset({"document_apply_patch", "search_replace"})

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
        self.temperature = self._normalize_temperature(self.temperature)
        self._rebuild_graph()
        self._subagent_runtime = SubagentRuntimeManager(tool_resolver=self._tool_registry_snapshot)
        self.configure_context_window(
            max_context_tokens=self.max_context_tokens,
            response_token_reserve=self.response_token_reserve,
        )
        self._configure_telemetry_sink()
        self.configure_budget_policy(self.budget_policy)
        self.configure_subagents(self.subagent_config)
        self.configure_analysis(self.analysis_config)
        self._subscribe_document_cache_bus()
        self._trace_compactor = TraceCompactor(
            pointer_builder=self._build_tool_pointer,
            estimate_message_tokens=self._estimate_message_tokens,
        )
        self.configure_debug_event_logging(enabled=self.debug_event_logging, event_log_dir=self.event_log_dir)

    @property
    def graph(self) -> Dict[str, Any]:
        """Return the current compiled graph representation."""

        return dict(self._graph)

    @property
    def plot_state_store(self) -> DocumentPlotStateStore | None:
        """Return the active plot/entity memory store (when enabled)."""

        runtime = self._subagent_runtime
        return runtime.plot_state_store if runtime else None

    @property
    def character_map_store(self) -> CharacterMapStore | None:
        """Return the active character/entity concordance store (when enabled)."""

        runtime = self._subagent_runtime
        return runtime.character_map_store if runtime else None

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

        registration = self._build_tool_registration(
            name,
            tool,
            description=description,
            parameters=parameters,
            strict=strict,
            summarizable=summarizable,
        )
        self._store_tool_registration(registration)

    def register_tools(self, tools: Mapping[str, Any]) -> None:
        """Register many tools while deferring graph rebuilds."""

        if not tools:
            return
        with self.suspend_graph_rebuilds():
            for name, tool in tools.items():
                registration = self._coerce_registration(name, tool)
                self._store_tool_registration(registration)

    @contextlib.contextmanager
    def suspend_graph_rebuilds(self) -> Iterable[None]:
        """Temporarily suppress graph rebuilds until the context exits."""

        self._graph_rebuild_suspensions += 1
        try:
            yield
        finally:
            self._graph_rebuild_suspensions = max(0, self._graph_rebuild_suspensions - 1)
            if self._graph_rebuild_suspensions == 0:
                self._flush_graph_rebuilds()

    def _build_tool_registration(
        self,
        name: str,
        tool: Any,
        *,
        description: str | None = None,
        parameters: Mapping[str, Any] | None = None,
        strict: bool | None = None,
        summarizable: bool | None = None,
    ) -> ToolRegistration:
        if summarizable is None:
            attr_flag = getattr(tool, "summarizable", None)
            summarizable_flag = True if attr_flag is None else bool(attr_flag)
        else:
            summarizable_flag = bool(summarizable)
        return ToolRegistration(
            name=name,
            impl=tool,
            description=description,
            parameters=parameters,
            strict=True if strict is None else bool(strict),
            summarizable=summarizable_flag,
        )

    def _coerce_registration(self, name: str, spec: Any) -> ToolRegistration:
        if isinstance(spec, ToolRegistration):
            return spec
        if isinstance(spec, Mapping):
            candidate = dict(spec)
            impl = candidate.get("impl")
            if impl is None:
                impl = candidate.get("tool")
            if impl is None:
                raise ValueError(f"Tool '{name}' mapping must include an 'impl' entry")
            return self._build_tool_registration(
                name,
                impl,
                description=candidate.get("description"),
                parameters=candidate.get("parameters"),
                strict=candidate.get("strict"),
                summarizable=candidate.get("summarizable"),
            )
        return self._build_tool_registration(name, spec)

    def _store_tool_registration(self, registration: ToolRegistration) -> None:
        self.tools[registration.name] = registration
        LOGGER.debug("Registered tool: %s", registration.name)
        self._schedule_graph_rebuild()

    def set_max_tool_iterations(self, iterations: int) -> None:
        """Update the maximum allowed tool iterations and rebuild the graph if needed."""

        normalized = self._normalize_iterations(iterations)
        config = self.agent_config or AgentConfig(max_iterations=self.max_tool_iterations)
        if normalized == config.max_iterations:
            return
        config.max_iterations = normalized
        self.agent_config = config
        self.max_tool_iterations = normalized
        self._schedule_graph_rebuild()

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

    def set_temperature(self, value: float | None) -> None:
        """Update the default sampling temperature used for chat turns."""

        self.temperature = self._normalize_temperature(value)

    def configure_chunking(
        self,
        *,
        default_profile: str | None = None,
        overlap_chars: int | None = None,
        max_inline_tokens: int | None = None,
        iterator_limit: int | None = None,
    ) -> None:
        """Update runtime chunking preferences for manifest + chunk tools."""

        config = self._chunk_config
        if default_profile is not None:
            normalized = str(default_profile).strip().lower()
            if normalized not in {"auto", "prose", "code", "notes"}:
                normalized = "auto"
            config.default_profile = normalized
        if overlap_chars is not None:
            try:
                config.overlap_chars = max(0, int(overlap_chars))
            except (TypeError, ValueError):
                pass
        if max_inline_tokens is not None:
            try:
                config.max_inline_tokens = max(256, int(max_inline_tokens))
            except (TypeError, ValueError):
                pass
        if iterator_limit is not None:
            try:
                config.iterator_limit = max(1, int(iterator_limit))
            except (TypeError, ValueError):
                pass

    def configure_analysis(
        self,
        config: AnalysisRuntimeConfig | None = None,
        *,
        enabled: bool | None = None,
        ttl_seconds: float | None = None,
    ) -> None:
        """Update analysis runtime preferences and rebuild the agent cache."""

        base = config or self.analysis_config
        normalized = AnalysisRuntimeConfig(
            enabled=bool(getattr(base, "enabled", True)),
            ttl_seconds=max(30.0, float(getattr(base, "ttl_seconds", 120.0) or 120.0)),
        )
        if enabled is not None:
            normalized.enabled = bool(enabled)
        if ttl_seconds is not None:
            try:
                normalized.ttl_seconds = max(30.0, float(ttl_seconds))
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass
        self.analysis_config = normalized
        self._analysis_agent = None

    def get_chunking_config(self) -> ChunkingRuntimeConfig:
        """Return a copy of the current chunking runtime config."""

        config = self._chunk_config
        return ChunkingRuntimeConfig(
            default_profile=config.default_profile,
            overlap_chars=config.overlap_chars,
            max_inline_tokens=config.max_inline_tokens,
            iterator_limit=config.iterator_limit,
        )

    def analysis_enabled(self) -> bool:
        """Return whether the preflight analysis pipeline is active."""

        return bool(getattr(self.analysis_config, "enabled", False))

    def ensure_chunk_index(self) -> ChunkIndex:
        """Return the lazily constructed chunk index shared across tools."""

        if self._chunk_index is None:
            self._chunk_index = ChunkIndex()
        return self._chunk_index

    def _ingest_snapshot_manifest(self, snapshot: Mapping[str, Any]) -> None:
        manifest = snapshot.get("chunk_manifest")
        if isinstance(manifest, Mapping):
            self._ingest_chunk_manifest(manifest)
        extras = snapshot.get("source_tab_snapshots")
        if isinstance(extras, Sequence):
            for candidate in extras:
                if isinstance(candidate, Mapping):
                    self._ingest_snapshot_manifest(candidate)

    def _ingest_chunk_manifest(self, manifest: Mapping[str, Any]) -> None:
        try:
            self.ensure_chunk_index().ingest_manifest(manifest)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Chunk manifest ingestion failed", exc_info=True)

    def _resolve_chunk_tool(self) -> Any | None:
        registration = self.tools.get("document_chunk")
        if registration is None:
            return None
        impl = getattr(registration, "impl", registration)
        runner = getattr(impl, "run", None)
        if callable(runner):
            return impl
        if callable(impl):
            return impl
        return None

    def _hydrate_chunk_text(
        self,
        *,
        chunk_id: str,
        document_id: str | None,
        cache_key: str | None,
        version: str | None,
    ) -> str:
        tool = self._resolve_chunk_tool()
        if tool is None:
            return ""
        kwargs: dict[str, Any] = {"chunk_id": chunk_id, "include_text": True}
        if document_id:
            kwargs["document_id"] = document_id
        if cache_key:
            kwargs["cache_key"] = cache_key
        if version:
            kwargs["version"] = version
        try:
            runner = getattr(tool, "run", None)
            if callable(runner):
                result = runner(**kwargs)
            else:
                result = tool(**kwargs)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Chunk hydration failed for %s", chunk_id, exc_info=True)
            return ""
        if not isinstance(result, Mapping):
            return ""
        chunk_payload = result.get("chunk")
        if isinstance(chunk_payload, Mapping):
            text = chunk_payload.get("text")
            if isinstance(text, str):
                return text
        return ""

    def _build_manifest_chunk_context(
        self,
        snapshot: Mapping[str, Any],
        selection_span: tuple[int, int],
        *,
        hydrate_text: bool,
    ) -> _ChunkContext | None:
        manifest = snapshot.get("chunk_manifest")
        if not isinstance(manifest, Mapping):
            return None
        chunks = manifest.get("chunks")
        if not isinstance(chunks, Sequence) or not chunks:
            return None
        self._ingest_chunk_manifest(manifest)
        chosen = self._select_manifest_chunk(chunks, selection_span)
        if chosen is None:
            return None
        chunk_id = str(chosen.get("id") or "").strip()
        if not chunk_id:
            return None
        document_id = str(manifest.get("document_id") or self._resolve_document_id(snapshot) or "document")
        start = self._coerce_index(chosen.get("start"), selection_span[0])
        end = self._coerce_index(chosen.get("end"), max(selection_span[1], start + 1))
        if end <= start:
            end = start + max(1, selection_span[1] - selection_span[0])
        chunk_hash = chosen.get("hash")
        pointer_id = chosen.get("outline_pointer_id")
        context = _ChunkContext(
            chunk_id=chunk_id,
            document_id=document_id,
            char_range=(start, end),
            chunk_hash=str(chunk_hash).strip() if isinstance(chunk_hash, str) and chunk_hash.strip() else None,
            pointer_id=str(pointer_id).strip() if isinstance(pointer_id, str) and pointer_id.strip() else None,
        )
        if hydrate_text:
            cache_key = manifest.get("cache_key")
            version = manifest.get("version")
            text = self._hydrate_chunk_text(
                chunk_id=chunk_id,
                document_id=document_id,
                cache_key=str(cache_key).strip() if isinstance(cache_key, str) and cache_key.strip() else None,
                version=str(version).strip() if isinstance(version, str) and version.strip() else None,
            )
            context.text = text or None
        return context

    def _analysis_hint_message(self, snapshot: Mapping[str, Any]) -> dict[str, str] | None:
        advice = self._run_preflight_analysis(snapshot, source="controller")
        if advice is None:
            return None
        hint_text = self._format_analysis_hint(advice)
        if not hint_text:
            return None
        return {"role": "system", "content": hint_text}

    def _run_preflight_analysis(
        self,
        snapshot: Mapping[str, Any],
        *,
        source: str = "controller",
        force_refresh: bool = False,
    ) -> AnalysisAdvice | None:
        if not self.analysis_enabled():
            return None
        document_id = self._resolve_document_id(snapshot)
        if not document_id:
            return None
        agent = self._ensure_analysis_agent()
        analysis_input = self._build_analysis_input(snapshot)
        advice = agent.analyze(analysis_input, force_refresh=force_refresh, source=source)
        self._analysis_advice_cache[document_id] = advice
        return advice

    def _ensure_analysis_agent(self) -> AnalysisAgent:
        agent = self._analysis_agent
        if agent is not None:
            return agent
        telemetry_emitter = getattr(telemetry_service, "emit", None)
        if not callable(telemetry_emitter):
            telemetry_emitter = None
        agent = AnalysisAgent(
            ttl_seconds=self.analysis_config.ttl_seconds,
            telemetry_emitter=telemetry_emitter,
        )
        self._analysis_agent = agent
        return agent

    def _build_analysis_input(self, snapshot: Mapping[str, Any]) -> AnalysisInput:
        selection_start, selection_end = self._analysis_selection_bounds(snapshot)
        manifest = snapshot.get("chunk_manifest") if isinstance(snapshot.get("chunk_manifest"), Mapping) else None
        chunk_profile = manifest.get("chunk_profile") if isinstance(manifest, Mapping) else None
        chunk_cache_key = manifest.get("cache_key") if isinstance(manifest, Mapping) else None
        outline_age = self._coerce_optional_float(snapshot.get("outline_age_seconds"))
        if outline_age is None:
            completed = snapshot.get("outline_completed_at")
            outline_age = self._outline_age_from_timestamp(completed)
        selection_hash = self._coerce_optional_str(snapshot.get("selection_hash"))
        document_chars = self._coerce_optional_int(snapshot.get("length"))
        if document_chars is None:
            text = snapshot.get("text")
            if isinstance(text, str):
                document_chars = len(text)
        chunk_flow_flags: tuple[str, ...] = ()
        tracker = self._chunk_flow_tracker
        if tracker and tracker.warning_active:
            chunk_flow_flags = (tracker.last_reason or "chunk_flow_warning",)
        plot_loop_flags: tuple[str, ...] = ()
        plot_state_status = self._coerce_optional_str(snapshot.get("plot_state_status"))
        loop_tracker = self._plot_loop_tracker
        if loop_tracker is not None:
            if loop_tracker.pending_update:
                plot_state_status = "pending_update"
                plot_loop_flags = ("pending_update",)
            elif loop_tracker.outline_called and not plot_state_status:
                plot_state_status = "ok"
        concordance_status = self._coerce_optional_str(snapshot.get("concordance_status"))
        concordance_age = self._coerce_optional_float(snapshot.get("concordance_age_seconds"))
        extras: dict[str, object] = {}
        if chunk_flow_flags:
            extras["chunk_flow"] = chunk_flow_flags
        if plot_loop_flags:
            extras["plot_loop"] = plot_loop_flags
        if manifest and manifest.get("generated_at") is not None:
            extras["chunk_manifest_generated_at"] = manifest.get("generated_at")
        document_id = self._resolve_document_id(snapshot) or "document"
        version = snapshot.get("version") or snapshot.get("version_id") or snapshot.get("document_version")
        return AnalysisInput(
            document_id=document_id,
            document_version=str(version) if version else None,
            document_path=self._coerce_optional_str(snapshot.get("path")),
            selection_start=selection_start,
            selection_end=selection_end,
            document_chars=document_chars,
            chunk_profile_hint=self._chunk_config.default_profile,
            chunk_index_ready=self._chunk_index is not None,
            chunk_manifest_profile=str(chunk_profile) if chunk_profile else None,
            chunk_manifest_cache_key=str(chunk_cache_key) if chunk_cache_key else None,
            outline_digest=self._coerce_optional_str(snapshot.get("outline_digest")),
            outline_age_seconds=outline_age,
            outline_version_id=self._coerce_optional_int(snapshot.get("outline_version_id")),
            plot_state_status=plot_state_status,
            plot_override_version=self._coerce_optional_int(snapshot.get("plot_override_version")),
            concordance_status=concordance_status,
            concordance_age_seconds=concordance_age,
            retrieval_enabled=True,
            selection_fingerprint=selection_hash,
            extra_metadata=extras or None,
            chunk_flow_warnings=chunk_flow_flags or None,
            plot_loop_flags=plot_loop_flags or None,
        )

    def _analysis_selection_bounds(self, snapshot: Mapping[str, Any]) -> tuple[int, int]:
        span = self._selection_span(snapshot)
        if span is not None:
            return span
        selection = snapshot.get("selection")
        start = 0
        end = 0
        if isinstance(selection, Mapping):
            start = self._coerce_index(selection.get("start"), 0)
            end = self._coerce_index(selection.get("end"), start)
        elif isinstance(selection, Sequence) and len(selection) >= 2:
            try:
                start = int(selection[0])
                end = int(selection[1])
            except (TypeError, ValueError):  # pragma: no cover - defensive
                start = 0
                end = 0
        document_length = self._coerce_optional_int(snapshot.get("length"))
        if document_length is not None:
            start = max(0, min(start, document_length))
            end = max(start, min(end, document_length))
        return (start, end)

    def _format_analysis_hint(self, advice: AnalysisAdvice) -> str:
        lines = [
            f"- Chunk profile: {advice.chunk_profile}",
            f"- Required tools: {', '.join(advice.required_tools) if advice.required_tools else 'none'}",
        ]
        if advice.optional_tools:
            lines.append(f"- Optional tools: {', '.join(advice.optional_tools)}")
        lines.append(f"- Outline refresh required: {'yes' if advice.must_refresh_outline else 'no'}")
        if advice.plot_state_status:
            lines.append(f"- Plot state status: {advice.plot_state_status}")
        if advice.concordance_status:
            lines.append(f"- Concordance status: {advice.concordance_status}")
        if advice.warnings:
            warning_lines = "\n".join(f"  - {warning.message}" for warning in advice.warnings)
            lines.append(f"- Warnings:\n{warning_lines}")
        return "Preflight analysis summary:\n" + "\n".join(lines)

    def _remember_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        document_id = self._resolve_document_id(snapshot)
        if not document_id:
            return
        self._latest_snapshot_cache[document_id] = dict(snapshot)

    def get_latest_analysis_advice(self, document_id: str | None) -> AnalysisAdvice | None:
        if not document_id:
            return None
        return self._analysis_advice_cache.get(document_id)

    def get_latest_snapshot(self, document_id: str | None) -> Mapping[str, Any] | None:
        if not document_id:
            return None
        cached = self._latest_snapshot_cache.get(document_id)
        if cached is None:
            return None
        return dict(cached)

    def _subscribe_document_cache_bus(self) -> None:
        try:
            bus = get_document_cache_bus()
        except Exception:  # pragma: no cover - defensive bus acquisition
            LOGGER.debug("Unable to acquire document cache bus", exc_info=True)
            return
        self._cache_bus = bus
        try:
            bus.subscribe(DocumentChangedEvent, self._handle_document_cache_event, weak=True)
            bus.subscribe(DocumentClosedEvent, self._handle_document_cache_event, weak=True)
        except Exception:  # pragma: no cover - cache bus subscription failures should not break controller
            LOGGER.debug("Failed to subscribe to document cache bus", exc_info=True)

    def _handle_document_cache_event(self, event: DocumentCacheEvent) -> None:
        document_id = getattr(event, "document_id", None)
        if not document_id:
            return
        self._analysis_advice_cache.pop(document_id, None)
        self._latest_snapshot_cache.pop(document_id, None)
        agent = self._analysis_agent
        if agent is None:
            return
        try:
            agent.invalidate_document(document_id)
        except Exception:  # pragma: no cover - cache invalidation best-effort
            LOGGER.debug("Failed to invalidate analysis cache for %s", document_id, exc_info=True)

    def _emit_analysis_invocation_event(
        self,
        *,
        event_name: str,
        document_id: str | None,
        selection_start: int | None,
        selection_end: int | None,
        force_refresh: bool,
        reason: str | None,
        snapshot_origin: str,
    ) -> None:
        emitter = getattr(telemetry_service, "emit", None)
        if not callable(emitter):
            return
        payload: dict[str, object] = {
            "document_id": document_id,
            "selection_start": selection_start if selection_start is not None else None,
            "selection_end": selection_end if selection_end is not None else None,
            "force_refresh": bool(force_refresh),
            "snapshot_origin": snapshot_origin,
            "source": "tool",
        }
        if reason:
            payload["reason"] = reason
        payload["has_selection_override"] = selection_start is not None and selection_end is not None
        emitter(event_name, payload)

    def request_analysis_advice(
        self,
        *,
        document_id: str | None = None,
        snapshot: Mapping[str, Any] | None = None,
        selection_start: int | None = None,
        selection_end: int | None = None,
        force_refresh: bool = False,
        reason: str | None = None,
    ) -> AnalysisAdvice | None:
        """Public entry point for UI code to rerun the analysis agent."""

        return self._advisor_tool_entrypoint(
            document_id=document_id,
            snapshot=snapshot,
            selection_start=selection_start,
            selection_end=selection_end,
            force_refresh=force_refresh,
            reason=reason,
            invocation_source="ui",
        )

    def _advisor_tool_entrypoint(
        self,
        *,
        document_id: str | None = None,
        snapshot: Mapping[str, Any] | None = None,
        selection_start: int | None = None,
        selection_end: int | None = None,
        force_refresh: bool = False,
        reason: str | None = None,
        invocation_source: str | None = None,
    ) -> AnalysisAdvice | None:
        source = (invocation_source or "tool").strip() or "tool"
        if not self.analysis_enabled():
            return None
        candidate_snapshot: Mapping[str, Any] | None = snapshot
        target_id = document_id
        if candidate_snapshot is None and target_id:
            cached = self._latest_snapshot_cache.get(target_id)
            if cached is not None:
                candidate_snapshot = dict(cached)
                snapshot_origin = "cache"
            else:
                snapshot_origin = "missing"
        else:
            snapshot_origin = "provided"
        if candidate_snapshot is None:
            raise ValueError("snapshot or document_id is required for analysis")
        snapshot_payload = dict(candidate_snapshot)
        resolved_id = target_id or self._resolve_document_id(snapshot_payload)
        if resolved_id:
            snapshot_payload.setdefault("document_id", resolved_id)
        if selection_start is not None and selection_end is not None:
            snapshot_payload["selection"] = {"start": selection_start, "end": selection_end}
        if source == "tool":
            self._emit_analysis_invocation_event(
                event_name="analysis.advisor_tool.invoked",
                document_id=resolved_id,
                selection_start=selection_start,
                selection_end=selection_end,
                force_refresh=force_refresh,
                reason=reason,
                snapshot_origin=snapshot_origin,
            )
        return self._run_preflight_analysis(snapshot_payload, source=source, force_refresh=force_refresh)

    @staticmethod
    def _select_manifest_chunk(
        chunks: Sequence[Any],
        selection_span: tuple[int, int],
    ) -> Mapping[str, Any] | None:
        start, end = selection_span
        width = max(1, end - start)
        center = start + width // 2
        fallback: Mapping[str, Any] | None = None
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            chunk_start = AIController._coerce_index(chunk.get("start"), start)
            chunk_end = AIController._coerce_index(chunk.get("end"), chunk_start + 1)
            if chunk_end <= chunk_start:
                chunk_end = chunk_start + 1
            if chunk_start <= center < chunk_end:
                return chunk
            if fallback is None and start < chunk_end and end > chunk_start:
                fallback = chunk
        return fallback

    def configure_budget_policy(self, policy: ContextBudgetPolicy | None) -> None:
        """Update (or initialize) the active budget policy."""

        model_name = getattr(getattr(self.client, "settings", None), "model", None)
        self.budget_policy = self._budget_manager.configure_policy(
            policy,
            model_name=model_name,
            max_context_tokens=self.max_context_tokens,
        )
        runtime = self._subagent_runtime
        if runtime and runtime.manager is not None:
            runtime.manager.update_budget_policy(self.budget_policy)

    def configure_subagents(self, config: SubagentRuntimeConfig | None) -> None:
        """Enable or disable the subagent sandbox at runtime."""

        runtime_manager = self._subagent_runtime
        if runtime_manager is None:
            runtime_manager = SubagentRuntimeManager(tool_resolver=self._tool_registry_snapshot)
            self._subagent_runtime = runtime_manager
        runtime = runtime_manager.configure(
            client=self.client,
            config=config,
            budget_policy=self.budget_policy,
        )
        self.subagent_config = runtime

    def configure_debug_event_logging(
        self,
        *,
        enabled: bool | None = None,
        event_log_dir: Path | str | None = None,
    ) -> None:
        """Rebuild the event logger with the latest toggle/directory."""

        if enabled is not None:
            self.debug_event_logging = bool(enabled)
        if event_log_dir is not None:
            self.event_log_dir = event_log_dir
        self._event_logger = ChatEventLogger(
            enabled=bool(self.debug_event_logging),
            base_dir=self.event_log_dir,
        )

    def unregister_tool(self, name: str) -> None:
        """Remove a tool and rebuild the agent graph."""

        if name in self.tools:
            self.tools.pop(name)
            LOGGER.debug("Unregistered tool: %s", name)
            self._schedule_graph_rebuild()

    def update_client(self, client: AIClient) -> None:
        """Swap the underlying AI client (e.g., when settings change)."""

        self.client = client
        if self._subagent_runtime is not None:
            self._subagent_runtime.update_client(client)

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
        self._ingest_snapshot_manifest(snapshot)
        self._remember_snapshot(snapshot)
        message_plan = self._build_messages(prompt, snapshot, history)
        outline_hint = self._outline_routing_hint(prompt, snapshot)
        analysis_hint = self._analysis_hint_message(snapshot)
        base_messages = list(message_plan.messages)
        insert_index = 1
        if analysis_hint:
            base_messages.insert(insert_index, analysis_hint)
            message_plan.prompt_tokens += self._estimate_message_tokens(analysis_hint)
            insert_index += 1
        if outline_hint:
            hint_message = {"role": "system", "content": outline_hint}
            base_messages.insert(insert_index, hint_message)
            message_plan.prompt_tokens += self._estimate_message_tokens(hint_message)
            insert_index += 1
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
            chunk_tracker = _ChunkFlowTracker(document_id=self._resolve_document_id(snapshot))
            self._chunk_flow_tracker = chunk_tracker
            plot_tracker: _PlotLoopTracker | None = None
            if self._should_enforce_plot_loop(snapshot):
                plot_tracker = _PlotLoopTracker(document_id=self._resolve_document_id(snapshot))
            self._plot_loop_tracker = plot_tracker
            turn_metrics = self._new_turn_context(
                snapshot=snapshot,
                prompt_tokens=message_plan.prompt_tokens,
                conversation_length=len(base_messages),
                response_reserve=completion_budget,
            )
            document_id = self._resolve_document_id(snapshot)
            document_path = snapshot.get("path") or snapshot.get("tab_name")
            event_logger = self._event_logger or ChatEventLogger(enabled=False)
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
                runtime = self._subagent_runtime
                if runtime is not None and runtime.manager is not None and self.subagent_config.enabled:
                    try:
                        subagent_jobs, subagent_messages = await self._run_subagent_pipeline(
                            prompt=prompt,
                            snapshot=snapshot,
                            turn_context=turn_metrics,
                        )
                    except Exception:  # pragma: no cover - defensive guard
                        LOGGER.debug("Subagent pipeline failed; continuing without helper context", exc_info=True)
                        subagent_jobs = []
                        subagent_messages = []
                    job_pointer_ids = tuple(job.job_id for job in subagent_jobs if job.job_id)
                    document_id_hint = self._resolve_document_id(snapshot)
                    for message in subagent_messages:
                        pending_tokens = self._estimate_message_tokens(message)
                        decision = self._evaluate_budget(
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
                        self._register_subagent_trace_entry(
                            message,
                            token_count=pending_tokens,
                            job_ids=job_pointer_ids,
                            document_id=document_id_hint,
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
                    assistant_message = turn.assistant_message
                    assistant_tokens = self._estimate_message_tokens(assistant_message)
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
                            if tool_followup_prompts_sent < self.max_tool_followup_prompts:
                                reminder_text = self._tool_only_response_prompt()
                                LOGGER.debug(
                                    "Injected tool follow-up reminder (empty assistant response, attempt=%s)",
                                    tool_followup_prompts_sent + 1,
                                )
                                reminder_message = {"role": "system", "content": reminder_text}
                                conversation.append(reminder_message)
                                conversation_tokens += self._estimate_message_tokens(reminder_message)
                                tool_followup_prompts_sent += 1
                                continue
                            if tool_followup_user_prompts_sent < self.max_tool_followup_user_prompts:
                                user_followup = self._tool_only_response_user_prompt(executed_tool_calls)
                                LOGGER.debug(
                                    "Injected tool follow-up user prompt (empty assistant response, attempt=%s)",
                                    tool_followup_user_prompts_sent + 1,
                                )
                                user_message = {"role": "user", "content": user_followup}
                                conversation.append(user_message)
                                conversation_tokens += self._estimate_message_tokens(user_message)
                                tool_followup_user_prompts_sent += 1
                                continue
                            response_text = self._tool_only_response_fallback(executed_tool_calls)
                            assistant_message["content"] = response_text
                            new_tokens = self._estimate_message_tokens(assistant_message)
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
                            and patch_reminders_sent < self.max_pending_patch_reminders
                        ):
                            reminder = self._pending_patch_prompt()
                            LOGGER.debug("Injected patch reminder (pending diff)")
                            reminder_message = {"role": "system", "content": reminder}
                            conversation.append(reminder_message)
                            conversation_tokens += self._estimate_message_tokens(reminder_message)
                            patch_reminders_sent += 1
                            continue
                        plot_tracker = self._plot_loop_tracker
                        if (
                            plot_tracker is not None
                            and plot_tracker.needs_update_prompt()
                            and plot_loop_reminders_sent < self.max_pending_patch_reminders
                        ):
                            reminder_message = {
                                "role": "system",
                                "content": plot_tracker.update_prompt(),
                            }
                            LOGGER.debug("Injected plot-state update reminder")
                            conversation.append(reminder_message)
                            conversation_tokens += self._estimate_message_tokens(reminder_message)
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
                    active_log.log_tool_batch(
                        turn_index=turn_count,
                        records=tool_records,
                        messages=tool_messages,
                    )
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
                        elif name == "plot_state_update" and not self._tool_call_failed(record):
                            plot_loop_reminders_sent = 0

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
                completion_warnings: list[str] = []
                if pending_patch_application:
                    completion_warnings.append("pending_patch_pending")
                if chunk_tracker.warning_active:
                    completion_warnings.append("chunk_flow_warning")
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
                    "graph": self.graph,
                    "trace_compaction": compaction_stats,
                    "subagent_jobs": [job.as_payload() for job in subagent_jobs],
                    "event_log_path": event_log_path,
                }
            except Exception as exc:
                log_context.__exit__(exc.__class__, exc, exc.__traceback__)
                raise
            finally:
                if self._chunk_flow_tracker is chunk_tracker:
                    self._chunk_flow_tracker = None
                if self._plot_loop_tracker is plot_tracker:
                    self._plot_loop_tracker = None

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

        return self._telemetry_manager.get_recent_events(limit)

    def get_budget_status(self) -> dict[str, object] | None:
        """Expose the most recent context budget policy decision."""

        return self._budget_manager.status_snapshot()

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
        self._graph_dirty = False

    def _schedule_graph_rebuild(self) -> None:
        if self._graph_rebuild_suspensions > 0:
            self._graph_dirty = True
            return
        self._rebuild_graph()

    def _flush_graph_rebuilds(self) -> None:
        if self._graph_dirty:
            self._rebuild_graph()

    def _configure_telemetry_sink(self) -> None:
        self._telemetry_manager.configure(
            enabled=self.telemetry_enabled,
            limit=self.telemetry_limit,
            sink=self.telemetry_sink,
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
        async for event in self.client.stream_chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=300,
        ):
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

    @staticmethod
    def _normalize_temperature(value: float | None) -> float:
        default = 0.2
        if value is None:
            return default
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(candidate, 2.0))

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

    def _tool_only_response_prompt(self) -> str:
        return (
            "You executed tools but did not provide any assistant response. "
            "Summarize the tool results for the user and describe next actions before ending the turn."
        )

    def _tool_only_response_user_prompt(self, records: Sequence[Mapping[str, Any]]) -> str:
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
            "Draft the assistant response now summarizing the tool output before taking any new tools.\nSummary:\n{summary}"
        ).format(tool=tool_name, status=status, summary=summary)

    def _tool_only_response_fallback(self, records: Sequence[Mapping[str, Any]]) -> str:
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
        self._record_analysis_summary(context, document_id)
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
        chunk_tracker = self._chunk_flow_tracker
        plot_tracker = self._plot_loop_tracker
        for record in records:
            payload = self._deserialize_tool_result(record)
            if chunk_tracker is not None:
                chunk_lines = chunk_tracker.observe_tool(record, payload if isinstance(payload, Mapping) else None)
                if chunk_lines:
                    chunk_hint = self._format_guardrail_hint("Chunk Flow", chunk_lines)
                    if chunk_hint and chunk_hint not in seen:
                        hints.append(chunk_hint)
                        seen.add(chunk_hint)
            if plot_tracker is not None:
                plot_lines = plot_tracker.observe_tool(record, payload if isinstance(payload, Mapping) else None)
                if plot_lines:
                    plot_hint = self._format_guardrail_hint("Plot Loop", plot_lines)
                    if plot_hint and plot_hint not in seen:
                        hints.append(plot_hint)
                        seen.add(plot_hint)
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
            telemetry_service.emit("chunk_flow.requested", {
                "document_id": chunk.get("document_id") or self.document_id,
                "chunk_id": chunk.get("chunk_id"),
                "chunk_length": length,
                "window_start": start,
                "window_end": end,
                "pointerized": bool(chunk.get("pointer")),
                "source": "document_chunk",
            })
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

    def _record_analysis_summary(self, context: dict[str, Any], document_id: str | None) -> None:
        if not document_id:
            return
        advice = self.get_latest_analysis_advice(document_id)
        if advice is None:
            return
        context["analysis_chunk_profile"] = advice.chunk_profile
        context["analysis_required_tools"] = tuple(advice.required_tools)
        context["analysis_optional_tools"] = tuple(advice.optional_tools)
        context["analysis_must_refresh_outline"] = bool(advice.must_refresh_outline)
        if advice.plot_state_status:
            context["analysis_plot_state_status"] = advice.plot_state_status
        if advice.concordance_status:
            context["analysis_concordance_status"] = advice.concordance_status
        warning_codes = tuple(warning.code for warning in advice.warnings if warning.code)
        if warning_codes:
            context["analysis_warning_codes"] = warning_codes
        if advice.cache_state:
            context["analysis_cache_state"] = advice.cache_state
        context["analysis_generated_at"] = advice.generated_at
        if advice.rule_trace:
            context["analysis_rule_trace"] = tuple(advice.rule_trace)

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
        return self._budget_manager.evaluate(
            prompt_tokens=prompt_tokens,
            response_reserve=response_reserve,
            document_id=self._resolve_document_id(snapshot),
            pending_tool_tokens=pending_tool_tokens,
            suppress_telemetry=suppress_telemetry,
        )

    def _record_budget_usage(
        self,
        *,
        run_id: str,
        prompt_tokens: int,
        response_reserve: int | None,
    ) -> None:
        self._budget_manager.record_usage(
            run_id=run_id,
            prompt_tokens=prompt_tokens,
            response_reserve=response_reserve,
        )

    def _emit_context_usage(self, context: dict[str, Any]) -> None:
        if not self.telemetry_enabled:
            return
        context.setdefault("timestamp", context.get("timestamp", time.time()))
        context.setdefault("run_id", context.get("run_id") or uuid.uuid4().hex)
        self._telemetry_manager.emit_context_usage(context)

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

    async def _run_subagent_pipeline(
        self,
        *,
        prompt: str,
        snapshot: Mapping[str, Any],
        turn_context: dict[str, Any],
    ) -> tuple[list[SubagentJob], list[dict[str, str]]]:
        runtime = self._subagent_runtime
        manager = runtime.manager if runtime else None
        if manager is None or not self.subagent_config.enabled:
            return [], []

        jobs = self._plan_subagent_jobs(prompt, snapshot, turn_context)
        if not jobs:
            return [], []

        results = await manager.run_jobs(jobs)
        summary_text = self._subagent_summary_message(results)
        messages: list[dict[str, str]] = []
        if summary_text:
            messages.append({"role": "system", "content": summary_text})
        plot_hint = self._maybe_update_plot_state(snapshot, results)
        if plot_hint:
            messages.append({"role": "system", "content": plot_hint})
        concordance_hint = self._maybe_update_character_map(snapshot, results)
        if concordance_hint:
            messages.append({"role": "system", "content": concordance_hint})
        turn_context["subagent_jobs"] = len(results)
        return results, messages

    def _plan_subagent_jobs(
        self,
        prompt: str,
        snapshot: Mapping[str, Any],
        turn_context: Mapping[str, Any],
    ) -> list[SubagentJob]:
        config = self.subagent_config
        if not config.enabled:
            return []
        selection_span = self._selection_span(snapshot)
        if selection_span is None:
            return []
        start, end = selection_span
        span_length = end - start
        if span_length <= 0 or span_length < config.selection_min_chars:
            return []
        text, window_start, window_end = self._snapshot_text_segment(snapshot)
        selection_within_window = bool(text) and start >= window_start and end <= window_end
        selection_text = ""
        if selection_within_window and text:
            local_start = start - window_start
            local_end = end - window_start
            selection_text = text[local_start:local_end]
        chunk_context = self._build_manifest_chunk_context(
            snapshot,
            selection_span,
            hydrate_text=not selection_within_window,
        )
        if not selection_text and (chunk_context is None or not chunk_context.text):
            return []
        context_text = chunk_context.text if chunk_context else None
        preview_source = (context_text or selection_text or "").strip()
        if not preview_source:
            return []
        preview = preview_source[: config.chunk_preview_chars].strip()
        if not preview:
            return []

        document_id = chunk_context.document_id if chunk_context else (self._resolve_document_id(snapshot) or "document")
        version = snapshot.get("version") or snapshot.get("document_version")
        chunk_id = chunk_context.chunk_id if chunk_context else f"selection:{start}-{end}"
        char_range = chunk_context.char_range if chunk_context else (start, end)
        chunk_hash = chunk_context.chunk_hash if chunk_context else None
        if not chunk_hash:
            chunk_hash = self._hash_subagent_chunk(document_id, version, preview_source)
        pointer_id = chunk_context.pointer_id if chunk_context and chunk_context.pointer_id else f"selection:{document_id}:{chunk_id}"
        token_source = context_text or selection_text or preview_source
        token_estimate = self._estimate_text_tokens(token_source)
        chunk_ref = ChunkReference(
            document_id=document_id,
            chunk_id=chunk_id,
            version_id=str(version) if version else None,
            pointer_id=pointer_id,
            char_range=char_range,
            token_estimate=token_estimate,
            chunk_hash=chunk_hash,
            preview=preview,
        )
        allowed_tools = tuple(tool for tool in config.allowed_tools if tool in self.tools)
        budget = self._build_subagent_budget(token_estimate)
        instructions = self._render_subagent_instructions(prompt, chunk_ref)
        job = SubagentJob(
            job_id=uuid.uuid4().hex,
            parent_run_id=str(turn_context.get("run_id") or uuid.uuid4().hex),
            instructions=instructions,
            chunk_ref=chunk_ref,
            allowed_tools=allowed_tools,
            budget=budget,
            dedup_hash=chunk_hash,
        )
        return [job]

    @staticmethod
    def _selection_span(snapshot: Mapping[str, Any]) -> tuple[int, int] | None:
        selection = snapshot.get("selection")
        start: int | None = None
        end: int | None = None
        if isinstance(selection, Mapping):
            start = selection.get("start")  # type: ignore[assignment]
            end = selection.get("end")  # type: ignore[assignment]
        elif isinstance(selection, Sequence) and len(selection) >= 2:
            try:
                start = int(selection[0])  # type: ignore[arg-type]
                end = int(selection[1])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
        if start is None or end is None:
            return None
        try:
            start_idx = int(start)
            end_idx = int(end)
        except (TypeError, ValueError):
            return None
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx
        doc_length = snapshot.get("length")
        if not isinstance(doc_length, int):
            text = snapshot.get("text")
            doc_length = len(text) if isinstance(text, str) else 0
        start_idx = max(0, min(start_idx, doc_length))
        end_idx = max(start_idx, min(end_idx, doc_length))
        if end_idx == start_idx:
            return None
        return (start_idx, end_idx)

    @staticmethod
    def _snapshot_text_segment(snapshot: Mapping[str, Any]) -> tuple[str, int, int]:
        text = snapshot.get("text")
        if not isinstance(text, str) or not text:
            return "", 0, 0
        text_range = snapshot.get("text_range")
        start_offset = 0
        end_offset = len(text)
        if isinstance(text_range, Mapping):
            start_offset = AIController._coerce_index(text_range.get("start"), 0)
            end_offset = AIController._coerce_index(text_range.get("end"), start_offset + len(text))
        if end_offset <= start_offset:
            end_offset = start_offset + len(text)
        return text, start_offset, end_offset

    @staticmethod
    def _coerce_index(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _hash_subagent_chunk(document_id: str, version: Any, chunk_text: str) -> str:
        token = f"{document_id}:{version}:{chunk_text}".encode("utf-8", errors="ignore")
        return hashlib.sha1(token).hexdigest()

    def _build_subagent_budget(self, token_estimate: int) -> SubagentBudget:
        prompt_cap = min(self.max_context_tokens // 2, max(512, token_estimate + 256))
        reserve_slice = max(256, self.response_token_reserve // 4 if self.response_token_reserve else 256)
        completion_cap = min(reserve_slice, self.response_token_reserve or reserve_slice)
        budget = SubagentBudget(
            max_prompt_tokens=prompt_cap,
            max_completion_tokens=max(256, completion_cap),
            max_runtime_seconds=45.0,
            max_tool_iterations=0,
        )
        return budget.clamp()

    def _render_subagent_instructions(self, prompt: str, chunk_ref: ChunkReference) -> str:
        base_prompt = self.subagent_config.instructions_template.strip()
        user_prompt = prompt.strip() or "(no additional user guidance provided)"
        document_label = chunk_ref.document_id or "document"
        return (
            f"{base_prompt}\n\nUser prompt:\n{user_prompt}\n\n"
            f"Focus on document '{document_label}' chunk {chunk_ref.chunk_id}. "
            "Summaries must stay under 200 tokens and include:"
            "\n1. Current intent\n2. Risks or continuity gaps\n3. Recommended follow-up tools."
        )

    def _subagent_summary_message(self, jobs: Sequence[SubagentJob]) -> str:
        if not jobs:
            return ""
        lines: list[str] = []
        for job in jobs:
            if job.state != SubagentJobState.SUCCEEDED or job.result is None:
                continue
            summary = (job.result.summary or "").strip()
            if not summary:
                continue
            label = job.chunk_ref.chunk_id or job.job_id
            trimmed = summary if len(summary) <= 280 else f"{summary[:277].rstrip()}…"
            lines.append(f"- {label}: {trimmed}")
            if len(lines) >= 4:
                break
        if not lines:
            if any(job.state == SubagentJobState.FAILED for job in jobs):
                return "Subagent scouting report: helper job failed; rerun if deeper analysis is required."
            return ""
        header = "Subagent scouting report (chunk-level analysis):"
        return "\n".join([header, *lines])

    def _maybe_update_plot_state(
        self,
        snapshot: Mapping[str, Any],
        jobs: Sequence[SubagentJob],
    ) -> str | None:
        if not jobs or not self.subagent_config.plot_scaffolding_enabled:
            return None
        runtime = self._subagent_runtime
        if runtime is None:
            return None
        store = runtime.ensure_plot_state_store()

        document_id = self._resolve_document_id(snapshot)
        ingested = 0
        for job in jobs:
            if job.state != SubagentJobState.SUCCEEDED or job.result is None:
                continue
            summary = (job.result.summary or "").strip()
            if not summary:
                continue
            chunk = job.chunk_ref
            target_document_id = chunk.document_id or document_id
            if not target_document_id:
                continue
            store.ingest_chunk_summary(
                target_document_id,
                summary,
                version_id=chunk.version_id,
                chunk_hash=chunk.chunk_hash,
                pointer_id=chunk.pointer_id,
                metadata={
                    "source_job_id": job.job_id,
                    "tokens_used": job.result.tokens_used,
                    "latency_ms": job.result.latency_ms,
                },
            )
            ingested += 1

        if not ingested:
            return None

        doc_label = document_id or jobs[0].chunk_ref.document_id or "document"
        return (
            f"Plot scaffolding refreshed for '{doc_label}'. Call PlotOutlineTool before editing for continuity "
            "and follow up with PlotStateUpdateTool after applying chunk edits."
        )

    def _maybe_update_character_map(
        self,
        snapshot: Mapping[str, Any],
        jobs: Sequence[SubagentJob],
    ) -> str | None:
        if not jobs or not self.subagent_config.plot_scaffolding_enabled:
            return None
        runtime = self._subagent_runtime
        if runtime is None:
            return None
        store = runtime.ensure_character_map_store()

        document_id = self._resolve_document_id(snapshot)
        ingested = 0
        for job in jobs:
            if job.state != SubagentJobState.SUCCEEDED or job.result is None:
                continue
            summary = (job.result.summary or "").strip()
            if not summary:
                continue
            chunk = job.chunk_ref
            target_document_id = chunk.document_id or document_id
            if not target_document_id:
                continue
            store.ingest_summary(
                target_document_id,
                summary,
                version_id=chunk.version_id,
                chunk_id=chunk.chunk_id,
                pointer_id=chunk.pointer_id,
                chunk_hash=chunk.chunk_hash,
                char_range=chunk.char_range,
            )
            ingested += 1

        if not ingested:
            return None

        doc_label = document_id or jobs[0].chunk_ref.document_id or "document"
        return (
            f"Character concordance refreshed for '{doc_label}'. Call CharacterMapTool to review "
            "entity mentions before editing across scenes."
        )

    def _register_subagent_trace_entry(
        self,
        message: MutableMapping[str, Any],
        *,
        token_count: int,
        job_ids: Sequence[str],
        document_id: str | None,
    ) -> None:
        compactor = self._trace_compactor
        if compactor is None:
            return
        record_id = "subagent:" + (job_ids[0] if job_ids else uuid.uuid4().hex)
        record: dict[str, Any] = {
            "id": record_id,
            "name": "subagent_scouting_report",
            "pointer_kind": "subagent_summary",
            "subagent_jobs": list(job_ids),
            "document_id": document_id,
            "summarizable": True,
        }
        entry = compactor.new_entry(
            message,
            record,
            raw_content=str(message.get("content") or ""),
            summarizable=True,
        )
        compactor.commit_entry(entry, current_tokens=token_count)

    def _tool_registry_snapshot(self) -> Mapping[str, ToolRegistration]:
        return dict(self.tools)

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
        doc_length = snapshot.get("length")
        if isinstance(doc_length, int):
            doc_chars = max(0, doc_length)
        elif isinstance(raw_text, str):
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
        stream_kwargs["temperature"] = self.temperature

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
        return _ModelTurnResult(assistant_message=assistant_message, response_text=response_text, tool_calls=tool_calls)

    def _parse_embedded_tool_calls(self, text: str, *, start_index: int = 0) -> tuple[str, list[_ToolCallRequest]]:
        if not text or "<" not in text:
            return text, []
        normalized, index_map = self._normalize_tool_marker_text(text)
        matches = list(_TOOL_CALLS_BLOCK_RE.finditer(normalized))
        if not matches:
            return text, []

        cleaned_parts: list[str] = []
        cursor = 0
        parsed_calls: list[_ToolCallRequest] = []
        for block in matches:
            block_start, block_end = block.span()
            orig_block_start = index_map[block_start]
            orig_block_end = index_map[block_end]
            cleaned_parts.append(text[cursor:orig_block_start])
            try:
                body_start, body_end = block.span("body")
            except IndexError:  # pragma: no cover - defensive guard
                body_start = body_end = -1
            block_calls: list[_ToolCallRequest] = []
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
    ) -> list[_ToolCallRequest]:
        # `body_start`/`body_end` indices are expressed in the original text.
        body_original = original_text[body_start:body_end]
        # Normalize the sliced body so we can reuse the regex tokenization logic reliably.
        body_normalized, body_index_map = self._normalize_tool_marker_text(body_original)
        parsed: list[_ToolCallRequest] = []
        for match in _TOOL_CALL_ENTRY_RE.finditer(body_normalized):
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
                .translate(_TOOL_MARKER_TRANSLATION)
                .strip()
                .strip("<>|")
            )
            args_text = body_original[orig_args_start:orig_args_end].strip()
            if not name_text:
                continue
            parsed_args = self._try_parse_json_block(args_text)
            ordinal = starting_index + len(parsed)
            call_id = self._parsed_tool_call_id(name_text, ordinal)
            parsed.append(
                _ToolCallRequest(
                    call_id=call_id,
                    name=name_text,
                    index=ordinal,
                    arguments=args_text,
                    parsed=parsed_args,
                )
            )
        return parsed

    @staticmethod
    def _normalize_tool_marker_text(text: str) -> tuple[str, list[int]]:
        if not text:
            return "", [0]
        translation = _TOOL_MARKER_TRANSLATION
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
    def _parsed_tool_call_id(name: str, ordinal: int) -> str:
        safe_name = re.sub(r"\s+", "_", name.strip().lower()) or "tool"
        return f"{safe_name}:{ordinal}"

    @staticmethod
    def _try_parse_json_block(text: str) -> Any | None:
        stripped = text.strip()
        if not stripped:
            return {}
        try:
            return json.loads(stripped, strict=False)
        except (TypeError, ValueError):
            return None

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
            content, resolved_arguments, raw_result, retry_context = await self._execute_tool_call(
                call,
                registration,
                on_event,
            )
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
            if retry_context:
                record["retry"] = dict(retry_context)
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

    def _build_subagent_pointer(self, record: Mapping[str, Any], content_text: str) -> ToolPointerMessage:
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
        pointer = build_pointer(summary, tool_name="SubagentScoutingReport", rehydrate_instructions=instructions)
        pointer.metadata.setdefault("source", "context_budget")
        pointer.metadata["pointer_kind"] = "subagent_summary"
        if job_ids:
            pointer.metadata["job_ids"] = list(job_ids)
        document_id = record.get("document_id")
        if document_id:
            pointer.metadata["document_id"] = str(document_id)
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
        resolved_arguments = self._normalize_tool_arguments(call, resolved_arguments)
        if registration is None:
            message = f"Tool '{call.name or 'unknown'}' is not registered."
            await self._emit_tool_result_event(call, message, None, on_event)
            return message, resolved_arguments, None, None

        block_reason = self._plot_loop_block(call.name)
        if block_reason:
            payload = {"status": "plot_loop_blocked", "reason": block_reason}
            await self._emit_tool_result_event(call, block_reason, payload, on_event)
            return block_reason, resolved_arguments, payload, None

        retry_context: dict[str, Any] | None = None
        try:
            result = await self._invoke_tool_impl(registration.impl, resolved_arguments)
        except DocumentVersionMismatchError as exc:
            if not self._supports_version_retry(call.name):
                raise
            result, retry_context = await self._handle_version_mismatch_retry(
                call,
                registration,
                resolved_arguments,
                exc,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.exception("Tool %s failed", call.name)
            result = f"Tool '{call.name}' failed: {exc}"

        serialized = self._serialize_tool_result(result)
        await self._emit_tool_result_event(call, serialized, result, on_event)
        return serialized, resolved_arguments, result, retry_context

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

    async def _handle_version_mismatch_retry(
        self,
        call: _ToolCallRequest,
        registration: ToolRegistration,
        resolved_arguments: Any,
        error: DocumentVersionMismatchError,
    ) -> tuple[Any, dict[str, Any]]:
        tab_id = self._extract_tab_id(resolved_arguments)
        snapshot = await self._refresh_document_snapshot(tab_id)
        document_id = self._snapshot_document_id(snapshot)
        base_context: dict[str, Any] = {
            "tool": call.name or "unknown",
            "tab_id": tab_id,
            "document_id": document_id,
            "cause": error.cause or "hash_mismatch",
            "attempts": 2,
        }
        LOGGER.warning(
            "Retrying %s after DocumentVersionMismatchError (cause=%s)",
            call.name,
            error.cause,
        )
        try:
            result = await self._invoke_tool_impl(registration.impl, resolved_arguments)
        except DocumentVersionMismatchError as retry_exc:
            failure_context = dict(base_context)
            failure_context["status"] = "failed"
            failure_context["reason"] = "retry_exhausted"
            failure_context["cause"] = retry_exc.cause or failure_context.get("cause")
            self._emit_version_retry_event(failure_context)
            message = self._format_retry_failure_message(call.name, retry_exc)
            return message, failure_context

        success_context = dict(base_context)
        success_context["status"] = "success"
        self._emit_version_retry_event(success_context)
        return result, success_context

    @classmethod
    def _supports_version_retry(cls, tool_name: str | None) -> bool:
        if not tool_name:
            return False
        return tool_name.strip().lower() in cls._RETRYABLE_VERSION_TOOLS

    @staticmethod
    def _extract_tab_id(arguments: Any) -> str | None:
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

    async def _refresh_document_snapshot(self, tab_id: str | None) -> Mapping[str, Any] | None:
        registration = self.tools.get("document_snapshot")
        if registration is None:
            return None
        runner = getattr(registration.impl, "run", registration.impl)
        if not callable(runner):  # pragma: no cover - defensive guard
            return None
        try:
            result = runner(tab_id=tab_id, delta_only=False, include_diff=False)
        except TypeError:
            try:
                result = runner()
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.debug("Snapshot refresh failed", exc_info=True)
                return None
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Snapshot refresh failed", exc_info=True)
            return None
        if inspect.isawaitable(result):
            result = await result
        return dict(result) if isinstance(result, Mapping) else None

    @staticmethod
    def _snapshot_document_id(snapshot: Mapping[str, Any] | None) -> str | None:
        if not isinstance(snapshot, Mapping):
            return None
        document_id = snapshot.get("document_id")
        if isinstance(document_id, str):
            text = document_id.strip()
            return text or None
        return None

    def _emit_version_retry_event(self, payload: Mapping[str, Any]) -> None:
        if not payload:
            return
        event_payload = dict(payload)
        event_payload.setdefault("event_source", "controller")
        telemetry_service.emit("document_edit.retry", event_payload)

    @staticmethod
    def _format_retry_failure_message(
        tool_name: str | None,
        error: DocumentVersionMismatchError,
    ) -> str:
        label = tool_name or "document_edit"
        cause = f" (cause={error.cause})" if getattr(error, "cause", None) else ""
        return (
            f"Tool '{label}' failed: document snapshot was stale even after an automatic retry{cause}. "
            "Call document_snapshot again and rebuild your diff before retrying."
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
            "raw_result": raw_result,
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
            return json.loads(text, strict=False)
        except (ValueError, TypeError):
            return text

    def _normalize_tool_arguments(self, call: _ToolCallRequest, arguments: Any) -> Any:
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

    def _plot_loop_block(self, tool_name: str | None) -> str | None:
        tracker = self._plot_loop_tracker
        if tracker is None:
            return None
        return tracker.before_tool(tool_name)

    def _should_enforce_plot_loop(self, snapshot: Mapping[str, Any]) -> bool:
        if not getattr(self.subagent_config, "plot_scaffolding_enabled", False):
            return False
        document_chars = self._snapshot_document_chars(snapshot) or 0
        if document_chars <= 0:
            return False
        min_chars_raw = getattr(self.subagent_config, "plot_outline_min_chars", 0)
        try:
            min_chars = int(min_chars_raw or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive cast
            min_chars = 0
        min_threshold = max(1, min_chars)
        return document_chars >= min_threshold

    def _snapshot_document_chars(self, snapshot: Mapping[str, Any]) -> int | None:
        document_chars = self._coerce_optional_int(snapshot.get("length"))
        if document_chars is None:
            text = snapshot.get("text")
            if isinstance(text, str):
                document_chars = len(text)
        return document_chars

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


