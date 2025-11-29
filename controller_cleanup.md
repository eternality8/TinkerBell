# Controller.py Full Rewrite Plan

This document outlines a comprehensive plan to restructure `src/tinkerbell/ai/orchestration/controller.py` from ~3,925 lines into a modular architecture with focused, single-responsibility components.

---

## Executive Summary

The current `AIController` class violates the Single Responsibility Principle by handling:
- Chat orchestration loop
- Tool registration and management
- Tool execution and dispatch
- Version retry logic
- Subagent pipeline coordination
- Preflight analysis
- Message building and token estimation
- Telemetry and metrics recording
- Suggestion generation
- Budget evaluation
- Guardrail hint generation

**Target Architecture**: Split into 8-10 focused modules, each under 500 lines, with `AIController` reduced to a ~400-line thin facade that delegates to specialized coordinators.

---

## Current State Analysis

### File Statistics
- **Current lines**: 3,925
- **Methods in AIController**: ~100+
- **Instance fields**: 50+
- **Already extracted modules**: 9 files (model_types, chunk_flow, turn_tracking, runtime_config, subagent_state, tool_call_parser, controller_utils, scope_helpers, guardrail_hints)

### Method Categories by Responsibility

| Category | Methods | Est. Lines | Target Module |
|----------|---------|------------|---------------|
| **Core Chat Loop** | `run_chat`, `_runner` (inner) | ~300 | `chat_orchestrator.py` |
| **Tool Execution** | `_handle_tool_calls`, `_execute_tool_call`, `_execute_via_dispatcher`, `_invoke_tool_impl`, `_serialize_tool_result` | ~350 | `tool_executor.py` |
| **Version Retry** | `_handle_version_mismatch_retry`, `_refresh_document_snapshot`, `_inject_snapshot_metadata`, `_supports_version_retry`, `_emit_version_retry_event`, `_format_retry_failure_message` | ~200 | `version_retry.py` |
| **NeedsRange Handling** | `_format_needs_range_payload`, `_format_needs_range_message`, `_resolve_needs_range_span_hint`, `_span_hint_from_*` (5 methods) | ~150 | `needs_range_handler.py` |
| **Subagent Pipeline** | `_run_subagent_pipeline`, `_plan_subagent_jobs`, `_subagent_*` (15+ methods), `_maybe_update_plot_state`, `_maybe_update_character_map` | ~500 | `subagent_coordinator.py` |
| **Analysis Pipeline** | `_run_preflight_analysis`, `_build_analysis_input`, `_ensure_analysis_agent`, `_format_analysis_hint`, `_analysis_hint_message`, `_advisor_tool_entrypoint`, `request_analysis_advice` | ~250 | `analysis_coordinator.py` |
| **Message Building** | `_build_messages`, `_sanitize_history`, `_estimate_message_tokens`, `_estimate_text_tokens`, `_effective_response_reserve`, `_outline_routing_hint` | ~200 | `message_builder.py` |
| **Metrics/Telemetry** | `_new_turn_context`, `_record_tool_names`, `_record_tool_metrics`, `_record_scope_metrics`, `_capture_*_metrics`, `_copy_snapshot_*`, `_record_analysis_summary`, `_emit_context_usage` | ~250 | `metrics_recorder.py` |
| **Suggestions** | `suggest_followups`, `_build_suggestion_messages`, `_complete_simple_chat`, `_parse_suggestion_response`, `_try_parse_json_suggestions`, `_sanitize_suggestions` | ~100 | `suggestions.py` |
| **Tool Registration** | `register_tool`, `register_tools`, `unregister_tool`, `_build_tool_registration`, `_coerce_registration`, `_store_tool_registration`, `available_tools` | ~120 | `tool_registry_facade.py` |
| **Model Turn Invocation** | `_invoke_model_turn`, `_parse_embedded_tool_calls`, `_parse_tool_call_entries`, `_normalize_tool_marker_text` | ~200 | `model_turn.py` |
| **Tool Compaction** | `_compact_tool_messages`, `_build_tool_pointer`, `_build_subagent_pointer`, `_pointer_rehydrate_instructions`, `_register_subagent_trace_entry` | ~150 | `trace_compaction.py` |
| **Guardrail Hints** | `_guardrail_hints_from_records` (orchestration only, hints already extracted) | ~80 | Keep in controller or `guardrail_coordinator.py` |
| **Configuration** | `configure_*` methods (8+), `set_*` methods, `__post_init__` | ~200 | Keep in controller facade |
| **Graph Management** | `_rebuild_graph`, `_schedule_graph_rebuild`, `_flush_graph_rebuilds`, `suspend_graph_rebuilds` | ~50 | Keep in controller facade |
| **Utility Delegations** | Static method wrappers to extracted utils | ~100 | Remove (call extracted utils directly) |

---

## Target Architecture

```
src/tinkerbell/ai/orchestration/
├── __init__.py                    # Re-exports (update)
├── controller.py                  # AIController facade (~400 lines)
├── chat_orchestrator.py           # NEW: run_chat loop (~400 lines)
├── tool_executor.py               # NEW: Tool execution (~350 lines)
├── version_retry.py               # NEW: Version mismatch handling (~200 lines)
├── needs_range_handler.py         # NEW: NeedsRange error handling (~150 lines)
├── subagent_coordinator.py        # NEW: Subagent pipeline (~500 lines)
├── analysis_coordinator.py        # NEW: Preflight analysis (~250 lines)
├── message_builder.py             # NEW: Message construction (~200 lines)
├── metrics_recorder.py            # NEW: Telemetry/metrics (~250 lines)
├── suggestions.py                 # NEW: Follow-up suggestions (~100 lines)
├── model_turn.py                  # NEW: Model turn invocation (~200 lines)
├── trace_compaction.py            # NEW: Tool result compaction (~150 lines)
├── tool_registry_facade.py        # NEW: Tool registration (~120 lines)
│
├── # Existing extracted modules (keep as-is):
├── budget_manager.py
├── chunk_flow.py
├── controller_utils.py
├── event_log.py
├── guardrail_hints.py
├── model_types.py
├── runtime_config.py
├── scope_helpers.py
├── subagent_executor.py
├── subagent_prompts.py
├── subagent_runtime.py
├── subagent_state.py
├── telemetry_manager.py
├── tool_call_parser.py
├── tool_dispatcher.py
├── transaction.py
├── turn_context.py
└── turn_tracking.py
```

---

## Implementation Plan

### Phase 1: Extract Independent Modules (Low Risk)

These modules have minimal dependencies on controller state and can be extracted first.

#### 1.1 `suggestions.py` (~100 lines)
**Dependencies**: `self.client`, `self.temperature`

```python
# suggestions.py
"""Follow-up suggestion generation for chat conversations."""

from __future__ import annotations

import json
from typing import Any, Iterable, Mapping, Sequence

from ..client import AIClient

_SUGGESTION_SYSTEM_PROMPT = (
    "You are a helpful writing copilot asked to propose up to {max_suggestions} focused follow-up suggestions. "
    "Each suggestion should be a short imperative phrase (no numbering) tailored to the prior conversation transcript."
)


class SuggestionGenerator:
    """Generates contextual follow-up suggestions based on chat history."""
    
    def __init__(self, client: AIClient, temperature: float = 0.2) -> None:
        self._client = client
        self._temperature = temperature
    
    async def generate(
        self,
        history: Sequence[Mapping[str, str]],
        *,
        max_suggestions: int = 4,
    ) -> list[str]:
        """Generate follow-up suggestions from conversation history."""
        if not history:
            return []
        messages = self._build_messages(history, max_suggestions)
        response_text = await self._complete_chat(messages)
        return self._parse_response(response_text, max_suggestions)
    
    def _build_messages(self, history: Sequence[Mapping[str, str]], max_suggestions: int) -> list[dict[str, str]]:
        # ... (move implementation)
    
    async def _complete_chat(self, messages: Sequence[Mapping[str, Any]]) -> str:
        # ... (move implementation)
    
    def _parse_response(self, text: str, max_suggestions: int) -> list[str]:
        # ... (move implementation)
    
    @staticmethod
    def sanitize(raw_items: Iterable[Any], max_suggestions: int) -> list[str]:
        # ... (move implementation)
```

**Controller changes**:
```python
# In AIController
self._suggestion_generator = SuggestionGenerator(self.client, self.temperature)

async def suggest_followups(self, history, *, max_suggestions=4):
    return await self._suggestion_generator.generate(history, max_suggestions=max_suggestions)
```

#### 1.2 `message_builder.py` (~200 lines)
**Dependencies**: `self.client`, `self.max_context_tokens`, `self.response_token_reserve`, prompts module

```python
# message_builder.py
"""Message construction and token estimation for chat turns."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .. import prompts
from ..client import AIClient
from .model_types import MessagePlan

_PROMPT_HEADROOM = 6_000


class MessageBuilder:
    """Builds and manages chat messages with token budget awareness."""
    
    def __init__(
        self,
        client: AIClient,
        max_context_tokens: int = 128_000,
        response_token_reserve: int = 16_000,
    ) -> None:
        self._client = client
        self._max_context_tokens = max_context_tokens
        self._response_token_reserve = response_token_reserve
    
    def build_messages(
        self,
        prompt: str,
        snapshot: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> MessagePlan:
        # ... (move implementation)
    
    def estimate_message_tokens(self, message: Mapping[str, Any]) -> int:
        # ... (move implementation)
    
    def estimate_text_tokens(self, text: str) -> int:
        # ... (move implementation)
    
    def sanitize_history(
        self,
        history: Sequence[Mapping[str, Any]],
        limit: int = 20,
        *,
        token_budget: int | None = None,
    ) -> list[dict[str, str]]:
        # ... (move implementation)
    
    def outline_routing_hint(self, prompt: str, snapshot: Mapping[str, Any]) -> str | None:
        # ... (move implementation)
    
    def effective_response_reserve(self, context_limit: int) -> int:
        # ... (move implementation)
    
    def update_limits(self, max_context_tokens: int | None = None, response_reserve: int | None = None) -> None:
        # ... update internal limits
```

#### 1.3 `metrics_recorder.py` (~250 lines)
**Dependencies**: `self.client`, telemetry service, analysis advice cache

```python
# metrics_recorder.py
"""Telemetry and metrics recording for chat turns."""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Any, Mapping, Sequence

from ...services import telemetry as telemetry_service
from .controller_utils import coerce_optional_int, coerce_optional_float, normalize_scope_origin
from .scope_helpers import scope_fields_from_summary, range_bounds_from_mapping


class MetricsRecorder:
    """Records metrics and telemetry for chat turn execution."""
    
    def __init__(self, telemetry_enabled: bool = False) -> None:
        self._telemetry_enabled = telemetry_enabled
    
    def new_turn_context(
        self,
        *,
        snapshot: Mapping[str, Any],
        prompt_tokens: int,
        conversation_length: int,
        response_reserve: int | None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        # ... (move implementation)
    
    def record_tool_names(self, context: dict[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
        # ... (move implementation)
    
    def record_tool_metrics(self, context: dict[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
        # ... (move implementation)
    
    def record_scope_metrics(self, context: dict[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
        # ... (move implementation)
    
    def capture_outline_tool_metrics(self, context: dict[str, Any], record: Mapping[str, Any]) -> None:
        # ... (move implementation)
    
    def capture_retrieval_tool_metrics(self, context: dict[str, Any], record: Mapping[str, Any]) -> None:
        # ... (move implementation)
    
    def copy_snapshot_outline_metrics(self, context: dict[str, Any], snapshot: Mapping[str, Any]) -> None:
        # ... (move implementation)
    
    def copy_snapshot_embedding_metadata(self, context: dict[str, Any], snapshot: Mapping[str, Any]) -> None:
        # ... (move implementation)
    
    def emit_context_usage(self, context: dict[str, Any]) -> None:
        # ... (move implementation)
```

### Phase 2: Extract Medium-Complexity Modules

These require passing controller references or callbacks for state access.

#### 2.1 `version_retry.py` (~200 lines)

```python
# version_retry.py
"""Version mismatch retry handling for tool execution."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Mapping, MutableMapping

from ...services import telemetry as telemetry_service
from ...services.bridge import DocumentVersionMismatchError
from .scope_helpers import scope_summary_from_arguments, scope_fields_from_summary

LOGGER = logging.getLogger(__name__)

_RETRYABLE_VERSION_TOOLS: frozenset[str] = frozenset({"document_apply_patch", "search_replace"})


class VersionRetryHandler:
    """Handles version mismatch errors with automatic retry."""
    
    def __init__(
        self,
        snapshot_refresher: Callable[[str | None], Mapping[str, Any] | None],
        tool_invoker: Callable[[Any, Any], Any],
    ) -> None:
        self._refresh_snapshot = snapshot_refresher
        self._invoke_tool = tool_invoker
    
    @staticmethod
    def supports_retry(tool_name: str | None) -> bool:
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
        # ... (move implementation)
    
    @staticmethod
    def inject_snapshot_metadata(arguments: Any, snapshot: Mapping[str, Any] | None) -> None:
        # ... (move implementation)
    
    @staticmethod
    def extract_tab_id(arguments: Any) -> str | None:
        # ... (move implementation)
    
    def _emit_retry_event(self, payload: Mapping[str, Any]) -> None:
        # ... (move implementation)
    
    @staticmethod
    def format_failure_message(tool_name: str | None, error: DocumentVersionMismatchError) -> str:
        # ... (move implementation)
```

#### 2.2 `needs_range_handler.py` (~150 lines)

```python
# needs_range_handler.py
"""Handler for NeedsRangeError in tool execution."""

from __future__ import annotations

from typing import Any, Mapping

from ..tools.errors import NeedsRangeError
from .scope_helpers import scope_summary_from_arguments, scope_fields_from_summary, extract_chunk_id, parse_chunk_bounds


class NeedsRangeHandler:
    """Handles NeedsRangeError by building informative payloads with span hints."""
    
    def __init__(
        self,
        snapshot_span_resolver: callable,
        selection_tool_invoker: callable | None = None,
    ) -> None:
        self._resolve_snapshot_span = snapshot_span_resolver
        self._invoke_selection_tool = selection_tool_invoker
    
    def format_payload(
        self,
        call: "ToolCallRequest",
        resolved_arguments: Any,
        error: NeedsRangeError,
    ) -> dict[str, Any]:
        # ... (move implementation)
    
    @staticmethod
    def format_message(tool_name: str | None, payload: Mapping[str, Any]) -> str:
        # ... (move implementation)
    
    def resolve_span_hint(self, resolved_arguments: Any, tab_id: str | None) -> dict[str, Any] | None:
        # ... (move implementation)
    
    # ... span_hint_from_* methods
```

#### 2.3 `tool_executor.py` (~350 lines)

```python
# tool_executor.py
"""Tool execution and result handling."""

from __future__ import annotations

import ast
import inspect
import json
import logging
import time
from typing import Any, Mapping, Sequence

from ..client import AIStreamEvent
from .model_types import ToolCallRequest
from .version_retry import VersionRetryHandler
from .needs_range_handler import NeedsRangeHandler

LOGGER = logging.getLogger(__name__)


class ToolExecutor:
    """Executes tool calls and handles results."""
    
    def __init__(
        self,
        tool_registry: Mapping[str, "OpenAIToolSpec"],
        dispatcher: "ToolDispatcher | None" = None,
        version_retry_handler: VersionRetryHandler | None = None,
        needs_range_handler: NeedsRangeHandler | None = None,
        token_estimator: callable | None = None,
    ) -> None:
        self._tools = tool_registry
        self._dispatcher = dispatcher
        self._version_retry = version_retry_handler
        self._needs_range = needs_range_handler
        self._estimate_tokens = token_estimator or (lambda x: len(x) // 4)
    
    async def handle_tool_calls(
        self,
        tool_calls: Sequence[ToolCallRequest],
        on_event: callable | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
        # ... (move implementation)
    
    async def execute_tool_call(
        self,
        call: ToolCallRequest,
        registration: "OpenAIToolSpec | None",
        on_event: callable | None,
    ) -> tuple[str, Any, Any, dict[str, Any] | None]:
        # ... (move implementation)
    
    async def execute_via_dispatcher(
        self,
        call: ToolCallRequest,
        resolved_arguments: Any,
        on_event: callable | None,
    ) -> tuple[str, Any, Any, dict[str, Any] | None]:
        # ... (move implementation)
    
    async def invoke_tool_impl(self, tool_impl: Any, arguments: Any) -> Any:
        # ... (move implementation)
    
    def serialize_result(self, result: Any) -> str:
        # ... (move implementation)
    
    def coerce_arguments(self, raw_arguments: str | None, parsed: Any | None) -> Any:
        # ... (move implementation)
    
    def normalize_arguments(self, call: ToolCallRequest, arguments: Any) -> Any:
        # ... (move implementation)
    
    def build_tool_record(self, ...) -> dict[str, Any]:
        # ... (move implementation)
```

### Phase 3: Extract Complex Coordinators

These modules orchestrate multiple subsystems.

#### 3.1 `subagent_coordinator.py` (~500 lines)

```python
# subagent_coordinator.py
"""Subagent pipeline coordination for chunk-level analysis."""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from ..ai_types import ChunkReference, SubagentBudget, SubagentJob, SubagentJobState, SubagentRuntimeConfig
from ...services import telemetry as telemetry_service
from .chunk_flow import ChunkContext
from .subagent_state import SubagentDocumentState
from .subagent_runtime import SubagentRuntimeManager

LOGGER = logging.getLogger(__name__)
_CODE_EXTENSIONS = frozenset({".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".rs", ".go"})


class SubagentCoordinator:
    """Coordinates subagent pipeline execution for chat turns."""
    
    def __init__(
        self,
        config: SubagentRuntimeConfig,
        runtime_manager: SubagentRuntimeManager,
        tool_registry: Mapping[str, "OpenAIToolSpec"],
        token_estimator: callable,
        chunk_hydrator: callable,
    ) -> None:
        self._config = config
        self._runtime = runtime_manager
        self._tools = tool_registry
        self._estimate_tokens = token_estimator
        self._hydrate_chunk = chunk_hydrator
        self._doc_states: dict[str, SubagentDocumentState] = {}
    
    async def run_pipeline(
        self,
        *,
        prompt: str,
        snapshot: Mapping[str, Any],
        turn_context: dict[str, Any],
    ) -> tuple[list[SubagentJob], list[dict[str, str]]]:
        # ... (move implementation)
    
    def plan_jobs(
        self,
        prompt: str,
        snapshot: Mapping[str, Any],
        turn_context: Mapping[str, Any],
    ) -> list[SubagentJob]:
        # ... (move implementation)
    
    def _state_for_document(self, document_id: str) -> SubagentDocumentState:
        # ... (move implementation)
    
    def _infer_document_kind(self, snapshot: Mapping[str, Any]) -> str:
        # ... (move implementation)
    
    def _prioritize_chunks(self, snapshot: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        # ... (move implementation)
    
    def _dirty_manifest_chunks(self, chunks: Sequence[Mapping[str, Any]], state: SubagentDocumentState) -> list[Mapping[str, Any]]:
        # ... (move implementation)
    
    def _build_chunk_context(self, snapshot: Mapping[str, Any], manifest: Mapping[str, Any], entry: Mapping[str, Any], *, hydrate_text: bool) -> ChunkContext | None:
        # ... (move implementation)
    
    def _trigger_reasons(self, snapshot: Mapping[str, Any], manifest: Mapping[str, Any], state: SubagentDocumentState, dirty_chunks: Sequence[Mapping[str, Any]]) -> list[str]:
        # ... (move implementation)
    
    def _build_budget(self, token_estimate: int) -> SubagentBudget:
        # ... (move implementation)
    
    def _render_instructions(self, prompt: str, chunk_ref: ChunkReference) -> str:
        # ... (move implementation)
    
    def _summary_message(self, jobs: Sequence[SubagentJob]) -> str:
        # ... (move implementation)
    
    def maybe_update_plot_state(self, snapshot: Mapping[str, Any], jobs: Sequence[SubagentJob]) -> str | None:
        # ... (move implementation)
    
    def maybe_update_character_map(self, snapshot: Mapping[str, Any], jobs: Sequence[SubagentJob]) -> str | None:
        # ... (move implementation)
    
    def handle_document_changed(self, document_id: str) -> None:
        # ... update edit churn
    
    def handle_document_closed(self, document_id: str) -> None:
        # ... cleanup state
```

#### 3.2 `analysis_coordinator.py` (~250 lines)

```python
# analysis_coordinator.py
"""Preflight analysis coordination for chat turns."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Mapping

from ...services import telemetry as telemetry_service
from ..analysis.agent import AnalysisAgent
from ..analysis.models import AnalysisAdvice, AnalysisInput
from .runtime_config import AnalysisRuntimeConfig, ChunkingRuntimeConfig
from .controller_utils import coerce_optional_int, coerce_optional_float, coerce_optional_str


class AnalysisCoordinator:
    """Coordinates preflight analysis for chat turns."""
    
    def __init__(
        self,
        config: AnalysisRuntimeConfig,
        chunking_config: ChunkingRuntimeConfig,
    ) -> None:
        self._config = config
        self._chunking_config = chunking_config
        self._agent: AnalysisAgent | None = None
        self._advice_cache: dict[str, AnalysisAdvice] = {}
        self._snapshot_cache: dict[str, Mapping[str, Any]] = {}
    
    @property
    def enabled(self) -> bool:
        return bool(getattr(self._config, "enabled", False))
    
    def run_analysis(
        self,
        snapshot: Mapping[str, Any],
        *,
        source: str = "controller",
        force_refresh: bool = False,
    ) -> AnalysisAdvice | None:
        # ... (move implementation)
    
    def build_analysis_input(self, snapshot: Mapping[str, Any]) -> AnalysisInput:
        # ... (move implementation)
    
    def format_hint(self, advice: AnalysisAdvice) -> str:
        # ... (move implementation)
    
    def hint_message(self, snapshot: Mapping[str, Any]) -> dict[str, str] | None:
        # ... (move implementation)
    
    def get_latest_advice(self, document_id: str | None) -> AnalysisAdvice | None:
        return self._advice_cache.get(document_id) if document_id else None
    
    def get_latest_snapshot(self, document_id: str | None) -> Mapping[str, Any] | None:
        if not document_id:
            return None
        cached = self._snapshot_cache.get(document_id)
        return dict(cached) if cached else None
    
    def remember_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        # ... (move implementation)
    
    def invalidate_document(self, document_id: str) -> None:
        self._advice_cache.pop(document_id, None)
        self._snapshot_cache.pop(document_id, None)
        if self._agent:
            self._agent.invalidate_document(document_id)
    
    def _ensure_agent(self) -> AnalysisAgent:
        # ... (move implementation)
```

#### 3.3 `chat_orchestrator.py` (~400 lines)

This is the core orchestration loop, extracted from `run_chat()`.

```python
# chat_orchestrator.py
"""Main chat turn orchestration loop."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Mapping, Sequence

from ..client import AIClient, AIStreamEvent
from .model_types import MessagePlan, ModelTurnResult
from .turn_context import TurnContext
from .turn_tracking import ChunkFlowTracker, SnapshotRefreshTracker, PlotLoopTracker
from .event_log import ChatEventLogger
from .tool_executor import ToolExecutor
from .message_builder import MessageBuilder
from .metrics_recorder import MetricsRecorder
from .subagent_coordinator import SubagentCoordinator
from .analysis_coordinator import AnalysisCoordinator
from .trace_compaction import TraceCompactor
from .budget_manager import BudgetManager

LOGGER = logging.getLogger(__name__)

ToolCallback = callable  # Callable[[AIStreamEvent], Awaitable[None] | None]


class ChatOrchestrator:
    """Orchestrates the main chat turn execution loop."""
    
    def __init__(
        self,
        client: AIClient,
        tool_executor: ToolExecutor,
        message_builder: MessageBuilder,
        metrics_recorder: MetricsRecorder,
        subagent_coordinator: SubagentCoordinator | None = None,
        analysis_coordinator: AnalysisCoordinator | None = None,
        trace_compactor: TraceCompactor | None = None,
        budget_manager: BudgetManager | None = None,
        event_logger: ChatEventLogger | None = None,
        max_tool_iterations: int = 8,
        max_edits_without_snapshot: int = 4,
        temperature: float = 0.2,
    ) -> None:
        self._client = client
        self._tool_executor = tool_executor
        self._message_builder = message_builder
        self._metrics_recorder = metrics_recorder
        self._subagent_coordinator = subagent_coordinator
        self._analysis_coordinator = analysis_coordinator
        self._trace_compactor = trace_compactor
        self._budget_manager = budget_manager
        self._event_logger = event_logger
        self._max_tool_iterations = max_tool_iterations
        self._max_edits_without_snapshot = max_edits_without_snapshot
        self._temperature = temperature
    
    async def run(
        self,
        prompt: str,
        snapshot: Mapping[str, Any],
        *,
        metadata: Mapping[str, str] | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
        on_event: ToolCallback | None = None,
        tool_specs: Sequence[Any] | None = None,
    ) -> dict:
        """Execute a complete chat turn."""
        # ... (move run_chat implementation)
    
    async def _invoke_model_turn(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        tool_specs: Sequence[Any] | None,
        metadata: Mapping[str, str] | None,
        on_event: ToolCallback | None,
        max_completion_tokens: int | None = None,
    ) -> ModelTurnResult:
        # ... (move implementation)
    
    def _create_trackers(self, snapshot: Mapping[str, Any]) -> tuple[ChunkFlowTracker, SnapshotRefreshTracker | None, PlotLoopTracker | None]:
        # ... create tracking instances
    
    def _build_metadata(self, snapshot: Mapping[str, Any], runtime_metadata: Mapping[str, str] | None) -> dict[str, str] | None:
        # ... (move implementation)
```

### Phase 4: Refactor AIController as Facade

After extracting all components, `AIController` becomes a thin facade:

```python
# controller.py (~400 lines)
"""Agent executor façade - thin coordinator delegating to specialized components."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from ..client import AIClient
from ..ai_types import AgentConfig, SubagentRuntimeConfig
from ..agents.graph import build_agent_graph
from ..services.context_policy import ContextBudgetPolicy
from .budget_manager import BudgetManager
from .chat_orchestrator import ChatOrchestrator
from .tool_executor import ToolExecutor
from .message_builder import MessageBuilder
from .metrics_recorder import MetricsRecorder
from .subagent_coordinator import SubagentCoordinator
from .analysis_coordinator import AnalysisCoordinator
from .suggestions import SuggestionGenerator
from .telemetry_manager import TelemetryManager
from .runtime_config import AnalysisRuntimeConfig, ChunkingRuntimeConfig
from .tool_registry_facade import ToolRegistryFacade
from .model_types import OpenAIToolSpec

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class AIController:
    """High-level interface invoked by the chat panel.
    
    This is a facade that delegates to specialized coordinators:
    - ChatOrchestrator: Main chat turn execution
    - ToolExecutor: Tool call handling
    - SubagentCoordinator: Subagent pipeline
    - AnalysisCoordinator: Preflight analysis
    - SuggestionGenerator: Follow-up suggestions
    """
    
    client: AIClient
    tools: MutableMapping[str, OpenAIToolSpec] = field(default_factory=dict)
    # ... configuration fields ...
    
    # Internal coordinators
    _orchestrator: ChatOrchestrator | None = field(default=None, init=False, repr=False)
    _tool_registry: ToolRegistryFacade | None = field(default=None, init=False, repr=False)
    _suggestion_generator: SuggestionGenerator | None = field(default=None, init=False, repr=False)
    # ... other coordinators ...
    
    def __post_init__(self) -> None:
        """Initialize all internal coordinators."""
        self._initialize_coordinators()
    
    def _initialize_coordinators(self) -> None:
        """Build and wire all internal components."""
        # Initialize in dependency order
        self._tool_registry = ToolRegistryFacade(self.tools)
        self._message_builder = MessageBuilder(...)
        self._metrics_recorder = MetricsRecorder(...)
        self._tool_executor = ToolExecutor(...)
        self._analysis_coordinator = AnalysisCoordinator(...)
        self._subagent_coordinator = SubagentCoordinator(...)
        self._suggestion_generator = SuggestionGenerator(self.client, self.temperature)
        self._orchestrator = ChatOrchestrator(
            client=self.client,
            tool_executor=self._tool_executor,
            message_builder=self._message_builder,
            ...
        )
    
    # === Public API (delegates to coordinators) ===
    
    async def run_chat(self, prompt: str, doc_snapshot: Mapping[str, Any] | None, **kwargs) -> dict:
        """Execute a chat turn."""
        return await self._orchestrator.run(prompt, doc_snapshot or {}, **kwargs)
    
    async def suggest_followups(self, history: Sequence[Mapping[str, str]], *, max_suggestions: int = 4) -> list[str]:
        """Generate follow-up suggestions."""
        return await self._suggestion_generator.generate(history, max_suggestions=max_suggestions)
    
    def register_tool(self, name: str, tool: Any, **kwargs) -> None:
        """Register a tool."""
        self._tool_registry.register(name, tool, **kwargs)
        self._schedule_graph_rebuild()
    
    def unregister_tool(self, name: str) -> None:
        """Unregister a tool."""
        self._tool_registry.unregister(name)
        self._schedule_graph_rebuild()
    
    def available_tools(self) -> tuple[str, ...]:
        """Return registered tool names."""
        return self._tool_registry.available_tools()
    
    # === Configuration methods ===
    
    def configure_context_window(self, **kwargs) -> None:
        self._message_builder.update_limits(**kwargs)
    
    def configure_analysis(self, config: AnalysisRuntimeConfig | None = None, **kwargs) -> None:
        self._analysis_coordinator.configure(config, **kwargs)
    
    def configure_subagents(self, config: SubagentRuntimeConfig | None) -> None:
        self._subagent_coordinator.configure(config)
    
    # ... other configure_* methods ...
    
    # === Graph management (kept in facade) ===
    
    def _rebuild_graph(self) -> None:
        # ... (keep implementation)
    
    @contextlib.contextmanager
    def suspend_graph_rebuilds(self) -> Iterable[None]:
        # ... (keep implementation)
    
    # === Lifecycle ===
    
    async def aclose(self) -> None:
        """Clean shutdown."""
        # ... cleanup coordinators
```

---

## Migration Strategy

### Step 1: Create New Modules Without Removing Code
1. Create each new module with implementations
2. Add comprehensive unit tests for new modules
3. Verify tests pass independently

### Step 2: Wire New Modules into Controller
1. Import new modules in controller.py
2. Initialize coordinators in `__post_init__`
3. Delegate method calls to coordinators
4. Run full test suite to verify behavior

### Step 3: Remove Duplicated Code
1. Delete moved implementations from controller.py
2. Remove now-unused imports
3. Update `__init__.py` exports
4. Run full test suite

### Step 4: Update Tests
1. Update `test_ai_controller.py` imports
2. Add tests for new coordinator classes
3. Remove tests for deleted internal methods
4. Ensure coverage remains high

---

## Risk Mitigation

### Testing Strategy
- Maintain 100% test pass rate at every step
- Add integration tests that exercise full flow
- Use feature flags if needed for gradual rollout

### Backwards Compatibility
- `AIController` public API remains unchanged
- Internal type aliases maintained for tests: `_ToolCallRequest`, `_ModelTurnResult`, etc.
- `__init__.py` exports preserved

### Rollback Plan
- Each phase is a separate commit
- Can revert individual phases if issues arise
- Original controller.py preserved in git history

---

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| controller.py lines | 3,925 | ~400 |
| Largest module | 3,925 | <500 |
| Methods in AIController | 100+ | ~20 |
| Instance fields | 50+ | ~15 |
| Test coverage | Current | Maintain or improve |
| Test pass rate | 100% | 100% |

---

## Implementation Order

1. **Week 1**: Phase 1 (suggestions.py, message_builder.py, metrics_recorder.py)
2. **Week 2**: Phase 2 (version_retry.py, needs_range_handler.py, tool_executor.py)
3. **Week 3**: Phase 3 (subagent_coordinator.py, analysis_coordinator.py)
4. **Week 4**: Phase 4 (chat_orchestrator.py, controller facade refactor)
5. **Week 5**: Testing, documentation, cleanup

---

## Open Questions

1. **Circular dependencies**: Some coordinators need references to each other. Consider:
   - Dependency injection via factory
   - Lazy initialization
   - Event-based communication

2. **Shared state**: Document caches, outline digest cache, etc. need to be accessible across coordinators. Options:
   - Shared state container passed to all coordinators
   - Event bus for cache invalidation
   - Single source of truth in controller

3. **Tool dispatcher integration**: The new `ToolDispatcher` is already separate. Ensure clean integration with `ToolExecutor`.

---

## Appendix: Method Mapping

Complete mapping of current methods to target modules:

| Current Method | Target Module | Notes |
|----------------|---------------|-------|
| `__post_init__` | controller.py | Simplified |
| `run_chat` | chat_orchestrator.py | Core loop |
| `suggest_followups` | suggestions.py | Delegate |
| `register_tool` | tool_registry_facade.py | Delegate |
| `unregister_tool` | tool_registry_facade.py | Delegate |
| `register_tools` | tool_registry_facade.py | Delegate |
| `available_tools` | tool_registry_facade.py | Delegate |
| `configure_*` | controller.py | Keep (thin wrappers) |
| `_build_messages` | message_builder.py | Move |
| `_sanitize_history` | message_builder.py | Move |
| `_estimate_*_tokens` | message_builder.py | Move |
| `_handle_tool_calls` | tool_executor.py | Move |
| `_execute_tool_call` | tool_executor.py | Move |
| `_execute_via_dispatcher` | tool_executor.py | Move |
| `_invoke_tool_impl` | tool_executor.py | Move |
| `_handle_version_mismatch_retry` | version_retry.py | Move |
| `_refresh_document_snapshot` | version_retry.py | Move |
| `_format_needs_range_*` | needs_range_handler.py | Move |
| `_run_subagent_pipeline` | subagent_coordinator.py | Move |
| `_plan_subagent_jobs` | subagent_coordinator.py | Move |
| `_run_preflight_analysis` | analysis_coordinator.py | Move |
| `_build_analysis_input` | analysis_coordinator.py | Move |
| `_new_turn_context` | metrics_recorder.py | Move |
| `_record_tool_*` | metrics_recorder.py | Move |
| `_invoke_model_turn` | chat_orchestrator.py | Move |
| `_compact_tool_messages` | trace_compaction.py | Move |
| `_build_tool_pointer` | trace_compaction.py | Move |
| `_guardrail_hints_from_records` | chat_orchestrator.py | Move |
| Static utility wrappers | Remove | Call utils directly |
