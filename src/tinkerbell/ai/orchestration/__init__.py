"""High-level AI orchestration utilities for the desktop app."""

from .budget_manager import ContextBudgetExceeded
from .chunk_flow import ChunkContext, ChunkFlowTracker
from .controller import AIController, OpenAIToolSpec, ToolRegistration
from .model_types import MessagePlan, ModelTurnResult, ToolCallRequest
from .runtime_config import AnalysisRuntimeConfig, ChunkingRuntimeConfig
from .subagent_state import SubagentDocumentState
from .turn_tracking import PlotLoopTracker, SnapshotRefreshTracker

# Phase 2 extractions: static utilities and helpers
from .controller_utils import (
    normalize_iterations,
    normalize_scope_origin,
    normalize_context_tokens,
    normalize_response_reserve,
    normalize_temperature,
    coerce_optional_int,
    coerce_optional_float,
    coerce_optional_str,
    sanitize_suggestions,
)
from .scope_helpers import (
    scope_summary_from_arguments,
    scope_fields_from_summary,
    extract_chunk_id,
    parse_chunk_bounds,
)
from .tool_call_parser import (
    parse_embedded_tool_calls,
    normalize_tool_marker_text,
    try_parse_json_block,
)
from .guardrail_hints import (
    format_guardrail_hint,
    outline_guardrail_hints,
    retrieval_guardrail_hints,
)

# WS4: Editor Lock & Diff Review
from .editor_lock import (
    EditorLockManager,
    LockSession,
    LockState,
    LockReason,
    LockableTab,
    TabProvider,
    LockStateListener,
    LockStatusUpdater,
    get_lock_manager,
    set_lock_manager,
    reset_lock_manager,
)
from .transaction import (
    Transaction,
    TransactionManager,
    TransactionState,
    ChangeType,
    StagedChange,
    DocumentSnapshot,
    TransactionError,
    CommitError,
    get_transaction_manager,
    set_transaction_manager,
    reset_transaction_manager,
)
from .checkpoints import (
    CheckpointStore,
    Checkpoint,
    CheckpointType,
    DocumentState,
    CheckpointDiff,
    compute_simple_diff,
    get_checkpoint_store,
    set_checkpoint_store,
    reset_checkpoint_store,
)

# WS6: Tool Dispatcher
from .tool_dispatcher import (
    ToolDispatcher,
    DispatchResult,
    ToolContextProvider,
    DispatchListener,
    create_tool_dispatcher,
)

# WS9: Subagent Executor
from .subagent_executor import (
    SubagentExecutor,
    SubagentExecutorConfig,
    ResponseParseError,
    extract_json_from_response,
    validate_response_schema,
    normalize_analysis_result,
    validate_transformation_output,
    create_subagent_orchestrator,
    configure_analyze_tool_executor,
    configure_transform_tool_executor,
)
from .subagent_prompts import (
    get_analysis_prompt,
    get_transform_prompt,
    ANALYSIS_SCHEMAS,
    TRANSFORM_SCHEMAS,
)

__all__ = [
    "AIController",
    "ContextBudgetExceeded",
    "OpenAIToolSpec",
    "ToolRegistration",  # Backwards compatibility alias for OpenAIToolSpec
    # Extracted model types
    "ToolCallRequest",
    "ModelTurnResult",
    "MessagePlan",
    # Chunk flow tracking
    "ChunkContext",
    "ChunkFlowTracker",
    # Turn tracking
    "SnapshotRefreshTracker",
    "PlotLoopTracker",
    # Runtime configs
    "ChunkingRuntimeConfig",
    "AnalysisRuntimeConfig",
    # Subagent state
    "SubagentDocumentState",
    # Phase 2: Controller utilities
    "normalize_iterations",
    "normalize_scope_origin",
    "normalize_context_tokens",
    "normalize_response_reserve",
    "normalize_temperature",
    "coerce_optional_int",
    "coerce_optional_float",
    "coerce_optional_str",
    "sanitize_suggestions",
    # Phase 2: Scope helpers
    "scope_summary_from_arguments",
    "scope_fields_from_summary",
    "extract_chunk_id",
    "parse_chunk_bounds",
    # Phase 2: Tool call parsing
    "parse_embedded_tool_calls",
    "normalize_tool_marker_text",
    "try_parse_json_block",
    # Phase 2: Guardrail hints
    "format_guardrail_hint",
    "outline_guardrail_hints",
    "retrieval_guardrail_hints",
    # WS4.1: Editor Lock
    "EditorLockManager",
    "LockSession",
    "LockState",
    "LockReason",
    "LockableTab",
    "TabProvider",
    "LockStateListener",
    "LockStatusUpdater",
    "get_lock_manager",
    "set_lock_manager",
    "reset_lock_manager",
    # WS4.2: Transaction
    "Transaction",
    "TransactionManager",
    "TransactionState",
    "ChangeType",
    "StagedChange",
    "DocumentSnapshot",
    "TransactionError",
    "CommitError",
    "get_transaction_manager",
    "set_transaction_manager",
    "reset_transaction_manager",
    # WS4.4: Checkpoints
    "CheckpointStore",
    "Checkpoint",
    "CheckpointType",
    "DocumentState",
    "CheckpointDiff",
    "compute_simple_diff",
    "get_checkpoint_store",
    "set_checkpoint_store",
    "reset_checkpoint_store",
    # WS6: Tool Dispatcher
    "ToolDispatcher",
    "DispatchResult",
    "ToolContextProvider",
    "DispatchListener",
    "create_tool_dispatcher",
    # WS9: Subagent Executor
    "SubagentExecutor",
    "SubagentExecutorConfig",
    "ResponseParseError",
    "extract_json_from_response",
    "validate_response_schema",
    "normalize_analysis_result",
    "validate_transformation_output",
    "create_subagent_orchestrator",
    "configure_analyze_tool_executor",
    "configure_transform_tool_executor",
    "get_analysis_prompt",
    "get_transform_prompt",
    "ANALYSIS_SCHEMAS",
    "TRANSFORM_SCHEMAS",
]
