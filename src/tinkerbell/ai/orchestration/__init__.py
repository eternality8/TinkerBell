"""High-level AI orchestration utilities for the desktop app."""

from .model_types import (
    MessagePlan,
    ModelTurnResult,
    OpenAIToolSpec,
    ToolCallRequest,
    ToolRegistration,
)
from .subagent_state import SubagentDocumentState

# New orchestrator facade
from .orchestrator import (
    AIOrchestrator,
    OrchestratorConfig,
    ChatResult,
    StreamCallback,
)

# Tool call parsing (used by pipeline)
from .tool_call_parser import (
    parse_embedded_tool_calls,
    normalize_tool_marker_text,
    try_parse_json_block,
)

# =============================================================================
# New Orchestration Pipeline (Phase 1 Cleanup)
# =============================================================================

# Core types
from .types import (
    TurnInput,
    TurnOutput,
    TurnConfig,
    TurnMetrics,
    Message,
    ModelResponse,
    ParsedToolCall,
    PreparedTurn,
    AnalyzedTurn,
    BudgetEstimate,
    ToolCallRecord,
)

# Turn runner
from .runner import (
    TurnRunner,
    RunnerConfig,
    ContentCallback,
    ToolCallback,
    create_runner,
)

# Tool system
from .tools import (
    ToolRegistry,
    ToolSpec,
    Tool,
    SimpleTool,
    ToolCategory,
    ToolRegistration as NewToolRegistration,
    DuplicateToolError,
    ToolNotFoundError,
)
from .tools.executor import (
    ToolExecutor as NewToolExecutor,
    ExecutorConfig,
    ToolExecutionError,
)

# Pipeline stages (for advanced usage)
from .pipeline.prepare import (
    prepare_turn,
    build_messages,
    estimate_budget,
    TokenCounter,
)
from .pipeline.analyze import (
    analyze_turn,
    generate_hints,
    AnalysisProvider,
)
from .pipeline.execute import (
    execute_model,
    parse_response,
    ModelClient,
)
from .pipeline.tools import (
    execute_tools,
    append_tool_results,
    ToolResults,
    ToolExecutionResult,
)
from .pipeline.finish import (
    finish_turn,
    finish_turn_with_error,
    collect_metrics,
    TurnTimer,
)

# =============================================================================
# End New Orchestration Pipeline
# =============================================================================

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
    # New orchestrator (replaces AIController)
    "AIOrchestrator",
    "OrchestratorConfig",
    "ChatResult",
    "StreamCallback",
    # Tool specifications
    "OpenAIToolSpec",
    "ToolRegistration",  # Backwards compatibility alias for OpenAIToolSpec
    # Extracted model types
    "ToolCallRequest",
    "ModelTurnResult",
    "MessagePlan",
    # Subagent state
    "SubagentDocumentState",
    # Tool call parsing
    "parse_embedded_tool_calls",
    "normalize_tool_marker_text",
    "try_parse_json_block",
    # ==========================================================================
    # New Orchestration Pipeline (Phase 1 Cleanup)
    # ==========================================================================
    # Core types
    "TurnInput",
    "TurnOutput",
    "TurnConfig",
    "TurnMetrics",
    "Message",
    "ModelResponse",
    "ParsedToolCall",
    "PreparedTurn",
    "AnalyzedTurn",
    "BudgetEstimate",
    "ToolCallRecord",
    # Turn runner
    "TurnRunner",
    "RunnerConfig",
    "ContentCallback",
    "ToolCallback",
    "create_runner",
    # Tool system
    "ToolRegistry",
    "ToolSpec",
    "Tool",
    "SimpleTool",
    "ToolCategory",
    "NewToolRegistration",
    "DuplicateToolError",
    "ToolNotFoundError",
    "NewToolExecutor",
    "ExecutorConfig",
    "ToolExecutionError",
    # Pipeline stages
    "prepare_turn",
    "build_messages",
    "estimate_budget",
    "TokenCounter",
    "analyze_turn",
    "generate_hints",
    "AnalysisProvider",
    "execute_model",
    "parse_response",
    "ModelClient",
    "execute_tools",
    "append_tool_results",
    "ToolResults",
    "ToolExecutionResult",
    "finish_turn",
    "finish_turn_with_error",
    "collect_metrics",
    "TurnTimer",
    # ==========================================================================
    # End New Orchestration Pipeline
    # ==========================================================================
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
