"""High-level AI orchestration utilities for the desktop app."""

from .budget_manager import ContextBudgetExceeded
from .controller import AIController, ToolRegistration

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

__all__ = [
    "AIController",
    "ContextBudgetExceeded",
    "ToolRegistration",
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
]
