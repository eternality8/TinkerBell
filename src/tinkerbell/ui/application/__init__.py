"""Application layer for UI architecture.

This package contains use cases that orchestrate domain operations.
Each use case encapsulates a specific user action or workflow,
coordinating multiple domain managers to accomplish a task.

Use Cases:
    - Document Operations: New, Open, Save, Close, Revert, Restore
    - AI Operations: Run AI Turn, Cancel AI Turn
    - Review Operations: Accept Review, Reject Review
    - Import Operations: Import Document

Coordinator:
    - AppCoordinator: Facade that delegates to use cases and provides
      widget reference management for the presentation layer.

All use cases:
    - Receive dependencies via constructor injection
    - Orchestrate domain managers without direct UI coupling
    - May emit events for cross-cutting concerns
"""

from __future__ import annotations

# Use cases and coordinator will be exported here as they are implemented:
from .document_ops import (
    DialogProvider,
    NewDocumentUseCase,
    OpenDocumentUseCase,
    SaveDocumentUseCase,
    CloseDocumentUseCase,
    RevertDocumentUseCase,
    RestoreWorkspaceUseCase,
)
from .ai_ops import (
    SnapshotProvider,
    RunAITurnUseCase,
    CancelAITurnUseCase,
)
from .review_ops import (
    DocumentRestorer,
    OverlayRestorer,
    ChatRestorer,
    WorkspaceSyncer,
    AcceptResult,
    RejectResult,
    AcceptReviewUseCase,
    RejectReviewUseCase,
)
from .import_ops import (
    ImportDialogProvider,
    ImportResult,
    ImportDocumentUseCase,
)
from .coordinator import AppCoordinator

__all__: list[str] = [
    # Document operations
    "DialogProvider",
    "NewDocumentUseCase",
    "OpenDocumentUseCase",
    "SaveDocumentUseCase",
    "CloseDocumentUseCase",
    "RevertDocumentUseCase",
    "RestoreWorkspaceUseCase",
    # AI operations
    "SnapshotProvider",
    "RunAITurnUseCase",
    "CancelAITurnUseCase",
    # Review operations
    "DocumentRestorer",
    "OverlayRestorer",
    "ChatRestorer",
    "WorkspaceSyncer",
    "AcceptResult",
    "RejectResult",
    "AcceptReviewUseCase",
    "RejectReviewUseCase",
    # Import operations
    "ImportDialogProvider",
    "ImportResult",
    "ImportDocumentUseCase",
    # Coordinator
    "AppCoordinator",
]
