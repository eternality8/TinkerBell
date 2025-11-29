"""Domain layer for UI architecture.

This package contains domain managers that encapsulate business logic
and state management, independent of UI concerns. Each manager is
responsible for a specific domain area and communicates via the event bus.

Domain Managers:
    - DocumentStore: Document/tab lifecycle management
    - AITurnManager: AI turn execution state machine
    - ReviewManager: Pending AI edit review management
    - OverlayManager: Diff overlay state tracking
    - SessionStore: Session persistence management
    - EmbeddingStore: Document embedding runtime management
    - OutlineStore: Document outline/summary management

All domain managers:
    - Receive dependencies via constructor injection
    - Emit events to notify other layers of state changes
    - Have no direct dependencies on Qt or UI widgets
"""

from __future__ import annotations

# Domain managers will be exported here as they are implemented:
from .document_store import DocumentStore
from .ai_turn_manager import AITurnManager
from .review_manager import ReviewManager
from .overlay_manager import OverlayManager
from .session_store import SessionStore
from .embedding_store import EmbeddingStore
from .outline_store import OutlineStore

__all__: list[str] = [
    "DocumentStore",
    "AITurnManager",
    "ReviewManager",
    "OverlayManager",
    "SessionStore",
    "EmbeddingStore",
    "OutlineStore",
]
