# TinkerBell UI Architecture Redesign

## Executive Summary

The current `ui/` folder is a tangled web of interdependent controllers, services, and helpers that evolved organically rather than being designed coherently. The ~2,200-line `main_window.py` acts as a god object, directly orchestrating everything from AI turns to file dialogs to telemetry updates.

This document proposes a clean, layered architecture that separates concerns and makes the codebase maintainable.

---

## Current State Analysis

### Files and Their Responsibilities

| File | LOC | Primary Purpose | Problems |
|------|-----|-----------------|----------|
| `main_window.py` | 2,234 | God object coordinating everything | Massive, handles too much, tight coupling |
| `main_window_helpers.py` | 152 | Utility functions and protocol adapters | Reasonable, but helpers indicate extraction needed |
| `window_shell.py` | 230 | Menu/toolbar/splitter assembly | Well-scoped, acceptable |
| `ai_turn_coordinator.py` | 184 | AI turn execution and streaming | Good extraction, but leaks state back to MainWindow |
| `ai_review_controller.py` | 275 | Pending edit review state machine | Well-designed state container |
| `document_session_service.py` | 393 | File dialogs, workspace restore, settings persistence | Mixed concerns: persistence + UI dialogs |
| `document_state_monitor.py` | 254 | Editor state tracking, autosave, cache bus | Reasonable, but callback-heavy |
| `document_status_service.py` | 315 | Status payload aggregation for status window | Good, but tightly coupled to internal types |
| `document_status.py` | 59 | Status formatting helpers | Fine |
| `embedding_controller.py` | 670 | Embedding runtime management | Huge, mixes provider configuration + validation + index management |
| `import_controller.py` | 103 | File import workflow | Well-scoped |
| `manual_tool_controller.py` | 150 | Manual command response formatting | Fine, but orphaned in MainWindow |
| `outline_runtime.py` | 82 | Outline worker lifecycle | Well-scoped |
| `review_overlay_manager.py` | 275 | Diff overlay display and accept/reject | Good extraction |
| `settings_runtime.py` | 247 | Theme, logging, AI client lifecycle | Reasonable scope |
| `telemetry_controller.py` | 397 | Status bar updates for subagent/chunk flow | Mixed: telemetry + status bar updates |
| `tool_trace_presenter.py` | 140 | Streaming tool trace aggregation | Fine |
| `tools/provider.py` | 82 | Tool wiring context factory | Fine |
| `models/*.py` | ~100 | Data classes for actions, window state, traces | Good |
| `widgets/*.py` | ~850 | Qt dialogs (command palette, status window) | Reasonable |

### Dependency Graph (Simplified)

```
MainWindow
├── WindowChrome (menus/toolbars)
├── TabbedEditorWidget (external)
├── ChatPanel (external)
├── StatusBar (external)
├── DocumentSessionService
│   ├── AIReviewController
│   └── ReviewOverlayManager
├── DocumentStateMonitor
├── AITurnCoordinator
│   └── EditorLockManager (external)
├── AIReviewController
│   └── ReviewOverlayManager
├── EmbeddingController
│   └── OutlineRuntime
├── TelemetryController
├── SettingsRuntime
├── ImportController
├── ToolProvider
├── ToolTracePresenter
└── DocumentStatusService
```

### Core Problems

1. **God Object**: `MainWindow` has ~100+ methods, 30+ instance variables, and coordinates every subsystem.

2. **Callback Hell**: Services communicate via callbacks injected in constructors:
   ```python
   DocumentStateMonitor(
       unsaved_cache_persister=self._document_session.persist_unsaved_cache,
       refresh_window_title=self._refresh_window_title,
       sync_workspace_state=lambda persist: self._document_session.sync_workspace_state(persist=persist),
       current_path_getter=self._document_session.get_current_path,
       ...
   )
   ```

3. **Circular Dependencies**: Controllers reference each other through MainWindow:
   - `AIReviewController` needs `clear_diff_overlay` from `ReviewOverlayManager`
   - `ReviewOverlayManager` needs `pending_turn_review` from `AIReviewController`

4. **Mixed Concerns**: 
   - `EmbeddingController` handles provider configuration, validation, AND index management
   - `DocumentSessionService` handles file dialogs AND workspace persistence
   - `TelemetryController` handles telemetry events AND status bar updates

5. **No Clear Boundaries**: Services reach into each other's internals via lambdas and direct property access.

---

## Proposed Architecture

### Design Principles

1. **Single Responsibility**: Each module does one thing well.
2. **Explicit Dependencies**: Use dependency injection with interfaces, not callbacks.
3. **Unidirectional Data Flow**: State changes flow through a central event bus.
4. **Layered Architecture**: Clear separation between coordination, domain, and infrastructure.

### Layer Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                        │
│  MainWindow  │  WindowChrome  │  Dialogs  │  StatusUpdaters     │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       APPLICATION LAYER                          │
│        AppCoordinator (orchestrates use cases)                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Use Cases:                                                │   │
│  │  • OpenDocumentUseCase                                    │   │
│  │  • SaveDocumentUseCase                                    │   │
│  │  • RunAITurnUseCase                                       │   │
│  │  • AcceptReviewUseCase                                    │   │
│  │  • RejectReviewUseCase                                    │   │
│  │  • ImportDocumentUseCase                                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DOMAIN LAYER                             │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │
│  │ DocumentStore  │  │ AITurnManager  │  │ ReviewManager   │   │
│  │ (workspace)    │  │ (orchestrator) │  │ (pending edits) │   │
│  └────────────────┘  └────────────────┘  └─────────────────┘   │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────┐   │
│  │ SessionStore   │  │ EmbeddingStore │  │ OutlineStore    │   │
│  │ (persistence)  │  │ (index + prov) │  │ (memory)        │   │
│  └────────────────┘  └────────────────┘  └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      INFRASTRUCTURE LAYER                        │
│  FileIO  │  SettingsStore  │  UnsavedCache  │  AIClient         │
└─────────────────────────────────────────────────────────────────┘
```

### Event-Driven Communication

Replace callback spaghetti with a typed event bus:

```python
# events.py
from dataclasses import dataclass
from typing import Protocol, TypeVar

T = TypeVar("T")

class Event:
    """Base class for all domain events."""
    pass

@dataclass(frozen=True, slots=True)
class DocumentOpened(Event):
    tab_id: str
    document_id: str
    path: str | None

@dataclass(frozen=True, slots=True)
class DocumentSaved(Event):
    tab_id: str
    path: str

@dataclass(frozen=True, slots=True)
class DocumentModified(Event):
    tab_id: str
    document_id: str
    version_id: int
    content_hash: str

@dataclass(frozen=True, slots=True)
class AITurnStarted(Event):
    turn_id: str
    prompt: str

@dataclass(frozen=True, slots=True)
class AITurnCompleted(Event):
    turn_id: str
    success: bool
    edit_count: int

@dataclass(frozen=True, slots=True)
class EditApplied(Event):
    tab_id: str
    edit_id: str
    action: str
    range: tuple[int, int]

@dataclass(frozen=True, slots=True)
class ReviewStateChanged(Event):
    turn_id: str | None
    ready_for_review: bool
    edit_count: int
    tabs_affected: int

@dataclass(frozen=True, slots=True)
class StatusMessage(Event):
    message: str
    timeout_ms: int | None = None

# Event bus with typed subscriptions
class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[type[Event], list[Callable[[Event], None]]] = {}
    
    def subscribe(self, event_type: type[T], handler: Callable[[T], None]) -> None:
        self._handlers.setdefault(event_type, []).append(handler)
    
    def publish(self, event: Event) -> None:
        for handler in self._handlers.get(type(event), []):
            handler(event)
```

---

## New Module Structure

```
src/tinkerbell/ui/
├── __init__.py
├── app.py                      # Application entry point, DI container setup
├── events.py                   # Event definitions and bus
│
├── presentation/               # UI Layer
│   ├── __init__.py
│   ├── main_window.py          # Thin shell: creates widgets, subscribes to events
│   ├── window_chrome.py        # Menu/toolbar assembly (existing, minimal changes)
│   ├── status_updaters.py      # StatusBar event handlers
│   └── dialogs/
│       ├── __init__.py
│       ├── command_palette.py  # (existing, cleaned)
│       ├── document_status.py  # (existing, cleaned)
│       └── settings_dialog.py  # (move from widgets/)
│
├── application/                # Application Layer (Use Cases)
│   ├── __init__.py
│   ├── coordinator.py          # AppCoordinator: wires use cases to events
│   ├── document_ops.py         # OpenDocument, SaveDocument, CloseDocument
│   ├── ai_ops.py               # RunAITurn, CancelAITurn
│   ├── review_ops.py           # AcceptReview, RejectReview
│   └── import_ops.py           # ImportDocument
│
├── domain/                     # Domain Layer (Business Logic)
│   ├── __init__.py
│   ├── document_store.py       # Wraps DocumentWorkspace with event emission
│   ├── ai_turn_manager.py      # AI turn state machine (from ai_turn_coordinator)
│   ├── review_manager.py       # Pending review state (from ai_review_controller)
│   ├── overlay_manager.py      # Diff overlay state (from review_overlay_manager)
│   ├── session_store.py        # Workspace persistence (from document_session_service)
│   ├── embedding_store.py      # Embedding index + provider (from embedding_controller)
│   └── outline_store.py        # Outline worker lifecycle (from outline_runtime)
│
├── infrastructure/             # Infrastructure adapters
│   ├── __init__.py
│   ├── settings_adapter.py     # Theme, logging, AI client (from settings_runtime)
│   ├── telemetry_adapter.py    # Telemetry event forwarding (from telemetry_controller)
│   └── tool_adapter.py         # Tool wiring context (from tools/provider)
│
└── models/                     # Shared data types
    ├── __init__.py
    ├── actions.py              # (existing)
    ├── window_state.py         # (existing)
    └── events.py               # Domain events (if not in top-level events.py)
```

---

## Key Component Designs

### 1. MainWindow (Thin Shell)

```python
# presentation/main_window.py
class MainWindow(QMainWindow):
    """Thin presentation shell. No business logic."""
    
    def __init__(
        self,
        event_bus: EventBus,
        coordinator: AppCoordinator,
    ) -> None:
        super().__init__()
        self._bus = event_bus
        self._coordinator = coordinator
        
        # Create widgets (pure UI)
        self._editor = TabbedEditorWidget(skip_default_tab=True)
        self._chat_panel = ChatPanel()
        self._status_bar = StatusBar()
        
        # Wire chrome
        self._chrome = WindowChrome(
            window=self,
            editor=self._editor,
            chat_panel=self._chat_panel,
            status_bar=self._status_bar,
            action_callbacks=self._action_callbacks(),
        )
        self._chrome.assemble()
        
        # Subscribe to events for UI updates
        self._subscribe_to_events()
        
        # Initialize coordinator with widget references
        self._coordinator.set_widgets(
            editor=self._editor,
            chat_panel=self._chat_panel,
            status_bar=self._status_bar,
        )
    
    def _subscribe_to_events(self) -> None:
        self._bus.subscribe(StatusMessage, self._handle_status)
        self._bus.subscribe(DocumentOpened, self._handle_document_opened)
        self._bus.subscribe(ReviewStateChanged, self._handle_review_changed)
        # ... other UI-relevant events
    
    def _handle_status(self, event: StatusMessage) -> None:
        self._status_bar.set_message(event.message, timeout_ms=event.timeout_ms)
    
    def _action_callbacks(self) -> dict[str, Callable[[], None]]:
        return {
            "file_new_tab": lambda: self._coordinator.new_document(),
            "file_open": lambda: self._coordinator.open_document(),
            "file_save": lambda: self._coordinator.save_document(),
            "ai_accept_changes": lambda: self._coordinator.accept_review(),
            "ai_reject_changes": lambda: self._coordinator.reject_review(),
            # ...
        }
```

### 2. AppCoordinator (Application Layer)

```python
# application/coordinator.py
class AppCoordinator:
    """Orchestrates use cases in response to user actions."""
    
    def __init__(
        self,
        event_bus: EventBus,
        document_store: DocumentStore,
        ai_turn_manager: AITurnManager,
        review_manager: ReviewManager,
        session_store: SessionStore,
    ) -> None:
        self._bus = event_bus
        self._documents = document_store
        self._ai = ai_turn_manager
        self._reviews = review_manager
        self._sessions = session_store
        
        # Use case instances (stateless operations)
        self._open_doc = OpenDocumentUseCase(document_store, session_store, event_bus)
        self._save_doc = SaveDocumentUseCase(document_store, session_store, event_bus)
        self._run_turn = RunAITurnUseCase(ai_turn_manager, review_manager, event_bus)
        self._accept = AcceptReviewUseCase(review_manager, event_bus)
        self._reject = RejectReviewUseCase(review_manager, document_store, event_bus)
    
    def new_document(self) -> None:
        tab = self._documents.create_tab()
        self._bus.publish(DocumentOpened(tab_id=tab.id, document_id=tab.document_id, path=None))
    
    def open_document(self, path: Path | None = None) -> None:
        self._open_doc.execute(path)
    
    def save_document(self, path: Path | None = None) -> None:
        self._save_doc.execute(path)
    
    async def run_ai_turn(self, prompt: str, metadata: dict) -> None:
        await self._run_turn.execute(prompt, metadata)
    
    def accept_review(self) -> None:
        self._accept.execute()
    
    def reject_review(self) -> None:
        self._reject.execute()
```

### 3. AITurnManager (Domain Layer)

```python
# domain/ai_turn_manager.py
@dataclass(slots=True)
class AITurnState:
    turn_id: str
    prompt: str
    status: Literal["pending", "running", "completed", "failed", "canceled"]
    edit_count: int = 0
    error: str | None = None

class AITurnManager:
    """Manages AI turn execution state."""
    
    def __init__(
        self,
        orchestrator_provider: Callable[[], AIOrchestrator | None],
        event_bus: EventBus,
    ) -> None:
        self._get_orchestrator = orchestrator_provider
        self._bus = event_bus
        self._current_turn: AITurnState | None = None
        self._task: asyncio.Task | None = None
    
    def is_running(self) -> bool:
        return self._current_turn is not None and self._current_turn.status == "running"
    
    async def start_turn(
        self,
        prompt: str,
        snapshot: dict,
        metadata: dict,
        history: list[dict],
        on_stream_event: Callable[[Any], None],
    ) -> AITurnState:
        if self.is_running():
            raise RuntimeError("AI turn already in progress")
        
        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            raise RuntimeError("AI orchestrator unavailable")
        
        turn_id = f"turn-{uuid.uuid4().hex[:8]}"
        self._current_turn = AITurnState(turn_id=turn_id, prompt=prompt, status="running")
        self._bus.publish(AITurnStarted(turn_id=turn_id, prompt=prompt))
        
        try:
            result = await orchestrator.run_chat(
                prompt, snapshot, metadata=metadata, history=history, on_event=on_stream_event,
            )
            self._current_turn.status = "completed"
            self._bus.publish(AITurnCompleted(
                turn_id=turn_id, success=True, edit_count=self._current_turn.edit_count
            ))
            return self._current_turn
        except Exception as exc:
            self._current_turn.status = "failed"
            self._current_turn.error = str(exc)
            self._bus.publish(AITurnCompleted(turn_id=turn_id, success=False, edit_count=0))
            raise
    
    def cancel(self) -> None:
        if not self.is_running():
            return
        orchestrator = self._get_orchestrator()
        if orchestrator:
            orchestrator.cancel()
        if self._current_turn:
            self._current_turn.status = "canceled"
```

### 4. ReviewManager (Domain Layer)

```python
# domain/review_manager.py
@dataclass(slots=True)
class PendingEdit:
    edit_id: str
    tab_id: str
    action: str
    range: tuple[int, int]
    diff: str
    spans: tuple[tuple[int, int], ...]

@dataclass(slots=True)
class ReviewSession:
    tab_id: str
    document_id: str
    snapshot: DocumentState
    edits: list[PendingEdit]
    previous_overlay: DiffOverlayState | None

@dataclass(slots=True)
class PendingReview:
    turn_id: str
    prompt: str
    sessions: dict[str, ReviewSession]
    ready: bool = False

class ReviewManager:
    """Manages pending AI edit reviews."""
    
    def __init__(self, event_bus: EventBus) -> None:
        self._bus = event_bus
        self._pending: PendingReview | None = None
    
    @property
    def pending_review(self) -> PendingReview | None:
        return self._pending
    
    def begin_review(self, turn_id: str, prompt: str) -> None:
        self._pending = PendingReview(turn_id=turn_id, prompt=prompt, sessions={})
        self._emit_state_changed()
    
    def record_edit(self, tab_id: str, document_snapshot: DocumentState, edit: PendingEdit) -> None:
        if self._pending is None:
            return
        session = self._pending.sessions.get(tab_id)
        if session is None:
            session = ReviewSession(
                tab_id=tab_id,
                document_id=document_snapshot.document_id,
                snapshot=document_snapshot,
                edits=[],
                previous_overlay=None,
            )
            self._pending.sessions[tab_id] = session
        session.edits.append(edit)
        self._emit_state_changed()
    
    def finalize(self, success: bool) -> None:
        if self._pending is None:
            return
        if success and self._pending.sessions:
            self._pending.ready = True
        else:
            self._pending = None
        self._emit_state_changed()
    
    def accept(self) -> list[str]:
        """Accept all edits, return list of affected tab IDs."""
        if self._pending is None:
            return []
        tabs = list(self._pending.sessions.keys())
        self._pending = None
        self._emit_state_changed()
        return tabs
    
    def reject(self) -> dict[str, DocumentState]:
        """Reject edits, return map of tab_id -> original snapshot to restore."""
        if self._pending is None:
            return {}
        snapshots = {
            tab_id: session.snapshot 
            for tab_id, session in self._pending.sessions.items()
        }
        self._pending = None
        self._emit_state_changed()
        return snapshots
    
    def _emit_state_changed(self) -> None:
        pending = self._pending
        self._bus.publish(ReviewStateChanged(
            turn_id=pending.turn_id if pending else None,
            ready_for_review=pending.ready if pending else False,
            edit_count=sum(len(s.edits) for s in pending.sessions.values()) if pending else 0,
            tabs_affected=len(pending.sessions) if pending else 0,
        ))
```

---

## Testing Strategy

### Unit Tests
- Each domain manager tested in isolation with mock event bus
- Use cases tested with mock domain managers
- No Qt dependencies in domain/application tests

### Integration Tests
- `AppCoordinator` with real domain managers, mock infrastructure
- Event flow verification through subscriptions

### UI Tests
- Existing Qt tests can remain largely unchanged
- New tests for event-based UI updates

---

## Benefits

1. **Testability**: Domain logic testable without Qt
2. **Maintainability**: Clear boundaries, single responsibility
3. **Extensibility**: New features add use cases, not sprawl
4. **Debuggability**: Event bus makes data flow visible
5. **Team Scale**: Multiple developers can work on different layers

---

## Risks

| Risk | Mitigation |
|------|------------|
| Event bus adds complexity | Typed events, central definitions |
| Performance overhead | Batch events, avoid excessive publishing |

---

## Conclusion

The current architecture is unsustainable. The proposed redesign:

1. Separates **what** (domain) from **how** (infrastructure) from **when** (application)
2. Replaces callback spaghetti with typed events
3. Makes `MainWindow` a thin UI shell
5. Enables proper testing of business logic
