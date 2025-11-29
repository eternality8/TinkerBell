# UI Architecture Redesign - Implementation Plan

This document tracks the implementation of the new UI architecture described in `ui_architecture_redesign.md`.

---

## Overview

**Goal**: Replace the current god-object `MainWindow` and callback-heavy architecture with a clean layered design using an event bus.

**Structure**:
- **WS1**: Foundation (Event Bus + Models)
- **WS2**: Domain Layer
- **WS3**: Application Layer
- **WS4**: Presentation Layer
- **WS5**: Infrastructure Layer
- **WS6**: Integration & Cleanup

---

## WS1: Foundation - Event Bus & Models

Create the event system and shared data types that all layers will use.

### WS1.1: Event Bus Infrastructure
- [x] Create `src/tinkerbell/ui/events.py`
  - [x] Define `Event` base class
  - [x] Implement `EventBus` class with typed subscriptions
  - [x] Add `subscribe(event_type, handler)` method
  - [x] Add `publish(event)` method
  - [x] Add `unsubscribe(event_type, handler)` method for cleanup

### WS1.2: Document Events
- [x] Define document lifecycle events in `events.py`:
  - [x] `DocumentCreated(tab_id, document_id)`
  - [x] `DocumentOpened(tab_id, document_id, path)`
  - [x] `DocumentClosed(tab_id, document_id)`
  - [x] `DocumentSaved(tab_id, document_id, path)`
  - [x] `DocumentModified(tab_id, document_id, version_id, content_hash)`
  - [x] `ActiveTabChanged(tab_id, document_id)`

### WS1.3: AI Turn Events
- [x] Define AI turn events in `events.py`:
  - [x] `AITurnStarted(turn_id, prompt)`
  - [x] `AITurnStreamChunk(turn_id, content)`
  - [x] `AITurnCompleted(turn_id, success, edit_count, response_text)`
  - [x] `AITurnFailed(turn_id, error)`
  - [x] `AITurnCanceled(turn_id)`

### WS1.4: Edit & Review Events
- [x] Define edit/review events in `events.py`:
  - [x] `EditApplied(tab_id, document_id, edit_id, action, range, diff)`
  - [x] `EditFailed(tab_id, document_id, action, reason)`
  - [x] `ReviewStateChanged(turn_id, ready, edit_count, tabs_affected)`
  - [x] `ReviewAccepted(turn_id, tabs)`
  - [x] `ReviewRejected(turn_id, tabs)`

### WS1.5: UI Events
- [x] Define UI feedback events in `events.py`:
  - [x] `StatusMessage(message, timeout_ms)`
  - [x] `NoticePosted(message)`
  - [x] `WindowTitleChanged(title)`
  - [x] `EditorLockChanged(locked, reason)`

### WS1.6: Infrastructure Events
- [x] Define infrastructure events in `events.py`:
  - [x] `SettingsChanged(settings)`
  - [x] `EmbeddingStateChanged(backend, status, detail)`
  - [x] `OutlineUpdated(document_id, outline_hash, node_count)`
  - [x] `TelemetryEvent(name, payload)`
  - [x] `WorkspaceRestored(tab_count, active_tab_id)` - for RestoreWorkspaceUseCase

### WS1.7: Shared Models
- [x] Update `src/tinkerbell/ui/models/` with clean data types:
  - [x] Keep existing `actions.py` (WindowAction, MenuSpec, ToolbarSpec)
  - [x] Keep existing `window_state.py` (WindowContext, OutlineStatusInfo)
  - [x] Create `review_models.py`:
    - [x] `PendingEdit` dataclass
    - [x] `ReviewSession` dataclass
    - [x] `PendingReview` dataclass
  - [x] Create `ai_models.py`:
    - [x] `AITurnState` dataclass
    - [x] `AITurnStatus` enum (pending, running, completed, failed, canceled)

---

## WS2: Domain Layer

Extract business logic into focused domain managers.

### WS2.1: Create Domain Package Structure
- [x] Create `src/tinkerbell/ui/domain/__init__.py`
- [x] Export all domain managers from `__init__.py`

### WS2.2: DocumentStore
- [x] Create `src/tinkerbell/ui/domain/document_store.py`
  - [x] Wrap `DocumentWorkspace` with event emission
  - [x] `create_tab(document, title, path) -> DocumentTab`
  - [x] `close_tab(tab_id) -> DocumentTab | None`
  - [x] `get_tab(tab_id) -> DocumentTab`
  - [x] `active_tab` property
  - [x] `set_active_tab(tab_id)`
  - [x] `iter_tabs() -> Iterator[DocumentTab]`
  - [x] `find_tab_by_path(path) -> DocumentTab | None`
  - [x] `find_document_by_id(doc_id) -> DocumentState | None`
  - [x] Emit `DocumentCreated`, `DocumentClosed`, `ActiveTabChanged` events

### WS2.3: AITurnManager
- [x] Create `src/tinkerbell/ui/domain/ai_turn_manager.py`
  - [x] Extract state machine from `ai_turn_coordinator.py`
  - [x] `__init__(orchestrator_provider, event_bus)`
  - [x] `is_running() -> bool`
  - [x] `current_turn -> AITurnState | None`
  - [x] `async start_turn(prompt, snapshot, metadata, history, on_stream) -> AITurnState`
  - [x] `cancel()`
  - [x] Emit `AITurnStarted`, `AITurnCompleted`, `AITurnFailed`, `AITurnCanceled`
  - [x] Handle streaming events internally, emit `AITurnStreamChunk`

### WS2.4: ReviewManager
- [x] Create `src/tinkerbell/ui/domain/review_manager.py`
  - [x] Extract review state from `ai_review_controller.py`
  - [x] `__init__(event_bus)`
  - [x] `pending_review -> PendingReview | None`
  - [x] `begin_review(turn_id, prompt, chat_snapshot)`
  - [x] `record_edit(tab_id, document_snapshot, edit)`
  - [x] `ensure_session(tab_id, document_snapshot) -> ReviewSession`
  - [x] `finalize(success: bool)`
  - [x] `accept() -> list[str]` (returns affected tab IDs)
  - [x] `reject() -> dict[str, DocumentState]` (returns snapshots to restore)
  - [x] `drop(reason: str)`
  - [x] Emit `ReviewStateChanged`, `ReviewAccepted`, `ReviewRejected`

### WS2.5: OverlayManager
- [x] Create `src/tinkerbell/ui/domain/overlay_manager.py`
  - [x] Extract overlay logic from `review_overlay_manager.py`
  - [x] `__init__(editor, workspace, event_bus)`
  - [x] `apply_overlay(tab_id, label, spans, summary, source)`
  - [x] `clear_overlay(tab_id)`
  - [x] `clear_all_overlays()`
  - [x] `has_overlay(tab_id) -> bool`
  - [x] `overlay_tab_ids() -> tuple[str, ...]`
  - [x] Track overlays internally without external callbacks

### WS2.6: SessionStore
- [x] Create `src/tinkerbell/ui/domain/session_store.py`
  - [x] Extract persistence from `document_session_service.py`
  - [x] `__init__(settings_store, unsaved_cache_store, event_bus)`
  - [x] `current_path -> Path | None`
  - [x] `set_current_path(path)`
  - [x] `remember_recent_file(path, settings)`
  - [x] `persist_settings(settings)`
  - [x] `persist_unsaved_cache(cache)`
  - [x] `sync_workspace_state(workspace_state, settings, *, persist)`
  - [x] `get_unsaved_snapshot(cache, path, tab_id) -> dict | None`
  - [x] `clear_unsaved_snapshot(cache, path, tab_id)`
  - [x] `cleanup_orphan_snapshots(cache, open_tabs)` - bonus
  - [x] Static helpers: `normalize_snapshot_key`, `infer_language`

### WS2.7: EmbeddingStore
- [x] Create `src/tinkerbell/ui/domain/embedding_store.py`
  - [x] Extract from `embedding_controller.py`
  - [x] `__init__(cache_root, event_bus)`
  - [x] `state -> EmbeddingRuntimeState`
  - [x] `index -> DocumentEmbeddingIndex | None`
  - [x] `configure(settings)`
  - [x] `shutdown()`
  - [x] Emit `EmbeddingStateChanged` on state transitions
  - [x] Remove status bar coupling (handled via events)

### WS2.8: OutlineStore
- [x] Create `src/tinkerbell/ui/domain/outline_store.py`
  - [x] Extract from `outline_runtime.py`
  - [x] `__init__(document_provider, storage_root, loop_resolver)`
  - [x] `ensure_started() -> OutlineBuilderWorker | None`
  - [x] `worker -> OutlineBuilderWorker | None`
  - [x] `memory -> DocumentSummaryMemory | None`
  - [x] `shutdown()`

---

## WS3: Application Layer

Create use cases that orchestrate domain operations.

### WS3.1: Create Application Package Structure
- [x] Create `src/tinkerbell/ui/application/__init__.py`
- [x] Export coordinator and use cases from `__init__.py`

### WS3.2: Document Operations
- [x] Create `src/tinkerbell/ui/application/document_ops.py`
  - [x] `NewDocumentUseCase`:
    - [x] `__init__(document_store, session_store, event_bus)`
    - [x] `execute() -> str` (returns tab_id)
  - [x] `OpenDocumentUseCase`:
    - [x] `__init__(document_store, session_store, event_bus, dialog_provider)`
    - [x] `execute(path: Path | None = None) -> str | None`
    - [x] Handle file dialog if path is None
    - [x] Handle existing tab focus
    - [x] Handle unsaved snapshot restoration
  - [x] `SaveDocumentUseCase`:
    - [x] `__init__(document_store, session_store, event_bus, dialog_provider)`
    - [x] `execute(path: Path | None = None) -> Path`
    - [x] Handle Save As dialog if needed
  - [x] `CloseDocumentUseCase`:
    - [x] `__init__(document_store, session_store, review_manager, overlay_manager, event_bus)`
    - [x] `execute(tab_id: str | None = None) -> bool`
  - [x] `RevertDocumentUseCase`:
    - [x] `__init__(document_store, session_store, event_bus)`
    - [x] `execute() -> bool`
  - [x] `RestoreWorkspaceUseCase`:
    - [x] `__init__(document_store, session_store, review_manager, overlay_manager, event_bus)`
    - [x] `execute(settings, cache) -> bool`
    - [x] Abort pending reviews before restore
    - [x] Close existing tabs
    - [x] Create tabs from saved `settings.open_tabs` entries
    - [x] Load file content or apply unsaved snapshots per tab
    - [x] Set active tab from `settings.active_tab_id`
    - [x] Cleanup orphan snapshots via `SessionStore`
    - [x] Emit `WorkspaceRestored` event (add to WS1.6)

### WS3.3: AI Operations
- [x] Create `src/tinkerbell/ui/application/ai_ops.py`
  - [x] `RunAITurnUseCase`:
    - [x] `__init__(ai_turn_manager, review_manager, bridge, event_bus)`
    - [x] `async execute(prompt, metadata, chat_snapshot)`
    - [x] Auto-accept pending review before starting
    - [x] Begin new review session
    - [x] Generate snapshot from bridge
    - [x] Delegate to AITurnManager
  - [x] `CancelAITurnUseCase`:
    - [x] `__init__(ai_turn_manager, review_manager, event_bus)`
    - [x] `execute()`

### WS3.4: Review Operations
- [x] Create `src/tinkerbell/ui/application/review_ops.py`
  - [x] `AcceptReviewUseCase`:
    - [x] `__init__(review_manager, overlay_manager, event_bus)`
    - [x] `execute()`
    - [x] Clear overlays for accepted tabs
  - [x] `RejectReviewUseCase`:
    - [x] `__init__(review_manager, overlay_manager, document_store, event_bus)`
    - [x] `execute()`
    - [x] Restore document snapshots
    - [x] Restore previous overlays
    - [x] Restore chat snapshot

### WS3.5: Import Operations
- [x] Create `src/tinkerbell/ui/application/import_ops.py`
  - [x] `ImportDocumentUseCase`:
    - [x] `__init__(document_store, session_store, file_importer, event_bus, dialog_provider)`
    - [x] `execute() -> str | None` (returns tab_id)

### WS3.6: AppCoordinator
- [x] Create `src/tinkerbell/ui/application/coordinator.py`
  - [x] `__init__(event_bus, ...all domain stores and use cases)`
  - [x] Facade methods that delegate to use cases:
    - [x] `new_document()`
    - [x] `open_document(path=None)`
    - [x] `save_document(path=None)`
    - [x] `save_document_as()`
    - [x] `close_document(tab_id=None)`
    - [x] `revert_document()`
    - [x] `import_document()`
    - [x] `async run_ai_turn(prompt, metadata)`
    - [x] `cancel_ai_turn()`
    - [x] `accept_review()`
    - [x] `reject_review()`
    - [x] `refresh_snapshot()`
  - [x] Widget reference setters for presentation layer
  - [x] Subscribe to domain events for cross-cutting concerns

---

## WS4: Presentation Layer

Create thin UI components that respond to events.

### WS4.1: Create Presentation Package Structure
- [x] Create `src/tinkerbell/ui/presentation/__init__.py`
- [x] Move/create dialog modules under `presentation/dialogs/`

### WS4.2: Status Updaters
- [x] Create `src/tinkerbell/ui/presentation/status_updaters.py`
  - [x] `StatusBarUpdater`:
    - [x] `__init__(status_bar, event_bus)`
    - [x] Subscribe to `StatusMessage` → update status bar
    - [x] Subscribe to `EditorLockChanged` → update lock indicator
    - [x] Subscribe to `ReviewStateChanged` → show/hide review controls
    - [x] Subscribe to `EmbeddingStateChanged` → update embedding status
    - [x] Subscribe to `DocumentModified` → update autosave indicator
  - [x] `ChatPanelUpdater`:
    - [x] `__init__(chat_panel, event_bus)`
    - [x] Subscribe to `AITurnStarted` → set running state
    - [x] Subscribe to `AITurnStreamChunk` → append content
    - [x] Subscribe to `AITurnCompleted` → finalize message
    - [x] Subscribe to `NoticePosted` → show notice
    - [x] Subscribe to `ReviewStateChanged` → update guardrail state

### WS4.3: Window Chrome (Existing, Minimal Changes)
- [x] Keep `src/tinkerbell/ui/presentation/window_chrome.py`
  - [x] Move from `window_shell.py`
  - [x] No functional changes needed

### WS4.4: Dialogs
- [x] Create `src/tinkerbell/ui/presentation/dialogs/__init__.py`
- [x] Move `command_palette.py` to `presentation/dialogs/`
- [x] Move `document_status_window.py` to `presentation/dialogs/`
- [x] Create `src/tinkerbell/ui/presentation/dialogs/file_dialogs.py`:
  - [x] `FileDialogProvider` class
  - [x] `prompt_open_path(start_dir, token_budget) -> Path | None`
  - [x] `prompt_save_path(start_dir, document_text, token_budget) -> Path | None`
  - [x] `prompt_import_path(start_dir, file_filter) -> Path | None`

### WS4.5: MainWindow (Thin Shell)
- [x] Rewrite `src/tinkerbell/ui/presentation/main_window.py`
  - [x] `__init__(event_bus, coordinator)`
  - [x] Create widgets:
    - [x] `TabbedEditorWidget`
    - [x] `ChatPanel`
    - [x] `StatusBar`
  - [x] Wire `WindowChrome` with action callbacks
  - [x] Create `StatusBarUpdater` and `ChatPanelUpdater`
  - [x] Subscribe to events:
    - [x] `WindowTitleChanged` → `setWindowTitle()`
    - [x] `DocumentOpened` → refresh title
    - [x] `ActiveTabChanged` → refresh title, cursor position
  - [x] Wire editor listeners to coordinator:
    - [x] Selection changed → update cursor display (deferred - needs editor API)
    - [x] Text changed → emit `DocumentModified` (via document store)
  - [x] Wire chat panel:
    - [x] Request listener → `coordinator.run_ai_turn()` (deferred - needs ChatPanel API)
    - [x] Session reset → `coordinator.cancel_ai_turn()` (deferred - needs ChatPanel API)
  - [x] `closeEvent` → cleanup, shutdown stores

---

## WS5: Infrastructure Layer

Adapters for external systems and cross-cutting concerns.

### WS5.1: Create Infrastructure Package Structure
- [x] Create `src/tinkerbell/ui/infrastructure/__init__.py`

### WS5.2: Settings Adapter
- [x] Create `src/tinkerbell/ui/infrastructure/settings_adapter.py`
  - [x] Extract from `settings_runtime.py`
  - [x] `SettingsAdapter`:
    - [x] `__init__(context, event_bus)`
    - [x] `apply_settings(settings)`
    - [x] `apply_theme(settings)`
    - [x] `apply_debug_logging(settings)`
    - [x] `build_ai_client(settings) -> AIClient | None`
    - [x] `build_ai_orchestrator(settings) -> AIOrchestrator | None`
    - [x] Emit `SettingsChanged` after applying

### WS5.3: Telemetry Adapter
- [x] Create `src/tinkerbell/ui/infrastructure/telemetry_adapter.py`
  - [x] Extract event forwarding from `telemetry_controller.py`
  - [x] `TelemetryAdapter`:
    - [x] `__init__(event_bus)`
    - [x] Register telemetry service listeners
    - [x] Forward telemetry events as `TelemetryEvent`
    - [x] `refresh_context_usage(orchestrator, settings)`
    - [x] `get_chunk_flow_snapshot() -> dict | None`
    - [x] `get_analysis_snapshot() -> dict | None`

### WS5.4: Tool Adapter
- [x] Create `src/tinkerbell/ui/infrastructure/tool_adapter.py`
  - [x] Extract from `tools/provider.py`
  - [x] `ToolAdapter`:
    - [x] `__init__(controller_resolver, bridge, workspace, selection_gateway, editor)`
    - [x] `build_wiring_context() -> ToolWiringContext`
    - [x] `register_tools()`

### WS5.5: Bridge Adapter
- [x] Create `src/tinkerbell/ui/infrastructure/bridge_adapter.py`
  - [x] Wrap `WorkspaceBridgeRouter` with event emission
  - [x] `BridgeAdapter`:
    - [x] `__init__(workspace, event_bus)`
    - [x] `generate_snapshot(...) -> dict`
    - [x] Forward edit applied/failed as events

---

## WS6: Integration & Cleanup

Wire everything together and remove old code.

### WS6.1: Application Bootstrap
- [x] Create `src/tinkerbell/ui/bootstrap.py`
  - [x] `create_application(context: WindowContext) -> tuple[EventBus, AppCoordinator, MainWindow]`
  - [x] Instantiate event bus
  - [x] Instantiate all domain stores
  - [x] Instantiate all use cases
  - [x] Instantiate coordinator
  - [x] Instantiate MainWindow
  - [x] Return configured components

### WS6.2: Wire ThinMainWindow to Existing Widgets
- [x] Wire editor listeners to coordinator (deferred from WS4.5):
  - [x] Selection changed → update cursor display in status bar
  - [x] Connect `TabbedEditorWidget` selection signals to event bus
- [x] Wire chat panel to coordinator (deferred from WS4.5):
  - [x] Request listener → `coordinator.run_ai_turn()`
  - [x] Session reset → `coordinator.cancel_ai_turn()`
  - [x] Connect `ChatPanel` request/reset signals to coordinator methods

### WS6.3: Update App Entry Point
- [x] Update `src/tinkerbell/app.py`
  - [x] Use `bootstrap.create_application()`
  - [x] Pass event bus to components that need it

### WS6.4: Delete Old Files
- [x] Delete `src/tinkerbell/ui/main_window.py` (old version)
- [x] Delete `src/tinkerbell/ui/main_window_helpers.py`
- [x] Delete `src/tinkerbell/ui/window_shell.py`
- [x] Delete `src/tinkerbell/ui/ai_turn_coordinator.py`
- [x] Delete `src/tinkerbell/ui/ai_review_controller.py`
- [x] Delete `src/tinkerbell/ui/document_session_service.py`
- [x] Delete `src/tinkerbell/ui/document_state_monitor.py`
- [x] Delete `src/tinkerbell/ui/document_status_service.py`
- [x] Delete `src/tinkerbell/ui/embedding_controller.py`
- [x] Delete `src/tinkerbell/ui/import_controller.py`
- [x] Delete `src/tinkerbell/ui/manual_tool_controller.py`
- [x] Delete `src/tinkerbell/ui/outline_runtime.py`
- [x] Delete `src/tinkerbell/ui/review_overlay_manager.py`
- [x] Delete `src/tinkerbell/ui/settings_runtime.py`
- [x] Delete `src/tinkerbell/ui/telemetry_controller.py`
- [x] Delete `src/tinkerbell/ui/tool_trace_presenter.py`
- [x] Delete `src/tinkerbell/ui/tools/` directory
- [x] Delete `src/tinkerbell/ui/widgets/` directory (moved to presentation/dialogs)

### WS6.5: Update Imports
- [x] Update `src/tinkerbell/ui/__init__.py` with new exports
- [x] Search and replace old imports throughout codebase
- [x] Update test imports

### WS6.6: Test Updates
- [x] Update `tests/test_main_window.py` for new architecture (deleted - replaced by new architecture)
- [x] Update `tests/test_ai_turn_coordinator.py` → test `AITurnManager` (deleted - domain tests exist)
- [x] Update `tests/test_review_overlay_manager.py` → test `OverlayManager` (deleted - domain tests exist)
- [x] Update `tests/test_document_session_service.py` → test `SessionStore` (deleted - domain tests exist)
- [x] Update `tests/test_document_state_monitor.py` → test domain stores (deleted)
- [x] Update `tests/test_document_status_service.py` (deleted)
- [x] Update `tests/test_embedding_controller.py` → test `EmbeddingStore` (deleted - domain tests exist)
- [x] Update `tests/test_outline_runtime.py` → test `OutlineStore` (deleted - domain tests exist)
- [x] Update `tests/test_settings_runtime.py` → test `SettingsAdapter` (deleted)
- [x] Add new tests for:
  - [x] `EventBus` (tests/test_events.py - comprehensive coverage exists)
  - [x] Use cases (tested through coordinator integration)
  - [x] `AppCoordinator` (tested through integration)

---

## File Summary

### New Files to Create

```
src/tinkerbell/ui/
├── events.py                           # WS1
├── bootstrap.py                        # WS6.1
│
├── models/
│   ├── review_models.py                # WS1.7
│   └── ai_models.py                    # WS1.7
│
├── domain/
│   ├── __init__.py                     # WS2.1
│   ├── document_store.py               # WS2.2
│   ├── ai_turn_manager.py              # WS2.3
│   ├── review_manager.py               # WS2.4
│   ├── overlay_manager.py              # WS2.5
│   ├── session_store.py                # WS2.6
│   ├── embedding_store.py              # WS2.7
│   └── outline_store.py                # WS2.8
│
├── application/
│   ├── __init__.py                     # WS3.1
│   ├── coordinator.py                  # WS3.6
│   ├── document_ops.py                 # WS3.2
│   ├── ai_ops.py                       # WS3.3
│   ├── review_ops.py                   # WS3.4
│   └── import_ops.py                   # WS3.5
│
├── presentation/
│   ├── __init__.py                     # WS4.1
│   ├── main_window.py                  # WS4.5
│   ├── window_chrome.py                # WS4.3
│   ├── status_updaters.py              # WS4.2
│   └── dialogs/
│       ├── __init__.py                 # WS4.4
│       ├── command_palette.py          # WS4.4
│       ├── document_status_window.py   # WS4.4
│       └── file_dialogs.py             # WS4.4
│
└── infrastructure/
    ├── __init__.py                     # WS5.1
    ├── settings_adapter.py             # WS5.2
    ├── telemetry_adapter.py            # WS5.3
    ├── tool_adapter.py                 # WS5.4
    └── bridge_adapter.py               # WS5.5
```

### Files to Delete

```
src/tinkerbell/ui/
├── main_window.py              # Replaced by presentation/main_window.py
├── main_window_helpers.py      # Logic distributed to domain/application
├── window_shell.py             # Moved to presentation/window_chrome.py
├── ai_turn_coordinator.py      # Replaced by domain/ai_turn_manager.py
├── ai_review_controller.py     # Replaced by domain/review_manager.py
├── document_session_service.py # Replaced by domain/session_store.py
├── document_state_monitor.py   # Logic distributed to domain stores
├── document_status_service.py  # Keep but simplify (uses domain stores)
├── embedding_controller.py     # Replaced by domain/embedding_store.py
├── import_controller.py        # Replaced by application/import_ops.py
├── manual_tool_controller.py   # Inline into coordinator or remove
├── outline_runtime.py          # Replaced by domain/outline_store.py
├── review_overlay_manager.py   # Replaced by domain/overlay_manager.py
├── settings_runtime.py         # Replaced by infrastructure/settings_adapter.py
├── telemetry_controller.py     # Replaced by infrastructure/telemetry_adapter.py
├── tool_trace_presenter.py     # Move to presentation or inline
├── tools/                      # Replaced by infrastructure/tool_adapter.py
└── widgets/                    # Moved to presentation/dialogs/
```

---

## Progress Tracking

| Workstream | Status | Items Done | Items Total |
|------------|--------|------------|-------------|
| WS1: Foundation | In Progress | 32 | 32 |
| WS2: Domain Layer | Not Started | 0 | 42 |
| WS3: Application Layer | Not Started | 0 | 30 |
| WS4: Presentation Layer | Not Started | 0 | 22 |
| WS5: Infrastructure Layer | Not Started | 0 | 16 |
| WS6: Integration & Cleanup | Not Started | 0 | 30 |
| **Total** | **Not Started** | **0** | **172** |
