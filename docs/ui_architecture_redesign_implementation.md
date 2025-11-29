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
- [ ] Create `src/tinkerbell/ui/events.py`
  - [ ] Define `Event` base class
  - [ ] Implement `EventBus` class with typed subscriptions
  - [ ] Add `subscribe(event_type, handler)` method
  - [ ] Add `publish(event)` method
  - [ ] Add `unsubscribe(event_type, handler)` method for cleanup

### WS1.2: Document Events
- [ ] Define document lifecycle events in `events.py`:
  - [ ] `DocumentCreated(tab_id, document_id)`
  - [ ] `DocumentOpened(tab_id, document_id, path)`
  - [ ] `DocumentClosed(tab_id, document_id)`
  - [ ] `DocumentSaved(tab_id, document_id, path)`
  - [ ] `DocumentModified(tab_id, document_id, version_id, content_hash)`
  - [ ] `ActiveTabChanged(tab_id, document_id)`

### WS1.3: AI Turn Events
- [ ] Define AI turn events in `events.py`:
  - [ ] `AITurnStarted(turn_id, prompt)`
  - [ ] `AITurnStreamChunk(turn_id, content)`
  - [ ] `AITurnCompleted(turn_id, success, edit_count, response_text)`
  - [ ] `AITurnFailed(turn_id, error)`
  - [ ] `AITurnCanceled(turn_id)`

### WS1.4: Edit & Review Events
- [ ] Define edit/review events in `events.py`:
  - [ ] `EditApplied(tab_id, document_id, edit_id, action, range, diff)`
  - [ ] `EditFailed(tab_id, document_id, action, reason)`
  - [ ] `ReviewStateChanged(turn_id, ready, edit_count, tabs_affected)`
  - [ ] `ReviewAccepted(turn_id, tabs)`
  - [ ] `ReviewRejected(turn_id, tabs)`

### WS1.5: UI Events
- [ ] Define UI feedback events in `events.py`:
  - [ ] `StatusMessage(message, timeout_ms)`
  - [ ] `NoticePosted(message)`
  - [ ] `WindowTitleChanged(title)`
  - [ ] `EditorLockChanged(locked, reason)`

### WS1.6: Infrastructure Events
- [ ] Define infrastructure events in `events.py`:
  - [ ] `SettingsChanged(settings)`
  - [ ] `EmbeddingStateChanged(backend, status, detail)`
  - [ ] `OutlineUpdated(document_id, outline_hash, node_count)`
  - [ ] `TelemetryEvent(name, payload)`

### WS1.7: Shared Models
- [ ] Update `src/tinkerbell/ui/models/` with clean data types:
  - [ ] Keep existing `actions.py` (WindowAction, MenuSpec, ToolbarSpec)
  - [ ] Keep existing `window_state.py` (WindowContext, OutlineStatusInfo)
  - [ ] Create `review_models.py`:
    - [ ] `PendingEdit` dataclass
    - [ ] `ReviewSession` dataclass
    - [ ] `PendingReview` dataclass
  - [ ] Create `ai_models.py`:
    - [ ] `AITurnState` dataclass
    - [ ] `AITurnStatus` enum (pending, running, completed, failed, canceled)

---

## WS2: Domain Layer

Extract business logic into focused domain managers.

### WS2.1: Create Domain Package Structure
- [ ] Create `src/tinkerbell/ui/domain/__init__.py`
- [ ] Export all domain managers from `__init__.py`

### WS2.2: DocumentStore
- [ ] Create `src/tinkerbell/ui/domain/document_store.py`
  - [ ] Wrap `DocumentWorkspace` with event emission
  - [ ] `create_tab(document, title, path) -> DocumentTab`
  - [ ] `close_tab(tab_id) -> DocumentTab | None`
  - [ ] `get_tab(tab_id) -> DocumentTab`
  - [ ] `active_tab` property
  - [ ] `set_active_tab(tab_id)`
  - [ ] `iter_tabs() -> Iterator[DocumentTab]`
  - [ ] `find_tab_by_path(path) -> DocumentTab | None`
  - [ ] `find_document_by_id(doc_id) -> DocumentState | None`
  - [ ] Emit `DocumentCreated`, `DocumentClosed`, `ActiveTabChanged` events

### WS2.3: AITurnManager
- [ ] Create `src/tinkerbell/ui/domain/ai_turn_manager.py`
  - [ ] Extract state machine from `ai_turn_coordinator.py`
  - [ ] `__init__(orchestrator_provider, event_bus)`
  - [ ] `is_running() -> bool`
  - [ ] `current_turn -> AITurnState | None`
  - [ ] `async start_turn(prompt, snapshot, metadata, history, on_stream) -> AITurnState`
  - [ ] `cancel()`
  - [ ] Emit `AITurnStarted`, `AITurnCompleted`, `AITurnFailed`, `AITurnCanceled`
  - [ ] Handle streaming events internally, emit `AITurnStreamChunk`

### WS2.4: ReviewManager
- [ ] Create `src/tinkerbell/ui/domain/review_manager.py`
  - [ ] Extract review state from `ai_review_controller.py`
  - [ ] `__init__(event_bus)`
  - [ ] `pending_review -> PendingReview | None`
  - [ ] `begin_review(turn_id, prompt, chat_snapshot)`
  - [ ] `record_edit(tab_id, document_snapshot, edit)`
  - [ ] `ensure_session(tab_id, document_snapshot) -> ReviewSession`
  - [ ] `finalize(success: bool)`
  - [ ] `accept() -> list[str]` (returns affected tab IDs)
  - [ ] `reject() -> dict[str, DocumentState]` (returns snapshots to restore)
  - [ ] `drop(reason: str)`
  - [ ] Emit `ReviewStateChanged`, `ReviewAccepted`, `ReviewRejected`

### WS2.5: OverlayManager
- [ ] Create `src/tinkerbell/ui/domain/overlay_manager.py`
  - [ ] Extract overlay logic from `review_overlay_manager.py`
  - [ ] `__init__(editor, workspace, event_bus)`
  - [ ] `apply_overlay(tab_id, label, spans, summary, source)`
  - [ ] `clear_overlay(tab_id)`
  - [ ] `clear_all_overlays()`
  - [ ] `has_overlay(tab_id) -> bool`
  - [ ] `overlay_tab_ids() -> tuple[str, ...]`
  - [ ] Track overlays internally without external callbacks

### WS2.6: SessionStore
- [ ] Create `src/tinkerbell/ui/domain/session_store.py`
  - [ ] Extract persistence from `document_session_service.py`
  - [ ] `__init__(settings_store, unsaved_cache_store, event_bus)`
  - [ ] `current_path -> Path | None`
  - [ ] `set_current_path(path)`
  - [ ] `remember_recent_file(path)`
  - [ ] `persist_settings(settings)`
  - [ ] `persist_unsaved_cache(cache)`
  - [ ] `sync_workspace_state(workspace)`
  - [ ] `restore_workspace(workspace, editor) -> bool`
  - [ ] `get_unsaved_snapshot(path, tab_id) -> dict | None`
  - [ ] `clear_unsaved_snapshot(path, tab_id)`

### WS2.7: EmbeddingStore
- [ ] Create `src/tinkerbell/ui/domain/embedding_store.py`
  - [ ] Extract from `embedding_controller.py`
  - [ ] `__init__(cache_root, event_bus)`
  - [ ] `state -> EmbeddingRuntimeState`
  - [ ] `index -> DocumentEmbeddingIndex | None`
  - [ ] `configure(settings)`
  - [ ] `shutdown()`
  - [ ] Emit `EmbeddingStateChanged` on state transitions
  - [ ] Remove status bar coupling (handled via events)

### WS2.8: OutlineStore
- [ ] Create `src/tinkerbell/ui/domain/outline_store.py`
  - [ ] Extract from `outline_runtime.py`
  - [ ] `__init__(document_provider, storage_root, loop_resolver)`
  - [ ] `ensure_started() -> OutlineBuilderWorker | None`
  - [ ] `worker -> OutlineBuilderWorker | None`
  - [ ] `memory -> DocumentSummaryMemory | None`
  - [ ] `shutdown()`

---

## WS3: Application Layer

Create use cases that orchestrate domain operations.

### WS3.1: Create Application Package Structure
- [ ] Create `src/tinkerbell/ui/application/__init__.py`
- [ ] Export coordinator and use cases from `__init__.py`

### WS3.2: Document Operations
- [ ] Create `src/tinkerbell/ui/application/document_ops.py`
  - [ ] `NewDocumentUseCase`:
    - [ ] `__init__(document_store, session_store, event_bus)`
    - [ ] `execute() -> str` (returns tab_id)
  - [ ] `OpenDocumentUseCase`:
    - [ ] `__init__(document_store, session_store, event_bus, dialog_provider)`
    - [ ] `execute(path: Path | None = None) -> str | None`
    - [ ] Handle file dialog if path is None
    - [ ] Handle existing tab focus
    - [ ] Handle unsaved snapshot restoration
  - [ ] `SaveDocumentUseCase`:
    - [ ] `__init__(document_store, session_store, event_bus, dialog_provider)`
    - [ ] `execute(path: Path | None = None) -> Path`
    - [ ] Handle Save As dialog if needed
  - [ ] `CloseDocumentUseCase`:
    - [ ] `__init__(document_store, session_store, review_manager, overlay_manager, event_bus)`
    - [ ] `execute(tab_id: str | None = None) -> bool`
  - [ ] `RevertDocumentUseCase`:
    - [ ] `__init__(document_store, session_store, event_bus)`
    - [ ] `execute() -> bool`

### WS3.3: AI Operations
- [ ] Create `src/tinkerbell/ui/application/ai_ops.py`
  - [ ] `RunAITurnUseCase`:
    - [ ] `__init__(ai_turn_manager, review_manager, bridge, event_bus)`
    - [ ] `async execute(prompt, metadata, chat_snapshot)`
    - [ ] Auto-accept pending review before starting
    - [ ] Begin new review session
    - [ ] Generate snapshot from bridge
    - [ ] Delegate to AITurnManager
  - [ ] `CancelAITurnUseCase`:
    - [ ] `__init__(ai_turn_manager, review_manager, event_bus)`
    - [ ] `execute()`

### WS3.4: Review Operations
- [ ] Create `src/tinkerbell/ui/application/review_ops.py`
  - [ ] `AcceptReviewUseCase`:
    - [ ] `__init__(review_manager, overlay_manager, event_bus)`
    - [ ] `execute()`
    - [ ] Clear overlays for accepted tabs
  - [ ] `RejectReviewUseCase`:
    - [ ] `__init__(review_manager, overlay_manager, document_store, event_bus)`
    - [ ] `execute()`
    - [ ] Restore document snapshots
    - [ ] Restore previous overlays
    - [ ] Restore chat snapshot

### WS3.5: Import Operations
- [ ] Create `src/tinkerbell/ui/application/import_ops.py`
  - [ ] `ImportDocumentUseCase`:
    - [ ] `__init__(document_store, session_store, file_importer, event_bus, dialog_provider)`
    - [ ] `execute() -> str | None` (returns tab_id)

### WS3.6: AppCoordinator
- [ ] Create `src/tinkerbell/ui/application/coordinator.py`
  - [ ] `__init__(event_bus, ...all domain stores and use cases)`
  - [ ] Facade methods that delegate to use cases:
    - [ ] `new_document()`
    - [ ] `open_document(path=None)`
    - [ ] `save_document(path=None)`
    - [ ] `save_document_as()`
    - [ ] `close_document(tab_id=None)`
    - [ ] `revert_document()`
    - [ ] `import_document()`
    - [ ] `async run_ai_turn(prompt, metadata)`
    - [ ] `cancel_ai_turn()`
    - [ ] `accept_review()`
    - [ ] `reject_review()`
    - [ ] `refresh_snapshot()`
  - [ ] Widget reference setters for presentation layer
  - [ ] Subscribe to domain events for cross-cutting concerns

---

## WS4: Presentation Layer

Create thin UI components that respond to events.

### WS4.1: Create Presentation Package Structure
- [ ] Create `src/tinkerbell/ui/presentation/__init__.py`
- [ ] Move/create dialog modules under `presentation/dialogs/`

### WS4.2: Status Updaters
- [ ] Create `src/tinkerbell/ui/presentation/status_updaters.py`
  - [ ] `StatusBarUpdater`:
    - [ ] `__init__(status_bar, event_bus)`
    - [ ] Subscribe to `StatusMessage` → update status bar
    - [ ] Subscribe to `EditorLockChanged` → update lock indicator
    - [ ] Subscribe to `ReviewStateChanged` → show/hide review controls
    - [ ] Subscribe to `EmbeddingStateChanged` → update embedding status
    - [ ] Subscribe to `DocumentModified` → update autosave indicator
  - [ ] `ChatPanelUpdater`:
    - [ ] `__init__(chat_panel, event_bus)`
    - [ ] Subscribe to `AITurnStarted` → set running state
    - [ ] Subscribe to `AITurnStreamChunk` → append content
    - [ ] Subscribe to `AITurnCompleted` → finalize message
    - [ ] Subscribe to `NoticePosted` → show notice
    - [ ] Subscribe to `ReviewStateChanged` → update guardrail state

### WS4.3: Window Chrome (Existing, Minimal Changes)
- [ ] Keep `src/tinkerbell/ui/presentation/window_chrome.py`
  - [ ] Move from `window_shell.py`
  - [ ] No functional changes needed

### WS4.4: Dialogs
- [ ] Create `src/tinkerbell/ui/presentation/dialogs/__init__.py`
- [ ] Move `command_palette.py` to `presentation/dialogs/`
- [ ] Move `document_status_window.py` to `presentation/dialogs/`
- [ ] Create `src/tinkerbell/ui/presentation/dialogs/file_dialogs.py`:
  - [ ] `FileDialogProvider` class
  - [ ] `prompt_open_path(start_dir, token_budget) -> Path | None`
  - [ ] `prompt_save_path(start_dir, document_text, token_budget) -> Path | None`
  - [ ] `prompt_import_path(start_dir, file_filter) -> Path | None`

### WS4.5: MainWindow (Thin Shell)
- [ ] Rewrite `src/tinkerbell/ui/presentation/main_window.py`
  - [ ] `__init__(event_bus, coordinator)`
  - [ ] Create widgets:
    - [ ] `TabbedEditorWidget`
    - [ ] `ChatPanel`
    - [ ] `StatusBar`
  - [ ] Wire `WindowChrome` with action callbacks
  - [ ] Create `StatusBarUpdater` and `ChatPanelUpdater`
  - [ ] Subscribe to events:
    - [ ] `WindowTitleChanged` → `setWindowTitle()`
    - [ ] `DocumentOpened` → refresh title
    - [ ] `ActiveTabChanged` → refresh title, cursor position
  - [ ] Wire editor listeners to coordinator:
    - [ ] Selection changed → update cursor display
    - [ ] Text changed → emit `DocumentModified` (via document store)
  - [ ] Wire chat panel:
    - [ ] Request listener → `coordinator.run_ai_turn()`
    - [ ] Session reset → `coordinator.cancel_ai_turn()`
  - [ ] `closeEvent` → cleanup, shutdown stores

---

## WS5: Infrastructure Layer

Adapters for external systems and cross-cutting concerns.

### WS5.1: Create Infrastructure Package Structure
- [ ] Create `src/tinkerbell/ui/infrastructure/__init__.py`

### WS5.2: Settings Adapter
- [ ] Create `src/tinkerbell/ui/infrastructure/settings_adapter.py`
  - [ ] Extract from `settings_runtime.py`
  - [ ] `SettingsAdapter`:
    - [ ] `__init__(context, event_bus)`
    - [ ] `apply_settings(settings)`
    - [ ] `apply_theme(settings)`
    - [ ] `apply_debug_logging(settings)`
    - [ ] `build_ai_client(settings) -> AIClient | None`
    - [ ] `build_ai_orchestrator(settings) -> AIOrchestrator | None`
    - [ ] Emit `SettingsChanged` after applying

### WS5.3: Telemetry Adapter
- [ ] Create `src/tinkerbell/ui/infrastructure/telemetry_adapter.py`
  - [ ] Extract event forwarding from `telemetry_controller.py`
  - [ ] `TelemetryAdapter`:
    - [ ] `__init__(event_bus)`
    - [ ] Register telemetry service listeners
    - [ ] Forward telemetry events as `TelemetryEvent`
    - [ ] `refresh_context_usage(orchestrator, settings)`
    - [ ] `get_chunk_flow_snapshot() -> dict | None`
    - [ ] `get_analysis_snapshot() -> dict | None`

### WS5.4: Tool Adapter
- [ ] Create `src/tinkerbell/ui/infrastructure/tool_adapter.py`
  - [ ] Extract from `tools/provider.py`
  - [ ] `ToolAdapter`:
    - [ ] `__init__(controller_resolver, bridge, workspace, selection_gateway, editor)`
    - [ ] `build_wiring_context() -> ToolWiringContext`
    - [ ] `register_tools()`

### WS5.5: Bridge Adapter
- [ ] Create `src/tinkerbell/ui/infrastructure/bridge_adapter.py`
  - [ ] Wrap `WorkspaceBridgeRouter` with event emission
  - [ ] `BridgeAdapter`:
    - [ ] `__init__(workspace, event_bus)`
    - [ ] `generate_snapshot(...) -> dict`
    - [ ] Forward edit applied/failed as events

---

## WS6: Integration & Cleanup

Wire everything together and remove old code.

### WS6.1: Application Bootstrap
- [ ] Create `src/tinkerbell/ui/bootstrap.py`
  - [ ] `create_application(context: WindowContext) -> tuple[EventBus, AppCoordinator, MainWindow]`
  - [ ] Instantiate event bus
  - [ ] Instantiate all domain stores
  - [ ] Instantiate all use cases
  - [ ] Instantiate coordinator
  - [ ] Instantiate MainWindow
  - [ ] Return configured components

### WS6.2: Update App Entry Point
- [ ] Update `src/tinkerbell/app.py`
  - [ ] Use `bootstrap.create_application()`
  - [ ] Pass event bus to components that need it

### WS6.3: Delete Old Files
- [ ] Delete `src/tinkerbell/ui/main_window.py` (old version)
- [ ] Delete `src/tinkerbell/ui/main_window_helpers.py`
- [ ] Delete `src/tinkerbell/ui/window_shell.py`
- [ ] Delete `src/tinkerbell/ui/ai_turn_coordinator.py`
- [ ] Delete `src/tinkerbell/ui/ai_review_controller.py`
- [ ] Delete `src/tinkerbell/ui/document_session_service.py`
- [ ] Delete `src/tinkerbell/ui/document_state_monitor.py`
- [ ] Delete `src/tinkerbell/ui/document_status_service.py`
- [ ] Delete `src/tinkerbell/ui/embedding_controller.py`
- [ ] Delete `src/tinkerbell/ui/import_controller.py`
- [ ] Delete `src/tinkerbell/ui/manual_tool_controller.py`
- [ ] Delete `src/tinkerbell/ui/outline_runtime.py`
- [ ] Delete `src/tinkerbell/ui/review_overlay_manager.py`
- [ ] Delete `src/tinkerbell/ui/settings_runtime.py`
- [ ] Delete `src/tinkerbell/ui/telemetry_controller.py`
- [ ] Delete `src/tinkerbell/ui/tool_trace_presenter.py`
- [ ] Delete `src/tinkerbell/ui/tools/` directory
- [ ] Delete `src/tinkerbell/ui/widgets/` directory (moved to presentation/dialogs)

### WS6.4: Update Imports
- [ ] Update `src/tinkerbell/ui/__init__.py` with new exports
- [ ] Search and replace old imports throughout codebase
- [ ] Update test imports

### WS6.5: Test Updates
- [ ] Update `tests/test_main_window.py` for new architecture
- [ ] Update `tests/test_ai_turn_coordinator.py` → test `AITurnManager`
- [ ] Update `tests/test_review_overlay_manager.py` → test `OverlayManager`
- [ ] Update `tests/test_document_session_service.py` → test `SessionStore`
- [ ] Update `tests/test_document_state_monitor.py` → test domain stores
- [ ] Update `tests/test_document_status_service.py`
- [ ] Update `tests/test_embedding_controller.py` → test `EmbeddingStore`
- [ ] Update `tests/test_outline_runtime.py` → test `OutlineStore`
- [ ] Update `tests/test_settings_runtime.py` → test `SettingsAdapter`
- [ ] Add new tests for:
  - [ ] `EventBus`
  - [ ] Use cases
  - [ ] `AppCoordinator`

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
| WS1: Foundation | Not Started | 0 | 32 |
| WS2: Domain Layer | Not Started | 0 | 42 |
| WS3: Application Layer | Not Started | 0 | 30 |
| WS4: Presentation Layer | Not Started | 0 | 22 |
| WS5: Infrastructure Layer | Not Started | 0 | 16 |
| WS6: Integration & Cleanup | Not Started | 0 | 30 |
| **Total** | **Not Started** | **0** | **172** |
