# Cleanup v2 Implementation Plan

This document expands `cleanup_v2.md` into actionable phases with granular tasks, ownership hints, and progress checkpoints.


## Phase A – Core Infrastructure (Tool Providers & Outline Runtime)
### A1. Tool Provider Module
1. [x] Create `src/tinkerbell/ui/tools/provider.py` with:
   - Dataclass capturing workspace, bridge, settings flags, and callbacks.
   - Methods: `build_tool_context()`, `ensure_outline_tool()`, `ensure_find_sections_tool()`, `ensure_plot_state_tool()`.
2. [x] Move `_tool_registry_context`, `_ensure_*` helpers, and plot-state wiring from `MainWindow` into the new module.
3. [x] Update `MainWindow` to instantiate `ToolProvider` and delegate tool creation.
4. [x] Add unit tests `tests/test_tool_provider.py` covering memoization, feature flag gating, and failure handling.

*Status:* Phase A1 merged. Verified via `uv run pytest tests/test_tool_provider.py` (pass) and regression `uv run pytest tests/test_main_window.py` to ensure wiring remained intact.

### A2. Outline Runtime Manager
1. [x] Create `src/tinkerbell/ui/outline_runtime.py` encapsulating worker lifecycle:
   - `OutlineRuntime.start(loop)` → returns worker.
   - `OutlineRuntime.shutdown()` handles async close.
   - Optional `resolve_memory()` helper for controllers.
2. [x] Move `_create_outline_worker`, `_start_outline_worker`, `_shutdown_outline_worker`, `_outline_memory` into the runtime class.
3. [x] Update `EmbeddingController` integration to depend on OutlineRuntime rather than direct attributes.
4. [x] Add tests `tests/test_outline_runtime.py` (use stub workers to verify start/stop behavior).

*Status:* Phase A2 landed. `MainWindow` now instantiates `OutlineRuntime` once and delegates lifecycle + memory lookups. Verified via `uv run pytest tests/test_outline_runtime.py tests/test_main_window.py`.

**Exit Criteria Phase A:** `MainWindow` no longer implements tool factories or outline worker lifecycle; all references flow through the new modules and tests cover them.

## Phase B – Interaction Controllers
### B1. Manual Tool Controller
1. [x] Introduce `manual_tool_controller.py` with methods to format outline/retrieval responses and record manual tool traces.
2. [x] Move `_render_manual_outline_response`, `_render_manual_retrieval_response`, `_format_retrieval_pointers`, `_summarize_manual_input`, `_record_manual_tool_trace`, and `_render_outline_tree_lines` into the controller.
3. [x] Provide dependency injection for `ChatPanel` so the controller raises tool traces independently.
4. [x] Tests `tests/test_manual_tool_controller.py` verifying formatting, truncation, notes, and metadata.

*Status:* Phase B1 complete. `MainWindow` now instantiates `ManualToolController` for manual outline/find commands and defers formatting + trace wiring. Verified via `uv run pytest tests/test_manual_tool_controller.py tests/test_main_window.py`.

### B2. AI Turn Coordinator & Tool Trace Presenter
1. [x] Create `ai_turn_coordinator.py` encapsulating `_run_ai_turn`, `_handle_ai_stream_event`, `_process_stream_event`, cancellation logic, and telemetry hooks.
2. [x] Create `tool_trace_presenter.py` managing `_record_tool_call_arguments_delta`, `_finalize_tool_call_arguments`, `_record_tool_call_result`, `_apply_tool_result_to_trace`, `_annotate_tool_traces_with_compaction`.
3. [x] `MainWindow` should register callbacks (status updater, `ChatPanel` appenders, `TelemetryController` hooks) with the coordinator.
4. [x] Tests:
   - `tests/test_ai_turn_coordinator.py`: simulate events, verify chat messages and review controller callbacks.
   - `tests/test_tool_trace_presenter.py`: feed fake events, ensure metadata/durations recorded and compaction flags applied.

*Status:* Coordinator and presenter are wired through `MainWindow`, replacing legacy helpers; dedicated suites landed and validated with `uv run pytest tests/test_ai_turn_coordinator.py tests/test_tool_trace_presenter.py`, then re-verified in the full `uv run pytest` sweep.

### B3. Document State Monitor
1. [x] Add `document_state_monitor.py` handling editor/workspace listeners:
   - Refresh window title, autosave indicators, chat suggestions, snapshot persistence.
   - Provide hooks: `on_clear_diff_overlay`, `on_status_update`, etc.
2. [x] Move `_handle_editor_snapshot`, `_handle_editor_text_changed`, `_handle_editor_selection_changed`, `_handle_active_tab_changed`, `_refresh_chat_suggestions`, `_build_chat_suggestions`, `_summarize_selection_text`, `_emit_outline_timeline_event`, autosave helpers, and snapshot persistence logic into the monitor.
3. [x] Update `MainWindow` to instantiate `DocumentStateMonitor` and register callbacks.
4. [x] Tests `tests/test_document_state_monitor.py` verifying selection summary logic, autosave labels, event fan-out, and timeline traces.

*Status:* Phase B3 wrapped. `DocumentStateMonitor` owns autosave/status logic, MainWindow delegates all editor listeners, and regressions were covered via `uv run pytest tests/test_document_state_monitor.py tests/test_main_window.py`.

### B4. Review Overlay Manager Enhancements
1. [x] Extend `AIReviewController` or add `review_overlay_manager.py` to own `_maybe_clear_diff_overlay`, `_clear_diff_overlay`, `_find_tab_id_for_document`, `_handle_accept_ai_changes`, `_handle_reject_ai_changes` workflows.
2. [x] Ensure the new component handles overlay restoration, orphan tracking, and notice text.
3. [x] Add tests `tests/test_review_overlay_manager.py` with mock tabs/workspace to validate branch coverage.

*Status:* Phase B4 completed. `MainWindow` now delegates overlay orchestration to `ReviewOverlayManager`, and unit coverage lives in `tests/test_review_overlay_manager.py` plus refreshed `tests/test_main_window.py`. Verified via `uv run pytest tests/test_review_overlay_manager.py tests/test_main_window.py`.

**Exit Criteria Phase B:** `MainWindow` delegates manual tool rendering, AI turn orchestration, document state monitoring, and review overlays to dedicated controllers; new tests cover these modules.

## Phase C – Persistence & System Services
### C1. Document Session Service
1. [x] Create `document_session_service.py` encapsulating:
   - File dialog prompts (open/save/save-as/import path wrappers).
   - Recent file tracking, `_remember_recent_file`, `_restore_last_session_document`.
   - Snapshot persistence + autosave interactions.
   - Workspace sync triggers (settings store updates, tab metadata serialization).
2. [x] Wire `MainWindow` file/menu actions (`_handle_open_requested`, `_handle_save_*`, `_handle_import_requested`, `_handle_new_tab_requested`, `_handle_close_tab_requested`, `_handle_revert_requested`) through the service.
3. [x] Tests `tests/test_document_session_service.py` mocking file system interactions and ensuring state updates.

*Status:* Service extracted and fully wired through `MainWindow`, including dialog passthrough helpers so legacy tests can monkeypatch window-level hooks. Dedicated coverage now lives in `tests/test_document_session_service.py` (recent files, workspace sync, snapshot restore, save dialog metadata). Verified via `uv run pytest tests/test_document_session_service.py` alongside the regression `tests/test_main_window.py` run.

### C2. Settings & Theme Runtime Wrapper
1. [x] Extract `_apply_runtime_settings`, `_apply_theme_setting`, `_ai_settings_signature`, and related helpers into `settings_runtime.py`.
2. [x] Provide idempotent APIs for applying settings, resolving AI controller signatures, and theme management.
3. [x] Update `MainWindow` initialization to rely on the wrapper.
4. [x] Tests `tests/test_settings_runtime.py` for theme application and AI controller reset behavior.

*Status:* Runtime/theme logic now lives in `src/tinkerbell/ui/settings_runtime.py`, with `MainWindow` delegating runtime updates and the new `tests/test_settings_runtime.py` covering theme application plus AI controller rebuild/disable flows (`uv run pytest tests/test_settings_runtime.py tests/test_main_window.py`).

**Exit Criteria Phase C:** File/dialog logic, persistence, and settings/theme application live in dedicated services with unit coverage; `MainWindow` handles only high-level orchestration.

## Phase D – Final Slimming & Validation
1. [x] Remove redundant helpers from `MainWindow`, keeping only composition wiring, event hookups, and simple delegations.
2. [x] Update `docs/operations/main_window_wiring.md` to reflect new controllers/services and signal flow changes.
3. [x] Re-run `pytest tests/test_main_window.py` plus all new suites to ensure stability.
4. [x] Capture post-cleanup metrics (LOC count, number of controllers, test coverage delta) and compare against Phase 0 baseline.
5. [x] Tag `cleanup_v2.md` milestones as completed, archive implementation doc if desired.

**Metrics snapshot (2025-11-18)**

- `src/tinkerbell/ui/main_window.py`: 1,513 lines (down from ~3,050 noted in Phase 0) while continuing to act purely as a composition root.
- Controllers/services now orchestrated externally: TelemetryController, AIReviewController, ReviewOverlayManager, DocumentStateMonitor, DocumentSessionService, ImportController, EmbeddingController, OutlineRuntime, ToolProvider, AITurnCoordinator, ToolTracePresenter, SettingsRuntime (12 total).
- Targeted regressions run: `pytest tests/test_main_window.py tests/test_tool_provider.py tests/test_outline_runtime.py` (81 tests passed).
- Coverage delta tooling still pending; no LCOV run in this pass.

## Risk & Coordination Notes
- Ensure each extraction lands with its own tests before removing code from `MainWindow` to avoid regressions.
- Maintain feature parity by integrating controllers one at a time; use feature flags (e.g., env vars) if partial rollouts are required.
- Communication checkpoints: after each phase, demo the slimmer `MainWindow` and updated documentation to validate direction.

This plan translates the high-level goals into trackable tasks, letting the team execute incrementally while keeping `MainWindow` focused on orchestration rather than implementation details.
