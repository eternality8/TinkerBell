# Main Window Cleanup v2 Plan

## 1. Current State & Targets
- `src/tinkerbell/ui/main_window.py` sits at ~3,050 lines even after the first wave of refactors.
- Responsibilities span Qt chrome, editor/chat signal wiring, AI run loops, manual tool emulation, autosave/UI adornments, telemetry fan-out, file dialog orchestration, and persistent settings.
- Goal: drive the file below 1,500 lines by the end of this pass while keeping `MainWindow` as a thin composition root.
- Strategy: extract cohesive controllers/services, document their contracts, and expand targeted unit coverage so regressions are caught outside of `tests/test_main_window.py`.

## 2. Responsibility Map (Remaining Hotspots)
1. **Manual tool command flow (≈250 lines)**
   - `_render_manual_outline_response`, `_render_manual_retrieval_response`, `_format_retrieval_pointers`, `_record_manual_tool_trace`, `_summarize_manual_input`.
   - Opportunity: move into `manual_tool_controller.py` that owns manual command parsing/rendering, returning chat-ready strings and tool traces.
2. **AI stream + tool-trace plumbing (≈300 lines)**
   - `_run_ai_turn`, `_handle_ai_stream_event`, `_process_stream_event`, `_record_tool_call_*`, `_apply_tool_result_to_trace`, `_annotate_tool_traces_with_compaction`.
   - Extract to `ai_turn_coordinator.py` + `tool_trace_presenter.py` so `MainWindow` only forwards events.
3. **Autosave/status & selection suggestion logic (≈200 lines)**
   - `_update_autosave_indicator`, `_format_autosave_label`, `_handle_editor_*`, `_refresh_chat_suggestions`, `_build_chat_suggestions`, `_summarize_selection_text`, `_emit_outline_timeline_event`.
   - Create `document_state_monitor.py` to listen to editor/workspace events, update status bar, and emit chat suggestions.
4. **Diff overlay + AI review interactions (≈220 lines)**
   - `_maybe_clear_diff_overlay`, `_clear_diff_overlay`, `_find_tab_id_for_document`, `_handle_accept_ai_changes`, `_handle_reject_ai_changes`.
   - Most logic belongs beside `AIReviewController`; extend that controller (or add `review_overlay_manager.py`) so `MainWindow` calls high-level APIs: `accept_pending_review()`, `reject_pending_review()`, `clear_overlays_for_tab()`.
5. **Document/dialog actions & persistence (≈250 lines)**
   - `_handle_open_requested`, `_handle_save_*`, `_prompt_for_*`, `_remember_recent_file`, `_sync_settings_workspace_state`, `_restore_last_session_document`, `_persist_settings` (earlier in file).
   - Create `document_session_service.py` to encapsulate file dialogs, recent-file bookkeeping, autosave snapshot persistence, and workspace sync.
6. **Outline worker + embedding integration (≈150 lines)**
   - `_create_outline_worker`, `_start_outline_worker`, `_shutdown_outline_worker`, `_resolve_embedding_index`, `_ensure_*` helpers.
   - These can move to `outline_runtime.py` (wrapping the worker lifecycle) and `tool_provider.py` (factory for outline/find-sections/plot tools) to keep `MainWindow` unaware of widget-level details.

## 3. Extraction Plan & Sequencing
### Phase A – Infrastructure
1. **Tool Provider Module**
   - Move `_tool_registry_context`, `_ensure_outline_tool`, `_ensure_find_sections_tool`, `_ensure_plot_state_tool`, and resolver methods into `src/tinkerbell/ui/tools/provider.py`.
   - Provide a small dataclass carrying the callbacks needed for registry wiring.
   - Tests: add unit coverage ensuring factories memoize correctly and propagate feature flags.
2. **Outline Runtime Manager**
   - Extract worker lifecycle helpers to `outline_runtime.py`; expose `OutlineRuntime.start_if_needed()` / `.shutdown()`.
   - `MainWindow` just instantiates runtime with dependencies and calls into it during setup/teardown.

### Phase B – Interaction Controllers
3. **Manual Tool Controller**
   - Own manual command formatting, retrieval outline responses, and trace emission.
   - Interface: `render_outline_status(response, label)`, `render_find_sections(response, query, doc_label)`, `record_manual_trace(name, args, response)`.
   - Tests: deterministic string formatting + trace metadata assertions (no Qt dependency).
4. **AI Turn Coordinator & Tool Trace Presenter**
   - Move async `_run_ai_turn` and stream event handling into `ai_turn_coordinator.py` with callbacks for UI updates (status text, chat append, telemetry, review controller hooks).
   - Tool trace helper becomes its own class with state dictionaries so we can unit test compaction annotations and duration math.
   - Tests: feed fake events, assert `ChatPanel` spy receives updates, ensure compaction metadata flows through.
5. **Document State Monitor**
   - Extract editor/workspace listeners plus autosave indicator logic into `document_state_monitor.py`.
   - Provide hooks to notify `MainWindow` when dirty state changes (to refresh title) and when diff overlays should clear.
   - Tests: feed fake documents/selections and assert suggestions + autosave labels.
6. **Review Overlay Manager Enhancements**
   - Extend `AIReviewController` or add companion `ReviewOverlayManager` to encapsulate `_handle_accept_ai_changes`, `_handle_reject_ai_changes`, `_maybe_clear_diff_overlay`, `_clear_diff_overlay`, `_find_tab_id_for_document`.
   - Tests: simulate tab sessions to verify orphan handling and overlay restoration.

### Phase C – Persistence & Actions
7. **Document Session Service**
   - Responsible for open/save/save-as/revert prompts, remembering recent files, syncing workspace metadata, and restoring last session documents.
   - Replace direct dialog usage with injection-friendly interfaces so tests can stub behavior without touching Qt.
   - Tests: file-path branching logic + persistence interactions.
8. **Settings & Theme Application Wrapper**
   - Wrap `_apply_runtime_settings`, `_apply_theme_setting`, `_ai_settings_signature`, etc., into `settings_runtime.py` to keep `MainWindow` from mutating settings store directly.

### Phase D – Final Slimming
9. **Event Wiring Pass**
   - After controllers exist, rewrite `_initialize_ui()` to instantiate them and register listeners.
   - Remove dead helpers from `MainWindow`, ensuring only high-level delegation remains (ideally <1,200 lines).
10. **Coverage Expansion & Docs**
   - Add targeted unit suites for each new controller, reducing reliance on `tests/test_main_window.py`.
   - Update `docs/operations/main_window_wiring.md` to reflect the new components and signal flow.

## 4. Testing & Validation Strategy
- Maintain `tests/test_main_window.py` as an integration smoke test (menu wiring, controller wiring, default tool registration).
- Create new unit modules:
  - `tests/test_manual_tool_controller.py`
  - `tests/test_ai_turn_coordinator.py`
  - `tests/test_document_state_monitor.py`
  - `tests/test_review_overlay_manager.py`
  - `tests/test_document_session_service.py`
- Each extraction should land with its own test file to keep the refactor incremental and verifiable.
- Continue running targeted PySide6-free tests via `.venv\Scripts\python.exe -m pytest tests/test_main_window.py` after each major move.

## 5. Risk Mitigation
- **Signal wiring regressions:** note every signal → handler pairing in `docs/operations/main_window_wiring.md` before/after each extraction.
- **Controller reentrancy:** ensure new services accept dependency injections (status updater, chat panel, workspace) rather than importing globals to simplify mocking.
- **Async AI runs:** maintain the existing cancelation semantics (e.g., `_suppress_cancel_abort`) inside `ai_turn_coordinator.py` to avoid regressions when canceling turns mid-stream.
- **User-visible messaging:** keep `update_status`/`_post_assistant_notice` calls centralized so UX text does not drift.

## 6. Milestone Checklist
- [x] Tool provider + outline runtime modules in place.
- [x] Manual tool controller extracted with tests.
- [x] AI turn coordinator + tool trace presenter extracted with tests.
- [x] Document state monitor plus review overlay manager extracted; `MainWindow` delegates editor signals accordingly.
- [x] Document session service handles open/save flows; `MainWindow` only calls high-level APIs.
- [x] Final cleanup pass removing obsolete helpers, updating docs, and confirming line count target.

This roadmap keeps the work incremental while steadily shrinking `main_window.py` into a composition shell. Each step introduces testable modules, making the next wave of UI changes safer and easier to reason about.
