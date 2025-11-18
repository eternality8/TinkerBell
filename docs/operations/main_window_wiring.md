# Main Window Wiring Reference

This document captures how `MainWindow` composes widgets, routes signals, and delegates to helper controllers. It is intended to satisfy the cleanup plan task of documenting the current wiring before deeper refactors.

## Widget & Controller Topology

| Component | Location | Owned By | Responsibility |
| --- | --- | --- | --- |
| `MainWindow` | `src/tinkerbell/ui/main_window.py` | Qt shell | Coordinates everything, owns async lifecycle, forwards UI events to controllers/services. |
| `TabbedEditorWidget` | `src/tinkerbell/editor/tabbed_editor.py` | `_editor` | Multi-document editor with workspace state + autosave wiring. |
| `Workspace` | `TabbedEditorWidget.workspace` | `_workspace` | Emits active tab events, exposes document lookup helpers. |
| `ChatPanel` | `src/tinkerbell/chat/chat_panel.py` | `_chat_panel` | Hosts conversation UI, tool activity panel, suggestion controls. |
| `WorkspaceBridgeRouter` | `src/tinkerbell/services/bridge_router.py` | `_bridge` | Applies AI edits to documents, reports failures to the window. |
| `StatusBar` | `src/tinkerbell/widgets/status_bar.py` | `_status_bar` | Displays status text plus telemetry badges. |
| `TelemetryController` | `src/tinkerbell/ui/telemetry_controller.py` | `_telemetry_controller` | Updates status bar indicators (memory, subagents, compaction stats). |
| `AIReviewController` | `src/tinkerbell/ui/ai_review_controller.py` | `_review_controller` | Owns AI review sessions, diff overlays, accept/reject UX. |
| `ImportController` | `src/tinkerbell/ui/import_controller.py` | `_import_controller` | Handles import dialogs, tab creation, persistence, and errors. |
| `EmbeddingController` | `src/tinkerbell/ui/embedding_controller.py` | `_embedding_controller` | Manages embedding cache root, outline worker propagation, runtime toggles. |
| `WindowChrome` | `src/tinkerbell/ui/window_shell.py` | `_initialize_ui()` | Builds splitter, menus, toolbars, and Qt actions, keeping `MainWindow` lean. |

## Signal & Callback Map

| Emitter | Hook | Purpose |
| --- | --- | --- |
| `TabbedEditorWidget` | `add_snapshot_listener(_handle_editor_snapshot)` | Persist editor snapshots for AI context + autosave cache. |
| `TabbedEditorWidget` | `add_text_listener(_handle_editor_text_changed)` | Track dirty state, schedule suggestion refresh, update outline digests. |
| `TabbedEditorWidget` | `add_selection_listener(_handle_editor_selection_changed)` | Keep status bar selection info, drive targeted AI prompts. |
| `TabbedEditorWidget` | `add_tab_created_listener(_bridge.track_tab)` | Let `WorkspaceBridgeRouter` observe new tabs for diff application. |
| `Workspace` | `add_active_listener(_handle_active_tab_changed)` | Update window title, autosave indicator, outline caches when focus changes. |
| `ChatPanel` | `add_request_listener(_handle_chat_request)` | Kick off AI turns, snapshot chat state, delegate to review controller. |
| `ChatPanel` | `add_session_reset_listener(_handle_chat_session_reset)` | Clear snapshots + review state when chat is reset. |
| `ChatPanel` | `add_suggestion_panel_listener(_handle_suggestion_panel_toggled)` | Toggle background suggestion refresh telemetry + layout. |
| `ChatPanel` | `set_stop_ai_callback(_cancel_active_ai_turn)` | Wire the stop button to the async task cancel path. |
| `WorkspaceBridgeRouter` | `add_edit_listener(_handle_edit_applied)` | Apply AI edits into the editor and sync status indicators. |
| `WorkspaceBridgeRouter` | `add_failure_listener(_handle_edit_failure)` | Surface tool/apply errors to chat panel + status bar. |
| `ImportController` | `prompt_for_path=lambda: _prompt_for_import_path()` | Reuse window dialogs when importing documents. |
| `ImportController` | `status_updater=update_status` | Centralize status bar updates for import flows. |
| `TelemetryController` | `register_subagent_listeners()` | Subscribes to AI controller updates to refresh badges. |

## Controller Delegations

- **Telemetry:** `MainWindow` instantiates `TelemetryController` with the status bar and registers listeners immediately after UI creation. Subagent toggles flow through `update_subagent_indicator()` so the window no longer mutates status widgets directly.
- **AI Review:** Accept/reject callbacks, diff overlay clearing, and status updates are passed into `AIReviewController`. The window delegates all review lifecycle calls (begin/abort/finalize) and no longer touches `_pending_turn_review` internals.
- **Importer:** Dialog prompts, error handling, autosave sync, and workspace persistence run through `ImportController`. Tests can still swap the underlying `FileImporter` via the shim property.
- **Embeddings/Outline:** `EmbeddingController` resolves cache roots, spawns/propagates outline workers, and exposes `resolve_index()` used by outline + find-sections tools.
- **Window Chrome:** `WindowChrome` (new in this wrap-up) centralizes splitter creation, menu/tool definitions, Qt action wiring, and status bar installation.

## Tool & AI Service Wiring

- `WindowChrome` builds the declarative action set used by menus/toolbars. `MainWindow._action_callbacks()` maps each action identifier to its handler, making the wiring explicit.
- `_tool_registry_context()` packages the bridge, outline helpers, and feature flags for the AI tool registry. Default tools register during `_initialize_ui()`, while Phase 3 and plot-state tools register lazily when features toggle.
- Outline + plot tooling lazily instantiate via `_ensure_*` helpers, with caches (`_outline_tool`, `_find_sections_tool`, `_plot_state_tool`) hanging off the window but with work delegated to controllers/workers.

## Event Flow Summary

1. Qt shell creation → `WindowChrome.assemble()` sets the splitter, menu bar, toolbars, and status bar.
2. Controllers (`TelemetryController`, `AIReviewController`, `ImportController`, `EmbeddingController`) begin observing the status bar, chat panel, workspace, and async loop.
3. Editor/chat/workspace signals feed into `_handle_*` methods, which in turn delegate to controllers (e.g., imports, review, telemetry) or services (e.g., outline worker, AI controller).
4. Tool registry context is built once and reused by feature toggles, ensuring controllers/services stay decoupled from Qt event glue.

Keep this map up to date when introducing new controllers or modifying signal hookups so the `cleanup_plan.md` stays trustworthy.
