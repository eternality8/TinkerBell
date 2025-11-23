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
| `ReviewOverlayManager` | `src/tinkerbell/ui/review_overlay_manager.py` | `_review_overlay_manager` | Restores/clears diff overlays per tab and coordinates notice text. |
| `DocumentStateMonitor` | `src/tinkerbell/ui/document_state_monitor.py` | `_document_monitor` | Observes editor/workspace events, manages snapshots, autosave labels, chat suggestions. |
| `DocumentSessionService` | `src/tinkerbell/ui/document_session_service.py` | `_document_session` | Handles open/save/import dialogs, recent files, workspace persistence, and snapshot replay. |
| `ImportController` | `src/tinkerbell/ui/import_controller.py` | `_import_controller` | Handles import dialogs, tab creation, persistence, and errors. |
| `EmbeddingController` | `src/tinkerbell/ui/embedding_controller.py` | `_embedding_controller` | Manages embedding cache root, outline worker propagation, runtime toggles. |
| `OutlineRuntime` | `src/tinkerbell/ui/outline_runtime.py` | `_outline_runtime` | Starts/shuts the outline worker and exposes shared outline memory. |
| `ToolProvider` | `src/tinkerbell/ui/tools/provider.py` | `_tool_provider` | Builds tool registry contexts and lazily caches outline, find-sections, and plot-state tools. |
| `AITurnCoordinator` | `src/tinkerbell/ui/ai_turn_coordinator.py` | `_ai_turn_coordinator` | Runs AI turns, streams updates to chat/review controllers, handles cancellation & telemetry. |
| `ToolTracePresenter` | `src/tinkerbell/ui/tool_trace_presenter.py` | `_tool_trace_presenter` | Aggregates tool call argument/result deltas and feeds the chat panel activity log. |
| `WindowChrome` | `src/tinkerbell/ui/window_shell.py` | `_initialize_ui()` | Builds splitter, menus, toolbars, and Qt actions, keeping `MainWindow` lean. |

## Signal & Callback Map

| Emitter | Hook | Purpose |
| --- | --- | --- |
| `TabbedEditorWidget` | `add_snapshot_listener(document_monitor.handle_editor_snapshot)` | Persist editor snapshots for AI context + autosave cache. |
| `TabbedEditorWidget` | `add_text_listener(document_monitor.handle_editor_text_changed)` | Track dirty state, schedule suggestion refresh, update outline digests. |
| `TabbedEditorWidget` | *(selection listener removed)* | Selection data stays private to the editor; controllers request spans through SelectionRangeTool or cached snapshots instead of live listeners. |
| `TabbedEditorWidget` | `add_tab_created_listener(_bridge.track_tab)` | Let `WorkspaceBridgeRouter` observe new tabs for diff application. |
| `Workspace` | `add_active_listener(document_monitor.handle_active_tab_changed)` | Update window title, autosave indicator, outline caches when focus changes. |
| `Workspace` | `add_active_listener(_handle_active_tab_for_review)` | Keep AI review state in sync with the focused document. |
| `ChatPanel` | `add_request_listener(_handle_chat_request)` | Kick off AI turns, snapshot chat state, delegate to review controller. |
| `ChatPanel` | `add_session_reset_listener(_handle_chat_session_reset)` | Clear snapshots + review state when chat is reset. |
| `ChatPanel` | `add_suggestion_panel_listener(_handle_suggestion_panel_toggled)` | Toggle background suggestion refresh telemetry + layout. |
| `ChatPanel` | `set_stop_ai_callback(_cancel_active_ai_turn)` | Wire the stop button to the async task cancel path. |
| `WorkspaceBridgeRouter` | `add_edit_listener(_handle_edit_applied)` | Apply AI edits into the editor and sync status indicators. |
| `WorkspaceBridgeRouter` | `add_failure_listener(_handle_edit_failure)` | Surface tool/apply errors to chat panel + status bar. |
| `ImportController` | `prompt_for_path=lambda: _prompt_for_import_path()` | Reuse window dialogs when importing documents. |
| `ImportController` | `status_updater=update_status` | Centralize status bar updates for import flows. |
| `TelemetryController` | `register_subagent_listeners()` | Subscribes to AI controller updates to refresh badges. |
| `DocumentSessionService` | `set_review_controller / set_review_overlay_manager / set_import_dialog_filter_provider` | Routes UI callbacks through service seams so tests and headless runs can override dialogs. |

## Controller Delegations

- **Telemetry:** `MainWindow` instantiates `TelemetryController` with the status bar and registers listeners immediately after UI creation. Subagent toggles flow through `update_subagent_indicator()` so the window no longer mutates status widgets directly.
- **AI Review:** Accept/reject callbacks, diff overlay clearing, and status updates are passed into `AIReviewController`. The window delegates all review lifecycle calls (begin/abort/finalize) and no longer touches `_pending_turn_review` internals.
- **Document State & Sessions:** `DocumentStateMonitor` owns autosave, suggestion, and outline digest bookkeeping, while `DocumentSessionService` centralizes open/save/import flows plus recent file tracking and snapshot restoration.
- **Importer:** Dialog prompts, error handling, autosave sync, and workspace persistence run through `ImportController`. Tests can still swap the underlying `FileImporter` via the shim property.
- **Embeddings/Outline:** `EmbeddingController` resolves cache roots and publishes embedding metadata. The `OutlineRuntime` now owns the `OutlineBuilderWorker` lifecycle, and `ToolProvider` supplies cached tool factories and registry contexts.
- **AI Turn Streaming:** `AITurnCoordinator` encapsulates `_run_ai_turn`, streaming callbacks, cancellation, and telemetry fan-out, while `ToolTracePresenter` mirrors tool call metadata back into the `ChatPanel` activity list.
- **Review Surface:** `AIReviewController` and `ReviewOverlayManager` manage diff overlays, orphan tracking, and notice lifecycles so the window only issues high-level commands.
- **Window Chrome:** `WindowChrome` centralizes splitter creation, menu/tool definitions, Qt action wiring, and status bar installation.

## Tool & AI Service Wiring

- `WindowChrome` builds the declarative action set used by menus/toolbars. `MainWindow._action_callbacks()` maps each action identifier to its handler, making the wiring explicit.
- `ToolProvider` builds the `ToolRegistryContext`, exposing lazy `ensure_*` hooks plus the auto-patch callback. Default tools register during `_initialize_ui()`, while Phase 3 and plot-state tools register (and unregister) in response to feature toggles.
- `OutlineRuntime` is the sole owner of the outline worker lifecycle, exposing `outline_memory()` to both the `ToolProvider` and `DocumentStateMonitor`. This keeps worker startup/shutdown logic out of the window class.
- Manual `/outline` and `/find` commands resolve their tools by calling into `ToolProvider`, so the window no longer keeps per-tool caches.

## Event Flow Summary

1. Qt shell creation → `WindowChrome.assemble()` sets the splitter, menu bar, toolbars, and status bar.
2. Controllers (`TelemetryController`, `AIReviewController`, `ReviewOverlayManager`, `ImportController`, `DocumentStateMonitor`, `DocumentSessionService`, `EmbeddingController`, `AITurnCoordinator`) begin observing the status bar, chat panel, workspace, and async loop.
3. Editor/chat/workspace signals feed into controller hooks (`DocumentStateMonitor`, `_handle_chat_request`, `_handle_active_tab_for_review`, etc.), which in turn delegate to services (imports, outline runtime, AI coordinator) instead of hosting logic inline.
4. `ToolProvider` builds the registry context on demand so feature toggles simply flip flags; Outline/tool construction stays outside the window class.

Keep this map up to date when introducing new controllers or modifying signal hookups so the `cleanup_plan.md` stays trustworthy.
