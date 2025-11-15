# Tabs Feature Implementation Plan

This document captures the end-to-end plan for adding multi-document tabs to TinkerBell. It focuses on UX, architecture, AI/tooling implications, persistence, and validation so the change can be implemented incrementally without regressing current single-document behavior.

## 1. Goals & Guardrails

- **Primary goal**: allow users to open, edit, and save multiple documents simultaneously within one TinkerBell window using a tabbed interface.
- **AI parity**: the chat assistant and AI controller (current LangGraph/LangChain-inspired executor) must be able to reason about every open tab, including reading from one tab and writing into another when asked (e.g., summarize Tab A into Tab B) while defaulting to the active tab when no target is specified.
- **Continuity**: preserve existing keyboard shortcuts, autosave safety rails, and document snapshots. Users who never open a second tab should see no regressions.
- **Testability**: expand the pytest/pytest-qt coverage to include tab lifecycle events (create, switch, close, restore) and AI interactions scoped to a tab.
- **Guardrails**: no multi-window workspace yet. Tabs remain within a single window, but AI tooling must safely target any open tab via explicit `tab_id` selection.

## 2. Current State Summary

- `MainWindow` owns a single `EditorWidget`, `DocumentBridge`, and `ChatPanel`.
- All document metadata (current path, dirty flag, unsaved snapshots, recent files) lives in `MainWindow` or `Settings` assuming one active document.
- AI tools (`document_snapshot`, `document_edit` running in patch-only mode, `document_apply_patch`, `diff_builder`, `search_replace`, `validate_snippet`) are registered once inside `MainWindow` and implicitly target the single bridge/editor instance.
- Autosave + restore logic keeps one unnamed draft (`unsaved_snapshot`) plus a dict of dirty file snapshots (`unsaved_snapshots[path]`), but `_current_document_path`, `last_open_file`, and dirty bookkeeping are all global and assume exactly one active editor.
- Status bar messaging, composer metadata, and the contextual suggestion system (`SUGGESTION_LOADING…` flows + cache) all read from the lone editor state managed via `_handle_editor_selection_changed`.

Any tab implementation must refactor these singletons into per-tab concerns while keeping a clean facade accessible to the rest of the window.

## 3. UX & Interaction Requirements

1. **Tab bar placement**: place a `QTabWidget` (or custom tab strip) above the editor panel. Each tab shows filename (with `*` for dirty documents) and an optional close button.
2. **Actions**:
   - `File → New Tab` (Ctrl+N) opens an empty Markdown document in a new tab.
   - `File → Open…` opens the selected file in a new tab if it isnt already open, otherwise focuses the existing tab.
   - `File → Save` / `Save As…` operate on the active tab.
  - `File → Close Tab` (Ctrl+W) closes the active tab with dirty-check prompts, persisting unsaved state if the user cancels.
  - `File → Revert` should discard unsaved changes only for the focused tab and leave the rest of the workspace untouched.
  - `AI → Refresh Snapshot` (Ctrl+Shift+R) captures the active tab’s state, but the action should eventually accept an optional `tab_id` so agents can refresh background tabs without stealing focus.
  - AI-directed actions can specify a target tab; for example, “copy the summary from the meeting notes tab into the roadmap tab” should resolve both the source snapshot and destination edit without requiring the user to manually switch tabs mid-flow.
   - Optional future: `Reopen Closed Tab`, but out of scope for this milestone.
3. **Tab switching**: clicking a tab or using `Ctrl+PageUp/PageDown` swaps the active editor, updates the chat selection summary, refreshes AI snapshots, and updates the window title.
4. **Status bar & chat**: always describe the active tab (cursor info, document name) while keeping the existing `StatusBar` widget + `update_status` helpers unchanged. Chat history remains global regardless of the selected tab, but composer metadata, selection summaries, and the contextual suggestion cache must be re-derived when the active tab changes so the AI sees the correct snippet even if the user didn’t manually edit anything in that tab.
5. **Startup/restore**: reopen tabs from the previous session (including untitled drafts) with dirty indicators and snapshots intact.

## 4. Architectural Changes

### 4.1 Document Workspace & Tab Data Model

Introduce a new module (e.g., `editor/workspace.py`) containing:

- `DocumentTab` dataclass summarizing `id` (UUID/str), `display_name`, `document_state`, `editor_widget`, `bridge`, `created_at`, `last_snapshot_digest`, etc.
- `DocumentWorkspace` manager responsible for:
  - Owning the collection of `DocumentTab` instances and the active tab ID.
  - Creating/destroying tabs, ensuring `EditorWidget` + `DocumentBridge` pairs are initialized per tab.
  - Emitting signals/callbacks when the active tab changes (so `MainWindow` can refresh menus, chat metadata, status bar).
  - Providing utility methods (`current_document()`, `find_tab_by_path()`, `iter_tabs()`), and orchestrating dirty checks before close.

This manager keeps `MainWindow` lean and makes headless tests easier (workspace logic can be tested without Qt by relying on the stub widgets already used in the repo).

### 4.2 Tabbed Editor Container

- Replace the single `EditorWidget` member in `MainWindow` with a `TabbedEditorWidget` wrapper:
  - Internally host a `QTabWidget` when Qt is available, or a lightweight list of `EditorWidget` stubs in headless mode.
  - Each tab stores its own `EditorWidget` reference; switching tabs emits callbacks to `MainWindow` and `DocumentWorkspace`.
  - Provide helper methods on the wrapper so existing code doesnt need to know about the tab control (`current_editor`, `current_bridge`, `add_tab`, `close_tab`, etc.).
  - Manage the `DocumentBridge` wiring per tab: `TabbedEditorWidget` can expose a `bridge_for_active_tab()` method to keep the AI tool registration path unchanged (it only needs the active bridge).
  - Re-emit the same text/selection/snapshot listener hooks that `MainWindow._wire_signals` currently registers on the lone `EditorWidget`, so chat metadata, autosave, and AI streaming keep functioning without refactoring every downstream consumer.

### 4.3 Menu & Command Routing Changes

- Update `WindowAction` callbacks to reference the workspace:
  - `open_document` becomes `open_document(path, *, in_new_tab=True)`; if the path is already open, focus it and decide whether to reuse the tab or duplicate (default: reuse).
  - `save_document` now accepts an optional `tab_id`; defaults to active tab.
  - Add new actions (`file_new_tab`, `file_close_tab`), register them in menus/toolbars, and add shortcuts.
  - Update undo/redo, toggle preview, `file_revert`, and the `ai_snapshot` action to scope to the active editor (most already do, but audit for any lingering references to `self._editor`).
  - Ensure chat-suggestion toggles (`_handle_suggestion_panel_toggled`) and metadata emitters always consult the workspace so switching tabs updates the composer context before the AI suggestion cache is evaluated.

### 4.4 Autosave, Snapshots & Persistence

- Extend `Settings` with new fields:
  - `open_tabs: list[dict]` capturing `tab_id`, `path`, `language`, `untitled_counter`, etc.
  - `active_tab_id: str | None` for startup restore.
  - `untitled_snapshots: dict[str, dict]` to store multiple untitled drafts keyed by tab ID (replace the single `unsaved_snapshot`).
- Keep the existing `recent_files`, `last_open_file`, and `unsaved_snapshots[path]` data structures but thread them through the workspace so MRU ordering and dirty prompts stay accurate even when multiple paths are open simultaneously.
- Update `_persist_unsaved_snapshot` and `_restore_unsaved_snapshot` to operate per tab:
  - For path-backed docs, reuse the existing `unsaved_snapshots[path]` map but include `tab_id` so we know which tab to hydrate.
  - For untitled docs, store snapshots under `untitled_snapshots[tab_id]`.
- When closing a tab, remove its unsaved snapshot entries (unless the user cancels). When switching tabs, ensure the workspace updates the `Settings.active_tab_id` so the last session restores correctly.

### 4.5 DocumentBridge Refactor

- Extract a thin `BridgeRouter` (or extend `DocumentBridge`) to support multiple editors without duplicating logic:
  - Option A: keep `DocumentBridge` untouched but instantiate one per tab. Tabs share no mutable state, so this is easiest. `TabbedEditorWidget` (or `DocumentWorkspace`) should keep a mapping from `tab_id → bridge` and expose `active_bridge`.
  - Option B: create `TabbedDocumentBridge` that holds a reference to the workspace + tab ID and proxies calls to `DocumentBridge`. This becomes handy for AI tools because they can take a `tab_id` parameter and route to the correct bridge.
- Regardless of approach, we need a central place for AI tools to ask, "Which bridge should I use for this tab?" and to reject directives referencing a stale tab.
- `DocumentApplyPatchTool` currently holds onto a single bridge + edit tool instance via `MainWindow._auto_patch_tool`; once tabs exist this either needs to become a lightweight router (wrapping per-tab bridges) or be created on-demand per tab so diff/patch requests always run against the correct snapshot.

### 4.6 AI & Tooling Updates

**Snapshot Tool**
- Add optional parameters `tab_id`, `source_tab_ids`, and `include_open_documents`:
  - If a `tab_id` is provided, return the snapshot for that tab (default to active tab). Allow batching via `source_tab_ids` so the agent can pull text from multiple tabs in one call when planning cross-tab operations.
  - Active tab snapshots should now include `tab_id`, `tab_title`, and perhaps `open_tabs` metadata (list of `{tab_id, name, path, dirty}`) when `include_open_documents=True`.
  - Attach `workspace_digest` or `tab_version` fields so the agent can detect cross-tab drift.

**Edit Tool**
- Extend the JSON schema (and `DIRECTIVE_SCHEMA`) with a `tab_id`/`document_id` property:
  - When present, `DocumentEditTool` resolves the appropriate bridge via the workspace. If absent, it uses the active tab but also injects the active `tab_id` into the directive so downstream tooling sees which document was edited.
  - Allow directives to reference a tab different from the snapshot source, enabling “write Tab A summary into Tab B” flows. The tool should validate that the destination tab still exists and that the referenced snapshot `version` hasn’t drifted to prevent stale writes.

**Apply Patch & Diff Tools**
- `DocumentApplyPatchTool` (and the helper `DocumentApplyPatchTool.run`) should accept `tab_id` + optional `document_version` overrides so AI requests can target background tabs. Under the hood it should fetch the relevant snapshot/selection via the workspace, build the diff with the appropriate filename, and then hand it to the per-tab `DocumentEditTool` instance.
- `DiffBuilderTool` remains stateless but should surface a `filename` derived from the tab metadata so diff headers stay meaningful when multiple documents are open.

**Search/Replace & Validation Tools**
- No schema change needed, but they should use the same bridge resolution logic so a tool can optionally accept `tab_id` (future-proofing). For now, they can just operate on the active tab while inheriting the new workspace-aware bridge interface.

**Tab enumeration tool (new)**
- Provide a lightweight `list_tabs` (or extend the snapshot tool) that enumerates `{tab_id, title, path, dirty}` so the agent can resolve natural-language requests like “the roadmap tab” before issuing cross-tab edits.

**Agent metadata**
- When `MainWindow` calls `controller.run_chat`, include `tab_id`, `tab_title`, `open_tab_count`, and a compact `{tab_id: title}` map in the metadata so prompts can mention which document(s) the user is referencing.
- Update prompts to clarify that the assistant may read from or edit any tab when explicitly requested, and should fall back to the active tab otherwise.
- Thread the same metadata through `controller.suggest_followups` so contextual suggestions can reference which tab supplied the selection summary (and so the cache key can incorporate `active_tab_id` to avoid collisions when multiple tabs share identical chat history).

### 4.7 Settings Dialog & Runtime Updates

- Surface tab-related preferences only if needed (e.g., "Restore last session tabs" checkbox). By default we can auto-restore tabs but let advanced users disable it.
- Ensure `SettingsStore` serialization handles the new fields while remaining backward compatible (default empty lists/dicts when older settings files are missing the fields).

### 4.8 Status Bar & Window Title

- Move window-title construction to use `DocumentWorkspace.current_display_name()` so it automatically includes `*` for any dirty active tab.
- Status bar should show `Tab N of M` (optional). At minimum, ensure cursor/line labels update when the active tab changes by re-subscribing to the correct editor signals.
- The existing `StatusBar` helper already exposes `set_message`; extend it (or feed it via `MainWindow`) with workspace data so AI status, cursor position, and suggestion hints remain accurate when focus jumps between tabs.

### 4.9 Testing Surface

- Update `tests/test_main_window.py` to cover:
  - Creating multiple tabs and switching between them updates suggestions and status text.
  - `open_document` reuses an existing tab when opening the same path twice.
  - `save_document` and `save_as` operate on the active tab while leaving others untouched.
  - `file_revert` only affects the targeted tab and leaves other dirty tabs unchanged.
  - `ai_snapshot` / `document_snapshot` actions return the active tab metadata by default and honor an explicit `tab_id` when provided.
  - Autosave snapshots persist per tab (untitled + file-backed).
  - Closing a tab drops its snapshot entries and updates the settings state.
- Add a new `tests/test_workspace.py` (or similar) for pure logic around `DocumentWorkspace` (tab lifecycle, serialization payloads, etc.).
- Expand AI tool tests (`tests/test_ai_tools.py`) to assert that `tab_id` is included in snapshot/edit/apply-patch payloads, that `DocumentApplyPatchTool` routes to the correct bridge when multiple tabs exist, and that invalid `tab_id`s raise errors.

## 5. Implementation Roadmap

1. **Phase 1 – Workspace scaffolding**
   - Create `DocumentWorkspace` + `DocumentTab` models and unit tests.
   - Allow creating tabs in headless mode without touching Qt yet.

2. **Phase 2 – UI integration**
   - Introduce `TabbedEditorWidget` and swap it into `MainWindow`.
   - Hook up menus/actions (`New Tab`, `Close Tab`, `Next/Previous Tab`).
   - Ensure status bar and chat selection refresh on tab changes.

3. **Phase 3 – Persistence & autosave**
   - Extend `Settings` schema and adjust load/save helpers.
   - Rework `_persist_unsaved_snapshot`, `_restore_*`, and startup flows to hydrate all tabs.
   - Implement dirty-check prompts when closing tabs.

4. **Phase 4 – AI/tool alignment**
   - Update `DocumentBridge` usage so each tab has its own bridge, plus a router for the AI tools.
   - Extend `DocumentSnapshotTool`, `DocumentEditTool`, and `DIRECTIVE_SCHEMA` with `tab_id` support.
   - Pass workspace metadata into `controller.run_chat` and update tests/mocks.

5. **Phase 5 – Polish & regression tests**
   - Add pytest-qt cases for tab interactions, autosave restore, and AI metadata.
   - Update docs (`README`, `plan.md`) to mention tab capability and shortcuts.
   - Manually verify Windows build (PySide6) with multiple tabs, saving, AI operations, and shutdown/startup cycles.

## 6. Open Questions & Follow-Ups

- **Chat per tab?** Chat history remains intentionally global to support cross-tab orchestration; revisit only if user feedback demands per-tab transcripts.
- **Cross-tab AI commands?** Addressed in this milestone via `tab_id`-aware snapshots/edits and the tab enumeration tool; future work may add richer multi-tab diffing if needed.
- **Memory/performance**: each tab spawns an `EditorWidget` + `DocumentBridge`. Monitor memory usage when dozens of tabs are open and consider lazy widget creation if needed.
- **MRU ordering & tab drag/drop**: not critical for MVP but plan the workspace API so reordering is possible later.

---

With this plan, implementation can proceed in small, testable steps while keeping AI tooling and persistence aligned with the new multi-document reality.
