# Change Control: Accept / Reject AI Edits

## Objective
- Give users explicit control to accept AI edits (clear the diff highlight) or reject them (revert both the document and chat to the state before the last AI turn).
- Preserve the existing diff overlay experience so the user still sees what changed, but make the follow-up action deterministic.

## Current Behaviour Summary
- `EditorWidget.show_diff_overlay` applies temporary highlighting per tab, and `MainWindow._tabs_with_overlay` tracks which tabs currently show changes. Highlights go away only when `MainWindow._clear_diff_overlay` is called or when `_maybe_clear_diff_overlay` sees a user edit.
- There is no structured notion of an "AI turn"; once `ChatPanel.send_prompt` fires, history is appended and cannot be rewound.
- `DocumentBridge` queues edits per tab and reports `_handle_edit_applied`, but there is no snapshot of the document/chat state from before the directive was applied.

## Proposed Implementation

### 1. Capture turn snapshots before each AI request
- Add a lightweight `ChatTurnSnapshot` dataclass (likely in `src/tinkerbell/chat/chat_panel.py`) with the chat history, tool traces, composer text/context, suggestion list, and `ai_running` flag.
- Extend `ChatPanel` with:
  - `capture_state()` (deep-copy `_messages`, `_tool_traces`, `_composer_context`, `_composer_text`, `_suggestions`).
  - `restore_state(snapshot: ChatTurnSnapshot)` that restores the copied structures and calls `_refresh_*` helpers so the UI reflects the rollback even in headless tests.
  - `consume_turn_snapshot()` that returns and clears the last snapshot captured right before `send_prompt` mutates state.
  - Modify `send_prompt()` to capture a snapshot before `append_user_message()` + composer clearing, then stash it so `MainWindow` can retrieve it when the request listener fires.
- In `MainWindow._handle_chat_request`, call `self._chat_panel.consume_turn_snapshot()` to get the chat state from *before* the just-sent user message, and fall back to `capture_state()` when metadata already came from manual commands (so we never store `None`).
- Initialize a `PendingTurnReview` envelope that contains the chat snapshot, prompt metadata, request/turn id, and an empty `tab_sessions: dict[str, PendingReviewSession]`. This envelope represents the whole AI turn and is the single thing users will accept or reject.
- Do **not** capture every tab eagerly. Instead, lazily snapshot each tab the first time a directive targets it: capture `DocumentState` via `copy.deepcopy(tab.document())` plus its current `DiffOverlayState` (if any) so earlier overlays can be restored if we roll back. Attach those to `tab_sessions[tab_id]` inside the envelope.
- Store related metadata (prompt text, selection summary, timestamp) **and** initialize an empty list per tab session that will accumulate per-edit summaries (action, spans, diff text) because the AI can fire multiple directives inside the same turn and across multiple tabs.

### 2. Track pending review sessions per tab in `MainWindow`
- Introduce two dataclasses:
   - `PendingTurnReview`: chat snapshot, prompt metadata, `turn_id`, `created_at`, `tab_sessions: dict[str, PendingReviewSession]`, `total_edit_count`, and flags for `completed` / `ready_for_review`.
   - `PendingReviewSession`: per-tab state containing `tab_id`, `document_id`, `document_snapshot`, `previous_overlay`, `applied_edits: list[EditSummary]`, merged overlay spans, and bookkeeping about whether the tab still exists.
- Maintain a single `self._pending_turn_review: PendingTurnReview | None`. This envelope owns the chat snapshot and orchestrates all per-tab sessions. A helper `self._pending_reviews_by_tab` dictionary can map tab ids to the per-tab session for quick lookup.
- When `_handle_chat_request` kicks off an AI turn, discard any prior envelope (auto-accept) and create a new one tied to the request id/prompt. No tab sessions are added yet.
- When `_handle_edit_applied` runs, inspect the directive’s `tab_id`/document id to resolve which tab was edited. If that tab has no session yet, create one by snapshotting the tab as described above and store it inside the envelope. Append the new edit metadata to `session.applied_edits`, merge the spans into a single overlay payload (union of all edit spans for that tab), and update `turn.total_edit_count` and a `total_tabs_affected` counter.
- Keep `turn.ready_for_review=False` until the controller signals the turn is complete (via `_on_ai_task_finished` or a new callback). Once the turn finishes, mark the entire envelope ready and call `_show_review_controls()` so the user can accept/reject the *whole* turn. The review UI should surface aggregate stats like “Turn changed 2 tabs / 5 edits.”
- If the AI turn completes without any edit (no call to `_handle_edit_applied`), discard the envelope so Accept/Reject never surfaces.

### 3. Review controls UI & commands
- Extend `widgets/status_bar.StatusBar` to support an optional "review controls" widget (two buttons + summary label). Implement it as a tiny helper class (e.g. `DiffReviewControls`) that renders `Accept` and `Reject` buttons when Qt is available and exposes noop setters when headless. The widget should surface both the edit count and the number of tabs affected (e.g., "5 edits across 2 tabs").
- Provide `StatusBar.set_review_state(summary: str | None, *, accept_callback, reject_callback)` so `MainWindow` can toggle the widget regardless of which tab is active (because the action applies to the entire turn). The summary should highlight the total edits/tabs (e.g., "Accept 5 edits across 2 tabs") and optionally show a tooltip listing each tab touched.
- Add keyboard/menu affordances:
  - New actions `ai_accept_changes` (e.g. shortcut `Ctrl+Shift+Enter`) and `ai_reject_changes` (e.g. `Ctrl+Shift+Backspace`) in `_create_actions`, wired to `_handle_accept_ai_changes` / `_handle_reject_ai_changes`.
  - Append these to the `AI` menu so the commands are available even without Qt widgets (tests can call the callbacks directly).
- Update `_handle_active_tab_changed` to refresh the review controls; when a pending envelope exists, the controls remain visible regardless of tab because they act on the whole turn. Optionally highlight the active tab’s contribution (e.g., show how many edits affected this tab in the status bar detail panel).

### 4. Accept flow (clearing the highlight)
1. Handler validates that the `PendingTurnReview` exists and is `ready_for_review`.
2. Iterate every tab session inside the envelope: if the tab still exists, call `_clear_diff_overlay(tab_id=session.tab_id)` so each editor clears its merged overlay spans.
3. Drop the entire envelope + lookup dicts and hide the review controls.
4. Emit a status update summarizing the bundle (e.g., "Accepted 5 edits across 2 tabs") and keep the chat history as-is (since the turn remains part of the conversation).

### 5. Reject flow (rollback document + chat)
1. Ensure the `PendingTurnReview` exists and every tab session can be safely rolled back (compare each tab’s current `version_signature()` to the snapshot signature; warn/abort if any diverged unless we allow forced reverts tab-by-tab).
2. For each session:
   - Reload the stored `DocumentState` via `tab.editor.load_document(deepcopy(session.document_snapshot))`. If the tab was closed after the edits landed, skip the reload altogether (we ignore closed tabs per the lifecycle note) and emit a debug log so telemetry still shows which documents weren’t restored.
   - Restore any previous overlay: if `session.previous_overlay` exists, call `tab.editor.show_diff_overlay(...)`; otherwise call `_clear_diff_overlay`.
   - Ask that tab’s `DocumentBridge` to refresh (`bridge.generate_snapshot(delta_only=True)` or a dedicated `reload_from_state` helper) so its version metadata realigns.
3. Restore chat/UI once by invoking `self._chat_panel.restore_state(turn.chat_snapshot)` so the user prompt and assistant response disappear and the composer text comes back. Because we bundled the entire turn, this single restore is sufficient even if multiple tool responses occurred.
4. Remove any tool traces that came from the rejected turn by relying on the restored chat state; also clear `_tool_trace_index` entries whose IDs vanished.
5. Update status ("Rejected 5 edits across 2 tabs"), hide review controls, and drop the pending envelope.

### 6. Lifecycle + edge cases
- **Manual edits after AI change:** `_maybe_clear_diff_overlay` currently fires when a user types; extend it so *any* user edit during a pending review immediately drops the whole turn envelope (clearing overlays + controls). This avoids rejecting a turn after the user has already modified any affected document.
- **Multiple tabs:** since an AI turn can touch any number of tabs, ensure `_handle_edit_applied` can resolve a tab even when it isn’t active (using `tab_id` metadata emitted by the directive or by scanning `DocumentWorkspace` for the document id). If a tab is closed before the turn ends, mark its session as `orphaned` and simply ignore it during accept/reject (no reopen/revert attempt, just drop the stored snapshot and log a note).
- **AI errors / cancellations:** whenever `_handle_ai_failure` or `_cancel_active_ai_turn` fires, drop the entire pending turn envelope (including all tab sessions) and rehydrate the composer with the user’s original prompt so they can immediately retry. `restore_state` therefore needs a way to repopulate only the composer without rewriting history.
- **Tab close / document reload / new chat:** ensure we clear any session tied to a tab when it’s closed, when `ChatPanel.start_new_chat()` fires, or when `WorkspaceBridgeRouter` removes a tab.
- **No edit produced:** if the AI turn only replied in chat but never invoked the edit tool, mark the pending session as completed-without-review so Accept/Reject never shows.
- **Telemetry & undo stack:** document reloading via `load_document` clears the undo stack. Note this in the implementation and consider pushing an undo snapshot first so the user can redo the rejection if needed.
- **Future enhancement – per-edit controls:** since we now bundle multiple edits under one decision, a future iteration could surface a breakdown (checkboxes or diff browser) to accept subsets or reorder edits, but that’s intentionally deferred.

### 7. Testing plan
1. `tests/test_chat_panel.py`:
   - New tests for `capture_state`/`restore_state` round-tripping history, composer text, and suggestions.
   - Ensure `send_prompt` caches a snapshot and `consume_turn_snapshot` exposes it exactly once.
2. `tests/test_main_window.py`:
   - Simulate a turn that edits a single tab by manually invoking `_handle_chat_request` (with a fake controller) + `_handle_edit_applied`, then call `_handle_accept_ai_changes` and assert that every tab in the envelope clears overlays and the pending envelope is removed.
   - Simulate a turn that edits **multiple** tabs: craft directives targeting two tabs, ensure each tab’s overlay appears, then exercise both accept and reject flows to confirm documents/overlays/chat are restored/cleared across tabs.
3. `tests/test_status_bar.py` (if available) or extend `test_widgets_status_bar.py`:
   - Validate that the new review controls widget toggles visibility and that callbacks fire when buttons are "clicked" (simulate via direct method calls when Qt absent).
4. `tests/test_workspace.py` or `test_editor_widget.py`:
   - Ensure `DiffOverlayState` survives deepcopy so we can safely stash/restore overlays.

### 8. Follow-ups / assumptions
- Storing entire `DocumentState` per turn could be heavy for very large documents; if this proves problematic we can store text only plus metadata diff.
- If future releases add multi-edit tool runs inside a single AI turn, we may need to support stacking multiple overlays per session; current plan handles one diff per tab.
- Consider persisting a short audit log (e.g., in telemetry) so we know when users reject edits and why.
