# Change Control Implementation Plan

## Goal
Deliver the accept/reject AI edits workflow exactly as described in `change_control.md`, covering multi-tab AI turns, turn-level rollback, and the associated UI/UX affordances.

## Workstreams & Tasks
Each checkbox represents an individually verifiable task. Complete them roughly in order unless a dependency is called out explicitly.

> **Test command reminder:** run any pytest suite via `uv run pytest …` so the virtual environment is configured automatically.

### 1. Chat turn snapshot groundwork
- [x] Add `ChatTurnSnapshot` dataclass to `src/tinkerbell/chat/chat_panel.py` capturing messages, tool traces, composer text/context, suggestions, and `ai_running`.
- [x] Implement `ChatPanel.capture_state()` and `restore_state()` that deep-copy/rehydrate all relevant structures and refresh UI helpers.
- [x] Implement `ChatPanel.consume_turn_snapshot()` with single-use semantics and update `send_prompt()` to capture a snapshot before it mutates chat state.
- [x] Update `MainWindow._handle_chat_request` to call `consume_turn_snapshot()` (falling back to `capture_state()` when needed) and stash the snapshot in the pending turn envelope.

### 2. Pending turn envelope & per-tab sessions
- [x] Introduce `PendingTurnReview` and `PendingReviewSession` dataclasses (likely in `main_window.py` or a companion module) with the fields defined in the plan.
- [x] Add `MainWindow._pending_turn_review` and `self._pending_reviews_by_tab` structures, plus helpers to create/drop envelopes.
- [x] Update `_handle_chat_request` to auto-accept any existing envelope, then create a fresh envelope tied to the new turn and prompt metadata.
- [x] Extend `_handle_edit_applied` to lazily snapshot each tab (document copy + prior overlay), append `EditSummary` entries, and merge overlay spans per session.
- [x] Ensure the envelope tracks aggregate counts (`total_edit_count`, `total_tabs_affected`) and flips `ready_for_review` only after the controller finishes the turn.
- [x] Auto-discard the envelope when the AI turn produces no document edits.

### 3. Review controls UI & commands
- [x] Implement a reusable `DiffReviewControls` widget (nested inside `widgets/status_bar.py`) that shows Accept/Reject buttons and a summary label.
- [x] Expose `StatusBar.set_review_state(summary: str | None, *, accept_callback, reject_callback)` to toggle the controls from `MainWindow`.
- [x] Wire `_show_review_controls()` / `_hide_review_controls()` helpers in `MainWindow` to set the summary ("X edits across Y tabs") and connect callbacks.
- [x] Add keyboard/menu actions (`ai_accept_changes`, `ai_reject_changes`) routed to `_handle_accept_ai_changes` / `_handle_reject_ai_changes`, and ensure they appear under the AI menu.
- [x] Keep controls visible across tab switches by updating `_handle_active_tab_changed` to reapply the current summary.

### 4. Accept flow (clear overlays)
- [x] Implement `_handle_accept_ai_changes` to guard on `ready_for_review`, clear overlays for every existing tab session, drop the envelope, hide controls, and emit telemetry/status.
- [x] Ensure closed/orphaned tabs are skipped gracefully (log + continue) when clearing overlays.

### 5. Reject flow (rollback documents + chat)
- [x] Implement `_handle_reject_ai_changes` validation (envelope present, ready, tab versions unchanged unless forced).
- [x] For each tab session, reload the stored `DocumentState`, restore/remove overlays, and refresh its `DocumentBridge` snapshot.
- [x] Skip closed tabs but log their omission so telemetry reflects the partial rollback.
- [x] Restore the chat panel using the stored snapshot, prune tool traces introduced during the rejected turn, and drop the envelope + controls with a status message.

### 6. Lifecycle edge cases
- [x] Extend `_maybe_clear_diff_overlay` (or equivalent hooks) so any manual edit during a pending review aborts the entire envelope and hides controls.
- [x] When a tab closes during a pending turn, mark its session orphaned and ensure accept/reject ignores it.
- [x] On AI cancellation or failure, drop the envelope and restore only the composer prompt text so the user can retry immediately.
- [x] Clear pending envelopes when `ChatPanel.start_new_chat()` runs or when the workspace unloads the affected document.

### 7. Testing & validation
- [x] Add unit tests in `tests/test_chat_panel.py` for snapshot capture/restore and `consume_turn_snapshot` semantics.
- [x] Expand `tests/test_main_window.py` (or similar) to cover single-tab turn accept, multi-tab accept, and multi-tab reject flows.
- [x] Update `tests/test_widgets_status_bar.py` (or nearest equivalent) to validate the review controls widget and callbacks.
- [x] Add regression tests for lifecycle behaviors: manual edit abort, closed tab ignore, AI cancellation dropping the envelope.
- [x] Run the full test suite (via `uv run pytest` or project-standard command) and ensure lint/type checks still pass.

## Completion Tracking
- [ ] All workstreams above complete and tests green.
- [ ] Documentation (`change_control.md`, release notes if needed) updated to reflect the shipped feature.
