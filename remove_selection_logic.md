# Removing Legacy Selection Logic

## Objective
- Enforce the constraint that the **SelectionRangeTool** is the *only* component allowed to observe or interact with editor selections.
- Delete historic selection plumbing (state, text snapshots, telemetry, CLI flags, UI affordances) so AI edits never couple to caret position again.
- Keep human-editing UX functional by containing any unavoidable selection handling inside the editor widget and the SelectionRangeTool data path.

## Current Usage Inventory (non-exhaustive code refs)

### Core data model & snapshots
- `src/tinkerbell/editor/document_model.py` defines `SelectionRange` and serializes `selection` inside every document snapshot.
- `src/tinkerbell/editor/editor_widget.py` keeps `SelectionRange` in `_state`, exposes `add_selection_listener`, restores selections during patches, and records them inside undo entries.
- Snapshot consumers (`ui/document_session_service.py`, `services/bridge.py`, `ui/document_status_service.py`, etc.) expect `selection` to exist and contain tuple/dict payloads.

### Bridge + AI controllers
- `src/tinkerbell/services/bridge.py` derives `selection_text`, `selection_hash`, `selection_fingerprint`, scopes diff windows to the selection, and forwards selection metadata to AI tools.
- `src/tinkerbell/chat/message_model.py` and `chat/commands.py` expose CLI overrides (`--start/--end`) and schema fields such as `selection_fingerprint`.
- Multiple AI tools/tests (`tests/test_ai_tools.py`, `tests/test_document_apply_patch.py`, `tests/test_search_replace...`) fabricate selections directly instead of calling the SelectionRangeTool.

### UI & UX surfaces
- `ui/document_state_monitor.py`, `ui/main_window.py`, `chat/chat_panel.py`, and `widgets/dialogs.py` build selection summaries, previews, and token counts.
- `ui/document_status_window.py` and `ui/document_status_service.py` render selection spans inside telemetry/status panes.
- `ui/document_session_service.py` persists `selection_text` for save prompts and session restore.

### Telemetry + tests
- Telemetry events (`span_snapshot_requested`, `span_overlap`, etc.) are emitted from non-tool code paths (bridge, status window, monitor).
- Every editor/bridge/unit test encodes assumptions about `SelectionRange` living on `DocumentState`.

These usages must be deleted or rewritten so that the only surviving selection touchpoint is the SelectionRangeTool querying the live editor state.

## Required Work

### Guardrails & Ownership
1. Introduce a lightweight `SelectionFacade` (e.g., `editor/selection_gateway.py`) that wraps the Qt/headless editor selection APIs.
2. Update SelectionRangeTool to depend on the facade instead of pulling `selection` out of document snapshots.
3. Add automation (lint/test) that fails whenever modules outside `editor/` + `ai/tools/selection_range.py` import `SelectionRange` or read `selection` snapshots.

### Data Model & Snapshot Cleanup
1. Remove `SelectionRange` from `DocumentState` and stop serializing `selection` inside `DocumentState.snapshot()`.
2. Delete selection persistence from `_UndoEntry`, session serialization (`ui/document_session_service.py`), and restore flows.
3. Replace APIs that accept `(start, end)` tuples with span objects (`documents/ranges.TextRange`) or explicit `target_span` dicts.
4. Update `DocumentState` tests/fixtures so any legacy snapshot containing `selection` fails fast.

### Editor Widget Containment
1. Keep selection tracking private inside `editor/editor_widget.py` solely for human editing convenience.
2. Remove exported helpers (`add_selection_listener`, `apply_selection`, `SelectionRange` access) so external callers must invoke SelectionRangeTool for span data.
3. Ensure undo/redo, patch application, and AI edit flows no longer persist or restore selections; they should adjust the caret internally and expose nothing outward.
4. Provide a `SelectionSnapshotProvider` that SelectionRangeTool can call to read the live selection + `line_offsets` without sharing that data elsewhere.

### Bridge & AI Controller Changes
1. Strip `selection_text`, `selection_hash`, and `selection_fingerprint` handling from `services/bridge.py`; replace anchoring logic with `target_span` + `content_hash` validation.
2. Remove `selection` fields from `EditDirective`, `DocumentEditTool`, `_PatchBridgeStub`, search/replace helpers, and any tests relying on tuples.
3. Delete CLI overrides and schema params like `selection_start`/`selection_end`; callers who need ranges must call SelectionRangeTool first.
4. Rewrite `AIController._plan_subagent_jobs` (and related config like `selection_min_chars`) so planning never inspects snapshot selections—require controllers to call SelectionRangeTool or use chunk manifests to define spans before queueing helper jobs.
5. Enforce chunk-first subagent workflows: planning must always hydrate or fetch a manifest-backed `DocumentChunkTool` slice, drop `selection_min_chars`, and ensure `SubagentJob` instances reference real chunk ids/hashes instead of ad-hoc selection ranges.
6. Replace selection-triggered scheduling with document-complexity heuristics (chunk count, format cues, edit churn) that proactively queue helper passes when a document warrants summaries, debounce reruns to avoid thrashing, and invalidate only the chunks whose hashes changed.
7. Update retry logic (e.g., handling `needs_range`) to fetch spans via SelectionRangeTool instead of cached tuples.

### UI & UX Surfaces
1. `document_state_monitor.py`: remove selection-summary generation; if editor context is still needed, derive it from the latest SelectionRangeTool response.
2. `chat/chat_panel.py`: drop `selection_summary` metadata; rely on explicit prompts or span descriptions.
3. `widgets/dialogs.py` (save/export) and `document_status_*` modules: remove selection previews, token counts, and overlap displays, or rewire them to use SelectionRangeTool output on demand.
5. Add a clear GUI indicator (status bar badge/toast) showing when subagent summaries are running or queued so users understand background analysis activity.
4. `document_session_service.py`: stop persisting `selection_text`; resume flows restore document text only.

### Tests, Fixtures, and Docs
1. Rewrite unit/integration tests to stop fabricating `selection` tuples; instead, stub the SelectionRangeTool/facade when spans are needed.
2. Delete helper attributes like `_PatchBridgeStub.selection`, `_SearchReplaceBridgeStub.selection`, and any `SelectionRange` imports outside the editor/tool packages.
3. Update docs (`docs/ai_v2.md`, partner guides) to state unequivocally that SelectionRangeTool is the sole selection API and that spans/replace-all semantics are mandatory elsewhere.

### Telemetry & Observability
1. Ensure only SelectionRangeTool emits `span_snapshot_requested`; remove `selection_*` fields from other events (bridge chunk metrics, status windows, etc.).
2. Introduce replacement metrics (e.g., `span_snapshot_requested`) if observability around span fetches is still required.
3. Purge dashboards/alerts tied to the deleted telemetry fields.

## Open Implementation Questions
- **Editor UX**: confirm whether collapsing selections after edits is still required for human workflows; if yes, keep that logic private to `editor_widget` only.
- **SelectionRangeTool inputs**: decide whether the tool should read directly from the editor widget or through a gateway so headless/test runs remain deterministic.

Completing the tasks above will confine selection awareness to SelectionRangeTool + its immediate adapter and eliminate the scattered, error-prone selection logic across the app.
