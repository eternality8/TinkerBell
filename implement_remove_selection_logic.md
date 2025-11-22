# Implementation Plan: Removing Legacy Selection Logic

This document converts the high-level removal plan into actionable workstreams with task checklists, dependencies, and completion criteria. Every workstream must keep SelectionRangeTool as the sole selection observer and move the platform toward chunk-first, selection-free AI workflows.

## Workstream 1: Guardrails & Ownership
**Goal:** Confine selection access to editor internals + SelectionRangeTool via a dedicated facade and enforce it through automation.

**Entry Criteria:** Repo still exposes `SelectionRange` publicly; SelectionRangeTool reads snapshots directly.

**Exit Criteria:** Only the facade + SelectionRangeTool interact with live selection data; CI fails if other modules attempt it.

**Tasks**
- [ ] Add `editor/selection_gateway.py` (or equivalent) that abstracts Qt/headless selection APIs.
- [ ] Refactor SelectionRangeTool to depend on the gateway and fetch live spans on demand.
- [ ] Remove snapshot-based selection reads from SelectionRangeTool and related helpers.
- [ ] Introduce lint/test automation that detects `SelectionRange` imports outside `editor/*` and `ai/tools/selection_range.py`.
- [ ] Document the new ownership rules inside `docs/ai_v2.md` and contributor guides.

**Dependencies:** Requires agreement on gateway surface; automation needs shared lint/test infra.

## Workstream 2: Data Model & Snapshot Cleanup
**Goal:** Remove selection data from all persisted/editor model paths.

**Entry Criteria:** `DocumentState`, undo stack, and session serialization still contain `selection`.

**Exit Criteria:** Snapshots no longer serialize selection; legacy snapshots that include the field fail loudly.

**Tasks**
- [ ] Delete `SelectionRange` usage from `DocumentState` and stop writing `selection` to snapshots.
- [ ] Remove selection persistence from `_UndoEntry` and undo/redo serialization.
- [ ] Strip selection text from `ui/document_session_service.py` save/restore payloads.
- [ ] Replace APIs consuming `(start, end)` tuples with span objects (`TextRange`, `target_span`).
- [ ] Update fixtures/tests so snapshots containing `selection` raise validation errors.

**Dependencies:** Requires editor widget containment work (Workstream 3) to avoid regressions; tests must be ready to change fixtures.

## Workstream 3: Editor Widget Containment
**Goal:** Keep selection logic private to the editor widget purely for human UX.

**Entry Criteria:** Editor exports listeners/helpers for external selection access.

**Exit Criteria:** External callers must use SelectionRangeTool; editor handles caret adjustments internally.

**Tasks**
- [ ] Remove `add_selection_listener`, `apply_selection`, and other selection exports from `editor/editor_widget.py`.
- [ ] Ensure undo/redo + patch flows adjust caret internally without persisting selections.
- [ ] Add `SelectionSnapshotProvider` callable used exclusively by SelectionRangeTool to fetch live spans + `line_offsets`.
- [ ] Validate human editing UX (cursor placement, focus retention) after stripping public APIs.

**Dependencies:** Depends on Workstream 1 (gateway) to provide the new access path; must coordinate with UI teams relying on old helpers.

## Workstream 4: Bridge & Core AI Tooling
**Goal:** Remove selection metadata from bridge plumbing, directives, CLI flags, and AI tools.

**Entry Criteria:** Bridge derives `selection_text/hash`, CLI exposes `--start/--end`, and tools/tests accept tuples.

**Exit Criteria:** Bridge only deals with chunk IDs, spans, and content hashes; CLI/tool schemas mention spans but never selection.

**Tasks**
- [ ] Delete `selection_text`, `selection_hash`, `selection_fingerprint` derivations from `services/bridge.py`.
- [ ] Remove selection fields from `EditDirective`, `DocumentEditTool`, `_PatchBridgeStub`, search/replace helpers, etc.
- [ ] Kill CLI overrides (`selection_start`, `selection_end`) and update schemas/messages accordingly.
- [ ] Update retry flows (e.g., `needs_range`) to call SelectionRangeTool or consume spans supplied by chunk manifests.
- [ ] Rebaseline bridge/tool tests to fabricate spans via chunk references or stubbed SelectionRangeTool responses only.

**Dependencies:** Requires Workstreams 1–3 to eliminate upstream selection data; ties directly into Workstreams 5 and 6 for scheduling and UI.

## Workstream 5: Subagent Planning, Chunk Heuristics, and UX Signaling
**Goal:** Rebuild subagent scheduling around chunk manifests and document complexity heuristics, while surfacing helper activity in the UI.

**Entry Criteria:** `AIController._plan_subagent_jobs` relies on selection spans and `selection_min_chars`; no UI visibility into helper runs.

**Exit Criteria:** Subagent jobs always reference chunk IDs/hashes, scheduling is heuristic-driven, and a GUI indicator shows pending/running helpers.

**Tasks**
- [ ] Remove `selection_min_chars` (and similar knobs) from controller config.
- [ ] Require `_plan_subagent_jobs` to hydrate a `DocumentChunkTool` manifest before constructing jobs.
- [ ] Implement document-complexity heuristics (chunk count thresholds, format cues, edit churn) to trigger helper passes proactively.
- [ ] Add edit debouncing + chunk-hash invalidation so only dirty chunks retrigger helpers.
- [ ] Update `SubagentJob` definitions to store chunk IDs/hashes instead of raw selection ranges.
- [ ] Introduce a status-bar badge or toast that reflects queued/running subagent summaries, wiring signals from the controller/runtime to UI.

**Dependencies:** Consumes chunk data from bridge/tooling; UI work requires coordination with `document_status_*` updates in Workstream 6.

## Workstream 6: UI & UX Surfaces
**Goal:** Remove selection summaries/previews from UI components and rewire any needed context to SelectionRangeTool outputs on demand.

**Entry Criteria:** Monitor, chat panel, dialogs, and status windows display selection data; session service stores `selection_text`.

**Exit Criteria:** UI surfaces either drop selection context or fetch spans via SelectionRangeTool lazily; new helper indicator is visible.

**Tasks**
- [ ] Update `ui/document_state_monitor.py` to stop generating selection summaries; if context is needed, call SelectionRangeTool directly.
- [ ] Remove `selection_summary` metadata from `chat/chat_panel.py` and related chat message schema.
- [ ] Strip selection previews/token counts from `widgets/dialogs.py`, `document_status_window.py`, and `document_status_service.py`; replace with span-aware info if necessary.
- [ ] Ensure `document_session_service.py` no longer persists `selection_text` during save/restore.
- [ ] Integrate the subagent activity indicator (from Workstream 5) into main window/status UI.

**Dependencies:** Requires Workstreams 1–3 to supply new SelectionRangeTool access; shares UI indicator requirement with Workstream 5.

## Workstream 7: Tests, Fixtures, and Developer Docs
**Goal:** Align verification and documentation with the new selection constraints.

**Entry Criteria:** Tests fabricate selection tuples; docs mention selection-based workflows.

**Exit Criteria:** Tests stub SelectionRangeTool/facade only; docs emphasize span requirements and chunk-first helpers.

**Tasks**
- [ ] Audit and rewrite unit/integration tests (`tests/test_ai_tools.py`, `test_document_apply_patch.py`, etc.) to stop using raw selection tuples.
- [ ] Delete helper attributes like `_PatchBridgeStub.selection`, `_SearchReplaceBridgeStub.selection`, and any stray `SelectionRange` imports.
- [ ] Update docs (`docs/ai_v2.md`, partner guides) explaining SelectionRangeTool exclusivity and span-based APIs.
- [ ] Add developer onboarding notes showing how to request spans through the facade/tool in tests.

**Dependencies:** Reflects structural changes from Workstreams 1–6; should run in parallel once APIs stabilize.

## Workstream 8: Telemetry & Observability
**Goal:** Ensure telemetry no longer reports selection-specific metrics and introduce span-focused replacements if needed.

**Entry Criteria:** Events such as `selection_snapshot_requested` emit outside SelectionRangeTool; dashboards rely on selection fields.

**Exit Criteria:** Only SelectionRangeTool emits span metrics; dashboards updated accordingly.

**Tasks**
- [ ] Remove `selection_*` fields from telemetry events produced by bridge/status windows/monitor.
- [ ] Rename/replace metrics with span-focused equivalents (e.g., `span_snapshot_requested`).
- [ ] Update dashboards/alerts to track new metrics and delete obsolete ones.
- [ ] Add tests/linters ensuring no new selection telemetry fields appear outside the allowed modules.

**Dependencies:** Telemetry updates trail other workstreams; requires coordination with observability owners.

## Cross-Workstream Coordination
- **API Freeze:** Workstreams 1–4 must stabilize new span interfaces before Workstreams 5–7 can fully migrate.
- **Testing:** Each workstream owns updates to its unit tests; Workstream 7 coordinates integration suites.
- **Rollout Strategy:** Favor feature flags for subagent heuristics (Workstream 5) and UI indicators (Workstream 6) to enable staged deployment.
- **Regression Monitoring:** After each workstream merges, run smoke tests covering AI edit flows, document load/save, undo/redo, and helper scheduling to ensure no selection references remain.
