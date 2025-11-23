# Implementation Plan: Removing Legacy Selection Logic

This document converts the high-level removal plan into actionable workstreams with task checklists, dependencies, and completion criteria. Every workstream must keep SelectionRangeTool as the sole selection observer and move the platform toward chunk-first, selection-free AI workflows.

## Workstream 1: Guardrails & Ownership
**Goal:** Confine selection access to editor internals + SelectionRangeTool via a dedicated facade and enforce it through automation.

**Entry Criteria:** Repo still exposes `SelectionRange` publicly; SelectionRangeTool reads snapshots directly.

**Exit Criteria:** Only the facade + SelectionRangeTool interact with live selection data; CI fails if other modules attempt it.

**Tasks**
- [x] Add `editor/selection_gateway.py` (or equivalent) that abstracts Qt/headless selection APIs.
- [x] Refactor SelectionRangeTool to depend on the gateway and fetch live spans on demand.
- [x] Remove snapshot-based selection reads from SelectionRangeTool and related helpers.
- [x] Introduce lint/test automation that detects `SelectionRange` imports outside `editor/*` and `ai/tools/selection_range.py`.
- [x] Document the new ownership rules inside `docs/ai_v2.md` and contributor guides.

**Status:** Completed via SelectionGateway rollout (see `selection_gateway.py`, SelectionRangeTool rewire, lint guard `tests/test_selection_guard.py`, and documentation updates in `docs/ai_v2.md`/`README.md`).

**Dependencies:** Requires agreement on gateway surface; automation needs shared lint/test infra.

## Workstream 2: Data Model & Snapshot Cleanup
**Goal:** Remove selection data from all persisted/editor model paths.

**Entry Criteria:** `DocumentState`, undo stack, and session serialization still contain `selection`.

**Exit Criteria:** Snapshots no longer serialize selection; legacy snapshots that include the field fail loudly.

**Tasks**
- [x] Delete `SelectionRange` usage from `DocumentState` and stop writing `selection` to snapshots.
- [x] Remove selection persistence from `_UndoEntry` and undo/redo serialization.
- [x] Strip selection text from `ui/document_session_service.py` save/restore payloads.
- [x] Replace APIs consuming `(start, end)` tuples with span objects (`TextRange`, `target_span`).
- [x] Update fixtures/tests so snapshots containing `selection` raise validation errors.

**Status (Nov 22, 2025):**
- DocumentState, undo/redo machinery, and snapshot serialization are fully selection-free. Legacy snapshots that include `selection` now trigger loud failures via both the bridge and session services, with tests covering the regression cases.
- Document session persistence and autosave paths are clean (see `ui/document_session_service.py` + `tests/test_document_session_service.py`).
- All bridge-level consumers now operate on `TextRange` instances instead of `(start, end)` tuples, so downstream tooling inherits span-only inputs by default. Workstream 2 is complete; future span work (CLI overrides, retry flows) will be tracked under Workstream 4.

**Dependencies:** Requires editor widget containment work (Workstream 3) to avoid regressions; tests must be ready to change fixtures.

## Workstream 3: Editor Widget Containment
**Goal:** Keep selection logic private to the editor widget purely for human UX.

**Entry Criteria:** Editor exports listeners/helpers for external selection access.

**Exit Criteria:** External callers must use SelectionRangeTool; editor handles caret adjustments internally.

**Tasks**
- [x] Remove `add_selection_listener`, `apply_selection`, and other selection exports from `editor/editor_widget.py`.
- [x] Ensure undo/redo + patch flows adjust caret internally without persisting selections.
- [x] Add `SelectionSnapshotProvider` callable used exclusively by SelectionRangeTool to fetch live spans + `line_offsets`.
- [ ] Validate human editing UX (cursor placement, focus retention) after stripping public APIs.

**Status (Nov 22, 2025):** TabbedEditor and the legacy UI main window no longer subscribe to editor selection events, and `EditorWidget` only exposes a private helper for caret adjustments. Chat suggestions now default to document-level prompts, aligning the UI with the new containment rules. Workspace bridge routers have since dropped SelectionGateway plumbing entirely, and the bridge no longer consults `editor.selection_span()`—patch flows rely on explicit range metadata and default window heuristics. Patch flows now rely on editor heuristics (`DocumentBridge` always passes `selection_hint` when a range is known and declines `preserve_selection`), and new regression tests in `tests/test_bridge.py` lock down the behavior. Remaining work is manual UX validation.

**Dependencies:** Depends on Workstream 1 (gateway) to provide the new access path; must coordinate with UI teams relying on old helpers.

## Workstream 4: Bridge & Core AI Tooling
**Goal:** Remove selection metadata from bridge plumbing, directives, CLI flags, and AI tools.

**Entry Criteria:** Bridge derives `selection_text/hash`, CLI exposes `--start/--end`, and tools/tests accept tuples.

**Exit Criteria:** Bridge only deals with chunk IDs, spans, and content hashes; CLI/tool schemas mention spans but never selection.

**Tasks**
- [x] Delete `selection_text`, `selection_hash`, `selection_fingerprint` derivations from `services/bridge.py`.
- [x] Remove selection fields from `EditDirective`, `DocumentEditTool`, `_PatchBridgeStub`, search/replace helpers, etc.
- [x] Kill CLI overrides (`selection_start`, `selection_end`) and update schemas/messages accordingly.
- [x] Update retry flows (e.g., `needs_range`) to call SelectionRangeTool or consume spans supplied by chunk manifests.
- [x] Rebaseline bridge/tool tests to fabricate spans via chunk references or stubbed SelectionRangeTool responses only.

**Status (Nov 22, 2025):** `DocumentApplyPatchTool` and `DocumentEditTool` both operate without selection metadata, telemetry now records span context instead of hashes, and the bridge no longer emits `selection_*` fields. `DocumentBridge` now prefers SelectionGateway spans when normalizing target ranges, and the new regression test in `tests/test_bridge.py` documents that contract. Manual entry points (manual `/analyze` command and UI handler in `ui/main_window.py`, plus docs in `docs/ai_v2.md`) no longer accept or advertise `--start/--end` overrides, so SelectionGateway spans remain the only path. Needs-range retries now attach `span_hint` data derived from chunk ids or SelectionRangeTool captures, and the bridge/tool test suites (`tests/test_ai_tools.py`, `tests/test_document_apply_patch.py`) now set `window`/`text_range` hints (or rely on SelectionRangeTool stubs) instead of legacy `selection` tuples. Workstream 4 now funnels remaining effort into Workstreams 5–7 (chunks-first scheduling, UI cleanup, and documentation hardening).

**Dependencies:** Requires Workstreams 1–3 to eliminate upstream selection data; ties directly into Workstreams 5 and 6 for scheduling and UI.

## Workstream 5: Subagent Planning, Chunk Heuristics, and UX Signaling
**Goal:** Rebuild subagent scheduling around chunk manifests and document complexity heuristics, while surfacing helper activity in the UI.

**Entry Criteria:** `AIController._plan_subagent_jobs` relies on selection spans and `selection_min_chars`; no UI visibility into helper runs.

**Exit Criteria:** Subagent jobs always reference chunk IDs/hashes, scheduling is heuristic-driven, and a GUI indicator shows pending/running helpers.

**Tasks**
- [x] Remove `selection_min_chars` (and similar knobs) from controller config.
- [x] Require `_plan_subagent_jobs` to hydrate a `DocumentChunkTool` manifest before constructing jobs.
- [x] Implement document-complexity heuristics (chunk count thresholds, format cues, edit churn) to trigger helper passes proactively.
- [x] Add edit debouncing + chunk-hash invalidation so only dirty chunks retrigger helpers.
- [x] Update `SubagentJob` definitions to store chunk IDs/hashes instead of raw selection ranges.
- [x] Introduce a status-bar badge or toast that reflects queued/running subagent summaries, wiring signals from the controller/runtime to UI.

**Status (Nov 22, 2025):** Workstream 5 is complete. `SubagentJob` now requires chunk IDs/document IDs and captures `chunk_hash` metadata end-to-end (controller → manager → cache/tests), so downstream tooling/telemetry no longer needs to infer ranges. Telemetry broadcasts `subagent.jobs_queued` events with chunk/reason context, and both `MainWindow` and `ui/telemetry_controller.py` track pending vs. active helper counts, display queue details in the status bar, and clear state when subagents are disabled. The status indicator now surfaces queued/running helpers with chunk previews, closing the loop between the new heuristics and the UX signaling requirement.

**Dependencies:** Consumes chunk data from bridge/tooling; UI work requires coordination with `document_status_*` updates in Workstream 6.

## Workstream 6: UI & UX Surfaces
**Goal:** Remove selection summaries/previews from UI components and rewire any needed context to SelectionRangeTool outputs on demand.

**Entry Criteria:** Monitor, chat panel, dialogs, and status windows display selection data; session service stores `selection_text`.

**Exit Criteria:** UI surfaces either drop selection context or fetch spans via SelectionRangeTool lazily; new helper indicator is visible.

**Tasks**
- [x] Update `ui/document_state_monitor.py` to stop generating selection summaries; if context is needed, call SelectionRangeTool directly.
- [x] Remove `selection_summary` metadata from `chat/chat_panel.py` and related chat message schema.
- [x] Strip selection previews/token counts from `widgets/dialogs.py`, `document_status_window.py`, and `document_status_service.py`; replace with span-aware info if necessary.
- [x] Ensure `document_session_service.py` no longer persists `selection_text` during save/restore.
- [x] Integrate the subagent activity indicator (from Workstream 5) into main window/status UI.

**Dependencies:** Requires Workstreams 1–3 to supply new SelectionRangeTool access; shares UI indicator requirement with Workstream 5.

**Status (Nov 22, 2025):** Chat, monitor, dialog, and status surfaces have dropped their selection summaries in favor of document-wide snapshots, and the export dialog now previews the entire document with the token budget gauge. `DocumentSessionService` already guards against persisting any `selection_text`, so save/restore flows rely solely on document text. Remaining workstream scope now lives in the validation/tests/doc follow-ups.

## Workstream 7: Tests, Fixtures, and Developer Docs
**Goal:** Align verification and documentation with the new selection constraints.

**Entry Criteria:** Tests fabricate selection tuples; docs mention selection-based workflows.

**Exit Criteria:** Tests stub SelectionRangeTool/facade only; docs emphasize span requirements and chunk-first helpers.

**Tasks**
- [x] Audit and rewrite unit/integration tests (`tests/test_ai_tools.py`, `test_document_apply_patch.py`, etc.) to stop using raw selection tuples.
- [x] Delete helper attributes like `_PatchBridgeStub.selection`, `_SearchReplaceBridgeStub.selection`, and any stray `SelectionRange` imports.
- [x] Update docs (`docs/ai_v2.md`, partner guides) explaining SelectionRangeTool exclusivity and span-based APIs.
- [x] Add developer onboarding notes showing how to request spans through the facade/tool in tests.

**Status (Nov 22, 2025):** `tests/test_document_apply_patch.py` and the broader AI tool/agent suites now fabricate spans via helper functions (`_selection_span()`, `_document_span()`) instead of raw tuples. `_PatchBridgeStub` and `_SearchReplaceBridgeStub` accept span dictionaries only, and `docs/ai_v2.md` documents the SelectionRangeTool-only contract plus the new testing/onboarding guidance. Workstream 7 is complete barring future regression tests.

**Dependencies:** Reflects structural changes from Workstreams 1–6; should run in parallel once APIs stabilize.

## Workstream 8: Telemetry & Observability
**Goal:** Ensure telemetry no longer reports selection-specific metrics and introduce span-focused replacements if needed.

**Entry Criteria:** Legacy selection-focused events still emitted outside SelectionRangeTool, and dashboards relied on those selection-prefixed fields.

**Exit Criteria:** Only SelectionRangeTool emits span metrics; dashboards updated accordingly.

**Tasks**
- [x] Remove `selection_*` fields from telemetry events produced by bridge/status windows/monitor.
- [x] Rename/replace metrics with span-focused equivalents (e.g., `span_snapshot_requested`).
- [x] Update dashboards/alerts to track new metrics and delete obsolete ones.
- [x] Add tests/linters ensuring no new selection telemetry fields appear outside the allowed modules.

**Status (Nov 24, 2025):** DocumentApplyPatch/DocumentEdit/bridge telemetry now reports `snapshot_span`, chunk manifests expose `span_overlap`, and SelectionRangeTool emits `span_snapshot_requested` with `span_length`. Docs/dashboards were refreshed to the new names, and `tests/test_selection_telemetry_guard.py` enforces that legacy `selection_*` telemetry identifiers never reappear in source or docs.

**Dependencies:** Telemetry updates trail other workstreams; requires coordination with observability owners.

## Workstream 9: Chunk/Ranged Runtime Rewrite
**Goal:** Eliminate residual selection/caret assumptions from bridge, tooling, controller, and prompts so every AI path operates on chunk manifests or explicit `target_span` data supplied by tool calls.

**Entry Criteria:** `DocumentBridge.generate_snapshot()` still centers windows on live selection spans, AI tools ingest/emit `analysis_selection`, controller heuristics (subagent planning, needs-range retries, advisor overrides) read selection lengths, and prompts/docs instruct agents to “work on the highlighted selection.”

**Exit Criteria:** Snapshots, controller payloads, and tool schemas rely solely on chunk metadata or explicit ranges provided in the request; `analysis_selection` and other implicit selection fields are removed, and prompts/docs describe chunk-first workflows only.

**Tasks**
- [x] Update `services/bridge.py` and snapshot providers so window defaults, chunk manifests, and retry hints derive from chunk stats or explicit tool-supplied ranges instead of `_active_selection_span()`; drop caret fallbacks entirely.
- [x] Remove `analysis_selection` from AI tool schemas (`document_edit`, `document_apply_patch`, `search_replace`, `tool_usage_advisor`) and replace with required `target_span`/chunk references sourced from SelectionRangeTool or DocumentChunkTool results.
- [x] Refactor `AIController` planning, needs-range retries, and telemetry to consume chunk manifests and `target_span` payloads, deleting `selection_min_chars` and any span overrides that hinge on live selections.
- [x] Rewrite planner/tool-loop prompts plus `docs/ai_v2.md`, `README.md`, and partner guides to emphasize chunk windows + explicit spans; remove guidance that mentions “current selection.”
- [x] Update unit/integration tests, benchmarks, and debug scripts to fabricate chunk/range inputs exclusively, ensuring guard tests fail if `analysis_selection` or caret fallbacks reappear.

**Status:** In progress (Dec 2025). DocumentBridge now ignores live selections entirely—snapshot windows clamp to chunk caps, chunk manifests mark overlap using the requested window span, and router/UI wiring no longer inject SelectionGateway instances. AI tools/tests now consume explicit `window`/`snapshot_span` data instead of `analysis_selection`, and the controller no longer exposes `selection_min_chars`, derives subagent focus from snapshot spans/windows, standardizes telemetry on `span_start/span_end`, and sources `needs_range` hints from chunk manifests or snapshot spans before touching SelectionRangeTool. Planner/tool-loop prompts (`src/tinkerbell/ai/prompts.py`) plus `docs/ai_v2.md`, `README.md`, and partner-facing wiring notes now describe the span-first workflow, so Workstream 9 shifts to validation/cleanup mode.

**Dependencies:** Requires stabilized chunk manifest APIs (Workstreams 4–5) and completed telemetry/docs cleanup (Workstreams 7–8) so new contracts propagate consistently.

## Workstream 10: Range-Only AI Logic Purge
**Goal:** Delete the remaining selection/caret fallback logic from AI tools, controller plumbing, prompts, and schemas so every decision or edit operates either on full-document chunks or on explicit spans supplied with the tool call.

**Entry Criteria:** `DocumentEditTool`, `DocumentApplyPatchTool`, `SearchReplaceTool`, and `ToolUsageAdvisorTool` still read `snapshot_span` / `selection_start/end`; `selection_utils.py` remains a shared helper; controller/analysis models expose `selection_length`; prompts, schemas, and telemetry still mention “selection” or caret handling.

**Exit Criteria:** All AI flows require explicit `target_span`/chunk references; `_selection_span`/`resolve_snapshot_span` helpers disappear from the AI stack; controller/analysis models drop selection fields entirely; prompts/docs/tool schemas describe chunk/range inputs only; telemetry no longer emits caret terminology outside SelectionRangeTool.

**Tasks**
- [x] Refactor `DocumentEditTool`/`DocumentApplyPatchTool` to reject implicit `selection` data, remove `_selection_span`/`_resolve_snapshot_span`, and require callers to provide explicit `target_span`/chunk metadata (falling back to chunk manifests only).
- [x] Delete `selection_utils.py` and replace call sites with range data derived from snapshot windows, chunk manifests, or explicit tool arguments.
- [x] Update `SearchReplaceTool`, `tool_usage_advisor`, and tool registry schemas so scopes are determined by caller-supplied spans/chunk references instead of live selections; enforce this via schema validation. *(ToolUsageAdvisor now accepts `target_range` and exposes the span schema via tool registration; SearchReplace work landed Nov 22, 2025.)*
- [x] Strip `selection_start`, `selection_end`, and `selection_length` from `AnalysisInput`, controller state, telemetry payloads, and planner/router heuristics—use document length + requested window/chunk metadata instead.
- [x] Rewrite planner/tool prompts plus `chat/commands.py` normalization to remove caret/selection wording, ensuring CLI errors/telemetry reference spans/chunks; update associated unit tests (`tests/test_ai_tools.py`, `tests/test_ai_controller.py`, `tests/test_analysis_agent.py`, etc.) to cover the new invariants.

**Status (Dec 2025):** Complete. `DocumentEditTool`, `DocumentApplyPatchTool`, and `SearchReplaceTool` now rely solely on caller-provided spans or snapshot windows, `selection_utils.py` has been removed in favor of inline range helpers, and ToolUsageAdvisor/controller schemas emit span terminology end-to-end (including AnalysisInput, telemetry, and tests). CLI validation rejects `selection` payloads in favor of explicit `target_range` data, error messaging now instructs agents to capture spans from DocumentSnapshot/chunk manifests/SelectionRangeTool, and planner/tool-loop prompts reference insert operations without invoking caret wording. Companion tests (`tests/test_chat_commands.py`) lock down the new behavior, so Workstream 10 is ready for sign-off.

**Dependencies:** Builds on Workstreams 4 and 9 to keep spans available everywhere and requires telemetry/doc baselines from Workstreams 7–8. Completion unblocks the final deletion of SelectionRangeTool fallbacks from AI workflows.

## Workstream 11: Chunk-Or-Explicit-Range Enforcement
**Goal:** Ensure every AI edit/mutation flow operates either on full-chunk payloads (verified via chunk manifests/cache keys) or on explicit ranges passed in the tool call, with no implicit fallbacks to snapshot-derived spans or live selections.

**Entry Criteria:** Some auto-convert/helpers still infer scopes from snapshot windows when tool calls omit explicit spans; streamed diff payloads do not record where the scope originated; controller retries rely on heuristics to reconstruct ranges; telemetry cannot prove edits were chunk/range backed.

**Exit Criteria:**
- Tool schemas require `target_span`/`target_range` or chunk manifest references (cache key + chunk id/hash) unless `scope="document"` is declared, and validation errors are surfaced otherwise.
- Patch/apply flows annotate every edit with a `scope_origin` (`chunk`, `explicit_span`, `document`) plus provenance metadata and refuse to execute when metadata is missing/mismatched.
- Controller orchestration, retries, and helper scheduling persist the provided scope metadata through every hop and only fall back to SelectionRangeTool when explicitly authorized.
- Telemetry/events/tests assert that each edit/tool call reports `scope_origin` and `scope_length`, preventing regressions to selection-centric logic.

**Tasks**
- [x] Harden tool schemas (`DocumentApplyPatchTool`, `DocumentEditTool`, `SearchReplaceTool`, `ToolUsageAdvisor`, CLI adapters) so callers must provide either explicit spans or chunk references, with `scope="document"` as the only implicit alternative.
- [x] Update diff/patch builders to attach `source_scope` metadata to streamed ranges and verify chunk IDs/chunk hashes or span offsets before applying changes.
- [x] Extend `DocumentBridge.generate_snapshot()` + apply flows to emit provenance metadata (chunk cache key, chunk hash, normalized span) and fail fast when edits omit it; add regression tests for malformed scopes.
- [x] Thread scope provenance through `AIController` planning, retry, and telemetry paths so `needs_range`/helper logic short-circuits when scope metadata becomes invalid; SelectionRangeTool should only run under explicit controller authorization.
- [x] Instrument telemetry (`patch.apply`, `document_edit`, controller tool traces, and context usage exports) with `scope_origin`/`scope_length` and add guard tests/linters to enforce their presence.
 - [x] Update prompts/docs (`docs/ai_v2.md`, partner guides) with chunk-or-explicit-range requirements, including troubleshooting for invalid scope errors and sample payloads for both flows.

**Status (Nov 23, 2025):** Tool-side enforcement remains solid, and DocumentBridge now *enforces* provenance instead of passively ingesting it. `services/bridge.py` validates every streamed range via `_range_payload`, rejects missing/invalid `scope.origin`/`scope_range`, normalizes scope summaries, and injects `scope_origin/scope_length/scope_range` into `patch.apply`, `hash_mismatch`, and failure telemetry. `tests/test_bridge.py` adds helper-backed range payloads plus assertions that both success/failure paths surface the new metadata. On the controller side, needs-range retries now prefer tool-supplied scope metadata (avoiding SelectionRangeTool fallbacks when provenance exists), propagate `scope_origin/scope_length/scope_range` into the user-facing hint payload, and stamp `document_edit.retry` telemetry with the same summary so downstream dashboards can prove chunk-or-range coverage. Prompts plus partner docs (`src/tinkerbell/ai/prompts.py`, `docs/ai_v2.md`, `docs/operations/telemetry.md`) now spell out chunk-or-explicit-range provenance requirements and troubleshooting steps. Controller planning + telemetry also aggregate scope coverage per turn (`scope_origin_counts`, `scope_missing_count`, `scope_total_length`) so Workstream dashboards can prove chunk-first compliance without scraping raw tool traces. Remaining work lives in any follow-up guardrails beyond the aggregated metrics.

**Dependencies:** Builds on Workstreams 4, 9, and 10 for finalized span/chunk schemas and requires telemetry plumbing from Workstream 8. Schema hardening should land before controller retry updates so validation errors surface early.

## Workstream 12: Final Caret-Free Cleanup
**Goal:** Close the remaining gaps by enforcing caret preservation in bridge flows, deleting legacy helpers, and scrubbing selection/caret wording from agent-facing contracts.

**Entry Criteria:** `DocumentBridge` still applies patches with `preserve_selection=False`; `ai/selection_utils.py` lingers in the repo; tool/agent messaging (tool registry, DocumentEdit/DocumentApplyPatch errors, controller retries, planner graph metadata) still mentions “selection”/“caret” even though the architecture is span-only.

**Exit Criteria:** Bridge always preserves the user caret during AI edits, the selection-utils helper is removed (with guard tests ensuring it stays gone), and all runtime/documentation strings outside SelectionRangeTool speak exclusively about spans/chunks.

**Tasks**
- [x] Flip `DocumentBridge._apply_patch_directive` (and associated tests) to pass `preserve_selection=True`, ensuring AI edits never collapse or move the live caret.
- [x] Delete `src/tinkerbell/ai/selection_utils.py`, drop the module from imports, and add a regression guard (e.g., selection guard test) to prevent it from reappearing.
- [x] Update tool/agent copy (`ai/tools/registry.py`, `ai/tools/document_edit.py`, `ai/tools/document_apply_patch.py`, controller retry guidance, planner graph metadata, docs) so guidance references spans/chunk manifests instead of selections/caret inserts.
- [x] Extend unit tests (`tests/test_bridge.py`, `tests/test_ai_tools.py`, controller/chat command suites) to assert the new wording/behavior and cover the caret-preservation regression.

**Status (Nov 23, 2025):** Bridge apply flows now preserve the caret by default, `selection_utils.py` is fully removed (guarded by `tests/test_selection_guard.py`), span-first wording replaced the last caret/selection references across tool registries, error messaging, and docs, and new regression tests in `tests/test_ai_tools.py` lock down the updated copy. Workstream 12 is now ready for sign-off once partner docs absorb the same language.

**Dependencies:** Relies on Workstreams 3–4 (bridge/tool plumbing) and Workstreams 9–11 (span/chunk runtime + enforcement) so the cleanup remains purely textual/behavioral.

## Cross-Workstream Coordination
- **API Freeze:** Workstreams 1–4 must stabilize new span interfaces before Workstreams 5–7 can fully migrate.
- **Testing:** Each workstream owns updates to its unit tests; Workstream 7 coordinates integration suites.
- **Rollout Strategy:** Favor feature flags for subagent heuristics (Workstream 5) and UI indicators (Workstream 6) to enable staged deployment.
- **Regression Monitoring:** After each workstream merges, run smoke tests covering AI edit flows, document load/save, undo/redo, and helper scheduling to ensure no selection references remain.
