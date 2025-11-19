# AI v2 Phase 5 Implementation Plan

_Last updated: 19 Nov 2025_

## Purpose & scope
Phase 5 ships the remaining large-document capabilities on top of the Phase 0–4 runtime (token policy, outline/retrieval guardrails, subagent + plot scaffolding). This document converts the roadmap bullets from `ai_v2_plan.md` into execution-ready workstreams with concrete tasks, file touchpoints, telemetry hooks, and validation requirements.

## Guiding assumptions
- Token accounting, outline/retrieval tooling, and subagent scaffolding from earlier phases are stable on the `AI_v2` branch.
- LangGraph agent flows already honor pointerized payloads; Phase 5 work must keep pointer semantics intact.
- Feature flags remain available for gradual rollout (`settings.experimental.*`, CLI/env mirrors).
- Large fixture set under `test_data/` plus benchmark harnesses in `benchmarks/` can be extended but not removed.

## Workstreams & task lists

### 1. Windowed snapshots + chunk manifests
**Objective:** Make slim, selection-scoped snapshots the default without breaking current callers.

**Key components:** `tinkerbell.services.bridge.DocumentBridge`, `tinkerbell.ai.tools.document_snapshot`, `tinkerbell.ai.prompts`, `tinkerbell.chat.commands`, new chunk manifest schema shared with `AIController`.

**Tasks**
1. **Schema design & prompts** ✅
   - Draft manifest shape (`window`, `chunk_profile`, offsets, outline pointer IDs) and document it in `docs/ai_v2.md`.
   - Update chat command schemas so tool calls can specify window constraints; add migration notes for existing tests.
2. **Bridge + tool implementation** ✅
   - Extend `DocumentBridge.generate_snapshot()` with optional `window`, `max_tokens`, `chunk_profile`, `include_text`, `delta_only` args; ensure defaults keep legacy behavior.
   - Cache recent chunk manifests (per tab) and return them with the snapshot payload.
   - Update `DocumentSnapshotTool` to pass through new params, include manifest metadata, and expose sane defaults (selection ± X chars).
3. **Controller & prompt integration** ✅
   - Teach `AIController` planners to request windowed snapshots first; fallback to full snapshots only behind explicit budget decisions.
   - Adjust system prompts to describe the new manifest + chunk flow.
4. **Validation** ✅
   - Add `tests/test_document_snapshot.py` covering parameter combinations + manifest caching.
   - Expand `tests/test_agent.py` scenarios ensuring controller prefers slim snapshots and emits guardrail hints when falling back.
   - Update docs + release notes with operator-facing guidance.

### 2. Chunk registry + `DocumentChunkTool`
**Objective:** Provide on-demand chunk fetches keyed by hashed descriptors.

**Key components:** new `tinkerbell.ai.memory.chunk_index`, `tinkerbell.ai.tools.document_chunk`, updates to `AIController`, `SubagentRuntimeManager`, telemetry.

**Tasks**
1. **Chunk index service**
   - ✅ Implemented `ChunkIndex` with cache-bus subscriptions, manifest ingestion, window metadata storage, iterator helpers, and automatic evictions on document change/close events.
   - ✅ `AIController.ensure_chunk_index()` now owns a per-runtime instance reused by tools and manifests.
2. **Tool surface + settings**
   - ✅ `DocumentChunkTool` ships with inline-token caps, iterator cursors, and pointer-only fallbacks plus `chunk_cache.hit/miss` + `chunk_tool.window_tokens` telemetry events.
   - ✅ Settings/runtime expose `chunk_profile`, `chunk_overlap_chars`, `chunk_max_inline_tokens`, and `chunk_iterator_limit`, all routed through `AIController.configure_chunking()`.
3. **Runtime integration** ✅
   - ✅ Wired `AIController` and subagent runtime to use chunk fetches when manifests indicate missing context (manifest ingestion + hydration helpers in `controller.py`).
   - ✅ Added telemetry events (`chunk_cache.hit/miss`, `chunk_tool.window_tokens`) and ensured pointer-only responses follow the same budget accounting as other pointerized payloads.
4. **Validation** ✅
   - ✅ Added `tests/test_chunk_index.py`, `tests/test_document_chunk_tool.py`, and extended `tests/test_ai_tools.py` to cover manifest ingestion, iterator semantics, and registry wiring.
   - ✅ Added controller regression coverage in `tests/test_agent.py` ensuring chunk-first planning ingests manifests and hydrates chunk references.

### 3. Streaming diff + multi-range patch flow ✅
**Objective:** Avoid full-buffer diffs by operating on chunk streams.

**Key components:** new `StreamedDiffBuilder` (under `tinkerbell.ai.tools.diff_builder`), updates to `DocumentApplyPatchTool`, `tinkerbell.services.bridge.DocumentBridge`, `editor.patches`.

**Tasks**
1. **Diff builder abstraction** ✅
   - ✅ `StreamedDiffBuilder`, `StreamedEditRequest`, and telemetry stats power range-based patches without buffering the full doc while keeping chunk metadata intact for traces.
2. **Patch/tool updates** ✅
   - ✅ `DocumentApplyPatchTool` now accepts `patches=` (multi-range) payloads, emits `diff.streamed` telemetry, and falls back to unified diffs for legacy callers.
   - ✅ Added `RangePatch` + `apply_streamed_ranges` to `editor.patches` alongside the legacy `apply_unified_diff` path.
3. **Bridge + cache bus integration** ✅
   - ✅ `DocumentBridge.queue_edit()` ingests streamed ranges, enforces match-text validation, publishes span metadata, and keeps the old diff flow for rollback.
4. **Validation** ✅
   - ✅ Regression coverage via new `tests/test_diff_builder.py`, `tests/test_document_apply_patch.py`, and expanded `tests/test_patches.py`, `tests/test_bridge.py`, `tests/test_ai_tools.py`.
   - ✅ Telemetry assertions ensure streamed diffs report range counts + bytes touched for dashboards.

### 3b. Diff tooling reliability fixes (parallel to Workstream 3)
Even after streaming diffs ship, we must harden the existing patch/edit path so legacy agents and UI actions stop corrupting text. The following tasks come directly from `ai_fixes.md` and should be executed alongside Workstream 3; treat them as a co-equal workstream rather than a follow-up.

#### 3b.1 Snapshot-anchored range resolution — ✅ landed 19 Nov
- `DocumentApplyPatchTool` and `DocumentEditTool` now accept `match_text`/`expected_text` plus `selection_fingerprint`, enforce anchor uniqueness, and refuse stale ranges with refresh guidance (see `tests/test_document_apply_patch.py`).
- Follow-up: monitor telemetry for anchor failure rates once counters from 3b.5 go live.

#### 3b.2 Inline edit auto-conversion safety — ✅ landed 19 Nov
- `_auto_convert_to_patch` mirrors anchoring validation, blocking bogus diffs when snapshots drift; guarded by new cases in `tests/test_ai_tools.py`.
- Follow-up: keep prompts aligned with the “always send anchors” rule (tracked under 3b.4).

#### 3b.3 Guardrails on implicit insertions — ✅ landed 19 Nov
- Replace operations now require explicit ranges or anchors, caret inserts stay scoped to explicit `insert` intents, and user-facing errors explain how to recover.
- Covered via regression updates in `tests/test_document_apply_patch.py` / `tests/test_ai_tools.py` plus bridge-level assertions.

#### 3b.4 Tool schema & instruction updates — ✅ landed 19 Nov
- System + tool prompts in `tinkerbell/ai/prompts.py` now spell out anchor requirements, copying `selection_text`/`selection_hash`, and the retry path when hashes drift.
- `docs/ai_v2.md` gained the "Snapshot-anchored editing guardrails" section with workflow tables, telemetry callouts, and migration guidance for agent authors.
- Tool manifests (`chat.commands`, `message_model`) remain the single source of truth for parameter schemas; prompts/docs now mirror them so agents and operators stay aligned.

#### 3b.5 Testing & telemetry — ✅ landed 19 Nov
- `DocumentApplyPatchTool`, `DocumentEditTool`, and `DocumentBridge` emit `patch.anchor` + `patch.apply` telemetry for success/conflict/stale paths; counters surface in dashboards.
- Editor widget now rejects zero-length replace directives, reproducing the historical failure locally and blocking it in production (`tinkerbell/editor/editor_widget.py`).
- Regression suite expanded: `tests/test_ai_tools.py`, `tests/test_bridge.py`, and `tests/test_editor_widget.py` assert telemetry payloads, anchor enforcement, and UI guardrails.

### 4. Prompt + telemetry guardrails for selective reads
**Objective:** Ensure the agent and operators stay on the chunk-first path.

**Key components:** `tinkerbell.ai.prompts`, `AIController` guardrail hints, `tinkerbell.ai.services.telemetry`, status bar/chat UI badges, docs.

**Tasks**
1. **Prompt refresh** ✅
   - System + tool prompts in `tinkerbell/ai/prompts.py` now explicitly require the "snapshot → chunk tool → outline" order and describe manifest hydration rules.
2. **Guardrail enforcement** ✅
   - `_ChunkFlowTracker` emits guardrail hints + `chunk_flow.*` telemetry whenever a full snapshot slips through, and the chat panel/status bar surface badges (`Chunk Flow Warning` / `Recovered`) so operators can react immediately.
3. **Telemetry + docs** ✅
   - `chunk_flow.requested`, `chunk_flow.escaped_full_snapshot`, and `chunk_flow.retry_success` events flow through the telemetry bus, with docs/readme updates explaining recovery steps.
4. **Validation** ✅
   - Added regression coverage: `tests/test_chat_panel.py`, `tests/test_widgets_status_bar.py`, and the new `tests/test_telemetry_controller.py` assert the badges + telemetry wiring; the existing controller tests already cover guardrail hint injection.

### 5. Character/entity concordance automation
**Objective:** Finish the entity pipeline so the agent can plan character-wide edits safely.

**Key components:** new `CharacterMapStore` (under `tinkerbell.ai.memory`), `CharacterMapTool`, `CharacterEditPlannerTool`, UI affordances, telemetry.

**Tasks**
1. **Store + pipeline** ✅
   - `CharacterMapStore` now records aliases/pronouns, exemplar mentions, cache-bus invalidations, and planner progress (`PlannerTaskProgress`).
2. **Tooling + planners** ✅
   - `CharacterMapTool` + `CharacterEditPlannerTool` are registered behind the plot-scaffolding flag, exposed via `ToolProvider`, and wired into `SubagentRuntimeManager`.
3. **UI & UX (delegated)** ✅
   - Scope now lives under Workstream 8’s Document Status console; WS5 deliverables are complete once data/modeling pieces land.
4. **Validation** ✅ (tooling)
   - Added `tests/test_character_map_store.py`, `tests/test_character_map_tool.py`, `tests/test_character_edit_planner_tool.py`, and extended `tests/test_tool_provider.py`; planner telemetry + UI tests remain once surfaces ship.

### 6. Storyline continuity orchestration
**Objective:** Upgrade plot scaffolding into a structured plot memory with enforced edit loops.

**Key components:** `PlotStateMemory` (timeline graphs, dependency tracking), `PlotOutlineTool`, `PlotStateUpdateTool`, `SubagentRuntimeManager`, `AIController`, persistence options for overrides.

**Tasks**
1. **Memory upgrade** ✅ (landed)
   - `PlotStateMemory` (`src/tinkerbell/ai/memory/plot_memory.py`) now tracks arcs, beats, dependencies, version metadata, and human overrides with cache-bus clears + cap enforcement.
   - `PlotOverrideStore` persists manual directives under `~/.tinkerbell/plot_overrides.json`, and overrides rehydrate at startup.
2. **Tool suite** ✅
   - `PlotOutlineTool` (alias `DocumentPlotStateTool`) now exposes enriched snapshots; `PlotStateUpdateTool` captures manual dependencies/overrides, both registered via `ui.tools.provider` + `ai.tools.registry` and available to subagents.
   - `_PlotLoopTracker` inside `AIController` enforces the "outline → edit → update" loop with guardrail hints, automatically activating when `SubagentRuntimeConfig.plot_scaffolding_enabled` is `True`. New controller tests (`tests/test_agent.py::test_plot_loop_*`) cover the block/allow/update-reminder paths.
3. **Telemetry/persistence** ✅
   - `plot_state.read`/`plot_state.write` telemetry events now fire from the outline/update tools; persistence of overrides validated via `PlotOverrideStore` round-trips.
   - `docs/ai_v2.md` + `docs/ai_v2_release_notes.md` now describe the telemetry fields, override file path, and operator workflow.
4. **Validation** ✅
   - `tests/test_plot_state.py`, `tests/test_document_plot_state_tool.py`, `tests/test_tool_provider.py`, and the new guardrail coverage in `tests/test_agent.py` exercise the memory store, tool factories, and controller enforcement.
   - Telemetry + persistence documented; release-note entry added under Phase 5.1.

### 7. Preflight analysis & tool recommendations
**Objective:** Inject document-aware planning before each run so the agent automatically chooses the right tool mix and operators see why.

**Key components:** new `tinkerbell.ai.analysis` package, `AnalysisAgent`, controller hooks (`AIController._build_messages`), `ToolUsageAdvisorTool`, chat panel/status-bar surfacing, telemetry additions.

**Tasks**
1. **Analysis module** — ✅ code + cache invalidation landed 19 Nov
   - `tinkerbell/ai/analysis/` now exists with `models.py`, `sources.py`, `rules.py`, `cache.py`, and `agent.py`; `AnalysisAgent` already powers controller preflight runs.
   - Inputs capture document metadata, chunk manifest hints, guardrail flags, plot/concordance freshness, and produce advice with `chunk_profile`, tool lists, outline refresh flags, and `rule_trace` telemetry.
   - Cache now subscribes to the document bus via `AIController`, so `DocumentChanged`/`DocumentClosed` events immediately invalidate advice + snapshot caches.
2. **Rule engine + telemetry plumbing** — ✅ analysis + override events wired
   - Base rule set (chunk profile, outline freshness, plot/concordance, retrieval) is active, emitting `AnalysisFinding` traces recorded inside `AnalysisAdvice`.
   - `AnalysisAgent` emits `analysis.preflight.*` plus the new `analysis.advisor_tool.invoked` / `analysis.ui_override.*` events. Advice metadata flows through `TelemetryManager`, `ContextUsageEvent`, and `scripts/export_context_usage.py`, and the docs/release notes describe the new columns.
3. **Controller + tool integration** — ✅ controller + tool + telemetry context complete (docs/tests follow-up)
   - `AIController` owns `_analysis_agent`, `configure_analysis()`, snapshot caching, `_run_preflight_analysis()`, `get_latest_analysis_advice()`, and injects a “Preflight analysis summary” system message ahead of prompts.
   - Advice objects (including cache state, tool lists, warnings, rule traces, timestamps) now flow into manual commands, telemetry contexts, and the `ToolUsageAdvisorTool` entry point, so dashboards/exports receive structured data.
4. **UI & operator transparency** — ✅ surfaced 19 Nov (status/chat badges + `/analyze`; docs/tests remain)
   - Status bar gains an analysis indicator, the chat panel shows a preflight badge + hover detail, TelemetryController refreshes them after each turn, and the `/analyze` manual command displays formatted advice or errors.
5. **Validation & docs** — ✅ coverage & docs updated
   - `tests/test_analysis_agent.py`, `tests/test_agent.py` (analysis hint, cache invalidation, telemetry emission), and `tests/test_ai_tools.py` lock down analyzer/tool behavior; follow-on widget tests remain optional.
   - `docs/ai_v2.md` and `docs/ai_v2_release_notes.md` now document the analyzer workflow, UI badges, manual `/analyze` command, and telemetry/export fields so ops/support can reference them.

### 8. Document status console & UX surfacing
**Objective:** Provide a unified "Document Status" window that exposes chunk manifests, plot/concordance summaries, planner progress, and telemetry state so operators can inspect readiness before edits.

**Key components:** new `DocumentStatusWindow` (Qt dialog) under `src/tinkerbell/ui/widgets/`, data adapters pulling from `OutlineRuntime`, `CharacterMapStore`, `DocumentPlotStateStore`, planner telemetry, and status bar/command palette hooks to launch the window.

**Tasks**
1. **Design & scaffolding**
   - ✅ Dialog now ships with a severity-aware header, metadata grid (path, selection, version, language), doc selector, and dedicated tabs for Chunks, Outline, Plot & Concordance, and Telemetry—including Copy/Save actions and manual refresh.
   - ✅ `DocumentStatusService` (in `tinkerbell/ui/document_status_service.py`) assembles payloads from controller caches, outline/plot/concordance stores, planner stats, and telemetry so the window stays data-driven.
2. **Data bindings**
   - ✅ Chunk manifests, outline freshness, plot/concordance snapshots, and planner stats all hydrate the window via `DocumentStatusWindow._update_views()`, with incremental refresh supported through the Refresh button and new signal-driven `MainWindow._handle_document_status_signal` hook.
   - ✅ Telemetry badges now style both the status bar indicator and the Telemetry tab (severity-aware backgrounds for chunk flow/analysis). Badge refreshes propagate to the open dialog whenever chunk-flow or analysis events fire, so operators see warnings without reopening the window.
3. **Interaction & commands**
   - ✅ `View → Document Status...` plus the global `Ctrl+Shift+D` action now launch the dialog, and the command palette (`Ctrl+Shift+P`) automatically indexes the `Document Status...` entry so operators can search for it alongside other window actions.
   - ✅ `/status` manual command resolves tab references/`--doc` flags, opens the dialog (or falls back to JSON summaries via `--json`) and keeps the dropdown in sync so multi-document inspection works from chat, the status bar, the menu, or the palette.
   - ✅ Document Status window exposes "Save JSON…" (writes via `export_payload`) plus the existing "Copy JSON" affordance so operators can persist the payload.
4. **Validation**
   - ✅ Headless + Qt coverage now lives in `tests/test_document_status_window.py` (payload loading, telemetry badges, Save JSON) and `tests/test_document_status_service.py` (payload assembly + severity selection), keeping the dialog + `/status` wiring locked down alongside the existing status bar indicator tests.

## Cross-cutting deliverables
- **Docs & runbooks:** Update README, `docs/ai_v2.md`, guardrail guides, and release notes after each workstream lands.
- **Telemetry:** Ensure every new event is enumerated in `tinkerbell.ai.services.telemetry` and exported via `scripts/export_context_usage.py`.
- **Benchmarks:** Add `benchmarks/measure_chunk_latency.py`, create or refresh `benchmarks/large_doc_report.md` (check the file into the repo if it does not yet exist), and capture concordance/planner metrics.
- **Testing:** Maintain fast unit coverage plus large-fixture regression passes (`uv run pytest -k chunk`, targeted benchmark scripts). Document any long-running tests in `docs/operations/`.
- **Rollout plan:** Ship each workstream behind settings/flags, gather telemetry for at least one week, then graduate to GA once dashboards remain green.

## Dependency matrix
| Workstream | Depends on | Enables |
| --- | --- | --- |
| 1. Windowed snapshots | Phase 2 budget policy, existing snapshot tool | Chunk tool (WS2), guardrails (WS4)
| 2. Chunk registry/tool | WS1 manifests, cache bus | Streaming diffs (WS3), concordance (WS5)
| 3. Streaming diff | WS2 chunk handles | Reliable mega-edit patches
| 3b. Diff reliability hardening | WS1 snapshots, WS3 diff payloads | Guardrails (WS4), safer legacy tooling
| 4. Guardrails | WS1–2 telemetry | Safer adoption of WS5–7
| 5. Concordance | WS2 chunk data, WS4 guardrails | Plot orchestration (WS6)
| 6. Storyline orchestration | WS5 entity signals, Phase 4 plot scaffolding | Analyzer plans (WS7)
| 7. Preflight analysis | WS1–6 metadata | Automated tool selection + operator transparency

## Acceptance checklist
1. All seven primary workstreams plus the Workstream 3b reliability package meet functional requirements with tests/docs merged.
2. Telemetry dashboards show chunk-first adoption and no regression in token budgets.
3. Benchmarks demonstrate measurable savings versus the Phase 4 baseline (documented in `benchmarks/large_doc_report.md`).
4. Feature flags default to GA only after staging soak tests and operator sign-off.
