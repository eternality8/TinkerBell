# AI v2 Phase 5 Implementation Plan

_Last updated: 18 Nov 2025_

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
1. **Schema design & prompts**
   - Draft manifest shape (`window`, `chunk_profile`, offsets, outline pointer IDs) and document it in `docs/ai_v2.md`.
   - Update chat command schemas so tool calls can specify window constraints; add migration notes for existing tests.
2. **Bridge + tool implementation**
   - Extend `DocumentBridge.generate_snapshot()` with optional `window`, `max_tokens`, `chunk_profile`, `include_text`, `delta_only` args; ensure defaults keep legacy behavior.
   - Cache recent chunk manifests (per tab) and return them with the snapshot payload.
   - Update `DocumentSnapshotTool` to pass through new params, include manifest metadata, and expose sane defaults (selection ± X chars).
3. **Controller & prompt integration**
   - Teach `AIController` planners to request windowed snapshots first; fallback to full snapshots only behind explicit budget decisions.
   - Adjust system prompts to describe the new manifest + chunk flow.
4. **Validation**
   - Add `tests/test_document_snapshot.py` covering parameter combinations + manifest caching.
   - Expand `tests/test_agent.py` scenarios ensuring controller prefers slim snapshots and emits guardrail hints when falling back.
   - Update docs + release notes with operator-facing guidance.

### 2. Chunk registry + `DocumentChunkTool`
**Objective:** Provide on-demand chunk fetches keyed by hashed descriptors.

**Key components:** new `tinkerbell.ai.memory.chunk_index`, `tinkerbell.ai.tools.document_chunk`, updates to `AIController`, `SubagentRuntimeManager`, telemetry.

**Tasks**
1. **Chunk index service**
   - Implement `ChunkIndex` that subscribes to `DocumentCacheBus`, builds semantic-aware chunks (configurable profiles), stores offsets/hashes/outline pointer IDs, and exposes lookup APIs.
   - Persist lightweight per-tab cache inside `AIController` for reuse.
2. **Tool surface + settings**
   - Create `DocumentChunkTool` returning single chunk or iterator handles whose inline text payloads are always <2K tokens; when a chunk exceeds that cap, return only pointer + manifest metadata so callers can request it through pointer-friendly channels.
   - Extend settings UI/CLI for profile selection (code/notes/prose) and overlap controls; surface active profile in the status bar.
3. **Runtime integration**
   - Wire `AIController` and subagent runtime to use chunk fetches when manifests indicate missing context.
   - Add telemetry events (`chunk_cache.hit/miss`, `chunk_tool.window_tokens`) and ensure pointer-only responses follow the same budget accounting as other pointerized payloads.
4. **Validation**
   - Write `tests/test_document_chunk_tool.py` + `tests/test_chunk_index.py` for boundary calculations, cache invalidation, iterator semantics.
   - Add guardrail tests ensuring chunk requests respect budget policy and pointerization rules.

### 3. Streaming diff + multi-range patch flow
**Objective:** Avoid full-buffer diffs by operating on chunk streams.

**Key components:** new `StreamedDiffBuilder` (under `tinkerbell.ai.tools.diff_builder`), updates to `DocumentApplyPatchTool`, `tinkerbell.services.bridge.DocumentBridge`, `editor.patches`.

**Tasks**
1. **Diff builder abstraction**
   - Design streaming API that consumes chunk iterators, emits multi-range diffs keyed by chunk hashes/spans, and reports telemetry metrics (range count, bytes touched).
   - Maintain compatibility with existing diff summaries for UI/traces.
2. **Patch/tool updates**
   - Enhance `DocumentApplyPatchTool` to accept multi-range payloads; serialize metadata so `DocumentBridge` can merge spans server-side.
   - Update `apply_unified_diff` helpers (or add sibling) to handle multiple disjoint ranges and overlapping edits.
3. **Bridge + cache bus integration**
   - Modify `DocumentBridge.queue_edit()` to accept new payloads, publish rich span metadata to the cache bus, and retain legacy behavior behind a feature flag.
4. **Validation**
   - Expand `tests/test_patches.py`, add `tests/test_diff_builder.py`, and introduce regression fixtures in `test_data/` representing mega-edits.
   - Add telemetry assertions verifying streamed diffs report latency + range counts.

### 3b. Diff tooling reliability fixes (parallel to Workstream 3)
Even after streaming diffs ship, we must harden the existing patch/edit path so legacy agents and UI actions stop corrupting text. The following tasks come directly from `ai_fixes.md` and should be executed alongside Workstream 3; treat them as a co-equal workstream rather than a follow-up.

#### 3b.1 Snapshot-anchored range resolution
- Extend `DocumentApplyPatchTool` (and `DocumentEditTool`) with optional `match_text`/`expected_text` plus `selection_fingerprint` parameters.
- When anchors are supplied, locate the text in the latest snapshot, realign `start/end`, or raise a refresh error if no unique match is found.
- Validate that any provided `target_range` still matches the anchor text; reject otherwise and instruct agents to refresh snapshots.

#### 3b.2 Inline edit auto-conversion safety
- Apply the same anchoring checks inside `DocumentEditTool._auto_convert_to_patch` before synthesizing diffs.
- Abort early when the snapshot slice no longer matches the requested content instead of emitting a bogus diff.

#### 3b.3 Guardrails on implicit insertions
- Stop defaulting to `(0, 0)` when the agent omits `target_range` and no anchor exists; require explicit ranges or anchors for replace operations.
- Allow caret-based inserts only when the action is explicitly `insert` or carries a flagged intent.
- Return actionable errors so agents learn to supply enough context, preventing duplicated paragraphs.

#### 3b.4 Tool schema & instruction updates
- Extend the tool manifest to document the new parameters and guide agents to copy `selection_text`/`selection_hash` from `document_snapshot`.
- Refresh system prompts/examples to show how to include anchors and handle validation failures.

#### 3b.5 Testing & telemetry
- Add unit tests for anchor realignment, mismatch rejection, missing-range errors, and inline conversion safety.
- Reproduce historic failure modes (mid-word insertions, duplicate replacements) inside `tests/test_patches.py` and `tests/test_editor_widget.py` to ensure they now fail fast.
- Instrument telemetry (e.g., anchor mismatch counters, patch success/conflict ratios) so we can verify improvements after rollout.

### 4. Prompt + telemetry guardrails for selective reads
**Objective:** Ensure the agent and operators stay on the chunk-first path.

**Key components:** `tinkerbell.ai.prompts`, `AIController` guardrail hints, `tinkerbell.ai.services.telemetry`, status bar/chat UI badges, docs.

**Tasks**
1. **Prompt refresh**
   - Update system + tool prompts to emphasize the order of operations (delta snapshot → chunk tool → outline/retrieval → plot/concordance as needed).
   - Add instructions on interpreting chunk manifests + analyzer recommendations (see Workstream 7).
2. **Guardrail enforcement**
   - Emit controller hints whenever a tool bypasses the chunk flow (e.g., full snapshot used) and require agents to acknowledge.
   - Add UI badges inside `tinkerbell.chat.panel.ChatPanel` and `tinkerbell.widgets.status_bar.StatusBarController` showing when the assistant deviates from selective read guidance.
3. **Telemetry + docs**
   - Introduce events such as `chunk_flow.requested`, `chunk_flow.escaped_full_snapshot`, `chunk_flow.retry_success`; wire them through `tinkerbell.ai.services.telemetry.TelemetryClient` so they propagate to existing exports and status bar debug counters.
   - Document operator recovery steps in `docs/ai_v2.md`, README, and guardrail playbooks.
4. **Validation**
   - Extend `tests/test_agent.py` and `tests/test_ai_turn_coordinator.py` to assert guardrail hints appear + agent acknowledgements are required.
   - Add widget tests for new badges and telemetry counters.

### 5. Character/entity concordance automation
**Objective:** Finish the entity pipeline so the agent can plan character-wide edits safely.

**Key components:** new `CharacterMapStore` (under `tinkerbell.ai.memory`), `CharacterMapTool`, `CharacterEditPlannerTool`, UI affordances, telemetry.

**Tasks**
1. **Store + pipeline**
   - Build `CharacterMapStore` reusing chunk manifests + plot state data, adding alias/pronoun mapping, exemplar quotes, and chunk IDs.
   - Subscribe to cache bus events so concordance data invalidates on edits/closures.
2. **Tooling + planners**
   - Implement `CharacterMapTool` returning entity lists, appearances, pointer IDs.
   - Add `CharacterEditPlannerTool` (subagent-friendly) that walks chunks, tracks completion state, and records results back into plot/character memory.
3. **UI & UX**
   - Provide optional dialog or status hints so users can inspect concordance output and monitor planner progress.
4. **Validation**
   - Author `tests/test_character_map_tool.py`, extend `tests/test_plot_state.py` & `tests/test_subagent_manager.py` for planner loops, add telemetry assertions.

### 6. Storyline continuity orchestration
**Objective:** Upgrade plot scaffolding into a structured plot memory with enforced edit loops.

**Key components:** `PlotStateMemory` (timeline graphs, dependency tracking), `PlotOutlineTool`, `PlotStateUpdateTool`, `SubagentRuntimeManager`, `AIController`, persistence options for overrides.

**Tasks**
1. **Memory upgrade**
   - Extend `DocumentPlotStateStore` into `PlotStateMemory` (new module/file) storing arcs, beats, dependencies, human overrides, and version metadata.
   - Ensure cache bus events wipe stale entries and that memory respects per-document caps.
2. **Tool suite**
   - Ship `PlotOutlineTool` + `PlotStateUpdateTool` with strict schemas; require the manager agent to call them before/after chunk edits.
   - Update prompts + controller logic to enforce the "read plot state → edit chunk → update plot state" loop.
3. **Telemetry/persistence**
   - Add telemetry events for plot state reads/writes and optional persistence of human-authored overrides (profile-level JSON under `~/.tinkerbell`).
4. **Validation**
   - Enhance `tests/test_plot_state.py`, add `tests/test_plot_state_tool.py`, and extend controller/subagent tests covering enforced loops + pointerization of plot payloads.

### 7. Preflight analysis & tool recommendations
**Objective:** Inject document-aware planning before each run so the agent automatically chooses the right tool mix.

**Key components:** new `tinkerbell.ai.analysis` package, `AnalysisAgent`, controller hooks (`AIController._build_messages`), `ToolUsageAdvisorTool`, UI transparency.

**Tasks**
1. **Analysis module**
   - Build rule-based `AnalysisAgent` that inspects document metadata (size, outline freshness, chunk index status, plot/concordance caches, guardrail flags) and emits structured advice (chunk profile, required tools, caution flags).
   - Support caching per `document_version` and TTL-based invalidation.
2. **Controller integration**
   - Invoke the analyzer before prompt construction; inject its output as system metadata and log decisions for telemetry/diagnostics.
   - Provide a `ToolUsageAdvisorTool` so the agent can re-run analysis mid-turn when conditions change.
3. **UI & telemetry**
   - Surface analyzer decisions in the chat panel (e.g., “Preflight recommends chunk profile: prose, retrieval: on, plot state: stale”).
   - Emit telemetry events capturing analyzer latency, cache hits, and operator overrides.
4. **Validation**
   - Create `tests/test_analysis_agent.py`, extend `tests/test_agent.py` to ensure analyzer output is honored, and add widget tests for UI surfacing.

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
