# AI v2 – Phase 4 Implementation Plan

## Objective
Deliver advanced coordination features that let the AI controller spawn scoped subagents, reuse chunk-level outputs, and scaffold narrative/entity memory without regressing latency, token budgets, or existing UX contracts. Shipping criteria:
- Manager agent can define, queue, and monitor `SubagentJob` executions inside existing `AIController` flows.
- Chunk result cache rehydrates prior subagent outputs deterministically and respects cache invalidation rules from earlier phases.
- Character/plot scaffolding persists lightweight entity+plot state per document and exposes it through guarded tools/APIs.
- Features remain optional via settings/toggles and pass full regression + new targeted suites.

## Scope & guardrails
- **In-scope:** Controller/runtime changes, new memory/caching modules, metadata schemas, telemetry, docs/tests.
- **Out-of-scope:** UI for human authoring of plot data (stub hooks only), long-lived background workers, or parallel subagent execution (sequential only this phase).
- **Assumptions:** Tokenizer + cache bus infra from Phases 0-3 remain stable; retrieval/outline tooling already available to subagents; OpenAI + local LLM backends share controller contracts.

## Architecture overview
1. **Subagent sandbox layer** (new `tinkerbell.ai.agents.subagents` module)
   - `SubagentJob` dataclass: `{job_id, parent_run_id, created_at, state, instructions, allowed_tools, chunk_ref, budget, result}`.
   - `SubagentManager`: orchestrates job lifecycle (enqueue → execute → finalize) within controller run loop.
   - `SubagentExecutor`: serializes tool calls, enforces token/time quotas, emits telemetry spans.
   - Feature flag: `settings.ai.enable_subagents` + prompt toggle to ensure backwards compatibility.
2. **Chunk result cache** (extend `ai/memory/cache_bus` + new `SubagentResultCache`)
   - Keyed by `{document_id, chunk_hash, job_type}` to allow reuse while honoring version IDs.
   - Hooks into cache bus `DocumentChanged` events to evict stale entries when chunk hash mismatch.
   - Storage adapter interface to support in-memory default + pluggable persistent backends later.
3. **Character/plot scaffolding**
   - `EntityIndex` service storing `{entity_id, name, type, attributes, supporting_chunks}` and `PlotState` objects with arcs/scenes.
   - Stub NER/structure extraction pipeline triggered optionally after subagent runs (flagged `experimental_plot_scaffolding`).
   - Surface read-only summaries via new `DocumentPlotStateTool` (hidden unless flag on).
4. **Observability + safety**
   - Telemetry events: `subagent.job_started`, `.job_completed`, `.job_failed`, cache hit/miss metrics.
   - Budget enforcement integrated with `ContextBudgetPolicy`; subagents inherit a portion of the parent budget.
   - Tests to ensure failures degrade gracefully (controller emits pointer message, no crash).

## Detailed implementation plan
### Phase 4.1 – Subagent sandbox MVP _(status: ✅ core runtime + telemetry/settings completed)_
- ✅ Define data models in `ai/ai_types.py` and serialization helpers for persistence/logging.
- ✅ Implement `SubagentJobQueue` (priority FIFO) with limits per controller turn.
- ✅ Update `AIController` planner to build job specs when prompts require chunk-specific analysis; fall back to inline reasoning when disabled.
- ✅ Wire executor to existing tool registry (diff, outline, retrieval) but restrict to read-only tools initially.
- ✅ Telemetry + settings plumbing: persisted flag + CLI/env overrides, prompt guardrails, and the status bar telemetry indicator for live subagent state.

### Phase 4.2 – Chunk result cache _(status: ✅ cache + invalidation shipped; diagnostics surfaced via Phase 4.4 telemetry)_
- ✅ Add `SubagentResultCache` under `ai/memory` with deterministic signatures, TTL, bus-driven eviction, and helper APIs already consumed by `SubagentManager`.
- ✅ Integrate with cache bus `DocumentChanged/Closed` events so stale entries clear automatically.
- ✅ Extend targeted tests (`tests/test_subagent_cache.py`) to cover hit/miss behavior, immutability guarantees, and event-driven invalidation (buffers coverage still queued for concurrency cases).
- ✅ Wire cache/subagent telemetry counters into diagnostics UI/CLI via the new `subagent.turn_summary` events emitted after each helper turn.

### Phase 4.3 – Character/plot scaffolding _(status: ✅ storage/tool/docs/tests shipped; extraction tuning ongoing)_
- ✅ Created `ai/memory/plot_state.py` with entity + plot schemas plus per-document storage tied to the cache bus.
- ✅ Implemented the stub extraction pipeline that reuses retrieval chunks, runs the summarizer for entity heuristics, and persists via cache events.
- ✅ Exposed the `DocumentPlotStateTool` (gated by settings/CLI/env toggles) and wired it into the controller + chat commands.
- ✅ Updated prompts/pointer messaging so assistants can reference plot scaffolding when the flag is on.
- ✅ Added docs (`README.md`, `docs/ai_v2.md`, `docs/operations/subagents.md`) and tests (`tests/test_plot_state.py`, `tests/test_document_plot_state_tool.py`) covering storage behavior, cache clearing, and tool responses.
- ⏳ Continue tuning extraction heuristics + telemetry once we gather operator feedback.

### Phase 4.4 – Integration, telemetry, hardening _(status: ✅ telemetry + docs landed; ongoing monitoring)_
- ✅ Added sequential multi-job + budget enforcement coverage (`tests/test_subagent_manager.py`) so helper queues stay deterministic and policy rejections skip execution.
- ✅ Registered subagent scouting reports with `TraceCompactor` and taught the pointer builder how to emit subagent-aware rehydrate instructions; controller tests verify ledger coverage.
- ✅ Emitted turn-level telemetry (`subagent.turn_summary`) plus cache-hit counts, and surfaced the data in diagnostics + release documentation.
- ✅ Authored `benchmarks/measure_subagent_latency.py` and the companion report (`benchmarks/subagent_latency.md`) to capture orchestration overhead vs. the manager-only baseline.
- ✅ Published `docs/ai_v2_release_notes.md` summarizing Phase 4, reiterated that flags stay default-off, and linked to the operator runbook.
- ⏳ Continue watching telemetry dashboards before flipping the default-on flags (staged rollout milestone).

## Data contracts
- **SubagentJobState:** `queued | running | succeeded | failed | skipped (cache hit)`.
- **Job instructions:** templated prompt snippet referencing chunk pointer or outline section.
- **Result payload:** structured summary, recommended edits, telemetry metadata, pointer to chunk hash.
- **Error handling:** on failure, emit `ToolPointerMessage` advising retry via same chunk.

## Edge cases & mitigations
- Empty/redundant jobs → detect duplicates via dedup hash before enqueue.
- Budget exhaustion → short-circuit job, emit warning pointer.
- Large documents with rapid edits → cache invalidates via bus; controller throttles job creation until debounce settles.
- NER pipeline hallucinations → keep outputs strictly advisory, no automatic edits.

## Validation strategy
1. **Unit tests** for job queue, executor quotas, cache invalidation, entity schemas.
2. **Integration tests** simulating controller run with two subagent jobs, verifying pointer messaging + cache hits.
3. **Performance tests** measuring added latency per job and cache reuse improvements.
4. **Manual QA** using large sample docs to confirm no UI regressions.

## Documentation & rollout
- Update `README.md` Phase 4 section + `docs/ai_v2.md` deep dive.
- Add `ai_v2_phase_4_plan.md` (this doc) + changelog entry.
- Provide operator guide in `docs/operations/subagents.md` (flag default-off instructions).
- Roll out: internal dogfood → staged beta (flag opt-in) → GA after telemetry stability (target 2 weeks of data).

## Task checklist for progress tracking
- [x] Define `SubagentJob` schema + serialization helpers.
- [x] Implement `SubagentManager/Executor` with quota enforcement + telemetry.
- [x] Add settings/flags + prompt toggles, ensure backwards compatibility.
- [x] Update `AIController` planner to enqueue jobs and consume results/pointers.
- [x] Create `SubagentResultCache` with bus-driven invalidation + tests.
- [x] Record cache telemetry (hit/miss, eviction) and surface in diagnostics (diagnostics UI exposure pending Phase 4.4).
- [x] Build entity/plot schemas + storage layer.
- [x] Implement stub entity extraction pipeline + background trigger hooks.
- [x] Expose `DocumentPlotStateTool` + controller wiring.
- [x] Extend docs (README, `docs/ai_v2.md`, operator guide) and sample workflows (now include plot-scaffolding quickstart + operator runbook).
- [x] Add unit/integration/perf tests; store/tool coverage (`tests/test_plot_state.py`, `tests/test_document_plot_state_tool.py`), helper sequencing/TraceCompactor coverage (`tests/test_subagent_manager.py`, `tests/test_agent.py::test_ai_controller_registers_subagent_messages_in_trace_compactor`), and full `pytest` (340 tests in 5.67 s).
- [x] Update benchmarks + telemetry dashboards; prep release notes (`benchmarks/measure_subagent_latency.py`, `benchmarks/subagent_latency.md`, `docs/ai_v2_release_notes.md`, and `subagent.turn_summary` events feeding diagnostics).
