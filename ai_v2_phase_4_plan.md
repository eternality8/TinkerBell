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
### Phase 4.1 – Subagent sandbox MVP
- Define data models in `ai/ai_types.py` and serialization helpers for persistence/logging.
- Implement `SubagentJobQueue` (priority FIFO) with limits per controller turn.
- Update `AIController` planner to build job specs when prompts require chunk-specific analysis; fall back to inline reasoning when disabled.
- Wire executor to existing tool registry (diff, outline, retrieval) but restrict to read-only tools initially.
- Telemetry + settings plumbing: settings schema, docs, status bar indicator for active subagents.

### Phase 4.2 – Chunk result cache
- Add `SubagentResultCache` under `ai/memory`; implement `get_or_run(job_context, executor)` helper that reuses cached payloads if chunk hash + job signature match.
- Integrate with cache bus to invalidate entries on `DocumentChanged(version_id, spans)` events.
- Extend tests (`tests/test_memory_buffers.py`, new `test_subagent_cache.py`) to cover cache hit/miss, invalidation, concurrent requests.
- Telemetry counters for cache effectiveness.

### Phase 4.3 – Character/plot scaffolding
- Create `ai/memory/plot_state.py` with entity + plot schemas, plus storage per document version.
- Implement stub extraction pipeline: reuse retrieval tool to gather chunk, run summarizer to extract entities (rule-based placeholder), persist via cache bus events.
- Expose new `DocumentPlotStateTool` returning latest entities/arcs, guarded by feature flag.
- Update prompts to mention optional pointer messages referencing plot state.
- Add docs (`docs/ai_v2.md` addendum) and tests (`tests/test_plot_state.py`, `test_document_plot_tool.py`).

### Phase 4.4 – Integration, telemetry, hardening
- Smoke test multi-job sequences to ensure sequential execution and token enforcement.
- Ensure TraceCompactor includes subagent traces but still respects summarization rules.
- Update benchmarks (new `benchmarks/subagent_latency.md`) to capture overhead vs. manager-only baseline.
- Finalize docs + release notes; keep flags default-off until stability confirmed.

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
- [ ] Define `SubagentJob` schema + serialization helpers.
- [ ] Implement `SubagentManager/Executor` with quota enforcement + telemetry.
- [ ] Add settings/flags + prompt toggles, ensure backwards compatibility.
- [ ] Update `AIController` planner to enqueue jobs and consume results/pointers.
- [ ] Create `SubagentResultCache` with bus-driven invalidation + tests.
- [ ] Record cache telemetry (hit/miss, eviction) and surface in diagnostics.
- [ ] Build entity/plot schemas + storage layer.
- [ ] Implement stub entity extraction pipeline + background trigger hooks.
- [ ] Expose `DocumentPlotStateTool` + controller wiring.
- [ ] Extend docs (README, `docs/ai_v2.md`, operator guide) and sample workflows.
- [ ] Add unit/integration/perf tests; ensure `uv run pytest` suite green.
- [ ] Update benchmarks + telemetry dashboards; prep release notes.
