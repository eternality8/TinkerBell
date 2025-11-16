# AI v2 Implementation Plan

## Objective
Deliver a second-generation AI editing loop that handles 100K+ token files without exhausting context windows. The plan phases features so each milestone produces measurable improvements while maintaining editing fidelity inside the existing `tinkerbell.ai` stack.

## Guiding principles
- **Instrumentation first:** Every phase reports token usage, tool payload size, cache hit rates, and latency so we can measure regressions immediately.
- **Deterministic token math:** Use the model-specific tokenizer for all budget calculations; heuristics are acceptable only as an explicit fallback.
- **Versioned data products:** All caches (chunks, embeddings, outlines, plot state) must carry document IDs + content hashes so edits invalidate stale artifacts automatically.
- **Prompt/backwards compatibility:** New tool schemas land behind optional parameters and prompt toggles so existing tests and user flows keep working during rollout.
- **Data locality:** Keep heavy processing (chunking, embeddings, NER) off the UI thread and avoid duplicating entire documents in memory when operating on large files.

## Current status (Nov 2025)
- ✅ **Phase 0 delivered:** Tokenizer registry + CLI, telemetry sinks/toggles, document versioning, cache bus, and supporting docs/benchmarks are merged with full pytest coverage.
- ✅ **Phase 1 delivered:** LangGraph-driven editor affordances (diff overlays, autosave signal, improved dialogs), deterministic tool traces, telemetry/memory upgrades, and benchmarking artifacts are complete with tests/docs.
- ✅ **Phase 2 delivered:** Context-budget enforcement, pointer summaries, TraceCompactor GA, refreshed docs/benchmarks, and opt-out controls now live on the AI v2 branch.
- 🔜 **Next focus – Phase 3:** Outline, retrieval, and chunk-aware tooling layered on top of the enforced policy described below.

## Phase 0 – Telemetry & shared infrastructure ✅
**Status:** Completed in November 2025. The token counter registry (`ai/client.py` + `scripts/inspect_tokens.py`), telemetry events + sinks (`ai/services/telemetry.py`, status bar hooks, settings toggles), document versioning (`editor/document_model.py`, `services/bridge.py`, optimistic patch enforcement), and the cache invalidation bus (`ai/memory/cache_bus.py`) are live with docs + benchmarks. These foundations now back every subsequent phase.

### Delivered scope
1. **Tokenizer parity layer**
   - Added a tokenizer registry in `ai/client.py` that maps model names to callable tokenizers (OpenAI tiktoken or local fallback).
   - Replaced `_estimate_text_tokens` usages with `_count_tokens(model, text)` and updated all callers; exposed via `scripts/inspect_tokens.py` for manual checks.
2. **Context usage instrumentation**
   - `AIController` now logs per-turn context size (prompt, history, tool payloads) and emits structured telemetry events.
   - Verbose token logging is gated behind new settings + status-bar toggles (`services/settings.py`, `widgets/status_bar.py`).
3. **Document version IDs**
   - `DocumentBridge.generate_snapshot` returns a monotonic `version_id` + content hash.
   - `DocumentApplyPatchTool` enforces optimistic concurrency and bumps version IDs after edits; cache bus consumers invalidate on mismatch.
4. **Cache registry + invalidation bus**
   - Introduced `DocumentCacheBus` under `ai/memory` so modules subscribe to `DocumentChanged(version_id, spans)` events and drop stale artifacts automatically.

### Validation
- `tests/test_ai_client.py`, `test_ai_tools.py`, `test_bridge.py`, `test_memory_buffers.py`, and `test_workspace.py` cover tokenizer routing, cache bus fan-out, and versioned snapshots; the full suite (`uv run pytest`) passes (217 tests).
- `scripts/inspect_tokens.py` exercises the tiktoken-backed counter against large fixture docs; telemetry toggles verified through widget tests.

### Risks & mitigations
- **Tokenizer packages unavailable offline:** ship a bundled fallback estimator.
- **Telemetry flood:** gate verbose logs behind settings and truncate traces after N entries.

## Phase 1 – Desktop UX & workflow hardening ✅
**Status:** Complete. This milestone focused on making the LangGraph assistant safe and ergonomic for large Markdown/YAML/JSON edits while wiring in deterministic telemetry.

### Delivered scope
1. **LangGraph + agent plumbing** – Updated planner/selector/tool loop contracts, structured tool traces, and prompt templates (`ai/agents`, `ai/prompts.py`) so every turn emits diff previews, selection metadata, and guardrail hints.
2. **Document safety + telemetry** – Hardened `DocumentApplyPatchTool`, diff builder, validation/search tools, and cache bus consumers; added autosave + diff overlay events in `main_window.py`, `editor/*`, `services/telemetry.py`, and scripts (`export_context_usage`, `measure_diff_latency`).
3. **UI polish tied to AI workflows** – Delivered chat panel tool timelines, status bar token/autosave indicators, tab-level diff overlays, and the new import/export dialogs with token budgets + sample loaders (`widgets/dialogs.py`, `widgets/status_bar.py`, `chat/*`).
4. **Docs/tests/benchmarks** – Expanded pytest coverage across `tests/test_chat_panel.py`, `test_main_window.py`, `test_editor_widget.py`, etc., refreshed `docs/ai_v2.md`, and updated `benchmarks/phase0_token_counts.md` with diff-latency metrics produced by `benchmarks/measure_diff_latency.py`.

### Validation & acceptance
- Full regression suites (`uv run pytest`, plus targeted dialog/main-window suites) run green; lint/mypy checks mirror CI.
- Manual smoke logs captured for large Markdown/YAML/JSON files (using the new dialog sample dropdown) and archived alongside telemetry exports.
- Architecture docs in `docs/ai_v2.md` + README reference the updated Phase 1 state; Phase 2 follow-ups live in `improvement_ideas.md`.

### Ready for Phase 2
With observability, diff safety, and UI affordances in place, Phase 2 can concentrate on the token budget policy + summarization work outlined below without blocking issues from earlier phases.

## Phase 2 – Token-aware gating & trace summarization ✅
**Status:** Completed (Nov 2025 GA rollout for the context policy + trace compactor stack).
**Goal:** Prevent oversized tool outputs from exhausting budgets and provide safe summaries.

### Work items
1. **Budget policy object**
   - Implement `ContextBudgetPolicy` configurable via settings (per model prompt budget, reserve for response).
   - `AIController` consults policy before appending messages; if over budget, trigger mitigation.
2. **Format-aware summarizer**
   - Add `summarize_tool_content(payload, schema_hint)` helper that supports plain text, diffs, and bullet lists.
   - Allow tools to set `summarizable=False` to skip compression.
3. **Pointer messages**
   - Define a `ToolPointerMessage` schema (e.g., `chunk:123`, `outline:v2`) with instructions in the prompt for how the agent can rehydrate via tool calls.
4. **Trace compactor**
   - After each tool round, if cumulative trace > threshold, replace earlier entries with summarizer output while retaining full logs in UI-only storage.

### Validation
- Unit tests ensuring tool responses are summarized only when thresholds are exceeded.
- Integration test simulating giant diff output to confirm controller substitutes pointer + summary.

### Risks & mitigations
- **Lossy summaries corrupting reasoning:** retain original payload IDs so the agent can refetch via chunk tools.
- **Performance hit from frequent summaries:** cache summarizer results per payload hash.

## Phase 3 – Outline + retrieval tools
**Goal:** Provide global navigation aids so the agent can target relevant chunks without full scans.

### Work items
1. **Hierarchical outline builder**
   - Extend `DocumentSummaryMemory` to store multi-level headings + blurbs per document version.
   - Background job (debounced) rebuilds outline when file size or edit distance crosses thresholds.
2. **`DocumentOutlineTool`**
   - Tool returns latest outline with metadata (heading level, offset, chunk IDs).
   - `DocumentSnapshotTool` optionally attaches outline digest when windowing requests it.
3. **Embedding-backed retrieval**
   - Add `DocumentEmbeddingIndex` using pluggable backends (OpenAI embeddings default, optional local model).
   - Implement `DocumentFindSectionsTool` returning top-k passages with chunk IDs.
   - Cache embeddings per chunk hash and invalidate via cache bus.

### Validation
- Tests generating outlines for markdown/YAML fixtures.
- Retrieval tests verifying changed chunk triggers re-embedding.
- Benchmark showing top-k retrieval reduces chunk calls vs regex search.

### Risks & mitigations
- **Embedding latency/cost:** batch chunk embeddings and reuse cached vectors; allow disabling embeddings in settings.
- **Outline staleness:** include version ID in tool response; controller requests fresh outline when mismatch occurs.

## Phase 4 – Advanced coordination (optional stretch)
**Goal:** Prepare for manager/subagent workflows without shipping the full orchestration yet.

### Work items
1. **Subagent sandbox API**
   - Define `SubagentJob` schema and lifecycle (inputs, allowed tools, outputs) within `AIController`.
   - Add queue executor that can run short-lived model calls sequentially (parallelism optional later).
2. **Chunk result cache**
   - Store subagent outputs (summaries, validations) keyed by chunk hash so they can be reused.
3. **Character/plot data scaffolding**
   - Implement a generic entity index pipeline (stub NER hooks) and plot state storage format, but gate full features behind experimental flags.

### Validation
- Controller tests showing manager agent can spawn a stub subagent that processes a chunk via `DocumentChunkTool` and returns a report kept under token budget.

### Risks & mitigations
- **Cost blowup from subagents:** enforce a hard per-run token cap and surface warnings in telemetry.
- **Complexity creep:** keep subagent feature flagged until earlier phases prove stable.

## Cross-cutting deliverables
- **Docs:** Update `README.md` and add `docs/ai_v2.md` walkthrough after each major phase.
- **Benchmarks:** Maintain a `benchmarks/large_doc_report.md` showing token usage, latency, and success criteria over time.
- **Testing:** Expand `tests/test_ai_tools.py`, `test_bridge.py`, `test_app.py`, and controller tests for every new tool or policy. Run the full regression suite with `uv run pytest` so results stay consistent with the shared environment.
- **Samples:** Add representative large docs (code, prose, YAML) in `assets/sample_docs/` for regression tests.

## Milestone readiness checklist
Before moving to the next phase, verify:
1. Telemetry dashboards show stable metrics and no regressions in existing tests.
2. All new tools have documentation + unit tests.
3. Token budgets stay within configured limits for sample docs (report stored alongside benchmarks).
4. Cache invalidation works across edit → snapshot → chunk → edit loops in `test_patches.py` scenarios.

With these phases, we build a reliable core (telemetry + chunking) before layering advanced retrieval and coordination features, ensuring every step is measurable and reversible.
