# AI v2 Implementation Plan

## Objective
Deliver a second-generation AI editing loop that handles 100K+ token files without exhausting context windows. The plan phases features so each milestone produces measurable improvements while maintaining editing fidelity inside the existing `tinkerbell.ai` stack.

## Guiding principles
- **Instrumentation first:** Every phase reports token usage, tool payload size, cache hit rates, and latency so we can measure regressions immediately.
- **Deterministic token math:** Use the model-specific tokenizer for all budget calculations; heuristics are acceptable only as an explicit fallback.
- **Versioned data products:** All caches (chunks, embeddings, outlines, plot state) must carry document IDs + content hashes so edits invalidate stale artifacts automatically.
- **Prompt/backwards compatibility:** New tool schemas land behind optional parameters and prompt toggles so existing tests and user flows keep working during rollout.
- **Data locality:** Keep heavy processing (chunking, embeddings, NER) off the UI thread and avoid duplicating entire documents in memory when operating on large files.

## Phase 0 – Telemetry & shared infrastructure
**Goal:** Establish observability and core utilities required by later phases.

### Work items
1. **Tokenizer parity layer**
   - Add a tokenizer registry in `ai/client.py` that maps model names to callable tokenizers (OpenAI tiktoken or local fallback).
   - Expose `_estimate_text_tokens` → `_count_tokens(model, text)` and update callers.
2. **Context usage instrumentation**
   - In `AIController`, log per-turn context size (prompt, history, tool payloads) and emit structured telemetry events.
   - Add toggle in `services/settings.py` to enable verbose token logging for debugging.
3. **Document version IDs**
   - Extend `DocumentBridge.generate_snapshot` to include `version_id` (monotonic counter or content hash).
   - Ensure `DocumentApplyPatchTool` bumps the version ID after successful edits.
4. **Cache registry + invalidation bus**
   - Introduce a lightweight pub/sub (e.g., `DocumentCacheBus`) under `ai/memory` so modules subscribe to `DocumentChanged(version_id, spans)` events.

### Validation
- New unit tests in `tests/test_ai_client.py` for tokenizer selection.
- Controller tests verifying telemetry records context usage per turn.
- Snapshot tests ensuring version IDs increment on edits.

### Risks & mitigations
- **Tokenizer packages unavailable offline:** ship a bundled fallback estimator.
- **Telemetry flood:** gate verbose logs behind settings and truncate traces after N entries.

## Phase 1 – Windowed snapshots + chunk system
**Goal:** Reduce default tool payload size by returning only needed regions while enabling on-demand chunk retrieval.

### Work items
1. **Snapshot windowing API**
   - Add `window`, `max_tokens`, `include_text`, and `delta_only` parameters to `DocumentSnapshotTool` with schema updates in `chat/commands.py`.
   - Bridge implementation trims text based on selection +/- configurable context, returning metadata if `include_text` is false.
2. **Chunk indexer**
   - Create `DocumentChunkIndex` under `ai/tools` that slices documents into presets (`code`, `notes`, `prose`) using structural heuristics.
   - Store chunk descriptors (start/end offsets, line span, hash, semantic tags) keyed by document + version ID.
3. **`DocumentChunkTool`**
   - New tool returns chunk text by ID or byte/line range, optionally expanding by overlap percentage.
   - Integrate with chunk cache in `AIController` (per tab LRU with cap).
4. **Prompt updates + migrations**
   - Update `prompts.py` to describe slim snapshot defaults and chunk usage.
   - Adjust integration tests (`tests/test_ai_tools.py`, `tests/test_bridge.py`) to cover new parameters.

### Validation
- Benchmark large sample docs (add under `assets/sample_docs`) comparing old vs new snapshot token counts; capture in markdown artifact.
- Unit tests for chunk boundary logic and cache invalidation on version mismatch.

### Risks & mitigations
- **Stale chunk offsets after edits:** subscribe to cache bus to drop affected chunks whenever patches land.
- **Memory pressure from caches:** expose settings for cache size and evict oldest chunks first.

## Phase 2 – Token-aware gating & trace summarization
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
- **Testing:** Expand `tests/test_ai_tools.py`, `test_bridge.py`, `test_app.py`, and controller tests for every new tool or policy.
- **Samples:** Add representative large docs (code, prose, YAML) in `assets/sample_docs/` for regression tests.

## Milestone readiness checklist
Before moving to the next phase, verify:
1. Telemetry dashboards show stable metrics and no regressions in existing tests.
2. All new tools have documentation + unit tests.
3. Token budgets stay within configured limits for sample docs (report stored alongside benchmarks).
4. Cache invalidation works across edit → snapshot → chunk → edit loops in `test_patches.py` scenarios.

With these phases, we build a reliable core (telemetry + chunking) before layering advanced retrieval and coordination features, ensuring every step is measurable and reversible.
