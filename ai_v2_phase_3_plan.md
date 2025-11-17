# AI v2 Phase 3 Technical Plan

## Objective & Success Criteria
- Deliver navigable outlines and embedding-backed retrieval so the agent can request narrowly scoped context without scanning entire files.
- Keep prompt + tool payloads within the Phase 2 context policy while serving 100K+ token documents.
- Ship with parity across Markdown, YAML/JSON, and plaintext sources with <150 ms outline latency for 1 MB files and <500 ms top-k retrieval (OpenAI embeddings backend) measured on the benchmark fixtures.
- Maintain full backwards compatibility: all new tool schemas stay opt-in until tests/docs pass, then become default under a `phase3_outline_tools` flag.

## Scope & Assumptions
- Operates within existing `tinkerbell.ai` desktop app; no server-side components are introduced in this phase.
- Document identifiers + content hashes from Phase 0 remain the single source of truth for cache invalidation.
- Embedding provider defaults to OpenAI via existing API client credentials; optional local embedding runner can be configured but not required for GA.
- Context budget enforcement from Phase 2 governs every new tool invocation; if an outline/retrieval payload would overflow, the controller must instead emit a pointer request.

## Architecture Overview
1. **DocumentSummaryMemory vNext (`ai/memory/buffers.py`):** stores multi-level outlines per `document_id` + `version_id`. Each record now includes an ordered tree of headings, character offsets, chunk IDs, and excerpt hashes.
2. **Outline Builder Worker (`ai/services`):** background job subscribed to `DocumentCacheBus` that computes/recomputes outlines asynchronously using tokenizer-aware chunking.
3. **DocumentOutlineTool (`ai/tools/document_outline.py`):** synchronous tool the agent can call; returns the most recent outline along with freshness metadata and digest (hash of outline payload for trace dedupe).
4. **DocumentEmbeddingIndex (`ai/memory/embeddings.py`):** manages chunk vectors, provider adapters, batching, and eviction keyed by `(document_id, chunk_hash)`.
5. **DocumentFindSectionsTool (`ai/tools/document_find_sections.py`):** exposes top-k retrieval results (chunk metadata + pointer handles) and integrates with Phase 2 pointer message schema.
6. **Controller integration (`ai/agents/executor.py` + `ai/client.py`):** planner inspects pointer summaries and decides whether to call Outline or Retrieval tools, feeding results into LangGraph nodes.
7. **Telemetry + Benchmarks:** `ai/services/telemetry.py`, `benchmarks/measure_diff_latency.py`, and a new `benchmarks/retrieval_latency.py` log outline/retrieval timings, cache hit ratios, and token savings.

### Data Flow (Outline)
1. `DocumentCacheBus` emits `DocumentChanged(version_id, spans)` after edits.
2. Outline worker debounces events per document (e.g., 750 ms) and reads the latest `DocumentSnapshot` via `DocumentBridge`.
3. Worker chunkifies using tokenizer-aware blocks, runs `OutlineExtractor` to build a tree (heading levels + blurbs), and persists to `DocumentSummaryMemory` + disk (MemoryStore).
4. Tool requests read from memory; if version mismatch, tool can trigger a synchronous rebuild (guarded by timeout + budget) or respond with `status="stale"` and pointer asking controller to retry later.

### Data Flow (Retrieval)
1. Outline worker emits chunk list + hashes to `DocumentEmbeddingIndex`.
2. Index verifies cache for each chunk; missing/dirty chunks are batched into provider requests (configurable batch size 16, concurrency 2) and stored in `embeddings.sqlite` (via `sqlite3` or disk-backed `faiss` if available).
3. `DocumentFindSectionsTool` receives a natural-language query, encodes it with the same backend, performs vector search (top-k default 6), and returns summary per chunk (start/end offsets, pointer ID, token cost estimate).
4. Controller either requests chunk bodies (Phase 2 pointer hydration) or uses returned highlight snippets directly when under budget.

## Workstreams & Deliverables
### 1. Outline Builder Enhancements _(Status: ✅ Complete)_
- **Schema:** Extend `SummaryRecord` with `version_id`, `outline_hash`, and `nodes: list[OutlineNode]` where `OutlineNode` includes `id`, `parent_id`, `level`, `text`, `char_range`, `chunk_id`, `children`. Store as nested dict for JSON persistence.
- **Algorithm:**
  - Tokenize per document type: Markdown headings, YAML keys, JSON structural braces.
  - Derive hierarchy using incremental parser; degrade gracefully to flat list when structure unavailable.
  - Compute blurbs using trimmed excerpts (max 240 chars) and store token estimate.
- **Storage:**
  - Update `MemoryStore.save_document_summaries` to persist outlines.
  - Add `OutlineCacheStore` (simple JSON/SQLite file) to support fast reload without warming DocumentSummaryMemory for all docs.
- **Debounce & Prioritization:**
  - Use `asyncio` worker (Qt-friendly) listening to cache bus; maintain priority queue sorted by document size and edit distance.
  - Hard cap rebuild frequency (e.g., max once per 5s per doc) to avoid thrash on rapid edits.

#### Checklist
- [x] Extend `SummaryRecord` and `OutlineNode` schema to capture `version_id`, `outline_hash`, hierarchy, and blurbs.
- [x] Implement tokenizer-aware outline parser for Markdown, YAML, JSON with graceful degradation to flat lists.
- [x] Persist outlines via `MemoryStore` plus the new `OutlineCacheStore`, and verify cold-start hydration.
- [x] Ship debounced outline worker with priority queue + rebuild frequency guardrails.
- [x] Add excerpt/token estimation logic and ensure `_max_summary_chars` bounds still apply.

### 2. `DocumentOutlineTool` _(Status: ✅ Complete)_
- **Contract:**
  - Input: `{document_id, desired_levels (default all), include_blurbs: bool, max_nodes}`.
  - Output: `{version_id, generated_at, is_stale, outline_digest, nodes: [...]}` with nodes referencing `pointer_id = f"outline:{document_id}:{node_id}"`.
- **Implementation:**
  - Lives under `src/tinkerbell/ai/tools/document_outline.py` with tests in `tests/test_ai_tools.py` + dedicated `tests/test_document_outline_tool.py`.
  - Validates context budget (rough token size of requested nodes) before returning; if over budget, degrade by trimming deeper levels first.
- **UI/Controller Integration:**
  - Chat timeline badges showing outline availability.
  - `DocumentSnapshotTool` optionally attaches digest (hash) so the agent can detect changes between requests without fetching full payload.

#### Checklist
- [x] Define tool schema inputs/outputs (including `outline_digest`, `pointer_id`) and document via docstring + JSON schema tests.
- [x] Implement tool module with budget-aware truncation and stale-outline handling.
- [x] Add automated tests (`tests/test_ai_tools.py`, `tests/test_document_outline_tool.py`) covering happy path + over-budget trimming.
- [x] Surface outline availability in chat timeline + `DocumentSnapshotTool` digest propagation.

### 3. Embedding Index Service _(Status: ✅ Complete)_
- **Module Layout:**
  - `ai/memory/embeddings.py` defines `DocumentEmbeddingIndex`, provider adapters (`OpenAIEmbeddingProvider`, `LocalEmbeddingProvider`), and `EmbeddingStore` persistence (SQLite with vector extension or fallback to disk JSON + `numpy` arrays).
- **Chunking Strategy:**
  - Reuse chunk boundaries from Outline worker to avoid double work.
  - Store `chunk_metadata` (document_id, version_id, chunk_id, start, end, token_count, outline_node_id?).
- **Invalidation:**
  - Subscribe to cache bus; when `version_id` mismatch or spans overlap chunk range, mark chunk dirty and queue re-embedding.
- **Batching & Rate Limits:**
  - Use `RateLimiter` utility (existing?) or create new simple token bucket to keep OpenAI usage within 60 RPM / 1500 RPD (configurable).

#### Checklist
- [x] Implement `DocumentEmbeddingIndex` with provider abstraction + persistence store.
- [x] Reuse outline chunk metadata for embeddings; ensure chunk IDs stay stable across rebuilds when content unchanged.
- [x] Wire cache-bus invalidation to mark dirty chunks and trigger re-embedding tasks.
- [x] Add batching + rate limiting knobs plus telemetry for queue depth and provider failures.

### 4. `DocumentFindSectionsTool` _(Status: ✅ Complete)_
- **Contract:**
  - Input: `{document_id, query, top_k=6, min_confidence=0.3, filters}`.
  - Output: `{version_id, latency_ms, pointers: [{chunk_id, pointer_id, score, preview, token_estimate}]}`.
- **Features:**
  - Optionally attach `outline_context` if retrieval nodes map to outlines (helpful for reasoning).
  - When embeddings disabled, fallback to regex + outline heuristics while surfacing `strategy="fallback"` for telemetry.
- **Testing:** add new suite `tests/test_retrieval_tool.py` with synthetic embeddings stub.

#### Checklist
- [x] Finalize tool schema (inputs, outputs, confidence thresholds) and document pointer hydration usage.
- [x] Implement retrieval flow that queries `DocumentEmbeddingIndex`, handles fallback paths, and estimates token costs.
- [x] Write tests that cover vector-based ranking, fallback search, and telemetry hooks.
- [x] Ensure controller can hydrate returned pointers or inline previews without busting budgets.

### 5. Controller & Agent Changes _(Status: ✅ Complete)_
- Update planner prompts in `ai/prompts.py` to describe Outline/Retrieval tools and pointer hydration rules.
- Extend `AIController` run loop to:
  - Detect when user ask references "section"/"heading" and bias Outline tool first.
  - Track `outline_digest` to avoid redundant tool calls (if digest matches last one seen this conversation, skip).
- Update `chat/commands.py` to let users manually invoke "Show outline" / "Find sections" commands for debugging.

#### Checklist
- [x] Update planner prompts/trainings to describe Outline/Retrieval capabilities and pointer etiquette.
- [x] Add routing heuristics inside `AIController` to detect outline-worthy queries and dedupe via `outline_digest` tracking.
- [x] Provide manual chat commands + UI affordances for QA/debug scenarios.
- [x] Expand regression tests ensuring controller chooses correct tool under varied prompts.

### 6. Settings, Flags, and UX _(Status: ✅ Complete)_
- Add `phase3_outline_tools` flag in `services/settings.py` with toggles in `widgets/dialogs.py`.
- Provide UI affordances: status bar icon when outline stale, tooltip with last build latency.
- Add CLI hooks in `scripts/export_context_usage.py` to include outline + retrieval stats.

#### Checklist
- [x] Introduce `phase3_outline_tools` flag in `services/settings.py` and thread through configuration loader.
- [x] Add toggle surfaces in settings dialog + CLI switches for headless runs.
- [x] Display outline staleness + latency indicators in status bar; confirm telemetry wiring.
- [x] Update export scripts to capture outline/retrieval metrics for audits.

**Status (2025-11-16):** Flag plumbing now spans settings/env/CLI, manual commands are gated accordingly, the status bar surfaces outline freshness, and the export CLI now records outline/retrieval metrics for audits.

### 7. LangChain Embedding Provider Integration _(Status: ✅ Complete)_
- Reuse the existing LangChain dependency to wrap multiple hosted/local embedding backends under a single `LangChainEmbeddingProvider` that plugs into `DocumentEmbeddingIndex`.
- Extend settings so users choose `embedding_backend="langchain"` and specify `embedding_model_name` (e.g., `deepseek-embedding`, `glm-4-embed`, `text-embedding-3-large`). Auto-detect provider families to pick the correct LangChain class and auth flow.
- Maintain compatibility with current OpenAI/local providers: controller checks settings, instantiates the LangChain adapter when selected, and falls back gracefully if the requested model isn’t supported.
- Surface telemetry (`embedding.provider=langchain`, `embedding.model`) and UI copy so users understand which backend is active.

#### Checklist
- [x] Implement `LangChainEmbeddingProvider` bridging `DocumentEmbeddingIndex` to `langchain.embeddings` classes with shared batching/rate-limit hooks.
- [x] Add `embedding_backend` + `embedding_model_name` settings (UI + CLI) and persist per-workspace overrides.
- [x] Auto-detect provider families (OpenAI/DeepSeek/GLM/etc.) to configure auth headers, dimensionality, and tokenizer hints; fall back to manual selection when unknown.
- [x] Integrate telemetry, logging, and error surfaces so unsupported models degrade to heuristics with actionable messages.
- [x] Document LangChain setup (API keys, model matrix) across README, `docs/ai_v2.md`, and troubleshooting guides.

## Telemetry & Observability _(Status: ✅ Complete)_
- Outline worker/tool instrumentation now emits `outline.build.start`, `outline.build.end`, `outline.tool.hit`, `outline.tool.miss`, and `outline.stale` with latency, node/token counts, cache status, and estimated `tokens_saved`. See `tinkerbell/ai/services/outline_worker.py` and `tinkerbell/ai/tools/document_outline.py` with coverage in `tests/test_document_outline_tool.py`.
- Retrieval and embedding services broadcast `retrieval.query`, `retrieval.provider.error`, `embedding.cache.hit`, and `embedding.cache.miss` with provider labels, cache ratios, pointer totals, and per-query token savings, enabling dashboards to track latency/cost regressions (`tests/test_retrieval_tool.py`, `tests/test_memory_embeddings.py`).
- `scripts/export_context_usage.py` and the existing `TelemetrySink` implementations can now persist/stream these events so shared Grafana/PowerBI dashboards and alert rules can key off latency, cache hit rates, and provider errors without further schema changes.

#### Checklist
- [x] Emit structured telemetry events for outline builds, tool hits/misses, retrieval queries, embedding cache stats.
- [x] Add dashboards/alerts that watch latency, cache hit rate, and provider error spikes (documented workflow: new events land in persistent sinks/exports consumed by shared dashboards).
- [x] Correlate telemetry with token savings estimates and ensure redaction policies hold.

## Edge Cases & Resilience _(Status: ✅ Complete)_
- **Huge documents (>5 MB):** `OutlineBuilderWorker` now clamps outlines to top-level headings via `_limit_outline_depth`, stamps `huge_document_guardrail=true`, and `DocumentOutlineTool` surfaces guardrail guidance plus document byte counts.
- **Binary / unsupported formats:** new `ai/utils/document_checks.py` performs MIME + binary sniffing; outline/retrieval tools return `status="unsupported_format"`, skip embeddings, and log reasons in metadata/telemetry.
- **Rapid edits:** the outline tool consults the worker’s `is_rebuild_pending` hook and responds with `status="pending"` + `retry_after_ms` (429-equivalent) whenever a rebuild is queued.
- **Offline mode:** retrieval now reports `offline_mode=True` with `status="offline_fallback"/"offline_no_results"` when the embedding provider is unavailable so controllers can degrade gracefully.
- **Cache corruption:** cached outlines persist `content_hash` + metadata; loader verifies hashes and enqueues rebuilds when mismatches appear, preventing stale/invalid payloads.

#### Checklist
- [x] Implement guardrails for huge docs (depth limiting, truncation flags) and verify prompts reference them (`outline_worker.py`, `document_outline.py`).
- [x] Add MIME sniffing + unsupported-format responses with clear telemetry codes (`document_checks.py`, `document_outline.py`, `document_find_sections.py`).
- [x] Handle rapid-edit backpressure (`OutlinePending`) and ensure controller retries gracefully (`DocumentOutlineTool.pending_outline_checker`).
- [x] Provide offline/air-gapped fallback paths and user-facing error states (`DocumentFindSectionsTool.offline_mode`).
- [x] Validate cache integrity by comparing stored hashes; auto-rebuild on mismatch (cache hydration logic + `_schedule_rebuild`).

## Testing Strategy _(Status: ❌ Not Started)_
1. **Unit tests**
   - Outline parser per format (Markdown headings, YAML keys) with fixtures under `tests/fixtures/outline/`.
   - Embedding index CRUD with stub provider (deterministic vectors) verifying caching + invalidation.
   - Tool schemas serialization/deserialization.
2. **Integration tests**
   - Simulate edit → cache bus → outline rebuild → tool request (use `tests/test_ai_tools.py`).
   - Retrieval query using synthetic embeddings ensures top-k ranking + fallback path.
3. **Performance tests**
   - Extend `benchmarks/measure_diff_latency.py` or add `benchmarks/retrieval_latency.py` to time outline builds for 1MB JSON + Markdown.
   - Ensure CPU usage <80% during rebuild on reference laptop (8-core, Windows) and memory footprint <200 MB.
4. **Regression suites**
   - Run `uv run pytest tests/test_ai_tools.py tests/test_memory_buffers.py tests/test_workflow_phase3.py` on every PR; nightly full run remains unchanged.

#### Checklist
- [ ] Build fixtures for Markdown/YAML outlines and deterministic embedding stubs.
- [ ] Cover unit + integration + performance paths in CI (fast) and nightly (full) suites.
- [ ] Automate regression command (`uv run pytest ...`) in PR template/checklist.
- [ ] Track benchmark outputs over time to catch latency regressions early.

## Benchmarks & KPIs _(Status: ❌ Not Started)_
- Outline build latency p95 <= 400 ms for 1 MB Markdown; <= 250 ms for 500 KB YAML.
- Retrieval query latency (including embedding encoding) p95 <= 500 ms for top_k=6.
- Token savings: average reduction of >=70% compared to naive full document send during scripted tasks (report in `benchmarks/large_doc_report.md`).
- Cache hit rate >=85% for outlines and >=75% for embeddings on steady-state editing sessions.

#### Checklist
- [ ] Implement `benchmarks/retrieval_latency.py` (or extend existing scripts) to capture outline/retrieval timings.
- [ ] Record token savings + cache hit metrics in `benchmarks/large_doc_report.md` per release.
- [ ] Define alert thresholds for KPI regressions and feed into telemetry dashboards.

## Rollout Plan _(Status: ❌ Not Started)_
1. **Phase 3a (feature-flagged)**
   - Ship outline worker + tool behind settings toggle; gather telemetry.
2. **Phase 3b (retrieval beta)**
   - Enable embeddings + retrieval for internal dogfooders; monitor cost dashboards.
3. **Phase 3c (GA)**
   - Default flag on once KPIs met; update docs (`docs/ai_v2.md`, README) and announce in release notes.
4. **Backout strategy**
   - Flags allow disabling entire subsystem within one release; cached data stored separately so removal is one `rm` command + restart.

#### Checklist
- [ ] Phase 3a: enable outline worker/tool behind flag and verify telemetry coverage.
- [ ] Phase 3b: roll embeddings/retrieval to dogfood cohort with cost monitoring + feedback loop.
- [ ] Phase 3c: flip default flag once KPIs met, update docs/release notes, and monitor for regressions.
- [ ] Document backout runbook (disable flag, purge caches, restart) and rehearse once pre-GA.

## Risks & Mitigations _(Status: ❌ Not Started)_
| Risk | Impact | Mitigation |
| --- | --- | --- |
| Embedding cost spikes | Unexpected API spend | Batch requests, expose per-doc budget, allow local provider fallback |
| Outline stale view leads to bad agent edits | Incorrect context | Include `version_id` + `is_stale` flag; controller revalidates before acting |
| Storage bloat from embeddings | Disk pressure on laptops | Evict least-recently-used chunk embeddings, compress SQLite page size, allow user to clear cache via settings |
| Latency regressions on large docs | Poor UX | Debounce rebuilds, limit depth, parallelize parsing in worker pool |
| Privacy concerns logging outlines | Sensitive data exposure | Sanitize telemetry payloads (counts only), keep content hashes not text |

#### Checklist
- [ ] Confirm mitigations above are implemented + tested (e.g., cost guardrails, stale detection, cache eviction).
- [ ] Review telemetry payloads for PII before GA.
- [ ] Add monitoring to ensure mitigation toggles stay enabled in production builds.

## Documentation & Samples _(Status: ✅ Complete)_
- `docs/ai_v2.md` now opens with a Phase 3 quickstart that explains the outline/retrieval architecture, guardrail hints, and a troubleshooting matrix that maps tool statuses to corrective actions.
- Added a "Phase 3 outline + retrieval" block to the README so Windows users know how to enable the feature flag, pick an embedding backend, and run the curated samples.
- Introduced a dedicated sample pack under `test_data/phase3/` (Markdown hierarchy, guardrail scenario cookbook, and binary placeholder) documented via `test_data/phase3/README.md` so QA has reproducible fixtures.

#### Checklist
- [x] Update `docs/ai_v2.md` diagrams + walkthrough.
- [x] Add sample docs under `test_data/phase3/` with README explaining usage.
- [x] Document new CLI/settings in README and changelog.
- [x] Provide troubleshooting guide for outline/retrieval failures (telemetry codes, cache reset instructions).

## Out-of-Scope / Future Work
- Manager/subagent orchestration (Phase 4).
- Semantic search across multiple documents simultaneously.
- Neural Named Entity extraction—left for experimental flag once retrieval proves stable.
