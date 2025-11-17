# AI v2 – Phase 0 implementation notes

Phase 0 introduces the shared infrastructure that every subsequent AI milestone depends on. This document summarizes what shipped, how to exercise it, and where to hook follow-on work.

## Phase 3 – Outline & Retrieval Quickstart

Phase 3 layers guardrail-aware outline/retrieval tooling on top of the Phase 0/1/2 foundation. Use this section as the TL;DR before diving into the historical notes below.

### Feature highlights

- **DocumentOutlineTool vNext** – Returns nested headings, blurbs, pointer IDs, digest hashes, and guardrail metadata (`status`, `guardrails`, `trimmed_reason`, `retry_after_ms`).
- **DocumentFindSectionsTool** – Queries the embedding index (OpenAI or LangChain) and falls back to regex/outline heuristics when offline. Responses now flag `offline_mode`, `fallback_reason`, and per-pointer outline context.
- **Controller guardrail hints** – `AIController` injects `Guardrail hint (…)` system messages whenever a tool reports pending, unsupported, stale, trimmed, or offline states so the agent must acknowledge constraints before making edits.
- **Telemetry** – Outline/retrieval events record latency, cache hits, and `tokens_saved`, and the trace compactor continues to pointerize oversized payloads whenever the budget policy demands it.

### Setup checklist

1. **Enable the flag** – Toggle **Settings → Experimental → Phase 3 outline tools** (or launch with `--enable-phase3-outline-tools`).
2. **Pick an embedding backend** – In **Settings → AI → Embeddings** choose `Auto/OpenAI`, `LangChain`, or `Disabled`. Env/CLI overrides remain available (`TINKERBELL_EMBEDDING_BACKEND`, `--embedding-backend`).
3. **Seed samples** – Open any file under `test_data/phase3/` (see that folder’s README) to exercise the workflows without hunting for fixtures.
4. **Watch guardrails** – When the controller inserts a guardrail hint, restate it to the user and follow the suggested remediation (work in chunks, wait for pending outlines, disable offline fallbacks, etc.).

### Sample walkthrough (`test_data/phase3`)

| File | Purpose | Suggested prompt |
| --- | --- | --- |
| `stacked_outline_demo.md` | Deeply nested headings for outline/retrieval smoke tests. | “Summarize the Control Plane goals and cite pointer IDs.” |
| `guardrail_scenarios.md` | Step-by-step guardrail reproductions referencing other large fixtures. | “Walk me through the pending outline scenario.” |
| `firmware_dump.bin` | Tiny binary placeholder that forces `status="unsupported_format"`. | “Generate an outline for this firmware dump.” |

Open a sample, ask the agent to outline or retrieve, and observe the tool payload plus guardrail hint that follows. The existing megabyte-scale fixtures (`5MB.json`, `War and Peace.txt`, `1MB.json`) remain useful for large-document depth limiting and pending rebuild tests.

### Guardrail & troubleshooting matrix

| Tool status / hint | What it means | Next step |
| --- | --- | --- |
| `guardrails[].type = "huge_document"`, `trimmed_reason = token_budget` | Outline capped depth/size. | Work pointer-by-pointer (DocumentFindSectionsTool + DocumentSnapshot) and mention the guardrail in your reasoning. |
| `status = "pending"`, `retry_after_ms` present | Worker is rebuilding the outline after rapid edits. | Wait the suggested delay or keep working off DocumentSnapshot until the rebuild completes. |
| `status = "unsupported_format"` / `reason = binary_*` | File detected as binary/unsupported (e.g., `.bin`). | Convert to Markdown/YAML/JSON/plain text or skip the tool entirely. |
| `status = "offline_fallback"`, `offline_mode = true` | Embedding provider is offline, retrieval switched to heuristics. | Treat previews as hints, rehydrate via DocumentSnapshot before editing, and restore connectivity/backend when possible. |
| `is_stale = true` | Cached outline digest no longer matches the current document version. | Trigger a rebuild (wait or poke the worker) and avoid committing edits that rely solely on stale headings. |

Additional reproduction ideas plus prompt language live in `test_data/phase3/README.md`.

## Phase 4 – Character & plot scaffolding (experimental)

Phase 4.3 adds an opt-in memory layer that captures lightweight character/entity and plot beat summaries from subagent jobs. The goal is to give the controller+agents continuity hints without persisting sensitive data or bloating prompts.

### Components

- **`DocumentPlotStateStore`** (`tinkerbell.ai.memory.plot_state`) keeps an in-memory map of `{document_id → DocumentPlotState}` with capped entity/beat counts (24 each by default). It subscribes to the shared cache bus so edits/closed tabs purge stale entries immediately. Entities are detected via conservative proper-noun heuristics; beats are appended to a primary arc along with metadata (chunk hash, pointer ID, timing).
- **Controller ingestion** – When `SubagentRuntimeConfig.plot_scaffolding_enabled` is true the `AIController` instantiates the store, and every successful subagent job calls `store.ingest_chunk_summary(...)`. After ingests the controller emits a short system hint (“Plot scaffolding refreshed…”) so prompt templates can remind the agent to call the new tool.
- **Tooling surface** – `DocumentPlotStateTool` exposes the cached payload via `document_plot_state`. It is marked `summarizable=False` to prevent the compactor from stripping entities/beats mid-turn. Responses include diagnostic statuses: `plot_state_disabled` (flag off), `plot_state_unavailable` (store missing), `no_document`, `no_plot_state`, or `status="ok"` with entity/arc arrays and metadata.
- **Settings & overrides** – The feature is disabled by default. Enable it via **Settings → Experimental → Plot scaffolding**, CLI flags (`--enable-plot-scaffolding` / `--disable-plot-scaffolding`), or the environment variable `TINKERBELL_ENABLE_PLOT_SCAFFOLDING`. Because ingestion relies on subagent summaries, the best experience pairs the flag with `enable_subagents=True`, but the tool remains callable even when no jobs have run (it will simply report `status="no_plot_state"`).

### Usage & limitations

1. Enable Phase 4 subagents and plot scaffolding in the settings dialog (or via CLI/env overrides) and restart the controller if it was already running.
2. Trigger a chat turn that causes the controller to spawn a subagent job (e.g., select >400 chars and ask for a scoped analysis). After the helper finishes, watch for the injected hint acknowledging that scaffolding refreshed for the document.
3. Within the same turn, the agent can call `document_plot_state` to retrieve the cached roster before drafting edits. The tool accepts `document_id`, `include_entities`, `include_arcs`, `max_entities`, and `max_beats` arguments to trim payloads when token budgets are tight.
4. No data is persisted between sessions. Closing a document or emitting a `DocumentChanged`/`DocumentClosed` event wipes its entry. This keeps the feature investigatory while the UX for human-authored plot data is still out-of-scope.

### Validation

- Unit tests in `tests/test_plot_state.py` validate entity heuristics, beat limits, cache-bus evictions, and telemetry stats.
- Tool tests in `tests/test_document_plot_state_tool.py` assert resolver fallbacks, feature-flag behavior, and response schemas for both populated and empty stores.
- Controller tests (`tests/test_agent.py`, `tests/test_ai_tools.py`) gained coverage for the runtime flag plumbing and the injected “plot scaffolding refreshed” hint to ensure the end-to-end flow remains guarded.

Future iterations may add richer arc detection, manual editing UI, or persistence hooks, but for Phase 4 the emphasis stays on advisory context that surfaces only when explicitly enabled.

### Phase 4.4 – Integration, telemetry, hardening

- **TraceCompactor coverage** – Subagent scouting reports now register ledger entries with `TraceCompactor`, so when later tool calls exceed the budget their summaries compact into pointers that explain how to rebuild the cached context (“rerun the helper or call `DocumentPlotStateTool`”).
- **Turn-level telemetry** – `SubagentManager` emits `subagent.turn_summary` after every run, reporting requested/scheduled jobs, cache hits, skipped or failed jobs, cumulative latency, and tokens consumed. Diagnostics UI widgets (and the persistent telemetry sink) pick these up automatically.
- **Sequential execution smoke tests** – `tests/test_subagent_manager.py` verifies multi-job queues stay strictly sequential and that budget-policy rejections skip work rather than touching the AI client. `tests/test_agent.py::test_ai_controller_registers_subagent_messages_in_trace_compactor` ensures the controller ledger captures helper summaries.
- **Benchmarks & docs** – `benchmarks/measure_subagent_latency.py` plus the published snapshot in `benchmarks/subagent_latency.md` quantify scheduling overhead (<4 ms/job on a 6 ms simulated helper). Release management details, toggles, and rollout guidance live in `docs/ai_v2_release_notes.md`.
- **Flags remain opt-in** – Even with the telemetry + testing guardrails, both `enable_subagents` and `plot_scaffolding_enabled` default to `False`. Keep them disabled outside staging until the new telemetry stays green for at least two weeks.

## 1. Tokenizer parity layer

- **Registry + protocols** – `TokenCounterRegistry` and `TokenCounterProtocol` live in `tinkerbell.ai.client` / `tinkerbell.ai.ai_types`. Every `AIClient` registers a counter for its active model and falls back to a deterministic byte estimator when no backend is available.
- **Backends** – `TiktokenCounter` is used when the optional [`ai_tokenizers`](../pyproject.toml) extra (`tiktoken>=0.12,<0.13`) can be installed. Otherwise `ApproxByteCounter` provides a predictable bytes-per-token approximation so counts are still monotonic and reproducible.
- **CLI helper** – `scripts/inspect_tokens.py` consumes stdin or `--file` input and prints both precise (tiktoken) and estimated counts. Example:
  ```powershell
  uv run python -m tinkerbell.scripts.inspect_tokens --file "test_data/Carmilla.txt"
  ```
  Pass `--estimate-only` to skip `tiktoken` even when it is installed.
- **Windows / Python 3.13 heads-up** – `tiktoken` does not yet ship wheels for Python 3.13. Installing the optional extra on this interpreter requires a Rust toolchain (`rustup` on Windows). Until that is available, the registry logs a single warning and keeps using `ApproxByteCounter`.

## 2. Context usage instrumentation

- **Event schema** – `ContextUsageEvent` (in `tinkerbell.ai.services.telemetry`) captures `document_id`, `model`, `prompt_tokens`, `tool_tokens`, `response_reserve`, `conversation_length`, `tool_names`, a monotonic timestamp, and the active embedding backend/model/status tuple so downstream dashboards can segment runs.
- **Collection points** – `AIController` (`ai.agents.executor.AIController`) emits an event per turn and aggregates tool invocations. Tool payload sizes are counted via the new token registry so metrics stay consistent.
- **Settings & UI** – The Settings dialog now exposes **Debug → Token logging enabled** and **Token log limit**. When enabled, the status bar shows running totals and the in-memory sink keeps the last *N* events for inspection/test assertions.
- **Programmatic access** – `AIController.get_recent_context_events()` exposes the rolling buffer for tests or external dashboards. Additional sinks can be registered via `TelemetrySink` to stream events elsewhere.
- **Export script** – `uv run python -m tinkerbell.scripts.export_context_usage --format csv --limit 50` dumps the persisted buffer (JSON/CSV) from `~/.tinkerbell/telemetry/context_usage.json` for audits or support bundles.
- **Outline/Retrieval telemetry** – Outline builder (`ai/services/outline_worker.py`), `DocumentOutlineTool`, `DocumentFindSectionsTool`, and `DocumentEmbeddingIndex` now emit structured events (`outline.build.start/end`, `outline.tool.hit/miss`, `outline.stale`, `retrieval.query`, `retrieval.provider.error`, `embedding.cache.hit/miss`) that include latency, cache hit counts, provider names, and `tokens_saved` deltas. Dashboards can subscribe via `TelemetrySink` or reuse `scripts/export_context_usage.py` to trend outline freshness, retrieval performance, and embedding cost spikes.

## 3. Document version IDs & optimistic patching

- **Document metadata** – `DocumentState` now tracks `document_id`, `version_id`, and `content_hash`; `document.snapshot()` and `DocumentBridge.generate_snapshot()` always include these fields along with a `version` token of the form `"{document_id}:{version_id}:{content_hash}"`.
- **Bridge enforcement** – `DocumentBridge.queue_edit()` rejects stale directives (raising `DocumentVersionMismatchError`) and annotates edits with selection hashes so callers must refresh snapshots when conflicts arise.
- **Tools** – `DocumentApplyPatchTool` insists on the current `document_version` before routing diffs through `DocumentEditTool`. Tests in `tests/test_bridge.py` and `tests/test_ai_tools.py` cover both the happy path and mismatch errors.

## 4. Cache registry & invalidation bus

- **Pub/sub bus** – `tinkerbell.ai.memory.cache_bus` introduces `DocumentCacheBus` plus standard events (`DocumentChangedEvent`, `DocumentClosedEvent`). Subscribers can be strong or weak references, and helper classes (`ChunkCacheSubscriber`, `OutlineCacheSubscriber`, `EmbeddingCacheSubscriber`) log every notification.
- **Publishers** – `DocumentBridge` publishes change events after every directive/patch and a closed event when `DocumentWorkspace` shuts down a tab. Future caches (chunking, outline, embeddings) can subscribe without tight coupling to the editor.
- **Tests** – `tests/test_memory_buffers.py`, `tests/test_bridge.py`, and `tests/test_workspace.py` validate event ordering, weak-reference cleanup, and integration with the workspace router.

## 5. Validation & follow-up

- **End-to-end tests** – `uv run pytest` exercises the full suite (217 tests as of this phase).
- **Benchmarks & observations** – See `benchmarks/phase0_token_counts.md` for sample token measurements and environment notes. Re-run them whenever you add a new model entry or upgrade `tiktoken`.
- **Snippet validators** – `tinkerbell.ai.tools.validation.validate_snippet` now supports Markdown lint stubs (heading jumps, unclosed fences), optional JSON schema inputs, and `register_snippet_validator()` so downstream teams can plug in bespoke formats without touching the core registry.
- **Next steps** – Later AI v2 phases can now rely on deterministic budgets, telemetry hooks, and cache invalidation semantics. Typical extensions include persisting telemetry, wiring cache subscribers that build chunk/embedding stores, and surfacing version IDs in future API responses.

## 6. Settings, secrets, and overrides

- **Secret providers** – `SettingsStore` now delegates encryption to a pluggable provider: Windows DPAPI when available and Fernet everywhere else. The active backend is recorded inside `settings.json` (`secret_backend`) and can be forced with `TINKERBELL_SECRET_BACKEND=fernet|dpapi` for deterministic test runs.
- **Migration safeguards** – Legacy plaintext API keys are detected during load, re-encrypted with the current provider, and the file is rewritten at version `2` automatically so future CLI tooling can rely on consistent metadata.
- **Deterministic override order** – Settings are merged as **UI defaults → CLI `--set key=value` flags → environment variables**. CLI overrides accept ints/floats/bools plus JSON blobs for structured fields, letting you tweak `max_tool_iterations`, request timeouts, or metadata without touching disk.
- **Settings inspector** – Running `uv run tinkerbell --dump-settings` prints the fully merged configuration (API keys redacted), the resolved settings path, the secret backend, applied CLI overrides, and the `TINKERBELL_*` variables that influenced the run.
- **Custom locations** – Pass `--settings-path` when launching or dumping to point at alternate profiles (useful for smoke tests or portable builds). The CLI honors the same override order, so env vars still win if both are supplied.

## 7. Desktop UX helpers (Phase 1)

- **Preview-first import/open dialogs** – `DocumentLoadDialog` replaces the native picker so every open/import run shows file size, inferred language, and token counts. The dialog highlights how much of the configured context budget a file would occupy and surfaces the first ~3k characters for sanity checks before loading a tab.
- **Selection-aware export dialog** – Saving a document now routes through `DocumentExportDialog`, which renders the current selection (or the start of the file), reports both selection + full-document token totals, and keeps the token budget gauge visible so authors know when exports approach window limits.
- **Curated sample library** – A dropdown in the open dialog pulls Markdown/JSON/YAML fixtures from `test_data/` (and `assets/sample_docs/` when present) so smoke tests on large files are one click away. Selecting a sample immediately previews its contents and token footprint before creating the new tab.

## 8. Benchmarking + performance checkpoints

- **Token + diff baselines** – `benchmarks/phase0_token_counts.md` now tracks both the original tokenizer sanity checks and the new Phase 1 diff latency table. Run `uv run python benchmarks/measure_diff_latency.py` to reproduce the War and Peace / Large JSON runtimes cited there.
- **Automation hook** – The benchmark helper accepts `--case LABEL=PATH` (and `--json`) so CI jobs or future profiling scripts can extend the dataset without editing the file.
- **Pointer impact follow-up** – Sprint 3 enables pointer-driven compaction by default, and the benchmark harness now prints `diff_tokens`, `pointer_tokens`, and `tokens_saved` for every oversized tool payload. War and Peace’s 88K-token diff collapses to ~250 tokens, so the controller stays comfortably below the watchdog ceiling.

## 9. Sprint 2 – Summaries & pointers

Phase 2 Sprint 2 builds on the dry-run budget policy by actually compacting oversized tool payloads. During this sprint the feature stayed behind the existing `context_policy.enabled` flag (defaulted to dry-run); Sprint 3 later flipped the default on for GA. Sprint 2 hinges on three pillars:

1. **Deterministic summarizer module** (`tinkerbell.ai.services.summarizer`)
  - Handles plaintext + diff payloads with conservative heuristics (line clamps, per-hunk stats, bullet synthesis) so results are reproducible in tests.
  - Emits `SummaryResult` records that report both estimated token savings and the pointer metadata needed downstream.
  - Tools can opt out by setting `summarizable = False` on their callable or dataclass; validators do this so lint findings never shrink.

2. **Pointer-aware chat schema** (`tinkerbell.chat.message_model.ToolPointerMessage`)
  - When `ContextBudgetPolicy.tokens_available()` returns `needs_summary`, the controller swaps the original tool response for a lightweight pointer message.
  - Pointer text explains why the payload shrank, includes key metadata (tool name, document/version IDs, diff stats), and always ends with explicit rehydration instructions for LangGraph agents.
  - Raw payloads are still preserved inside `executed_tool_calls` for UI expansion, export scripts, and audits.

3. **Controller + prompt integration** (`tinkerbell.ai.agents.executor.AIController` & `ai/prompts.py`)
  - `_handle_tool_calls` records per-tool `summarizable` flags, feeds payloads to the summarizer, and caches pointer instructions alongside tool traces.
  - `_compact_tool_messages` retries the budget calculation after each summarization while suppressing duplicate telemetry so dashboards stay readable.
  - Prompt templates now brief the agent on pointer semantics (“If you see a pointer message, rerun the referenced tool with the provided parameters to fetch the full data”).

### Usage notes

- **Settings & telemetry** – No new toggles were added; reuse `context_policy.enabled` and the response reserve fields from Phase 1. Telemetry gains `summary_count` + `tokens_saved` deltas per turn, visible in the status bar debug counters.
- **Tests** – `tests/test_ai_tools.py` and `tests/test_agent.py` contain regression coverage for summarizer budgets, pointer serialization, and non-summarizable tools. Run `uv run pytest -k pointer` for a focused sweep.
- **Manual validation** – Load `test_data/War and Peace.txt`, ask the Workspace Agent for a snapshot, and confirm the tool entry shows a pointer badge. Clicking it in the chat panel or invoking “Show full payload” should rehydrate the original diff.

Sprint 3 completes the experience with a dedicated trace compactor service and UI badges everywhere pointers appear (see §10).

## 10. Sprint 3 – Trace compactor GA

Sprint 3 flips the context policy + trace compactor stack to General Availability. Oversized tool responses are now pointerized automatically, the UI surfaces compaction badges, and telemetry/status widgets expose savings without requiring debug flags.

1. **TraceCompactor service** (`tinkerbell.ai.services.trace_compactor`)
  - Maintains a rolling ledger of tool outputs, tracks token savings, and swaps entries for pointer summaries whenever the budget policy reports `needs_summary`.
  - Integrates with `AIController` so compaction happens off the main thread while raw payloads continue to flow to transcripts/export scripts.
2. **UI + transcript affordances** (`chat.chat_panel`, `main_window`, `services.bridge_router`)
  - Tool traces show “Compacted” badges plus pointer metadata, and the transcript/export paths retain the original payload for auditing.
  - Status bar text now appends `tokens_saved` / `total_compactions` counters so long-running sessions have immediate feedback.
3. **Telemetry + benchmarks** (`services.telemetry`, `benchmarks/measure_diff_latency.py`)
  - `trace_compaction` events capture `entries_tracked`, `total_compactions`, and `tokens_saved` for every turn. Tests assert the payloads so dashboards stay consistent.
  - The benchmark helper injects a massive diff per document, showing savings like “War and Peace: 88K diff tokens → 247 pointer tokens (88K saved) in 66 ms” to validate both latency and GA savings.
4. **Defaults flipped** (`services/settings.ContextPolicySettings`)
  - `context_policy.enabled` now defaults to `True` with `dry_run=False`, so fresh installs enforce the policy immediately. Users can still opt out or revert to dry-run via settings or environment overrides.

### Usage notes

- The Settings dialog highlights the active policy + compaction stats, and toggling “Dry run only” is now an explicit opt-in.
- Status bar and chat panel badges require no extra flags; pointer text always includes rehydrate instructions so LangGraph agents can fetch the raw data on demand.
- Telemetry exports (`scripts/export_context_usage.py`) include the latest `trace_compaction` snapshot alongside `context_budget_decision` events for auditors.

### Validation

- New tests `tests/test_trace_compactor.py`, `tests/test_chat_panel.py`, and `tests/test_main_window.py` cover ledger math, pointer metadata propagation, and status-bar stats.
- Full regression suites (`uv run pytest`) run green, and the refreshed benchmark numbers in `benchmarks/phase0_token_counts.md` document the <50 ms overhead requirement despite compaction.

## 11. Embedding runtime & LangChain backends (Phase 3 preview)

- **Feature flag** – Everything is gated behind `settings.phase3_outline_tools` (UI toggle or CLI `--enable-phase3-outline-tools`). When disabled, the embedding runtime tears down automatically and retrieval tools fall back to outline heuristics.
- **Backend selection** – `settings.embedding_backend` accepts `auto` (OpenAI), `langchain`, or `disabled`. Users can override via CLI (`--embedding-backend langchain`) or env vars (`TINKERBELL_EMBEDDING_BACKEND`, `TINKERBELL_EMBEDDING_MODEL`).
- **Model routing** – `settings.embedding_model_name` feeds both OpenAI and LangChain adapters. LangChain mode defaults to `langchain_openai.OpenAIEmbeddings` but now auto-detects known provider families (OpenAI, DeepSeek, GLM/Zhipu, Moonshot/Kimi) to preconfigure base URLs, tokenizer hints, and embedding dimensions. Provide family-specific keys via `settings.metadata["<family>_api_key"]` or env vars such as `DEEPSEEK_API_KEY`, `GLM_API_KEY`, and `MOONSHOT_API_KEY`; force a family with `settings.metadata.langchain_provider_family` or `TINKERBELL_LANGCHAIN_PROVIDER_FAMILY`. You can still point at any custom class by populating `settings.metadata.langchain_embeddings_class` and `settings.metadata.langchain_embeddings_kwargs` (or the mirrored env vars `TINKERBELL_LANGCHAIN_EMBEDDINGS_CLASS` / `..._KWARGS`).
- **Credentials & packages** – LangChain mode requires the provider-specific package (e.g., `langchain-openai`, `langchain-community`) and whatever API keys/base URLs those classes expect. The kwargs blob supports JSON strings or literal dicts in `settings.json` so secrets can come from the same vault.
- **Status + telemetry** – The status bar now surfaces `Embeddings: {backend}` labels, and every `ContextUsageEvent` records `embedding_backend`, `embedding_model`, `embedding_status`, and `embedding_detail`. Export scripts and dashboards use those fields to separate OpenAI vs. LangChain usage and spot misconfigurations quickly.
- **Troubleshooting** – Runtime errors set `embedding_status="error"` and bubble the message into both the status bar tooltip and telemetry exports. Clearing `~/.tinkerbell/cache/embeddings/` plus toggling the backend forces a clean reinitialization.

## 12. Edge cases & resilience guardrails (Phase 3)

- **Document checks** – `ai/utils/document_checks.py` centralizes size + MIME heuristics (`document_size_bytes`, `is_huge_document`, `unsupported_format_reason`). Both the outline worker and retrieval tool reuse these helpers so every entry point agrees on what “unsupported” means.
- **Huge doc throttling** – `OutlineBuilderWorker` switches to top-level headings once `is_huge_document()` trips, tags the cache entry with `huge_document_guardrail`, and `DocumentOutlineTool` relays those guardrails + byte counts to the controller along with guidance to run targeted scans.
- **Unsupported / pending statuses** – Outline/retrieval tools now respond with `status="unsupported_format"` (reason string included) whenever binary files slip in, and they advertise `status="pending"` with a `retry_after_ms` when edits arrive faster than the worker debounce window.
- **Offline embeddings** – `DocumentFindSectionsTool` reports `offline_mode=True` and `status="offline_fallback"` whenever the embedding provider is unavailable so calling agents can treat results as heuristic-only.
- **Cache validation** – Outline cache entries persist the originating `content_hash`; mismatches trigger `_schedule_rebuild`, ensuring corrupted files never hydrate stale outlines.
- **Regression coverage** – Tests in `tests/test_outline_worker.py`, `tests/test_document_outline_tool.py`, `tests/test_retrieval_tool.py`, and `tests/test_memory_buffers.py` cover guardrails, unsupported flows, offline fallbacks, and metadata persistence.
