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
2. **Pick an embedding mode/backend** – In **Settings → AI → Embeddings** choose a mode (`Same API`, `Custom API`, `Local`) and backend (`Auto/OpenAI`, `LangChain`, `SentenceTransformers`, or `Disabled`). CLI/env overrides remain available for the backend (`--embedding-backend`, `TINKERBELL_EMBEDDING_BACKEND`).
3. **Install the embeddings extra when needed** – Local `SentenceTransformers` mode depends on PyTorch. Run `uv sync --extra embeddings` (or `pip install -e '.[embeddings]'`) before selecting the mode so `sentence-transformers`, `torch`, and `numpy` are available.
4. **Seed samples** – Open any file under `test_data/phase3/` (see that folder’s README) to exercise the workflows without hunting for fixtures.
5. **Watch guardrails** – When the controller inserts a guardrail hint, restate it to the user and follow the suggested remediation (work in chunks, wait for pending outlines, disable offline fallbacks, etc.).

### Sample walkthrough (`test_data/phase3`)

| File | Purpose | Suggested prompt |
| --- | --- | --- |
| `stacked_outline_demo.md` | Deeply nested headings for outline/retrieval smoke tests. | “Summarize the Control Plane goals and cite pointer IDs.” |
Open a sample, ask the agent to outline or retrieve, and observe the tool payload plus guardrail hint that follows. The existing megabyte-scale fixtures (`5MB.json`, `War and Peace.txt`, `1MB.json`) remain useful for large-document depth limiting and pending rebuild tests.

### Guardrail & troubleshooting matrix

| Tool status / hint | What it means | Next step |
| --- | --- | --- |
| `guardrails[].type = "huge_document"`, `trimmed_reason = token_budget` | Outline capped depth/size. | Work pointer-by-pointer (DocumentFindSectionsTool + DocumentSnapshot) and mention the guardrail in your reasoning. |
| `status = "offline_fallback"`, `offline_mode = true` | Embedding provider is offline, retrieval switched to heuristics. | Treat previews as hints, rehydrate via DocumentSnapshot before editing, and restore connectivity/backend when possible. |
| `is_stale = true` | Cached outline digest no longer matches the current document version. | Trigger a rebuild (wait or poke the worker) and avoid committing edits that rely solely on stale headings. |

Additional reproduction ideas plus prompt language live in `test_data/phase3/README.md`.

## Phase 4 – Character & plot scaffolding (experimental)

Phase 4.3 adds an opt-in memory layer that captures lightweight character/entity and plot beat summaries from subagent jobs. The goal is to give the controller+agents continuity hints without persisting sensitive data or bloating prompts.

### Components


### Usage & limitations

1. Enable Phase 4 subagents and plot scaffolding in the settings dialog (or via CLI/env overrides) and restart the controller if it was already running.
2. Trigger a chat turn that causes the controller to spawn a subagent job (e.g., select >400 chars and ask for a scoped analysis). After the helper finishes, watch for the injected hint acknowledging that scaffolding refreshed for the document.
3. Within the same turn, the agent can call `document_plot_state` to retrieve the cached roster before drafting edits. The tool accepts `document_id`, `include_entities`, `include_arcs`, `max_entities`, and `max_beats` arguments to trim payloads when token budgets are tight.
4. In Phase 4 the store was transient—closing a document or emitting a `DocumentChanged`/`DocumentClosed` event wiped its entry. Phase 5 (see below) introduces optional persistence for operator overrides.

## Phase 5 – Storyline continuity orchestration (experimental)

Phase 5 builds on the same plot scaffolding flag to keep long-form edits narratively coherent. When the flag is enabled, the controller enforces a `plot_outline → document_edit/document_apply_patch → plot_state_update` loop before it allows the agent to wrap a turn.

### PlotStateMemory, overrides, and persistence

- `PlotStateMemory` replaces the transient store with dependency tracking, version metadata, and operator overrides. It subscribes to `DocumentCacheBus` events so stale documents are purged automatically.
- Manual overrides and dependency notes persist to `~/.tinkerbell/plot_overrides.json`. Each entry records `override_id`, summary text, optional `arc_id`/`beat_id`, author, and timestamps so human continuity decisions survive restarts.
- `PlotOutlineTool` (alias `DocumentPlotStateTool`) now hydrates enriched snapshots from `PlotStateMemory.snapshot_enriched()` including overrides, dependencies, and `version_metadata`.

### Enforced plot loop

1. **Read** – `_PlotLoopTracker` blocks `document_edit`/`document_apply_patch` if the agent has not called `plot_outline`/`document_plot_state` first, returning a guardrail hint that explains how to recover.
2. **Edit** – Successful edit calls mark the document as “pending update”. The very next planner request injects a system reminder instructing the agent to run `plot_state_update` before it can finish the turn.
3. **Update** – Once `plot_state_update` succeeds, the tracker clears the pending flag and emits a confirmation hint so operators know the continuity metadata is back in sync.

Disable the plot scaffolding flag to bypass this enforcement entirely (useful for smoketests or legacy sessions).

### Telemetry & observability

- `plot_state.read` fires whenever `PlotOutlineTool` runs, reporting `document_id`, entity/arc counts, override totals, and dependency totals.
- `plot_state.write` records every `PlotStateUpdateTool` call with deltas plus the persistence status for `plot_overrides.json`.
- Tool traces include `plot_loop_blocked` outcomes, and the chat panel mirrors the guardrail hints so operators can see when the agent skipped the enforced loop.

### Validation


## Phase 5 – Preflight analysis & tool recommendations

Phase 5.2 layers a rule-based analyzer (`tinkerbell.ai.analysis`) on top of the chunk/plot/concordance metadata so the controller can select the right tool mix before each turn.

### Analyzer + cache lifecycle

- `AnalysisAgent` ingests `AnalysisInput` snapshots generated by `AIController._build_analysis_input()` and emits structured `AnalysisAdvice` records (chunk profile, required/optional tools, outline refresh flag, warnings, trace metadata, cache state).
- Advice now travels with every chat turn (`AIController._analysis_hint_message`), manual `/analyze` command, and telemetry export via `TelemetryManager.emit_context_usage()` (`analysis_chunk_profile`, `analysis_required_tools`, `analysis_warning_codes`, etc.).
- The TTL cache subscribes to `DocumentCacheBus` events, so `DocumentChangedEvent`/`DocumentClosedEvent` immediately invalidate cached advice and drop stale snapshots; manual edits no longer reuse outdated recommendations.

### Operator interactions

- The status bar exposes a dedicated **Preflight** badge that summarizes the latest advice, while the chat panel mirrors it with hover text (chunk profile, required tools, warnings).
- Operators can run `/analyze [--doc ...] [--start N --end M] [--force-refresh] [--reason note]` to rerun the analyzer on demand. The helper leverages `AIController.request_analysis_advice()`, posts a formatted notice, refreshes badges, and records the run via `analysis.ui_override.*` telemetry.
- LangGraph planners call `ToolUsageAdvisorTool` when they need fresh guidance mid-turn; the tool bridges into `_advisor_tool_entrypoint()` and returns the serialized advice payload.

### Telemetry & exports

- `analysis.advisor_tool.invoked` fires whenever the tool surface runs, capturing `document_id`, selection overrides, force-refresh flag, and whether a cached snapshot was reused.
- `analysis.ui_override.requested/completed/failed` events trace manual `/analyze` executions along with reasons, cache states, and resulting tool lists so dashboards can flag repeated operator overrides.
- `scripts/export_context_usage.py` now prints the analysis columns alongside the existing outline/retrieval metrics, so CSV exports include chunk profile, tool lists, warnings, cache state, and rule traces for each turn.

### Phase 4.4 – Integration, telemetry, hardening

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
- **Collection points** – `AIController` (`ai.orchestration.AIController`) emits an event per turn and aggregates tool invocations. Tool payload sizes are counted via the new token registry so metrics stay consistent.
- **Settings & UI** – The Settings dialog now exposes **Debug → Token logging enabled** and **Token log limit**. When enabled, the status bar shows running totals and the in-memory sink keeps the last *N* events for inspection/test assertions.
- **Programmatic access** – `AIController.get_recent_context_events()` exposes the rolling buffer for tests or external dashboards. Additional sinks can be registered via `TelemetrySink` to stream events elsewhere.
- **Export script** – `uv run python -m tinkerbell.scripts.export_context_usage --format csv --limit 50` dumps the persisted buffer (JSON/CSV) from `~/.tinkerbell/telemetry/context_usage.json` for audits or support bundles.
- **Outline/Retrieval telemetry** – Outline builder (`ai/services/outline_worker.py`), `DocumentOutlineTool`, `DocumentFindSectionsTool`, and `DocumentEmbeddingIndex` now emit structured events (`outline.build.start/end`, `outline.tool.hit/miss`, `outline.stale`, `retrieval.query`, `retrieval.provider.error`, `embedding.cache.hit/miss`) that include latency, cache hit counts, provider names, and `tokens_saved` deltas. Dashboards can subscribe via `TelemetrySink` or reuse `scripts/export_context_usage.py` to trend outline freshness, retrieval performance, and embedding cost spikes.

## 3. Document version IDs & optimistic patching

- **Document metadata** – `DocumentState` now tracks `document_id`, `version_id`, and `content_hash`; `document.snapshot()` and `DocumentBridge.generate_snapshot()` always include these fields along with a `version` token of the form `"{document_id}:{version_id}:{content_hash}"`.
- **Bridge enforcement** – `DocumentBridge.queue_edit()` rejects stale directives (raising `DocumentVersionMismatchError`) and annotates edits with selection hashes so callers must refresh snapshots when conflicts arise.
- **Tools** – `DocumentApplyPatchTool` insists on the current `document_version` before routing diffs through `DocumentEditTool`. Tests in `tests/test_bridge.py` and `tests/test_ai_tools.py` cover both the happy path and mismatch errors.

## 3b. Snapshot-anchored editing guardrails (Phase 5)

Phase 5 hardens diff tooling so every edit is anchored to an explicit snapshot slice. The goal is to stop legacy inline edits from duplicating paragraphs or inserting mid-word glitches when the document shifts between tool calls.

### Tool & schema changes

- `DocumentSnapshot` always exposes `selection_text` and `selection_hash` (SHA-1 of the selected text) so agents can round-trip anchors without recomputing hashes.
- `DocumentApplyPatchTool` and `DocumentEditTool` accept `match_text`, `expected_text`, and `selection_fingerprint` parameters. `selection_fingerprint` is compared against the snapshot `selection_hash`, while `match_text`/`expected_text` are used to relocate the edit if the offsets drift.
- The tool manifest plus system prompts (`src/tinkerbell/ai/prompts.py`) now instruct agents to copy these fields from `document_snapshot` before calling diff/edit tools.

### Recommended workflow

1. Call `document_snapshot` with `include_text=true` and capture `selection_text`, `selection_hash`, `document_version`, and the exact `target_range` you plan to edit.
2. When building a patch, include at least one of `target_range` or `match_text`. Prefer to send both so the bridge can double-check offsets.
3. Copy `selection_hash` into the `selection_fingerprint` field and copy the literal snippet into `match_text` (and `expected_text` if the schema requires it for your caller).
4. If the snapshot slice no longer matches the document, let the tools raise their guardrail errors and immediately refresh the snapshot instead of guessing offsets.
5. Reserve caret inserts (`target_range` start == end) for explicit `action="insert"` directives. Replace operations now require anchors or a non-empty target range.

### Error handling quick reference

| Message fragment | Meaning | Next step |
| --- | --- | --- |
| `Edits must include target_range or match_text` | The edit lacked both a range and an anchor. | Re-run `document_snapshot` and send either the captured range or `match_text`/`selection_fingerprint` bundle. |
| `selection_fingerprint does not match the latest snapshot` | The hash no longer matches the live document. | Refresh the snapshot and rebuild the patch from the new selection window. |
| `Snapshot selection_text no longer matches document content` | The auto-anchor pulled from the snapshot was stale. | Provide explicit `match_text` from the user-visible context or fetch a new snapshot. |
| `match_text matched multiple ranges` | Anchoring text was ambiguous in the current document. | Narrow the selection and resend the edit with a more specific snippet. |
| `match_text did not match any content` | The document drifted beyond the provided snippet. | Refresh the snapshot before retrying the edit. |

### Telemetry & dashboards

- `patch.anchor` – emitted whenever `DocumentApplyPatchTool` or the inline `DocumentEdit` auto-convert path validates anchors. Payload highlights `status` (`success` or `reject`), `phase` (`requirements`, `alignment`, etc.), `anchor_source` (`match_text`, `selection_text`, `fingerprint`, or `range_only`), plus document/tab identifiers. Use this to trend anchor mismatch rates after rollout.
- `patch.apply` – emitted by `DocumentBridge` when a patch succeeds, conflicts, or arrives stale. Includes the duration in milliseconds, `range_count` (for streamed diffs), diff summary, and whether the bridge had to fall back because of a conflict. Dashboards can now plot success/conflict ratios directly from telemetry instead of scraping logs.

These events piggyback on the existing telemetry bus, so the status bar debug widgets and `scripts/export_context_usage.py` will surface them automatically once the sinks are subscribed.

## 3c. Chunk-first selective read guardrails (Phase 5)

Phase 5 tightens the read flow so the agent stays on the "selection snapshot → chunk tool → outline/retrieval" path before touching edits.

- **Controller tracking** – `_ChunkFlowTracker` (inside `ai/orchestration/controller.py`) watches every `DocumentSnapshot`/`DocumentChunk` call. When the agent grabs a full snapshot without chunk hydration, it injects `Guardrail hint (Chunk Flow …)` system messages and emits telemetry.
- **Telemetry events** – New events `chunk_flow.requested`, `chunk_flow.escaped_full_snapshot`, and `chunk_flow.retry_success` ride the telemetry bus. Dashboards (or the status bar debug widgets) can subscribe to those names to trend how often the guardrail fires and how quickly agents recover.
- **UI badges** – The chat panel shows a `Chunk Flow` banner whenever a warning or recovery fires, mirroring the system hint. The status bar adds a matching badge so operators can monitor long-running sessions without scrolling through the transcript.
- **Recovery workflow** – When the badge reports `Chunk Flow Warning`, follow the guardrail hint: re-run `DocumentSnapshot` with a selection window or hydrate the manifest via `DocumentChunkTool`. Once the controller sees a chunk hydrate succeed, the badge flips to `Chunk Flow Recovered` until the next turn.
- **Testing** – `tests/test_chat_panel.py`, `tests/test_widgets_status_bar.py`, and `tests/test_telemetry_controller.py` cover the UI + telemetry plumbing, while the existing controller tests assert the hint injection logic.

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

3. **Controller + prompt integration** (`tinkerbell.ai.orchestration.AIController` & `ai/prompts.py`)
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
- **Mode + backend selection** – The settings dialog exposes an `embedding_mode` field (`same-api`, `custom-api`, `local`) plus the legacy backend dropdown (`auto`, `openai`, `langchain`, `sentence-transformers`, `disabled`). Remote modes reuse the OpenAI/LangChain stack, while `local` forces the backend to `sentence-transformers`. CLI/env overrides still target the backend (`--embedding-backend`, `TINKERBELL_EMBEDDING_BACKEND`, `TINKERBELL_EMBEDDING_MODEL`).
- **Optional embeddings extra** – Local mode requires PyTorch + SentenceTransformers. Run `uv sync --extra embeddings` (or `pip install -e '.[embeddings]'`) to pull in `sentence-transformers>=3.0`, `torch>=2.2`, and their transitive deps before flipping the flag.
- **Model routing** – `settings.embedding_model_name` feeds both OpenAI and LangChain adapters. LangChain mode defaults to `langchain_openai.OpenAIEmbeddings` but auto-detects known provider families (OpenAI, DeepSeek, GLM/Zhipu, Moonshot/Kimi) to preconfigure base URLs, tokenizer hints, and embedding dimensions. Provide family-specific keys via `settings.metadata["<family>_api_key"]` or env vars such as `DEEPSEEK_API_KEY`, `GLM_API_KEY`, and `MOONSHOT_API_KEY`; force a family with `settings.metadata.langchain_provider_family` or `TINKERBELL_LANGCHAIN_PROVIDER_FAMILY`. You can still point at any custom class by populating `settings.metadata.langchain_embeddings_class` and `settings.metadata.langchain_embeddings_kwargs` (or the mirrored env vars `TINKERBELL_LANGCHAIN_EMBEDDINGS_CLASS` / `..._KWARGS`).
- **Local SentenceTransformers workflow** – Selecting `embedding_mode="local"` unlocks fields for `st_model_path`, device (`cpu`, `cuda:0`, `mps`), dtype overrides, cache directory, and batch size. The **Test Embeddings** button now runs the same validator used in the controller so you can confirm the model loads before saving. Metadata is persisted under `settings.metadata` and redacted in `--dump-settings` so secrets/path hints stay private.
- **Credentials & packages** – Remote backends reuse your primary API key/base URL or the dedicated `metadata.embedding_api.*` bundle when `custom-api` is active. LangChain mode requires the provider-specific package (e.g., `langchain-openai`, `langchain-community`) plus whatever headers you list in the metadata. Local mode only touches the files/directories you supply and never ships third-party weights with the app.
- **Status + telemetry** – The status bar surfaces `Embeddings: OpenAI/LangChain/SentenceTransformers/Error` labels, and every `ContextUsageEvent` records `embedding_backend`, `embedding_model`, `embedding_status`, and `embedding_detail`. Export scripts and dashboards use those fields to separate OpenAI vs. LangChain vs. local runs and spot misconfigurations quickly.
- **Licensing & troubleshooting** – Runtime errors set `embedding_status="error"` and bubble the message into both the status bar tooltip and telemetry exports. Clearing `~/.tinkerbell/cache/embeddings/` plus toggling the backend forces a clean reinitialization for remote modes; local failures usually indicate a missing model path or GPU driver. Remember that every third-party model/provider ships under its own license—you must review and honor those terms (and any attribution requirements) before pointing TinkerBell at them.

## 12. Edge cases & resilience guardrails (Phase 3)

- **Document checks** – `ai/utils/document_checks.py` centralizes size + MIME heuristics (`document_size_bytes`, `is_huge_document`, `unsupported_format_reason`). Both the outline worker and retrieval tool reuse these helpers so every entry point agrees on what “unsupported” means.
- **Huge doc throttling** – `OutlineBuilderWorker` switches to top-level headings once `is_huge_document()` trips, tags the cache entry with `huge_document_guardrail`, and `DocumentOutlineTool` relays those guardrails + byte counts to the controller along with guidance to run targeted scans.
- **Unsupported / pending statuses** – Outline/retrieval tools now respond with `status="unsupported_format"` (reason string included) whenever binary files slip in, and they advertise `status="pending"` with a `retry_after_ms` when edits arrive faster than the worker debounce window.
- **Offline embeddings** – `DocumentFindSectionsTool` reports `offline_mode=True` and `status="offline_fallback"` whenever the embedding provider is unavailable so calling agents can treat results as heuristic-only.
- **Cache validation** – Outline cache entries persist the originating `content_hash`; mismatches trigger `_schedule_rebuild`, ensuring corrupted files never hydrate stale outlines.
- **Regression coverage** – Tests in `tests/test_outline_worker.py`, `tests/test_document_outline_tool.py`, `tests/test_retrieval_tool.py`, and `tests/test_memory_buffers.py` cover guardrails, unsupported flows, offline fallbacks, and metadata persistence.
