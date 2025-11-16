# AI v2 – Phase 0 implementation notes

Phase 0 introduces the shared infrastructure that every subsequent AI milestone depends on. This document summarizes what shipped, how to exercise it, and where to hook follow-on work.

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

- **Event schema** – `ContextUsageEvent` (in `tinkerbell.ai.services.telemetry`) captures `document_id`, `model`, `prompt_tokens`, `tool_tokens`, `response_reserve`, `conversation_length`, `tool_names`, and a monotonic timestamp.
- **Collection points** – `AIController` (`ai.agents.executor.AIController`) emits an event per turn and aggregates tool invocations. Tool payload sizes are counted via the new token registry so metrics stay consistent.
- **Settings & UI** – The Settings dialog now exposes **Debug → Token logging enabled** and **Token log limit**. When enabled, the status bar shows running totals and the in-memory sink keeps the last *N* events for inspection/test assertions.
- **Programmatic access** – `AIController.get_recent_context_events()` exposes the rolling buffer for tests or external dashboards. Additional sinks can be registered via `TelemetrySink` to stream events elsewhere.
- **Export script** – `uv run python -m tinkerbell.scripts.export_context_usage --format csv --limit 50` dumps the persisted buffer (JSON/CSV) from `~/.tinkerbell/telemetry/context_usage.json` for audits or support bundles.

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
