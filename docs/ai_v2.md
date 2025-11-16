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
