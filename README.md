# TinkerBell

> A Windows-first, agent-aware desktop text editor that lets you co-write Markdown, YAML, JSON, and plain text with an AI assistant that can *see* and *edit* your document through guarded, structured actions.

TinkerBell pairs a PySide6 editor with a full LangChain/LangGraph tool stack so you can ask for summaries, rewrites, annotations, or scoped refactors and review every change before it lands. The app ships with a qasync event loop bridge, encrypted settings storage, autosave-aware document bridge, and an extensible toolbox for future agents.

## Contents

- [Highlights](#highlights)
- [Architecture at a glance](#architecture-at-a-glance)
- [Getting started](#getting-started)
- [Configuring AI access](#configuring-ai-access)
- [Everyday workflow](#everyday-workflow)
- [Built-in agent tools](#built-in-agent-tools)
- [Safety, privacy, and reliability](#safety-privacy-and-reliability)
- [Testing & developer experience](#testing--developer-experience)
- [Roadmap & contributing](#roadmap--contributing)

## Highlights

- **Dual-pane UX** – A splitter-based main window (`tinkerbell.main_window.MainWindow`) keeps the editor and chat panes in sync with status indicators, toolbars, and autosave messaging.
- **AI-native editing** – The chat panel streams OpenAI-compatible responses, shows tool traces, and forwards structured commands (`EditDirective`) to the editor bridge.
- **Diff-based edits by default** – Agents fetch snapshots, build unified diffs, and submit `action="patch"` directives so every change is reproducible and undo-friendly.
- **Structured safety rails** – All AI edits flow through `DocumentBridge` where schema validation, document-version checks, and diff summaries prevent stale or destructive operations.
- **Deterministic token budgets** – A shared `TokenCounterRegistry` normalizes counts per model, falls back to a byte-length estimator when `tiktoken` is unavailable, and ships with `scripts/inspect_tokens.py` for quick sanity checks.
- **Context budget policy + trace compactor** – Budget checks now run (and enforce) by default, compacting oversized tool payloads into pointer summaries while keeping the original trace available for auditing. Settings still expose dry-run/override switches when you need to experiment.
- **Phase 3 outline + retrieval** – Guardrail-aware `DocumentOutlineTool` and `DocumentFindSectionsTool` stream structured digests, surface pending/unsupported/huge-document hints, and fall back to offline heuristics while the controller injects "Guardrail hint" messages the agent must obey.
- **Turn-level telemetry** – The status bar and debug settings can stream per-turn prompt/tool budgets via `ContextUsageEvent` objects so regressions are visible before shipping.
- **Versioned document bus** – Every snapshot now includes `{document_id, version_id, content_hash}` and `DocumentCacheBus` broadcasts `DocumentChanged`/`DocumentClosed` events so downstream caches can stay coherent.
- **Markdown-first editor** – `EditorWidget` wraps `QPlainTextEdit`/`QsciScintilla` with headless fallbacks, Markdown preview, undo/redo, selection tracking, and theme hooks.
- **One-click imports** – `File → Import…` converts PDFs (and future formats via pluggable handlers) into fresh, editable tabs so you can work with research papers or specs that aren’t plain text.
- **Async everywhere** – LangGraph-powered agents run on asyncio while `qasync` ensures Qt stays responsive during streaming conversations.
- **Windows-friendly secrets** – API keys are encrypted with DPAPI when available (fallback to Fernet) so dropping your laptop does not leak credentials.
- **Tested components** – `pytest` + `pytest-qt` suites cover agents, bridge logic, widgets, syntax helpers, dialogs, and service layers.

## Architecture at a glance

```
┌───────────────────────────────────────────────┐
│ MainWindow                                   │
│  ├─ EditorWidget (Markdown/YAML/JSON)        │
│  ├─ ChatPanel (history, composer, traces)    │
│  └─ StatusBar + menu/toolbar actions         │
│                                               │
│  Documents ↔ DocumentBridge ↔ Agent Tools    │
│                                               │
│  AI stack                                    │
│  ├─ AIClient (OpenAI-compatible streaming)   │
│  ├─ LangGraph Agent (planning + retries)     │
│  └─ Tools: snapshot, edit, search/replace,   │
│      validation, memory buffers              │
└───────────────────────────────────────────────┘
```

Reference docs:

- `plan.md` – end-to-end product scope and sequencing.
- `module_plan.md` – per-module responsibilities and APIs.
- `src/tinkerbell/**` – concrete implementations referenced above.

## Getting started

### Prerequisites

- Windows 10/11 (works on macOS/Linux with PySide6, but the packaged secrets vault currently targets Windows).
- Python **3.11** or newer on your PATH.
- [uv](https://docs.astral.sh/uv/) for fast dependency management (falls back to pip if you prefer, but uv is what we test).
- Optional: VS Code + Python extension for richer dev tooling.

### 1. Clone & install

```powershell
git clone https://github.com/eternality8/TinkerBell.git
cd TinkerBell
uv sync
```

`uv sync` creates `.venv/` and installs runtime + dev dependencies (PySide6, LangChain, LangGraph, OpenAI SDK, pytest, etc.). If you cannot install `uv`, run `pip install uv` first or translate the dependencies from `pyproject.toml` into your environment manager of choice.

### 2. Launch the desktop app

```powershell
uv run tinkerbell
```

The console script calls `tinkerbell.app:main`, which boots the qasync-enabled Qt application, warms FAISS for vector searches, and opens the main window. The first launch will create `~/.tinkerbell/settings.json` for preferences and encrypted credentials.

### 3. Explore sample docs (optional)

`test_data/phase3/` now ships the Phase 3 sample pack (stacked outline demo, guardrail scenario cookbook, and a binary placeholder). Read `test_data/phase3/README.md` for reproduction steps. The legacy `assets/sample_docs/` snippets and large fixtures under `test_data/` remain available for broader smoke tests.

### Optional extras (tokenizers)

Install the optional tokenizer extra (`tiktoken>=0.12,<0.13`) whenever you need model-authentic counts or want to run token benchmarks:

```powershell
uv sync --extra ai_tokenizers
```

> `tiktoken` currently requires a Rust toolchain for Python 3.13 on Windows. Install [rustup](https://rustup.rs/) or run the command under Python 3.12 until official wheels are published. Without the extra, the editor automatically falls back to the deterministic byte-length estimator.

### Embedding backends & configuration

Phase 3 outline/retrieval tooling relies on an embedding index that can speak either native OpenAI embeddings or any LangChain-compatible provider. The runtime picks the backend from **Settings → AI → Embeddings** and mirrors the choice into telemetry/status widgets.

1. **Enable the tools** – Toggle **Settings → Experimental → Phase 3 outline tools** (or launch with `--enable-phase3-outline-tools`) so the outline worker + embedding runtime spin up.
2. **Choose a backend** – In the settings dialog select `Auto/OpenAI`, `LangChain`, or `Disabled`. `Auto` resolves to OpenAI unless the feature flag is off. The same knobs are exposed via CLI/environment overrides:
	 - CLI: `uv run tinkerbell --embedding-backend langchain --embedding-model deepseek-embedding`
	 - Env vars: `TINKERBELL_EMBEDDING_BACKEND=langchain`, `TINKERBELL_EMBEDDING_MODEL=deepseek-embedding`
3. **Provide credentials** – OpenAI embeddings reuse the global API key/base URL/org. LangChain adapters fall back to `langchain_openai.OpenAIEmbeddings`, so install `langchain-openai` and set the relevant API key/env vars for whichever provider you target.
	- The app auto-detects common LangChain families (OpenAI, DeepSeek, GLM/Zhipu, Moonshot/Kimi). It inspects `embedding_model_name` (or `settings.metadata["langchain_provider_family"]` / `TINKERBELL_LANGCHAIN_PROVIDER_FAMILY`) and wires the correct base URL, tokenizer hint, and embedding dimensionality. Provider-specific API keys can live in `settings.metadata["<family>_api_key"]` or env vars such as `DEEPSEEK_API_KEY`, `GLM_API_KEY`, and `MOONSHOT_API_KEY`. Override URLs per family with `settings.metadata["<family>_base_url"]` when needed. Unknown models fall back to the stock OpenAI configuration until you supply manual overrides.
4. **Advanced overrides** – When you need a non-OpenAI LangChain class, drop this into `settings.metadata` (or export `TINKERBELL_LANGCHAIN_EMBEDDINGS_CLASS/KWARGS`):

	 ```jsonc
	 {
		 "langchain_embeddings_class": "langchain_community.embeddings.DeepSeekEmbeddings",
		 "langchain_embeddings_kwargs": {
			 "model": "deepseek-embedding",
			 "api_key": "${DEEPSEEK_API_KEY}",
			 "base_url": "https://api.deepseek.com/v1"
		 }
	 }
	 ```

	 The kwargs blob can be stored as a dict in `settings.metadata` or as a JSON string in `TINKERBELL_LANGCHAIN_EMBEDDINGS_KWARGS`. All fields merge with the automatically supplied `model` argument.
5. **Observe the runtime** – The status bar shows `Embeddings: LangChain/OpenAI/Error` labels, and every `ContextUsageEvent` now includes `embedding_backend`, `embedding_model`, and `embedding_status` so exports/audits can segment LangChain usage.

If embeddings are disabled or initialization fails, the outline worker keeps running but retrieval calls degrade gracefully and telemetry marks the backend as `unavailable`.

#### Phase 3 outline + retrieval field guide

- The controller injects `Guardrail hint (…)` system messages whenever outline/retrieval responses include `guardrails`, `status != "ok"`, or `offline_mode=true`. Restate the warning to the user and follow the suggested remediation before editing.
- Use the curated fixtures under `test_data/phase3/` to validate each guardrail quickly. For example, `stacked_outline_demo.md` is perfect for pointer hydration loops, `guardrail_scenarios.md` documents huge/pending/offline playbooks, and `firmware_dump.bin` triggers the unsupported-format path instantly.
- Full troubleshooting steps (including a matrix that maps tool statuses to actions) live in `docs/ai_v2.md` under “Phase 3 – Outline & Retrieval Quickstart.”

## Configuring AI access

You can supply OpenAI-compatible credentials in three interchangeable ways:

1. **Settings dialog** – Press `Ctrl+,` or use **Settings → Preferences…** to enter a base URL, API key, model name, and retry/backoff settings. Keys are stored with DPAPI (Windows) or Fernet (cross-platform) via `SettingsStore`. The dialog also exposes **Max Context Tokens**, **Response Token Reserve**, and the **Context Budget Policy** toggles (enforce vs. dry-run plus optional prompt/reserve overrides) so you can dial in budgets even though enforcement is now on by default.
2. **Environment variables** – Set any subset of the following before launching the app:
	 - `TINKERBELL_API_KEY`
	 - `TINKERBELL_BASE_URL` (e.g., `https://api.openai.com/v1` or your proxy)
	 - `TINKERBELL_MODEL` (defaults to `gpt-4o-mini`)
	 - `TINKERBELL_THEME` / `TINKERBELL_ORGANIZATION`
	 - `TINKERBELL_DEBUG_LOGGING` (set to `1`/`true` to force verbose logging and prompt dumps)
	 - `TINKERBELL_REQUEST_TIMEOUT` (seconds before an AI request fails; defaults to `90`)
3. **Programmatic injection** – Instantiate `Settings` or `ClientSettings` yourself if you embed TinkerBell in a larger Python workflow.
4. **CLI overrides** – Pass `--set key=value` flags to the launcher to override persisted settings for a single session:

```powershell
uv run tinkerbell --set base_url=https://proxy.example.com --set max_tool_iterations=5
```

Add `--settings-path` when you want to point at a custom `settings.json` file.

Precedence is deterministic: **environment variables override CLI flags, which in turn override the values saved via the UI**. This matches the debugging workflow where you might hardcode safe defaults, tweak them per-session with CLI overrides, and fall back to environment variables for quick emergency switches (e.g., rotating API keys).

Need to confirm what the app sees? Use the built-in inspector to dump the fully merged configuration with secrets redacted:

```powershell
uv run tinkerbell --dump-settings --set base_url=https://proxy.example.com
```

The output includes the resolved settings path, the active secret backend (DPAPI or Fernet; override with `TINKERBELL_SECRET_BACKEND`), and whichever environment variables were applied.

Test credentials via the **Refresh Snapshot** or a simple “Say hello” chat message. Failures are surfaced in the chat panel and status bar, and logs are written to your platform-specific temp directory.

## Context budgets & trace compaction

- **Enforced budget policy (opt-out available)** – Context budgets ship enabled with `dry_run=False`, so every new session automatically enforces prompt/reserve math. Toggle **Settings → Context budget policy → Dry run only** if you want to inspect decisions without blocking, or disable the policy entirely for legacy debugging.
- **Automatic tool pointerization** – When the policy reports `needs_summary`, oversized tool payloads (snapshots, diffs, search results) are compacted into `[pointer:…]` summaries before they’re handed back to the model. Each pointer carries human-readable text plus rehydrate instructions so LangGraph nodes (and future tools) can call the originating tool again with a narrower scope whenever the raw payload is required. Validators and other critical outputs opt out via the `summarizable=False` flag on their registrations.
- **UI + telemetry hooks** – Chat tool traces now show “Compacted” badges, the status bar publishes `tokens_saved` / `total_compactions`, and telemetry gains a `trace_compaction` event alongside `context_budget_decision`. Exported benchmarks (`benchmarks/measure_diff_latency.py`) demonstrate savings like “War and Peace diff: 88K tokens → 247 pointer tokens” to keep regression budgets honest.

## Everyday workflow

1. **Open, create, or import a document** – Markdown, YAML, JSON, or plain text files are supported out of the box, and PDFs can be converted to text via **File → Import…**. Syntax detection drives highlighting and validation helpers.
2. **Compose a prompt** – Select the text you want help with, describe the task (“Rewrite the introduction in an encouraging tone”), and hit **Send**. The selection summary and a fresh document snapshot are automatically attached.
3. **Watch the agent work** – Streaming responses land in the chat history. Enable the Tool Activity panel from **Settings → Show tool activity panel** whenever you need to inspect each LangChain tool invocation (snapshot, diff builder, edits, etc.).
4. **Apply or rollback edits** – Structured payloads go through the bridge, which enforces document-version checks and emits diff summaries (e.g., `+128 chars`). Undo/redo is still available because edits use the regular editor APIs.
5. **Iterate rapidly** – Refresh snapshots, toggle Markdown preview, or enable autosave intervals from settings. Each AI turn records metadata so future prompts understand the document state.

## Built-in agent tools

| Tool | Module | Purpose |
| --- | --- | --- |
| `DocumentSnapshotTool` | `tinkerbell.ai.tools.document_snapshot` | Returns the latest document text, metadata, selection, preview flag, and diff token so the agent can reason safely.
| `DocumentEditTool` | `tinkerbell.ai.tools.document_edit` | Applies validated insert/replace/annotate directives and patch diffs through the bridge with undo support and diff summaries.
| `DocumentApplyPatchTool` | `tinkerbell.ai.tools.document_apply_patch` | Uses the live snapshot to build a diff for the requested range/content, then routes it through `DocumentEdit` so the agent never forgets to apply the edit.
| `DiffBuilderTool` | `tinkerbell.ai.tools.diff_builder` | Generates unified diffs from before/after snippets so agents never have to handcraft patch formatting.
| `SearchReplaceTool` | `tinkerbell.ai.tools.search_replace` | Provides regex/literal transforms with capped replacements, diff previews, and optional dry-run summaries before edits are enqueued.
| `ValidationTool` | `tinkerbell.ai.tools.validation` | Checks YAML/JSON snippets via `ruamel.yaml`/`jsonschema`, lint-stubs Markdown for heading/fence issues, and exposes hooks for custom validators.
| `ListTabsTool` | `tinkerbell.ai.tools.list_tabs` | Enumerates open tabs (`tab_id`, title, path, dirty flag) so agents can target any document without stealing focus.
| `Memory Buffers` | `tinkerbell.ai.memory.buffers` | Maintains conversation + document summaries so prompts stay concise without losing context.

You can register custom tools at runtime via `AIController.register_tool`, and the LangGraph plan automatically picks them up.

### Tool parameter reference

- **`document_snapshot`** — accepts `delta_only` (bool) to request only changed fields, `tab_id` to target a non-active document, `source_tab_ids` to batch additional read-only snapshots, and `include_open_documents` to embed lightweight metadata for every tab. Each response still carries the latest diff summary and digest so agents can detect drift.
- **`document_edit`** — consumes either a native `EditDirective` or a JSON/mapping payload matching the schema exposed during registration. Prefer `action="patch"` plus a unified diff and `document_version`; legacy `insert`/`replace` actions remain available for small, cursor-relative tweaks. Provide `tab_id` whenever the edit should be applied to a background tab.
- **`document_apply_patch`** — requires `content` (replacement text) and optionally `target_range`, `document_version`, `rationale`, `context_lines`, and `tab_id`. It snapshots the targeted document, builds a diff for the requested range, and immediately sends it through `document_edit` so diff construction + application happen in one step.
- **`diff_builder`** — accepts `original`, `updated`, optional `filename`, and optional `context` (default 3) to produce a ready-to-send unified diff string compatible with `document_edit` patch directives.
- **`search_replace`** — parameters include `pattern`, `replacement`, `is_regex`, `scope` (`document` or `selection`), `dry_run`, `max_replacements` (defaults to a guarded cap), `match_case`, and `whole_word`. Each call reports replacement counts, whether the cap was hit, and a unified diff preview; with `dry_run=True`, no edit occurs and only previews/diff metadata are returned.
- **`validate_snippet`** — requires `text` and `fmt` (`yaml`, `yml`, `json`, `markdown`, or `md`) and responds with a `ValidationOutcome` describing the first issue plus a count of remaining problems. JSON calls optionally accept a schema, Markdown checks flag heading jumps/unclosed fences, and you can register additional formats at runtime via `tinkerbell.ai.tools.validation.register_snippet_validator`.
- **`list_tabs`** — no parameters; returns `{tabs: [...], active_tab_id, total}` so the agent can map natural-language references ("roadmap tab") to actual `tab_id` values before issuing snapshot/edit requests.

## Phase 0 instrumentation & observability

### Deterministic token counters

- `TokenCounterRegistry` keeps a per-model tokenizer (`TiktokenCounter` when the optional extra is installed, otherwise `ApproxByteCounter`).
- The registry is shared by the AI client, controller, memory buffers, and telemetry emitters so every component speaks the same budget language.
- Run `scripts/inspect_tokens.py` to inspect counts from stdin or a file:
	```powershell
	uv run python -m tinkerbell.scripts.inspect_tokens --file "test_data/Romeo and Juliet.txt"
	```
- Install `tiktoken` via `uv sync --extra ai_tokenizers` for exact parity. On Python 3.13 you currently need a Rust toolchain; without it, the CLI still prints the byte-length approximation and logs a single warning per session.

### Context usage telemetry

- `ContextUsageEvent` objects capture per-turn prompt tokens, tool tokens, response reserves, tool names, timestamps, and the active embedding backend/model/status so audits can distinguish LangChain vs. OpenAI runs.
- Toggle **Settings → Debug → Token logging enabled** (or set `settings.debug.token_logging_enabled` programmatically) to surface live totals in the status bar and capture up to *N* events (default 200).
- Retrieve the rolling buffer via `AIController.get_recent_context_events()` for assertions, diagnostics, or custom sinks.
- Export recent events anytime with the CLI below (JSON or CSV) which reads the persisted buffer from `~/.tinkerbell/telemetry/context_usage.json`:
	```powershell
	uv run python -m tinkerbell.scripts.export_context_usage --format csv --limit 50 --output usage.csv
	```
	Use `--format json` (default) for structured blobs, `--source` to point at a custom buffer, and omit `--output` to stream directly to stdout for piping.

### Document version IDs

- `DocumentState` snapshots now embed `{document_id, version_id, content_hash}` along with a combined `version` token.
- `DocumentApplyPatchTool` refuses to run without a matching `document_version` and `DocumentBridge` raises `DocumentVersionMismatchError` when callers use stale snapshots.
- Selection fingerprints and diff summaries keep optimistic concurrency obvious without leaking implementation details to end users.

### Cache registry & invalidation bus

- `tinkerbell.ai.memory.cache_bus` provides `DocumentCacheBus` plus `DocumentChangedEvent` / `DocumentClosedEvent` types and helper subscribers (`ChunkCacheSubscriber`, `OutlineCacheSubscriber`, `EmbeddingCacheSubscriber`).
- `DocumentBridge` publishes change events automatically after every edit/patch and `DocumentWorkspace.close_tab()` now emits a closed event, keeping downstream caches honest.
- Future chunk/outline/embedding caches only need to subscribe; no additional wiring is required inside the editor.

> Need more detail? See [`docs/ai_v2.md`](docs/ai_v2.md) for implementation notes that map directly to the Phase 0 checklist.

## Safety, privacy, and reliability

- **Document versioning & patch guards** – Every snapshot includes a SHA-1 digest and selection hash so stale edits are rejected before they touch the editor; patch directives must cite the version they were built against.
- **Guarded parsing** – `chat.commands` normalizes and JSON-validates any agent payload (even if it arrives as a markdown code fence) before it becomes an `EditDirective`.
- **Encrypted secrets** – API keys reside in `%USERPROFILE%\.tinkerbell\settings.json`, encrypted with Windows DPAPI when possible; other platforms fall back to Fernet with a locally stored key (`settings.key`).
- **Error transparency** – All warnings bubble up via the status bar, chat notices, and structured logs, making it easy to debug misconfigurations or flaky endpoints.

## Testing & developer experience

- **Run the full suite**:

	```powershell
	uv run pytest
	```

	GUI-sensitive tests rely on `pytest-qt`, so running them on Windows with a display server is recommended. Headless CI uses the stub widgets built into the editor and chat modules.

- **Static analysis**: `ruff`, `black`, and `mypy` configs live in `pyproject.toml`. Run `uv run ruff check .` or `uv run mypy src` as needed.
- **Logging**: Set `TINKERBELL_DEBUG=1` before launching to enable verbose logs (including Qt message handler output).
- **Debug prompt logging**: Toggle **Enable debug logging** inside the Settings dialog (or set `TINKERBELL_DEBUG_LOGGING=1`) to emit full AI prompt payloads to the log file for troubleshooting. Diff-based edits are always enabled now—if a patch conflicts, capture a fresh snapshot and retry rather than switching modes.
- **Tool traces**: Toggle **Show tool activity panel** in the Settings dialog (or set `TINKERBELL_TOOL_ACTIVITY_PANEL=1`) when you need the debug-only LangChain trace list; it's hidden by default to maximize editing space.
- **Scripts**: `src/tinkerbell/scripts/seed_examples.py` seeds demonstration docs; extend it when adding new showcase flows.
- **Token inspector**: `scripts/inspect_tokens.py` compares precise (tiktoken) vs. approximate counts so you can size prompts or reproduce the data in `benchmarks/phase0_token_counts.md`.

## Roadmap & contributing

Planned improvements (see `plan.md` for details):

- Multi-document tabs and workspace-aware retrieval.
- Markdown preview polish (themes, live sync scrolling).
- Additional tools (Git diffing, grammar checks, local vector stores).
- Packaging experiments (PyInstaller, MSIX) for one-click Windows installs.

Contributions are welcome! Please open an issue describing the change, keep PRs focused, and accompany code changes with tests/docs. The project is licensed under the MIT License (see `LICENSE`).
