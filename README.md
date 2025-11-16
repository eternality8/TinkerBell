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

`assets/sample_docs/` contains Markdown and YAML snippets used by tests and demos. Use **File → Open…** to load them or seed your own via `src/tinkerbell/scripts/seed_examples.py`.

### Optional extras (tokenizers)

Install the optional tokenizer extra (`tiktoken>=0.12,<0.13`) whenever you need model-authentic counts or want to run token benchmarks:

```powershell
uv sync --extra ai_tokenizers
```

> `tiktoken` currently requires a Rust toolchain for Python 3.13 on Windows. Install [rustup](https://rustup.rs/) or run the command under Python 3.12 until official wheels are published. Without the extra, the editor automatically falls back to the deterministic byte-length estimator.

## Configuring AI access

You can supply OpenAI-compatible credentials in three interchangeable ways:

1. **Settings dialog** – Press `Ctrl+,` or use **Settings → Preferences…** to enter a base URL, API key, model name, and retry/backoff settings. Keys are stored with DPAPI (Windows) or Fernet (cross-platform) via `SettingsStore`. The dialog also exposes **Max Context Tokens** and **Response Token Reserve** so you can cap how much history is sent while guaranteeing ~16k tokens of headroom for replies.
2. **Environment variables** – Set any subset of the following before launching the app:
	 - `TINKERBELL_API_KEY`
	 - `TINKERBELL_BASE_URL` (e.g., `https://api.openai.com/v1` or your proxy)
	 - `TINKERBELL_MODEL` (defaults to `gpt-4o-mini`)
	 - `TINKERBELL_THEME` / `TINKERBELL_ORGANIZATION`
	 - `TINKERBELL_DEBUG_LOGGING` (set to `1`/`true` to force verbose logging and prompt dumps)
	 - `TINKERBELL_REQUEST_TIMEOUT` (seconds before an AI request fails; defaults to `90`)
3. **Programmatic injection** – Instantiate `Settings` or `ClientSettings` yourself if you embed TinkerBell in a larger Python workflow.

Test credentials via the **Refresh Snapshot** or a simple “Say hello” chat message. Failures are surfaced in the chat panel and status bar, and logs are written to your platform-specific temp directory.

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
| `SearchReplaceTool` | `tinkerbell.ai.tools.search_replace` | Provides regex/literal transforms with optional dry-run previews before edits are enqueued.
| `ValidationTool` | `tinkerbell.ai.tools.validation` | Checks YAML/JSON snippets via `ruamel.yaml`/`jsonschema` prior to insertion.
| `ListTabsTool` | `tinkerbell.ai.tools.list_tabs` | Enumerates open tabs (`tab_id`, title, path, dirty flag) so agents can target any document without stealing focus.
| `Memory Buffers` | `tinkerbell.ai.memory.buffers` | Maintains conversation + document summaries so prompts stay concise without losing context.

You can register custom tools at runtime via `AIController.register_tool`, and the LangGraph plan automatically picks them up.

### Tool parameter reference

- **`document_snapshot`** — accepts `delta_only` (bool) to request only changed fields, `tab_id` to target a non-active document, `source_tab_ids` to batch additional read-only snapshots, and `include_open_documents` to embed lightweight metadata for every tab. Each response still carries the latest diff summary and digest so agents can detect drift.
- **`document_edit`** — consumes either a native `EditDirective` or a JSON/mapping payload matching the schema exposed during registration. Prefer `action="patch"` plus a unified diff and `document_version`; legacy `insert`/`replace` actions remain available for small, cursor-relative tweaks. Provide `tab_id` whenever the edit should be applied to a background tab.
- **`document_apply_patch`** — requires `content` (replacement text) and optionally `target_range`, `document_version`, `rationale`, `context_lines`, and `tab_id`. It snapshots the targeted document, builds a diff for the requested range, and immediately sends it through `document_edit` so diff construction + application happen in one step.
- **`diff_builder`** — accepts `original`, `updated`, optional `filename`, and optional `context` (default 3) to produce a ready-to-send unified diff string compatible with `document_edit` patch directives.
- **`search_replace`** — parameters include `pattern`, `replacement`, `is_regex`, `scope` (`document` or `selection`), `dry_run`, `max_replacements`, `match_case`, and `whole_word`. With `dry_run=True`, the tool performs no edit and instead returns a preview plus match counts; otherwise it enqueues a single replace directive scoped to the resolved range and refreshes the document version.
- **`validate_snippet`** — requires `text` and `fmt` (`yaml`, `yml`, or `json`) and responds with a `ValidationOutcome` describing the first issue along with a count of remaining problems so agents can deliver actionable feedback.
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

- `ContextUsageEvent` objects capture per-turn prompt tokens, tool tokens, response reserves, tool names, and timestamps.
- Toggle **Settings → Debug → Token logging enabled** (or set `settings.debug.token_logging_enabled` programmatically) to surface live totals in the status bar and capture up to *N* events (default 200).
- Retrieve the rolling buffer via `AIController.get_recent_context_events()` for assertions, diagnostics, or custom sinks.

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
