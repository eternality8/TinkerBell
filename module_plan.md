# Module Implementation Plan

Below are per-module implementation notes derived from `plan.md`. Each entry lists responsibilities, key classes/functions, dependencies, and open considerations to streamline future development.

---

## `app.py`
- **Responsibilities**: bootstrap logging, read settings, initialize `QApplication` via `qasync`, start event loop, and open `MainWindow`.
- **Key functions/classes**:
  - `load_settings(path: Path) -> Settings`: read persisted configuration or create defaults.
  - `configure_logging(debug: bool)`.
  - `create_qapp(settings: Settings) -> qasync.QApplication`: wires asyncio event loop with Qt.
  - `main()`: orchestrates boot (settings, faiss warmup, window show, `asyncio.run`).
- **Dependencies**: `qasync`, `asyncio`, `services.settings`, `main_window`, `utils.logging`.
- **Notes**: ensure single-instance guard (optional), catch top-level exceptions and surface dialog + log.

## `main_window.py`
- **Responsibilities**: define `MainWindow(QMainWindow)` layout (splitter, menus, toolbars), connect editor/chat signals, manage status bar.
- **Key methods**:
  - `__init__(settings: Settings, ai_controller: AIController)`.
  - `create_actions()`, `create_menus()`, `create_toolbars()`.
  - `wire_signals()`: editor ↔ chat events, autosave timers.
  - `open_document(path)`, `save_document()`, `update_status(message)`.
- **Dependencies**: `editor.editor_widget`, `chat.chat_panel`, `widgets.status_bar`, `services.bridge`.
- **Notes**: anticipate tabbed editing; keep command routing centralized for undo/redo, AI actions, previews.

## `editor/document_model.py`
- **Responsibilities**: typed representation of document state (path, text, syntax mode, selection, hashes, dirty flag).
- **Key classes**:
  - `DocumentMetadata`: filename, format, language mode, created/updated timestamps.
  - `SelectionRange`: start/end positions (line/column + absolute offset helpers).
  - `DocumentState`: dataclass w/ text, metadata, selection, sha digest, undo stack info.
- **Methods**: `from_path(path)`, `update_text(new_text)`, `apply_patch(patch)`, `snapshot(delta_only=False)` returning dict for AI tools.
- **Dependencies**: `dataclasses`, `pydantic` or `attrs` for validation, `hashlib`.
- **Notes**: include diff generation helper for AI context (maybe `compute_diff(previous: str) -> str`).

## `editor/editor_widget.py`
- **Responsibilities**: wrap text editor widget (prefer `QsciScintilla` w/ fallback), expose signals for edits, selections, file ops, integrate syntax themes.
- **Key classes/methods**:
  - `EditorWidget(QWidget)`: orchestrates editor component + optional preview toggle.
  - `load_document(document: DocumentState)` / `to_document()`.
  - `apply_ai_edit(action: EditDirective)` (insert/replace/annotate) w/ transaction + undo.
  - `request_snapshot() -> DocumentSnapshot` event.
  - `set_preview_mode(enabled: bool)` hooking to Markdown preview.
- **Dependencies**: `PySide6`, `Qsci`, `editor.syntax.*`, `services.bridge`.
- **Notes**: emit Qt signals such as `selectionChanged`, `textEdited`, `snapshotRequested(DocumentSnapshot)`; ensure async-safe updates via slots.

## `editor/syntax/markdown.py`
- **Responsibilities**: Markdown-specific highlighting, preview rendering hooks.
- **Key pieces**:
  - `MarkdownLexer(QsciLexerCustom)` or fallback highlighter.
  - `render_preview(text: str, theme: Theme) -> str` (HTML for preview pane).
  - `detect_frontmatter(text) -> dict` for metadata hints.
- **Dependencies**: `markdown-it-py` or `python-markdown`, `PySide6.QtWebEngineWidgets` (optional), `themes` module.
- **Notes**: keep preview optional to avoid heavy deps during MVP.

## `editor/syntax/yaml_json.py`
- **Responsibilities**: syntax highlighting + validation helpers for YAML/JSON.
- **Key functions**:
  - `YamlJsonLexer` configuration (styles, folding).
  - `validate_yaml(text)`, `validate_json(text)` returning structured errors for inspector tool.
- **Dependencies**: `ruamel.yaml`, `jsonschema` (if schema provided).
- **Notes**: share diagnostics with `FormatInspectorTool`.

## `editor/syntax/themes.py`
- **Responsibilities**: central theme definitions for editors + preview.
- **Key classes/functions**:
  - `Theme` dataclass (fonts, colors, highlight map).
  - `load_theme(name)`, `available_themes()`, `apply_theme(widget, theme)`.
- **Notes**: store defaults matching Qt palettes; allow overrides via settings.

## `chat/chat_panel.py`
- **Responsibilities**: UI panel containing chat history view, composer, suggestions, tool trace view.
- **Key classes/methods**:
  - `ChatPanel(QWidget)` with nested `QListView`/`QTextBrowser` for history.
  - `append_user_message(content, selection_summary)`.
  - `append_ai_message(message: ChatMessage, streaming: bool=False)`.
  - `emit_request(prompt: str)` signal hooking to AI controller.
  - `show_tool_trace(step: ToolTrace)` for ReAct transparency.
- **Dependencies**: `chat.message_model`, `chat.commands`, `services.bridge` for document snapshots.
- **Notes**: ensure asynchronous streaming updates (append tokens as they arrive).

## `chat/message_model.py`
- **Responsibilities**: data models for chat messages, tool traces, metadata persisted per session.
- **Key classes**:
  - `ChatRole(Enum)` {user, assistant, system, tool}.
  - `ChatMessage`: role, content, timestamp, attachments, directives.
  - `ToolTrace`: tool name, input, output summary, duration.
- **Methods**: serialization helpers `to_dict()`, `from_dict()`, `truncate_tokens(max_tokens)`.
- **Notes**: integrate with LangChain memory format.

## `chat/commands.py`
- **Responsibilities**: define structured commands AI can issue (insert, replace, summarize, annotate) plus validation.
- **Key items**:
  - `ActionType(Enum)`; `EditDirective` dataclass {action, target_range, content, rationale}.
  - `parse_agent_payload(payload: str|dict) -> EditDirective|PlainText`.
  - `validate_directive(directive, doc_state) -> ValidationResult`.
- **Notes**: include schema (JSON Schema) shared with guard node + tests.

## `ai/client.py`
- **Responsibilities**: wrap `openai` SDK with configurable base URL/model, handle streaming + retries.
- **Key classes/methods**:
  - `AIClient`: init with `Settings`, holds `AsyncOpenAI` client.
  - `stream_chat(messages, tools=None, **kwargs)` -> async generator.
  - `list_models()` to validate credentials.
  - `with_proxy(endpoint, key)` context manager (optional).
- **Dependencies**: `openai >= 1.0`, `tenacity` for retries, `asyncio`.
- **Notes**: support SSE/responses for OpenAI-compatible endpoints.

## `ai/prompts.py`
- **Responsibilities**: system prompts/templates for agent instructions, doc summaries, tool outputs.
- **Key functions**:
  - `base_system_prompt()` returning string with rules.
  - `format_user_prompt(user_prompt, doc_snapshot, safety_hints)`.
  - `tool_contract_prompt(tool_specs)`.
- **Notes**: store as multiline constants or `jinja2` templates; include placeholders for tokens limit.

## `ai/tools/document_snapshot.py`
- **Responsibilities**: LangChain `Tool` exposing current document snapshot to agent.
- **Key functions**:
  - `class DocumentSnapshotTool(BaseTool)` with `name`, `description`, `args_schema`.
  - `_run`/`_arun` retrieving `DocumentState.snapshot(delta_only: bool)`.
- **Dependencies**: `langchain.tools`, `services.bridge`, `editor.document_model`.
- **Notes**: enforce size limits, redact secrets, cache last snapshot to avoid heavy recompute.

## `ai/tools/document_edit.py`
- **Responsibilities**: apply AI-proposed edits safely through bridge.
- **Key elements**:
  - `DocumentEditTool(BaseTool)` that accepts JSON patch/diff.
  - Validation: ensure action + range + content, confirm diff preview.
  - On success, return diff summary + undo reference.
- **Dependencies**: `services.bridge`, `chat.commands`, `langchain` tool base.
- **Notes**: guard rails (require rationale, length thresholds, multi-step confirm).

## `ai/tools/search_replace.py`
- **Responsibilities**: scoped find/replace with regex/literal options, optional dry-run preview.
- **Key methods**:
  - Args schema: `pattern`, `replace`, `is_regex`, `scope` (selection/current doc).
  - Implementation uses `re` or `regex` module with compiled patterns.
  - Returns count + snippet preview; optionally auto-apply via edit tool.
- **Notes**: ensure timeouts and match limits to avoid freezing.

## `ai/tools/validation.py`
- **Responsibilities**: verify YAML/JSON snippets before insertion.
- **Key functions**:
  - `validate_snippet(text: str, format: Literal["yaml","json"])` returning success/errors.
  - Optional schema argument for JSON Schema.
- **Dependencies**: `ruamel.yaml`, `jsonschema`, `editor.syntax.yaml_json`.
- **Notes**: integrate with guard pipeline to block invalid edits.

## `ai/agents/graph.py`
- **Responsibilities**: define LangGraph flow (ingest → planner → tool loop → guard → respond).
- **Key constructs**:
  - `build_agent_graph(client: AIClient, tools: list[BaseTool], memory: MemoryManager)` returning `CompiledGraph`.
  - Nodes: `ingest_node`, `agent_node`, `tool_node`, `guard_node`.
  - Edges/resolvers specifying transitions + retries.
- **Notes**: log ReAct traces, enforce max iterations/timeouts.

## `ai/agents/executor.py`
- **Responsibilities**: provide `AgentExecutor` façade to UI.
- **Key methods**:
  - `AIController`: `async run_chat(prompt, doc_snapshot)` using graph; `cancel()` for stop button.
  - `register_tool(tool)`, `update_settings(settings)`.
- **Dependencies**: `langchain.agents`, `langgraph`, `ai.client`, `chat.commands`.
- **Notes**: unify streaming callbacks to chat panel + status bar.

## `ai/memory/buffers.py`
- **Responsibilities**: conversation + document summary buffers.
- **Key classes**:
  - `ConversationMemory`: wrap `LangChainConversationBuffer` with token-aware trimming.
  - `DocumentSummaryMemory`: maintain rolling summary per document (maybe via map-reduce summarizer).
  - `MemoryStore`: persistence layer (JSON per doc) w/ load/save.
- **Notes**: ensure thread-safe access when called from async tasks.

## `services/bridge.py`
- **Responsibilities**: mediate between AI directives and Qt editor operations.
- **Key classes/methods**:
  - `DocumentBridge(QObject)`: holds reference to `EditorWidget` + document state.
  - `queue_edit(directive: EditDirective)`; internally marshals to main thread via signals.
  - `generate_snapshot(delta_only: bool=False)` for tools + agent context.
  - `emit_diff_summary(old_text, new_text)` for chat log.
- **Notes**: integrate undo/redo, conflict detection (if doc changed since directive computed).

## `services/settings.py`
- **Responsibilities**: load/save settings, manage API keys, encryption.
- **Key classes/functions**:
  - `Settings` dataclass (paths, theme, ai endpoint/key/model, autosave interval, preview toggles).
  - `SettingsStore`: `load()`, `save(settings)`, `set_api_key(key)` with encryption via Windows Credential Locker or `cryptography` Fernet.
  - `SettingsDialog` hooks (if not in widgets).
- **Notes**: watch for cross-platform secrets; provide validation + defaults.

## `widgets/status_bar.py`
- **Responsibilities**: custom status bar showing cursor position, doc format, AI status spinner, memory usage.
- **Key classes/methods**:
  - `StatusBar(QStatusBar)`.
  - `update_cursor(line, column)`, `set_message(text, timeout)`, `set_ai_state(state: Enum)`.
- **Notes**: include small task indicators for autosave + agent activity.

## `widgets/dialogs.py`
- **Responsibilities**: shared dialogs (open/save, API settings, confirmation prompts, validation results).
- **Key components**:
  - `OpenFileDialog`, `SaveFileDialog` wrappers customizing filters.
  - `SettingsDialog(QDialog)` (if not in services): binds to `Settings` dataclass, includes API key input + validation button.
  - `ValidationErrorsDialog(errors)`.
- **Notes**: consider asynchronous validation (model list ping) with spinner.

## `utils/file_io.py`
- **Responsibilities**: file read/write helpers with encoding detection, autosave temp files, change detection.
- **Key functions**:
  - `read_text(path) -> str`, `write_text(path, text)` with atomic writes.
  - `detect_format(path, text) -> FormatEnum`.
  - `ensure_autosave_dir()` + `write_autosave(document)`.
- **Notes**: use `pathlib`, handle BOM + newline normalization.

## `utils/logging.py`
- **Responsibilities**: configure structured logging, integrate with Qt + LangChain traces.
- **Key functions**:
  - `setup_logging(level, log_dir)`, `get_logger(name)`.
  - `qt_message_handler(mode, context, message)` to redirect Qt warnings.
- **Notes**: consider rotating logs (e.g., `logging.handlers.RotatingFileHandler`).

## `utils/telemetry.py` (optional)
- **Responsibilities**: capture anonymous usage metrics (feature toggles) respecting opt-in.
- **Key functions**:
  - `TelemetryClient`: `track_event(name, props)`, `flush()`.
  - `is_enabled(settings)` to guard calls.
- **Notes**: keep stubbed until privacy policy defined.

## `tests/`
- **Focus areas**:
  - `test_agent.py`: simulate LangGraph runs w/ mocked tools; ensure directives produced.
  - `test_bridge.py`: verify edits apply correctly + handle conflicts.
  - `test_editor_widget.py`: GUI tests using `pytest-qt` for menu actions, selection signals, markdown preview toggles.
- **Helpers**: fixtures for sample documents, fake AI responses.

## `scripts/seed_examples.py`
- **Responsibilities**: populate `assets/sample_docs` + conversation transcripts for demos.
- **Key steps**:
  - CLI interface (`argparse`) to copy sample files.
  - Optionally call AI client to precompute example chats (with stub credentials).
- **Notes**: keep deterministic; useful for automated QA.
