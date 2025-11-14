## TinkerBell – Implementation Plan

### 1. Goals & Scope
- Provide a desktop text editor focused on Markdown and structured-text formats (Markdown, plain text, YAML, JSON).
- Pair the editor with an “agentic” AI chat panel that can read/write the current document through natural-language requests.
- Allow the AI to use any OpenAI-compatible API endpoint + API key supplied by the user at runtime.
- Keep the stack Python-centric for easier packaging/distribution (Windows-first, cross-platform later).

### 2. Tech Stack Decisions
| Layer | Choice | Rationale |
| --- | --- | --- |
| **Desktop shell & UI toolkit** | `PySide6` (Qt for Python) | Native desktop look, built-in split layouts, good text widget support, permissive license |
| **Text editing component** | `QsciScintilla` via `PyQt6.Qsci` or fallback to `QPlainTextEdit` w/ syntax highlighting | Advanced editing features, folding, line numbers; fallback keeps dependency lighter |
| **AI client** | `openai` (>=1.0) SDK w/ configurable base URL + key | Supports OpenAI + compatible providers (Anthropic, local proxies) |
| **Agent framework** | `LangChain` + `langgraph` | Built-in tool abstractions, memory, and guardrails for multi-step agent plans |
| **Async orchestration** | `asyncio` + `qasync` bridge | Needed to run async chat calls without freezing UI |
| **State management** | Local Python dataclasses / pydantic models | Typed document + chat session state |
| **Persistence** | Local filesystem autosave, JSON for session metadata | Simple, portable |
| **Retrieval store** | `faiss-cpu` (via LangChain retrievers) | Lightweight local vector DB for document memory + snippets |
| **Packaging** | `uv` for dependency management/sync; optional `pyinstaller` build | Fast, reproducible env + single-file distributable |
| **Testing** | `pytest`, `pytest-qt` | GUI + logic unit tests |

### 3. High-Level Architecture
1. **Main Window** – `QMainWindow` hosting a horizontal splitter.
2. **Editor Panel (left)** – rich text widget with tabs (optional), file tree, format detection, Markdown preview toggle.
3. **AI Chat Panel (right)** – conversation list, message composer, agent suggestions, streaming responses.
4. **AI Controller** – wraps OpenAI-compatible API, handles tool-style requests (insert, replace, summarize, outline).
5. **Document Bridge** – exposes the current document (full text + selections + metadata) to the AI controller and applies modifications returned by the agent.
6. **Settings / Auth** – modal dialog for API endpoint + key + model selection; persisted encrypted (Windows Credential Locker / simple obfuscation initially).

### 4. Core Features & Contracts
- **Editing Basics**: open/save files, autosave drafts, syntax highlighting, search/replace, Markdown preview.
- **AI Actions**: user prompts referencing current doc; AI can request operations (insert text, rewrite selection, annotate). Contract: AI responses encapsulated in structured payloads `{action, target, content}`.
- **Agent Context**: always send current doc summary + viewport selection + user prompt; configurable tokens limit.
- **Error Handling**: non-blocking notifications, retry button, offline fallback.
- **Extensibility**: plugin-like registry for future tools (e.g., grammar check, snippet library).

### 5. Agent Tooling Strategy
- **Agent type**: LangChain `AgentExecutor` orchestrated via `langgraph` for explicit planning + retries; ReAct reasoning traces logged in the chat panel so users can audit tool calls.
- **Tool registry**: each capability is a LangChain `Tool` with async implementations:
	- `DocumentSnapshotTool` – returns current document text, metadata, selection, diffs.
	- `DocumentEditTool` – validates JSON patch (insert/replace/comment) before routing through the bridge.
	- `SearchReplaceTool` – scoped regex or literal replacements with dry-run preview.
	- `FormatInspectorTool` – uses `ruamel.yaml`/`jsonschema` to validate YAML/JSON before insertion.
	- `KnowledgeTool` – optional retrieval augmented generation hooking into a local vector store (FAISS) seeded with help docs/snippets.
- **Memory & state**: LangChain ConversationBuffer for short context + summarized long-term memory persisted per document; tool outputs truncated and stored in message metadata for replay.
- **Safety rails**: guard function that refuses destructive edits unless agent provides diff + rationale; rate limiting + timeout wrappers around each tool.
- **Extensibility**: future tools (Git integration, file tree ops) register through a plugin manifest consumed by the LangChain tool loader.

### 6. Proposed File/Directory Structure
```
tinkerbell/
├─ plan.md
├─ README.md                      # project overview & setup
├─ pyproject.toml / uv.lock       # dependencies & scripts
├─ src/
│  ├─ tinkerbell/
│  │  ├─ __init__.py
│  │  ├─ app.py                  # entry point that boots Qt event loop
│  │  ├─ main_window.py          # QMainWindow layout + actions
│  │  ├─ editor/
│  │  │  ├─ document_model.py    # dataclasses for document state
│  │  │  ├─ editor_widget.py     # wrapper around text widget + helpers
│  │  │  └─ syntax/
│  │  │     ├─ markdown.py
│  │  │     ├─ yaml_json.py
│  │  │     └─ themes.py
│  │  ├─ chat/
│  │  │  ├─ chat_panel.py        # UI elements for chat history
│  │  │  ├─ message_model.py
│  │  │  └─ commands.py          # agent actions (insert, replace, summarize)
│  │  ├─ ai/
│  │  │  ├─ client.py            # OpenAI-compatible client wrapper
│  │  │  ├─ prompts.py           # system prompts/templates
│  │  │  ├─ tools/
│  │  │  │  ├─ document_snapshot.py
│  │  │  │  ├─ document_edit.py
│  │  │  │  ├─ search_replace.py
│  │  │  │  └─ validation.py
│  │  │  ├─ agents/
│  │  │  │  ├─ graph.py          # langgraph definitions
│  │  │  │  └─ executor.py       # LangChain AgentExecutor wiring
│  │  │  └─ memory/
│  │  │     └─ buffers.py        # conversation + summary memory helpers
│  │  ├─ services/
│  │  │  ├─ bridge.py            # Document ↔ AI synchronization logic
│  │  │  └─ settings.py          # API key storage, model prefs
│  │  ├─ widgets/
│  │  │  ├─ status_bar.py
│  │  │  └─ dialogs.py           # open/save, settings, confirmation
│  │  └─ utils/
│  │     ├─ file_io.py
│  │     └─ logging.py
│  └─ scripts/
│     └─ seed_examples.py        # sample docs/conversations
├─ tests/
│  ├─ conftest.py
│  ├─ test_agent.py
│  ├─ test_bridge.py
│  └─ test_editor_widget.py
└─ assets/
	├─ icons/
	├─ qss/
	└─ sample_docs/
```

### 7. Module Responsibilities
1. **`app.py`** – initializes logging, loads settings, sets up `QApplication`/`qasync`, instantiates `MainWindow`.
2. **`main_window.py`** – configures menus, toolbar actions, splitter layout, event wiring between editor + chat.
3. **Editor module** – handles file IO, syntax detection, line numbers, selection events, preview toggle.
4. **Chat module** – view models for chat history, emits requests to AI agent, displays streaming tokens.
5. **AI module** – `client.py` configures transport; LangChain executor + `langgraph` manage planning, retries, memory summaries, and structured outputs.
6. **Agent tools & memory** – each tool module enforces validation (e.g., schema-check edits, token limits). Memory helpers persist conversation buffers + document summaries for retrieval tools.
7. **Bridge service** – listens for AI commands (replace selection, append at cursor, comment block) and mutates the editor safely (Qt signal/slot) while surfacing diffs back to the agent.
8. **Settings service** – secure storage, validation, model list caching, UI dialog to update credentials.
9. **Utilities** – logging wrappers and temporary file helpers.

### 8. AI Interaction Flow
1. User selects text or positions cursor and asks AI for an action.
2. Chat panel emits `ChatRequest` containing prompt + editor snapshot (full text, selection, file metadata, diff since last request).
3. LangGraph “ingest” node packages system prompt (tool contract, safety rules), user prompt, doc summary chunks, and memory snippets into the agent scratchpad.
4. LangChain ReAct agent reasons step-by-step, invoking registered tools (`DocumentSnapshotTool`, `SearchReplaceTool`, etc.) until it produces a final plan or edit directive. Tool inputs/outputs stream back to the UI for transparency.
5. Final response is validated by the guard node; if it contains directives (e.g., `{"action":"replace","range":"line 10-20","content":"..."}`), it moves to the bridge queue; otherwise it is shown as plain text.
6. Bridge applies edit via Qt signals on the main thread, updates chat history with diff summary, and records outcome in LangChain memory.
7. Errors/safety: timeouts -> toast + log; invalid responses trigger automatic retry with shorter context or fallback to plain text insertion.

### 9. Settings & Security
- Store API key encrypted using `cryptography` + key derived from OS credential store; otherwise, let user enter per session.
- Allow custom base URL/model names; validate with a quick `/models` call.
- Config file (YAML) for default preferences (theme, font, autosave interval, default model).

### 10. Testing Strategy
- **Unit tests**: LangChain graph transition tests (happy path + guardrail failures), bridge edit application, file IO edge cases.
- **Tool tests**: each tool module gets schema validation tests plus property-based tests for diff/replace invariants.
- **GUI tests**: `pytest-qt` to ensure menu actions trigger expected signals, chat panel updates without blocking, tool traces render correctly.
- **Integration**: mock OpenAI-compatible endpoint to simulate streaming tokens + multi-tool runs; verify text replacement + undo stack coherence.
- **Regression**: sample Markdown/YAML docs to ensure syntax highlighting loads without crash.

### 11. Delivery Roadmap
1. Bootstrap project (uv, linting, formatter).
2. Implement barebones editor + file IO.
3. Add chat panel UI and fake agent to prove layout.
4. Wire real AI client with streaming + structured commands.
5. Harden document bridge, add undo/redo integration.
6. Polish UX (themes, icons, markdown preview, status indicators).
7. Package for Windows (PyInstaller) and add setup guide.

### 12. Open Questions / Next Steps
- How opinionated should the agent be (auto tool invocation vs confirm-before-apply)?
- Should we support collaborative editing (multi-tab, remote sync)?
- Need to define a schema for AI commands (JSON schema + validation) early to avoid parsing drift.
- What permission + auditing model should govern destructive tools (e.g., multi-file edits, shell commands)?
- Do we need local vector-store caching per workspace or a shared global index?

