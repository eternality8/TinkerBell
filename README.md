# TinkerBell

AI‑assisted text editor built with PySide6 and modern LLM APIs.

## Highlights

- **Multi-document tabbed UI** – open, edit, and compare several files at once
- **Persistent chat** – conversation history survives across sessions
- **Review mode** – inspect AI‑proposed changes before accepting or rejecting
- **Streaming AI responses** – see results as they arrive
- **Context-aware tools** – AI can read, search, and modify your documents

## Architecture

```
src/tinkerbell/
├── app/              # Application entry, main window, tabs
├── ai/               # AI client, turn management, context budgeting
├── chat/             # Chat panel, commands, formatting
├── document/         # Document model, storage, importers
├── editor/           # Text editor widget, syntax highlighting
├── orchestration/    # Multi-step AI pipelines (analyze → prepare → execute → finish)
├── review/           # Diff display, accept/reject workflows
├── services/         # Container, caching, telemetry
├── settings/         # User preferences and model configuration
├── tools/            # AI-callable tools (navigation, editing, search)
└── utils/            # Shared helpers
```

## Getting Started

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python      | 3.9+    |
| uv          | latest  |

### Clone & Install

```bash
git clone https://github.com/eternality8/TinkerBell.git
cd TinkerBell
uv sync
```

### Launch

```bash
uv run tinkerbell
```

### Optional Extras

For development tools:

```bash
uv sync --all-extras
```

## Configuring AI Access

Open **Settings → AI** and enter credentials for your preferred provider:

| Provider  | Required Settings       |
|-----------|-------------------------|
| OpenAI    | API key                 |
| Anthropic | API key                 |
| Google    | API key                 |
| OpenRouter| API key                 |
| Local     | Base URL (e.g., Ollama) |

The editor supports any OpenAI-compatible endpoint.

## AI Tools

The AI assistant can use these tools during a conversation:

### Navigation & Reading

| Tool | Description |
|------|-------------|
| `get_tab_info` | Get active tab metadata |
| `list_tabs` | List all open documents |
| `read_document_lines` | Read specific line ranges |
| `read_selection` | Get currently selected text |
| `search_document` | Regex search within a document |

### Editing

| Tool | Description |
|------|-------------|
| `edit_document` | Replace text ranges with new content |
| `apply_edits` | Batch multiple edits atomically |

### Memory & Context

| Tool | Description |
|------|-------------|
| `write_to_memory` | Store information for later |
| `read_from_memory` | Retrieve stored information |
| `list_memory_buffers` | List available memory buffers |

## Safety & Privacy

- **Local-first** – all data stays on your machine
- **Review before apply** – AI edits require explicit approval
- **No telemetry** – usage data is never sent externally
- **Bring your own keys** – API credentials are stored locally

## Testing

Run the full test suite:

```bash
uv run pytest
```

Run a specific test file:

```bash
uv run pytest tests/test_orchestrator.py -v
```

Run with coverage:

```bash
uv run pytest --cov=src/tinkerbell --cov-report=html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `uv run pytest` to ensure tests pass
5. Submit a pull request

## License

MIT – see [LICENSE](LICENSE) for details.
