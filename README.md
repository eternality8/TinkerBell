# TinkerBell

TinkerBell is an agent-aware desktop text editor focused on Markdown and other structured text formats. The application pairs a PySide6-based editor pane with an AI chat/agent panel capable of reading and modifying the current document via structured actions.

## Project Overview

- **UI Toolkit:** PySide6 with QSplitter-based layout for editor + chat panel.
- **Agent Stack:** OpenAI-compatible client, LangChain + LangGraph orchestration, and custom tools for document access/edits.
- **Async Runtime:** `asyncio` bridged into Qt via `qasync` so streaming responses do not block the UI thread.
- **Persistence:** Local filesystem autosave plus JSON metadata describing chat sessions and document summaries.

## Repository Layout

```
README.md
plan.md
module_plan.md
pyproject.toml
src/tinkerbell/...               # application source code
tests/...                        # pytest + pytest-qt suites
assets/icons|qss|sample_docs     # static resources
```

Refer to `plan.md` and `module_plan.md` for deeper architectural notes.

## Quick Start (planned)

1. Install [uv](https://docs.astral.sh/uv/) (and ensure Python 3.11+ is available on your PATH).
2. Run `uv sync` to create/update the virtual environment.
3. Launch the app with `uv run tinkerbell` once the UI bootstrap is implemented.

> **Status:** This repository currently contains scaffolding and stubs aligned with the design plan. Implementations will arrive in subsequent milestones.
