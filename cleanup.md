# Cleanup Targets

## AI controller monolith & rebuild cost (`src/tinkerbell/ai/orchestration/controller.py`)
- `AIController` now spans ~2.3k lines and blends budgeting, telemetry, LangGraph orchestration, subagent plumbing, and tool registry CRUD. The class is effectively untestable in isolation and small changes require scrolling through thousands of lines.
- Every call to `register_tool()` or `unregister_tool()` triggers `_rebuild_graph()`, so the default tool bootstrap rebuilds the LangGraph 7–10 times on startup. Profiling during `register_default_tools()` shows ~250ms wasted in repeated graph builds.
- Action: extract budgeting/telemetry/subagent helpers into dedicated modules and add a batch registration path (e.g., `register_tools(dict)` or a `suspend_rebuild` context manager) so the graph only recompiles once per batch.

## UI main window sprawl (`src/tinkerbell/ui/main_window.py`)
- `MainWindow` ballooned past 4,100 lines and owns unrelated concerns: Qt wiring, telemetry, embeddings, importer error handling, AI tool registration, and subagent orchestration. The file mixes Qt stubs, dataclasses, business logic, and async AI coordination, making it nearly impossible to reason about or test.
- Action: split the class into focused controllers (window chrome, AI turn review, embeddings panel, telemetry HUD) and move the long dataclass definitions into their own modules. Add unit coverage for the extracted services before refactoring UI glue.

## Tool registry error masking (`src/tinkerbell/ai/tools/registry.py`)
- `register_default_tools()` wraps the entire wiring sequence in a single `try/except`. If any one tool constructor raises (e.g., `DocumentSnapshotTool` due to a bridge change), the exception is swallowed and **all** downstream tools silently fail to register.
- Action: register each tool inside its own guarded block (or collect errors and raise one aggregated exception) so partial failures are visible and other tools still wire up.

