# Cleanup Implementation Plan

## Tasklist
- [x] AI controller modularization & batch rebuild
   - [x] Baseline coverage for budgeting, telemetry, and subagent policies
   - [x] Extract helper services and wire through dependency injection
   - [x] Introduce batch registration API and update call sites
   - [x] Add regression tests for graph rebuild counts and controller behaviors
- [x] Main window decomposition
   - [x] Capture current widget wiring and signal flows
      - Documented emitter/handler pairs, controller ownership, and tool wiring in `docs/operations/main_window_wiring.md` so future refactors have an authoritative map.
   - [ ] Extract controllers/services for telemetry, embeddings, AI review, importer handling
      - [x] Embeddings controller extracted (`src/tinkerbell/ui/embedding_controller.py`) and wired into `MainWindow`
      - [x] Telemetry overlay/controller
         - Delivered via `src/tinkerbell/ui/telemetry_controller.py`, which now owns status-bar memory updates, compaction stats, and subagent telemetry wiring; `MainWindow` delegates to it and tests cover the controller surface.
         - Follow-up: keep extending controller as future telemetry widgets emerge (e.g., budget HUD tweaks) to prevent logic from slipping back into the window class.
      - [x] AI review flow controller
         - Delivered via `src/tinkerbell/ui/ai_review_controller.py`, which now owns pending-turn envelopes, session bookkeeping, composer restoration, overlay clearing, and status-bar review controls.
         - `MainWindow` delegates begin/finalize/abort paths, active-tab orphan tracking, and accept/reject handlers to controller APIs; the window no longer mutates `_pending_turn_review` or review UI state directly.
         - Updated `tests/test_main_window.py` to exercise the controller via `window._review_controller` helpers.
      - [x] Importer handling controller
         - Delivered via `src/tinkerbell/ui/import_controller.py`, encapsulating dialog prompts, error/status handling, tab creation, and workspace persistence for imports.
         - `MainWindow` delegates import actions and exposes a compatibility shim so tests can still swap the underlying `FileImporter` facade.
   - [x] Move dataclasses into dedicated modules with tests
      - Introduced `src/tinkerbell/ui/models/actions.py`, `window_state.py`, and `tool_traces.py`, updating `tinkerbell.ui` exports plus `MainWindow` and dependent controllers to import from the new modules.
      - TODO follow-up: add focused unit coverage for the extracted models (hash/serialization helpers, pending trace accumulation) once the broader migration settles.
   - [x] Reassemble a slimmer `MainWindow` and backfill UI/unit coverage
      - Added `src/tinkerbell/ui/window_shell.py` to own splitter/menu/tool wiring, shrinking `MainWindow` by ~180 lines and confining Qt glue to a dedicated helper.
      - Updated `tests/test_main_window.py` (via existing suite) to validate the refactor; `pytest tests/test_main_window.py` passes post-change.
- [x] Tool registry error handling hardening
  - [x] Rework registration flow to scope exceptions per tool
     - Added `ToolRegistrationFailure` + `ToolRegistrationError`, wrapping each tool factory/register call so one failure no longer aborts the batch.
  - [x] Aggregate/emit failures while continuing healthy registrations
     - `MainWindow` now surfaces partial failures via status + chat notices while healthy tools keep working; registry raises aggregated errors for upstream telemetry/logging.
  - [x] Extend tests to cover partial failure scenarios
     - `tests/test_ai_tools.py` now exercises both the happy path and a partial-failure controller stub to ensure aggregation works.

---

## 1. AI Controller Monolith & Rebuild Cost (`src/tinkerbell/ai/orchestration/controller.py`)

### Status (Nov 18, 2025)
- Extracted `BudgetManager`, `TelemetryManager`, and `SubagentRuntimeManager` helpers plus supporting modules under `src/tinkerbell/ai/orchestration/`, slimming the controller to primarily coordinate orchestration and lifecycle.
- Added `register_tools()` and the `suspend_graph_rebuilds()` context manager so LangGraph rebuilds occur once per batch; `register_default_tools()` now uses the suspension guard to avoid repeated compilation.
- Refreshed tests in `tests/test_agent.py` to cover batched registration rebuild counts and controller behaviors, and ran `pytest tests/test_agent.py` to validate the refactor.
- Telemetry and budget enforcement now run through the extracted managers, improving isolation and enabling future focused unit coverage.

### Current Pain Points
- ~2.3k-line `AIController` mixes budgeting, telemetry, LangGraph orchestration, subagent plumbing, and tool registry CRUD, making the class untestable and risky to change.
- `register_tool()`/`unregister_tool()` each trigger `_rebuild_graph()`, causing 7–10 LangGraph rebuilds (~250ms) during startup when default tools register sequentially.

### Objectives
1. Reduce the responsibilities handled directly by `AIController` to orchestration and lifecycle coordination.
2. Provide a batch registration path (API or context manager) so LangGraph rebuilds only once per batch.
3. Improve testability with unit coverage around the extracted services and controller behaviors.

### Proposed Architecture & Steps
1. **Baseline Behavior & Tests**
   - Capture current interactions (budgeting decisions, telemetry emission, subagent routing, graph rebuild counts) via dedicated tests in `tests/test_agent.py`, `tests/test_ai_client.py`, or new unit suites under `tests/test_ai_controller.py` to ensure refactor safety.
2. **Responsibility Extraction**
   - Introduce modules:
     - `budgeting_service.py`: encapsulate cost tracking and limit enforcement.
     - `telemetry_service.py`: own telemetry batching and submission.
     - `subagent_manager.py`: wrap LangGraph subagent plumbing helpers currently embedded in the controller.
   - Update `AIController` to depend on these via constructor injection so behaviors remain mockable.
3. **Tool Registration API**
   - Add `register_tools(mapping: dict[str, ToolSpec])` and/or `suspend_rebuild()` context manager that defers `_rebuild_graph()` until exiting the batch.
   - Update existing bootstrap flows (e.g., `register_default_tools()` and UI bindings) to use the batch path.
4. **Graph Rebuild Optimization**
   - Ensure `_rebuild_graph()` tracks dirty state and only executes when necessary.
   - Instrument with telemetry counters to confirm only one rebuild occurs per batch.
5. **Testing & Validation**
   - Add regression tests verifying a single rebuild during batch registration and correct handling of individual register/unregister calls.
   - Mock extracted services in controller tests to assert orchestration logic without long files.

### Deliverables
- New service modules under `src/tinkerbell/ai/orchestration/` (or `/services/`) with unit tests.
- Updated controller with slimmer responsibilities and documented batch registration workflow.
- Telemetry metrics validating reduced rebuild counts.

### Risks & Mitigations
- **Risk:** Hidden coupling within controller may surface missing dependencies. *Mitigation:* incremental extraction with feature flags and comprehensive tests before removal.
- **Risk:** Downstream code may rely on side effects of immediate `_rebuild_graph()`. *Mitigation:* maintain default immediate rebuild outside batches and document new API expectations.

---

## 2. UI Main Window Sprawl (`src/tinkerbell/ui/main_window.py`)

### Current Pain Points
- `MainWindow` exceeds 4,100 lines and mixes Qt setup, telemetry, embeddings, importer error handling, AI tool wiring, and subagent coordination.
- Dataclasses and business logic coexist with UI glue, hindering readability and tests.

### Objectives
1. Split `MainWindow` into focused controllers/services for discrete concerns (window chrome, AI turn review, telemetry HUD, embeddings panel, importer handling).
2. Relocate large dataclasses into domain modules under `src/tinkerbell/ui/` or `src/tinkerbell/chat/` with their own tests.
3. Increase unit coverage—especially for extracted services—before refactoring UI glue.

### Proposed Architecture & Steps
1. **Map Current Responsibilities**
   - Document widget hierarchy, signal/slot connections, and service dependencies (AI controller, tool registry, telemetry) to understand separation boundaries.
2. **Introduce Feature Controllers**
   - Create modules such as `window_shell.py`, `embeddings_controller.py`, `telemetry_overlay.py`, `ai_review_panel.py`, and `importer_errors_controller.py` to own logic currently embedded in `MainWindow`.
   - Each controller exposes a small interface consumed by the slimmed `MainWindow`.
3. **Move Dataclasses Out**
   - Extract long dataclass definitions to `src/tinkerbell/ui/models/*.py`. Provide serialization/unit tests verifying behavior independently of Qt widgets.
4. **Refactor `MainWindow`**
   - Reduce the class to orchestrating controllers, wiring signals, and delegating to services.
   - Ensure `MainWindow` remains the integration point but not the logic owner.
5. **Testing Strategy**
   - Add Qt-less unit tests for the new controllers using dependency injection/mocks.
   - Extend existing UI tests (`tests/test_main_window.py`, `tests/test_widgets_*`) to assert controllers integrate correctly.

### Deliverables
- New controller/service modules with tests.
- Slimmed `MainWindow` file (<1,500 lines target) focusing on composition.
- Updated documentation or diagrams describing the UI architecture split.

### Risks & Mitigations
- **Risk:** Qt signal/slot behavior may be harder to test post extraction. *Mitigation:* abstract signal emitters and use fake Qt objects in tests.
- **Risk:** Refactor may disrupt telemetry or importer flows. *Mitigation:* prioritize coverage and stage rollout feature-by-feature.

---

## 3. Tool Registry Error Masking (`src/tinkerbell/ai/tools/registry.py`)

### Current Pain Points
- `register_default_tools()` wraps all registrations in a single `try/except`, so one failure prevents every subsequent tool from registering while hiding the root cause.

### Objectives
1. Surface individual tool registration failures without stopping other tools from wiring up.
2. Provide actionable error reporting to developers and telemetry.
3. Add tests confirming partial registration succeeds and failures are visible.

### Proposed Architecture & Steps
1. **Per-Tool Guarding**
   - Refactor registration loop so each tool is wrapped in its own `try/except` block.
   - Collect exceptions in a list and raise an aggregated `ToolRegistrationError` (containing tool names and stack traces) after the loop if any failures occurred.
2. **Telemetry & Logging**
   - Emit telemetry/log entries per failed tool and ensure the aggregated error reaches the caller/UI.
   - Optionally expose a callback hook so the UI can surface partial failures.
3. **Tests**
   - Extend `tests/test_ai_tools.py` or add new suites to simulate a failing tool constructor and assert:
     - Remaining tools register.
     - Aggregated error includes failing tool details.
     - Telemetry/log hooks fire.

### Deliverables
- Updated `registry.py` with robust error handling and a new `ToolRegistrationError` class.
- Unit tests covering success, partial failure, and all-fail scenarios.
- Updated documentation (e.g., `docs/operations/subagents.md`) explaining error reporting behavior.

### Risks & Mitigations
- **Risk:** Existing callers might rely on blanket suppression. *Mitigation:* audit call sites (AI controller, UI) and handle new exception types explicitly, possibly gating enforcement behind a feature flag during rollout.

---

## Cross-Cutting Considerations
- **Sequencing:** Begin with AI controller modularization to unlock cleaner tool registration and reduce coupling before tackling MainWindow refactors that depend on those services.
- **Telemetry:** Instrument each refactor phase to measure improvements (graph rebuild counts, UI controller load times, tool registration error rates).
- **Documentation:** Update `docs/ai_v2.md` or release notes to highlight architecture changes and developer workflows.
- **Rollout:** Stage refactors behind feature flags or branch toggles when possible, merging incremental improvements with comprehensive tests.
