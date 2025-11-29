# Orchestration Cleanup Implementation Plan

This document tracks the detailed implementation of the orchestration rewrite. See `orchestration_cleanup_plan.md` for the architectural rationale.

---

## Phase 1: Build New Pipeline (Clean Room)

> **Rule:** Do not modify existing code. Build the new system alongside it.

### 1.1 Core Types (`types.py`)

- [ ] Create `orchestration/types.py`
- [ ] Define `TurnInput` dataclass (frozen)
  - [ ] `prompt: str`
  - [ ] `snapshot: DocumentSnapshot`
  - [ ] `history: tuple[Message, ...]`
  - [ ] `config: TurnConfig`
- [ ] Define `TurnOutput` dataclass (frozen)
  - [ ] `response: str`
  - [ ] `tool_calls: tuple[ToolCallRecord, ...]`
  - [ ] `metrics: TurnMetrics`
- [ ] Define `TurnConfig` dataclass
  - [ ] `max_iterations: int`
  - [ ] `analysis_enabled: bool`
  - [ ] `budget_policy: BudgetPolicy`
  - [ ] Other config options as needed
- [ ] Define `PreparedTurn` dataclass
- [ ] Define `AnalyzedTurn` dataclass
- [ ] Define `ModelResponse` dataclass
- [ ] Define `TurnMetrics` dataclass
- [ ] Define `Message` type (or import from existing)
- [ ] Define `DocumentSnapshot` (or import from existing)
- [ ] Write unit tests for type serialization/validation

### 1.2 Pipeline Stage: Prepare (`pipeline/prepare.py`)

- [ ] Create `orchestration/pipeline/` directory
- [ ] Create `orchestration/pipeline/__init__.py`
- [ ] Create `orchestration/pipeline/prepare.py`
- [ ] Implement `build_messages(prompt, snapshot, history) -> list[Message]`
- [ ] Implement `estimate_budget(messages, config) -> BudgetEstimate`
- [ ] Implement `prepare_turn(input: TurnInput) -> PreparedTurn`
- [ ] Write unit tests for `prepare_turn`
- [ ] Write unit tests for `build_messages`
- [ ] Write unit tests for `estimate_budget`

### 1.3 Pipeline Stage: Analyze (`pipeline/analyze.py`)

- [ ] Create `orchestration/pipeline/analyze.py`
- [ ] Implement `run_analysis(snapshot) -> AnalysisAdvice | None`
- [ ] Implement `generate_hints(advice, snapshot) -> list[str]`
- [ ] Implement `analyze_turn(prepared, snapshot, config) -> AnalyzedTurn`
- [ ] Write unit tests for `analyze_turn`
- [ ] Write unit tests for analysis logic

### 1.4 Pipeline Stage: Execute Model (`pipeline/execute.py`)

- [ ] Create `orchestration/pipeline/execute.py`
- [ ] Implement `parse_response(raw_response) -> ModelResponse`
- [ ] Implement `execute_model(turn, client) -> ModelResponse` (async)
- [ ] Handle streaming responses
- [ ] Handle tool call parsing
- [ ] Write unit tests with mocked AI client

### 1.5 Pipeline Stage: Tools (`pipeline/tools.py`)

- [ ] Create `orchestration/pipeline/tools.py`
- [ ] Implement `execute_tools(response, executor) -> ToolResults` (async)
- [ ] Implement `append_tool_results(messages, results) -> list[Message]`
- [ ] Write unit tests for tool execution flow

### 1.6 Pipeline Stage: Finish (`pipeline/finish.py`)

- [ ] Create `orchestration/pipeline/finish.py`
- [ ] Implement `collect_metrics(...) -> TurnMetrics`
- [ ] Implement `finish_turn(response, tool_results, metrics) -> TurnOutput`
- [ ] Write unit tests for `finish_turn`

### 1.7 Tool System (`tools/`)

- [ ] Create `orchestration/tools/` directory
- [ ] Create `orchestration/tools/__init__.py`
- [ ] Create `orchestration/tools/types.py`
  - [ ] Define `ToolCall` dataclass
  - [ ] Define `ToolResult` dataclass (with `.success()` and `.error()` constructors)
  - [ ] Define `ToolCallRecord` dataclass
- [ ] Create `orchestration/tools/registry.py`
  - [ ] Implement `ToolRegistry` class
  - [ ] `register(name, tool)` method
  - [ ] `get(name) -> Tool | None` method
  - [ ] `list_tools() -> list[ToolSpec]` method
- [ ] Create `orchestration/tools/executor.py`
  - [ ] Implement `ToolExecutor` class
  - [ ] `execute(call: ToolCall) -> ToolResult` (async)
  - [ ] Error handling and logging
- [ ] Write unit tests for `ToolRegistry`
- [ ] Write unit tests for `ToolExecutor`
- [ ] Migrate `transaction.py` logic to `tools/transaction.py` if needed

### 1.8 Turn Runner (`runner.py`)

- [ ] Create `orchestration/runner.py`
- [ ] Implement `TurnRunner` class
  - [ ] `__init__(client, tool_executor, config)`
  - [ ] `run(input: TurnInput) -> TurnOutput` (async)
- [ ] Implement the tool loop (iterate until no tool calls or max iterations)
- [ ] Wire all pipeline stages together
- [ ] Write integration tests for `TurnRunner`
- [ ] Write unit tests with mocked stages

### 1.9 Public API (`__init__.py`)

- [ ] Update `orchestration/__init__.py`
- [ ] Export `TurnRunner`
- [ ] Export `TurnInput`, `TurnOutput`, `TurnConfig`
- [ ] Export `ToolRegistry`, `ToolExecutor`
- [ ] Export any other public types

---

## Phase 2: Wire Up Services

### 2.1 Services Directory Structure

- [ ] Create `orchestration/services/` directory
- [ ] Create `orchestration/services/__init__.py`

### 2.2 Document Cache Service

- [ ] Create `orchestration/services/document_cache.py`
- [ ] Implement `DocumentCache` class
  - [ ] `get(doc_id) -> DocumentSnapshot | None`
  - [ ] `set(doc_id, snapshot)`
  - [ ] `invalidate(doc_id)`
- [ ] Write unit tests

### 2.3 Analysis Cache Service

- [ ] Create `orchestration/services/analysis_cache.py`
- [ ] Implement `AnalysisCache` class
  - [ ] `get(snapshot_hash) -> AnalysisAdvice | None`
  - [ ] `set(snapshot_hash, advice)`
  - [ ] Cache invalidation logic
- [ ] Write unit tests

### 2.4 Budget Service

- [ ] Create `orchestration/services/budget.py`
- [ ] Implement `BudgetPolicy` class
- [ ] Implement budget evaluation logic
- [ ] Migrate relevant logic from `budget_manager.py`
- [ ] Write unit tests

### 2.5 Telemetry Service

- [ ] Create `orchestration/services/telemetry.py`
- [ ] Implement `TelemetryService` class
  - [ ] `record_turn(metrics: TurnMetrics)`
  - [ ] `record_tool_call(call: ToolCallRecord)`
  - [ ] Other telemetry methods
- [ ] Write unit tests

### 2.6 Services Container

- [ ] Define `Services` dataclass in `services/__init__.py`
- [ ] Wire services into `TurnRunner` constructor
- [ ] Update pipeline stages to accept services where needed

---

## Phase 3: Switch Over

### 3.1 Identify All Callers

- [ ] Search codebase for all `AIController` usages
- [ ] Document each caller and what API it uses
- [ ] List files that need updating:
  - [ ] `src/tinkerbell/ui/chat_panel.py` (likely)
  - [ ] `src/tinkerbell/ui/main_window.py` (likely)
  - [ ] Other files TBD

### 3.2 Update Callers

- [ ] Update each caller to use `TurnRunner` API
- [ ] For each caller:
  - [ ] Replace `AIController` instantiation with `TurnRunner`
  - [ ] Replace `run_chat()` calls with `runner.run(TurnInput(...))`
  - [ ] Update result handling for `TurnOutput` type
- [ ] Test each updated caller manually

### 3.3 Delete Old Code

- [ ] Delete `orchestration/controller.py`
- [ ] Delete `orchestration/chat_orchestrator.py`
- [ ] Delete `orchestration/controller_utils.py`
- [ ] Delete `orchestration/scope_helpers.py`
- [ ] Delete `orchestration/message_builder.py`
- [ ] Delete `orchestration/tool_executor.py` (old one)
- [ ] Clean up any other dead code

### 3.4 Move Non-Orchestration Code

- [ ] Move `checkpoints.py` to `services/checkpoints.py`
- [ ] Move `editor_lock.py` to `services/editor_lock.py`
- [ ] Move `event_log.py` to `utils/event_log.py`
- [ ] Move `telemetry_manager.py` to `services/telemetry.py`
- [ ] Update all imports for moved files

### 3.5 Delete Old Tests

- [ ] Delete `tests/test_ai_controller.py`
- [ ] Delete any other tests for deleted code
- [ ] Identify tests that need updating vs deleting

### 3.6 Write New Tests

- [ ] Write integration tests for full turn flow
- [ ] Write tests for error handling scenarios
- [ ] Write tests for tool loop edge cases
- [ ] Write tests for budget exceeded scenarios
- [ ] Write tests for analysis integration
- [ ] Ensure >90% coverage on new code

### 3.7 Final Validation

- [ ] Run full test suite
- [ ] Fix any failing tests
- [ ] Run linter, fix any issues
- [ ] Run type checker, fix any issues
- [ ] Manual smoke test of chat functionality
- [ ] Review for any remaining dead code

---

## Success Criteria Checklist

- [ ] Total lines in `orchestration/` < 2,000
- [ ] No single file > 300 lines
- [ ] Cyclomatic complexity < 10 for all functions
- [ ] Test coverage > 90%
- [ ] All existing functionality preserved
- [ ] No `AIController` references remaining
- [ ] Clean import structure (no circular imports)

---

## Notes & Decisions

_Record any implementation decisions, blockers, or changes to the plan here._

| Date | Note |
|------|------|
| 2025-11-29 | Plan created |

---

## File Checklist

New files to create:
- [ ] `orchestration/__init__.py` (update)
- [ ] `orchestration/types.py`
- [ ] `orchestration/runner.py`
- [ ] `orchestration/pipeline/__init__.py`
- [ ] `orchestration/pipeline/prepare.py`
- [ ] `orchestration/pipeline/analyze.py`
- [ ] `orchestration/pipeline/execute.py`
- [ ] `orchestration/pipeline/tools.py`
- [ ] `orchestration/pipeline/finish.py`
- [ ] `orchestration/tools/__init__.py`
- [ ] `orchestration/tools/types.py`
- [ ] `orchestration/tools/registry.py`
- [ ] `orchestration/tools/executor.py`
- [ ] `orchestration/services/__init__.py`
- [ ] `orchestration/services/document_cache.py`
- [ ] `orchestration/services/analysis_cache.py`
- [ ] `orchestration/services/budget.py`
- [ ] `orchestration/services/telemetry.py`

Files to delete:
- [ ] `orchestration/controller.py`
- [ ] `orchestration/chat_orchestrator.py`
- [ ] `orchestration/controller_utils.py`
- [ ] `orchestration/scope_helpers.py`
- [ ] `orchestration/message_builder.py`
- [ ] `orchestration/tool_executor.py`
- [ ] `tests/test_ai_controller.py`

Files to move:
- [ ] `checkpoints.py` → `services/checkpoints.py`
- [ ] `editor_lock.py` → `services/editor_lock.py`
- [ ] `event_log.py` → `utils/event_log.py`
- [ ] `telemetry_manager.py` → `services/telemetry.py`

---

*Last updated: 2025-11-29*
