# Orchestration Rewrite Architecture

## Why Rewrite, Not Refactor

The previous cleanup attempt failed because **extraction doesn't fix the fundamental problem**: the `AIController` abstraction is wrong. It's not that the code is poorly organized—it's that the *concept* of a single "controller" that orchestrates everything is inherently flawed.

No matter how much we extract into subcomponents:
- The controller remains the "brain" that coordinates everything
- Every new feature requires touching the controller
- The controller accumulates implicit state and hidden dependencies
- Testing requires mocking the entire world

**The real problem**: We modeled this as a single stateful object when it should be a **pipeline of stateless transformations**.

---

## Current Architecture (What's Wrong)

```
┌─────────────────────────────────────────────────────────────────┐
│                        AIController                              │
│  (4,000 lines, owns everything, knows everything)               │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │ Tools    │ │ Budget   │ │ Subagent │ │ Analysis │  ...      │
│  │ Registry │ │ Manager  │ │ Runtime  │ │ Cache    │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│                                                                  │
│  run_chat() → 500 lines of interleaved logic                    │
│  _plan_subagent_jobs() → duplicates SubagentCoordinator         │
│  _build_messages() → duplicates MessageBuilder                  │
│  ...                                                             │
└─────────────────────────────────────────────────────────────────┘
```

The controller is a **god object** that:
1. Owns all state
2. Orchestrates all operations  
3. Has methods for every concern
4. Cannot be decomposed without breaking everything

---

## New Architecture: Pipeline Model

Instead of one object that does everything, we model chat as a **pipeline** where each stage:
- Takes input, produces output
- Has no hidden state
- Can be tested in isolation
- Can be replaced independently

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ Prepare │ -> │ Analyze │ -> │ Execute │ -> │ Tools   │ -> │ Finish  │
│ Context │    │ Intent  │    │ Turn    │    │ Loop    │    │ Turn    │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
     │              │              │              │              │
     v              v              v              v              v
  TurnInput    AnalyzedTurn   ModelResponse  ToolResults   TurnOutput
```

### Core Insight

A chat turn is a **pure function**:

```python
TurnOutput = execute_turn(TurnInput)
```

Where:
- `TurnInput` = prompt + snapshot + history + config
- `TurnOutput` = response + tool_calls + metrics

The "controller" becomes just **wiring**—connecting pipeline stages and holding configuration.

---

## New Design

### 1. Data Flows, Not Objects

Everything is a dataclass that flows through the pipeline:

```python
@dataclass(frozen=True)
class TurnInput:
    """Immutable input to a chat turn."""
    prompt: str
    snapshot: DocumentSnapshot
    history: tuple[Message, ...]
    config: TurnConfig

@dataclass(frozen=True)  
class TurnOutput:
    """Immutable output from a chat turn."""
    response: str
    tool_calls: tuple[ToolCallRecord, ...]
    metrics: TurnMetrics
```

### 2. Pipeline Stages

Each stage is a **function** (or small stateless class):

```python
# Stage 1: Prepare
def prepare_turn(input: TurnInput) -> PreparedTurn:
    """Build messages, estimate tokens, check budget."""
    messages = build_messages(input.prompt, input.snapshot, input.history)
    budget = estimate_budget(messages, input.config)
    return PreparedTurn(messages=messages, budget=budget)

# Stage 2: Analyze (optional)
def analyze_turn(prepared: PreparedTurn, snapshot: DocumentSnapshot) -> AnalyzedTurn:
    """Run preflight analysis if enabled."""
    advice = run_analysis(snapshot) if config.analysis_enabled else None
    hints = generate_hints(advice, snapshot)
    return AnalyzedTurn(prepared=prepared, hints=hints, advice=advice)

# Stage 3: Execute Model
async def execute_model(turn: AnalyzedTurn, client: AIClient) -> ModelResponse:
    """Stream model response, parse tool calls."""
    messages = turn.messages_with_hints()
    response = await client.stream_chat(messages)
    return parse_response(response)

# Stage 4: Execute Tools  
async def execute_tools(
    response: ModelResponse, 
    executor: ToolExecutor,
) -> ToolResults:
    """Execute tool calls, collect results."""
    results = []
    for call in response.tool_calls:
        result = await executor.execute(call)
        results.append(result)
    return ToolResults(results)

# Stage 5: Finish
def finish_turn(
    response: ModelResponse,
    tool_results: ToolResults | None,
    metrics: TurnMetrics,
) -> TurnOutput:
    """Assemble final output."""
    return TurnOutput(
        response=response.text,
        tool_calls=tuple(tool_results.records) if tool_results else (),
        metrics=metrics,
    )
```

### 3. The Turn Runner

The only "orchestrator" is a simple loop:

```python
class TurnRunner:
    """Runs the turn pipeline. No business logic, just wiring."""
    
    def __init__(
        self,
        client: AIClient,
        tool_executor: ToolExecutor,
        config: TurnConfig,
    ):
        self.client = client
        self.tool_executor = tool_executor
        self.config = config
    
    async def run(self, input: TurnInput) -> TurnOutput:
        # Prepare
        prepared = prepare_turn(input)
        analyzed = analyze_turn(prepared, input.snapshot)
        
        # Tool loop
        messages = analyzed.messages_with_hints()
        all_tool_calls = []
        
        for iteration in range(self.config.max_iterations):
            response = await execute_model(messages, self.client)
            
            if not response.tool_calls:
                break
            
            results = await execute_tools(response, self.tool_executor)
            messages = append_tool_results(messages, results)
            all_tool_calls.extend(results.records)
        
        # Finish
        return finish_turn(response, all_tool_calls)
```

**That's it.** ~50 lines instead of 4,000.

### 4. Tool Executor

Tools are executed by a simple, focused class:

```python
class ToolExecutor:
    """Executes tools. No orchestration, no state management."""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
    
    async def execute(self, call: ToolCall) -> ToolResult:
        tool = self.registry.get(call.name)
        if tool is None:
            return ToolResult.error(f"Unknown tool: {call.name}")
        
        try:
            result = await tool.run(call.arguments)
            return ToolResult.success(result)
        except ToolError as e:
            return ToolResult.error(str(e))
```

### 5. Services (Shared State)

Stateful things become **services** that are injected:

```python
@dataclass
class Services:
    """Container for shared services. Explicit dependencies."""
    document_cache: DocumentCache
    analysis_cache: AnalysisCache  
    telemetry: TelemetryService
    budget_policy: BudgetPolicy
```

Services are:
- Created once at startup
- Passed explicitly to functions that need them
- Never accessed globally

---

## File Structure

```
orchestration/
├── __init__.py              # Public API
│
├── types.py                 # All dataclasses (TurnInput, TurnOutput, etc.)
│
├── pipeline/                # The turn pipeline
│   ├── __init__.py
│   ├── prepare.py          # prepare_turn()
│   ├── analyze.py          # analyze_turn()
│   ├── execute.py          # execute_model()
│   ├── tools.py            # execute_tools()
│   └── finish.py           # finish_turn()
│
├── runner.py               # TurnRunner (~100 lines)
│
├── tools/                   # Tool system
│   ├── __init__.py
│   ├── registry.py         # ToolRegistry
│   ├── executor.py         # ToolExecutor
│   └── types.py            # ToolCall, ToolResult
│
└── services/               # Stateful services
    ├── __init__.py
    ├── document_cache.py
    ├── analysis_cache.py
    ├── budget.py
    └── telemetry.py
```

**Total: ~1,500 lines** (vs current ~8,000+)

---

## What Happens to Existing Code

### Delete Entirely
| File | Reason |
|------|--------|
| `controller.py` | Replaced by `runner.py` + pipeline |
| `chat_orchestrator.py` | Absorbed into pipeline |
| `controller_utils.py` | Inline into pipeline stages |
| `scope_helpers.py` | Inline where needed |
| `message_builder.py` | Becomes `pipeline/prepare.py` |

### Keep & Clean
| File | New Location |
|------|--------------|
| `model_types.py` | → `types.py` |
| `turn_context.py` | → `types.py` |
| `runtime_config.py` | → `types.py` |
| `tool_dispatcher.py` | → `tools/executor.py` |
| `transaction.py` | → `tools/transaction.py` |
| `budget_manager.py` | → `services/budget.py` |
| `analysis_coordinator.py` | → `pipeline/analyze.py` |
| `subagent_coordinator.py` | → Separate module (not in orchestration) |

### Move Out of Orchestration
| File | New Home | Reason |
|------|----------|--------|
| `checkpoints.py` | `services/checkpoints.py` | Not orchestration-specific |
| `editor_lock.py` | `services/editor_lock.py` | Not orchestration-specific |
| `event_log.py` | `utils/event_log.py` | General utility |
| `telemetry_manager.py` | `services/telemetry.py` | Not orchestration-specific |

---

## Implementation Plan

### Phase 1: Build New Pipeline (Clean Room)
**Do not touch existing code.** Build the new system alongside:

1. Create `types.py` with all dataclasses
2. Implement pipeline stages as pure functions
3. Build `TurnRunner`
4. Build `ToolExecutor` + `ToolRegistry`
5. Write comprehensive tests for new code

### Phase 2: Wire Up Services
1. Create service classes for stateful concerns
2. Implement `DocumentCache`, `AnalysisCache`, etc.
3. Wire services into pipeline where needed

### Phase 3: Switch Over
1. Update all callers to use new API
2. Delete old `controller.py` and related files
3. Delete old controller tests (`test_ai_controller.py`, etc.)
4. Write new tests for pipeline stages and `TurnRunner`
5. Run full test suite, fix any issues

---

## Why This Will Work

1. **No migration** - We build new, then switch
2. **No god object** - Pipeline stages are independent
3. **Testable** - Each stage is a pure function
4. **Extensible** - Add stages without touching others
5. **Debuggable** - Data flows visibly through pipeline
6. **Maintainable** - Each file is <300 lines with single purpose

---

## API Examples

```python
from tinkerbell.ai.orchestration import TurnRunner, TurnInput, TurnConfig

runner = TurnRunner(
    client=ai_client,
    tool_executor=tool_executor,
    config=TurnConfig(max_iterations=8),
)

output = await runner.run(TurnInput(
    prompt="Fix the typo in chapter 3",
    snapshot=document_snapshot,
    history=conversation_history,
))

print(output.response)
print(output.tool_calls)
```

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Total lines in orchestration/ | ~8,000 | <2,000 |
| Largest file | 4,000 | <300 |
| Cyclomatic complexity (max) | 50+ | <10 |
| Test coverage | ~70% | >90% |
| Time to understand codebase | Hours | Minutes |

---

*Generated: 2025-11-29*
