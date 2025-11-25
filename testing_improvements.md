# Automated Testing Improvements for AI Text Editor

## Objectives
- Validate that the AI correctly interprets natural-language prompts and issues the right tool calls.
- Prevent regressions in tool selection, argument construction, and sequencing.
- Reduce manual QA effort by giving engineers fast feedback (local + CI) and nightly deep coverage.

## Scope & Assumptions
- Focus on the AI orchestration layer (prompt parsing, plan generation, controller/tool bridges).
- Editor UI, rendering, and low-level document diffing already covered elsewhere; only stub those pieces.
- Tests will reuse existing pytest infrastructure and available mock services where possible.
- Real AI API end-to-end tests incur cost and require credentials, so they must stay opt-in and never run with the default `pytest` target; use markers/flags to gate them.

## Test Layers
1. **Conversation Harness (End-to-End Lite)**
   - Spin up the AI backend with minimal document context.
   - Replay canonical prompts and assert on emitted tool calls + arguments.
   - Store expected transcripts as golden files (`tests/fixtures/conversation_traces/*.json`).
2. **Agent/Controller Unit Suites**
   - Mock downstream services to isolate planner/controller logic.
   - Parametrize prompts to confirm routing decisions (e.g., "outline chapter" -> `OutlineTool`).
   - Assert both chosen tool and structured payload.
3. **Log Replay Regression Tests**
   - Capture real manual sessions once; serialize prompt/tool-call sequences.
   - Re-run them automatically to detect behavioral drift.
4. **Prompt Fuzz/Property Tests**
   - Generate noisy/ambiguous instructions (typos, mixed directives).
   - Ensure safe fallbacks: either reasonable tool choice or explicit rejection.
5. **CI Smoke + Nightly Deep Runs**
   - CI: 3-5 representative prompts to catch blatant failures.
   - Nightly: full golden suite + fuzz batches + latency tracking.

## Implementation Workstreams
| Order | Workstream | Key Tasks | Output |
| --- | --- | --- | --- |
| 1 | Fixture Format & Harness | Define JSON schema for prompt/tool transcripts; build helper to run a conversation and collect events. | `tests/helpers/conversation_harness.py`, sample fixtures. |
| 2 | Controller/Agent Tests | Add pytest modules targeting planner, controller, and router. Use mocks to assert tool selection & args. | `tests/test_ai_controller.py` additions, new parametrized tests. |
| 3 | Golden Trace Suite | Record 5-10 high-value scenarios (outline, rewrite, diff apply, multi-step). Build runner that compares actual vs. stored traces with helpful diffs. | `tests/conversation_traces/test_golden_traces.py`. |
| 4 | Log Replay Pipeline | Add script to ingest manual session logs, normalize UUIDs/timestamps, and convert to fixtures. Integrate into test runner. | `scripts/convert_logs_to_traces.py`. |
| 5 | Prompt Fuzzing | Implement generator utilities (typo injection, multi-step permutations). Add property tests ensuring safe tool fallback or rejection. | `tests/test_prompt_fuzzing.py`. |
| 6 | CI Integration | Create two pytest markers: `smoke_conversation` (fast) and `conversation_full`. Wire `smoke` to CI, full suite to nightly workflow. Document commands in `testing.md`. | Updated workflows + docs. |
| 7 | Telemetry & Alerting (Stretch) | Track pass/fail + latency per scenario, emit summary artifact for dashboards. | CI metrics + optional Slack alert. |

## Tooling & Infrastructure
- Extend existing pytest fixtures (`tests/conftest.py`) to provide mock editor/session objects.
- Use `freezegun` or similar to stabilize timestamps within transcripts.
- Adopt `deepdiff` for expressive diffs when traces mismatch.
- Store large fixtures under `tests/fixtures/` and gate >1MB files.

## Validation & Rollout
1. Prototype harness + one golden scenario locally; ensure deterministic replays.
2. Expand to 5 scenarios and land controller tests; enable `pytest -m smoke_conversation` in CI.
3. Add log replay + fuzz once initial suite stable; monitor flake rate.
4. Document workflow in `testing.md` (how to record, update, and approve new traces).

## Risks & Mitigations
- **Non-determinism**: Normalize timestamps/IDs, seed RNG, stub network responses.
- **Large fixture churn**: Provide tooling to auto-update traces while reviewing diffs.
- **Slow tests**: Keep harness lightweight (mock heavy services) and separate slow nightly runs.

## Success Metrics
- 90% of AI regressions caught by automated tests before manual QA.
- CI smoke suite runtime < 5 minutes.
- Nightly suite produces actionable report with per-scenario latency + pass/fail counts.
