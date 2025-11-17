# AI v2 Phase 4 release notes

_Last updated: 17 Nov 2025_

## Highlights

- **Scoped subagents (Phase 4.1)** – Planner can spin up helper jobs tied to the current selection with quota-aware execution.
- **Subagent result cache (Phase 4.2)** – Deterministic keying, cache-bus eviction, and telemetry surfaced through the diagnostics sink.
- **Character & plot scaffolding (Phase 4.3)** – Document-scoped entity memory (`DocumentPlotStateTool` + operator guide) gated behind the experimental plot toggle.
- **Integration, telemetry, hardening (Phase 4.4)** – Sequential multi-job smoke tests, cache-hit telemetry, TraceCompactor coverage for helper summaries, and a repeatable latency benchmark (`benchmarks/subagent_latency.md`).

## Feature flags

Both the subagent sandbox and plot scaffolding remain **off by default**. Enable them using any of the following:

| Toggle | Description |
| --- | --- |
| Settings UI → AI → “Enable experimental subagents” | Persists to the profile config file. |
| Environment: `TINKERBELL_ENABLE_SUBAGENTS=1` | Forces subagents on for the current session. |
| CLI: `tinkerbell.exe --enable-subagents` | Useful for portable builds and CI. |
| Plot scaffolding: `TINKERBELL_ENABLE_PLOT_SCAFFOLDING=1` or `--enable-plot-scaffolding` | Enables the entity/plot store, guardrailed tools, and operator hints. |

If the flags are left off, the runtime behaves identically to Phase 3.

## Phase 4.4 additions

| Area | Details |
| --- | --- |
| Trace compaction | Subagent scouting reports are now registered with `TraceCompactor`, so pointer summaries can reclaim tokens when later tool calls pressure the budget. The pointers instruct operators to rerun the helper or `DocumentPlotStateTool` to rebuild the scouting report. |
| Telemetry | `SubagentManager` emits `subagent.turn_summary` events (requested vs. scheduled jobs, cache hits, latency/tokens) after each turn. Cache hits and budget skips surface in diagnostics UI via the existing telemetry sink. |
| Tests | New coverage under `tests/test_subagent_manager.py` (sequential execution and budget enforcement) plus `tests/test_agent.py::test_ai_controller_registers_subagent_messages_in_trace_compactor`. Plot-state tooling already covered in `tests/test_plot_state.py` and `tests/test_document_plot_state_tool.py`. |
| Benchmarks | `benchmarks/measure_subagent_latency.py` plus the accompanying write-up (`benchmarks/subagent_latency.md`) quantify orchestration overhead before engaging real models. |
| Docs | README quickstart, `docs/ai_v2.md` Phase 4 notes, operator runbook, and this release-note all reference the new toggles and telemetry behaviors. |

## Validation matrix

- ✅ `uv run pytest` (entire test suite, 340 tests, 5.7 s on Windows 11 / Python 3.13.3)
- ✅ Benchmarks executed locally to capture Table 1 above.
- 🔜 Long-form latency + telemetry dashboard review will continue during the staged rollout (flags remain opt-in until we collect two weeks of stability data).

## Upgrade checklist

1. Sync to the `AI_v2` branch and run `uv sync --all-extras`.
2. Review `docs/operations/subagents.md` for enablement guidance.
3. (Optional) Enable `TINKERBELL_ENABLE_SUBAGENTS=1` and `TINKERBELL_ENABLE_PLOT_SCAFFOLDING=1` in a staging profile.
4. Tail telemetry (`uv run python -m tinkerbell.cli trace --follow`) to confirm `subagent.turn_summary` events before enabling in production.
5. Run the new benchmark or `uv run pytest tests/test_subagent_manager.py` if you modify helper flow.
6. Keep the feature flags disabled in production until telemetry dashboards confirm stability.
