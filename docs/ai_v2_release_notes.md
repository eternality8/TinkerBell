# AI v2 Phase 4–5 release notes

_Last updated: 19 Nov 2025_

## Highlights

- **Scoped subagents (Phase 4.1)** – Planner can spin up helper jobs tied to the current selection with quota-aware execution.
- **Subagent result cache (Phase 4.2)** – Deterministic keying, cache-bus eviction, and telemetry surfaced through the diagnostics sink.
- **Character & plot scaffolding (Phase 4.3)** – Document-scoped entity memory (`DocumentPlotStateTool` + operator guide) gated behind the experimental plot toggle.
- **Integration, telemetry, hardening (Phase 4.4)** – Sequential multi-job smoke tests, cache-hit telemetry, TraceCompactor coverage for helper summaries, and a repeatable latency benchmark (`benchmarks/subagent_latency.md`).
- **Storyline continuity orchestration (Phase 5.1)** – `PlotStateMemory`, `PlotOutlineTool`, and `PlotStateUpdateTool` now enforce the “outline → edit → update” loop with guardrail hints, persistence for operator overrides, and `plot_state.*` telemetry.
- **Preflight analysis & tool recommendations (Phase 5.2)** – Rule-based analyzer feeds proactive tool hints into the controller, status bar/chat badges, manual `/analyze`, ContextUsage exports, and telemetry (`analysis.advisor_tool.*`, `analysis.ui_override.*`).
- **Caret-free AI editing (Workstream 1)** – `DocumentBridge` applies patches with `preserve_selection`, so AI edits can no longer move the user’s caret. Partners must rely on `document_snapshot` + `selection_range` fingerprints, honor `selection_fingerprint`/`match_text`, and monitor `caret_call_blocked` telemetry until all clients migrate.
- **Configurable sampling temperature** – The AI settings dialog now exposes the OpenAI-compatible temperature control (0.0–2.0) so creative writing sessions can lean toward higher variety without editing `settings.json` manually.
- **Per-turn debug event logs** – When `debug_logging` (or `TINKERBELL_DEBUG=1`) is enabled, every AI turn now emits a JSONL trace under `~/.tinkerbell/logs/events/` capturing the active snapshot text, assistant messages, tool payloads, and edit outcomes for easier post-mortem debugging. The Settings dialog (Features tab) exposes a “Capture per-chat event logs (JSONL)” toggle so you can enable/disable this without editing `settings.json`.

## Feature flags

Both the subagent sandbox and plot scaffolding remain **off by default**. Enable them using any of the following:

| Toggle | Description |
| --- | --- |
| Settings UI → AI → “Enable experimental subagents” | Persists to the profile config file. |
| Environment: `TINKERBELL_ENABLE_SUBAGENTS=1` | Forces subagents on for the current session. |
| CLI: `tinkerbell.exe --enable-subagents` | Useful for portable builds and CI. |
| Plot scaffolding: `TINKERBELL_ENABLE_PLOT_SCAFFOLDING=1` or `--enable-plot-scaffolding` | Enables the entity/plot store, guardrailed tools, and (as of Phase 5) the PlotStateMemory + enforced edit loop. |

If the flags are left off, the runtime behaves identically to Phase 3.

## Phase 4.4 additions

| Area | Details |
| --- | --- |
| Trace compaction | Subagent scouting reports are now registered with `TraceCompactor`, so pointer summaries can reclaim tokens when later tool calls pressure the budget. The pointers instruct operators to rerun the helper or `DocumentPlotStateTool` to rebuild the scouting report. |
| Telemetry | `SubagentManager` emits `subagent.turn_summary` events (requested vs. scheduled jobs, cache hits, latency/tokens) after each turn. Cache hits and budget skips surface in diagnostics UI via the existing telemetry sink. |
| Tests | New coverage under `tests/test_subagent_manager.py` (sequential execution and budget enforcement) plus `tests/test_agent.py::test_ai_controller_registers_subagent_messages_in_trace_compactor`. Plot-state tooling already covered in `tests/test_plot_state.py` and `tests/test_document_plot_state_tool.py`. |
| Benchmarks | `benchmarks/measure_subagent_latency.py` plus the accompanying write-up (`benchmarks/subagent_latency.md`) quantify orchestration overhead before engaging real models. |
| Docs | README quickstart, `docs/ai_v2.md` Phase 4 notes, operator runbook, and this release-note all reference the new toggles and telemetry behaviors. |

## Phase 5.1 additions – Storyline continuity orchestration

| Area | Details |
| --- | --- |
| PlotStateMemory | `PlotStateMemory` supersedes the transient plot store with dependency tracking, version metadata, and operator overrides persisted to `~/.tinkerbell/plot_overrides.json`. Cache-bus events still wipe stale documents automatically. |
| Tools | `PlotOutlineTool` (alias `DocumentPlotStateTool`) now surfaces overrides/dependencies, while `PlotStateUpdateTool` lets agents log manual adjustments. Both register through `ui.tools.provider`/`ai.tools.registry` and emit telemetry. |
| Controller guardrail | `_PlotLoopTracker` blocks `document_edit`/`document_apply_patch` until an outline tool runs, then reminds the agent to call `plot_state_update` after each edit. Tool traces label `plot_loop_blocked` runs so operators can spot violations immediately. |
| Telemetry | `plot_state.read` and `plot_state.write` events include `document_id`, entity/arc counts, override totals, and persistence status for dashboards. |
| Tests & docs | `tests/test_plot_state.py`, `tests/test_document_plot_state_tool.py`, `tests/test_tool_provider.py`, and new `tests/test_agent.py` guardrail cases cover the memory and enforcement flow. `docs/ai_v2.md` documents the new workflow and telemetry fields. |

## Phase 5.2 additions – Preflight analysis & tool recommendations

| Area | Details |
| --- | --- |
| Analyzer | `tinkerbell.ai.analysis.AnalysisAgent` consumes controller snapshots (chunk manifest hints, outline/plot/concordance freshness, chunk-flow flags) and emits `AnalysisAdvice` (chunk profile, required/optional tools, outline refresh flag, warnings, rule traces). Advice is cached with a TTL and invalidated automatically whenever the document bus publishes `DocumentChangedEvent`/`DocumentClosedEvent`. |
| UI & commands | The status bar now shows a **Preflight** badge, the chat panel mirrors it with hover detail, and `/analyze` lets operators rerun the analyzer with optional document/selection overrides plus a `--reason` note. Successful runs post a formatted notice back into chat and refresh the UI badges. |
| Tooling | `ToolUsageAdvisorTool` exposes `_advisor_tool_entrypoint()` to the agent graph so LangGraph flows can request advice mid-turn. Controller telemetry records every tool invocation via `analysis.advisor_tool.invoked`. |
| Telemetry & exports | Manual overrides emit `analysis.ui_override.requested/completed/failed` (document/selection info, force-refresh flag, cache state, tool lists, warning codes). ContextUsage exports now include analysis columns (chunk profile, required/optional tools, warnings, cache state, rule trace timestamps) so dashboards can track analyzer coverage. |
| Tests & docs | `tests/test_agent.py` asserts cache invalidation and telemetry emission, while `docs/ai_v2.md` documents the end-to-end workflow plus the new telemetry events. Release notes (this section) highlight the operator workflow so support can reference it quickly. |

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
