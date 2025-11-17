# Subagent & Plot Scaffolding Operations Guide

This guide summarizes how to enable, monitor, and troubleshoot the experimental Phase 4 features: the subagent sandbox and the new plot/entity scaffolding layer.

## 1. Enabling the features

| Toggle | Settings dialog | CLI flag | Environment variable | Notes |
| --- | --- | --- | --- | --- |
| Subagent sandbox | **Settings → Experimental → Phase 4 subagents** | `--enable-subagents` / `--disable-subagents` | `TINKERBELL_ENABLE_SUBAGENTS=1` | Required for chunk-level helper jobs and strongly recommended before enabling scaffolding. |
| Plot scaffolding | **Settings → Experimental → Plot scaffolding** | `--enable-plot-scaffolding` / `--disable-plot-scaffolding` | `TINKERBELL_ENABLE_PLOT_SCAFFOLDING=1` | Gated feature that exposes cached character/entity + arc summaries via `DocumentPlotStateTool`. |

Both toggles default to **off**. Flip them via the settings dialog when the app is idle, then click **Apply** so the controller can restart with the new runtime config. CLI/env overrides win over the persisted settings, which is useful when running short-lived smoke tests or CI pipelines.

## 2. Runtime expectations

1. **Status bar** – The "Subagents" pill in the status bar turns green when the sandbox is enabled and shows live counters (`completed/failed/skipped`). Plot scaffolding has no separate badge yet; rely on controller hints (see below) to confirm ingestion.
2. **Ingestion flow** – Every successful subagent job publishes a ~200-token summary. When `plot_scaffolding_enabled=True`, `AIController` feeds those summaries into `DocumentPlotStateStore`, storing the chunk hash, pointer ID, and timestamp per beat. The store keeps only the latest 24 entities and 24 beats per arc per document, and cache-bus events (`DocumentChanged`/`DocumentClosed`) wipe stale entries automatically.
3. **Controller hints** – After a document ingests at least one chunk, the chat transcript receives a short system message: `"Plot scaffolding refreshed for '<doc>'…"`. Use this as your signal that `document_plot_state` will return `status="ok"` for the active tab.
4. **Tool access** – Agents (and manual tool runners) can call `DocumentPlotStateTool` once the flag is on. Pass `document_id` to inspect a non-active tab, or omit it to target the active document. The tool never mutates state and refuses to run when the feature is disabled.
5. **Persistence** – Plot scaffolding is intentionally ephemeral. Closing a document, clearing caches, or disabling the flag flushes the corresponding entry. There is no on-disk persistence in Phase 4.

## 3. Troubleshooting checklist

| Symptom | Likely cause | Remediation |
| --- | --- | --- |
| Subagents never appear in the status bar | Flag disabled or CLI/env override forcing `enable_subagents=False`. | Verify **Settings → Experimental → Phase 4 subagents**, remove conflicting overrides, and restart the controller. |
| `document_plot_state` returns `plot_state_disabled` | Plot scaffolding toggle is off. | Enable the flag (settings/CLI/env) and re-run the tool. |
| `document_plot_state` returns `plot_state_unavailable` | Controller has not instantiated the store (rare; usually because the AI runtime has not been configured). | Ensure the AI controller is running (valid API key/model) and that subagents completed at least one job. |
| `document_plot_state` returns `no_plot_state` | Feature is on but no subagent summaries have been ingested for the requested document. | Trigger a subagent job (select >400 chars and ask for analysis) or wait for existing jobs to finish, then retry. |
| Plot hints stop appearing mid-session | Document was closed, cache bus cleared the entry, or the flag was toggled off. | Reopen the document and rerun subagents, or re-enable the flag. |

## 4. Operational tips

- Keep the feature behind CLI/env overrides in CI until you are comfortable with the added latency. Subagent jobs add ~2–4 seconds each depending on the LLM backend.
- Because the store is per-process state, running multiple app instances splits the cached data. Use a single window when testing scaffolding-driven workflows.
- Telemetry emits `plot_state.ingested` events with `document_id`, entity/arc counts, and chunk hashes. Subscribe via `TelemetrySink` if you need centralized monitoring.
- The feature is read-only. Do not attempt to edit plot data manually; future phases will ship a human-facing editor once the ingestion quality matures.
