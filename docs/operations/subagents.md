# Subagent & Plot Scaffolding Operations Guide

This guide summarizes how to monitor and troubleshoot the subagent sandbox and plot/entity scaffolding features.

## 1. Overview

Both subagents and plot scaffolding are **always enabled** in the current release. No feature flags or toggles are required.

## 2. Runtime expectations

1. **Status bar** – The "Subagents" pill in the status bar turns green when the sandbox is active and shows live counters (`completed/failed/skipped`).
2. **Ingestion flow** – Every successful subagent job publishes a ~200-token summary. The `AIController` feeds those summaries into `DocumentPlotStateStore`, storing the chunk hash, pointer ID, and timestamp per beat. The store keeps only the latest 24 entities and 24 beats per arc per document, and cache-bus events (`DocumentChanged`/`DocumentClosed`) wipe stale entries automatically.
3. **Controller hints** – After a document ingests at least one chunk, the chat transcript receives a short system message: `"Plot scaffolding refreshed for '<doc>'…"`. Use this as your signal that `document_plot_state` will return `status="ok"` for the active tab.
4. **Tool access** – Agents (and manual tool runners) can call `DocumentPlotStateTool`. Pass `document_id` to inspect a non-active tab, or omit it to target the active document. The tool never mutates state.
5. **Persistence** – Plot scaffolding is intentionally ephemeral. Closing a document or clearing caches flushes the corresponding entry. There is no on-disk persistence.

## 3. Troubleshooting checklist

| Symptom | Likely cause | Remediation |
| --- | --- | --- |
| Subagents never appear in the status bar | AI controller not running. | Ensure the AI controller is running (valid API key/model). |
| `document_plot_state` returns `plot_state_unavailable` | Controller has not instantiated the store (rare; usually because the AI runtime has not been configured). | Ensure the AI controller is running (valid API key/model) and that subagents completed at least one job. |
| `document_plot_state` returns `no_plot_state` | No subagent summaries have been ingested for the requested document. | Trigger a subagent job (select >400 chars and ask for analysis) or wait for existing jobs to finish, then retry. |
| Plot hints stop appearing mid-session | Document was closed or cache bus cleared the entry. | Reopen the document and rerun subagents. |

## 4. Operational tips

- Subagent jobs add ~2–4 seconds each depending on the LLM backend.
- Because the store is per-process state, running multiple app instances splits the cached data. Use a single window when testing scaffolding-driven workflows.
- Telemetry emits `plot_state.ingested` events with `document_id`, entity/arc counts, and chunk hashes. Subscribe via `TelemetrySink` if you need centralized monitoring.
- The feature is read-only. Do not attempt to edit plot data manually; future phases will ship a human-facing editor once the ingestion quality matures.
