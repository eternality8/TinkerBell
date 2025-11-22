# Safe AI Editing Telemetry Reference

This reference captures the telemetry events emitted by the AI editing guardrails so ingestion pipelines and dashboards stay aligned with the on-disk schema. Every event flows through `tinkerbell.services.telemetry.emit(...)`, so custom sinks (status-bar badges, export scripts, log forwarders) all observe the same payloads.

## 1. Patch lifecycle events (`DocumentBridge`)

| Event | When it fires | Key fields |
| --- | --- | --- |
| `patch.apply` | After every patch attempt. Emitted with `status="success"` once the diff lands, or `status="stale"/"conflict"/"rejected"` when version checks, hash validation, or the inspector block the change. | `document_id`, `version_id`, `content_hash`, optional `tab_id`, `status`, `diff_summary` (success only), `duration_ms`, `range_count`, `streamed`, optional `reason`, `cause` (`hash_mismatch`, `chunk_hash_mismatch`, `inspector_failure`). |
| `edit_rejected` | Mirrors every failure path so dashboards can separate success latency from rejection counts. Fired with the same cause codes whenever `_record_patch_rejection` runs. | `document_id`, `version_id`, `content_hash`, `action` (`patch`), `reason`, optional `cause`, `tab_id`, `range_count`, `streamed`, optional `diagnostics` (includes inspector details when duplicates/boundaries tripped). |

**Tips**
- Use `status` + `cause` to build stacked rejection charts. `status="rejected"` with `cause="inspector_failure"` maps to duplicate/corruption catches, while `status="stale"` indicates the agent skipped a resnapshot.
- `diff_summary` formats as `+X/-Y lines across H hunk(s)`; it is omitted on failures so you can join against `patch.apply` success rows only.
- `range_count` lets you spot streamed (range-based) diffs versus one-shot unified diffs.

## 2. Guardrail helper events (`DocumentApplyPatchTool` / `DocumentEditTool`)

| Event | Purpose | Fields |
| --- | --- | --- |
| `patch.anchor` | Fired at each anchor validation phase (`requirements`, `alignment`, `caret-guard`). Helps quantify how often agents provide hashes/ranges versus falling back to snapshots. | `document_id`, optional `tab_id`, `status` (`success` or `reject`), `phase`, `source` (`document_apply_patch` or `document_edit`), `anchor_source` (`match_text`, `selection_text`, `fingerprint`, `range_only`), `range_provided`, optional `selection_span`, optional `resolved_range`, optional `reason`. |
| `diff.streamed` | Emitted when streamed patches are constructed so we can trend range counts and inserted text volume. | `document_id`, optional `tab_id`, `range_count`, `replaced_chars`, `inserted_chars`. |

**Usage notes**
- Anchor rejects with `phase="requirements"` usually mean the agent skipped `document_snapshot`; flag them so prompt authors can tighten their workflows.
- `anchor_source="fingerprint"` indicates the agent copied `selection_hash` successfully—track it separately from `match_text` heuristics.

## 3. Controller retry telemetry (`AIController`)

| Event | When it fires | Fields |
| --- | --- | --- |
| `document_edit.retry` | Emitted whenever the controller catches a `DocumentVersionMismatchError` from a retryable tool (`document_edit`, `document_apply_patch`, `search_replace`). Fired twice: once for the failure (`status="failed"`, `reason="retry_exhausted"`) and once if the second attempt succeeds. | `tool`, `tab_id`, `document_id`, `cause` (`hash_mismatch`, `chunk_hash_mismatch`, `inspector_failure`), `attempts` (currently `2`), `status`, optional `reason`, `event_source="controller"`. |

**Usage notes**
- Join retry events with `patch.apply` using `document_id` and `tab_id` to see whether automatic retries eventually land.
- A sustained stream of `status="failed"` rows usually means the agent ignored the “refresh snapshot” guardrail; bubble that back to prompts or planners.

## 4. Quick integration checklist

1. **Subscribe once** – Register a `TelemetrySink` (or reuse `scripts/export_context_usage.py`) so all events stream through a single dispatcher.
2. **Normalize cause codes** – Downstream dashboards should treat `hash_mismatch` as the general “stale snapshot” bucket, `chunk_hash_mismatch` as manifest drift, and `inspector_failure` as duplicate/boundary catches.
3. **Correlate with UI badges** – The chat panel/status bar consume the same payloads, so dashboards built from these events will mirror what operators already see live.
4. **Benchmark tie-in** – When running `benchmarks/measure_diff_latency.py --mode pipeline`, watch for matching `patch.apply`/`edit_rejected` events to confirm safe-edit thresholds before rolling changes into production.

## 5. Guardrail & selection signals

| Event | When it fires | Key fields |
| --- | --- | --- |
| `hash_mismatch` | Emitted by `DocumentApplyPatchTool` preflight validators (`stage=document_version`, `version_id`, or `content_hash`) and by `DocumentBridge` when the live buffer diverges or a streamed chunk hash fails validation (`stage=bridge`/`chunk_hash`). | `document_id`, `version_id`, `content_hash`, `stage`, `cause` (`hash_mismatch` or `chunk_hash_mismatch`), `reason`, optional `tab_id`, `range_count`, `streamed`, `details` (provided vs. expected tokens). |
| `needs_range` | Raised when large (>1 KB) inserts omit explicit bounds, triggering a `NeedsRangeError`. | `document_id`, `tab_id`, `selection_span`, `content_length`, `threshold`, `source="document_apply_patch"`, `reason`. |
| `caret_call_blocked` | Recorded whenever the caller omits `target_span`/`match_text` and tries to rely on the caret. | `document_id`, `tab_id`, `range_provided`, `replace_all`, `selection_span`, optional `range_payload`, `reason`. |
| `selection_snapshot_requested` | Fired by `SelectionRangeTool` so rollout dashboards can compare read-only selection usage vs. blocked caret calls. | `document_id`, `tab_id`, `start_line`, `end_line`, `selection_span`, `selection_length`, `content_hash`. |
