# AI Editing Reliability v2 – Implementation Plan

This document breaks down the `fixing_ai_v2.md` strategy into concrete workstreams, milestones, and trackable tasks.

## Workstream 1 – Caret-Free AI Tools (P0)
- [ ] Remove caret/selection mutation APIs from `document_service` and `editor_bridge`.
- [ ] Deprecate client parameters `insert_at_cursor`, `cursor_offset`, `selection_start`, `selection_end`.
- [ ] Implement read-only `get_selection_range` tool returning `{start_line, end_line, content_hash}`.
- [ ] Update agent runtime to require `{range|replace_all}` on every edit request.
- [ ] Emit telemetry counters `caret_call_blocked`, `selection_snapshot_requested`.
- [ ] Migration guide + changelog entry for partners.

## Workstream 2 – Replace-All Semantics (P0)
- [ ] Add `operation: "replace"` / `replace_all: true` to `document_apply_patch` schema.
- [ ] Enforce replace-all path when incoming content length ±5% of document length and no range provided.
- [ ] Reject insert-only payloads >1 KB without range; return structured `needs_range` error.
- [ ] Update agent-side diff builder to re-issue request with explicit range when `needs_range` received.

## Workstream 3 – Snapshot + Plot-State Validation (P0)
- [ ] Require both `content_hash` and `version_id` in API contract; reject mismatches before applying patch.
- [ ] Cache latest `plot_outline` entities/beats; compare against candidate edit for missing tracked actors.
- [ ] Emit soft warning + guidance when entity/beat disappears unexpectedly.
- [ ] Document retry protocol for clients upon hash mismatch or outline drift warnings.

## Workstream 4 – Duplicate & Corruption Detection (P0)
- [ ] Implement post-edit diff that hashes paragraphs / 10-line windows.
- [ ] Detect duplicated adjacent hashes; flag as duplication anomaly.
- [ ] Add regex heuristics for split tokens (e.g., `[a-z]{2}\n#`).
- [ ] Auto-revert buffer to pre-edit snapshot upon anomaly.
- [ ] Emit `auto_revert` event with reason + diff summary.
- [ ] Surface structured error to caller with remediation steps.

## Workstream 5 – Range-Safe Editing API (P1)
- [ ] Replace raw offset parameters with `{start_line, end_line}` (or CRDT span) objects in public API.
- [ ] Server-side normalization: expand partial spans to word/paragraph boundaries.
- [ ] Provide compatibility adapter translating old offset-based requests during rollout (behind flag).
- [ ] Update SDKs/tooling to consume new span objects.

## Workstream 6 – Regression Tests (P1)
- [ ] Unit tests: hash mismatch rejection, replace-all enforcement, duplicate detector auto-revert, selection read-only guarantee.
- [ ] Integration tests via pytest simulating full-document rewrite; assert single copy remains.
- [ ] Add load test covering concurrent edit attempts + auto-retry flow.
- [ ] Wire tests into CI and block merge on failures.

## Workstream 7 – Observability & UX (P1)
- [ ] Emit structured events: `edit_rejected`, `auto_revert`, `duplicate_detected`, `hash_mismatch`, `needs_range`, `caret_call_blocked`, `selection_snapshot_requested`.
- [ ] Build telemetry dashboard (e.g., Grafana/Looker) showing rolling counts + anomalies.
- [ ] Editor UI: toast + inline banner describing failure reason and suggested action.
- [ ] Documentation updates for support playbooks.

## Workstream 8 – Rollout Plan (P2)
- [ ] Feature gate `safe_ai_edits_v2` controlling all new behaviors.
- [ ] Stage rollout: internal dogfood → 10% → 50% → 100%.
- [ ] Collect metrics each stage: rejection rate, retry success, auto-revert count, duplicate incidents.
- [ ] Create rollback checklist (toggle flag, revert configs, notify stakeholders).
- [ ] Final postmortem + lessons learned report once rollout completes.

