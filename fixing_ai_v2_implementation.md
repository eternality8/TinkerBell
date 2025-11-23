# AI Editing Reliability v2 – Implementation Plan

This document breaks down the `fixing_ai_v2.md` strategy into concrete workstreams, milestones, and trackable tasks.

## Workstream 1 – Caret-Free AI Tools (P0)
- [x] Remove caret/selection mutation APIs from `document_service` and `editor_bridge`.
- [x] Deprecate client parameters `insert_at_cursor`, `cursor_offset`, `selection_start`, `selection_end` (schema now rejects legacy caret fields and emits `caret_call_blocked`).
- [x] Implement read-only `get_selection_range` tool returning `{start_line, end_line, content_hash}` (SelectionRangeTool registered + covered by pytest).
- [x] Update agent runtime to require `{range|replace_all}` on every edit request (DocumentEditTool/ApplyPatch enforce range, match_text, or `replace_all=true`).
- [x] Emit telemetry counters `caret_call_blocked`, `span_snapshot_requested` (events wired + asserted in `tests/test_ai_tools.py`).
- [x] Migration guide + changelog entry for partners.

_Status_: Code/tests for caret-free enforcement and selection snapshots merged on branch `AI_v2`; caret-free guidance now lives in `docs/ai_v2.md` and the release notes. Bridge-layer caret mutation APIs now preserve the original user selection, and the partner migration guide/changelog updates are published.

## Workstream 2 – Replace-All Semantics (P0)
- [x] Add `operation: "replace"` / `replace_all: true` to `document_apply_patch` schema.
- [x] Enforce replace-all path when incoming content length ±5% of document length and no range provided.
- [x] Reject insert-only payloads >1 KB without range; return structured `needs_range` error.
- [x] Update agent-side diff builder to re-issue request with explicit range when `needs_range` received (controller now emits structured `needs_range` payloads so the diff builder can auto-retry with explicit bounds).

## Workstream 3 – Snapshot + Plot-State Validation (P0)
- [x] Require both `content_hash` and `version_id` in API contract; reject mismatches before applying patch (enforced in `DocumentApplyPatchTool`, schema/provider wiring updated, pytest coverage via `_run_with_meta`).
- [x] Cache latest `plot_outline` entities/beats; compare against candidate edit for missing tracked actors (plot-state resolver now injected throughout registry/provider stack).
- [x] Emit soft warning + guidance when entity/beat disappears unexpectedly (tool surfaces warning text + telemetry event, tests exercising warning path in progress).
- [x] Document retry protocol for clients upon hash mismatch or outline drift warnings (see `docs/ai_v2.md` Workstream 3 section).

_Status_: Metadata enforcement, plot-state drift checks, controller retry plumbing, and partner documentation are all live on `AI_v2`; no remaining tasks for this workstream.

## Workstream 4 – Duplicate & Corruption Detection (P0)
- [x] Implement post-edit diff that hashes paragraphs / 10-line windows.
- [x] Detect duplicated adjacent hashes; flag as duplication anomaly.
- [x] Add regex heuristics for split tokens (e.g., `[a-z]{2}\n#`).
- [x] Auto-revert buffer to pre-edit snapshot upon anomaly.
- [x] Emit `auto_revert` event with reason + diff summary.
- [x] Surface structured error to caller with remediation steps.

## Workstream 5 – Range-Safe Editing API (P1)
- [x] Replace raw offset parameters with `{start_line, end_line}` (or CRDT span) objects in public API. (`document_apply_patch` + registry schema now expose `target_span`, UI forces spans when safe edits are enabled.)
- [x] Server-side normalization: expand partial spans to word/paragraph boundaries. (Line spans convert through `LineRange` helpers + `NormalizedTextRange`, preserving explicit spans while still widening legacy offsets.)
- [x] Provide compatibility adapter translating old offset-based requests during rollout (behind flag). (Legacy adapter and telemetry flag wiring allow offsets when span policy disabled.)
- [x] Update SDKs/tooling to consume new span objects. (Prompts + `docs/ai_v2.md` now instruct agents/partners to call `selection_range` and send `target_span`, and guardrail errors reference the span-first workflow.)

## Workstream 6 – Regression Tests (P1)
- [x] Unit tests: hash mismatch rejection, replace-all enforcement, duplicate detector auto-revert, selection read-only guarantee.
- [x] Integration tests via pytest simulating full-document rewrite; assert single copy remains.
- [x] Add load test covering concurrent edit attempts + auto-retry flow.

## Workstream 7 – Observability & UX (P1)
- [x] Emit structured events: `edit_rejected`, `auto_revert`, `duplicate_detected`, `hash_mismatch`, `needs_range`, `caret_call_blocked`, `span_snapshot_requested`.
- [x] Editor UI: toast + inline banner describing failure reason and suggested action.

_Status_: Hash mismatch telemetry now fires from both DocumentApplyPatch preflight validators and the DocumentBridge rejection path, and the chat/status UI surfaces guardrail toasts for snapshot/streamed chunk failures.


## Workstream 8 – Rollout Plan (P2)
- [ ] Feature gate `safe_ai_edits_v2` controlling all new behaviors.

