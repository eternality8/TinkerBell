# Plan: AI Editing Reliability v2

## Background
Incidents on 2025-11-21 show the assistant still pastes full-story rewrites after the existing text, creating duplicated documents even after the "+dramatic detail" request. Logs (`chat-20251121-090923-e7b1d33b8e49.jsonl`) confirm `document_apply_patch` inserted ~5.1k chars without deleting anything, so the buffer became `Original + Rewrite`. The safeguards described in `fixing_ai.md` were not implemented (no replace-all command, no duplicate detector, no hash gate). This plan prioritizes the minimal set of changes required to stop duplication, then layers observability and regression coverage so the issue cannot recur silently.

## Goals
1. Guarantee that any rewrite either fully replaces the targeted span or is rejected before reaching the buffer.
2. Detect and auto-revert duplicate or corrupted text within one edit cycle.
3. Provide actionable telemetry and UX feedback whenever an edit is blocked or reverted.
4. Land automated tests that fail if duplication, mid-word inserts, or stale snapshots regress.

## Guardrails & Principles
- **Idempotent Edits**: applying the same tool call twice must yield the same document (no cumulative duplication).
- **Latest Snapshot Only**: every mutation must reference the current `content_hash`; anything older must be rejected and retried.
- **Explain Failures**: developers and end-users should see why an edit was blocked (hash mismatch, duplicate detection, etc.).
- **Caret-Free Automation**: AI tools never move or depend on the user caret; all edits must be described via explicit ranges or replace-all semantics.

## Workstreams

### 1. Caret-Free AI Tools (P0)
- Remove every code path that reads or writes the user's caret/selection on behalf of the AI agent.
- Deprecate API parameters such as `insert_at_cursor`, `cursor_offset`, or implicit "current selection" writes; return `unsupported_operation` if older clients attempt to use them.
- Introduce a single read-only `get_selection_range` tool so the agent can inspect what the human highlighted without mutating it; the returned `{start_line, end_line}` must be pinned to the snapshot hash included in the edit request.
- Update the agent runtime to always compute edits as diff patches (range-based or replace-all) derived from the fetched snapshot, never from editor UI state.
- Add telemetry counters `caret_call_blocked` and `span_snapshot_requested` so we can track lingering clients and real span usage until they migrate.

### 2. Replace-All Semantics (P0)
- Add an explicit `operation: "replace"` flag (or `replace_all: true`) to `document_apply_patch`.
- When the incoming content length is ±5% of the document length and no `range` is supplied, force a replace-all execution path that overwrites the entire buffer rather than inserting at the cursor.
- Reject insert-only calls that exceed 1 KB without an explicit target range; respond with `needs_range=true` so the agent re-fetches and reissues a bounded diff.

### 3. Snapshot + Plot-State Validation (P0)
- Require both `content_hash` and `version_id` on every edit. Reject when the live hash differs, instructing the agent to refresh.
- Compare requested entity/beat set (from `plot_outline`) to the candidate rewrite; if tracked entities disappear unexpectedly (e.g., Geoffrey missing), raise a soft warning so the agent can reconcile before writing.

### 4. Duplicate & Corruption Detection (P0)
- After applying a patch (but before committing), diff the buffer against the previous version:
  - Hash paragraphs (or 10-line windows) and flag duplicates that appear twice consecutively.
  - Run regex heuristics for split tokens (e.g., `[a-z]{2}\n#`, single-character tokens flanking large inserts).
- If either rule triggers, automatically revert to the pre-edit snapshot, emit an `auto_revert` event, and return a structured error to the caller.

### 5. Range-Safe Editing API (P1)
- Move from raw offsets to `{start_line, end_line}` (or CRDT spans) in the public API.
- Server-side, expand partial selections up to word/paragraph boundaries before executing the diff, guaranteeing we never slice a token in half.

### 6. Regression Tests (P1)
- Unit tests for:
  - Hash mismatch rejection.
  - Replace-all enforcement when no range is supplied.
  - Duplicate detector auto-reverting a contrived double-insert.
- Integration tests (pytest) that drive the editing tool end-to-end: simulate "rewrite whole story" and assert the final buffer contains only one copy.

### 7. Observability & UX (P1)
- Emit structured events: `edit_rejected`, `auto_revert`, `duplicate_detected`, `hash_mismatch`, `needs_range`.
- Editor UI: toast + inline banner summarizing the failure reason and suggested next action.
- Add a lightweight dashboard (e.g., in `docs/operations`) listing last N blocked edits for debugging.

### 8. Rollout Plan (P2)
- Gate changes behind `safe_ai_edits_v2` feature flag.
- Enable in staging, capture metrics (rejection rate, retry success, auto-revert count) for one week.
- Roll out progressively (10% → 50% → 100%), ensuring no spike in failed edits before full release.

## Milestones
| Milestone | Scope | Owner | Target |
|-----------|-------|-------|--------|
| M1 | Caret-free tooling + replace-all semantics + hash enforcement | Platform | Week 1 |
| M2 | Duplicate detector + auto-revert + telemetry events | Platform | Week 2 |
| M3 | Range-safe API + regression tests | SDK Team | Week 3 |
| M4 | UX feedback + rollout + dashboard | Product | Week 4 |

Deliverables are complete when duplication attempts are blocked, tests codify the behavior, and telemetry proves the guardrails fire in practice.
