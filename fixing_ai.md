# Plan: Stabilize AI Editing Pipeline

## Objectives
- Stop the editor from duplicating entire sections when LLM responses overwrite existing text.
- Prevent insertions from splitting words or corrupting surrounding characters.
- Detect and surface botched edits immediately so they can be rolled back or retried with a fresh snapshot.

## Observed Failure Modes
1. **Stale snapshot writes**: the agent captured version hash `87f8…` but never revalidated before applying its new draft, so the original block remained and the new draft appended right after it.
2. **Mid-word insert**: offsets were calculated against the earlier snapshot, landing in the middle of `thing`, producing `th` + new block + `ing`.
3. **No post-edit validation**: neither the service nor UI warned that the document now contained two copies of the story with a corrupted word.

## Implementation Plan

### 1. Snapshot Integrity Guardrails
- Extend the document apply API to require the caller's `content_hash` (already present in logs) and reject the patch if the live buffer hash diverges.
- On rejection, force the agent/process to re-fetch the latest text, recompute the diff, and retry once.
- Log every rejection with document id and diff summary for telemetry dashboards.

### 2. Range-Safe Editing
- Switch from absolute integer offsets to explicit `start_line`, `end_line` (or CRDT spans) so replacements align with textual boundaries.
- Normalize selections server-side: expand partial tokens to encompass the full word/paragraph before writing.
- When a tool wants to replace "the whole document", send a `replace_all` command rather than `insert at cursor` so we never leave the source copy behind.

### 3. Duplicate/Corruption Detection Pass
- After each automated edit, diff the new buffer against the prior version:
  - Flag if identical paragraphs (same hash) repeat more than once within a small window.
  - Flag if a token was split (regex for `[a-z]{2}\n#` or unusually short tokens flanking a large insert).
- If anomalies are found, mark the operation as failed, revert the buffer to the pre-edit snapshot, and surface the error to the agent.

### 4. Regression Tests
- Add unit tests around `document_apply_patch` to cover:
  - Concurrent edit rejection (hash mismatch).
  - Word-boundary expansion logic.
  - No-duplication guarantee when replacing a block.
- Add integration tests that drive the editor via test commands to simulate "rewrite paragraph" and assert the final document matches expected text.

### 5. Observability & UX Feedback
- Emit structured events (`edit_rejected`, `auto_revert`, `duplicate_detected`) consumed by the existing telemetry pipeline in `docs/operations`.
- Surface toast/inline warnings in the editor so humans know a botched edit was rolled back and can retry.
- Build a lightweight dashboard panel listing the last N failed edits with reasons to speed up debugging.

### 6. Rollout & Validation
- Ship behind a feature flag `safe_ai_edits` default-off; enable in internal testing first.
- Collect metrics for rejection rate, auto-revert count, and mean retry success to confirm improvements.
- Once stable, flip the flag on for all users and archive metrics snapshots for future regressions.
