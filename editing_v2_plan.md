# Editing v2 – Diff-Based Agent Edits

## 1. Objectives
- Eliminate "replace selection" brittleness by letting the AI describe edits as unified diffs anchored to a snapshot digest.
- Support both existing atomic actions (`insert`, `replace`, `annotate`) and the new `patch` flow while keeping backward compatibility for incremental rollout.
- Surface clear success/failure telemetry (diff summary, conflict cause, ranges touched) for both the UI and the agent.
- Maintain deterministic editor state even when the AI operates asynchronously or multiple directives queue up.

## 2. Guiding Principles
1. **Snapshot-first contract** – every structural edit must reference a known `document_version`; the bridge rejects stale deltas early.
2. **Textual anchoring** – diffs describe context lines + hunks so edits survive cursor drift.
3. **Deterministic apply** – patch application runs on the UI/main thread, emits a single undo entry, and reports the resulting selection.
4. **Graceful fallback** – when diffs fail, the agent is nudged to fetch a fresh snapshot rather than corrupt the buffer.

## 3. Directive Schema & Validation
- Update `ActionType` enum (`tinkerbell/chat/commands.py`) to add `PATCH`.
- Extend `DIRECTIVE_SCHEMA`:
  - `if action == "patch"`: require `diff` (string, non-empty), forbid bare `content/target_range`.
  - For legacy actions: keep existing fields, but add optional `selection_fingerprint` (hash of original text) for soft verification.
- Update `validate_directive` to enforce the conditional schema and to ensure the declared `document_version` is present on patch directives.
- Provide helper `create_patch_directive(diff: str, version: str, rationale: str | None)` for agent/test code.

## 4. Patch Application Pipeline
1. **Payload normalization** (`DocumentBridge._normalize_directive`):
   - When `action == "patch"`, keep raw `diff`, store `context_version` from `document_version|snapshot_version|version|document_digest`.
   - Skip `_normalize_target_range`; derive it later from applied hunks for telemetry.
2. **Patch executor** (`tinkerbell/editor/patches.py`, new module):
   - Parse unified diff (at minimum handle headers like `--- a/...` / `+++ b/...` and hunks `@@ -l,s +l,s @@`).
   - Apply sequentially using the in-memory buffer; raise `PatchApplyError` on mismatched context.
   - Return both `new_text` and metadata: list of affected `(start, end)` spans + diff summary.
   - Prefer a small dependency-free implementation; if adopting `unidiff`, wire it through `pyproject.toml` + vendor tests.
3. **Bridge application** (`DocumentBridge._apply_edit`):
   - Branch on directive action: for patch, call `apply_patch(document_before.text, queued.directive.diff)`.
   - Update `_last_edit_context` using computed spans (e.g., smallest union of added/changed ranges).
   - Feed the resulting text into `editor.set_text` via a dedicated method `EditorWidget.apply_patch(new_text, selection_hint)` to ensure undo bookkeeping matches a normal replace.
4. **Editor widget helper** (`EditorWidget`):
   - Add `apply_patch_result(result: PatchResult)` that:
     - pushes undo snapshot once,
     - swaps `_text_buffer` -> `result.text`,
     - sets selection to the last modified span or caret end, and
     - emits text/selection change notifications.

## 5. Agent Tool Updates
- **`DocumentSnapshotTool`**: include `version` (already), plus optional `selection_hash` and `line_offsets` to aid diff creation.
- **`DocumentEditTool`**:
  - Accept `action="patch"` with `diff` field; forbid mixing with `content/target_range`.
  - Elaborate help text/exception messages so the agent can correct bad payloads.
  - Echo patch status: e.g., `applied: +120 chars (patch, version=abc)`.
- **Prompting/Docs**: Update `tinkerbell/ai/prompts.py` and README tool docs to describe the new diff workflow ("fetch snapshot → compute unified diff → call DocumentEdit with patch").
- **Optional helper tool**: add `DiffBuilderTool` (LangChain tool) that takes `original` + `updated` text slices and returns a ready-to-send diff; helps smaller models avoid formatting mistakes.

## 6. Conflict Detection & Telemetry
- On `PatchApplyError`, include:
  - reason (`context mismatch`, `stale version`, etc.),
  - the expected vs actual context snippet,
  - the version token comparison.
- Surface failures in both the chat transcript and `_last_diff_summary` (e.g., `failed: stale patch vs digest-xyz`).
- Add a lightweight metrics struct to `DocumentBridge` capturing `total_patches`, `patch_conflicts`, `avg_patch_latency` for future debugging (optional but easy).

## 7. UI/UX Touchpoints
- Chat panel should render patch tool calls distinctly (icon + diff preview snippet) so users can audit changes.
- Provide an option in settings (e.g., `Settings.use_patch_edits`) to toggle patch mode by default; fallback to legacy mode for users on older models.
- Add a toast/notification when a patch is rejected, suggesting "Ask TinkerBell to re-sync".

## 8. Testing Strategy
1. **Unit tests** (`tests/test_ai_tools.py`):
   - Validate schema acceptance/rejection for patch directives.
   - Ensure `DocumentEditTool` routes diff payloads correctly and reports statuses.
2. **Patch parser tests** (`tests/test_patches.py`, new):
   - Happy path diff apply, multiple hunks, newline-at-EOF behavior.
   - Failure cases: missing context, reversed hunks, overlapping edits.
3. **Bridge tests** (`tests/test_bridge.py`):
   - Applying a patch updates `_last_edit_context`, diff summaries, and undo stack once.
   - Conflicting patch raises and leaves document untouched.
4. **Integration tests** (`tests/test_agent.py`):
   - Simulate agent sending patch + verifying `DocumentSnapshot` version handshake.
5. **Prompt regression**: run a fixture ensuring prompt text describes new contract (snapshot -> diff -> apply).

## 9. Migration & Rollout Steps
1. Implement patch parser + tests.
2. Update schema, directive validation, and bridge normalization.
3. Wire editor + bridge patch application and ensure undo/redo coherence.
4. Update AI tools + prompts + docs.
5. Add feature flag in settings + UI surfaces.
6. Run full test suite; add regression tests for legacy insert/replace to ensure no behavioral regressions.
7. Document the new workflow in `README.md` and changelog; include examples of valid diff payloads and troubleshooting tips.

## 10. Open Questions
- Should we store the last few applied patches for quick rollback/history in the chat transcript?
- Do we want to support multi-file diffs eventually (requires file targeting in directives)?
- Is adopting `diff-match-patch` preferable for fuzzy matches, or do we keep strict unified diffs for now?
- How do we guard against extremely large diffs (size limits, streaming apply)?
