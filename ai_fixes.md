# AI Diff Tooling Fixes Plan

## 1. Snapshot-Anchored Range Resolution
- Add optional `match_text`/`expected_text` (and `selection_fingerprint`) arguments to `DocumentApplyPatchTool` and propagate them to `DocumentEditTool`.
- When present, locate the text inside the current snapshot and realign `start/end` or raise a version mismatch error if no unambiguous hit exists.
- If both a range and anchor are provided, validate that `base_text[start:end]` matches; reject the edit when it does not.
- Surface clear error messages instructing agents to refresh `document_snapshot` when anchors fail, preventing silent mid-word overwrites.

## 2. Inline Edit Auto-Conversion Safety
- Mirror the anchoring checks inside `DocumentEditTool._auto_convert_to_patch` so forced inline edits undergo the same validation.
- Abort early when the snapshot slice no longer matches the requested content, instead of emitting a diff for the wrong region.

## 3. Guardrails on Implicit Insertions
- Stop defaulting to `(0, 0)` when the agent omits `target_range` and no anchor is present; require an explicit range or `match_text` for replace operations.
- Only allow caret-based inserts when the action is explicitly "insert" or when the AI provides a flagged intent.
- Return actionable errors so agents learn to send enough context instead of duplicating paragraphs.

## 4. Tool Schema & Instruction Updates
- Extend the tool manifest so the agent sees the new arguments plus guidance on copying `selection_text`/`selection_hash` from `document_snapshot`.
- Update system prompts/examples demonstrating how to include anchors and handle validation failures.

## 5. Testing & Telemetry
- Add targeted unit tests covering: valid anchor realignment, anchor mismatch rejection, missing-range errors, and inline conversion safety.
- Build regression tests reproducing the prior failure modes (mid-word insertion, duplicate replacements) to ensure they now error out.
- Instrument bridge telemetry to log anchor-mismatch counts and patch success/conflict ratios so we can verify improvements post-release.
