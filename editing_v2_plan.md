# Editing v2 – Diff-Based Agent Edits

## 0. Current Status (November 2025)

- The schema/validation path already understands `action="patch"`. `ActionType` includes the new enum, `DIRECTIVE_SCHEMA` enforces diffs + version tokens, and `DocumentBridge._normalize_directive` honors optional `selection_fingerprint` hashes for legacy insert/replace flows.
- The full patch pipeline has shipped: `tinkerbell/editor/patches.py` parses multi-hunk unified diffs and returns `PatchResult.spans`, `DocumentBridge` routes patch directives through `apply_unified_diff` while tracking `_last_edit_context` and `PatchMetrics`, and `EditorWidget.apply_patch_result` collapses each patch into a single undo snapshot.
- Agents/tools now default to diff-first workflows. `DocumentSnapshotTool` returns selection text/hash/line offsets, `DocumentEditTool` enforces document versions (and can auto-convert insert/replace payloads when `patch_only=True`), `DocumentApplyPatchTool` bundles snapshot→diff→edit, and the prompts in `tinkerbell/ai/prompts.py` explicitly instruct models to fetch a snapshot, build a diff, then call `document_edit`.
- UI + settings work landed: the Preferences dialog now reflects the always-on diff workflow, chat tool traces badge patch tools and embed diff previews, and `_handle_edit_failure` posts a toast telling users to re-sync whenever a patch conflicts.
- Telemetry/tests/docs exist today. `DocumentBridge.patch_metrics` tracks totals/conflicts/latency, tool traces store `diff_preview` metadata, README already documents diff-based editing, and pytest suites (`tests/test_patches.py`, `tests/test_bridge.py`, `tests/test_ai_tools.py`, `tests/test_agent.py`, `tests/test_main_window.py`) exercise the parser, bridge, tools, agent reminders, and settings wiring.

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

**Shipped**

- `ActionType` already includes `PATCH`, and `DIRECTIVE_SCHEMA` enforces `diff` + `document_version` for patch directives while prohibiting raw `content/target_range`. Insert/replace/annotate directives gained the optional `selection_fingerprint` used by the bridge to guard stale selections.
- `validate_directive` trims payloads, ensures non-empty diffs, and reuses `_extract_version_token` so every patch cites a digest. `_normalize_directive` in `DocumentBridge` rejects mismatched fingerprints before queuing edits, keeping the editor safe even when agents race the user.
- `create_patch_directive` centralizes helper logic for tests/agents so they can construct compliant payloads without duplicating schema assumptions.

**Next refinements**

- Consider making either `document_version` or `selection_fingerprint` mandatory for *all* directives so schema validation, not runtime errors, catch stale edits.
- Extend `validate_directive` to return structured error codes/messages; `DocumentEditTool` could then translate those codes into friendly chat-level guidance instead of a generic "Directive payload must..." string.

## 4. Patch Application Pipeline

**Shipped**

1. **Payload normalization** – `_normalize_directive` maps alias fields into `target_range`, extracts `context_version` from any version key, and when `action="patch"` it preserves the raw diff while skipping `_normalize_target_range`. Selection fingerprints are verified immediately for legacy edits.
2. **Patch executor** – `tinkerbell/editor/patches.py` parses full unified diffs (headers, multiple hunks, blank context, `\ No newline` markers) and returns `PatchResult(text, spans, summary)` using a dependency-free implementation.
3. **Bridge application** – `_apply_patch_directive` runs `apply_unified_diff`, captures `_last_edit_context` (spans, diff body), tracks `PatchMetrics`, and forwards the result to `editor.apply_patch_result`, ensuring `_last_snapshot_token` stays in sync.
4. **Editor widget helper** – `EditorWidget.apply_patch_result` pushes a single undo snapshot, updates `_text_buffer` + preview listeners, and restores the selection to the final modified span (falling back to hints when no span is reported).

**Next refinements**

- Bubble `PatchApplyError.details()` (expected text, actual text, hunk header) through `_record_failure` so the UI can show conflict snippets without digging through logs.
- Evaluate guardrails for extremely large diffs (pre-flight line counts or chunked application) so `_locate_hunk` on megabyte-scale documents cannot starve the UI thread.

## 5. Agent Tool Updates

**Shipped**

- `DocumentSnapshotTool` now includes document length, line offsets, `selection_text`, `selection_hash`, and the latest diff summary so diff builders have the exact anchors they need.
- `DocumentEditTool` handles JSON strings/mappings, enforces patch schema rules, exposes `patch_only` mode, and can auto-convert insert/replace payloads into diffs by calling `generate_snapshot` + `DiffBuilderTool`.
- `DocumentApplyPatchTool` (new since the original plan) bundles snapshot→diff→`document_edit` into a single tool, respecting optional `target_range`, `document_version`, `rationale`, and `context_lines` knobs.
- Prompts/docs already tell models to favor `DocumentApplyPatch`/`DocumentEdit` with diffs. `AIController` reinforces this by injecting reminders whenever diff_builder runs without a follow-up patch call.
- README + MainWindow tool registration expose the refreshed toolbox: snapshot, edit, apply_patch, diff_builder, search_replace, and validation.

**Next refinements**

- Return structured responses (spans touched, new version token, diff summary) from `DocumentEditTool`/`DocumentApplyPatchTool` so downstream agent logic can react programmatically instead of parsing human-readable strings.
- Extend `DocumentApplyPatchTool` to optionally patch multiple disjoint ranges or include surrounding context in its response, reducing the need for manual `DiffBuilder + DocumentEdit` fallbacks.

## 6. Conflict Detection & Telemetry

**Shipped**

- `PatchApplyError` carries reason + expected/actual context details, and `_apply_patch_directive` increments `PatchMetrics.conflicts` while setting `_last_diff_summary = "failed:<reason>"` so tool traces and chat notices stay informative.
- `_last_edit_context` stores the diff body + spans, and `MainWindow._handle_edit_applied` attaches that as `diff_preview` metadata so the tool activity panel can show what changed.
- `PatchMetrics` (total/conflicts/avg_latency_ms) are already tracked, paving the way for in-app telemetry widgets.

**Next refinements**

- Thread `PatchApplyError.details()` through `_record_failure` and into the UI toast/tool trace so users can see the exact conflicting snippet without digging into logs.
- Expose `PatchMetrics` via a listener or status bar so long-running sessions can monitor patch health without attaching a debugger.

## 7. UI/UX Touchpoints

**Shipped**

- Tool traces badge patch tools with `[patch]`, include the diff summary, and store a `diff_preview` block that shows up when copying trace details.
- Settings default to the diff-based workflow now—`DocumentEditTool.patch_only` is always enforced, and prompts permanently instruct models to ship patches.
- `_handle_edit_failure` posts a toast plus status text (“Patch rejected – ask TinkerBell to re-sync”), matching the planned fallback affordance.

**Next refinements**

- Provide an inline diff viewer (or clickable preview) when a patch lands so users can audit changes without opening the tool trace panel.
- Surface per-edit spans/diff summaries directly in the chat transcript or status bar to make navigation to recent changes simpler.

## 8. Testing Strategy

**Current coverage**

1. `tests/test_ai_tools.py` exercises JSON parsing, patch validation, patch-only mode conversions, DocumentApplyPatchTool’s diff generation, and DiffBuilder no-op detection.
2. `tests/test_patches.py` + `tests/test_bridge.py` cover multi-hunk parsing, CRLF handling, context mismatches, patch metrics, and undo-safe bridge application.
3. `tests/test_agent.py` ensures the AI controller injects reminders until `document_edit` runs, and `tests/test_main_window.py` verifies the UI always registers the document_edit tool in patch-only mode.

**Remaining gaps**

- Add an end-to-end test that simulates a patch failure (DocumentApplyPatch + bridge conflict) and asserts that the toast, tool trace metadata, and undo stack stay consistent.
- Property-test `apply_unified_diff` with randomized CR/LF combinations and large documents to guard against quadratic `_locate_hunk` regressions.
- Exercise the telemetry path (e.g., verifying `patch_metrics.avg_latency_ms` smoothing) so refactors don’t silently drop metrics updates.

## 9. Migration & Rollout Steps

**Completed**

1. Patch parser + tests (`tinkerbell/editor/patches.py`, `tests/test_patches.py`).
2. Schema/validation updates plus bridge normalization + selection fingerprint checks.
3. Bridge/editor wiring that applies patches with a single undo snapshot and records `_last_edit_context`.
4. AI tooling/prompt/docs refresh (DocumentEdit patch-only, DocumentApplyPatch, DiffBuilder guidance, README updates).
5. Diff-first workflow hardened across UI + runtime (patch-only controller/tool wiring, no fallback toggle).
6. Full pytest suite (including legacy insert/replace regressions) green via `uv run pytest`.
7. README + tool registration already document diff workflows and troubleshooting tips.

**Upcoming hardening**

- Surface richer failure diagnostics (expected vs actual context, version tokens) in `_handle_edit_failure` and chat notices.
- Add rollout controls so teams can gather telemetry (e.g., per-conversation patch-only toggles) before making diff mode mandatory everywhere.
- Define the pathway for multi-file diffs/edits and decide whether DocumentApplyPatch should orchestrate them automatically or leave it to future tooling.

## 10. Open Questions
- Should we store the last few applied patches for quick rollback/history in the chat transcript?
- Do we want to support multi-file diffs eventually (requires file targeting in directives)?
- Is adopting `diff-match-patch` preferable for fuzzy matches, or do we keep strict unified diffs for now?
- How do we guard against extremely large diffs (size limits, streaming apply)?
- Should `DocumentApplyPatch` return the generated diff (and maybe spans) so the agent can cite it in explanations or reuse it across follow-up tool calls?
