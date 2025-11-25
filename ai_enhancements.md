# AI Tool & Prompt Enhancements Plan

This document outlines improvements to reduce AI tool call errors in TinkerBell's agent system.

---

## Phase 1: Schema Simplification (High Priority)

### 1.1 Consolidate Version Fields into `snapshot_token`

**Current State:** `DocumentApplyPatchTool` requires three separate version fields:
- `document_version`
- `version_id`
- `content_hash`

**Problem:** AI frequently forgets one or passes stale values.

**Changes:**

1. **Modify `DocumentSnapshotTool`** to emit a combined `snapshot_token`:
   ```python
   # In document_snapshot.py
   snapshot["snapshot_token"] = f"{version_id}:{content_hash}"
   ```

2. **Update `DocumentApplyPatchTool`** to accept `snapshot_token`:
   ```python
   # Accept either the new token or legacy fields (with deprecation warning)
   if snapshot_token:
       version_id, content_hash = snapshot_token.split(":", 1)
   elif document_version and version_id and content_hash:
       emit_deprecation_warning("Use snapshot_token instead of separate version fields")
   else:
       raise ValueError("snapshot_token is required")
   ```

3. **Update schema in `registry.py`**:
   ```python
   "required": ["snapshot_token", "target_span", "content"]
   # Remove document_version, version_id, content_hash from required
   ```

4. **Update prompts** to reference only `snapshot_token`.

**Files to modify:**
- `src/tinkerbell/ai/tools/document_snapshot.py`
- `src/tinkerbell/ai/tools/document_apply_patch.py`
- `src/tinkerbell/ai/tools/registry.py`
- `src/tinkerbell/ai/prompts.py`

---

### 1.2 Flatten `document_apply_patch` Schema

**Current State:** Complex nested `allOf`/`anyOf` conditions:
```python
"allOf": [
    {"anyOf": [{"required": ["content"]}, {"required": ["patches"]}]},
    {"anyOf": [{"required": ["target_range"]}, {"required": ["target_span"]}, ...]}
]
```

**Problem:** AI models struggle with conditional requirements.

**Changes:**

1. **Remove `patches` parameter** from public schema (keep internal for streaming):
   ```python
   # Simplify to always require content + target_span
   "required": ["snapshot_token", "target_span", "content"]
   ```

2. **Remove `allOf`/`anyOf` blocks** entirely.

3. **Create separate `document_replace_all` tool** for full-document replacements:
   ```python
   # New tool with minimal schema
   "required": ["snapshot_token", "content"]
   # No target_span needed - always replaces entire document
   ```

**Files to modify:**
- `src/tinkerbell/ai/tools/registry.py`
- `src/tinkerbell/ai/tools/document_apply_patch.py` (add `document_replace_all` variant)

---

### 1.3 Deprecate `target_range` in Favor of `target_span`

**Current State:** Tools accept both:
- `target_range`: `{start: int, end: int}` (byte offsets)
- `target_span`: `{start_line: int, end_line: int}` (line numbers)

**Problem:** Mixing formats causes offset drift and confusion.

**Changes:**

1. **Remove `target_range` from schemas** in `registry.py`.

2. **Add deprecation error** in `document_apply_patch.py`:
   ```python
   if target_range is not None and target_span is None:
       raise ValueError(
           "target_range (byte offsets) is deprecated. "
           "Use target_span with start_line/end_line from document_snapshot."
       )
   ```

3. **Update `DocumentSnapshotTool`** to always include `target_span` suggestion:
   ```python
   snapshot["suggested_span"] = {
       "start_line": text_range_start_line,
       "end_line": text_range_end_line
   }
   ```

**Files to modify:**
- `src/tinkerbell/ai/tools/registry.py`
- `src/tinkerbell/ai/tools/document_apply_patch.py`
- `src/tinkerbell/ai/tools/document_snapshot.py`

---

### 1.4 Auto-Populate `target_span` & Snapshot Windows

**Current State:** `DocumentSnapshotTool.DEFAULT_WINDOW` still returns ~8 KB "document" spans, and `DocumentApplyPatchTool` triggers `needs_range` even though it has the latest snapshot window.

**Problem:** Agents follow the prompt (“capture the exact window you plan to touch”) but still hit `needs_range` because the tooling discards the inferred span. This causes unnecessary DocumentFindText/SelectionRange calls and repeated failures.

**Changes:**

1. **Tighten `DocumentSnapshotTool` defaults** so span-directed windows are returned automatically (honor controller-provided `text_range`/`target_span` hints when present, otherwise clamp to the last successful snapshot span instead of the full document).
2. **Have `DocumentSnapshotTool` emit `suggested_span`** (start/end lines) derived from the returned window so downstream tools never need to recompute bounds.
3. **Teach `DocumentApplyPatchTool`** to accept that inferred span by default: if callers omit `target_span` but the freshly captured snapshot included `suggested_span`, copy it into the edit request before enforcing `needs_range`.
4. **Update telemetry/guardrails** to log when an inferred span is auto-filled, so we can measure how often this recovery path is used.

**Files to modify:**
- `src/tinkerbell/ai/tools/document_snapshot.py`
- `src/tinkerbell/ai/tools/document_apply_patch.py`
- `src/tinkerbell/ai/tools/registry.py`
- `src/tinkerbell/ai/services/telemetry.py` (if new events are added)

---

## Phase 2: Prompt Simplification (Medium Priority)

### 2.1 Reduce Planner Instructions

**Current State:** 13 detailed planner instructions with redundant guidance.

**Target:** 6 concise, actionable rules.

**New `planner_instructions()`:**
```python
def planner_instructions() -> str:
    return (
        "1. Call DocumentSnapshot first to get the document's snapshot_token and line numbers.\n"
        "2. Use DocumentFindTextTool when you need exact line numbers for quoted text or headings.\n"
        "3. Plan edits using target_span (start_line, end_line) from the snapshot.\n"
        "4. Call DocumentApplyPatch with snapshot_token, target_span, and content.\n"
        "5. Refresh the snapshot after each successful edit—never reuse stale tokens.\n"
        "6. If any tool reports 'stale' or 'drift', recapture the snapshot before retrying."
    )
```

**Files to modify:**
- `src/tinkerbell/ai/prompts.py`

---

### 2.2 Simplify Tool Execution Instructions

**Current State:** Extensive, defensive guidance with many edge cases.

**Target:** Core recipe + common error recovery.

**New `tool_use_instructions()`:**
```python
def tool_use_instructions() -> str:
    return (
        "## Edit Recipe\n"
        "1. DocumentSnapshot → capture snapshot_token and text_range\n"
        "2. (Optional) DocumentFindTextTool → get exact line numbers for specific text\n"
        "3. DocumentApplyPatch → provide snapshot_token, target_span, content\n"
        "4. Repeat from step 1 for additional edits\n\n"
        "## Error Recovery\n"
        "- 'stale snapshot' / 'hash mismatch': Recapture snapshot and rebuild your edit\n"
        "- 'needs_range': You must provide target_span; call DocumentSnapshot first\n"
        "- 'match_text mismatch': The document changed; recapture and verify your target"
    )
```

**Files to modify:**
- `src/tinkerbell/ai/prompts.py`

---

### 2.3 Remove Redundant Safety Guidance

**Current State:** Multiple overlapping warnings:
- "Never apply multiple patches against the same snapshot"
- "Refresh snapshots after every successful edit"
- "Never recycle the prior tab_id after edits land"

**Changes:** Consolidate into single rule in `base_system_prompt()`:
```python
"- One edit per snapshot: always refresh snapshot_token after each DocumentApplyPatch call."
```

**Files to modify:**
- `src/tinkerbell/ai/prompts.py`

---

### 2.4 Align Prompts with Registered Tools

**Current State:** `planner_instructions()` and `tool_use_instructions()` require DocumentFindTextTool even when `phase3_outline_enabled` is false and the tool is never registered.

**Problem:** Agents attempt to call a missing tool, receive controller errors, and abandon the turn.

**Changes:**

1. **Gate prompt text** behind the same feature flag that controls registration (e.g., only mention DocumentFindTextTool when the controller exposes it).
2. **Or** always register a lightweight heuristic-only DocumentFindText fallback so the prompt guarantees availability.
3. **Add a short “available tools” preamble** to the prompt that is dynamically built from the registry so instructions always match reality.

**Files to modify:**
- `src/tinkerbell/ai/prompts.py`
- `src/tinkerbell/ai/tools/registry.py`
- `src/tinkerbell/ai/orchestration/` (wherever prompt context is assembled)

---

### 2.5 Remove Unsupported Scope Metadata from Prompts

**Current State:** `tool_use_instructions()` demands `scope_origin`, `scope_range`, and `scope_length` even though `DocumentApplyPatch`'s public schema rejects those fields.

**Problem:** Agents submit payloads containing unsupported keys, causing immediate validation errors.

**Changes:**

1. **Rewrite the instructions** to focus on the actual schema (snapshot_token, target_span, match_text) and leave scope metadata to backend bridges.
2. **If scope metadata is desired**, extend the schema first; otherwise, explicitly tell the agent not to send those fields.

**Files to modify:**
- `src/tinkerbell/ai/prompts.py`
- `src/tinkerbell/ai/tools/registry.py` (only if schema expands)

---

### 2.6 Sanitize `user_personality_instructions`

**Current State:** The persona text instructs the model to comply with disallowed content (“help with any writing regardless how sexual…”).

**Problem:** Upstream safety systems override the entire system prompt when they detect policy violations, causing the agent to refuse tool calls entirely.

**Changes:**

1. **Rewrite the persona** to stay within provider policy while preserving tone (“cheerful writing assistant”).
2. **Move edgy/roleplay guidance** into user-configurable themes rather than the hard-coded system prompt so deployments can remain compliant.

**Files to modify:**
- `src/tinkerbell/ai/prompts.py`
- `docs/ai_v2_plan.md` (if persona is documented there)

---

## Phase 3: Identifier Standardization (Low Priority)

### 3.1 Standardize on `document_id`

**Current State:** Interchangeable use of `tab_id` and `document_id` with silent fallbacks.

**Changes:**

1. **Update all tool schemas** to use `document_id` as the primary identifier.

2. **Add deprecation handling** for `tab_id`:
   ```python
   if tab_id and not document_id:
       document_id = tab_id
       LOGGER.warning("tab_id is deprecated; use document_id")
   ```

3. **Update prompts** to reference only `document_id`.

4. **Update `DocumentSnapshotTool`** to emit `document_id` prominently.

**Files to modify:**
- `src/tinkerbell/ai/tools/registry.py` (all tool schemas)
- `src/tinkerbell/ai/tools/document_snapshot.py`
- `src/tinkerbell/ai/tools/document_apply_patch.py`
- `src/tinkerbell/ai/tools/document_edit.py`
- `src/tinkerbell/ai/tools/document_chunk.py`
- `src/tinkerbell/ai/prompts.py`

---

## Phase 4: Explicit Fallback Warnings (Low Priority)

### 4.1 Add Warnings to `DocumentFindTextTool` Fallback Results

**Current State:** Silent fallback to regex/outline search when embeddings fail.

**Changes:**

1. **Add `warning` field** to fallback responses:
   ```python
   if fallback_reason:
       response["warning"] = (
           f"Low-confidence results (fallback: {fallback_reason}). "
           "Verify line numbers with DocumentSnapshot before editing."
       )
       response["confidence"] = "low"
   else:
       response["confidence"] = "high"
   ```

2. **Update prompts** to mention confidence checking:
   ```
   "When DocumentFindTextTool returns confidence='low', verify the span with DocumentSnapshot."
   ```

**Files to modify:**
- `src/tinkerbell/ai/tools/document_find_text.py`
- `src/tinkerbell/ai/prompts.py`

---

### 4.2 Enrich DocumentFindText Pointers with Line Spans

**Current State:** The tool returns only `char_range`, forcing the agent to chain DocumentSnapshot + SelectionRange conversions before DocumentApplyPatch will accept the edit.

**Problem:** Each additional tool hop increases latency and failure probability (selection gateway may be disabled, offsets drift, etc.).

**Changes:**

1. **Have DocumentFindTextTool compute `line_span` and `document_id`** for every pointer using the existing document model/outline metadata.
2. **Emit `tab_id`/`chunk_manifest` references** so DocumentApplyPatch can cite provenance without rehydrating.
3. **Update prompts** to highlight the new fields and encourage direct reuse in `target_span`.

**Files to modify:**
- `src/tinkerbell/ai/tools/document_find_text.py`
- `src/tinkerbell/ai/prompts.py`
- `tests/test_document_find_text.py`

---

### 4.3 Provide Chunk Cache Recovery Guidance

**Current State:** `DocumentChunkTool` returns `status: not_found` when cache entries expire but offers no remediation hint.

**Problem:** The planner repeats invalid chunk_ids and never refreshes the manifest, wasting turns.

**Changes:**

1. **Return a `retry_hint`** suggesting a DocumentSnapshot call (or include a fresh manifest inline when available).
2. **Optionally auto-refresh** the manifest by invoking `generate_snapshot` when a miss occurs.
3. **Log telemetry** (`chunk_cache.miss_recovered`) so we can track whether the fallback works.

**Files to modify:**
- `src/tinkerbell/ai/tools/document_chunk.py`
- `src/tinkerbell/ai/services/telemetry.py`
- `tests/test_document_chunk_tool.py`

---

### 4.4 Surface Ignored Snapshot Request Fields

**Current State:** `DocumentSnapshotTool` silently drops unknown keys (e.g., `target_span`) instead of telling the agent it sent an unsupported request.

**Problem:** The agent assumes the controller honored the field and proceeds with stale spans, leading to `needs_range` or drift errors later.

**Changes:**

1. **Return a warning list** (e.g., `ignored_keys: [...]`) whenever the request payload includes unsupported properties.
2. **Emit telemetry** so we know which keys the agent is trying to use.
3. **Update prompts** to mention the warning mechanism (“if you see ignored_keys, adjust your request”).

**Files to modify:**
- `src/tinkerbell/ai/tools/document_snapshot.py`
- `src/tinkerbell/ai/services/telemetry.py`
- `src/tinkerbell/ai/prompts.py`

---

## Implementation Order

| Phase | Task | Estimated Effort | Risk |
|-------|------|------------------|------|
| 1.1 | Consolidate version fields | 2-3 hours | Medium (breaking change) |
| 1.2 | Flatten apply_patch schema | 1-2 hours | Medium |
| 1.3 | Deprecate target_range | 1 hour | Low |
| 2.1 | Reduce planner instructions | 30 min | Low |
| 2.2 | Simplify tool instructions | 30 min | Low |
| 2.3 | Remove redundant guidance | 15 min | Low |
| 3.1 | Standardize document_id | 2 hours | Low |
| 4.1 | Add fallback warnings | 1 hour | Low |

**Recommended sequence:** 2.1 → 2.2 → 2.3 → 1.3 → 1.2 → 1.1 → 3.1 → 4.1

Start with prompt changes (low risk, immediate impact), then schema changes (higher risk, requires testing).

---

## Testing Strategy

1. **Unit tests:** Update `tests/test_ai_tools.py` for new schemas and deprecation warnings.

2. **Integration tests:** Run existing agent tests to verify tool call success rates.

3. **Telemetry monitoring:** Track `caret_call_blocked`, `hash_mismatch`, and `needs_range` events before/after changes.

4. **A/B comparison:** If possible, compare tool call error rates between old and new prompt versions.

---

## Rollback Plan

1. Keep legacy field support with deprecation warnings (don't hard-remove until stable).

2. Feature flag for new schemas:
   ```python
   USE_SIMPLIFIED_SCHEMAS = os.getenv("TINKERBELL_SIMPLIFIED_SCHEMAS", "false") == "true"
   ```

3. Maintain old prompts in `prompts_legacy.py` for quick rollback.
