# AI Tool & Prompt Enhancements Plan

This document outlines improvements to reduce AI tool call errors in TinkerBell's agent system.

---

## Phase 1: Schema Simplification (High Priority)

### 1.1 Simplify Versioning to `tab_id:version_id`

**Current State:** `DocumentApplyPatchTool` requires three separate version fields:
- `document_version`
- `version_id`
- `content_hash`

Plus there's confusion between `tab_id` and `document_id` throughout the codebase.

**Problem:** 
- AI frequently forgets one field or passes stale values
- Redundant identifiers waste tokens (~20-30 tokens per edit cycle)
- `content_hash` is redundant since `version_id` is monotonic and increments on every edit

**Key Insight:** In a tabbed editor, one tab = one document. The AI operates on tabs, not abstract document IDs. We only need:
1. **Which tab?** → `tab_id`
2. **Is it stale?** → `version_id` (monotonic counter)

**Changes:**

1. **Modify `DocumentSnapshotTool`** to emit a compact `snapshot_token`:
   ```python
   # In document_snapshot.py
   snapshot["snapshot_token"] = f"{tab_id}:{version_id}"
   # Example: "tab_3:42" (~8-12 chars vs 60+ chars before)
   ```

2. **Update `DocumentApplyPatchTool`** to accept `snapshot_token`:
   ```python
   if snapshot_token:
       tab_id, version_id = snapshot_token.split(":", 1)
   elif tab_id and version_id:  # Legacy support with deprecation warning
       emit_deprecation_warning("Use snapshot_token instead of separate fields")
   else:
       raise ValueError("snapshot_token is required")
   ```

3. **Remove from schema** in `registry.py`:
   - Remove `document_version`, `content_hash` from required fields
   - Remove `document_id` references (use `tab_id` exclusively)
   ```python
   "required": ["snapshot_token", "target_span", "content"]
   ```

4. **Deprecate `document_id`** across all tools—use `tab_id` exclusively.

5. **Update prompts** to reference only `snapshot_token` and `tab_id`.

**Token savings:** ~20-30 tokens per edit cycle (from ~60+ char tokens down to ~10 chars).

**Files to modify:**
- `src/tinkerbell/ai/tools/document_snapshot.py`
- `src/tinkerbell/ai/tools/document_apply_patch.py`
- `src/tinkerbell/ai/tools/document_chunk.py`
- `src/tinkerbell/ai/tools/document_find_text.py`
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

`DocumentApplyPatchTool` already has `require_line_spans` and `legacy_range_adapter_enabled` flags, plus emits `target_range.legacy_adapter` telemetry when adapting byte offsets.

**Problem:** Mixing formats causes offset drift and confusion.

**Changes:**

1. **Enable `require_line_spans=True` by default** and set `legacy_range_adapter_enabled=False`.

2. **Remove `target_range` from schema** in `registry.py` (keep internal handling for streaming patches).

3. **Update `DocumentSnapshotTool`** to compute and emit `suggested_span` from `text_range`:
   ```python
   snapshot["suggested_span"] = {
       "start_line": computed_start_line,
       "end_line": computed_end_line
   }
   ```

4. **Update prompts** to remove `target_range` references.

**Files to modify:**
- `src/tinkerbell/ai/tools/registry.py`
- `src/tinkerbell/ai/tools/document_apply_patch.py`
- `src/tinkerbell/ai/tools/document_snapshot.py`
- `src/tinkerbell/ai/prompts.py`

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

### 2.3 Align Prompts with Registered Tools

**Current State:** `planner_instructions()` and `tool_use_instructions()` reference DocumentFindTextTool unconditionally, but it's only registered when `phase3_outline_enabled` is true (see `register_phase3_tools()`).

**Problem:** Agents attempt to call a missing tool, receive controller errors, and abandon the turn.

**Changes:**

1. **Option A: Gate prompt text** – Parameterize `planner_instructions()` and `tool_use_instructions()` with a `phase3_enabled` flag so DocumentFindTextTool references are omitted when the tool isn't registered.
2. **Option B: Always register fallback** – Register a lightweight heuristic-only DocumentFindText fallback even when embeddings are unavailable (the tool already supports `offline_fallback` mode).
3. **Add dynamic tool preamble** – Build an "available tools" section from the registry at prompt assembly time so instructions always match reality.

**Recommendation:** Option B is simpler since the fallback path already exists.

**Files to modify:**
- `src/tinkerbell/ai/prompts.py`
- `src/tinkerbell/ai/tools/registry.py`

---

### 2.4 Remove Unsupported Scope Metadata from Prompts

**Current State:** `tool_use_instructions()` mentions `scope_origin`, `scope_range`, and `scope_length` as if agents should provide them, but these are **internal metadata fields** populated by `DocumentApplyPatchTool._build_scope_details()` and `_range_scope_fields()`. The public schema in `registry.py` does not expose these parameters.

**Problem:** Agents may attempt to include these fields, causing confusion or silent drops (schema uses `additionalProperties: false`).

**Changes:**

1. **Remove scope metadata references** from `tool_use_instructions()` and `planner_instructions()`.
2. **Clarify in prompts** that agents provide `target_span` and the tool internally computes scope provenance.

**Files to modify:**
- `src/tinkerbell/ai/prompts.py`

---

## Phase 3: Reserved for Future Use

*Identifier standardization has been consolidated into Phase 1.1 (standardize on `tab_id`).*

---

## Phase 4: Explicit Fallback Warnings (Low Priority)

### 4.1 Add Confidence Field to `DocumentFindTextTool` Results

**Current State:** The tool already exposes `strategy` ("embedding" vs "fallback") and `fallback_reason` in its response, but agents don't consistently check these fields before editing.

**Changes:**

1. **Add explicit `confidence` field** to make low-confidence results more prominent:
   ```python
   if fallback_reason:
       response["confidence"] = "low"
       response["warning"] = (
           f"Low-confidence results (fallback: {fallback_reason}). "
           "Verify line numbers with DocumentSnapshot before editing."
       )
   else:
       response["confidence"] = "high"
   ```

2. **Update prompts** to mention confidence checking:
   ```
   "When DocumentFindTextTool returns confidence='low' or strategy='fallback', verify the span with DocumentSnapshot before editing."
   ```

**Files to modify:**
- `src/tinkerbell/ai/tools/document_find_text.py`
- `src/tinkerbell/ai/prompts.py`

---

### 4.2 Enrich DocumentFindText Pointers with Line Spans

**Current State:** The tool returns `char_range` (byte offsets) but not `line_span` (start_line/end_line). Agents must chain DocumentSnapshot or SelectionRange calls to convert offsets before DocumentApplyPatch will accept the edit.

**Problem:** Each additional tool hop increases latency and failure probability.

**Changes:**

1. **Compute `line_span`** for every pointer by using the document's line offset table (similar to `DocumentApplyPatchTool._line_span_from_offsets()`).
2. **Include `tab_id` and `version_id`** in pointer responses so agents can build `snapshot_token` directly.
3. **Update prompts** to highlight direct `target_span` reuse from pointer results.

**Files to modify:**
- `src/tinkerbell/ai/tools/document_find_text.py`
- `src/tinkerbell/ai/prompts.py`
- `tests/test_document_find_text.py`

---

### 4.3 Provide Chunk Cache Recovery Guidance

**Current State:** `DocumentChunkTool` returns `status: not_found` with `chunk_id`, `document_id`, and `cache_key` but no actionable recovery hint. The `_emit_cache_miss()` logs telemetry but the response doesn't guide the agent.

**Problem:** The planner repeats invalid chunk_ids and never refreshes the manifest, wasting turns.

**Changes:**

1. **Add `retry_hint` field** to not_found responses:
   ```python
   return {
       "status": "not_found",
       "chunk_id": chunk_id,
       "document_id": document_id,
       "cache_key": cache_key,
       "retry_hint": "Call DocumentSnapshot to refresh the chunk_manifest before retrying.",
   }
   ```
2. **Optionally auto-refresh** the manifest by calling `self.bridge.generate_snapshot()` on cache miss and returning the fresh manifest inline.
3. **Add telemetry event** `chunk_cache.miss_recovered` when auto-refresh succeeds.

**Files to modify:**
- `src/tinkerbell/ai/tools/document_chunk.py`
- `tests/test_document_chunk_tool.py`

---

### 4.4 Surface Ignored Snapshot Request Fields

**Current State:** `DocumentSnapshotTool._coerce_request_mapping()` logs ignored keys via `LOGGER.debug()` but doesn't include them in the response. Agents don't see the warning.

**Problem:** The agent assumes the controller honored unsupported fields (e.g., `target_span` passed to snapshot) and proceeds with incorrect assumptions.

**Changes:**

1. **Track ignored keys** during request coercion and include them in the response:
   ```python
   if request_kwargs:
       snapshot["ignored_keys"] = sorted(request_kwargs.keys())
   ```
2. **Emit telemetry** with the ignored key names for monitoring.
3. **Update prompts** to mention: "If DocumentSnapshot returns `ignored_keys`, those parameters were not applied."

**Files to modify:**
- `src/tinkerbell/ai/tools/document_snapshot.py`
- `src/tinkerbell/ai/prompts.py`

---
