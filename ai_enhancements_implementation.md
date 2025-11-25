# AI Enhancements Implementation Plan

This document provides a detailed, trackable implementation plan for the AI Tool & Prompt Enhancements outlined in `ai_enhancements.md`.

---

## Overview

| Workstream | Phase | Priority | Estimated Effort |
|------------|-------|----------|------------------|
| WS1: Schema Simplification | Phase 1 | High | 3-4 days |
| WS2: Prompt Simplification | Phase 2 | Medium | 1-2 days |
| WS3: Fallback & Recovery | Phase 4 | Low | 2-3 days |

---

## Workstream 1: Schema Simplification (Phase 1)

### 1.1 Simplify Versioning to `tab_id:version_id`

**Goal:** Replace redundant version fields with a compact `snapshot_token` format.

#### Implementation Tasks

- [x] **1.1.1** Modify `DocumentSnapshotTool` to emit `snapshot_token`
  - File: `src/tinkerbell/ai/tools/document_snapshot.py`
  - Add `snapshot["snapshot_token"] = f"{tab_id}:{version_id}"` to response
  - Remove old version fields (`document_version`, `content_hash`)

- [x] **1.1.2** Update `DocumentApplyPatchTool` to accept `snapshot_token`
  - File: `src/tinkerbell/ai/tools/document_apply_patch.py`
  - Add parsing logic: `tab_id, version_id = snapshot_token.split(":", 1)`
  - Add validation for malformed tokens

- [x] **1.1.3** Update `DocumentChunkTool` to use `snapshot_token`
  - File: `src/tinkerbell/ai/tools/document_chunk.py`
  - Accept `snapshot_token` as primary identifier
  - Remove `document_id` references

- [x] **1.1.4** Update `DocumentFindTextTool` to use `snapshot_token`
  - File: `src/tinkerbell/ai/tools/document_find_text.py`
  - Standardize on `tab_id` throughout
  - Include `snapshot_token` in pointer responses

- [x] **1.1.5** Update tool registry schemas
  - File: `src/tinkerbell/ai/tools/registry.py`
  - Remove `document_version`, `content_hash` from required fields
  - Remove `document_id` references
  - Update `required` arrays to use `snapshot_token`

- [x] **1.1.6** Update prompts to reference `snapshot_token`
  - File: `src/tinkerbell/ai/prompts.py`
  - Replace all references to separate version fields
  - Update examples to use new format

- [x] **1.1.7** Add unit tests for `snapshot_token` parsing
  - File: `tests/test_document_apply_patch.py`
  - Test valid tokens and malformed tokens

- [x] **1.1.8** Add integration tests for version flow
  - File: `tests/test_document_snapshot.py`
  - Verify end-to-end snapshot → edit → refresh cycle

---

### 1.2 Flatten `document_apply_patch` Schema

**Goal:** Remove complex `allOf`/`anyOf` conditions that confuse AI models.

#### Implementation Tasks

- [x] **1.2.1** Remove `patches` parameter from schema
  - File: `src/tinkerbell/ai/tools/registry.py`
  - Simplify to `required: ["snapshot_token", "target_span", "content"]`

- [x] **1.2.2** Remove `allOf`/`anyOf` blocks from schema
  - File: `src/tinkerbell/ai/tools/registry.py`
  - Replace with flat required array

- [x] **1.2.3** Create `DocumentReplaceAllTool` for full-document replacements
  - File: `src/tinkerbell/ai/tools/document_replace_all.py` (new file)
  - Minimal schema: `required: ["snapshot_token", "content"]`
  - No target_span needed

- [x] **1.2.4** Register `DocumentReplaceAllTool` in registry
  - File: `src/tinkerbell/ai/tools/registry.py`
  - Add to appropriate tool groups

- [x] **1.2.5** Add tests for `DocumentReplaceAllTool`
  - File: `tests/test_document_replace_all.py` (new file)
  - Test full replacement scenarios

- [x] **1.2.6** Update prompts to mention `DocumentReplaceAll` for full replacements
  - File: `src/tinkerbell/ai/prompts.py`

---

### 1.3 Deprecate `target_range` in Favor of `target_span`

**Goal:** Standardize on line-based spans to prevent offset drift.

#### Implementation Tasks

- [x] **1.3.1** Enable `require_line_spans=True` by default
  - File: `src/tinkerbell/ai/tools/document_apply_patch.py`
  - Set `legacy_range_adapter_enabled=False`

- [x] **1.3.2** Remove `target_range` from schema
  - File: `src/tinkerbell/ai/tools/registry.py`

- [x] **1.3.3** Update `DocumentSnapshotTool` to emit `suggested_span`
  - File: `src/tinkerbell/ai/tools/document_snapshot.py`
  - Compute `start_line` and `end_line` from `text_range`
  - Add to snapshot response

- [x] **1.3.4** Remove `target_range` references from prompts
  - File: `src/tinkerbell/ai/prompts.py`
  - Update all examples to use `target_span`

- [x] **1.3.5** Update tests to use `target_span` exclusively
  - Files: `tests/test_document_apply_patch.py`, `tests/test_patches.py`

---

### 1.4 Auto-Populate `target_span` & Snapshot Windows

**Goal:** Reduce tool call chains by auto-filling inferred spans.

#### Implementation Tasks

- [x] **1.4.1** Tighten `DocumentSnapshotTool` defaults
  - File: `src/tinkerbell/ai/tools/document_snapshot.py`
  - Honor controller-provided `text_range`/`target_span` hints
  - Clamp to last successful snapshot span when no hint provided

- [x] **1.4.2** Emit `suggested_span` in snapshot responses
  - File: `src/tinkerbell/ai/tools/document_snapshot.py`
  - Derive from returned window bounds
  - Include `start_line` and `end_line`

- [x] **1.4.3** Auto-fill `target_span` in `DocumentApplyPatchTool`
  - File: `src/tinkerbell/ai/tools/document_apply_patch.py`
  - If `target_span` omitted but `suggested_span` available, copy it
  - Apply before `needs_range` enforcement

- [x] **1.4.4** Add telemetry for auto-fill recovery path
  - File: `src/tinkerbell/ai/services/telemetry.py`
  - Log when inferred span is auto-filled
  - Track success/failure rates

- [x] **1.4.5** Update registry with new response fields
  - File: `src/tinkerbell/ai/tools/registry.py`
  - Document `suggested_span` in schema

- [x] **1.4.6** Add tests for auto-fill behavior
  - File: `tests/test_document_apply_patch.py`
  - Test with/without `suggested_span` context

---

## Workstream 2: Prompt Simplification (Phase 2)

### 2.1 Reduce Planner Instructions

**Goal:** Condense 13 instructions to 6 concise, actionable rules.

#### Implementation Tasks

- [x] **2.1.1** Rewrite `planner_instructions()` function
  - File: `src/tinkerbell/ai/prompts.py`
  - Reduce to 6 core rules
  - Focus on snapshot → find → edit → refresh cycle

- [x] **2.1.2** Review and test with sample prompts
  - Verify AI follows simplified instructions
  - Document any edge cases needing explicit guidance

---

### 2.2 Simplify Tool Execution Instructions

**Goal:** Replace extensive guidance with core recipe + error recovery.

#### Implementation Tasks

- [x] **2.2.1** Rewrite `tool_use_instructions()` function
  - File: `src/tinkerbell/ai/prompts.py`
  - Create "Edit Recipe" section with numbered steps
  - Create "Error Recovery" section with common issues

- [x] **2.2.2** Remove redundant edge case documentation
  - File: `src/tinkerbell/ai/prompts.py`
  - Keep only most common errors

---

### 2.3 Align Prompts with Registered Tools

**Goal:** Ensure prompts only reference tools that are actually available.

#### Implementation Tasks

- [x] **2.3.1** Register fallback `DocumentFindTextTool` unconditionally
  - File: `src/tinkerbell/ai/tools/registry.py`
  - Use `offline_fallback` mode when embeddings unavailable
  - Ensures tool is always available

- [x] **2.3.2** Add dynamic tool preamble to prompt assembly
  - File: `src/tinkerbell/ai/prompts.py`
  - Build "available tools" section from registry
  - Instructions match registered tools

- [x] **2.3.3** Test prompt generation with various configurations
  - Verify tool references match availability

---

### 2.4 Remove Unsupported Scope Metadata from Prompts

**Goal:** Stop referencing internal-only fields in agent-facing prompts.

#### Implementation Tasks

- [x] **2.4.1** Remove scope metadata references from prompts
  - File: `src/tinkerbell/ai/prompts.py`
  - Remove `scope_origin`, `scope_range`, `scope_length` mentions
  - Clarify that agents provide `target_span` only

- [x] **2.4.2** Add clarifying note about internal scope computation
  - File: `src/tinkerbell/ai/prompts.py`
  - Explain tool internally computes provenance

---

## Workstream 3: Fallback & Recovery Improvements (Phase 4)

### 4.1 Add Confidence Field to `DocumentFindTextTool`

**Goal:** Make low-confidence results more visible to agents.

#### Implementation Tasks

- [x] **4.1.1** Add `confidence` field to response
  - File: `src/tinkerbell/ai/tools/document_find_text.py`
  - Set to "high" or "low" based on strategy/fallback
  - Add `warning` message for low confidence

- [x] **4.1.2** Update prompts with confidence checking guidance
  - File: `src/tinkerbell/ai/prompts.py`
  - Instruct agents to verify low-confidence results

- [x] **4.1.3** Add tests for confidence field
  - File: `tests/test_document_find_text.py`
  - Test embedding vs fallback scenarios

---

### 4.2 Enrich DocumentFindText Pointers with Line Spans

**Goal:** Reduce tool chains by including line spans in pointer results.

#### Implementation Tasks

- [x] **4.2.1** Compute `line_span` for every pointer
  - File: `src/tinkerbell/ai/tools/document_find_text.py`
  - Use document's line offset table
  - Include `start_line` and `end_line`

- [x] **4.2.2** Include `tab_id` and `version_id` in pointer responses
  - File: `src/tinkerbell/ai/tools/document_find_text.py`
  - Allow agents to build `snapshot_token` directly

- [x] **4.2.3** Update prompts to highlight direct `target_span` reuse
  - File: `src/tinkerbell/ai/prompts.py`
  - Show example of using pointer results directly

- [x] **4.2.4** Add tests for enriched pointers
  - File: `tests/test_document_find_text.py`

---

### 4.3 Provide Chunk Cache Recovery Guidance

**Goal:** Help agents recover from chunk cache misses.

#### Implementation Tasks

- [x] **4.3.1** Add `retry_hint` to not_found responses
  - File: `src/tinkerbell/ai/tools/document_chunk.py`
  - Include actionable recovery instruction

- [x] **4.3.2** Implement optional auto-refresh on cache miss
  - File: `src/tinkerbell/ai/tools/document_chunk.py`
  - Call `bridge.generate_snapshot()` automatically
  - Return fresh manifest inline

- [x] **4.3.3** Add telemetry for cache miss recovery
  - File: `src/tinkerbell/ai/tools/document_chunk.py`
  - Event: `chunk_cache.miss_recovered`

- [x] **4.3.4** Add tests for recovery behavior
  - File: `tests/test_document_chunk_tool.py`

---

### 4.4 Surface Ignored Snapshot Request Fields

**Goal:** Make ignored request fields visible to agents.

#### Implementation Tasks

- [x] **4.4.1** Track and return ignored keys in response
  - File: `src/tinkerbell/ai/tools/document_snapshot.py`
  - Add `ignored_keys` array to response
  - Populate from `_coerce_request_mapping()` rejects

- [x] **4.4.2** Add telemetry for ignored fields
  - File: `src/tinkerbell/ai/tools/document_snapshot.py`
  - Log ignored key names for monitoring

- [x] **4.4.3** Update prompts with ignored keys guidance
  - File: `src/tinkerbell/ai/prompts.py`
  - Warn that ignored parameters were not applied

- [x] **4.4.4** Add tests for ignored keys reporting
  - File: `tests/test_document_snapshot.py`

---

## Testing & Validation Checklist

### Unit Tests
- [ ] All new functions have corresponding unit tests
- [ ] Existing tests updated for schema changes
- [ ] Edge cases covered (malformed tokens, missing fields, etc.)

### Integration Tests
- [ ] End-to-end snapshot → edit → refresh cycle works
- [ ] Tool registration matches prompt references

### Manual Testing
- [ ] AI agent successfully completes edit tasks with new schema
- [ ] Error recovery paths work as documented
- [ ] Token savings measured and validated

---

## Rollout Plan

### Stage 1: Development (Week 1-2)
- [ ] Complete Workstream 1 (Schema Simplification)
- [ ] Complete Workstream 2 (Prompt Simplification)
- [ ] All unit tests passing

### Stage 2: Integration (Week 3)
- [ ] Complete Workstream 3 (Fallback & Recovery)
- [ ] Integration tests passing
- [ ] Internal dogfooding

### Stage 3: Validation (Week 4)
- [ ] Measure error rate reduction
- [ ] Measure token savings
- [ ] Documentation updates

### Stage 4: Release
- [ ] Update user-facing documentation
- [ ] Monitor telemetry for regressions

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Tool call error rate | TBD | -50% |
| Tokens per edit cycle | ~60+ | ~10-15 |
| Retry rate due to stale data | TBD | -70% |
| Agent task completion rate | TBD | +20% |

---

## Dependencies & Risks

### Dependencies
- Existing telemetry infrastructure for monitoring
- Test fixtures for schema changes

### Risks
- **AI behavior regression**: Mitigated by testing
- **Token budget impact**: Mitigated by measuring before/after

---

## Notes

- Consider feature flags for gradual rollout if needed
