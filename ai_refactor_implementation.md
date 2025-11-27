# AI Toolset Redesign - Technical Implementation Plan

This document provides a detailed, trackable implementation plan for the AI toolset redesign outlined in `ai_refactor.md`. The plan is organized into parallel workstreams with dependency tracking.

---

## Overview

| Workstream | Description | Est. Effort | Dependencies |
|------------|-------------|-------------|--------------|
| WS1: Core Infrastructure | Version tokens, locking, atomicity | 5-7 days | None |
| WS2: Navigation Tools | `list_tabs`, `read_document`, `get_outline` | 3-4 days | WS1.1 |
| WS3: Search Tools | `search_document` with semantic/exact/regex | 3-4 days | WS2.1 |
| WS4: Writing Tools | `insert_lines`, `replace_lines`, `delete_lines` | 4-5 days | WS1, WS2 |
| WS5: Bulk Operations | `find_and_replace`, `write_document` | 2-3 days | WS4 |
| WS6: Subagent System | `analyze_document`, `transform_document` | 6-8 days | WS1-WS5 |
| WS7: UI Integration | Diff review, turn checkpoints, editor locking | 4-5 days | WS1 |
| WS8: Migration & Cleanup | Deprecation, prompt updates, tests | 3-4 days | WS1-WS7 |

**Total Estimated Effort:** 30-40 days (can be parallelized to ~15-20 days with 2 developers)

---

## Workstream 1: Core Infrastructure

### 1.1 Simplified Version Token System

**Goal:** Replace complex `snapshot_token = "tab_id:version_id"` with simple opaque `version = "42"` integer string.

#### Current State
- `DocumentSnapshotTool` emits `snapshot_token = f"{tab_id}:{version_id}"`
- `DocumentApplyPatchTool` parses token to extract components
- Version tracking tied to `DocumentState.version_id`

#### Implementation Tasks

- [ ] **1.1.1** Add `version` field to `DocumentState`
  - File: `src/tinkerbell/editor/document_model.py`
  - Add `version: str` field (simple incrementing integer as string)
  - Increment on every content modification
  - Separate from internal `version_id` used for conflict detection

- [ ] **1.1.2** Update `DocumentBridge.generate_snapshot()` to emit `version`
  - File: `src/tinkerbell/services/bridge.py`
  - Add `version` to snapshot response (alongside existing fields for compatibility)
  - Version should be the simple string format: `"1"`, `"2"`, etc.

- [ ] **1.1.3** Create version validation utility
  - File: `src/tinkerbell/ai/tools/validation.py`
  - Add `validate_version(provided: str, current: str) -> bool`
  - Add `VersionMismatchError` exception class

- [ ] **1.1.4** Add version reset behavior
  - File: `src/tinkerbell/editor/document_model.py`
  - Reset to `"1"` on document reload from disk
  - Continue incrementing on save (no reset)

- [ ] **1.1.5** Add unit tests for version system
  - File: `tests/test_version_tokens.py` (new)
  - Test increment, validation, reset behaviors

---

### 1.2 Global Editor Lock During AI Turns

**Goal:** Lock all editor tabs to view-only mode while AI is working.

#### Implementation Tasks

- [ ] **1.2.1** Create `AITurnLock` class
  - File: `src/tinkerbell/ai/orchestration/turn_lock.py` (new)
  - Methods: `acquire()`, `release()`, `is_locked() -> bool`
  - Signal/callback for UI to respond to lock state changes

- [ ] **1.2.2** Integrate lock into `AIController`
  - File: `src/tinkerbell/ai/orchestration/controller.py`
  - Acquire lock at turn start
  - Release lock at turn end (success, error, or cancel)

- [ ] **1.2.3** Add lock status to `EditorWidget`
  - File: `src/tinkerbell/editor/editor_widget.py`
  - Connect to lock signal
  - Set read-only mode when locked

- [ ] **1.2.4** Add status bar indicator
  - File: `src/tinkerbell/ui/status_bar.py` (or similar)
  - Show "🤖 AI working..." when locked
  - Show subagent count if applicable

- [ ] **1.2.5** Add cancel mechanism
  - File: `src/tinkerbell/ai/orchestration/controller.py`
  - User can cancel AI turn, releasing lock immediately
  - Emit cancellation telemetry

- [ ] **1.2.6** Add tests for lock behavior
  - File: `tests/test_turn_lock.py` (new)
  - Test acquire/release, cancel, concurrent access

---

### 1.3 Atomic Operations with Rollback

**Goal:** Stage all edits within a turn; commit on success, rollback on failure.

#### Implementation Tasks

- [ ] **1.3.1** Create `EditTransaction` class
  - File: `src/tinkerbell/ai/orchestration/transaction.py` (new)
  - Collect pending edits per tab
  - Methods: `stage(tab_id, edit)`, `commit()`, `rollback()`
  - Store pre-edit snapshots for rollback

- [ ] **1.3.2** Integrate transaction into `AIController`
  - File: `src/tinkerbell/ai/orchestration/controller.py`
  - Create transaction at turn start
  - Tools stage edits to transaction instead of applying directly
  - Commit/rollback at turn end

- [ ] **1.3.3** Update `DocumentBridge` to support staged edits
  - File: `src/tinkerbell/services/bridge.py`
  - Add `stage_edit()` method (does not apply immediately)
  - Add `apply_staged_edits()` method (batch apply)
  - Add `discard_staged_edits()` method

- [ ] **1.3.4** Update writing tools to use staging
  - Files: All new writing tools
  - Call `transaction.stage()` instead of direct bridge methods

- [ ] **1.3.5** Add transaction telemetry
  - Emit `transaction.commit` / `transaction.rollback` events
  - Include edit count, affected tabs, duration

- [ ] **1.3.6** Add tests for transaction system
  - File: `tests/test_transaction.py` (new)
  - Test multi-edit commit, partial failure rollback

---

### 1.4 Turn Checkpoint System

**Goal:** Create restorable checkpoints after each AI turn.

#### Implementation Tasks

- [ ] **1.4.1** Create `TurnCheckpoint` dataclass
  - File: `src/tinkerbell/ai/orchestration/checkpoints.py` (new)
  - Fields: `turn_id`, `timestamp`, `summary`, `tab_snapshots: dict[str, DocumentState]`

- [ ] **1.4.2** Create `CheckpointManager` class
  - File: `src/tinkerbell/ai/orchestration/checkpoints.py`
  - Methods: `create_checkpoint()`, `restore_checkpoint()`, `list_checkpoints()`
  - Per-document checkpoint storage
  - Configurable max checkpoints (default: 20)

- [ ] **1.4.3** Integrate checkpoints into `AIController`
  - File: `src/tinkerbell/ai/orchestration/controller.py`
  - Create checkpoint before each turn
  - Store turn summary in checkpoint

- [ ] **1.4.4** Add checkpoint UI panel (optional for Phase 1)
  - File: `src/tinkerbell/ui/checkpoint_panel.py` (new)
  - List checkpoints with timestamps and summaries
  - Restore button with confirmation

- [ ] **1.4.5** Add tests for checkpoint system
  - File: `tests/test_checkpoints.py` (new)

---

## Workstream 2: Navigation Tools

### 2.1 `list_tabs` Tool

**Goal:** Refactor existing `ListTabsTool` to match new schema.

#### Current State
- `ListTabsTool` exists in `src/tinkerbell/ai/tools/list_tabs.py`
- Returns basic tab metadata

#### Implementation Tasks

- [ ] **2.1.1** Extend `ListTabsTool` response schema
  - File: `src/tinkerbell/ai/tools/list_tabs.py`
  - Add `file_type`, `version`, `size_chars`, `line_count` fields
  - Add `is_active` boolean per tab
  - Rename internal fields for consistency

- [ ] **2.1.2** Update `TabListingProvider` protocol
  - File: `src/tinkerbell/ai/tools/list_tabs.py`
  - Add methods for additional metadata retrieval

- [ ] **2.1.3** Update `DocumentWorkspace` to provide extended metadata
  - File: `src/tinkerbell/documents/workspace.py` (or equivalent)
  - Implement new protocol methods

- [ ] **2.1.4** Update tool registry schema
  - File: `src/tinkerbell/ai/tools/registry.py`
  - Update `list_tabs` description and response schema

- [ ] **2.1.5** Add/update tests
  - File: `tests/test_list_tabs.py` (new or existing)

---

### 2.2 `read_document` Tool

**Goal:** Create new tool replacing `document_snapshot` with simpler line-based reading.

#### Implementation Tasks

- [ ] **2.2.1** Create `ReadDocumentTool` class
  - File: `src/tinkerbell/ai/tools/read_document.py` (new)
  - Parameters: `tab_id`, `start_line`, `end_line`
  - Returns: `content`, `version`, `lines` metadata, `has_more`, `continuation_hint`

- [ ] **2.2.2** Implement automatic token budgeting
  - File: `src/tinkerbell/ai/tools/read_document.py`
  - Default max ~6000 tokens per response
  - Calculate end_line automatically if not specified

- [ ] **2.2.3** Add line offset tracking to `DocumentState`
  - File: `src/tinkerbell/editor/document_model.py`
  - Efficiently convert between line numbers and character offsets

- [ ] **2.2.4** Handle empty document case
  - Return `content: ""`, `lines.total: 0`, valid version

- [ ] **2.2.5** Register tool in registry
  - File: `src/tinkerbell/ai/tools/registry.py`
  - Add schema and description

- [ ] **2.2.6** Add comprehensive tests
  - File: `tests/test_read_document.py` (new)
  - Test pagination, empty docs, token limiting

---

### 2.3 `get_outline` Tool

**Goal:** Refactor `DocumentOutlineTool` to match new schema with file-type-aware detection.

#### Current State
- `DocumentOutlineTool` exists with budget-aware trimming
- Uses cached outlines from `DocumentSummaryMemory`

#### Implementation Tasks

- [ ] **2.3.1** Add file-type detection to outline generation
  - File: `src/tinkerbell/ai/tools/document_outline.py`
  - Detect markdown headings (`#`, `##`)
  - Detect JSON/YAML structure (top-level keys)
  - Add `detection_method` to response

- [ ] **2.3.2** Implement plain text heuristics
  - File: `src/tinkerbell/ai/tools/document_outline.py`
  - Chapter marker detection (Chapter 1, CHAPTER ONE, etc.)
  - Centered/emphasized title detection
  - Paragraph break fallback
  - Add `detection_confidence` field

- [ ] **2.3.3** Update response schema
  - File: `src/tinkerbell/ai/tools/document_outline.py`
  - Add `line_start`, `line_end` to each node
  - Add `file_type`, `detection_method`, `detection_confidence`
  - Remove internal fields not needed by AI

- [ ] **2.3.4** Handle unstructured documents
  - Return empty outline with helpful suggestion
  - Guide AI to use `search_document` instead

- [ ] **2.3.5** Update tool registry
  - File: `src/tinkerbell/ai/tools/registry.py`

- [ ] **2.3.6** Add tests for all detection methods
  - File: `tests/test_document_outline_tool.py`

---

## Workstream 3: Search Tools

### 3.1 `search_document` Tool

**Goal:** Create unified search tool combining semantic, exact, and regex modes.

#### Current State
- `DocumentFindTextTool` handles embedding-based search with fallback
- `SearchReplaceTool` has regex/exact matching logic

#### Implementation Tasks

- [ ] **3.1.1** Create `SearchDocumentTool` class
  - File: `src/tinkerbell/ai/tools/search_document.py` (new)
  - Parameters: `tab_id`, `query`, `match_type`, `max_results`, `include_context`
  - Returns: matches with `line`, `score`, `preview`, `context`

- [ ] **3.1.2** Implement semantic search mode
  - Integrate with `DocumentEmbeddingIndex`
  - Add `embedding_status` to response ("ready", "indexing", "unavailable")
  - Just-in-time indexing for first search

- [ ] **3.1.3** Implement exact search mode
  - Case-sensitive literal text matching
  - Return all occurrences with context

- [ ] **3.1.4** Implement regex search mode
  - Pattern validation with helpful error messages
  - Support common flags (case-insensitive, multiline)

- [ ] **3.1.5** Add context extraction
  - Configurable context lines before/after match
  - Include `start_line`, `end_line` for direct use in edits

- [ ] **3.1.6** Register tool and add tests
  - File: `src/tinkerbell/ai/tools/registry.py`
  - File: `tests/test_search_document.py` (new)

---

## Workstream 4: Writing Tools

### 4.1 `create_document` Tool

**Goal:** Simple tool for creating new documents.

#### Implementation Tasks

- [ ] **4.1.1** Create `CreateDocumentTool` class
  - File: `src/tinkerbell/ai/tools/create_document.py` (new)
  - Parameters: `title`, `content`, `file_type`
  - Returns: `tab_id`, `title`, `version`

- [ ] **4.1.2** Integrate with `DocumentWorkspace`
  - Create new tab with specified content
  - Auto-detect file type from extension if not specified

- [ ] **4.1.3** Handle title conflicts
  - Return `title_exists` error if duplicate
  - Suggest alternative name

- [ ] **4.1.4** Register and test
  - File: `src/tinkerbell/ai/tools/registry.py`
  - File: `tests/test_create_document.py` (new)

---

### 4.2 `insert_lines` Tool

**Goal:** Purely additive insertion tool.

#### Current State
- `DocumentInsertTool` exists but uses different parameter schema
- Delegates to `DocumentApplyPatchTool`

#### Implementation Tasks

- [ ] **4.2.1** Create `InsertLinesTool` class
  - File: `src/tinkerbell/ai/tools/insert_lines.py` (new)
  - Parameters: `version`, `after_line`, `content`, `tab_id`, `dry_run`, `match_text`
  - Returns: `version` (new), `status`, `inserted_at` details

- [ ] **4.2.2** Implement insertion logic
  - Convert line number to character offset
  - Ensure newline handling is consistent
  - Support `-1` for document start

- [ ] **4.2.3** Implement dry_run mode
  - Validate without applying
  - Return what would happen

- [ ] **4.2.4** Implement match_text drift recovery
  - If provided, locate text and adjust insertion point
  - Error if not found or ambiguous

- [ ] **4.2.5** Integrate with transaction system (WS1.3)
  - Stage edit instead of direct apply

- [ ] **4.2.6** Register and test
  - File: `src/tinkerbell/ai/tools/registry.py`
  - File: `tests/test_insert_lines.py` (new)

---

### 4.3 `replace_lines` Tool

**Goal:** Line-range replacement tool.

#### Implementation Tasks

- [ ] **4.3.1** Create `ReplaceLinesTool` class
  - File: `src/tinkerbell/ai/tools/replace_lines.py` (new)
  - Parameters: `version`, `start_line`, `end_line`, `content`, `tab_id`, `dry_run`, `match_text`
  - Returns: `version` (new), `status`, `lines_affected`

- [ ] **4.3.2** Implement replacement logic
  - Validate line range (end >= start)
  - Convert to character offsets
  - Support empty content for deletion

- [ ] **4.3.3** Implement dry_run mode
  - Return preview without applying

- [ ] **4.3.4** Implement match_text drift recovery
  - Locate anchor text and adjust range

- [ ] **4.3.5** Integrate with transaction system
  - Stage edit for atomic commit

- [ ] **4.3.6** Register and test
  - File: `src/tinkerbell/ai/tools/registry.py`
  - File: `tests/test_replace_lines.py` (new)

---

### 4.4 `delete_lines` Tool

**Goal:** Explicit deletion tool (convenience wrapper).

#### Implementation Tasks

- [ ] **4.4.1** Create `DeleteLinesTool` class
  - File: `src/tinkerbell/ai/tools/delete_lines.py` (new)
  - Parameters: `version`, `start_line`, `end_line`, `tab_id`, `dry_run`
  - Internally calls `ReplaceLinesTool` with `content=""`

- [ ] **4.4.2** Register and test
  - File: `src/tinkerbell/ai/tools/registry.py`
  - File: `tests/test_delete_lines.py` (new)

---

## Workstream 5: Bulk Operations

### 5.1 `find_and_replace` Tool

**Goal:** Document-wide or scoped search/replace with preview.

#### Current State
- `SearchReplaceTool` exists with similar functionality
- Needs schema alignment and preview mode refinement

#### Implementation Tasks

- [ ] **5.1.1** Create `FindAndReplaceTool` class
  - File: `src/tinkerbell/ai/tools/find_and_replace.py` (new)
  - Parameters: `version`, `find`, `replace`, `is_regex`, `case_sensitive`, `whole_word`, `max_replacements`, `tab_id`, `preview`, `scope`

- [ ] **5.1.2** Implement preview mode
  - When `preview=true`, don't modify document
  - Return match list with before/after snippets
  - Truncate to 20 matches with `preview_truncated` flag

- [ ] **5.1.3** Implement scoped replacement
  - Accept `scope: {start_line, end_line}`
  - Limit replacements to specified range

- [ ] **5.1.4** Integrate with transaction system
  - Stage changes for atomic commit

- [ ] **5.1.5** Register and test
  - File: `src/tinkerbell/ai/tools/registry.py`
  - File: `tests/test_find_and_replace.py` (new)

---

### 5.2 `write_document` Tool

**Goal:** Full document replacement (refactor existing).

#### Current State
- `DocumentReplaceAllTool` exists
- Needs schema alignment

#### Implementation Tasks

- [ ] **5.2.1** Refactor `DocumentReplaceAllTool` to `WriteDocumentTool`
  - File: `src/tinkerbell/ai/tools/write_document.py` (new or rename)
  - Parameters: `version`, `content`, `tab_id`
  - Returns: `version` (new), `lines_affected`

- [ ] **5.2.2** Integrate with transaction system

- [ ] **5.2.3** Update registry and tests

---

## Workstream 6: Subagent System

### 6.1 Subagent Coordinator Enhancement

**Goal:** Enhance existing `SubagentManager` for document-wide operations.

#### Current State
- `SubagentManager` exists with job queue and executor
- Handles chunk-based processing

#### Implementation Tasks

- [ ] **6.1.1** Create document chunking strategy
  - File: `src/tinkerbell/ai/agents/subagents/chunking.py` (new)
  - Auto-chunk documents exceeding ~20,000 characters
  - Target ~4,000 tokens per chunk
  - Overlap for context continuity

- [ ] **6.1.2** Create result aggregator
  - File: `src/tinkerbell/ai/agents/subagents/aggregator.py` (new)
  - Combine chunk results into unified output
  - Deduplication and conflict resolution

- [ ] **6.1.3** Add subagent status to UI
  - Show active subagent count in status bar
  - Update `AITurnLock` to track subagent state

---

### 6.2 `analyze_document` Tool

**Goal:** Subagent-powered document analysis.

#### Implementation Tasks

- [ ] **6.2.1** Create `AnalyzeDocumentTool` class
  - File: `src/tinkerbell/ai/tools/analyze_document.py` (new)
  - Parameters: `tab_id`, `task`, `custom_prompt`, `output_tab`
  - Tasks: "characters", "plot", "style", "summary", "custom"

- [ ] **6.2.2** Implement analysis task handlers
  - Character extraction: names, mentions, first appearance
  - Plot analysis: threads, conflicts, resolutions
  - Style analysis: tone, vocabulary, sentence structure
  - Summary: condensed document overview

- [ ] **6.2.3** Implement output tab creation
  - Auto-create tab named `"{source} - {task}"`
  - Overwrite if `output_tab` specified

- [ ] **6.2.4** Integrate with `SubagentManager`
  - Spawn chunk analyzers in parallel
  - Aggregate results

- [ ] **6.2.5** Register and test
  - File: `src/tinkerbell/ai/tools/registry.py`
  - File: `tests/test_analyze_document.py` (new)

---

### 6.3 `transform_document` Tool

**Goal:** Subagent-powered document transformation.

#### Implementation Tasks

- [ ] **6.3.1** Create `TransformDocumentTool` class
  - File: `src/tinkerbell/ai/tools/transform_document.py` (new)
  - Parameters: `tab_id`, `transformation`, `params`, `custom_prompt`, `output_mode`, `version`

- [ ] **6.3.2** Implement transformation handlers
  - `character_rename`: Name + pronoun updates
  - `setting_change`: Location + cultural detail adaptation
  - `style_rewrite`: Tone/formality conversion
  - `custom`: User-defined transformation

- [ ] **6.3.3** Implement consistency checker
  - Validate changes don't break continuity
  - Track entity references across chunks

- [ ] **6.3.4** Implement output modes
  - `new_tab`: Create new document with result
  - `in_place`: Modify source (requires version, goes through diff review)

- [ ] **6.3.5** Integrate with transaction system
  - Stage all chunk transformations
  - Atomic commit/rollback

- [ ] **6.3.6** Register and test
  - File: `src/tinkerbell/ai/tools/registry.py`
  - File: `tests/test_transform_document.py` (new)

---

## Workstream 7: UI Integration

### 7.1 Diff Review System

**Goal:** Show diff view after AI turn completes, user accepts/rejects.

#### Implementation Tasks

- [ ] **7.1.1** Create `DiffReviewPanel` widget
  - File: `src/tinkerbell/ui/diff_review_panel.py` (new)
  - Side-by-side or unified diff view
  - Syntax highlighting for changes
  - Accept/Reject buttons

- [ ] **7.1.2** Create `DiffReviewManager` class
  - File: `src/tinkerbell/ai/orchestration/diff_review.py` (new)
  - Generate diff from staged edits
  - Handle multi-tab diffs with tabbed view

- [ ] **7.1.3** Integrate with `AIController`
  - After turn completes, show diff review
  - Block until user accepts/rejects
  - On accept: commit transaction
  - On reject: rollback transaction

- [ ] **7.1.4** Add keyboard shortcuts
  - Accept: Ctrl+Enter or similar
  - Reject: Escape

- [ ] **7.1.5** Add tests
  - File: `tests/test_diff_review.py` (new)

---

### 7.2 Turn History Panel

**Goal:** Browse and restore past checkpoints.

#### Implementation Tasks

- [ ] **7.2.1** Create `TurnHistoryPanel` widget
  - File: `src/tinkerbell/ui/turn_history_panel.py` (new)
  - List checkpoints with timestamp, summary
  - Select to preview diff from current
  - Restore button

- [ ] **7.2.2** Integrate with `CheckpointManager`
  - Subscribe to checkpoint creation events
  - Update panel on new checkpoints

- [ ] **7.2.3** Implement restore confirmation
  - Show diff before restore
  - Create new checkpoint on restore (non-destructive)

---

### 7.3 Status Bar Integration

**Goal:** Show AI turn status in status bar.

#### Implementation Tasks

- [ ] **7.3.1** Add AI status section to status bar
  - Show when AI is idle/working
  - Show subagent count when active
  - Show cancel button

- [ ] **7.3.2** Connect to `AITurnLock` signals

---

## Workstream 8: Migration & Cleanup

### 8.1 Deprecate Old Tools

**Goal:** Mark old tools as deprecated, guide migration.

#### Implementation Tasks

- [ ] **8.1.1** Add deprecation warnings to old tools
  - `document_snapshot` → `read_document`
  - `document_apply_patch` → `insert_lines`/`replace_lines`
  - `document_insert` → `insert_lines`
  - `document_replace_all` → `write_document`

- [ ] **8.1.2** Update tool registry to show deprecation
  - Add `deprecated: true` flag
  - Add `replacement` suggestion

- [ ] **8.1.3** Add migration guide to documentation
  - Map old tool calls to new equivalents
  - Sample code transformations

---

### 8.2 Update Prompts

**Goal:** Update AI prompts to use new tools.

#### Implementation Tasks

- [ ] **8.2.1** Rewrite `planner_instructions()`
  - File: `src/tinkerbell/ai/prompts.py`
  - Focus on new tool workflow
  - Remove references to old tools

- [ ] **8.2.2** Rewrite `tool_use_instructions()`
  - File: `src/tinkerbell/ai/prompts.py`
  - Document new tool parameters
  - Include error recovery guidance

- [ ] **8.2.3** Update sample workflows in prompts
  - Show read → find → edit patterns
  - Show subagent usage patterns

---

### 8.3 Comprehensive Testing

**Goal:** Ensure all new tools are thoroughly tested.

#### Implementation Tasks

- [ ] **8.3.1** Unit tests for all new tools
  - Cover happy path and error cases
  - Test parameter validation

- [ ] **8.3.2** Integration tests
  - End-to-end read → edit → verify cycles
  - Multi-tab edit scenarios
  - Subagent workflows

- [ ] **8.3.3** Performance tests
  - Large document handling
  - Many-chunk subagent processing

---

### 8.4 Documentation Update

**Goal:** Update all documentation for new system.

#### Implementation Tasks

- [ ] **8.4.1** Update README.md
  - New tool table
  - Updated architecture diagram

- [ ] **8.4.2** Update docs/operations/*.md
  - New telemetry events
  - New error codes

- [ ] **8.4.3** Create migration guide
  - File: `docs/tool_migration_guide.md` (new)

---

## Implementation Order & Dependencies

### Phase 1: Foundation (Week 1-2)
```
WS1.1 (Version Tokens) ──────────────┐
WS1.2 (Editor Lock) ─────────────────┼──► WS2 (Navigation Tools)
WS1.3 (Transactions) ────────────────┤
WS1.4 (Checkpoints) ─────────────────┘
```

### Phase 2: Core Tools (Week 2-3)
```
WS2 (Navigation) ──────────────────┐
                                   ├──► WS4 (Writing Tools)
WS3 (Search) ──────────────────────┘
```

### Phase 3: Advanced Features (Week 3-4)
```
WS4 (Writing) ─────────────────────┐
                                   ├──► WS5 (Bulk Operations)
WS3 (Search) ──────────────────────┘    WS6 (Subagents)
```

### Phase 4: UI & Polish (Week 4-5)
```
WS1-6 ─────────────────────────────┬──► WS7 (UI Integration)
                                   └──► WS8 (Migration & Cleanup)
```

---

## File Change Summary

### New Files to Create
| File | Purpose |
|------|---------|
| `src/tinkerbell/ai/tools/read_document.py` | `read_document` tool |
| `src/tinkerbell/ai/tools/search_document.py` | `search_document` tool |
| `src/tinkerbell/ai/tools/create_document.py` | `create_document` tool |
| `src/tinkerbell/ai/tools/insert_lines.py` | `insert_lines` tool |
| `src/tinkerbell/ai/tools/replace_lines.py` | `replace_lines` tool |
| `src/tinkerbell/ai/tools/delete_lines.py` | `delete_lines` tool |
| `src/tinkerbell/ai/tools/find_and_replace.py` | `find_and_replace` tool |
| `src/tinkerbell/ai/tools/write_document.py` | `write_document` tool |
| `src/tinkerbell/ai/tools/analyze_document.py` | `analyze_document` tool |
| `src/tinkerbell/ai/tools/transform_document.py` | `transform_document` tool |
| `src/tinkerbell/ai/orchestration/turn_lock.py` | Editor lock during AI turns |
| `src/tinkerbell/ai/orchestration/transaction.py` | Atomic edit transactions |
| `src/tinkerbell/ai/orchestration/checkpoints.py` | Turn checkpoint system |
| `src/tinkerbell/ai/orchestration/diff_review.py` | Diff review manager |
| `src/tinkerbell/ai/agents/subagents/chunking.py` | Document chunking strategy |
| `src/tinkerbell/ai/agents/subagents/aggregator.py` | Result aggregation |
| `src/tinkerbell/ui/diff_review_panel.py` | Diff review UI widget |
| `src/tinkerbell/ui/turn_history_panel.py` | Turn history UI widget |
| `tests/test_version_tokens.py` | Version system tests |
| `tests/test_turn_lock.py` | Lock system tests |
| `tests/test_transaction.py` | Transaction tests |
| `tests/test_checkpoints.py` | Checkpoint tests |
| `tests/test_read_document.py` | `read_document` tests |
| `tests/test_search_document.py` | `search_document` tests |
| `tests/test_create_document.py` | `create_document` tests |
| `tests/test_insert_lines.py` | `insert_lines` tests |
| `tests/test_replace_lines.py` | `replace_lines` tests |
| `tests/test_delete_lines.py` | `delete_lines` tests |
| `tests/test_find_and_replace.py` | `find_and_replace` tests |
| `tests/test_analyze_document.py` | `analyze_document` tests |
| `tests/test_transform_document.py` | `transform_document` tests |
| `tests/test_diff_review.py` | Diff review tests |
| `docs/tool_migration_guide.md` | Migration documentation |

### Files to Modify
| File | Changes |
|------|---------|
| `src/tinkerbell/editor/document_model.py` | Add `version` field, line offset tracking |
| `src/tinkerbell/services/bridge.py` | Add staged edit support, emit `version` |
| `src/tinkerbell/ai/tools/list_tabs.py` | Extend response schema |
| `src/tinkerbell/ai/tools/document_outline.py` | File-type detection, line numbers |
| `src/tinkerbell/ai/tools/registry.py` | Register new tools, deprecate old |
| `src/tinkerbell/ai/orchestration/controller.py` | Integrate lock, transaction, checkpoints |
| `src/tinkerbell/ai/agents/subagents/manager.py` | Enhanced chunking/aggregation |
| `src/tinkerbell/ai/prompts.py` | Update for new tools |
| `src/tinkerbell/editor/editor_widget.py` | Read-only mode for lock |
| `README.md` | Updated tool documentation |

---

## Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Tool call error rate | TBD | -60% |
| Tokens per edit cycle | ~60+ | ~10-15 |
| Version mismatch errors | TBD | -80% |
| AI task completion rate | TBD | +25% |
| User satisfaction (diff review) | N/A | >80% |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking existing workflows | Deprecation period, not removal; dual support |
| Performance regression | Benchmark before/after; lazy loading |
| Subagent reliability | Result caching, retry logic, graceful degradation |
| UI complexity | Phased rollout; simple defaults |
| Test coverage gaps | Mandatory tests for each task |

---

## Notes

- All checkboxes can be checked off as tasks are completed
- Each workstream can have multiple developers working in parallel on non-dependent tasks
- Consider feature flags for gradual rollout of UI changes
- Telemetry should be added for all new tools to track adoption and errors
