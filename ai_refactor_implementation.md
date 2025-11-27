# AI Toolset Refactor - Technical Implementation Plan

This document provides a detailed technical implementation plan for the complete rewrite of TinkerBell's AI toolset, as designed in `ai_refactor.md`.

---

## Overview

**Objective**: Replace the existing complex, overlapping AI tool system with a clean, minimal toolset organized around clear responsibilities.

**Approach**: Complete rewrite—remove all legacy tools and code, implement new tools from scratch.

**Timeline Reference**: 5 phases as outlined in the design document.

---

## Current Code Inventory (To Be Removed)

### Tools to Delete (`src/tinkerbell/ai/tools/`)
- [ ] `document_snapshot.py` - Replaced by `read_document`
- [ ] `document_apply_patch.py` - Replaced by `replace_lines` + `insert_lines`
- [ ] `document_chunk.py` - Automatic via subagents
- [ ] `document_edit.py` - Replaced by explicit edit tools
- [ ] `document_find_text.py` - Replaced by `search_document`
- [ ] `document_insert.py` - Replaced by `insert_lines`
- [ ] `document_replace_all.py` - Replaced by `write_document`
- [ ] `document_outline.py` - Replaced by `get_outline`
- [ ] `document_plot_state.py` - Replaced by `analyze_document`
- [ ] `plot_state_update.py` - Replaced by `transform_document`
- [ ] `character_edit_planner.py` - Replaced by `transform_document`
- [ ] `character_map.py` - Replaced by `analyze_document`
- [ ] `diff_builder.py` - No longer needed (edits are direct)
- [ ] `search_replace.py` - Replaced by `find_and_replace`
- [ ] `selection_range.py` - Remove (not in new design)
- [ ] `tool_usage_advisor.py` - Remove (simplified flow)
- [ ] `validation.py` - Keep/adapt for format validation
- [ ] `registry.py` - Complete rewrite

### Memory/State to Delete or Refactor (`src/tinkerbell/ai/memory/`)
- [ ] `chunk_index.py` - Rewrite for new chunking system
- [ ] `plot_memory.py` - Absorbed into `analyze_document` subagent
- [ ] `plot_state.py` - Absorbed into `analyze_document` subagent
- [ ] `character_map.py` - Absorbed into `analyze_document` subagent
- [ ] `buffers.py` - Review for relevance
- [ ] `cache_bus.py` - Keep, adapt for new version system
- [ ] `embeddings.py` - Keep for semantic search
- [ ] `result_cache.py` - Keep, adapt for new caching needs

### Services to Refactor (`src/tinkerbell/ai/services/`)
- [ ] `context_policy.py` - Adapt for new token budgeting
- [ ] `outline_worker.py` - Rewrite for new `get_outline` tool
- [ ] `summarizer.py` - Keep for subagent summarization
- [ ] `telemetry.py` - Keep, update event names
- [ ] `trace_compactor.py` - Review for relevance

### Orchestration to Refactor (`src/tinkerbell/ai/orchestration/`)
- [ ] `controller.py` - Major refactor for new tool dispatch
- [ ] `budget_manager.py` - Adapt for new budget system
- [ ] `event_log.py` - Keep, update for new tools
- [ ] `subagent_runtime.py` - Major refactor for new subagent architecture
- [ ] `telemetry_manager.py` - Keep, update event types

### Tests to Rewrite (`tests/`)
- [ ] `test_document_snapshot.py` → `test_read_document.py`
- [ ] `test_document_apply_patch.py` → `test_replace_lines.py`
- [ ] `test_document_chunk_tool.py` → Remove (automatic chunking)
- [ ] `test_document_insert.py` → `test_insert_lines.py`
- [ ] `test_document_replace_all.py` → `test_write_document.py`
- [ ] `test_document_outline_tool.py` → `test_get_outline.py`
- [ ] `test_character_map_tool.py` → Remove
- [ ] `test_character_edit_planner_tool.py` → Remove
- [ ] `test_document_plot_state_tool.py` → Remove
- [ ] All other tool tests need review/rewrite

---

## Workstream 1: Core Infrastructure

### WS1.1: Version Token System
**Files**: New `src/tinkerbell/ai/tools/version.py`, modify `src/tinkerbell/services/bridge.py`

- [ ] **WS1.1.1**: Design `VersionManager` class
  - Per-tab version tracking (simple incrementing integer)
  - Version validation on write operations
  - Version bump on successful edits

- [ ] **WS1.1.2**: Implement version storage in `DocumentBridge`
  - Add `_version: int` field to bridge/tab state
  - Increment on any document mutation
  - Reset to `1` on reload from disk

- [ ] **WS1.1.3**: Create `VersionMismatchError` exception
  - Include `your_version`, `current_version`, `suggestion`
  - JSON-serializable for tool responses

- [ ] **WS1.1.4**: Add version to all document state responses
  - Ensure `read_document`, `search_document`, `get_outline` all return version

- [ ] **WS1.1.5**: Unit tests for version system
  - Test increment behavior
  - Test mismatch detection
  - Test reset on reload

### WS1.2: Tool Base Classes
**Files**: New `src/tinkerbell/ai/tools/base.py`

- [ ] **WS1.2.1**: Create `BaseTool` abstract class
  - Standardized `execute()` signature
  - Built-in error formatting
  - Telemetry hooks

- [ ] **WS1.2.2**: Create `ReadOnlyTool` base class
  - For tools that don't require version token
  - Standardized response format with version

- [ ] **WS1.2.3**: Create `WriteTool` base class
  - Requires version token validation
  - Automatic version bump on success
  - Dry-run support built-in

- [ ] **WS1.2.4**: Create `SubagentTool` base class
  - Spawns subagents for execution
  - Progress tracking
  - Result aggregation

### WS1.3: Error Response System
**Files**: New `src/tinkerbell/ai/tools/errors.py`

- [ ] **WS1.3.1**: Define error code constants
  - `version_mismatch`, `invalid_tab_id`, `invalid_line_range`, etc.

- [ ] **WS1.3.2**: Create `ToolError` base exception
  - `error_code`, `message`, `details` fields
  - `to_dict()` for JSON serialization

- [ ] **WS1.3.3**: Create specific error subclasses
  - `VersionMismatchError`
  - `InvalidTabIdError`
  - `InvalidLineRangeError`
  - `UnsupportedFileTypeError`
  - `ContentRequiredError`

- [ ] **WS1.3.4**: Unit tests for error serialization

---

## Workstream 2: Navigation & Reading Tools

### WS2.1: `list_tabs` Tool
**Files**: Modify `src/tinkerbell/ai/tools/list_tabs.py`

- [ ] **WS2.1.1**: Update response format
  - Add `version`, `size_chars`, `line_count` to each tab
  - Add `is_active` flag
  - Add `file_type` detection

- [ ] **WS2.1.2**: Add file type detection
  - Based on extension: `.md`, `.txt`, `.json`, `.yaml`
  - Flag `binary` or `unknown` for unsupported types

- [ ] **WS2.1.3**: Unit tests for list_tabs

### WS2.2: `read_document` Tool
**Files**: New `src/tinkerbell/ai/tools/read_document.py`

- [ ] **WS2.2.1**: Implement basic line-range reading
  - 0-indexed, inclusive ranges
  - Default to active tab if `tab_id` omitted
  - Return content with line metadata

- [ ] **WS2.2.2**: Implement automatic pagination
  - Default ~6000 token window
  - `has_more` flag
  - `continuation_hint` with next `start_line`

- [ ] **WS2.2.3**: Implement token estimation
  - `tokens.returned` (actual)
  - `tokens.total_estimate` (whole document)

- [ ] **WS2.2.4**: Handle empty documents
  - Return `content: ""`, `lines.total: 0`
  - Version still valid for editing

- [ ] **WS2.2.5**: Handle unsupported file types
  - Return `unsupported_file_type` error for binary

- [ ] **WS2.2.6**: Unit tests for read_document
  - Normal reading
  - Pagination
  - Empty documents
  - Error cases

### WS2.3: `search_document` Tool
**Files**: New `src/tinkerbell/ai/tools/search_document.py`

- [ ] **WS2.3.1**: Implement exact text search
  - Literal string matching
  - Case sensitivity option
  - Return line numbers with context

- [ ] **WS2.3.2**: Implement regex search
  - Pattern validation
  - Match highlighting
  - Multiple results

- [ ] **WS2.3.3**: Integrate semantic search
  - Use existing `embeddings.py`
  - Just-in-time indexing if not ready
  - `embedding_status` field in response

- [ ] **WS2.3.4**: Result formatting
  - `line`, `score`, `preview`
  - Context window with `start_line`, `end_line`

- [ ] **WS2.3.5**: Handle unavailable embeddings
  - Graceful fallback to exact search
  - Clear error message with suggestion

- [ ] **WS2.3.6**: Unit tests for search_document
  - Each match type
  - Embedding availability scenarios
  - Context extraction

### WS2.4: `get_outline` Tool
**Files**: New `src/tinkerbell/ai/tools/get_outline.py`

- [ ] **WS2.4.1**: Implement Markdown heading detection
  - `#`, `##`, `###` markers
  - Proper nesting hierarchy

- [ ] **WS2.4.2**: Implement JSON/YAML structure detection
  - Top-level keys
  - Nested structure representation

- [ ] **WS2.4.3**: Implement plain text heuristics
  - Chapter markers (`Chapter 1`, `CHAPTER ONE`, etc.)
  - Visual patterns (ALL CAPS, separators)
  - Paragraph break fallback

- [ ] **WS2.4.4**: Confidence scoring
  - `detection_confidence`: high/medium/low
  - `detection_method` field
  - Suggestion for unstructured text

- [ ] **WS2.4.5**: Response formatting
  - Hierarchical `outline` array
  - `line_start`, `line_end` (null for last section)
  - `children` for nested sections

- [ ] **WS2.4.6**: Unit tests for get_outline
  - Each file type
  - Heuristic detection cases
  - Edge cases (empty, unstructured)

---

## Workstream 3: Writing Tools

### WS3.1: `create_document` Tool
**Files**: New `src/tinkerbell/ai/tools/create_document.py`

- [ ] **WS3.1.1**: Implement tab creation
  - Title/filename parameter
  - Optional initial content
  - File type hint

- [ ] **WS3.1.2**: Integrate with workspace
  - Create new tab via `TabbedEditorWidget`
  - Return `tab_id` and initial `version`

- [ ] **WS3.1.3**: Handle title conflicts
  - Return `title_exists` error

- [ ] **WS3.1.4**: Unit tests for create_document

### WS3.2: `insert_lines` Tool
**Files**: New `src/tinkerbell/ai/tools/insert_lines.py`

- [ ] **WS3.2.1**: Implement insertion logic
  - `after_line` parameter (-1 for start)
  - Never deletes existing content
  - Multi-line content support

- [ ] **WS3.2.2**: Version token validation
  - Require valid version
  - Return new version on success

- [ ] **WS3.2.3**: Dry-run support
  - Validate without applying
  - Version not consumed

- [ ] **WS3.2.4**: Optional `match_text` drift recovery
  - Find anchor text
  - Adjust insertion point if lines shifted

- [ ] **WS3.2.5**: Response formatting
  - `inserted_at.after_line`
  - `inserted_at.lines_added`
  - `inserted_at.new_lines.start/end`

- [ ] **WS3.2.6**: Unit tests for insert_lines
  - Normal insertion
  - Start/end insertion
  - Dry-run
  - Drift recovery

### WS3.3: `replace_lines` Tool
**Files**: New `src/tinkerbell/ai/tools/replace_lines.py`

- [ ] **WS3.3.1**: Implement replacement logic
  - `start_line`, `end_line` (inclusive)
  - `content` parameter (empty for delete)

- [ ] **WS3.3.2**: Version token validation
  - Same pattern as insert_lines

- [ ] **WS3.3.3**: Dry-run support

- [ ] **WS3.3.4**: Optional `match_text` drift recovery

- [ ] **WS3.3.5**: Response formatting
  - `lines_affected.removed`, `.added`, `.net_change`

- [ ] **WS3.3.6**: Unit tests for replace_lines

### WS3.4: `delete_lines` Tool
**Files**: New `src/tinkerbell/ai/tools/delete_lines.py`

- [ ] **WS3.4.1**: Implement as wrapper around replace_lines
  - `content=""` internally
  - Clear intent in API

- [ ] **WS3.4.2**: Response formatting
  - `lines_deleted` count

- [ ] **WS3.4.3**: Unit tests for delete_lines

### WS3.5: `write_document` Tool
**Files**: New `src/tinkerbell/ai/tools/write_document.py`

- [ ] **WS3.5.1**: Implement full document replacement
  - Version token required
  - Complete content parameter

- [ ] **WS3.5.2**: Response formatting
  - `lines_affected.previous`, `.current`

- [ ] **WS3.5.3**: Unit tests for write_document

### WS3.6: `find_and_replace` Tool
**Files**: New `src/tinkerbell/ai/tools/find_and_replace.py`

- [ ] **WS3.6.1**: Implement search logic
  - Literal and regex modes
  - Case sensitivity, whole word options
  - Scope limiting by line range

- [ ] **WS3.6.2**: Implement replacement logic
  - Batch replacement
  - `max_replacements` cap
  - Version token required

- [ ] **WS3.6.3**: Preview mode
  - `preview=true` doesn't apply changes
  - Version not consumed in preview
  - Truncate preview to 20 matches

- [ ] **WS3.6.4**: Response formatting
  - `matches_found`, `replacements_made`
  - `preview` array with before/after

- [ ] **WS3.6.5**: Unit tests for find_and_replace

---

## Workstream 4: Editor Lock & Diff Review

### WS4.1: Global Editor Lock
**Files**: Modify `src/tinkerbell/ui/ai_turn_coordinator.py`, `src/tinkerbell/editor/tabbed_editor.py`

- [ ] **WS4.1.1**: Implement lock acquisition
  - Lock all tabs on AI turn start
  - Visual indicator in status bar

- [ ] **WS4.1.2**: Implement lock release
  - Release on turn completion
  - Release on cancel
  - Release on error/timeout

- [ ] **WS4.1.3**: Cancel functionality
  - User-triggered cancel button
  - Immediate lock release

- [ ] **WS4.1.4**: Unit tests for editor lock

### WS4.2: Atomic Operations & Rollback
**Files**: New `src/tinkerbell/ai/orchestration/transaction.py`

- [ ] **WS4.2.1**: Implement staged changes
  - Buffer edits in memory
  - Don't apply to document immediately

- [ ] **WS4.2.2**: Implement commit
  - Apply all staged changes atomically
  - Single version bump

- [ ] **WS4.2.3**: Implement rollback
  - Discard staged changes on failure
  - Document remains unchanged

- [ ] **WS4.2.4**: Multi-document transaction support
  - Track changes across tabs
  - All-or-nothing commit

- [ ] **WS4.2.5**: Unit tests for transactions

### WS4.3: Diff Review UI
**Files**: Modify `src/tinkerbell/ui/review_overlay_manager.py`, `src/tinkerbell/editor/editor_widget.py`

- [ ] **WS4.3.1**: Single-document diff view
  - Red/green highlighting
  - Accept/Reject buttons

- [ ] **WS4.3.2**: Multi-document diff view
  - Tabbed interface for multiple files
  - Accept All / Reject All (no partial)

- [ ] **WS4.3.3**: Accept flow
  - Commit staged changes
  - Increment versions
  - Create checkpoint

- [ ] **WS4.3.4**: Reject flow
  - Discard staged changes
  - Restore pre-turn state

- [ ] **WS4.3.5**: Unit tests for diff review

### WS4.4: Turn Checkpoints
**Files**: New `src/tinkerbell/ai/memory/checkpoints.py`

- [ ] **WS4.4.1**: Checkpoint creation
  - Capture document state
  - Store turn number, timestamp, action summary

- [ ] **WS4.4.2**: Checkpoint storage
  - Per-document checkpoint list
  - Session-scoped (cleared on close)

- [ ] **WS4.4.3**: Checkpoint restoration
  - Restore document to checkpoint state
  - Create new checkpoint (non-destructive)

- [ ] **WS4.4.4**: Turn history UI
  - List of checkpoints
  - Diff from current state
  - Restore button

- [ ] **WS4.4.5**: Unit tests for checkpoints

---

## Workstream 5: Subagent Architecture

### WS5.1: Subagent Infrastructure
**Files**: Major refactor of `src/tinkerbell/ai/orchestration/subagent_runtime.py`

- [ ] **WS5.1.1**: Define subagent types
  - Chunk Analyzer
  - Consistency Checker
  - Transformer

- [ ] **WS5.1.2**: Implement chunk coordination
  - Automatic document chunking (~4000 tokens)
  - Parallel chunk processing
  - Result aggregation

- [ ] **WS5.1.3**: Implement progress tracking
  - Active subagent counter
  - Status bar integration

- [ ] **WS5.1.4**: Error handling
  - Partial failure rollback
  - Clear error messages

- [ ] **WS5.1.5**: Unit tests for subagent runtime

### WS5.2: `analyze_document` Tool
**Files**: New `src/tinkerbell/ai/tools/analyze_document.py`

- [ ] **WS5.2.1**: Implement task routing
  - `characters`, `plot`, `style`, `summary`, `custom`
  - Task-specific subagent prompts

- [ ] **WS5.2.2**: Implement chunking strategy
  - Auto-chunk documents > 20k chars
  - Parallel chunk analysis

- [ ] **WS5.2.3**: Result aggregation
  - Synthesize chunk results
  - Write to output tab

- [ ] **WS5.2.4**: Output tab management
  - Auto-create if not specified
  - Overwrite existing with diff review

- [ ] **WS5.2.5**: Unit tests for analyze_document

### WS5.3: `transform_document` Tool
**Files**: New `src/tinkerbell/ai/tools/transform_document.py`

- [ ] **WS5.3.1**: Implement transformation types
  - `character_rename`
  - `setting_change`
  - `style_rewrite`
  - `custom`

- [ ] **WS5.3.2**: Character rename implementation
  - Find all mentions
  - Update pronouns if requested
  - Maintain consistency

- [ ] **WS5.3.3**: Setting change implementation
  - Find setting references
  - Adapt cultural details

- [ ] **WS5.3.4**: Output mode handling
  - `new_tab` (default)
  - `in_place` (requires version, uses diff review)

- [ ] **WS5.3.5**: Consistency checking
  - Validate no continuity breaks
  - Flag ambiguous cases

- [ ] **WS5.3.6**: Unit tests for transform_document

---

## Workstream 6: Tool Registry & Integration

### WS6.1: New Tool Registry
**Files**: Complete rewrite of `src/tinkerbell/ai/tools/registry.py`

- [ ] **WS6.1.1**: Define new tool registration
  - Register only new tools
  - Clean parameter schemas

- [ ] **WS6.1.2**: Remove all legacy tool registration
  - Delete all old tool imports
  - Remove feature flags for old tools

- [ ] **WS6.1.3**: Implement tool schema generation
  - JSON Schema for each tool
  - Consistent parameter naming

- [ ] **WS6.1.4**: Unit tests for registry

### WS6.2: Controller Integration
**Files**: Major refactor of `src/tinkerbell/ai/orchestration/controller.py`

- [ ] **WS6.2.1**: Update tool dispatch
  - Route to new tool implementations
  - Remove legacy tool handling

- [ ] **WS6.2.2**: Integrate transaction system
  - Wrap tool calls in transactions
  - Handle atomic commit/rollback

- [ ] **WS6.2.3**: Update telemetry
  - New tool names in events
  - New metrics for subagents

- [ ] **WS6.2.4**: Integration tests

### WS6.3: Prompt Updates
**Files**: Modify `src/tinkerbell/ai/prompts.py`

- [ ] **WS6.3.1**: Update system prompt
  - Document new tool names
  - Remove references to old tools

- [ ] **WS6.3.2**: Update tool usage instructions
  - New workflow examples
  - Version token handling

- [ ] **WS6.3.3**: Review all prompt templates

---

## Workstream 7: Cleanup & Testing

### WS7.1: Delete Legacy Code
**Files**: Multiple deletions

- [ ] **WS7.1.1**: Delete old tool files
  - All files listed in inventory above
  - Remove from `__init__.py`

- [ ] **WS7.1.2**: Delete old memory components
  - `plot_memory.py`, `plot_state.py` if fully replaced
  - `character_map.py` memory component

- [ ] **WS7.1.3**: Clean up imports
  - Remove unused imports across codebase
  - Fix any broken references

- [ ] **WS7.1.4**: Delete old tests
  - All tests for removed tools

### WS7.2: New Test Suite
**Files**: New test files in `tests/`

- [ ] **WS7.2.1**: Unit tests for each new tool
  - `test_list_tabs.py`
  - `test_read_document.py`
  - `test_search_document.py`
  - `test_get_outline.py`
  - `test_create_document.py`
  - `test_insert_lines.py`
  - `test_replace_lines.py`
  - `test_delete_lines.py`
  - `test_write_document.py`
  - `test_find_and_replace.py`
  - `test_analyze_document.py`
  - `test_transform_document.py`

- [ ] **WS7.2.2**: Integration tests
  - Multi-tool workflows
  - Version token flows
  - Transaction commit/rollback

- [ ] **WS7.2.3**: End-to-end tests
  - Sample workflow: "Write me a story"
  - Sample workflow: "Change John to James"
  - Sample workflow: "Find and expand scene"

### WS7.3: Documentation
**Files**: Update `README.md`, `docs/`

- [ ] **WS7.3.1**: Update API documentation
  - New tool reference
  - Parameter schemas

- [ ] **WS7.3.2**: Update user guide
  - New workflow examples
  - UI changes (diff review)

- [ ] **WS7.3.3**: Update developer guide
  - Architecture overview
  - Adding new tools

---

## Implementation Order

### Phase 1: Core Foundation (WS1, WS2.1, WS2.2)
1. Version token system (WS1.1)
2. Tool base classes (WS1.2)
3. Error response system (WS1.3)
4. `list_tabs` update (WS2.1)
5. `read_document` implementation (WS2.2)

### Phase 2: Navigation & Search (WS2.3, WS2.4)
1. `search_document` with exact/regex (WS2.3.1-2)
2. `search_document` semantic integration (WS2.3.3-5)
3. `get_outline` implementation (WS2.4)

### Phase 3: Writing Tools (WS3)
1. `create_document` (WS3.1)
2. `insert_lines` (WS3.2)
3. `replace_lines` (WS3.3)
4. `delete_lines` (WS3.4)
5. `write_document` (WS3.5)
6. `find_and_replace` (WS3.6)

### Phase 4: Editor Integration (WS4)
1. Global editor lock (WS4.1)
2. Atomic operations & rollback (WS4.2)
3. Diff review UI (WS4.3)
4. Turn checkpoints (WS4.4)

### Phase 5: Subagents (WS5)
1. Subagent infrastructure (WS5.1)
2. `analyze_document` (WS5.2)
3. `transform_document` (WS5.3)

### Phase 6: Integration & Cleanup (WS6, WS7)
1. New tool registry (WS6.1)
2. Controller integration (WS6.2)
3. Prompt updates (WS6.3)
4. Delete legacy code (WS7.1)
5. New test suite (WS7.2)
6. Documentation (WS7.3)

---

## Risk Mitigation

### Testing Strategy
- Write tests for new tools before deleting old ones
- Run full test suite after each phase
- Manual testing of key workflows

### Rollback Plan
- Git tags at each phase completion
- Feature branch until fully validated
- No incremental deployment—big-bang switch

### Dependencies
- Embedding service must remain functional for semantic search
- UI components (diff overlay) must be adapted, not replaced
- Telemetry pipeline must be updated for new events

---

## File Summary

### New Files to Create
```
src/tinkerbell/ai/tools/
├── base.py                    # Tool base classes
├── errors.py                  # Error types
├── version.py                 # Version management
├── read_document.py           # WS2.2
├── search_document.py         # WS2.3
├── get_outline.py             # WS2.4
├── create_document.py         # WS3.1
├── insert_lines.py            # WS3.2
├── replace_lines.py           # WS3.3
├── delete_lines.py            # WS3.4
├── write_document.py          # WS3.5
├── find_and_replace.py        # WS3.6
├── analyze_document.py        # WS5.2
└── transform_document.py      # WS5.3

src/tinkerbell/ai/orchestration/
└── transaction.py             # WS4.2

src/tinkerbell/ai/memory/
└── checkpoints.py             # WS4.4
```

### Files to Delete
```
src/tinkerbell/ai/tools/
├── document_snapshot.py
├── document_apply_patch.py
├── document_chunk.py
├── document_edit.py
├── document_find_text.py
├── document_insert.py
├── document_replace_all.py
├── document_outline.py
├── document_plot_state.py
├── plot_state_update.py
├── character_edit_planner.py
├── character_map.py
├── diff_builder.py
├── search_replace.py
├── selection_range.py
└── tool_usage_advisor.py
```

### Files to Heavily Modify
```
src/tinkerbell/ai/tools/registry.py          # Complete rewrite
src/tinkerbell/ai/tools/__init__.py          # Update exports
src/tinkerbell/ai/orchestration/controller.py # Major refactor
src/tinkerbell/ai/orchestration/subagent_runtime.py # Major refactor
src/tinkerbell/ai/prompts.py                 # Update for new tools
src/tinkerbell/ui/ai_turn_coordinator.py     # Add lock mechanism
src/tinkerbell/ui/review_overlay_manager.py  # Enhance diff review
src/tinkerbell/services/bridge.py            # Version system integration
```
