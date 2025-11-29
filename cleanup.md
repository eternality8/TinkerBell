# Code Cleanup Plan for TinkerBell

This document outlines identified code quality issues and a phased plan for cleanup and refactoring.

---

## Guiding Principles

> **No Deprecation. No Backwards Compatibility. Just Clean Code.**

This cleanup effort is focused on **removing unnecessary code**, not adding compatibility layers or deprecation warnings. When we identify legacy or redundant code:

1. **Delete it** - Don't wrap it, don't deprecate it, remove it entirely
2. **Update all call sites** - Fix every reference, don't leave stubs
3. **Remove related tests** - Tests for deleted code should be deleted too
4. **Update documentation** - Remove all references to deleted functionality

We are not maintaining backwards compatibility with any previous versions. This is a single-user application, not a library with external consumers.

---

## Completed Work

### Legacy Feature Flag Removal ✅

**Date:** Completed

The following legacy feature flags have been removed from the codebase as they are now always enabled:

**Removed Settings Fields:**
- `phase3_outline_tools` - Phase 3 outline/retrieval tools (now always enabled)
- `enable_outline_generation` - Background outline generation (now always enabled)
- `enable_subagents` - Phase 4 subagent sandbox (now always enabled)
- `enable_plot_scaffolding` - Plot/entity scaffolding (now always enabled)
- `safe_ai_edits` - Post-edit inspection guardrails (now always enabled)
- `safe_ai_duplicate_threshold` - Duplicate threshold setting (removed)
- `safe_ai_token_drift` - Token drift tolerance setting (removed)

**Files Modified:**
- `src/tinkerbell/services/settings.py` - Removed fields from `Settings` dataclass and environment variable overrides
- `src/tinkerbell/app.py` - Removed CLI arguments (`--enable-phase3-outline-tools`, `--enable-subagents`, etc.)
- `src/tinkerbell/widgets/dialogs.py` - Removed UI checkboxes from Settings dialog
- `src/tinkerbell/ui/main_window.py` - Removed feature flag handling methods
- `src/tinkerbell/ui/settings_runtime.py` - Simplified `apply_runtime_settings()` method
- `src/tinkerbell/ui/telemetry_controller.py` - Removed `initial_subagent_enabled` parameter
- `src/tinkerbell/ui/embedding_controller.py` - Removed `phase3_outline_enabled` parameter

**Tests Updated:**
- `tests/test_main_window.py` - Removed tests for feature flag behavior
- `tests/test_settings.py` - Removed tests for legacy environment variable overrides
- `tests/test_settings_runtime.py` - Simplified handler tests
- `tests/test_widgets_dialogs.py` - Removed assertions for removed UI elements
- `tests/test_telemetry_controller.py` - Removed `initial_subagent_enabled` parameter
- `tests/test_embedding_controller.py` - Removed `phase3_outline_enabled` parameter

**Rationale:** These features were experimental flags during development. Now that they are stable, the toggles add unnecessary complexity. All functionality is always available.

---

## Executive Summary

After reviewing the codebase, several categories of issues have been identified:
1. **Massive files** that need splitting
2. **Duplicate code** across modules
3. **Legacy typing patterns** (pre-Python 3.9 style)
4. **Parallel prompt/tool systems** that need consolidation
5. **Overly broad exception handling**
6. **Incomplete async transitions**
7. **Dead code and stale comments**

---

## Priority 1: Critical Refactoring

### 1.1 Split `controller.py` (Partial ✅)

**Location:** `src/tinkerbell/ai/orchestration/controller.py`

**Phase 1 Complete:** The controller.py file has been split from 4,636 lines to ~4,250 lines (~395 lines extracted).

**Extracted Files (Phase 1):**
- `model_types.py` - `ToolCallRequest`, `ModelTurnResult`, `MessagePlan`
- `chunk_flow.py` - `ChunkContext`, `ChunkFlowTracker`
- `turn_tracking.py` - `SnapshotRefreshTracker`, `PlotLoopTracker`
- `runtime_config.py` - `ChunkingRuntimeConfig`, `AnalysisRuntimeConfig`
- `subagent_state.py` - `SubagentDocumentState`

**Phase 2 Partial Complete:** Additional extractions brought the file down to ~3,925 lines (~720 total lines extracted).

**Extracted Files (Phase 2):**
- ✅ `tool_call_parser.py` - Tool call parsing constants and functions (`TOOL_MARKER_TRANSLATION`, regex patterns, `parse_embedded_tool_calls`, `normalize_tool_marker_text`, `try_parse_json_block`)
- ✅ `controller_utils.py` - Static utility functions (`normalize_iterations`, `normalize_scope_origin`, `normalize_context_tokens`, `normalize_response_reserve`, `normalize_temperature`, `coerce_optional_int`, `coerce_optional_float`, `coerce_optional_str`, `sanitize_suggestions`)
- ✅ `scope_helpers.py` - Scope/range handling (~250 lines: `scope_summary_from_arguments`, `scope_summary_from_metadata`, `scope_summary_from_ranges`, `scope_summary_from_target_range`, `scope_summary_from_chunk_arguments`, `scope_fields_from_summary`, `range_bounds_from_entry`, `range_bounds_from_mapping`, `extract_chunk_id`, `parse_chunk_bounds`)
- ✅ `guardrail_hints.py` - Guardrail hint generation (~120 lines: `format_guardrail_hint`, `outline_guardrail_hints`, `retrieval_guardrail_hints`)

**Phase 2 Remaining:**

| Extraction Target | Est. Lines | Risk | Priority | Status |
|-------------------|------------|------|----------|--------|
| `tool_call_parser.py` | ~150 | Low | 1 | ✅ Done |
| `controller_utils.py` | ~100 | Low | 2 | ✅ Done |
| `scope_helpers.py` | ~300 | Low | 3 | ✅ Done |
| `guardrail_hints.py` | ~200 | Low | 4 | ✅ Done |
| `suggestions.py` | ~100 | Low | 5 | ⏳ Pending |
| `tool_execution.py` | ~350 | Medium | 6 | ⏳ Pending |
| `version_retry.py` | ~150 | Medium | 7 | ⏳ Pending |
| `metrics_recorder.py` | ~200 | Low | 8 | ⏳ Pending |
| `analysis_pipeline.py` | ~200 | Medium | 9 | ⏳ Pending |
| `subagent_pipeline.py` | ~400 | Medium | 10 | ⏳ Pending |

**Action Items:**
- [x] Extract `tool_call_parser.py` (standalone, no controller state dependency)
- [x] Extract `controller_utils.py` (pure functions)
- [x] Extract `scope_helpers.py` (mostly stateless helpers)
- [x] Extract `guardrail_hints.py` (self-contained logic)
- [ ] Extract `suggestions.py` (isolated feature)
- [ ] Extract `tool_execution.py` (core tool handling)
- [ ] Extract `version_retry.py` (retry logic)
- [ ] Extract `metrics_recorder.py` (telemetry helpers)
- [ ] Extract `analysis_pipeline.py` (preflight analysis)
- [ ] Extract `subagent_pipeline.py` (subagent orchestration)

---

### 1.2 Split `main_window.py` (Partial ✅)

**Location:** `src/tinkerbell/ui/main_window.py`

**Current Status:** 2,519 → 2,233 lines (~11% reduction, 286 lines extracted)

**Extracted to `main_window_helpers.py` (220 lines):**
- Protocol implementations: `WriteToolDispatchListener`, `EditorTabWrapper`, `WorkspaceTabProvider`
- Pure utility functions:
  - `condense_whitespace` - Collapse whitespace to single spaces
  - `line_column_from_offset` - Convert offset to line/column
  - `coerce_stream_text` - Extract text from streaming payloads
  - `infer_language` - Infer language from file extension
  - `directive_parameters_schema` - Build directive schema copy
  - `serialize_chat_history` - Serialize chat history to dicts
  - `history_signature` - Compute SHA-256 of chat history

**Also Cleaned:**
- Removed duplicate `file_io` import
- Removed `hashlib` import (no longer needed)
- Removed 4 no-op methods (`_register_phase3_ai_tools`, etc.)
- Removed unused `DIRECTIVE_SCHEMA` import

**Remaining Work:** The remaining ~2,233 lines are tightly coupled to instance state. Further extraction requires creating coordinator classes that receive dependencies via constructor injection, similar to `AITurnCoordinator` and `AIReviewController`.

Contains UI, business logic, and coordination all mixed together.

**Current Responsibilities:**
- Window shell management
- Document session coordination
- AI controller integration
- Tool registration and wiring
- Settings management
- Embedding controller coordination
- Review overlay management
- Status bar updates

**Recommended Split:**
```
ui/
├── main_window.py           # Core window, ~500 lines
├── main_window_ai.py        # AI-related methods (mixin or extracted)
├── main_window_documents.py # Document handling (mixin or extracted)
├── main_window_settings.py  # Settings handling (mixin or extracted)
└── coordinators/
    ├── ai_coordinator.py
    └── document_coordinator.py
```

**Alternative:** Use composition over inheritance:
```python
class MainWindow:
    def __init__(self):
        self._ai_coord = AICoordinator(self)
        self._doc_coord = DocumentCoordinator(self)
```

---

### 1.3 Split `dialogs.py` ✅

**Location:** `src/tinkerbell/widgets/dialogs.py` → `src/tinkerbell/widgets/dialogs/` (package)

**Status:** Complete. All dialog classes extracted, legacy file deleted.

**Extracted Files:**
```
widgets/dialogs/
├── __init__.py              # Re-exports + open_file_dialog, save_file_dialog (133 lines)
├── common.py                # DEFAULT_FILE_FILTER, constants, utilities (72 lines)
├── sample_document.py       # SampleDocument, discover_sample_documents (63 lines)
├── document_load.py         # DocumentLoadDialog (252 lines)
├── document_export.py       # DocumentExportDialog (159 lines)
├── validation_errors.py     # ValidationErrorsDialog, show_validation_errors (58 lines)
├── settings_dialog.py       # SettingsDialog, SettingsDialogResult, test functions (1,403 lines)
```

**Summary:**
- Original `dialogs.py` (1,948 lines) → 7 focused modules (~2,140 total lines with better organization)
- `SettingsDialog` (~1,100 lines of the original) extracted to `settings_dialog.py`
- Helper functions `show_settings_dialog`, `test_ai_api_settings`, `test_embedding_settings` moved to `settings_dialog.py`
- `ValidationResult`, `SettingsValidator`, `SettingsTester` types defined in `settings_dialog.py`
- Legacy file `_dialogs_legacy.py` deleted
- All 1,102 tests pass

**Import Path:** `from tinkerbell.widgets.dialogs import ...` works for all dialog classes and utilities.

---

### 1.4 Split `bridge.py` ✅

**Location:** `src/tinkerbell/services/bridge.py`

**Status:** Complete. Reduced from 1,856 lines to ~1,417 lines (~24% reduction, 439 lines extracted).

**Extracted Files:**
```
services/
├── bridge.py                # DocumentBridge core (1,417 lines)
├── bridge_types.py          # Types and dataclasses (137 lines)
│   - PatchRangePayload, EditContext, PatchMetrics
│   - SafeEditSettings, DocumentVersionMismatchError
│   - QueuedEdit, EditorAdapter protocol, Executor type
├── bridge_versioning.py     # Version token utilities (122 lines)
│   - format_version_token, is_version_current
│   - hash_text, extract_context_version, extract_content_hash
│   - parse_chunk_bounds, compute_line_start_offsets
│   - clamp_range, summarize_diff
├── bridge_inspection.py     # Post-edit inspection helpers (111 lines)
│   - auto_revert_remediation, format_auto_revert_message
│   - build_failure_metadata, attach_scope_metadata
└── bridge_queue.py          # Edit queue management (330 lines)
    - normalize_directive, normalize_patch_ranges
    - scope helpers: normalize_scope_origin, coerce_scope_length
    - summarize_patch_scopes, range_hint_from_payload
    - validate_scope_requirements, refresh_scope_span
```

**Summary:**
- Original `bridge.py` (1,856 lines) → 5 focused modules (~2,117 total lines with better organization)
- All static utility methods moved to appropriate helper modules
- `DocumentBridge` class retained in `bridge.py` with core orchestration logic
- Types re-exported from `services/__init__.py` for backwards compatibility
- All 1,102 tests pass

**Import Path:** Original imports still work: `from tinkerbell.services.bridge import DocumentBridge, DocumentVersionMismatchError`

---

## Priority 2: Duplicate Code Elimination

### 2.1 Duplicate `estimate_tokens` Functions

**Locations:**
- `src/tinkerbell/ai/tools/read_document.py:37` - Uses `len(text) / 4`
- `src/tinkerbell/ai/tools/subagent.py:267` - Uses `len(text.encode("utf-8")) // 4`
- `src/tinkerbell/ai/services/summarizer.py:19` - Uses `math.ceil(len(text.encode("utf-8", errors="ignore")) / 4)` (private `_estimate_tokens`)

These should be consolidated into a single utility function.

**Action:**
- [ ] Create `src/tinkerbell/ai/utils/tokens.py`
- [ ] Move `estimate_tokens` there
- [ ] Update all imports

---

### 2.2 Delete Legacy Prompt System

**Issue:** Two parallel prompt systems exist:
- `src/tinkerbell/ai/prompts.py` - Legacy system (DELETE)
- `src/tinkerbell/ai/prompts_v2.py` - Current system (KEEP)

**Comparison Analysis:**

| Aspect | `prompts.py` (Legacy) | `prompts_v2.py` (Current) |
|--------|----------------------|--------------------------|
| Tool Listing | Incomplete, missing 6+ tools | Complete, all 12 WS1-6 tools |
| Tab ID Guidance | None | Explicit section on opaque tab IDs |
| Workflow Examples | Generic, no code | Detailed code snippets |
| Error Handling | Brief mention | Complete error table with recovery |
| Tool-Specific Docs | None | Individual exportable functions |
| Line Numbering | Not mentioned | Explicitly states 0-based |
| Version Token | Uses outdated `snapshot_token` | Correctly uses `version_token` |

**`prompts.py` Problems:**
1. Uses `snapshot_token` instead of `version_token` (line 32)
2. Missing tools: `list_tabs`, `create_document`, `delete_lines`, `find_and_replace`, `analyze_document`, `transform_document`
3. No tab ID guidance (AI often confuses document names with tab IDs)
4. References `document_version` instead of `version_token`

**`prompts_v2.py` Minor Issues to Fix:**
1. Missing `aliases` parameter mention in `transform_document_instructions()`
2. Missing `summary` analysis type in `analyze_document` section

**Files importing from legacy `prompts.py`:**
- `tests/test_prompts.py`
- `tests/test_agent.py`

**Action:**
- [ ] Fix minor issues in `prompts_v2.py` (add `aliases`, `summary` type)
- [ ] Update `tests/test_prompts.py` to import from `prompts_v2.py`
- [ ] Update `tests/test_agent.py` to import from `prompts_v2.py`
- [ ] Delete `prompts.py` entirely
- [ ] Rename `prompts_v2.py` to `prompts.py`
- [ ] Update import in `tests/test_ws6_registry.py` after rename

---

### 2.3 Duplicate `ToolRegistration` Class ✅

**Date:** Completed

**Resolution:** The two classes serve different purposes and should remain separate. To avoid confusion, they have been renamed:

- `src/tinkerbell/ai/tools/tool_registry.py` - `ToolRegistration` (tool lifecycle management, categories, feature flags)
- `src/tinkerbell/ai/orchestration/controller.py` - **Renamed to `OpenAIToolSpec`** (runtime API formatting for OpenAI function calling)

A backwards compatibility alias `ToolRegistration = OpenAIToolSpec` is maintained in `controller.py` for existing code.

---

## Priority 3: Modernize Type Annotations

### 3.1 Replace Legacy `typing` Imports

Multiple files use old-style type hints that can be replaced with built-in generics (Python 3.9+).

**Examples Found:**
```python
# Old style (should be replaced)
from typing import Dict, List, Optional, Tuple

# New style
dict, list, tuple, X | None
```

**Files Requiring Updates:**
- `src/tinkerbell/ui/main_window.py` - Uses `Dict, Optional, Sequence`
- `src/tinkerbell/ai/orchestration/controller.py` - Uses `Dict, MutableMapping`
- `src/tinkerbell/ai/client.py` - Uses `Dict, List`
- `src/tinkerbell/services/bridge.py` - Uses `Optional, Sequence`
- `src/tinkerbell/chat/chat_panel.py` - Uses `List, Optional`
- `src/tinkerbell/chat/commands.py` - Uses `Dict`
- `src/tinkerbell/editor/document_model.py` - Uses `Dict, Optional`
- `src/tinkerbell/editor/patches.py` - Uses `List, Optional, Tuple`
- `src/tinkerbell/theme/models.py` - Uses `Dict, Tuple`
- And 10+ more files

**Action:**
- [ ] Add `from __future__ import annotations` to all files (if not present)
- [ ] Replace `Dict[K, V]` → `dict[K, V]`
- [ ] Replace `List[T]` → `list[T]`
- [ ] Replace `Tuple[...]` → `tuple[...]`
- [ ] Replace `Optional[T]` → `T | None`
- [ ] Replace `Sequence[T]` → Consider `Iterable` or `list` where appropriate

---

## Priority 4: Exception Handling Improvements

### 4.1 Replace Bare `except Exception:` Clauses

**Issue:** Many files have overly broad exception handlers that silently swallow errors. **100+ instances** found across the codebase.

**Files with excessive bare exceptions:**
- `src/tinkerbell/widgets/status_bar.py` - 30+ instances
- `src/tinkerbell/ui/main_window.py` - 20+ instances
- `src/tinkerbell/ui/widgets/document_status_window.py` - 15+ instances
- `src/tinkerbell/ui/embedding_controller.py` - 5+ instances
- `src/tinkerbell/widgets/dialogs.py` - 3+ instances

**Pattern to fix:**
```python
# Bad
except Exception:
    pass

# Better
except Exception:
    LOGGER.debug("Context for why this might fail", exc_info=True)

# Best (when possible)
except SpecificError as exc:
    LOGGER.debug("Specific context: %s", exc)
```

**Action:**
- [ ] Audit all `except Exception:` clauses
- [ ] Add logging with context to silent exception handlers
- [ ] Replace with specific exception types where possible

---

## Priority 5: Complete TODO Items

### 5.1 Incomplete Async Transitions

**Locations:**
- `src/tinkerbell/ai/tools/transform_document.py:673`
  ```python
  # TODO: When async execution is implemented, this should queue the task
  ```
- `src/tinkerbell/ai/tools/analyze_document.py:459`
  ```python
  # TODO: When async execution is implemented, this should queue the task
  ```

**Action:**
- [ ] Implement proper async execution path
- [ ] Or document why sync fallback is acceptable

---

## Priority 6: Dead Code Removal

### 6.1 Delete Legacy Tool System

**Location:** `src/tinkerbell/ai/tools/deprecation.py`

This entire module exists to maintain backwards compatibility with old tool names. Per our guiding principles, we delete it entirely instead of maintaining it.

**Legacy tools to delete (not deprecate):**
- `document_snapshot`
- `document_edit`
- `document_apply_patch`
- `document_chunk`
- `document_find_text`
- `document_outline`
- `document_insert`
- `document_replace_all`
- `search_replace`
- `selection_range`

**Action:**
- [ ] Delete `src/tinkerbell/ai/tools/deprecation.py`
- [ ] Delete `tests/test_ws7_deprecation.py`
- [ ] Remove all imports of deprecation module
- [ ] Search for and remove any string references to legacy tool names

---

### 6.2 No-op Methods in `main_window.py`

**Locations (lines 662-674):**
```python
def _register_phase3_ai_tools(self) -> None:
    # Phase3 tools have been deprecated; this is now a no-op
    pass

def _unregister_phase3_ai_tools(self) -> None:
    # Phase3 tools have been deprecated; this is now a no-op
    pass

def _register_plot_state_tool(self) -> None:
    # Plot state tools have been deprecated; this is now a no-op
    pass

def _unregister_plot_state_tool(self) -> None:
    # Plot state tools have been deprecated; this is now a no-op
    pass
```

**Action:**
- [ ] Remove these methods entirely
- [ ] Remove all call sites
- [ ] Delete any tests that exercise these no-ops

---

## Priority 7: Test Suite Cleanup

### 7.1 Remove Redundant Tests

**Goal:** Identify and remove tests that:
- Test deleted functionality
- Duplicate other tests with minor variations
- Test internal implementation details that should be free to change
- Have excessive mocking that makes them brittle and low-value

**Audit Checklist:**
- [ ] Review `tests/test_ws7_deprecation.py` - Tests for the deprecation layer we're removing
- [ ] Review tests for legacy tool names (`document_snapshot`, `document_edit`, etc.)
- [ ] Identify tests with 50+ lines of mock setup for trivial assertions
- [ ] Find duplicate test scenarios across different test files
- [ ] Remove tests for removed feature flags (already done partially)

### 7.2 Reduce `# type: ignore` Comments in Tests

**Issue:** 20+ `# type: ignore` comments in test files, indicating either:
- Tests need better type stubs
- Test fixtures need proper typing
- Mock objects need protocol compliance

**Affected test files:**
- `tests/test_ws2_tools.py`
- `tests/test_ws9_subagent_execution.py`
- `tests/test_workspace.py`
- `tests/test_turn_context.py`
- `tests/test_tool_base.py`

**Action:**
- [ ] Create properly typed test fixtures
- [ ] Add type stubs for mock objects
- [ ] Use `cast()` instead of `# type: ignore` where appropriate

### 7.3 Consolidate Test Helpers

**Issue:** Similar test fixtures and helpers are duplicated across test files.

**Action:**
- [ ] Audit `conftest.py` for underused fixtures
- [ ] Identify repeated `_StubController`, `_FakeSettings`, etc. patterns
- [ ] Consolidate into shared fixtures in `conftest.py`
- [ ] Remove duplicate helper classes from individual test files

---

## Priority 8: Documentation Rewrite

### 8.1 Complete Documentation Overhaul

**Problem:** The current documentation is a mix of:
- Outdated feature descriptions
- References to removed/renamed functionality
- Planning documents that are no longer relevant
- Implementation notes that describe code that no longer exists

**Current Documentation Files:**
```
ai_enhancements.md           # Outdated planning doc
ai_enhancements_implementation.md  # Stale implementation notes
ai_refactor.md               # Legacy refactor plan
ai_refactor_implementation.md  # Legacy implementation notes
testing.md                   # Outdated test documentation
testing_improvements.md      # Stale improvement notes
docs/ai_v2.md               # May contain legacy references
docs/ai_v2_release_notes.md  # Historical notes
docs/operations/*.md         # Need review for accuracy
```

**Action Plan:**

1. **Delete all outdated planning/implementation docs:**
   - [ ] Delete `ai_enhancements.md` and `ai_enhancements_implementation.md`
   - [ ] Delete `ai_refactor.md` and `ai_refactor_implementation.md`
   - [ ] Delete `testing.md` and `testing_improvements.md`

2. **Rewrite `README.md` from scratch:**
   - [ ] Current architecture overview (not historical)
   - [ ] Setup and installation instructions
   - [ ] Basic usage guide
   - [ ] Development setup

3. **Create new `docs/` structure:**
   ```
   docs/
   ├── architecture.md      # Current system architecture
   ├── tools.md             # Current AI tools (WS1-6 only)
   ├── settings.md          # Current settings reference
   └── development.md       # Contributing guide
   ```

4. **Review and update `docs/operations/*.md`:**
   - [ ] Verify all referenced code still exists
   - [ ] Remove references to deleted features
   - [ ] Update code examples to match current implementation

### 8.2 Documentation Standards

Going forward, documentation should:
- Describe **what exists now**, not historical evolution
- Include **working code examples** that are tested
- Avoid phrases like "previously", "legacy", "deprecated", "will be removed"
- Be updated **as part of the PR** when code changes

---

## Priority 9: Repository Structure Reorganization

### 9.1 Clean Up Root Directory

**Current root-level clutter:**
```
example.json                    # DELETE or move to test_data/
cleanup.md                      # KEEP until cleanup complete, then DELETE
```

> **Note:** Previous planning docs (`ai_enhancements.md`, `ai_refactor.md`, `testing.md`, etc.) have already been deleted.

**Target root structure:**
```
.gitignore
LICENSE
README.md
pyproject.toml
uv.lock
src/
tests/
docs/
assets/
```

**Action:**
- [ ] Move `example.json` to `test_data/` or delete
- [ ] Delete `benchmarks/` if no longer used, or move to `tests/benchmarks/`
- [ ] Delete `.retrieval-test/` and any other temp directories

### 9.2 Reorganize `src/tinkerbell/` Structure

**Current structure issues:**
- `chat/` and `widgets/` overlap (chat panel is a widget)
- `documents/` is ambiguous (models? file operations?)
- `ui/` and `widgets/` are confusingly separate
- `services/` is a dumping ground for unrelated modules

**Current layout:**
```
src/tinkerbell/
├── ai/              # AI orchestration (KEEP, needs internal cleanup)
├── app.py           # Entry point (KEEP)
├── chat/            # Chat panel components
├── documents/       # Document models?
├── editor/          # Editor widget
├── scripts/         # CLI scripts
├── services/        # Mixed bag of services
├── theme/           # Theme system
├── ui/              # UI controllers and coordinators
├── utils/           # Utilities
├── widgets/         # Qt widgets
└── __init__.py
```

**Proposed layout:**
```
src/tinkerbell/
├── ai/              # AI orchestration (unchanged)
├── core/            # Core models and types (from documents/, services/)
│   ├── document.py
│   ├── settings.py
│   └── types.py
├── ui/              # All UI code consolidated
│   ├── main_window.py
│   ├── chat/        # Chat panel (moved from chat/)
│   ├── editor/      # Editor widget (moved from editor/)
│   ├── widgets/     # Reusable widgets (moved from widgets/)
│   ├── dialogs/     # Dialog classes
│   └── theme/       # Theme system (moved from theme/)
├── services/        # Backend services only
│   ├── bridge.py
│   ├── telemetry.py
│   └── storage.py
├── utils/           # Utilities (unchanged)
├── scripts/         # CLI scripts (unchanged)
├── app.py           # Entry point (unchanged)
└── __init__.py
```

**Benefits:**
- Clear separation: `ui/` = presentation, `services/` = backend, `core/` = domain
- No more confusion between `widgets/`, `ui/`, `chat/`
- `documents/` renamed to clearer `core/`
- Theme is clearly part of UI

### 9.3 Reorganize `src/tinkerbell/ai/` Structure

**Current layout:**
```
ai/
├── agents/          # Analysis agents
├── ai_types.py      # Type definitions
├── analysis/        # Analysis adapters
├── client.py        # API client
├── memory/          # Memory/embeddings
├── orchestration/   # Main controller
├── prompts.py       # DELETE (legacy)
├── prompts_v2.py    # RENAME to prompts.py
├── services/        # AI services (context policy, etc.)
├── tools/           # Tool implementations
└── utils/           # AI utilities
```

**Proposed layout:**
```
ai/
├── client.py        # API client
├── controller.py    # Main controller (extracted from orchestration/)
├── prompts.py       # Prompts (renamed from prompts_v2.py)
├── types.py         # Type definitions (renamed from ai_types.py)
├── tools/           # Tool implementations
│   ├── read.py      # read_document, list_tabs
│   ├── write.py     # write_document, replace_lines, etc.
│   ├── search.py    # search_document, find_and_replace
│   ├── transform.py # transform_document
│   ├── analyze.py   # analyze_document
│   └── registry.py  # Tool registration
├── analysis/        # Analysis system
├── memory/          # Memory/embeddings
├── agents/          # Subagents
└── services/        # Supporting services
```

**Benefits:**
- Flat structure for main components (client, controller, prompts)
- Tools organized by operation type, not arbitrary groupings
- No more `orchestration/` directory with single file
- Cleaner imports: `from tinkerbell.ai import controller, prompts`

### 9.4 Reorganize Test Directory

**Current test structure mirrors old code structure - needs updating.**

**Proposed test layout:**
```
tests/
├── conftest.py          # Shared fixtures
├── fixtures/            # Complex test fixtures
├── unit/                # Unit tests (fast, isolated)
│   ├── test_tools.py
│   ├── test_controller.py
│   └── ...
├── integration/         # Integration tests
│   └── test_tool_wiring.py
└── data/                # Test data (moved from test_data/)
```

**Action:**
- [ ] Move `test_data/` contents to `tests/data/`
- [ ] Delete empty `test_data/` directory
- [ ] Consider splitting tests into `unit/` and `integration/`
- [ ] Consolidate `test_ws*.py` files into logical groupings

---

## Implementation Order

### Phase 1 (High Impact, Lower Risk)
1. Split `dialogs.py` - Low coupling, easy to test
2. Consolidate `estimate_tokens` duplicates
3. Modernize typing annotations
4. Remove dead no-op methods
5. **Delete deprecation layer entirely**
6. **Clean up root directory (delete outdated docs)**

### Phase 2 (Medium Impact, Medium Risk)  
1. Split `bridge.py`
2. Clean up exception handling
3. Remove duplicate `ToolRegistration`
4. Complete TODO items or document decisions
5. **Test suite cleanup - remove redundant tests**

### Phase 3 (High Impact, Higher Risk)
1. Split `main_window.py`
2. Split `controller.py`
3. Consolidate prompt systems (migrate to v2, delete v1)
4. **Complete documentation rewrite**

### Phase 4 (Structural Changes)
1. **Reorganize `src/tinkerbell/` directory structure**
2. **Reorganize `src/tinkerbell/ai/` structure**
3. **Reorganize test directory**
4. Update all imports throughout codebase
5. Final documentation pass

---

## Metrics to Track

| Metric | Current | Target |
|--------|---------|--------|
| Largest file (lines) | 4,241 (controller.py) | < 1,000 |
| Files > 1,000 lines | 7 | 0 |
| `# type: ignore` in src | ~5 | 0 |
| Bare `except Exception:` | 100+ | < 10 |
| Duplicate utilities | 3+ | 0 |

---

## Notes

- All changes should maintain 100% test pass rate
- Large refactors should be done in feature branches
- Consider adding pre-commit hooks for:
  - File length limits
  - Type checking (mypy)
  - Import sorting