# AI Toolset Redesign

This document outlines a complete redesign of the AI toolset for TinkerBell, a tabbed text editor. The goal is to create a simpler, more robust system that enables AI-assisted creative writing, editing, and data manipulation for documents that may exceed context size.

---

## Executive Summary

The current tool system has grown complex with overlapping responsibilities, confusing version tokens, and multiple ways to accomplish the same task. This redesign proposes a clean, minimal toolset organized around clear responsibilities:

1. **Navigation Tools** - Understanding what's open and where content is
2. **Reading Tools** - Getting document content into context
3. **Writing Tools** - Creating and modifying document content
4. **Analysis Tools** - Understanding document structure and content (via subagents)

---

## Design Principles

### 1. Simplicity First
- Each tool does ONE thing well
- No overlapping functionality between tools
- Minimal required parameters
- Clear, predictable behavior

### 2. Context-Aware by Default
- Tools automatically work on the active tab unless specified
- Continuation is built-in for long documents
- No manual token counting or chunking by the AI

### 3. Consistency Guaranteed
- Simple version tokens prevent stale edits
- All edits use line-based addressing (no byte offsets)
- Atomic operations with clear success/failure

### 4. Long Document Support
- Streaming reads with automatic pagination
- Subagents for document-wide operations
- Smart summarization for context management

---

## Core Toolset

### Navigation Tools

#### `list_tabs`
Returns information about all open tabs.

**Parameters:** None

**Returns:**
```json
{
  "tabs": [
    {
      "tab_id": "tab_1",
      "title": "story.md",
      "path": "/documents/story.md",
      "file_type": "markdown",
      "version": "42",
      "is_active": true,
      "is_dirty": false,
      "size_chars": 45000,
      "line_count": 892
    }
  ],
  "active_tab_id": "tab_1"
}
```

**Use Cases:**
- Determine which documents are available
- Find a specific document by name
- Check if changes are unsaved

---

### Reading Tools

#### `read_document`
Read content from a document by line range. For finding specific content by description, use `search_document` instead.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `tab_id` | string | No | Target tab (defaults to active) |
| `start_line` | integer | No | First line to read (0-indexed, default 0) |
| `end_line` | integer | No | Last line to read (0-indexed, inclusive, default: auto-sized window) |

**Returns:**
```json
{
  "tab_id": "tab_1",
  "version": "42",
  "content": "The story begins...",
  "lines": {
    "start": 0,
    "end": 150,
    "total": 892
  },
  "tokens": {
    "returned": 5800,
    "total_estimate": 35000
  },
  "has_more": true,
  "continuation_hint": "Call read_document with start_line=151 to continue"
}
```

**Key Features:**
- Returns a `version` token that must be passed to edit operations
- All line numbers are 0-indexed and ranges are **inclusive** (consistent across all tools)
- Automatically limits response size to fit context budget (configurable, default ~6000 tokens)
- `tokens.returned` shows actual tokens in response; `tokens.total_estimate` helps plan chunking strategy
- `has_more` indicates if document continues beyond returned content
- Use `get_outline` first to find line numbers for specific sections

**Use Cases:**
- Read the beginning of a document
- Continue reading from where you left off
- Read a specific line range (after finding it via `get_outline` or `search_document`)

**Note on version tokens:** The `version` returned is valid for editing immediately. However, if you need more context before editing (e.g., reading additional sections), always use the version from your most recent `read_document` call.

**Empty document behavior:**
- Returns `content: ""`, `lines.total: 0`, `has_more: false`
- Version is still returned and valid for editing
- `insert_lines(after_line=-1)` works to add initial content
- `replace_lines` returns `invalid_line_range` error (no lines exist to replace)

---

#### `search_document`
Find content in a document by natural language description or exact text. Returns matching locations with surrounding context.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `tab_id` | string | No | Target tab (defaults to active) |
| `query` | string | Yes | What to find - natural language description OR exact text |
| `match_type` | string | No | "semantic" (default), "exact", or "regex" |
| `max_results` | integer | No | Maximum matches to return (default: 5) |
| `include_context` | boolean | No | Include surrounding lines (default: true) |

**Returns:**
```json
{
  "tab_id": "tab_1",
  "version": "42",
  "embedding_status": "ready",
  "matches": [
    {
      "line": 245,
      "score": 0.92,
      "preview": "...the fierce battle raged on as Alice drew her sword...",
      "context": {
        "start_line": 240,
        "end_line": 255,
        "line_count": 16,
        "content": "Chapter 12: The Battle\n\nThe morning sun rose..."
      }
    },
    {
      "line": 567,
      "score": 0.78,
      "preview": "...remembering the battle, Alice shuddered...",
      "context": {
        "start_line": 562,
        "end_line": 577,
        "line_count": 16,
        "content": "..."
      }
    }
  ],
  "total_matches": 2
}
```

**Key Features:**
- **Semantic search** (default): Finds content by meaning ("the scene where they fight")
- **Exact search**: Finds literal text matches
- **Regex search**: Pattern matching for advanced users
- Returns line numbers so you can use `read_document`, `insert_lines`, or `replace_lines` on that location
- Context helps verify you found the right section
- The returned `version` is valid for immediate editing (same as `read_document`)

**Semantic Search Availability:**
- Semantic search requires embeddings, which may not always be available
- If embeddings are not ready, the document is indexed just-in-time (may add latency on first search)
- If embeddings are unavailable (e.g., no embedding model configured), returns error with suggestion to use `match_type="exact"` or `match_type="regex"`
- The response includes `"embedding_status": "ready" | "indexing" | "unavailable"` to indicate state

**Use Cases:**
- Find a specific scene by description ("where Alice meets the Queen")
- Locate all mentions of a character
- Find a specific phrase or code pattern

---

#### `get_outline`
Get document structure (headings, sections) for navigation. Works differently based on file type.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `tab_id` | string | No | Target tab (defaults to active) |
| `max_depth` | integer | No | Maximum heading depth to include (default: 3) |
| `detect_chapters` | boolean | No | For plain text: attempt to detect chapter-like patterns (default: true) |

**Returns:**
```json
{
  "tab_id": "tab_1",
  "version": "42",
  "file_type": "markdown",
  "detection_method": "headings",
  "outline": [
    {
      "title": "Chapter 1: The Beginning",
      "level": 1,
      "line_start": 10,
      "line_end": 244,
      "children": [
        {
          "title": "The Journey Starts",
          "level": 2,
          "line_start": 25,
          "line_end": 89
        }
      ]
    }
  ]
}
```

**Detection Methods by File Type:**

| File Type | Method | What it detects |
|-----------|--------|-----------------|
| Markdown (`.md`) | `headings` | `#`, `##`, `###` heading markers |
| JSON (`.json`) | `keys` | Top-level object keys, array indices |
| YAML (`.yaml`) | `keys` | Top-level keys, nested structure |
| Plain text (`.txt`) | `heuristic` | See below |

**Plain Text Heuristics (`detection_method: "heuristic"`):**

For `.txt` files, the tool attempts multiple detection strategies in order:

1. **Explicit chapter markers**: Lines matching patterns like:
   - `Chapter 1`, `CHAPTER ONE`, `Chapter I`
   - `Part 1`, `PART ONE`, `Part I`
   - `Act 1`, `Scene 1`
   - `Section 1`, `1.`, `1)`

2. **Centered/emphasized titles**: Lines that are:
   - ALL CAPS and short (< 50 chars)
   - Surrounded by blank lines
   - Followed by a separator (`---`, `***`, `===`)

3. **Paragraph breaks**: If no chapters detected, falls back to:
   - Major breaks (3+ blank lines) → level 1 sections
   - Minor breaks (1-2 blank lines) → level 2 sections
   - Returns first ~20 words as section "title"

**Example for unstructured text:**
```json
{
  "tab_id": "tab_1",
  "version": "42",
  "file_type": "txt",
  "detection_method": "heuristic",
  "detection_confidence": "low",
  "outline": [
    {
      "title": "CHAPTER ONE",
      "level": 1,
      "line_start": 1,
      "line_end": 44,
      "detected_by": "chapter_marker"
    },
    {
      "title": "The sun rose slowly over the mountains...",
      "level": 2,
      "line_start": 45,
      "line_end": null,
      "detected_by": "paragraph_break"
    }
  ],
  "note": "Structure detected heuristically. Use search_document for more precise navigation."
}
```

**Note on `line_end`:** The `line_end` value is `null` for the last section in a document (extends to end of file). This allows the AI to read or replace an entire section without guessing where it ends.

**When heuristics fail:**
- If no structure is detected, returns empty outline with suggestion to use `search_document`
- `detection_confidence` field indicates reliability: "high", "medium", "low"
- For truly unstructured text, `search_document` is the better navigation tool

**Use Cases:**
- Navigate to specific chapters/sections in structured documents
- Get rough overview of plain text document structure
- Understand JSON/YAML schema before editing

**Note on version:** The `version` field is included so the AI can detect if the document changed between calling `get_outline` and a subsequent read/write operation. This is useful when the outline is used to plan edits - if the version differs from a later `read_document` call, the line numbers may have shifted.

---

### Writing Tools

#### `create_document`
Create a new document in a new tab.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `title` | string | Yes | Filename/title for the new document |
| `content` | string | No | Initial content (empty if not provided) |
| `file_type` | string | No | File type hint: "md", "txt", "json", "yaml" |

**Returns:**
```json
{
  "tab_id": "tab_2",
  "title": "new_story.md",
  "version": "1",
  "status": "created"
}
```

**Use Cases:**
- Start a new story
- Create a summary document
- Generate test data file

---

#### `insert_lines`
Insert new content at a specific location WITHOUT removing any existing content.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `version` | string | Yes | Version token from `read_document` |
| `after_line` | integer | Yes | Insert after this line (0-indexed). Use -1 to insert at document start |
| `content` | string | Yes | Content to insert (can be multiple lines) |
| `tab_id` | string | No | Target tab (defaults to active) |
| `dry_run` | boolean | No | If true, validate and return what would happen without making changes (default: false) |
| `match_text` | string | No | Optional anchor text for drift recovery. If provided, locates this text and inserts after the line containing it. |

**Returns:**
```json
{
  "tab_id": "tab_1",
  "version": "43",
  "status": "success",
  "inserted_at": {
    "after_line": 50,
    "lines_added": 15,
    "new_lines": { "start": 51, "end": 65 }
  }
}
```

**Key Features:**
- **Never deletes anything** - purely additive
- Supports optional `match_text` for drift recovery (same as `replace_lines`)
- Inserts AFTER the specified line:
  - Use `-1` to prepend to document start
  - Use `lines.total - 1` from `read_document` to append to end (e.g., if `lines.total` is 100, use `after_line=99`)
- Returns new version token for subsequent edits
- **Dry run mode**: Set `dry_run=true` to validate the operation and see what would happen without making changes. Version token is NOT consumed in dry run mode.

**Use Cases:**
- Add a new paragraph after an existing one
- Insert a new chapter between existing chapters
- Add content at the beginning of a document
- Append content to the end of a document

---

#### `replace_lines`
Replace a range of existing lines with new content. Use when you need to modify or delete existing text.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `version` | string | Yes | Version token from `read_document` |
| `start_line` | integer | Yes | First line to replace (0-indexed, inclusive) |
| `end_line` | integer | Yes | Last line to replace (0-indexed, inclusive) |
| `content` | string | Yes | Replacement content (empty string to delete) |
| `tab_id` | string | No | Target tab (defaults to active) |
| `dry_run` | boolean | No | If true, validate and return what would happen without making changes (default: false) |
| `match_text` | string | No | Optional anchor text for drift recovery. If provided and line content has shifted, the tool attempts to locate this exact text and adjust the range accordingly. |

**Returns:**
```json
{
  "tab_id": "tab_1",
  "version": "43",
  "status": "success",
  "lines_affected": {
    "removed": 10,
    "added": 15,
    "net_change": 5
  }
}
```

**Key Features:**
- **Requires valid range**: `end_line` must be >= `start_line`
- **Inclusive range**: Both start and end lines are replaced
- **Delete by empty content**: Set `content=""` to delete the range
- Returns new version token for subsequent edits
- **Dry run mode**: Set `dry_run=true` to validate the operation and see what would happen without making changes. Version token is NOT consumed in dry run mode.
- **Drift recovery**: Optional `match_text` parameter allows recovery when lines have shifted since the snapshot. If provided, the tool locates the exact text and adjusts the target range accordingly. Returns error if `match_text` is not found or matches multiple locations.

**Use Cases:**
- Rewrite a paragraph
- Fix a typo or error
- Delete unwanted content
- Replace a section with expanded content

---

#### `delete_lines`
Delete a range of lines from the document. Convenience wrapper that makes deletion intent explicit.

**Design Note:** This tool is functionally equivalent to `replace_lines(content="")`, which technically violates the "no overlapping functionality" principle. However, this exception is intentional:
- Makes deletion intent explicit and unambiguous in logs/history
- Reduces errors from forgetting to pass empty string
- Clearer for AI to choose the right tool for the task
- The overlap cost is minimal (simple wrapper, shared implementation)

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `version` | string | Yes | Version token from `read_document` |
| `start_line` | integer | Yes | First line to delete (0-indexed, inclusive) |
| `end_line` | integer | Yes | Last line to delete (0-indexed, inclusive) |
| `tab_id` | string | No | Target tab (defaults to active) |
| `dry_run` | boolean | No | If true, validate and return what would happen without making changes (default: false) |

**Returns:**
```json
{
  "tab_id": "tab_1",
  "version": "43",
  "status": "success",
  "lines_deleted": 10
}
```

**Use Cases:**
- Remove an unwanted paragraph
- Delete a chapter
- Clean up redundant content

---

#### `write_document`
Replace the ENTIRE document content. Use sparingly - prefer `insert_lines` or `replace_lines` for partial edits.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `version` | string | Yes | Version token from `read_document` |
| `content` | string | Yes | Complete new document content |
| `tab_id` | string | No | Target tab (defaults to active) |

**Returns:**
```json
{
  "tab_id": "tab_1",
  "version": "43",
  "status": "success",
  "lines_affected": {
    "previous": 892,
    "current": 950
  }
}
```

**Use Cases:**
- Complete document rewrite
- Generating small documents (< 500 lines)
- Format conversion

---

#### `find_and_replace`
Search and replace text across the document.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `version` | string | Yes | Version token from `read_document` |
| `find` | string | Yes | Text or regex pattern to find |
| `replace` | string | Yes | Replacement text |
| `is_regex` | boolean | No | Treat `find` as regex (default: false) |
| `case_sensitive` | boolean | No | Match case (default: true) |
| `whole_word` | boolean | No | Match whole words only (default: false) |
| `max_replacements` | integer | No | Limit number of replacements |
| `tab_id` | string | No | Target tab (defaults to active) |
| `preview` | boolean | No | If true, don't apply changes, just show matches |
| `scope` | object | No | Limit search to line range: `{"start_line": 100, "end_line": 200}` |

**Returns:**
```json
{
  "tab_id": "tab_1",
  "version": "43",
  "status": "success",
  "matches_found": 23,
  "replacements_made": 23,
  "preview": [
    {
      "line": 15,
      "before": "The character John said...",
      "after": "The character James said..."
    }
  ]
}
```

**Use Cases:**
- Rename a character throughout a story
- Fix repeated typos
- Update terminology
- Preview changes before applying

**Note:** When `preview: true`, the document is NOT modified and the version token is NOT consumed. You can preview, review the results, then call again with `preview: false` (or omit it) using the same version token.

**Preview truncation:** If there are more than 20 matches, the `preview` array is truncated to the first 20 entries and the response includes `"preview_truncated": true`. The `matches_found` field always reflects the true total count.

---

### Analysis Tools (Subagent-Powered)

These tools spawn subagents to process documents that exceed context size.

#### `analyze_document`
Perform deep analysis of a document using subagents.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `tab_id` | string | No | Target tab (defaults to active) |
| `task` | string | Yes | What to analyze: "characters", "plot", "style", "summary", "custom" |
| `custom_prompt` | string | No | Custom analysis instructions (when task="custom") |
| `output_tab` | string | No | Tab ID to write results (creates new tab if not specified) |

**Returns:**
```json
{
  "status": "complete",
  "result_tab_id": "tab_3",
  "summary": "Found 12 characters, 3 major plot threads...",
  "details": {
    "characters": [
      {"name": "Alice", "mentions": 234, "first_appearance": 5},
      {"name": "Bob", "mentions": 156, "first_appearance": 89}
    ]
  }
}
```

**Subagent Workflow:**
1. Document is chunked automatically
2. Subagents analyze each chunk in parallel
3. Results are aggregated and synthesized
4. Final result written to output tab

**Output Tab Behavior:**
- If `output_tab` is omitted: creates new tab named `"{source_title} - {task}"` (e.g., "story.md - summary")
- If `output_tab` specifies existing tab with content: **overwrites** the content (with diff review)
- If `output_tab` specifies non-existent tab ID: returns `invalid_tab_id` error

**Use Cases:**
- Extract character list from a novel
- Summarize a long document
- Analyze writing style
- Custom analysis queries

---

#### `transform_document`
Apply a transformation across an entire document using subagents.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `tab_id` | string | No | Source tab (defaults to active) |
| `transformation` | string | Yes | Type: "character_rename", "setting_change", "style_rewrite", "custom" |
| `params` | object | No | Transformation-specific parameters |
| `custom_prompt` | string | No | Custom transformation instructions |
| `output_mode` | string | No | "new_tab" (default) or "in_place" (modifies source, still goes through diff review) |
| `version` | string | Conditional | Required when `output_mode="in_place"`. Prevents stale transforms. |

**Transformation Parameters:**

For `character_rename`:
```json
{
  "old_name": "John",
  "new_name": "James",
  "update_pronouns": true,
  "gender": "male"
}
```

For `setting_change`:
```json
{
  "old_setting": "London",
  "new_setting": "Tokyo",
  "adapt_details": true
}
```

**Returns:**
```json
{
  "status": "complete",
  "result_tab_id": "tab_4",
  "changes_made": 145,
  "summary": "Renamed 'John' to 'James' in 145 locations across 892 lines"
}
```

**Subagent Workflow:**
1. First pass: Analyze document for transformation targets
2. Plan changes to maintain consistency
3. Apply changes chunk by chunk
4. Verify no continuity breaks

**Use Cases:**
- Rename character throughout long novel (with pronoun updates)
- Change story setting (adapting cultural details)
- Convert writing style (formal to casual)
- Gender swap a character

---

## Subagent Architecture

### When to Use Subagents

Subagents are spawned automatically for:
1. Documents exceeding ~20,000 characters
2. Operations requiring whole-document consistency
3. Analysis tasks needing multiple passes

### Subagent Types

#### Chunk Analyzer
- Processes document chunks in parallel
- Extracts entities, summaries, structural elements
- Results cached for reuse

#### Consistency Checker
- Validates changes don't break continuity
- Tracks entity references across chunks
- Flags potential issues for human review

#### Transformer
- Applies localized changes
- Uses context from analyzer for consistency
- Can request human confirmation for ambiguous cases

### Coordination Flow

```
User Request
    │
    ▼
┌─────────────────┐
│ Main Agent      │──── Simple task? ──► Direct tool call
│ (Coordinator)   │
└────────┬────────┘
         │
         │ Complex task
         ▼
┌─────────────────┐
│ Document        │
│ Chunker         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Chunk Analyzer  │ ◄──►│ Chunk Analyzer  │  (parallel)
│ (chunk 1)       │     │ (chunk 2)       │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌─────────────────┐
         │ Result          │
         │ Aggregator      │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Change          │
         │ Planner         │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Transformer     │──► Apply changes
         │ (per chunk)     │
         └─────────────────┘
```

---

## Version Token System

### Design
- Simple incrementing integer per document: `1`, `2`, `3`...
- Changes on ANY edit (even single character)
- Token is document-specific (not global)
- Tokens are opaque strings - AI should not parse or manipulate them

### Workflow
```
1. read_document → returns version="5"
2. replace_lines(version="5", ...) → returns version="6"
3. replace_lines(version="5", ...) → ERROR: stale version
```

### Error Handling
When version mismatch occurs:
```json
{
  "error": "version_mismatch",
  "message": "Document changed since you read it",
  "your_version": "5",
  "current_version": "7",
  "suggestion": "Call read_document to get current content and version"
}
```

---

## Sample Workflows

### 1. "Write me a story"
```
1. AI calls create_document(title="my_story.md", content="Once upon a time...") → returns version="1"
```

### 2. "Change John to James in this story"
```
Option A (simple, for short docs):
1. AI calls read_document()
2. AI calls find_and_replace(find="John", replace="James", preview=true)
3. AI reviews preview, then calls find_and_replace(find="John", replace="James")

Option B (comprehensive, for long docs):
1. AI calls transform_document(
     transformation="character_rename",
     params={"old_name": "John", "new_name": "James", "update_pronouns": true}
   )
```

### 3. "Add details to chapter 3"
```
1. AI calls get_outline() → finds Chapter 3 at line 450
2. AI calls read_document(start_line=450, end_line=550) → returns version="5"
3. AI understands context, plans additions
4. AI calls insert_lines(version="5", after_line=479, content="New paragraph...")
```

### 4. "Find the scene where Alice fights the dragon and expand it"
```
1. AI calls search_document(query="Alice fights the dragon")
2. AI gets match at line 623 with context
3. AI calls read_document(start_line=610, end_line=680) → returns version="12"
4. AI calls replace_lines(version="12", start_line=623, end_line=630, content="Expanded scene...") to rewrite
   OR AI calls insert_lines(version="12", after_line=630, content="Additional details...") to add more
```

### 5. "Change this story from London to Tokyo"
```
1. AI calls transform_document(
     transformation="setting_change",
     params={"old_setting": "London", "new_setting": "Tokyo", "adapt_details": true}
   )
```

### 6. "Summarize this document to another tab"
```
1. AI calls analyze_document(task="summary")
   → Creates new tab automatically with results
```

### 7. "Edit this YAML section"
```
1. AI calls search_document(query="database configuration", match_type="semantic")
2. AI gets match showing the config section at line 45
3. AI calls read_document(start_line=42, end_line=60) → returns version="3"
4. AI calls replace_lines(version="3", start_line=45, end_line=52, content="new yaml...")
```

### 8. "Generate JSON test data"
```
1. AI calls create_document(title="test_data.json", file_type="json", content='{"users": [...]}') → returns version="1"
```

---

## Comparison: Old vs New

| Old System | New System |
|------------|------------|
| `document_snapshot` + `document_apply_patch` with complex tokens | `read_document` + `insert_lines`/`replace_lines`/`delete_lines` with simple version |
| `snapshot_token = "tab_id:version_id"` parsing | Simple `version = "42"` (opaque string) |
| `target_span`, `target_range`, `match_text` options | Just `start_line` and `end_line` (0-indexed, inclusive) |
| Ambiguous insert/replace via same tool | Explicit `insert_lines`, `replace_lines`, `delete_lines` |
| Manual chunking with `document_chunk` | Automatic via subagents |
| Complex character tracking tools | `transform_document` with subagent coordination |
| `DocumentFindTextTool` mixed into snapshot | Dedicated `search_document` tool |

---

## Migration Path

### Phase 1: Core Tools
1. Implement `list_tabs`, `read_document`, `create_document`
2. Implement `insert_lines`, `replace_lines`, `delete_lines`
3. Add version token system
4. Basic tests and validation

### Phase 2: Navigation & Search
1. Add `get_outline` for structure navigation
2. Add `search_document` with semantic, exact, and regex modes
3. Automatic pagination for `read_document`

### Phase 3: Bulk Operations
1. Implement `find_and_replace`
2. Add `write_document`
3. Preview mode

### Phase 4: Subagent Integration
1. Implement `analyze_document`
2. Implement `transform_document`
3. Chunk coordination system
4. Consistency checking

### Phase 5: Polish
1. Error handling refinement
2. Telemetry integration
3. Performance optimization
4. Documentation

---

## Tool Summary Table

| Tool | Purpose | Requires Version | Modifies Content |
|------|---------|-----------------|------------------|
| `list_tabs` | See open documents | No | No |
| `read_document` | Get content by line range | No | No |
| `search_document` | Find content by query | No | No |
| `get_outline` | Navigate structure | No | No |
| `create_document` | New tab | No | Yes (creates) |
| `insert_lines` | Add new content (no deletions) | Yes | Yes (additive only) |
| `replace_lines` | Modify/overwrite existing content | Yes | Yes (replaces range) |
| `delete_lines` | Remove content | Yes | Yes (removes range) |
| `write_document` | Replace entire document | Yes | Yes (full rewrite) |
| `find_and_replace` | Bulk text replacement | Yes | Yes (or No if preview=true) |
| `analyze_document` | Deep analysis | No | Yes (to output tab) |
| `transform_document` | Bulk transform | Conditional* | Yes |

*`transform_document` requires version only when `output_mode="in_place"`

---

## Open Questions

### 1. ~~Undo support~~ → RESOLVED: Turn-Based Checkpoints + Diff Review
~~Should write operations support undo? How does this interact with version tokens?~~

### 5. ~~Error recovery~~ → RESOLVED: Atomic Operations + Rollback
~~If a multi-chunk transform fails partway, how to rollback or resume?~~

**Combined Solution: Atomic Operations + Turn Checkpoints + Diff Review**

#### For AI Operations (Atomicity)

All tool calls within a single AI turn are atomic:
- Changes are staged in memory, not applied immediately
- If any operation in a multi-step process fails, ALL changes roll back
- Only on complete success are changes committed to the document
- This applies to both single tools and multi-chunk subagent operations

**Example - Multi-chunk transform:**
```
transform_document("rename John to James")
  ├── Chunk 1: Success (staged)
  ├── Chunk 2: Success (staged)
  ├── Chunk 3: FAIL (API error)
  └── ROLLBACK: Chunks 1 & 2 discarded, document unchanged
```

#### For User Review (Diff Review)

After AI turn completes successfully:
1. Show diff view comparing before/after
2. User can Accept or Reject the changes
3. Rejecting restores the pre-turn state

**UI Flow:**
```
┌─────────────────────────────────────────┐
│ 📝 story.md - Review AI Changes         │
├─────────────────────────────────────────┤
│ - The character John walked in.         │  (red = removed)
│ + The character James walked in.        │  (green = added)
│                                         │
│   [Accept Changes]  [Reject Changes]    │
└─────────────────────────────────────────┘
```

#### For History (Turn Checkpoints)

Every AI turn creates a checkpoint that can be restored later:
- Checkpoints stored per-document
- User can browse turn history and rollback to any point
- Checkpoint includes: document state, turn number, timestamp, AI action summary

**UI - Turn History Panel:**
```
Turn History - story.md
─────────────────────────
● Current
○ Turn 5: "Renamed John to James" (2 min ago)
○ Turn 4: "Added chapter 3" (5 min ago)
○ Turn 3: "Fixed dialogue" (10 min ago)
○ Turn 2: "Expanded intro" (15 min ago)
○ Turn 1: "Initial draft" (20 min ago)

[Restore Selected Turn]
```

**Checkpoint behavior:**
- Selecting a past turn shows diff from current state
- "Restore" reverts document to that checkpoint
- Restoring creates a NEW checkpoint (non-destructive)
- Checkpoints persist for session (cleared on document close, or optionally saved)

#### Version Token Interaction

- Version tokens increment on commit (after user accepts)
- Rejected changes don't increment version
- Rollback to checkpoint sets version to that checkpoint's version
- This keeps version tokens simple and sequential

**Summary:**
| Scenario | Behavior |
|----------|----------|
| Tool fails mid-operation | Automatic rollback, document unchanged |
| AI turn completes | Show diff, user accepts/rejects |
| User rejects changes | Restore pre-turn state |
| User wants to undo older turn | Use turn history to restore checkpoint |

### 2. ~~Concurrent edits~~ → RESOLVED: Global Editor Lock
~~What happens if user edits document while AI is working?~~

**Solution: Lock All Editing During AI Turn**

When AI turn begins:
1. Lock ALL editor tabs to view-only mode
2. Show visual indicator (e.g., "AI working..." in status bar)
3. User can scroll and read any tab, but cannot edit any document
4. Lock releases when AI turn completes (success or failure)

**Implementation details:**
- Global lock (all tabs locked, not just the one AI is working on)
- Lock acquired when: AI turn starts (user sends message)
- Lock released when: AI turn ends (response complete or error/timeout)
- User can cancel AI turn, which releases lock immediately

**UI indicator:**
```
┌─────────────────────────────────────────┐
│ 📝 story.md                              │
├─────────────────────────────────────────┤
│  (document content - read only)         │
└─────────────────────────────────────────┘
│ 🤖 AI working...              [Cancel]  │
└─────────────────────────────────────────┘
```

**Benefits:**
- Simplest possible solution
- No version conflicts possible
- No per-tab lock tracking needed
- User expectation is clear: wait for AI or cancel

### 3. ~~Large content generation~~ → RESOLVED: Not a Real Concern
~~For `insert_lines` or `replace_lines` with very large content, should we stream or chunk the write?~~

**Analysis:** AI output is bounded by context window size (~100-200k tokens max, typically much less). Even at the extreme upper bound, this translates to ~400-800KB of text. Modern text editors handle multi-megabyte files without issue, so any AI-generated write will complete instantly from a UI perspective.

**Conclusion:** No special handling needed. Standard synchronous writes are fine.

### 4. ~~Subagent visibility~~ → RESOLVED: Status Bar Counter
~~Should the user see subagent progress? How to surface partial results?~~

**Solution:** Simple status bar indicator showing active subagent count.

**UI:**
```
│ 🤖 AI working... (3 subagents active)   [Cancel] │
```

**Behavior:**
- Counter increments as subagents spawn
- Counter decrements as subagents complete
- No need to show partial results or per-chunk progress
- Existing status bar infrastructure can be reused

### 6. Version Token Scope

**Clarifications:**
- Version tokens are **per-tab**, not per-file-path
- If the same file is open in two tabs, each tab has independent versions
- On save: version is NOT reset (continues incrementing)
- On reload from disk: version IS reset to `1` (document state changed externally)
- On close/reopen: new tab gets new `tab_id` and version starts at `1`

### 7. Tab ID Stability

**Clarifications:**
- `tab_id` values are stable within a session but NOT across sessions
- Closing and reopening a document creates a new `tab_id`
- Saving to a new location does NOT change `tab_id` (tab identity preserved)
- Use `list_tabs` to discover current tab IDs; don't cache them across sessions

### 8. Unsupported File Types

**Behavior for binary/unsupported files:**
- `list_tabs` includes ALL open tabs, with `file_type: "binary"` or `file_type: "unknown"`
- `read_document` returns error: `{"error": "unsupported_file_type", "message": "Cannot read binary file"}`
- Write operations similarly return errors for unsupported types
- This prevents accidental corruption of binary files

### 9. AI Turn Definition

**What constitutes a "turn":**
- A turn begins when the user sends a message and ends when the AI's response is complete (no pending tool calls)
- If AI makes multiple tool calls, they are all part of the same turn
- All tool calls within a turn are atomic (all succeed or all roll back)
- Streaming responses: turn ends when stream completes, not when first tokens arrive

### 10. Diff Review Timing

**When diff review appears:**
- Diff review is shown AFTER the AI's response text is displayed
- User sees the AI's explanation first, then reviews the proposed changes
- This allows users to understand what was done before deciding to accept/reject

**Flow:**
```
1. User sends message
2. AI processes, calls tools (staged, not applied)
3. AI response text streams to user
4. Response completes→ Diff review appears
5. User accepts or rejects
6. If accepted: changes committed, version incremented
```

### 11. Multi-Document Edits in One Turn

**When AI edits multiple documents in a single turn:**
- Diff review shows changes per-tab (tabbed diff view)
- User must Accept or Reject the **entire turn** (all-or-nothing)
- This prevents inconsistent state when changes across tabs are interdependent (e.g., character renamed in both `story.md` and `character_notes.md`)
- On Accept: all version tokens increment, all changes committed
- On Reject: all tabs remain unchanged

**UI:**
```
┌─────────────────────────────────────────┐
│ Review AI Changes (2 files)             │
├─────────────────────────────────────────┤
│ [📄 story.md] [📄 notes.md]            │
├─────────────────────────────────────────┤
│ - The character John walked in.         │
│ + The character James walked in.        │
│                                         │
│   [✓ Accept All] [✗ Reject All]         │
└─────────────────────────────────────────┘
```

---

## Appendix: File Type Support

| Extension | Type | Outline Support | Detection Method | Validation |
|-----------|------|-----------------|------------------|------------|
| `.md` | Markdown | Excellent | Heading markers (`#`) | Optional lint |
| `.txt` | Plain text | Heuristic | Chapter patterns, breaks | None |
| `.json` | JSON | Good | Object keys | JSON schema |
| `.yaml`/`.yml` | YAML | Good | Top-level keys | YAML syntax |

### Plain Text Outline Detection Details

Since `.txt` files have no formal structure, we use a multi-pass heuristic approach:

**Pass 1 - Explicit Markers (high confidence):**
```
Chapter 1          ✓ detected
CHAPTER ONE        ✓ detected  
Chapter I          ✓ detected
Part One           ✓ detected
Act II, Scene 3    ✓ detected
--- Section ---    ✓ detected
```

**Pass 2 - Visual Patterns (medium confidence):**
```
                   
THE BEGINNING      ← ALL CAPS, surrounded by blanks
                   
---                ← followed by separator

* * *              ← scene break marker
```

**Pass 3 - Paragraph Analysis (low confidence):**
- Groups text by blank line patterns
- Uses first sentence fragment as "title"
- Best for documents with clear paragraph structure

**Fallback behavior:**
When no structure is detected, `get_outline` returns:
```json
{
  "outline": [],
  "detection_method": "none",
  "suggestion": "This document has no detectable structure. Use search_document to find specific content, or read_document to browse sequentially."
}
```

This guides the AI to use appropriate alternative tools rather than failing silently.

---

## Appendix: Token Budget Strategy

For documents exceeding context:

1. **Initial read**: Return first ~6000 tokens worth of content
2. **Outline always fits**: Structure summary stays small
3. **Targeted reads**: Use `query` or line ranges to get specific sections
4. **Subagent chunking**: ~4000 tokens per chunk for analysis
5. **Response budget**: Reserve ~2000 tokens for AI response

This ensures the AI can always work effectively without manual token management.

**Configuration:** These defaults can be adjusted in settings:
- `ai.tokenBudget.initialRead`: Default 6000
- `ai.tokenBudget.chunkSize`: Default 4000
- `ai.tokenBudget.responseReserve`: Default 2000

---

## Appendix: Error Response Schema

All tools return errors in a consistent format:

```json
{
  "error": "error_code",
  "message": "Human-readable description",
  "details": { /* optional additional context */ }
}
```

**Common Error Codes:**

| Error Code | Description | Returned By |
|------------|-------------|-------------|
| `version_mismatch` | Document changed since version token was obtained | All write tools |
| `invalid_tab_id` | Tab ID not found (tab may have been closed) | All tools with `tab_id` |
| `title_exists` | A tab with this title already exists | `create_document` |
| `invalid_line_range` | Line numbers out of bounds or start > end | `read_document`, `replace_lines`, `delete_lines` |
| `unsupported_file_type` | Operation not supported for this file type | `read_document`, write tools |
| `invalid_regex` | Regex pattern syntax error | `search_document`, `find_and_replace` |
| `content_required` | Empty content provided where content is required | `insert_lines` |
| `embeddings_unavailable` | Semantic search requested but embeddings not configured | `search_document` |
| `subagent_failed` | Subagent encountered an unrecoverable error | `analyze_document`, `transform_document` |

**Example - Invalid Line Range:**
```json
{
  "error": "invalid_line_range",
  "message": "Line range 500-600 exceeds document length of 450 lines",
  "details": {
    "requested_start": 500,
    "requested_end": 600,
    "document_lines": 450
  }
}
```
