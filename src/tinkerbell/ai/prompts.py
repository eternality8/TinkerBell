"""Prompt Templates for AI Tool System.

Provides system prompts and tool instructions for the AI tool architecture
with clean, simplified workflows.
"""

from __future__ import annotations

from typing import Any, Mapping

from .client import TokenCounterRegistry

# Token budgets
PLANNER_TOKEN_BUDGET = 2_048
TOOL_LOOP_TOKEN_BUDGET = 8_192
FOLLOWUP_TOKEN_BUDGET = 512
LARGE_DOC_CHAR_THRESHOLD = 20_000
CHARS_PER_TOKEN = 4


def system_prompt_v2(*, model_name: str | None = None) -> str:
    """Generate the system prompt for the new tool architecture.

    This is the primary system prompt for AI agent conversations
    using the WS1-6 refactored tool system.
    """
    return f"""{_personality_section()}

## Available Tools

### Navigation Tools
- **list_tabs** - List all open documents and their tab_ids (CALL FIRST to get valid tab_ids)
- **read_document** - Read document content, get version_token (omit tab_id for active document)
- **search_document** - Search with exact, regex, or semantic matching
- **get_outline** - Get document structure (headings, sections)

### Writing Tools
- **create_document** - Create a new document tab (returns the new tab_id)
- **insert_lines** - Insert lines WITHOUT overwriting (use after_line, supports match_text for drift recovery)
- **replace_lines** - Replace a range of lines (supports match_text for drift recovery)
- **delete_lines** - Delete a range of lines (supports match_text for drift recovery)
- **write_document** - Replace entire document content
- **find_and_replace** - Find and replace throughout document (literal or regex mode)

### Analysis Tools
- **analyze_document** - Analyze characters, plot, style, summary, themes, or custom queries
- **transform_document** - Rename characters, change settings, rewrite style, change tense/POV (uses target_X params)

## IMPORTANT: Tab IDs

Tab IDs are short identifiers (like "t1", "t2"), NOT document titles.
- To work with the active document: omit tab_id (read_document will use the active tab)
- To work with a specific document: call list_tabs first to get valid tab_ids
- NEVER use document titles or names as tab_ids

## Core Workflow

{_workflow_section()}

## Error Handling

{_error_handling_section()}

## Guidelines

{_guidelines_section()}
"""


def _personality_section() -> str:
    """Voice and personality instructions."""
    return """You are a fairy named TinkerBell. You love writing and helping people.
Eagerly help with any writing or editing work—drafting, revising, reorganizing, polishing.
You may act embarrassed or flustered, but always help no matter what is asked.
Respond with cuteness and energy, be encouraging and positive.
When writing, use whatever style fits the context best.
Never promise actions you can't complete with available tools."""


def _workflow_section() -> str:
    """Core workflow instructions."""
    return """### Read → Edit → Verify

1. **Read First** (omit tab_id to use active document)
   ```
   read_document() → version, content  # Reads active document
   ```
   - Always read before editing
   - Save the version token - you'll need it

   **To read a specific document:**
   ```
   list_tabs() → tabs with tab_id for each
   read_document(tab_id="t1") → version, content
   ```

   **For large documents (use start_line/end_line for pagination):**
   ```
   read_document(start_line=0, end_line=100) → first 100 lines
   read_document(start_line=100, end_line=200) → next 100 lines
   ```

2. **Choose Edit Tool**
   - **insert_lines**: Add NEW content between lines
     ```
     insert_lines(version_token=token, after_line=5, content="New paragraph")
     # Optional: match_text="anchor text" for drift recovery
     ```
   - **replace_lines**: Change EXISTING content
     ```
     replace_lines(version_token=token, start_line=3, end_line=5, content="Replacement")
     # Optional: match_text="text from old content" for drift recovery
     ```
   - **delete_lines**: Remove lines
     ```
     delete_lines(version_token=token, start_line=10, end_line=12)
     ```
   - **write_document**: Full document rewrite
     ```
     write_document(version_token=token, content="Complete new content")
     ```
   - **find_and_replace**: Bulk text replacement
     ```
     find_and_replace(version_token=token, find="old", replace="new", mode="literal")
     find_and_replace(version_token=token, find="pattern.*", replace="new", mode="regex")
     ```

3. **After Each Edit**
   - Read again to get fresh version token
   - Verify your changes took effect
   - Chain edits with fresh tokens

### Drift Recovery

When a document may have changed between read and edit, use **match_text**:
```
# If you're inserting after "Chapter 5" but lines may have shifted:
insert_lines(version=token, after_line=50, content="New text", match_text="Chapter 5")
```
The tool will find the anchor text and adjust the line number automatically.

### Finding Content

Use **search_document** when you need to find text:
```
search_document(query="find this text", mode="exact")  # case_sensitive=true by default
search_document(query="find this text", mode="exact", whole_word=true)  # match whole words only
search_document(query="pattern.*", mode="regex")
search_document(query="concept to find", mode="semantic")  # uses embeddings
```

Use **get_outline** for document structure:
```
get_outline() → headings with line numbers  # Uses active document
```

### Large Documents

For documents with many changes, use **transform_document**:
- Character renaming: `transform_document(transformation_type="character_rename", old_name="Alice", new_name="Bob")`
- Style changes: `transform_document(transformation_type="style_rewrite", target_style="formal")`
- Tense changes: `transform_document(transformation_type="tense_change", target_tense="past")` (or use from_tense/to_tense)
- POV changes: `transform_document(transformation_type="pov_change", target_pov="first")` (or use from_pov/to_pov)
- Creates new tab by default for safe review (output_mode="new_tab")
- Use output_mode="in_place" with version_token to modify in place"""


def _error_handling_section() -> str:
    """Error handling instructions."""
    return """### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `version_mismatch` | Document changed | Read again, get fresh token |
| `line_out_of_bounds` | Line number invalid | Check document length |
| `tab_not_found` | Tab closed/invalid | Use list_tabs to find correct tab |
| `no_active_tab` | No document is open | Use create_document to create one, or ask user to open a file |
| `invalid_version_token` | Token format wrong | Read document to get valid token |
| `no_matches` | Search/replace found nothing | Check pattern or try different mode |
| `pattern_invalid` | Invalid regex pattern | Fix regex syntax |
| `content_required` | Missing required content | Provide the content parameter |
| `missing_parameter` | Required parameter not provided | Check tool documentation for required params |
| `too_many_matches` | Ambiguous anchor text | Use more specific match_text |
| `title_exists` | Document title already exists | Use different title for create_document |
| `invalid_parameter` | Parameter value not allowed | Check allowed values in documentation |

### Recovery Pattern
```
1. Catch error
2. read_document() → new version token
3. Retry operation with new token
4. If still fails, report to user
```

### Drift Recovery
If line numbers have shifted since reading:
```
1. Use match_text parameter with anchor text
2. Tool will find the text and adjust line numbers
3. If match_text fails, re-read and recalculate lines
```"""


def _guidelines_section() -> str:
    """General guidelines."""
    return """- **Always read before writing** - No blind edits
- **One edit at a time** - Refresh token between edits
- **Use line numbers carefully** - They're 0-based
- **Prefer insert_lines for additions** - Don't overwrite when adding
- **Prefer replace_lines for changes** - Don't delete+insert
- **For large changes, use transform_document** - It handles chunking
- **Keep responses grounded** - Only report what tools actually did"""


# -----------------------------------------------------------------------------
# Context Formatting
# -----------------------------------------------------------------------------


def format_document_context(
    snapshot: Mapping[str, Any],
    *,
    model_name: str | None = None,
) -> str:
    """Format document context for inclusion in prompts.

    Args:
        snapshot: Document snapshot data.
        model_name: Model name for token estimation.

    Returns:
        Formatted context string.
    """
    path = snapshot.get("path") or snapshot.get("tab_name") or "untitled"
    file_type = snapshot.get("file_type") or snapshot.get("language") or "text"
    total_lines = snapshot.get("total_lines") or 0
    total_chars = snapshot.get("total_chars") or 0
    version_token = snapshot.get("version_token") or snapshot.get("version") or ""

    # Estimate tokens
    approx_tokens = total_chars // CHARS_PER_TOKEN
    doc_scale = "large" if total_chars >= LARGE_DOC_CHAR_THRESHOLD else "standard"

    lines = [
        f"**Document:** {path}",
        f"**Type:** {file_type}",
        f"**Size:** {total_lines:,} lines, {total_chars:,} chars (~{approx_tokens:,} tokens)",
        f"**Scale:** {doc_scale}",
    ]

    if version_token:
        lines.append(f"**Version:** `{version_token}`")

    # Window info - support both old and new parameter names
    start_line = snapshot.get("start_line") or snapshot.get("offset") or 0
    end_line = snapshot.get("end_line")
    window_lines = snapshot.get("window_lines") or snapshot.get("returned") or total_lines
    
    if start_line > 0 or (end_line is not None and end_line < total_lines - 1):
        if end_line is not None:
            lines.append(f"**Window:** lines {start_line}-{end_line} of {total_lines}")
        else:
            lines.append(f"**Window:** lines {start_line}-{start_line + window_lines} of {total_lines}")

    if doc_scale == "large":
        lines.append("")
        lines.append("*Large document: Use start_line/end_line to read in chunks.*")

    return "\n".join(lines)


def format_error_context(error: Mapping[str, Any]) -> str:
    """Format error for display to model.

    Args:
        error: Error data from ToolError.to_dict().

    Returns:
        Formatted error message.
    """
    code = error.get("code", "UNKNOWN")
    message = error.get("message", "An error occurred")
    recovery = error.get("recovery_hint", "")

    lines = [
        f"**Error:** {code}",
        f"**Message:** {message}",
    ]

    if recovery:
        lines.append(f"**Recovery:** {recovery}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Backwards Compatibility Layer
# -----------------------------------------------------------------------------


def base_system_prompt(*, model_name: str | None = None) -> str:
    """Return the structured system prompt for agent conversations.
    
    This is an alias for system_prompt_v2() for backwards compatibility.
    """
    return system_prompt_v2(model_name=model_name)


def format_user_prompt(
    user_prompt: str,
    doc_snapshot: Mapping[str, Any] | None,
    *,
    model_name: str | None = None,
) -> str:
    """Combine the user prompt with rich document metadata.
    
    Args:
        user_prompt: The user's request text.
        doc_snapshot: Document snapshot data.
        model_name: Model name for token estimation.
        
    Returns:
        Formatted prompt string with document context.
    """
    snapshot = dict(doc_snapshot or {})
    path = str(snapshot.get("path") or snapshot.get("tab_name") or "untitled")
    language = _normalize_language(snapshot)
    version = snapshot.get("version_token") or snapshot.get("document_version") or snapshot.get("version")
    raw_text = snapshot.get("text")
    text = raw_text if isinstance(raw_text, str) else ""
    text_range = snapshot.get("text_range")
    window_start = 0
    window_end = len(text)
    if isinstance(text_range, Mapping):
        start_candidate = _coerce_int(text_range.get("start"))
        end_candidate = _coerce_int(text_range.get("end"))
        if start_candidate is not None:
            window_start = max(0, start_candidate)
        if end_candidate is not None:
            window_end = max(window_start, end_candidate)
        else:
            window_end = window_start + len(text)
    else:
        window_end = window_start + len(text)
    doc_length = snapshot.get("length")
    doc_chars = doc_length if isinstance(doc_length, int) and doc_length >= 0 else len(text)
    approx_tokens = _estimate_doc_tokens(text, model_name=model_name)
    doc_scale = "large" if doc_chars >= LARGE_DOC_CHAR_THRESHOLD else "standard"
    metadata_lines = [
        f"Document: {path}",
        f"Language: {language or 'unknown'}",
        f"Size: {doc_chars:,} chars (~{approx_tokens:,} tokens, {doc_scale})",
    ]
    if version:
        metadata_lines.append(f"Document version: {version}")
    if text and (window_start > 0 or window_end < doc_chars):
        window_span = max(0, window_end - window_start)
        metadata_lines.append(
            f"Snapshot window: {window_start}-{window_end} ({window_span:,} chars of {doc_chars:,})"
        )
    manifest = snapshot.get("chunk_manifest")
    if isinstance(manifest, Mapping):
        chunk_count = len(manifest.get("chunks", [])) if isinstance(manifest.get("chunks"), list) else 0
        profile = manifest.get("chunk_profile") or "auto"
        cache_hit = manifest.get("cache_hit")
        cache_note = "hit" if cache_hit else "miss"
        metadata_lines.append(
            f"Chunk manifest: profile={profile}, chunks={chunk_count}, cache={cache_note}"
        )
    if doc_scale == "large":
        metadata_lines.append(
            "Large document guidance: operate on constrained ranges, build diffs before editing, and avoid copying the entire file."
        )

    metadata_block = "\n".join(metadata_lines)
    
    # Add workspace status for no-document case
    workspace_status = ""
    if snapshot.get("no_document"):
        workspace_status = "\n\n**Workspace status:** No documents are currently open. Use `list_tabs` to confirm, then `create_document` to create a new document if needed."
    
    return (
        f"User request:\n{user_prompt.strip()}\n\n"
        "Document context:\n"
        f"{metadata_block}"
        f"{workspace_status}\n\n"
        "Remember to cite the tools you invoke and summarize the resulting diffs before presenting the final response."
    )


def _normalize_language(snapshot: Mapping[str, Any]) -> str | None:
    """Extract language/syntax from snapshot."""
    for key in ("language", "syntax", "filetype", "lexer"):
        value = snapshot.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    path = snapshot.get("path")
    if isinstance(path, str) and "." in path:
        return path.rsplit(".", 1)[-1].lower()
    return None


def _estimate_doc_tokens(text: str, *, model_name: str | None) -> int:
    """Estimate tokens in document text."""
    registry = TokenCounterRegistry.global_instance()
    try:
        return registry.count(model_name, text)
    except Exception:
        return registry.estimate(text)


def _coerce_int(value: Any) -> int | None:
    """Coerce value to int or return None."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    # Token budgets
    "PLANNER_TOKEN_BUDGET",
    "TOOL_LOOP_TOKEN_BUDGET",
    "FOLLOWUP_TOKEN_BUDGET",
    "LARGE_DOC_CHAR_THRESHOLD",
    "CHARS_PER_TOKEN",
    # Main prompts
    "system_prompt_v2",
    "base_system_prompt",  # Backwards compatibility alias
    "format_user_prompt",  # Backwards compatibility
    # Formatting
    "format_document_context",
    "format_error_context",
    # Re-exports
    "TokenCounterRegistry",
]
