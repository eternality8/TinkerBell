"""Find and Replace Tool for AI operations.

Provides functionality to find and replace text patterns in documents
with support for literal and regex modes, preview, and scope limiting.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar

from .base import WriteTool, ToolContext
from .errors import (
    InvalidParameterError,
    MissingParameterError,
    PatternInvalidError,
    NoMatchesError,
    TabNotFoundError,
)
from .version import VersionManager, VersionToken
from .insert_lines import split_lines, DocumentEditor

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

MAX_PREVIEW_MATCHES = 20
DEFAULT_MAX_REPLACEMENTS = 1000


# -----------------------------------------------------------------------------
# Match Result
# -----------------------------------------------------------------------------


@dataclass
class ReplacementMatch:
    """Represents a single match for replacement."""

    line: int  # 0-indexed line number
    column: int  # 0-indexed column
    original: str  # Original matched text
    replacement: str  # Replacement text
    line_before: str  # Full line before replacement
    line_after: str  # Full line after replacement


# -----------------------------------------------------------------------------
# Search and Replace Logic
# -----------------------------------------------------------------------------


def find_matches_literal(
    text: str,
    pattern: str,
    *,
    case_sensitive: bool = True,
    whole_word: bool = False,
    start_line: int | None = None,
    end_line: int | None = None,
) -> list[tuple[int, int, int]]:
    """Find all literal matches in text.

    Args:
        text: Document text.
        pattern: Literal pattern to find.
        case_sensitive: Whether to match case.
        whole_word: Whether to match whole words only.
        start_line: Start of scope (0-indexed, inclusive).
        end_line: End of scope (0-indexed, inclusive).

    Returns:
        List of (line, start_offset, end_offset) tuples.
    """
    lines = split_lines(text)
    matches: list[tuple[int, int, int]] = []

    search_pattern = pattern
    if not case_sensitive:
        search_pattern = pattern.lower()

    for line_num, line in enumerate(lines):
        # Apply scope filter
        if start_line is not None and line_num < start_line:
            continue
        if end_line is not None and line_num > end_line:
            continue

        search_line = line if case_sensitive else line.lower()
        start = 0

        while True:
            pos = search_line.find(search_pattern, start)
            if pos == -1:
                break

            # Check whole word boundary if required
            if whole_word:
                before_ok = pos == 0 or not search_line[pos - 1].isalnum()
                after_pos = pos + len(pattern)
                after_ok = after_pos >= len(search_line) or not search_line[after_pos].isalnum()
                if not (before_ok and after_ok):
                    start = pos + 1
                    continue

            matches.append((line_num, pos, pos + len(pattern)))
            start = pos + 1

    return matches


def find_matches_regex(
    text: str,
    pattern: str,
    *,
    case_sensitive: bool = True,
    start_line: int | None = None,
    end_line: int | None = None,
) -> list[tuple[int, int, int, re.Match[str]]]:
    """Find all regex matches in text.

    Args:
        text: Document text.
        pattern: Regex pattern to find.
        case_sensitive: Whether to match case.
        start_line: Start of scope (0-indexed, inclusive).
        end_line: End of scope (0-indexed, inclusive).

    Returns:
        List of (line, start_offset, end_offset, match_object) tuples.

    Raises:
        PatternInvalidError: If regex pattern is invalid.
    """
    flags = 0 if case_sensitive else re.IGNORECASE

    try:
        compiled = re.compile(pattern, flags)
    except re.error as exc:
        raise PatternInvalidError(
            message=f"Invalid regex pattern: {exc}",
            pattern=pattern,
            reason=str(exc),
        ) from exc

    lines = split_lines(text)
    matches: list[tuple[int, int, int, re.Match[str]]] = []

    for line_num, line in enumerate(lines):
        # Apply scope filter
        if start_line is not None and line_num < start_line:
            continue
        if end_line is not None and line_num > end_line:
            continue

        for match in compiled.finditer(line):
            matches.append((line_num, match.start(), match.end(), match))

    return matches


def apply_replacements(
    text: str,
    matches: list[tuple[int, int, int]],
    replacement: str,
    *,
    is_regex: bool = False,
    regex_matches: list[re.Match[str]] | None = None,
    max_replacements: int = DEFAULT_MAX_REPLACEMENTS,
) -> tuple[str, list[ReplacementMatch]]:
    """Apply replacements to text.

    Args:
        text: Original text.
        matches: List of (line, start, end) matches.
        replacement: Replacement string (may contain backreferences for regex).
        is_regex: Whether replacement supports backreferences.
        regex_matches: Original regex match objects for backreference expansion.
        max_replacements: Maximum replacements to apply.

    Returns:
        Tuple of (new_text, list of ReplacementMatch details).
    """
    if not matches:
        return text, []

    lines = split_lines(text)
    replacement_details: list[ReplacementMatch] = []

    # Group matches by line (reversed for safe replacement)
    matches_by_line: dict[int, list[tuple[int, int, int | None]]] = {}
    for i, match in enumerate(matches[:max_replacements]):
        line_num, start, end = match[:3]
        match_idx = i if regex_matches else None
        if line_num not in matches_by_line:
            matches_by_line[line_num] = []
        matches_by_line[line_num].append((start, end, match_idx))

    # Process each line with matches
    for line_num in sorted(matches_by_line.keys()):
        line_matches = sorted(matches_by_line[line_num], reverse=True)  # Process right to left
        line_before = lines[line_num]
        line_after = line_before

        for start, end, match_idx in line_matches:
            original_text = line_before[start:end]

            # Expand replacement (handle backreferences for regex)
            if is_regex and regex_matches and match_idx is not None:
                expanded = regex_matches[match_idx].expand(replacement)
            else:
                expanded = replacement

            # Apply replacement
            line_after = line_after[:start] + expanded + line_after[end:]

            replacement_details.append(ReplacementMatch(
                line=line_num,
                column=start,
                original=original_text,
                replacement=expanded,
                line_before=line_before,
                line_after=line_after,
            ))

        lines[line_num] = line_after

    # Reverse the details to be in document order
    replacement_details.reverse()

    return "\n".join(lines), replacement_details


# -----------------------------------------------------------------------------
# Find and Replace Tool
# -----------------------------------------------------------------------------


@dataclass
class FindAndReplaceTool(WriteTool):
    """Tool for finding and replacing text patterns in a document.

    Supports both literal and regex search modes with various options.
    Can preview changes before applying them.

    Parameters:
        tab_id: Target tab ID (optional, defaults to active tab).
        version: Version token from a previous read (required).
        find: Pattern to search for (required).
        replace: Replacement text (required, use "" for deletion).
        mode: Search mode - "literal" (default) or "regex".
        case_sensitive: Whether to match case (default True).
        whole_word: For literal mode, match whole words only (default False).
        scope: Optional scope limiting.
            - start_line: First line to search (0-indexed, inclusive).
            - end_line: Last line to search (0-indexed, inclusive).
        max_replacements: Maximum replacements to apply (default 1000).
        preview: If True, show matches without applying (default False).
        dry_run: If True, validate and preview without applying (default False).

    Returns:
        matches_found: Total number of matches found.
        replacements_made: Number of replacements applied (0 if preview/dry_run).
        preview: Array of match details (limited to 20 in preview mode).
            - line: Line number of match.
            - column: Column position of match.
            - original: Original matched text.
            - replacement: Replacement text.
            - line_before: Full line before replacement.
            - line_after: Full line after replacement.
        version: New version token (or unchanged if preview/dry_run).
    """

    name: ClassVar[str] = "find_and_replace"
    summarizable: ClassVar[bool] = False

    version_manager: VersionManager
    document_editor: DocumentEditor | None = None

    def validate(self, params: dict[str, Any]) -> None:
        """Validate parameters."""
        super().validate(params)

        find_pattern = params.get("find")
        if not find_pattern:
            raise MissingParameterError(
                message="Parameter 'find' is required",
                parameter="find",
            )

        if "replace" not in params:
            raise MissingParameterError(
                message="Parameter 'replace' is required",
                parameter="replace",
            )

        mode = params.get("mode", "literal")
        if mode not in ("literal", "regex"):
            raise InvalidParameterError(
                message=f"Invalid mode: {mode}. Must be 'literal' or 'regex'",
                parameter="mode",
                value=mode,
                expected="'literal' or 'regex'",
            )

    def write(
        self,
        context: ToolContext,
        params: dict[str, Any],
        token: VersionToken,
    ) -> dict[str, Any]:
        """Perform the find and replace operation."""
        tab_id = token.tab_id
        find_pattern = params["find"]
        replace_text = params["replace"]
        mode = params.get("mode", "literal")
        case_sensitive = params.get("case_sensitive", True)
        whole_word = params.get("whole_word", False)
        scope = params.get("scope", {})
        max_replacements = params.get("max_replacements", DEFAULT_MAX_REPLACEMENTS)
        preview_only = params.get("preview", False)

        # Get current document content
        doc_content = context.document_provider.get_document_content(tab_id)
        if doc_content is None:
            raise TabNotFoundError(
                message=f"Document not found: {tab_id}",
                tab_id=tab_id,
            )

        # Extract scope
        start_line = scope.get("start_line")
        end_line = scope.get("end_line")

        # Find matches
        if mode == "regex":
            raw_matches = find_matches_regex(
                doc_content,
                find_pattern,
                case_sensitive=case_sensitive,
                start_line=start_line,
                end_line=end_line,
            )
            matches = [(m[0], m[1], m[2]) for m in raw_matches]
            regex_match_objects = [m[3] for m in raw_matches]
        else:
            matches = find_matches_literal(
                doc_content,
                find_pattern,
                case_sensitive=case_sensitive,
                whole_word=whole_word,
                start_line=start_line,
                end_line=end_line,
            )
            regex_match_objects = None

        matches_found = len(matches)

        if matches_found == 0:
            raise NoMatchesError(
                message=f"No matches found for pattern: {find_pattern}",
                pattern=find_pattern,
            )

        # If preview only, don't apply changes
        if preview_only:
            # Generate preview for first N matches
            preview_matches = matches[:MAX_PREVIEW_MATCHES]
            lines = split_lines(doc_content)

            preview_list = []
            for i, match in enumerate(preview_matches):
                line_num, start, end = match
                original = lines[line_num][start:end]

                # Expand replacement for preview
                if mode == "regex" and regex_match_objects:
                    expanded = regex_match_objects[i].expand(replace_text)
                else:
                    expanded = replace_text

                line_before = lines[line_num]
                line_after = line_before[:start] + expanded + line_before[end:]

                preview_list.append({
                    "line": line_num,
                    "column": start,
                    "original": original,
                    "replacement": expanded,
                    "line_before": line_before,
                    "line_after": line_after,
                })

            return {
                "matches_found": matches_found,
                "replacements_made": 0,
                "preview": preview_list,
                "preview_truncated": matches_found > MAX_PREVIEW_MATCHES,
            }

        # Apply replacements
        new_text, replacement_details = apply_replacements(
            doc_content,
            matches,
            replace_text,
            is_regex=(mode == "regex"),
            regex_matches=regex_match_objects,
            max_replacements=max_replacements,
        )

        # Apply edit via document provider
        if hasattr(context.document_provider, 'set_document_content'):
            context.document_provider.set_document_content(tab_id, new_text)
        elif self.document_editor:
            self.document_editor.set_document_text(tab_id, new_text)

        # Build result
        replacements_made = len(replacement_details)

        # Include preview of changes made (limited)
        preview_list = []
        for detail in replacement_details[:MAX_PREVIEW_MATCHES]:
            preview_list.append({
                "line": detail.line,
                "column": detail.column,
                "original": detail.original,
                "replacement": detail.replacement,
            })

        return {
            "matches_found": matches_found,
            "replacements_made": replacements_made,
            "preview": preview_list,
            "preview_truncated": replacements_made > MAX_PREVIEW_MATCHES,
            "_new_text": new_text,
        }

    def preview(
        self,
        context: ToolContext,
        params: dict[str, Any],
        token: VersionToken,
    ) -> dict[str, Any]:
        """Preview the find and replace without applying."""
        # Set preview flag and delegate to write
        preview_params = dict(params)
        preview_params["preview"] = True

        # Call write but without the _new_text (which triggers version update)
        result = self.write(context, preview_params, token)
        result.pop("_new_text", None)

        return result
