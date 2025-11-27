"""Tool for searching document content.

WS2.3: Implements exact text search, regex search, and semantic search
with embedding fallback.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, Sequence

from .base import ReadOnlyTool, ToolContext
from .errors import (
    InvalidParameterError,
    InvalidTabIdError,
    PatternInvalidError,
    TabNotFoundError,
    UnsupportedFileTypeError,
)
from .list_tabs import detect_file_type, is_supported_file_type
from .version import VersionManager, compute_content_hash, get_version_manager


# Default context lines around matches
DEFAULT_CONTEXT_LINES = 2

# Maximum number of results to return
MAX_RESULTS = 50

# Minimum semantic search score threshold
DEFAULT_SEMANTIC_MIN_SCORE = 0.5


class EmbeddingSearchProvider(Protocol):
    """Protocol for semantic search via embeddings."""

    async def similarity_search(
        self,
        document_id: str,
        *,
        query_text: str,
        top_k: int = 6,
        min_score: float = 0.0,
    ) -> Sequence[Any]:
        """Search for semantically similar chunks.

        Returns a sequence of match objects with `record` and `score` attributes.
        """
        ...

    def is_ready(self, document_id: str) -> bool:
        """Check if embeddings are ready for the given document."""
        ...


@dataclass(slots=True)
class SearchMatch:
    """Represents a single search result."""

    line: int
    column: int
    text: str  # The matched text
    preview: str  # Line containing the match
    context: dict[str, Any]  # start_line, end_line, context_preview
    score: float = 1.0  # Relevance score (1.0 for exact matches)


def split_lines(text: str) -> list[str]:
    """Split text into lines."""
    if not text:
        return []
    return text.split("\n")


def get_line_offsets(text: str) -> list[int]:
    """Get character offsets for each line start."""
    if not text:
        return []

    offsets = [0]
    for i, char in enumerate(text):
        if char == "\n":
            offsets.append(i + 1)
    return offsets


def offset_to_line_col(text: str, offset: int, line_offsets: list[int] | None = None) -> tuple[int, int]:
    """Convert character offset to line and column (0-indexed)."""
    if line_offsets is None:
        line_offsets = get_line_offsets(text)

    if not line_offsets:
        return (0, offset)

    # Binary search for the line
    left, right = 0, len(line_offsets) - 1
    while left < right:
        mid = (left + right + 1) // 2
        if line_offsets[mid] <= offset:
            left = mid
        else:
            right = mid - 1

    line = left
    col = offset - line_offsets[line]
    return (line, col)


def extract_context(
    lines: list[str],
    match_line: int,
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> dict[str, Any]:
    """Extract context around a match."""
    start_line = max(0, match_line - context_lines)
    end_line = min(len(lines) - 1, match_line + context_lines)

    context_text = "\n".join(lines[start_line : end_line + 1])

    return {
        "start_line": start_line,
        "end_line": end_line,
        "context_preview": context_text,
    }


def search_exact(
    text: str,
    pattern: str,
    *,
    case_sensitive: bool = True,
    whole_word: bool = False,
    max_results: int = MAX_RESULTS,
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> list[SearchMatch]:
    """Perform exact text search.

    Args:
        text: Document text to search.
        pattern: Text pattern to find.
        case_sensitive: Whether to match case.
        whole_word: Whether to match whole words only.
        max_results: Maximum number of results.
        context_lines: Lines of context around each match.

    Returns:
        List of SearchMatch objects.
    """
    if not text or not pattern:
        return []

    lines = split_lines(text)
    line_offsets = get_line_offsets(text)

    # Build search flags
    flags = 0 if case_sensitive else re.IGNORECASE

    # Escape pattern for regex and optionally add word boundaries
    escaped = re.escape(pattern)
    if whole_word:
        escaped = rf"\b{escaped}\b"

    matches: list[SearchMatch] = []
    try:
        for match in re.finditer(escaped, text, flags):
            if len(matches) >= max_results:
                break

            start = match.start()
            line, col = offset_to_line_col(text, start, line_offsets)

            matches.append(SearchMatch(
                line=line,
                column=col,
                text=match.group(),
                preview=lines[line] if line < len(lines) else "",
                context=extract_context(lines, line, context_lines),
                score=1.0,
            ))
    except re.error:
        # This shouldn't happen with escaped patterns, but just in case
        return []

    return matches


def search_regex(
    text: str,
    pattern: str,
    *,
    case_sensitive: bool = True,
    max_results: int = MAX_RESULTS,
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> list[SearchMatch]:
    """Perform regex search.

    Args:
        text: Document text to search.
        pattern: Regex pattern to find.
        case_sensitive: Whether to match case.
        max_results: Maximum number of results.
        context_lines: Lines of context around each match.

    Returns:
        List of SearchMatch objects.

    Raises:
        PatternInvalidError: If the regex pattern is invalid.
    """
    if not text or not pattern:
        return []

    lines = split_lines(text)
    line_offsets = get_line_offsets(text)

    flags = 0 if case_sensitive else re.IGNORECASE

    try:
        compiled = re.compile(pattern, flags)
    except re.error as e:
        raise PatternInvalidError(
            message=f"Invalid regex pattern: {e}",
            pattern=pattern,
            reason=str(e),
        )

    matches: list[SearchMatch] = []
    for match in compiled.finditer(text):
        if len(matches) >= max_results:
            break

        start = match.start()
        line, col = offset_to_line_col(text, start, line_offsets)

        matches.append(SearchMatch(
            line=line,
            column=col,
            text=match.group(),
            preview=lines[line] if line < len(lines) else "",
            context=extract_context(lines, line, context_lines),
            score=1.0,
        ))

    return matches


@dataclass
class SearchDocumentTool(ReadOnlyTool):
    """Search document content with multiple search modes.

    This tool supports:
    - Exact text search (literal string matching)
    - Regex search (pattern matching)
    - Semantic search (similarity via embeddings, when available)

    Parameters:
        tab_id: Target tab ID (optional, defaults to active tab)
        query: Search query string
        mode: Search mode - "exact", "regex", or "semantic" (default: "exact")
        case_sensitive: Whether to match case (default: true, not for semantic)
        whole_word: Match whole words only (exact mode only)
        max_results: Maximum results to return (default: 50)
        context_lines: Lines of context around matches (default: 2)

    Response includes:
        - matches: Array of match objects with line, column, preview, context
        - total_matches: Total number of matches found
        - mode: The search mode used
        - version: Version token for subsequent operations
        - embedding_status: Status of semantic search availability
    """

    name: ClassVar[str] = "search_document"
    description: ClassVar[str] = "Search document content with exact, regex, or semantic search"
    summarizable: ClassVar[bool] = True

    version_manager: VersionManager = field(default_factory=get_version_manager)
    embedding_provider: EmbeddingSearchProvider | None = None

    def read(self, context: ToolContext, params: dict[str, Any]) -> dict[str, Any]:
        """Execute the search_document tool.

        Args:
            context: Tool execution context.
            params: Tool parameters.

        Returns:
            Search results with metadata.
        """
        # Resolve tab ID
        tab_id = context.resolve_tab_id(params.get("tab_id"))
        if tab_id is None:
            raise InvalidTabIdError(
                message="No tab specified and no active tab available",
                tab_id=params.get("tab_id"),
            )

        # Get document content
        content = context.document_provider.get_document_content(tab_id)
        if content is None:
            raise TabNotFoundError(
                message=f"Tab '{tab_id}' not found",
                tab_id=tab_id,
            )

        # Get file metadata for type checking
        metadata = context.document_provider.get_document_metadata(tab_id)
        path = metadata.get("path") if metadata else None
        language = metadata.get("language") if metadata else None
        file_type = detect_file_type(path, language)

        if not is_supported_file_type(file_type):
            raise UnsupportedFileTypeError(
                message=f"Cannot search {file_type} files",
                file_type=file_type,
                file_path=path,
            )

        # Parse parameters
        query = params.get("query", "")
        if not query:
            raise InvalidParameterError(
                message="query parameter is required",
                parameter="query",
                value=query,
                expected="non-empty string",
            )

        mode = params.get("mode", "exact").lower()
        if mode not in ("exact", "regex", "semantic"):
            raise InvalidParameterError(
                message=f"Invalid search mode: {mode}",
                parameter="mode",
                value=mode,
                expected="'exact', 'regex', or 'semantic'",
            )

        case_sensitive = params.get("case_sensitive", True)
        whole_word = params.get("whole_word", False)
        max_results = params.get("max_results", MAX_RESULTS)
        context_lines = params.get("context_lines", DEFAULT_CONTEXT_LINES)

        # Perform search based on mode
        matches: list[SearchMatch] = []
        embedding_status = "not_available"

        if mode == "exact":
            matches = search_exact(
                content,
                query,
                case_sensitive=case_sensitive,
                whole_word=whole_word,
                max_results=max_results,
                context_lines=context_lines,
            )
        elif mode == "regex":
            matches = search_regex(
                content,
                query,
                case_sensitive=case_sensitive,
                max_results=max_results,
                context_lines=context_lines,
            )
        elif mode == "semantic":
            # Try semantic search, fall back to exact if unavailable
            matches, embedding_status = self._search_semantic(
                tab_id,
                content,
                query,
                max_results=max_results,
                context_lines=context_lines,
            )
            if not matches and embedding_status != "ready":
                # Fall back to exact search
                matches = search_exact(
                    content,
                    query,
                    case_sensitive=False,  # Semantic is case-insensitive
                    whole_word=False,
                    max_results=max_results,
                    context_lines=context_lines,
                )
                if matches:
                    embedding_status = "fallback_to_exact"

        # Register version for this tab
        content_hash = compute_content_hash(content)
        if not self.version_manager.get_current_token(tab_id):
            doc_id = uuid.uuid4().hex
            self.version_manager.register_tab(tab_id, doc_id, content_hash)
        token = self.version_manager.get_current_token(tab_id)

        # Format results
        results = [
            {
                "line": m.line,
                "column": m.column,
                "text": m.text,
                "preview": m.preview,
                "context": m.context,
                "score": round(m.score, 4),
            }
            for m in matches
        ]

        return {
            "matches": results,
            "total_matches": len(results),
            "mode": mode,
            "query": query,
            "tab_id": tab_id,
            "file_type": file_type,
            "version": token.to_string() if token else None,
            "embedding_status": embedding_status if mode == "semantic" else None,
        }

    def _search_semantic(
        self,
        tab_id: str,
        content: str,
        query: str,
        *,
        max_results: int,
        context_lines: int,
    ) -> tuple[list[SearchMatch], str]:
        """Perform semantic search using embeddings.

        Returns:
            Tuple of (matches, embedding_status).
        """
        if self.embedding_provider is None:
            return [], "not_configured"

        # Check if embeddings are ready
        try:
            if not self.embedding_provider.is_ready(tab_id):
                return [], "indexing"
        except Exception:
            return [], "error"

        # Perform async search (run in sync context)
        try:
            import asyncio

            # Try to get or create an event loop
            try:
                loop = asyncio.get_running_loop()
                # If we're in an async context, we can't run_until_complete
                # This is a limitation - semantic search needs async support
                return [], "async_context_unsupported"
            except RuntimeError:
                # No running loop, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(
                        self.embedding_provider.similarity_search(
                            tab_id,
                            query_text=query,
                            top_k=max_results,
                            min_score=DEFAULT_SEMANTIC_MIN_SCORE,
                        )
                    )
                finally:
                    loop.close()

            # Convert embedding matches to SearchMatch objects
            lines = split_lines(content)
            matches: list[SearchMatch] = []

            for result in results:
                record = result.record
                score = result.score

                # Convert offset to line number
                start_offset = record.start_offset
                line = 0
                char_count = 0
                for i, line_text in enumerate(lines):
                    if char_count + len(line_text) >= start_offset:
                        line = i
                        break
                    char_count += len(line_text) + 1  # +1 for newline

                # Get the matched text (chunk content)
                end_offset = min(record.end_offset, len(content))
                matched_text = content[start_offset:end_offset]

                # Truncate for preview
                preview = lines[line] if line < len(lines) else ""

                matches.append(SearchMatch(
                    line=line,
                    column=0,  # Semantic matches are chunk-level, not precise
                    text=matched_text[:100] + ("..." if len(matched_text) > 100 else ""),
                    preview=preview,
                    context=extract_context(lines, line, context_lines),
                    score=score,
                ))

            return matches, "ready"

        except Exception as e:
            return [], f"error: {str(e)[:50]}"


# Factory function for creating the tool
def create_search_document_tool(
    version_manager: VersionManager | None = None,
    embedding_provider: EmbeddingSearchProvider | None = None,
) -> SearchDocumentTool:
    """Create a SearchDocumentTool instance.

    Args:
        version_manager: Optional version manager.
        embedding_provider: Optional embedding search provider for semantic search.

    Returns:
        Configured SearchDocumentTool instance.
    """
    if version_manager is None:
        version_manager = get_version_manager()
    return SearchDocumentTool(
        version_manager=version_manager,
        embedding_provider=embedding_provider,
    )
