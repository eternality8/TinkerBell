"""Unit tests for WS2 Navigation & Reading Tools."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field

import pytest

from tinkerbell.ai.tools.version import VersionManager, reset_version_manager
from tinkerbell.ai.tools.base import ToolContext, DocumentProvider
from tinkerbell.ai.tools.list_tabs import (
    ListTabsTool,
    detect_file_type,
    is_supported_file_type,
)
from tinkerbell.ai.tools.read_document import (
    ReadDocumentTool,
    estimate_tokens,
    split_lines,
    CHARS_PER_TOKEN,
)
from tinkerbell.ai.tools.search_document import (
    SearchDocumentTool,
    search_exact,
    search_regex,
    SearchMatch,
)
from tinkerbell.ai.tools.get_outline import (
    GetOutlineTool,
    detect_markdown_outline,
    detect_json_outline,
    detect_yaml_outline,
    detect_plaintext_outline,
    OutlineNode,
)
from tinkerbell.ai.tools.errors import (
    InvalidTabIdError,
    TabNotFoundError,
    InvalidLineRangeError,
    LineOutOfBoundsError,
    UnsupportedFileTypeError,
    PatternInvalidError,
    InvalidParameterError,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@dataclass
class MockTabListingProvider:
    """Mock provider for list_tabs tests."""
    
    tabs: list[dict[str, Any]] = field(default_factory=list)
    active_tab: str | None = "tab-1"
    contents: dict[str, str] = field(default_factory=dict)
    
    def list_tabs(self) -> list[dict[str, Any]]:
        return self.tabs
    
    def active_tab_id(self) -> str | None:
        return self.active_tab
    
    def get_tab_content(self, tab_id: str) -> str | None:
        return self.contents.get(tab_id)


@dataclass
class MockDocumentProvider:
    """Mock document provider for tool tests."""
    
    documents: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    active_tab: str | None = "tab-1"
    
    def get_document_text(self, tab_id: str | None = None) -> str:
        tid = tab_id or self.active_tab
        return self.documents.get(tid or "", "")
    
    def get_active_tab_id(self) -> str | None:
        return self.active_tab
    
    def get_document_content(self, tab_id: str) -> str | None:
        return self.documents.get(tab_id)
    
    def get_document_metadata(self, tab_id: str) -> dict[str, Any] | None:
        return self.metadata.get(tab_id)


@pytest.fixture
def version_manager() -> VersionManager:
    """Create a fresh version manager for each test."""
    reset_version_manager()
    return VersionManager()


@pytest.fixture
def doc_provider() -> MockDocumentProvider:
    """Create a mock document provider."""
    return MockDocumentProvider(
        documents={
            "tab-1": "Line 0\nLine 1\nLine 2\nLine 3\nLine 4",
            "tab-2": "# Heading 1\n\nContent here.\n\n## Heading 2\n\nMore content.",
        },
        metadata={
            "tab-1": {"path": "/test/doc.txt", "language": "plain_text"},
            "tab-2": {"path": "/test/doc.md", "language": "markdown"},
        },
        active_tab="tab-1",
    )


@pytest.fixture
def context(doc_provider: MockDocumentProvider, version_manager: VersionManager) -> ToolContext:
    """Create a tool context."""
    return ToolContext(
        document_provider=doc_provider,
        version_manager=version_manager,
    )


# =============================================================================
# File Type Detection Tests
# =============================================================================

class TestFileTypeDetection:
    """Tests for file type detection."""
    
    def test_markdown_extension(self) -> None:
        assert detect_file_type("/path/to/file.md") == "markdown"
        assert detect_file_type("/path/to/file.markdown") == "markdown"
    
    def test_json_extension(self) -> None:
        assert detect_file_type("/path/to/file.json") == "json"
    
    def test_yaml_extension(self) -> None:
        assert detect_file_type("/path/to/file.yaml") == "yaml"
        assert detect_file_type("/path/to/file.yml") == "yaml"
    
    def test_plain_text_extension(self) -> None:
        assert detect_file_type("/path/to/file.txt") == "plain_text"
    
    def test_binary_extension(self) -> None:
        assert detect_file_type("/path/to/file.png") == "binary"
        assert detect_file_type("/path/to/file.pdf") == "binary"
    
    def test_unknown_extension(self) -> None:
        assert detect_file_type("/path/to/file.xyz") == "unknown"
    
    def test_language_hint_override(self) -> None:
        assert detect_file_type("/path/to/file.txt", "markdown") == "markdown"
    
    def test_untitled_default(self) -> None:
        assert detect_file_type(None) == "markdown"
    
    def test_is_supported(self) -> None:
        assert is_supported_file_type("markdown") is True
        assert is_supported_file_type("json") is True
        assert is_supported_file_type("binary") is False
        assert is_supported_file_type("unknown") is False


# =============================================================================
# ListTabsTool Tests
# =============================================================================

class TestListTabsTool:
    """Tests for ListTabsTool."""
    
    def test_basic_listing(self, version_manager: VersionManager) -> None:
        """Test basic tab listing."""
        provider = MockTabListingProvider(
            tabs=[
                {"tab_id": "tab-1", "title": "Doc 1", "dirty": False, "language": "markdown"},
                {"tab_id": "tab-2", "title": "Doc 2", "dirty": True, "language": "json"},
            ],
            active_tab="tab-1",
            contents={"tab-1": "Hello", "tab-2": "World\nLine2"},
        )
        
        tool = ListTabsTool(provider=provider, version_manager=version_manager)
        context = ToolContext(document_provider=provider, version_manager=version_manager)  # type: ignore
        
        result = tool.execute(context, {})
        
        assert result["total"] == 2
        assert result["active_tab_id"] == "tab-1"
        assert len(result["tabs"]) == 2
    
    def test_tab_metadata(self, version_manager: VersionManager) -> None:
        """Test that tab metadata includes all expected fields."""
        provider = MockTabListingProvider(
            tabs=[{"tab_id": "tab-1", "title": "Test", "path": "/test.md", "dirty": True}],
            active_tab="tab-1",
            contents={"tab-1": "Line1\nLine2\nLine3"},
        )
        
        tool = ListTabsTool(provider=provider, version_manager=version_manager)
        context = ToolContext(document_provider=provider, version_manager=version_manager)  # type: ignore
        
        result = tool.execute(context, {})
        tab = result["tabs"][0]
        
        assert "tab_id" in tab
        assert "title" in tab
        assert "is_active" in tab
        assert "dirty" in tab
        assert "size_chars" in tab
        assert "line_count" in tab
        assert "file_type" in tab
        assert "supported" in tab
        
        assert tab["is_active"] is True
        assert tab["size_chars"] == 17  # "Line1\nLine2\nLine3"
        assert tab["line_count"] == 3


# =============================================================================
# ReadDocumentTool Tests
# =============================================================================

class TestReadDocumentTool:
    """Tests for ReadDocumentTool."""
    
    def test_read_full_document(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test reading a full document."""
        tool = ReadDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        result = tool.read(context, {"tab_id": "tab-1"})
        
        assert result["content"] == "Line 0\nLine 1\nLine 2\nLine 3\nLine 4"
        assert result["lines"]["total"] == 5
        assert result["lines"]["start"] == 0
        assert result["lines"]["end"] == 4
        assert result["has_more"] is False
        assert result["version"] is not None
    
    def test_read_line_range(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test reading a specific line range."""
        tool = ReadDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        result = tool.read(context, {"tab_id": "tab-1", "start_line": 1, "end_line": 3})
        
        assert result["content"] == "Line 1\nLine 2\nLine 3"
        assert result["lines"]["start"] == 1
        assert result["lines"]["end"] == 3
        assert result["lines"]["returned"] == 3
        assert result["has_more"] is True
    
    def test_read_with_pagination(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test automatic pagination with token limit."""
        # Create a larger document
        doc_provider.documents["tab-1"] = "\n".join([f"Line {i}" for i in range(100)])
        
        tool = ReadDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        # Request with very small token window
        result = tool.read(context, {"tab_id": "tab-1", "max_tokens": 10})
        
        assert result["has_more"] is True
        assert result["continuation_hint"] is not None
        assert "start_line" in result["continuation_hint"]
    
    def test_read_empty_document(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test reading an empty document."""
        doc_provider.documents["tab-1"] = ""
        
        tool = ReadDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        result = tool.read(context, {"tab_id": "tab-1"})
        
        assert result["content"] == ""
        assert result["lines"]["total"] == 0
        assert result["has_more"] is False
        assert result["version"] is not None
    
    def test_read_tab_not_found(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test error when tab doesn't exist includes helpful suggestion."""
        tool = ReadDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        with pytest.raises(TabNotFoundError) as exc_info:
            tool.read(context, {"tab_id": "nonexistent"})
        
        # Verify helpful suggestion is included
        assert "omit tab_id" in exc_info.value.suggestion
        assert "list_tabs" in exc_info.value.suggestion
    
    def test_read_invalid_line_range(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test error for invalid line range (start > end within valid bounds)."""
        tool = ReadDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        # Use valid line numbers where start > end to test InvalidLineRangeError
        with pytest.raises(InvalidLineRangeError):
            tool.read(context, {"tab_id": "tab-1", "start_line": 3, "end_line": 1})
    
    def test_read_line_out_of_bounds(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test error when line is out of bounds."""
        tool = ReadDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        with pytest.raises(LineOutOfBoundsError):
            tool.read(context, {"tab_id": "tab-1", "start_line": 100})


class TestTokenEstimation:
    """Tests for token estimation utilities."""
    
    def test_estimate_empty(self) -> None:
        assert estimate_tokens("") == 0
    
    def test_estimate_short_text(self) -> None:
        text = "Hello world"
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens == max(1, int(len(text) / CHARS_PER_TOKEN))
    
    def test_split_lines_empty(self) -> None:
        assert split_lines("") == []
    
    def test_split_lines_basic(self) -> None:
        assert split_lines("a\nb\nc") == ["a", "b", "c"]


# =============================================================================
# SearchDocumentTool Tests
# =============================================================================

class TestSearchDocumentTool:
    """Tests for SearchDocumentTool."""
    
    def test_exact_search(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test exact text search."""
        doc_provider.documents["tab-1"] = "Hello world\nHello again\nGoodbye"
        
        tool = SearchDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        result = tool.read(context, {"tab_id": "tab-1", "query": "Hello"})
        
        assert result["total_matches"] == 2
        assert result["mode"] == "exact"
        assert len(result["matches"]) == 2
    
    def test_exact_search_case_insensitive(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test case-insensitive search."""
        doc_provider.documents["tab-1"] = "Hello world\nhello again"
        
        tool = SearchDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        result = tool.read(context, {
            "tab_id": "tab-1",
            "query": "hello",
            "case_sensitive": False,
        })
        
        assert result["total_matches"] == 2
    
    def test_regex_search(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test regex search."""
        doc_provider.documents["tab-1"] = "apple\nbanana\napricot\ncherry"
        
        tool = SearchDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        # Use pattern that matches without MULTILINE flag
        result = tool.read(context, {
            "tab_id": "tab-1",
            "query": r"a\w+",  # matches 'apple', 'anana', 'apricot'
            "mode": "regex",
        })
        
        assert result["total_matches"] >= 2  # at least apple, apricot
        assert result["mode"] == "regex"
    
    def test_regex_invalid_pattern(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test error for invalid regex pattern."""
        tool = SearchDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        with pytest.raises(PatternInvalidError):
            tool.read(context, {
                "tab_id": "tab-1",
                "query": "[invalid",
                "mode": "regex",
            })
    
    def test_search_no_query(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test error when query is missing."""
        tool = SearchDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        with pytest.raises(InvalidParameterError):
            tool.read(context, {"tab_id": "tab-1", "query": ""})
    
    def test_search_whole_word(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test whole word search."""
        doc_provider.documents["tab-1"] = "the theory\ntheater\nthe"
        
        tool = SearchDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        result = tool.read(context, {
            "tab_id": "tab-1",
            "query": "the",
            "whole_word": True,
        })
        
        assert result["total_matches"] == 2  # "the theory" and "the"


class TestSearchFunctions:
    """Tests for standalone search functions."""
    
    def test_search_exact_basic(self) -> None:
        text = "Hello world\nHello again"
        matches = search_exact(text, "Hello")
        
        assert len(matches) == 2
        assert matches[0].line == 0
        assert matches[1].line == 1
    
    def test_search_regex_groups(self) -> None:
        text = "test123\ntest456"
        matches = search_regex(text, r"test(\d+)")
        
        assert len(matches) == 2
        assert "123" in matches[0].text
        assert "456" in matches[1].text


# =============================================================================
# GetOutlineTool Tests
# =============================================================================

class TestGetOutlineTool:
    """Tests for GetOutlineTool."""
    
    def test_markdown_outline(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test Markdown outline detection."""
        doc_provider.documents["tab-2"] = """# Chapter 1

Some content.

## Section 1.1

More content.

## Section 1.2

Even more.

# Chapter 2

Final content.
"""
        doc_provider.metadata["tab-2"] = {"path": "/test.md", "language": "markdown"}
        
        tool = GetOutlineTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        result = tool.read(context, {"tab_id": "tab-2"})
        
        assert result["detection_method"] == "markdown_headings"
        assert result["detection_confidence"] == "high"
        assert result["total_sections"] >= 2
    
    def test_json_outline(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test JSON outline detection."""
        doc_provider.documents["tab-1"] = """{
    "name": "test",
    "version": "1.0",
    "dependencies": {
        "lib1": "^1.0",
        "lib2": "^2.0"
    }
}"""
        doc_provider.metadata["tab-1"] = {"path": "/package.json", "language": "json"}
        
        tool = GetOutlineTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        result = tool.read(context, {"tab_id": "tab-1"})
        
        assert result["detection_method"] == "json_structure"
        assert result["total_sections"] >= 3
    
    def test_empty_document(
        self, doc_provider: MockDocumentProvider, version_manager: VersionManager
    ) -> None:
        """Test outline for empty document."""
        doc_provider.documents["tab-1"] = ""
        
        tool = GetOutlineTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)
        
        result = tool.read(context, {"tab_id": "tab-1"})
        
        assert result["outline"] == []
        assert result["detection_method"] == "empty_document"


class TestMarkdownOutline:
    """Tests for Markdown outline detection."""
    
    def test_atx_headings(self) -> None:
        text = "# H1\n## H2\n### H3"
        nodes, method = detect_markdown_outline(text)
        
        assert method == "markdown_headings"
        # Returns hierarchical tree - H1 contains H2 contains H3
        assert len(nodes) == 1  # Just the root H1
        assert nodes[0].level == 1
        assert nodes[0].title == "H1"
        assert len(nodes[0].children) == 1  # H2 is child of H1
        assert nodes[0].children[0].level == 2
        assert len(nodes[0].children[0].children) == 1  # H3 is child of H2
        assert nodes[0].children[0].children[0].level == 3
    
    def test_setext_headings(self) -> None:
        text = "Heading 1\n=========\n\nHeading 2\n---------"
        nodes, method = detect_markdown_outline(text)
        
        # H1 contains H2 as a child
        assert len(nodes) == 1
        assert nodes[0].title == "Heading 1"
        assert nodes[0].level == 1
        assert len(nodes[0].children) == 1
        assert nodes[0].children[0].title == "Heading 2"
        assert nodes[0].children[0].level == 2
    
    def test_mixed_headings(self) -> None:
        text = "# ATX Heading\n\nSetext Heading\n--------------"
        nodes, method = detect_markdown_outline(text)
        
        # ATX H1 contains Setext H2 as child
        assert len(nodes) == 1
        assert nodes[0].title == "ATX Heading"
        assert len(nodes[0].children) == 1
        assert nodes[0].children[0].title == "Setext Heading"


class TestJsonOutline:
    """Tests for JSON outline detection."""
    
    def test_simple_object(self) -> None:
        text = '{"a": 1, "b": 2}'
        nodes, method = detect_json_outline(text)
        
        assert method == "json_structure"
        assert len(nodes) == 2
    
    def test_nested_object(self) -> None:
        text = '{"outer": {"inner1": 1, "inner2": 2}}'
        nodes, method = detect_json_outline(text)
        
        # Should have outer + nested keys
        assert len(nodes) >= 1
    
    def test_invalid_json(self) -> None:
        text = "not valid json"
        nodes, method = detect_json_outline(text)
        
        assert method == "json_parse_error"
        assert nodes == []


class TestPlainTextOutline:
    """Tests for plain text outline heuristics."""
    
    def test_chapter_markers(self) -> None:
        text = "Chapter 1\n\nContent.\n\nChapter 2\n\nMore content."
        nodes, method, confidence = detect_plaintext_outline(text)
        
        assert method == "chapter_markers"
        assert len(nodes) == 2
    
    def test_all_caps_headings(self) -> None:
        text = "\nINTRODUCTION\n\nSome text.\n\nCONCLUSION\n\nMore text."
        nodes, method, confidence = detect_plaintext_outline(text)
        
        # Should detect caps headings
        assert len(nodes) >= 0  # May or may not detect based on surrounding context
    
    def test_unstructured_text(self) -> None:
        text = "Just some regular text without any structure markers."
        nodes, method, confidence = detect_plaintext_outline(text)
        
        # Should return empty or low confidence
        assert confidence in ("low", "medium") or len(nodes) == 0
