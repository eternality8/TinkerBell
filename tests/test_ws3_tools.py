"""Tests for Workstream 3: Writing Tools.

Tests for create_document, insert_lines, replace_lines, delete_lines,
write_document, and find_and_replace tools.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest

from tinkerbell.ai.tools import (
    # WS3 Tools
    CreateDocumentTool,
    InsertLinesTool,
    ReplaceLinesTool,
    DeleteLinesTool,
    WriteDocumentTool,
    FindAndReplaceTool,
    # Supporting classes
    VersionManager,
    ToolContext,
    # Errors
    InvalidParameterError,
    MissingParameterError,
    InvalidLineRangeError,
    LineOutOfBoundsError,
    ContentRequiredError,
    NoMatchesError,
    TooManyMatchesError,
    PatternInvalidError,
    TabNotFoundError,
    TitleExistsError,
    # Helpers
    find_anchor_text,
    find_matches_literal,
    find_matches_regex,
    apply_replacements,
    infer_file_type,
    suggest_extension,
)
from tinkerbell.ai.tools.version import compute_content_hash


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


class MockDocumentProvider:
    """Mock document provider for testing."""

    def __init__(self) -> None:
        self.documents: dict[str, str] = {
            "tab-1": "Line 0\nLine 1\nLine 2\nLine 3\nLine 4",
            "tab-2": "",
        }
        self.active_tab = "tab-1"
        self.metadata: dict[str, dict[str, Any]] = {
            "tab-1": {"path": "/test/doc.md", "language": "markdown"},
            "tab-2": {"path": "/test/empty.txt", "language": "plain"},
        }

    def get_document_text(self, tab_id: str | None = None) -> str:
        tid = tab_id or self.active_tab
        return self.documents.get(tid, "")

    def get_active_tab_id(self) -> str | None:
        return self.active_tab

    def get_document_content(self, tab_id: str) -> str | None:
        return self.documents.get(tab_id)

    def get_document_metadata(self, tab_id: str) -> dict[str, Any] | None:
        return self.metadata.get(tab_id)


class MockDocumentEditor:
    """Mock document editor for testing."""

    def __init__(self, provider: MockDocumentProvider) -> None:
        self.provider = provider
        self.edit_history: list[tuple[str, str]] = []

    def set_document_text(self, tab_id: str, new_text: str) -> None:
        self.edit_history.append((tab_id, new_text))
        self.provider.documents[tab_id] = new_text


class MockDocumentCreator:
    """Mock document creator for testing."""

    def __init__(self, provider: MockDocumentProvider) -> None:
        self.provider = provider
        self.create_count = 0

    def create_document(
        self,
        title: str,
        content: str = "",
        file_type: str | None = None,
    ) -> str:
        # Check for existing title
        for tab_id, meta in self.provider.metadata.items():
            if meta.get("title") == title or meta.get("path", "").endswith(title):
                raise TitleExistsError(title=title, existing_tab_id=tab_id)

        self.create_count += 1
        tab_id = f"tab-new-{self.create_count}"
        self.provider.documents[tab_id] = content
        self.provider.metadata[tab_id] = {
            "title": title,
            "path": f"/test/{title}",
            "language": file_type or "plain",
        }
        return tab_id

    def document_exists(self, title: str) -> tuple[bool, str | None]:
        for tab_id, meta in self.provider.metadata.items():
            if meta.get("title") == title or meta.get("path", "").endswith(title):
                return True, tab_id
        return False, None


@pytest.fixture
def doc_provider() -> MockDocumentProvider:
    return MockDocumentProvider()


@pytest.fixture
def doc_editor(doc_provider: MockDocumentProvider) -> MockDocumentEditor:
    return MockDocumentEditor(doc_provider)


@pytest.fixture
def doc_creator(doc_provider: MockDocumentProvider) -> MockDocumentCreator:
    return MockDocumentCreator(doc_provider)


@pytest.fixture
def version_manager() -> VersionManager:
    vm = VersionManager()
    # Register and initialize versions for test tabs
    vm.register_tab("tab-1", "doc-1", compute_content_hash("Line 0\nLine 1\nLine 2\nLine 3\nLine 4"))
    vm.register_tab("tab-2", "doc-2", compute_content_hash(""))
    return vm


# -----------------------------------------------------------------------------
# Test: File Type Inference
# -----------------------------------------------------------------------------


class TestFileTypeInference:
    """Tests for file type inference helpers."""

    def test_infer_from_extension(self) -> None:
        assert infer_file_type("document.md") == "markdown"
        assert infer_file_type("data.json") == "json"
        assert infer_file_type("config.yaml") == "yaml"
        assert infer_file_type("notes.txt") == "plain_text"
        assert infer_file_type("unknown.xyz") == "plain_text"

    def test_infer_with_hint_override(self) -> None:
        assert infer_file_type("document.txt", "markdown") == "markdown"
        assert infer_file_type("data.json", "YAML") == "yaml"

    def test_suggest_extension(self) -> None:
        assert suggest_extension("markdown") == ".md"
        assert suggest_extension("json") == ".json"
        assert suggest_extension("unknown") == ".txt"


# -----------------------------------------------------------------------------
# Test: CreateDocumentTool
# -----------------------------------------------------------------------------


class TestCreateDocumentTool:
    """Tests for CreateDocumentTool."""

    def test_create_basic_document(
        self,
        doc_provider: MockDocumentProvider,
        doc_creator: MockDocumentCreator,
        version_manager: VersionManager,
    ) -> None:
        """Test creating a basic document."""
        tool = CreateDocumentTool(
            version_manager=version_manager,
            document_creator=doc_creator,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        result = tool.run(context, {"title": "new_doc.md"})

        assert result.success
        assert result.data is not None
        assert "tab_id" in result.data
        assert "version" in result.data
        assert result.data["file_type"] == "markdown"

    def test_create_with_content(
        self,
        doc_provider: MockDocumentProvider,
        doc_creator: MockDocumentCreator,
        version_manager: VersionManager,
    ) -> None:
        """Test creating a document with initial content."""
        tool = CreateDocumentTool(
            version_manager=version_manager,
            document_creator=doc_creator,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        content = "# Title\n\nParagraph text."
        result = tool.run(context, {"title": "content.md", "content": content})

        assert result.success
        assert result.data is not None
        assert result.data["lines"] == 3
        assert result.data["size_chars"] == len(content)

    def test_create_missing_title(
        self,
        doc_provider: MockDocumentProvider,
        version_manager: VersionManager,
    ) -> None:
        """Test error when title is missing."""
        tool = CreateDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        result = tool.run(context, {})

        assert not result.success
        assert result.error is not None
        assert "title" in result.error.message.lower()

    def test_create_invalid_title_chars(
        self,
        doc_provider: MockDocumentProvider,
        version_manager: VersionManager,
    ) -> None:
        """Test error for invalid characters in title."""
        tool = CreateDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        result = tool.run(context, {"title": "invalid<>name.txt"})

        assert not result.success
        assert result.error is not None


# -----------------------------------------------------------------------------
# Test: InsertLinesTool
# -----------------------------------------------------------------------------


class TestInsertLinesTool:
    """Tests for InsertLinesTool."""

    def test_insert_at_beginning(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test inserting at the beginning of document."""
        tool = InsertLinesTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "after_line": -1,
            "content": "New first line",
        })

        assert result.success
        assert result.data is not None
        assert result.data["inserted_at"]["after_line"] == -1
        assert result.data["inserted_at"]["lines_added"] == 1

    def test_insert_in_middle(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test inserting in the middle of document."""
        tool = InsertLinesTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "after_line": 2,
            "content": "Inserted line",
        })

        assert result.success
        assert result.data is not None
        assert result.data["inserted_at"]["after_line"] == 2
        # Check that insertion happened
        assert "Inserted line" in doc_provider.documents["tab-1"]

    def test_insert_multiline(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test inserting multiple lines."""
        tool = InsertLinesTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "after_line": 0,
            "content": "New A\nNew B\nNew C",
        })

        assert result.success
        assert result.data is not None
        assert result.data["inserted_at"]["lines_added"] == 3

    def test_insert_dry_run(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test dry run doesn't modify document."""
        original_content = doc_provider.documents["tab-1"]
        tool = InsertLinesTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "after_line": 0,
            "content": "Should not appear",
            "dry_run": True,
        })

        assert result.success
        assert result.data is not None
        assert result.data.get("dry_run") is True
        # Document should be unchanged
        assert doc_provider.documents["tab-1"] == original_content

    def test_insert_missing_content(
        self,
        doc_provider: MockDocumentProvider,
        version_manager: VersionManager,
    ) -> None:
        """Test error when content is missing."""
        tool = InsertLinesTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "after_line": 0,
        })

        assert not result.success


# -----------------------------------------------------------------------------
# Test: ReplaceLinesTool
# -----------------------------------------------------------------------------


class TestReplaceLinesTool:
    """Tests for ReplaceLinesTool."""

    def test_replace_single_line(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test replacing a single line."""
        tool = ReplaceLinesTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "start_line": 1,
            "end_line": 1,
            "content": "Replaced line 1",
        })

        assert result.success
        assert result.data is not None
        assert result.data["lines_affected"]["removed"] == 1
        assert result.data["lines_affected"]["added"] == 1
        assert "Replaced line 1" in doc_provider.documents["tab-1"]

    def test_replace_multiple_lines(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test replacing multiple lines."""
        tool = ReplaceLinesTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "start_line": 1,
            "end_line": 3,
            "content": "Single replacement",
        })

        assert result.success
        assert result.data is not None
        assert result.data["lines_affected"]["removed"] == 3
        assert result.data["lines_affected"]["added"] == 1
        assert result.data["lines_affected"]["net_change"] == -2

    def test_replace_with_empty_deletes(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test replacing with empty content effectively deletes."""
        tool = ReplaceLinesTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "start_line": 1,
            "end_line": 1,
            "content": "",
        })

        assert result.success
        assert result.data is not None
        assert result.data["lines_affected"]["removed"] == 1
        assert result.data["lines_affected"]["added"] == 0

    def test_replace_invalid_range(
        self,
        doc_provider: MockDocumentProvider,
        version_manager: VersionManager,
    ) -> None:
        """Test error for invalid line range."""
        tool = ReplaceLinesTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "start_line": 3,
            "end_line": 1,  # end < start
            "content": "test",
        })

        assert not result.success


# -----------------------------------------------------------------------------
# Test: DeleteLinesTool
# -----------------------------------------------------------------------------


class TestDeleteLinesTool:
    """Tests for DeleteLinesTool."""

    def test_delete_single_line(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test deleting a single line."""
        tool = DeleteLinesTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "start_line": 2,
            "end_line": 2,
        })

        assert result.success
        assert result.data is not None
        assert result.data["lines_deleted"] == 1
        assert "Line 2" in result.data["deleted_content"]

    def test_delete_multiple_lines(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test deleting multiple lines."""
        tool = DeleteLinesTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "start_line": 1,
            "end_line": 3,
        })

        assert result.success
        assert result.data is not None
        assert result.data["lines_deleted"] == 3

    def test_delete_from_empty_document(
        self,
        doc_provider: MockDocumentProvider,
        version_manager: VersionManager,
    ) -> None:
        """Test error when deleting from empty document."""
        tool = DeleteLinesTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-2")
        result = tool.run(context, {
            "tab_id": "tab-2",
            "version_token": token.to_string(),
            "start_line": 0,
            "end_line": 0,
        })

        assert not result.success


# -----------------------------------------------------------------------------
# Test: WriteDocumentTool
# -----------------------------------------------------------------------------


class TestWriteDocumentTool:
    """Tests for WriteDocumentTool."""

    def test_write_full_document(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test full document replacement."""
        tool = WriteDocumentTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        new_content = "Completely new content\nWith two lines"
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "content": new_content,
        })

        assert result.success
        assert result.data is not None
        assert result.data["lines_affected"]["previous"] == 5
        assert result.data["lines_affected"]["current"] == 2
        assert doc_provider.documents["tab-1"] == new_content

    def test_write_empty_content(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test clearing document content."""
        tool = WriteDocumentTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "content": "",
        })

        assert result.success
        assert result.data is not None
        assert result.data["lines_affected"]["current"] == 0

    def test_write_missing_content(
        self,
        doc_provider: MockDocumentProvider,
        version_manager: VersionManager,
    ) -> None:
        """Test error when content is missing."""
        tool = WriteDocumentTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
        })

        assert not result.success


# -----------------------------------------------------------------------------
# Test: FindAndReplaceTool
# -----------------------------------------------------------------------------


class TestFindAndReplaceTool:
    """Tests for FindAndReplaceTool."""

    def test_literal_replace(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test literal find and replace."""
        tool = FindAndReplaceTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "find": "Line",
            "replace": "Row",
        })

        assert result.success
        assert result.data is not None
        assert result.data["matches_found"] == 5
        assert result.data["replacements_made"] == 5
        assert "Row 0" in doc_provider.documents["tab-1"]

    def test_regex_replace(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test regex find and replace."""
        tool = FindAndReplaceTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "find": r"Line (\d)",
            "replace": r"Row \1",
            "mode": "regex",
        })

        assert result.success
        assert result.data is not None
        assert result.data["replacements_made"] == 5
        assert "Row 0" in doc_provider.documents["tab-1"]

    def test_preview_mode(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test preview mode doesn't modify document."""
        original_content = doc_provider.documents["tab-1"]
        tool = FindAndReplaceTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "find": "Line",
            "replace": "Row",
            "preview": True,
        })

        assert result.success
        assert result.data is not None
        assert result.data["replacements_made"] == 0
        assert result.data["matches_found"] == 5
        # Document should be unchanged
        assert doc_provider.documents["tab-1"] == original_content

    def test_case_insensitive(
        self,
        doc_provider: MockDocumentProvider,
        doc_editor: MockDocumentEditor,
        version_manager: VersionManager,
    ) -> None:
        """Test case-insensitive search."""
        doc_provider.documents["tab-1"] = "LINE 0\nline 1\nLine 2"
        version_manager.increment_version("tab-1", compute_content_hash(doc_provider.documents["tab-1"]))

        tool = FindAndReplaceTool(
            version_manager=version_manager,
            document_editor=doc_editor,
        )
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "find": "line",
            "replace": "ROW",
            "case_sensitive": False,
        })

        assert result.success
        assert result.data is not None
        assert result.data["matches_found"] == 3

    def test_no_matches_error(
        self,
        doc_provider: MockDocumentProvider,
        version_manager: VersionManager,
    ) -> None:
        """Test error when no matches found."""
        tool = FindAndReplaceTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "find": "NOTFOUND",
            "replace": "whatever",
        })

        assert not result.success

    def test_invalid_regex(
        self,
        doc_provider: MockDocumentProvider,
        version_manager: VersionManager,
    ) -> None:
        """Test error for invalid regex pattern."""
        tool = FindAndReplaceTool(version_manager=version_manager)
        context = ToolContext(document_provider=doc_provider, version_manager=version_manager)

        token = version_manager.get_current_token("tab-1")
        result = tool.run(context, {
            "tab_id": "tab-1",
            "version_token": token.to_string(),
            "find": "[invalid",
            "replace": "test",
            "mode": "regex",
        })

        assert not result.success


# -----------------------------------------------------------------------------
# Test: Drift Recovery
# -----------------------------------------------------------------------------


class TestDriftRecovery:
    """Tests for drift recovery functionality."""

    def test_find_anchor_exact_match(self) -> None:
        """Test finding anchor text with exact match."""
        lines = ["Line 0", "Target line", "Line 2"]
        found_line, confidence = find_anchor_text(lines, "Target line", expected_line=1)
        assert found_line == 1
        assert confidence == 1.0

    def test_find_anchor_with_drift(self) -> None:
        """Test finding anchor text when lines have shifted."""
        lines = ["New line", "Line 0", "Target line", "Line 2"]
        found_line, confidence = find_anchor_text(lines, "Target line", expected_line=1)
        assert found_line == 2  # Drifted by 1
        assert confidence > 0.5

    def test_find_anchor_not_found(self) -> None:
        """Test error when anchor text not found."""
        lines = ["Line 0", "Line 1", "Line 2"]
        with pytest.raises(NoMatchesError):
            find_anchor_text(lines, "Not here")

    def test_find_anchor_multiple_matches(self) -> None:
        """Test error when multiple matches with similar confidence."""
        lines = ["Target", "Target", "Target"]
        with pytest.raises(TooManyMatchesError):
            find_anchor_text(lines, "Target")


# -----------------------------------------------------------------------------
# Test: Search Helpers
# -----------------------------------------------------------------------------


class TestSearchHelpers:
    """Tests for search helper functions."""

    def test_find_matches_literal_basic(self) -> None:
        """Test basic literal matching."""
        text = "apple banana apple cherry"
        matches = find_matches_literal(text, "apple")
        assert len(matches) == 2

    def test_find_matches_literal_whole_word(self) -> None:
        """Test whole word matching."""
        text = "apple pineapple apple"
        matches = find_matches_literal(text, "apple", whole_word=True)
        assert len(matches) == 2  # Not pineapple

    def test_find_matches_regex_basic(self) -> None:
        """Test basic regex matching."""
        text = "Line 1\nLine 2\nLine 3"
        matches = find_matches_regex(text, r"Line \d")
        assert len(matches) == 3

    def test_apply_replacements(self) -> None:
        """Test applying replacements to text."""
        text = "Hello world, hello everyone"
        matches = [(0, 0, 5), (0, 13, 18)]  # "Hello", "hello"
        new_text, details = apply_replacements(text, matches, "Hi")
        assert new_text == "Hi world, Hi everyone"
        assert len(details) == 2
