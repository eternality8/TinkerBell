"""Tests for DocumentInsertTool."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from tinkerbell.ai.tools.document_insert import DocumentInsertTool
from tinkerbell.ai.tools.document_apply_patch import DocumentApplyPatchTool
from tinkerbell.ai.tools.document_edit import DocumentEditTool
from tinkerbell.chat.message_model import EditDirective


class _InsertBridgeStub:
    """Stub bridge for testing DocumentInsertTool."""

    def __init__(self, *, text: str = "Line 0\nLine 1\nLine 2\n", version: str = "v1") -> None:
        self._text = text
        self.snapshot = {
            "text": text,
            "version": version,
            "version_id": 1,
            "document_id": "doc-insert",
            "path": "doc.md",
            "content_hash": "hash-1",
            "length": len(text),
            "window": {"start": 0, "end": len(text)},
            "text_range": {"start": 0, "end": len(text)},
            "line_start_offsets": self._build_line_start_offsets(text),
        }
        self.calls: list[dict[str, Any]] = []
        self.snapshot_requests: list[dict[str, Any]] = []

    def queue_edit(
        self,
        directive: EditDirective | Mapping[str, Any],
        *,
        tab_id: str | None = None,
    ) -> None:
        if isinstance(directive, Mapping):
            payload = dict(directive)
        else:
            payload = {
                "action": directive.action,
                "content": directive.content,
                "target_range": directive.target_range.to_tuple(),
                "diff": directive.diff,
            }
        payload["tab_id"] = tab_id
        self.calls.append(payload)

    def generate_snapshot(
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_text: bool = True,
        window: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        assert delta_only is False
        self.snapshot_requests.append({
            "tab_id": tab_id,
            "include_text": include_text,
            "window": dict(window) if isinstance(window, Mapping) else None,
        })
        snapshot = dict(self.snapshot)
        text = snapshot.get("text", "")
        doc_text = text if isinstance(text, str) else ""
        length = len(doc_text)
        snapshot["length"] = length
        snapshot["line_start_offsets"] = self._build_line_start_offsets(self._text)
        if isinstance(window, Mapping):
            start = max(0, min(int(window.get("start", 0)), length))
            end = max(start, min(int(window.get("end", length)), length))
            snapshot["window"] = {"start": start, "end": end}
            if include_text:
                snapshot["text"] = doc_text[start:end]
            else:
                snapshot["text"] = ""
        elif not include_text:
            snapshot["text"] = ""
        else:
            snapshot.setdefault("window", {"start": 0, "end": length})
        return snapshot

    @staticmethod
    def _build_line_start_offsets(text: str) -> list[int]:
        offsets = [0]
        cursor = 0
        for segment in text.splitlines(keepends=True):
            cursor += len(segment)
            offsets.append(cursor)
        if offsets[-1] < len(text):
            offsets.append(len(text))
        return offsets

    def get_last_diff_summary(self, tab_id: str | None = None) -> str | None:
        return None

    def get_last_snapshot_version(self, tab_id: str | None = None) -> str | None:
        return self.snapshot.get("version")


def _make_insert_tool(text: str) -> tuple[DocumentInsertTool, _InsertBridgeStub]:
    """Create an insert tool with the given document text."""
    bridge = _InsertBridgeStub(text=text)
    edit_tool = DocumentEditTool(bridge=bridge)
    patch_tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)
    insert_tool = DocumentInsertTool(patch_tool=patch_tool)
    return insert_tool, bridge


@pytest.fixture()
def insert_tool() -> tuple[DocumentInsertTool, _InsertBridgeStub]:
    """Fixture providing an insert tool with default document."""
    return _make_insert_tool("Line 0\nLine 1\nLine 2\n")


class TestDocumentInsertBasic:
    """Basic insertion tests."""

    def test_insert_after_first_line(self, insert_tool):
        """Insert content after line 0."""
        tool, bridge = insert_tool

        status = tool.run(
            snapshot_token="doc-insert:1",
            after_line=0,
            content="New line",
        )

        assert "applied" in status or "queued" in status
        assert bridge.calls, "Expected edit to be queued"
        payload = bridge.calls[-1]
        # The insertion should be at offset 7 (after "Line 0\n")
        ranges = payload["ranges"]
        assert len(ranges) == 1
        assert ranges[0]["start"] == 7
        assert ranges[0]["end"] == 7  # Zero-width = insertion
        assert "New line" in ranges[0]["replacement"]

    def test_insert_at_beginning(self, insert_tool):
        """Insert content at the very beginning using after_line=-1."""
        tool, bridge = insert_tool

        status = tool.run(
            snapshot_token="doc-insert:1",
            after_line=-1,
            content="First line",
        )

        assert "applied" in status or "queued" in status
        payload = bridge.calls[-1]
        ranges = payload["ranges"]
        assert ranges[0]["start"] == 0
        assert ranges[0]["end"] == 0

    def test_insert_at_end(self):
        """Insert content at the end of the document."""
        tool, bridge = _make_insert_tool("Line 0\nLine 1\n")

        status = tool.run(
            snapshot_token="doc-insert:1",
            after_line=1,  # After the last line
            content="Final line",
        )

        assert "applied" in status or "queued" in status
        payload = bridge.calls[-1]
        ranges = payload["ranges"]
        # Should insert at the end
        assert ranges[0]["start"] == len("Line 0\nLine 1\n")

    def test_insert_includes_newline_when_needed(self, insert_tool):
        """Inserted content should get a trailing newline for proper separation."""
        tool, bridge = insert_tool

        tool.run(
            snapshot_token="doc-insert:1",
            after_line=0,
            content="No newline",
        )

        payload = bridge.calls[-1]
        replacement = payload["ranges"][0]["replacement"]
        assert replacement.endswith("\n"), "Inserted content should end with newline"

    def test_insert_preserves_existing_newline(self, insert_tool):
        """Content that already has a newline should not get another."""
        tool, bridge = insert_tool

        tool.run(
            snapshot_token="doc-insert:1",
            after_line=0,
            content="Has newline\n",
        )

        payload = bridge.calls[-1]
        replacement = payload["ranges"][0]["replacement"]
        assert replacement == "Has newline\n"
        assert not replacement.endswith("\n\n")


class TestDocumentInsertValidation:
    """Validation and error handling tests."""

    def test_insert_rejects_invalid_line_number(self, insert_tool):
        """Should reject line numbers below -1."""
        tool, bridge = insert_tool

        with pytest.raises(ValueError, match="after_line must be >= -1"):
            tool.run(
                snapshot_token="doc-insert:1",
                after_line=-2,
                content="Bad",
            )

        assert not bridge.calls

    def test_insert_rejects_out_of_bounds_line(self, insert_tool):
        """Should reject line numbers beyond document length."""
        tool, bridge = insert_tool

        with pytest.raises(ValueError, match="exceeds document line count"):
            tool.run(
                snapshot_token="doc-insert:1",
                after_line=100,
                content="Out of bounds",
            )

        assert not bridge.calls

    def test_insert_requires_valid_snapshot_token(self, insert_tool):
        """Should reject malformed snapshot_token."""
        tool, _ = insert_tool

        with pytest.raises(Exception):  # InvalidSnapshotTokenError
            tool.run(
                snapshot_token="invalid-token-format",
                after_line=0,
                content="Bad token",
            )

    def test_insert_detects_document_id_used_instead_of_tab_id(self):
        """Should give helpful error when AI uses document_id instead of tab_id."""

        class _BridgeWithDifferentIds(_InsertBridgeStub):
            """Bridge that simulates document_id != tab_id scenario."""

            def __init__(self) -> None:
                super().__init__(text="Line 0\nLine 1\n")
                # Set different document_id and tab_id
                self.snapshot["document_id"] = "doc-id-12345"
                self.snapshot["tab_id"] = "tab-id-67890"

        bridge = _BridgeWithDifferentIds()
        edit_tool = DocumentEditTool(bridge=bridge)
        patch_tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)
        tool = DocumentInsertTool(patch_tool=patch_tool)

        # AI mistakenly uses document_id in the snapshot_token
        with pytest.raises(ValueError, match="document_id.*instead of.*tab_id"):
            tool.run(
                snapshot_token="doc-id-12345:1",  # Wrong! Should be tab-id-67890:1
                after_line=0,
                content="Test",
            )


class TestDocumentInsertMetadata:
    """Tests for metadata and telemetry."""

    def test_insert_includes_operation_metadata(self, insert_tool):
        """Inserted edits should include operation metadata."""
        tool, bridge = insert_tool

        tool.run(
            snapshot_token="doc-insert:1",
            after_line=1,
            content="Middle content",
        )

        payload = bridge.calls[-1]
        metadata = payload.get("metadata", {})
        assert metadata.get("operation") == "insert"
        assert "after_line" in metadata
        assert metadata["after_line"] == 1

    def test_insert_includes_rationale(self, insert_tool):
        """Rationale should be passed through to the edit."""
        tool, bridge = insert_tool

        tool.run(
            snapshot_token="doc-insert:1",
            after_line=0,
            content="With reason",
            rationale="Adding new paragraph",
        )

        payload = bridge.calls[-1]
        assert payload.get("rationale") == "Adding new paragraph"

    def test_insert_includes_scope_metadata(self, insert_tool):
        """Range entries must include scope metadata for edit tool."""
        tool, bridge = insert_tool

        tool.run(
            snapshot_token="doc-insert:1",
            after_line=0,
            content="Scoped insert",
        )

        payload = bridge.calls[-1]
        ranges = payload["ranges"]
        assert len(ranges) == 1
        
        # Verify scope is present in the range entry
        range_entry = ranges[0]
        assert "scope" in range_entry, "Range must include scope dict"
        assert range_entry["scope"]["origin"] == "insert"
        assert "scope_origin" in range_entry
        assert range_entry["scope_origin"] == "insert"


class TestDocumentInsertZeroWidth:
    """Tests verifying zero-width range semantics for pure insertion."""

    def test_insert_creates_zero_width_range(self, insert_tool):
        """Insertion must use zero-width range (start == end) to avoid overwriting."""
        tool, bridge = insert_tool

        tool.run(
            snapshot_token="doc-insert:1",
            after_line=0,
            content="Inserted",
        )

        payload = bridge.calls[-1]
        ranges = payload["ranges"]
        assert len(ranges) == 1
        assert ranges[0]["start"] == ranges[0]["end"], (
            "Insert must use zero-width range to avoid overwriting"
        )

    def test_insert_does_not_include_existing_content_in_match_text(self, insert_tool):
        """The match_text should be empty for insertions."""
        tool, bridge = insert_tool

        tool.run(
            snapshot_token="doc-insert:1",
            after_line=0,
            content="Pure insert",
        )

        payload = bridge.calls[-1]
        ranges = payload["ranges"]
        assert ranges[0]["match_text"] == "", (
            "Insert should have empty match_text since nothing is being replaced"
        )


class TestDocumentInsertMultiline:
    """Tests for multi-line content insertion."""

    def test_insert_multiline_content(self, insert_tool):
        """Should handle multi-line content correctly."""
        tool, bridge = insert_tool
        multiline = "First inserted\nSecond inserted\nThird inserted"

        status = tool.run(
            snapshot_token="doc-insert:1",
            after_line=0,
            content=multiline,
        )

        assert "applied" in status or "queued" in status
        # Verify the content was properly passed through
        payload = bridge.calls[-1]
        assert "First inserted" in payload["ranges"][0]["replacement"]

    def test_insert_empty_document(self):
        """Should handle insertion into an empty document."""
        tool, bridge = _make_insert_tool("")

        status = tool.run(
            snapshot_token="doc-insert:1",
            after_line=-1,
            content="First content",
        )

        assert "applied" in status or "queued" in status
        payload = bridge.calls[-1]
        ranges = payload["ranges"]
        assert ranges[0]["start"] == 0
        assert ranges[0]["end"] == 0
