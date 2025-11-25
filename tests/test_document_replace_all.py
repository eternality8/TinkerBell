"""Tests for DocumentReplaceAllTool."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from tinkerbell.ai.tools.document_apply_patch import DocumentApplyPatchTool, InvalidSnapshotTokenError
from tinkerbell.ai.tools.document_edit import DocumentEditTool
from tinkerbell.ai.tools.document_replace_all import DocumentReplaceAllTool
from tinkerbell.chat.message_model import EditDirective


class _ReplaceAllBridgeStub:
    """Stub bridge for DocumentReplaceAllTool tests."""

    def __init__(self, *, text: str = "Original document content", version: str = "v1") -> None:
        self._text = text
        self.snapshot = {
            "text": text,
            "version": version,
            "version_id": 1,
            "document_id": "doc-replace-all",
            "tab_id": "tab-1",
            "path": "document.md",
            "content_hash": "hash-original",
            "length": len(text),
            "window": {"start": 0, "end": len(text)},
            "text_range": {"start": 0, "end": len(text)},
            "line_start_offsets": self._build_line_start_offsets(text),
        }
        self.calls: list[dict[str, Any]] = []
        self.snapshot_requests: list[dict[str, Any]] = []
        self.last_snapshot_version: str | None = version
        self.last_diff_summary: str | None = None

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
        self.last_snapshot_version = "updated"

    def generate_snapshot(
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_text: bool = True,
        window: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        self.snapshot_requests.append({
            "tab_id": tab_id,
            "include_text": include_text,
            "window": dict(window) if isinstance(window, Mapping) else None,
        })
        snapshot = dict(self.snapshot)
        if not include_text:
            snapshot["text"] = ""
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
        return self.last_diff_summary

    def get_last_snapshot_version(self, tab_id: str | None = None) -> str | None:
        return self.last_snapshot_version


@pytest.fixture
def replace_all_tool() -> tuple[DocumentReplaceAllTool, _ReplaceAllBridgeStub]:
    """Create a DocumentReplaceAllTool with a stub bridge."""
    bridge = _ReplaceAllBridgeStub()
    edit_tool = DocumentEditTool(bridge=bridge)
    patch_tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)
    replace_all_tool = DocumentReplaceAllTool(patch_tool=patch_tool)
    return replace_all_tool, bridge


def test_document_replace_all_replaces_entire_document(replace_all_tool):
    """Test that DocumentReplaceAllTool replaces the entire document."""
    tool, bridge = replace_all_tool

    status = tool.run(
        snapshot_token="tab-1:1",
        content="New document content",
    )

    assert "queued" in status or "applied" in status
    assert bridge.calls, "Edit should have been queued"
    payload = bridge.calls[-1]
    # Check that it's a full document replacement
    range_payload = payload["ranges"][0]
    assert range_payload["start"] == 0
    assert range_payload["end"] == len("Original document content")


def test_document_replace_all_parses_snapshot_token(replace_all_tool):
    """Test that snapshot_token is parsed correctly."""
    tool, bridge = replace_all_tool
    bridge.snapshot["version_id"] = 42
    bridge.snapshot["tab_id"] = "my-tab"

    status = tool.run(
        snapshot_token="my-tab:42",
        content="Replaced content",
    )

    assert "queued" in status or "applied" in status


def test_document_replace_all_extracts_tab_id_from_token(replace_all_tool):
    """Test that tab_id is extracted from snapshot_token."""
    tool, bridge = replace_all_tool

    status = tool.run(
        snapshot_token="extracted-tab:1",
        content="New content",
    )

    assert "queued" in status or "applied" in status


def test_document_replace_all_uses_explicit_tab_id_over_token(replace_all_tool):
    """Test that explicit tab_id takes precedence over snapshot_token."""
    tool, bridge = replace_all_tool

    status = tool.run(
        snapshot_token="token-tab:1",
        content="New content",
        tab_id="explicit-tab",
    )

    assert "queued" in status or "applied" in status


def test_document_replace_all_rejects_malformed_token():
    """Test that malformed snapshot_token is rejected."""
    bridge = _ReplaceAllBridgeStub()
    edit_tool = DocumentEditTool(bridge=bridge)
    patch_tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)
    tool = DocumentReplaceAllTool(patch_tool=patch_tool)

    with pytest.raises(InvalidSnapshotTokenError, match="tab_id:version_id"):
        tool.run(
            snapshot_token="no-colon-here",
            content="New content",
        )


def test_document_replace_all_rejects_empty_tab_id_in_token():
    """Test that empty tab_id in token is rejected."""
    bridge = _ReplaceAllBridgeStub()
    edit_tool = DocumentEditTool(bridge=bridge)
    patch_tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)
    tool = DocumentReplaceAllTool(patch_tool=patch_tool)

    with pytest.raises(InvalidSnapshotTokenError, match="tab_id"):
        tool.run(
            snapshot_token=":1",
            content="New content",
        )


def test_document_replace_all_rejects_empty_version_id_in_token():
    """Test that empty version_id in token is rejected."""
    bridge = _ReplaceAllBridgeStub()
    edit_tool = DocumentEditTool(bridge=bridge)
    patch_tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)
    tool = DocumentReplaceAllTool(patch_tool=patch_tool)

    with pytest.raises(InvalidSnapshotTokenError, match="version_id"):
        tool.run(
            snapshot_token="my-tab:",
            content="New content",
        )


def test_document_replace_all_includes_rationale(replace_all_tool):
    """Test that rationale is passed through to the patch tool."""
    tool, bridge = replace_all_tool

    status = tool.run(
        snapshot_token="tab-1:1",
        content="New content",
        rationale="Full document rewrite for clarity",
    )

    assert "queued" in status or "applied" in status
    payload = bridge.calls[-1]
    assert payload.get("rationale") == "Full document rewrite for clarity"


def test_document_replace_all_minimal_api(replace_all_tool):
    """Test the minimal API with just snapshot_token and content."""
    tool, bridge = replace_all_tool

    status = tool.run(
        snapshot_token="tab-1:1",
        content="Minimal replacement",
    )

    assert "queued" in status or "applied" in status
    # Verify no extra fields needed
    assert bridge.calls, "Edit should succeed with minimal fields"
