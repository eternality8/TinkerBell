"""Targeted tests for DocumentApplyPatchTool streaming diffs."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import pytest

from tinkerbell.ai.tools.document_apply_patch import DocumentApplyPatchTool
from tinkerbell.ai.tools.document_edit import DocumentEditTool
from tinkerbell.chat.message_model import EditDirective


class _StreamingBridgeStub:
    def __init__(self, *, text: str = "Alpha beta gamma", version: str = "digest-1") -> None:
        self._text = text
        self.snapshot = {
            "text": text,
            "selection": (0, 0),
            "version": version,
            "document_id": "doc-stream",
            "path": "doc.md",
            "content_hash": "hash-1",
            "length": len(text),
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
    ) -> None:  # type: ignore[override]
        if isinstance(directive, Mapping):
            payload = dict(directive)
        else:
            payload = {
                "action": directive.action,
                "content": directive.content,
                "target_range": directive.target_range,
                "diff": directive.diff,
            }
        payload["tab_id"] = tab_id
        self.calls.append(payload)
        self.last_snapshot_version = "updated"

    def generate_snapshot(  # type: ignore[override]
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_text: bool = True,
        window: Mapping[str, Any] | None = None,
        **_: Any,
    ) -> Mapping[str, Any]:
        assert delta_only is False
        self.snapshot_requests.append(
            {
                "tab_id": tab_id,
                "include_text": include_text,
                "window": dict(window) if isinstance(window, Mapping) else None,
            }
        )
        snapshot = dict(self.snapshot)
        text = snapshot.get("text", "")
        doc_text = text if isinstance(text, str) else ""
        length = len(doc_text)
        snapshot["length"] = length
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

    def get_last_diff_summary(self, tab_id: str | None = None) -> str | None:  # noqa: ARG002 - unused
        return self.last_diff_summary

    def get_last_snapshot_version(self, tab_id: str | None = None) -> str | None:  # noqa: ARG002 - unused
        return self.last_snapshot_version


@pytest.fixture()
def streaming_tool() -> tuple[DocumentApplyPatchTool, _StreamingBridgeStub]:
    bridge = _StreamingBridgeStub()
    edit_tool = DocumentEditTool(bridge=bridge)
    patch_tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)
    return patch_tool, bridge


def _patch_tool(text: str, *, selection: tuple[int, int] = (0, 0)) -> tuple[DocumentApplyPatchTool, _StreamingBridgeStub]:
    bridge = _StreamingBridgeStub(text=text)
    bridge.snapshot["selection"] = selection
    edit_tool = DocumentEditTool(bridge=bridge)
    return DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool), bridge


def test_document_apply_patch_streams_ranges_and_attaches_metadata(streaming_tool):
    tool, bridge = streaming_tool

    status = tool.run(
        patches=[{"start": 0, "end": 5, "replacement": "Omega"}],
        rationale="stream",
    )

    assert isinstance(status, str)
    assert bridge.snapshot_requests[0]["include_text"] is False
    window_request = bridge.snapshot_requests[-1]
    assert window_request["window"]["start"] == 0
    payload = bridge.calls[-1]
    ranges = payload["ranges"]
    assert isinstance(ranges, list) and ranges
    assert ranges[0]["replacement"] == "Omega"
    assert ranges[0]["match_text"].lower() == "alpha"
    assert "streamed_diff" in payload.get("metadata", {})


def test_document_apply_patch_sorts_multi_range_payload(streaming_tool):
    tool, bridge = streaming_tool
    patches = [
        {"start": 10, "end": 14, "replacement": "DELTA"},
        {"start": 0, "end": 5, "replacement": "OMEGA"},
    ]

    tool.run(patches=patches)

    payload = bridge.calls[-1]
    starts = [entry["start"] for entry in payload["ranges"]]
    assert starts == sorted(starts), "ranges should be normalized and sorted"


def test_document_apply_patch_requires_range_or_anchor_when_selection_unknown():
    tool, bridge = _patch_tool("Hello world", selection=(0, 0))

    with pytest.raises(ValueError, match="target_range or match_text"):
        tool.run(content="Hi there")

    assert bridge.calls == []


def test_document_apply_patch_realigns_stale_range_with_match_text():
    tool, bridge = _patch_tool("Alpha BETA Gamma", selection=(0, 5))

    status = tool.run(content="beta", target_range=(0, 5), match_text="BETA")

    assert "queued" in status
    diff = bridge.calls[-1]["diff"]
    assert "+Alpha beta" in diff


def test_document_apply_patch_rejects_stale_selection_without_anchor():
    tool, bridge = _patch_tool("Alpha beta", selection=(0, 5))
    bridge.snapshot["selection_text"] = "stale"

    with pytest.raises(ValueError, match="selection_text no longer matches"):
        tool.run(content="BETA", target_range=(0, 5))


def test_document_apply_patch_rejects_selection_fingerprint_mismatch():
    tool, bridge = _patch_tool("Alpha beta", selection=(0, 5))
    bridge.snapshot["selection_text"] = "Alpha"
    bridge.snapshot["selection_hash"] = hashlib.sha1(b"Alpha").hexdigest()

    with pytest.raises(ValueError, match="selection_fingerprint"):
        tool.run(content="ALPHA", target_range=(0, 5), selection_fingerprint="mismatch")


def test_document_apply_patch_rejects_ambiguous_match_text():
    tool, _ = _patch_tool("beta one beta two", selection=(0, 4))

    with pytest.raises(ValueError, match="matched multiple"):
        tool.run(content="BETA", target_range=(4, 7), match_text="beta")
