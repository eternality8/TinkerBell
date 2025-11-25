"""Targeted tests for DocumentApplyPatchTool streaming diffs."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from tinkerbell.ai.tools import document_apply_patch as document_apply_patch_module
from tinkerbell.ai.tools.document_apply_patch import DocumentApplyPatchTool, NeedsRangeError, InvalidSnapshotTokenError
from tinkerbell.ai.tools.document_edit import DocumentEditTool
from tinkerbell.chat.message_model import EditDirective
from tinkerbell.services.bridge import DocumentVersionMismatchError


class _StreamingBridgeStub:
    def __init__(self, *, text: str = "Alpha beta gamma", version: str = "digest-1") -> None:
        self._text = text
        self.snapshot = {
            "text": text,
            "version": version,
            "version_id": 1,
            "document_id": "doc-stream",
            "path": "doc.md",
            "content_hash": "hash-1",
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
    ) -> None:  # type: ignore[override]
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


def _patch_tool(text: str) -> tuple[DocumentApplyPatchTool, _StreamingBridgeStub]:
    bridge = _StreamingBridgeStub(text=text)
    edit_tool = DocumentEditTool(bridge=bridge)
    return DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool), bridge


def _run_with_meta(tool: DocumentApplyPatchTool, **kwargs: Any) -> str:
    payload = {"document_version": "digest-1", "version_id": 1, "content_hash": "hash-1"}
    payload.update(kwargs)
    return tool.run(**payload)


def test_document_apply_patch_streams_ranges_and_attaches_metadata(streaming_tool):
    tool, bridge = streaming_tool

    status = _run_with_meta(
        tool,
        patches=[{"start": 0, "end": 5, "replacement": "Omega"}],
        rationale="stream",
    )

    assert isinstance(status, str)
    assert bridge.snapshot_requests[0]["include_text"] is False
    assert any(entry.get("include_text") for entry in bridge.snapshot_requests[1:]), (
        "streamed diffs should fetch document text for normalization"
    )
    payload = bridge.calls[-1]
    ranges = payload["ranges"]
    assert isinstance(ranges, list) and ranges
    assert ranges[0]["replacement"] == "Omega"
    assert ranges[0]["match_text"].lower() == "alpha"
    assert "streamed_diff" in payload.get("metadata", {})
    assert payload["metadata"]["scope_origin"] == "explicit_span"
    scope = ranges[0].get("scope")
    assert isinstance(scope, dict)
    assert scope["origin"] == "explicit_span"
    assert scope["range"] == {"start": 0, "end": 5}


def test_document_apply_patch_streaming_copies_snapshot_version_and_hash(streaming_tool):
    tool, bridge = streaming_tool

    _run_with_meta(tool, patches=[{"start": 0, "end": 5, "replacement": "OMEGA"}])

    payload = bridge.calls[-1]
    assert payload["document_version"] == bridge.snapshot["version"]
    assert payload["content_hash"] == bridge.snapshot["content_hash"]


def test_document_apply_patch_sorts_multi_range_payload(streaming_tool):
    tool, bridge = streaming_tool
    patches = [
        {"start": 10, "end": 14, "replacement": "DELTA"},
        {"start": 0, "end": 5, "replacement": "OMEGA"},
    ]

    _run_with_meta(tool, patches=patches)

    payload = bridge.calls[-1]
    starts = [entry["start"] for entry in payload["ranges"]]
    assert starts == sorted(starts), "ranges should be normalized and sorted"


def test_document_apply_patch_streaming_normalizes_ranges_before_queueing():
    bridge = _StreamingBridgeStub(text="alpha beta")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    status = _run_with_meta(tool, patches=[{"start": 2, "end": 5, "replacement": "BETA"}])

    assert isinstance(status, str)
    payload = bridge.calls[-1]
    entry = payload["ranges"][0]
    assert entry["start"] == 0
    assert entry["end"] == 5
    assert entry["match_text"] == "alpha"
    assert entry["replacement"] == "alBETA"


def test_document_apply_patch_accepts_target_span():
    text = "Alpha\nBeta\nGamma\n"
    tool, bridge = _patch_tool(text)

    status = _run_with_meta(
        tool,
        content="Beta replacement\n",
        target_span={"start_line": 1, "end_line": 1},
    )

    assert "queued" in status or "applied" in status
    payload = bridge.calls[-1]
    range_payload = payload["ranges"][0]
    assert range_payload["start"] == len("Alpha\n")
    assert range_payload["end"] == len("Alpha\nBeta\n")
    assert "Beta replacement" in payload["diff"]
    assert payload["metadata"]["scope_origin"] == "explicit_span"
    assert payload["metadata"]["scope_range"] == {
        "start": len("Alpha\n"),
        "end": len("Alpha\nBeta\n"),
    }
    assert range_payload["scope"]["origin"] == "explicit_span"


def test_document_apply_patch_rejects_out_of_bounds_line_span():
    tool, bridge = _patch_tool("Alpha\nBeta\n")

    with pytest.raises(ValueError, match="line count"):
        _run_with_meta(
            tool,
            content="Gamma\n",
            target_span={"start_line": 10, "end_line": 12},
        )

    assert bridge.calls == []


def test_document_apply_patch_marks_document_scope_metadata():
    text = "Alpha Beta"
    tool, bridge = _patch_tool(text)

    status = _run_with_meta(
        tool,
        content="Gamma Delta",
        scope="document",
    )

    assert "queued" in status or "applied" in status
    payload = bridge.calls[-1]
    metadata = payload["metadata"]
    assert metadata["scope_origin"] == "document"
    assert metadata["scope_range"] == {"start": 0, "end": len(text)}
    range_payload = payload["ranges"][0]
    assert range_payload["scope"]["origin"] == "document"
    assert range_payload["scope_length"] == len(text)


def test_document_apply_patch_emits_legacy_adapter_event(monkeypatch: pytest.MonkeyPatch):
    tool, bridge = _patch_tool("Hello world")
    # Enable legacy adapter for this test
    tool.configure_line_span_policy(require_line_spans=True, adapt_legacy_ranges=True)
    captured: list[tuple[str, dict | None]] = []

    def _emit(event: str, payload: dict | None = None) -> None:
        captured.append((event, payload))

    monkeypatch.setattr(document_apply_patch_module, "telemetry_emit", _emit)

    _run_with_meta(tool, content="Hi", target_range=(0, 5))

    events = [name for name, _ in captured]
    assert "target_range.legacy_adapter" in events
    legacy_payload = next(payload for name, payload in captured if name == "target_range.legacy_adapter")
    assert legacy_payload["start_line"] == 0
    assert legacy_payload["end_line"] == 0
    assert legacy_payload["range_end"] == 5


def test_document_apply_patch_requires_spans_when_configured():
    tool, _ = _patch_tool("Hello world")
    tool.configure_line_span_policy(require_line_spans=True, adapt_legacy_ranges=False)

    with pytest.raises(ValueError, match="target_span"):
        _run_with_meta(tool, content="Hi", target_range=(0, 2))


def test_document_apply_patch_allows_span_when_required():
    tool, bridge = _patch_tool("Alpha Beta")
    tool.configure_line_span_policy(require_line_spans=True, adapt_legacy_ranges=False)

    status = _run_with_meta(
        tool,
        content="Alpha",
        target_span=(0, 0),
    )

    assert "queued" in status
    assert bridge.calls, "Edit should have been queued when target_span provided"


def test_document_apply_patch_requires_range_or_anchor_when_selection_unknown():
    tool, bridge = _patch_tool("Hello world")

    with pytest.raises(ValueError, match="target_span .*target_range"):
        _run_with_meta(tool, content="Hi there")

    assert bridge.calls == []


def test_document_apply_patch_forces_replace_all_when_content_matches_document():
    tool, bridge = _patch_tool("abcdefghij")

    status = _run_with_meta(tool, content="jihgfedcba", operation="replace")

    assert "queued" in status or "applied" in status
    payload = bridge.calls[-1]
    ranges = payload["ranges"]
    assert ranges[0]["start"] == 0
    assert ranges[0]["end"] == len("abcdefghij")
    assert ranges[0]["match_text"] == "abcdefghij"


def test_document_apply_patch_realigns_stale_range_with_match_text():
    tool, bridge = _patch_tool("Alpha BETA Gamma")
    # Enable legacy adapter for this test
    tool.configure_line_span_policy(require_line_spans=True, adapt_legacy_ranges=True)

    status = _run_with_meta(tool, content="beta", target_range=(0, 5), match_text="BETA")

    assert "queued" in status
    diff = bridge.calls[-1]["diff"]
    assert "+Alpha beta" in diff


def test_document_apply_patch_widens_ranges_before_building_diff():
    tool, bridge = _patch_tool("alpha beta")
    # Enable legacy adapter for this test
    tool.configure_line_span_policy(require_line_spans=True, adapt_legacy_ranges=True)

    status = _run_with_meta(tool, content="BETA", target_range=(2, 5))

    assert "queued" in status
    diff = bridge.calls[-1]["diff"]
    assert "-alpha" in diff
    assert "+alBETA" in diff


def test_document_apply_patch_rejects_ambiguous_match_text():
    tool, _ = _patch_tool("beta one beta two")
    # Enable legacy adapter for this test
    tool.configure_line_span_policy(require_line_spans=True, adapt_legacy_ranges=True)

    with pytest.raises(ValueError, match="matched multiple"):
        _run_with_meta(tool, content="BETA", target_range=(4, 7), match_text="beta")


def test_document_apply_patch_rejects_large_insert_without_range():
    tool, bridge = _patch_tool("Short text")
    long_text = "x" * 2000

    with pytest.raises(NeedsRangeError, match="needs_range"):
        _run_with_meta(tool, content=long_text, operation="replace")

    assert bridge.calls == []


def test_document_apply_patch_rejects_hash_mismatch():
    tool, bridge = _patch_tool("Alpha beta")
    # Enable legacy adapter for this test so we can test hash mismatch specifically
    tool.configure_line_span_policy(require_line_spans=True, adapt_legacy_ranges=True)

    with pytest.raises(DocumentVersionMismatchError) as exc:
        _run_with_meta(
            tool,
            content="BETA",
            target_range=(0, 5),
            content_hash="hash-mismatch",
        )

    assert exc.value.cause == "hash_mismatch"
    assert bridge.calls == []


def test_document_apply_patch_forces_replace_all_with_length_tolerance():
    text = "A" * 200
    replacement = "B" * 198  # Within ±5% of document length
    tool, bridge = _patch_tool(text)

    status = _run_with_meta(tool, content=replacement)

    assert "queued" in status or "applied" in status
    payload = bridge.calls[-1]
    range_payload = payload["ranges"][0]
    assert range_payload["start"] == 0
    assert range_payload["end"] == len(text)
    assert range_payload["match_text"] == text


# -----------------------------------------------------------------------
# WS1.1.7: snapshot_token parsing tests
# -----------------------------------------------------------------------


def test_document_apply_patch_parses_valid_snapshot_token():
    """Test that a valid snapshot_token is parsed into tab_id and version_id."""
    tool, bridge = _patch_tool("Alpha\nBeta\n")
    bridge.snapshot["version_id"] = 42
    
    # Use snapshot_token instead of separate fields
    status = tool.run(
        snapshot_token="doc-stream:42",
        content="New Beta\n",
        target_span={"start_line": 1, "end_line": 1},
        document_version="digest-1",
        version_id=42,
        content_hash="hash-1",
    )
    
    assert "queued" in status or "applied" in status


def test_document_apply_patch_extracts_tab_id_from_snapshot_token():
    """Test that tab_id is extracted from snapshot_token when not explicitly provided."""
    tool, bridge = _patch_tool("Alpha Beta")
    
    status = tool.run(
        snapshot_token="my-tab:1",
        content="Gamma",
        target_span={"start_line": 0, "end_line": 0},
        document_version="digest-1",
        version_id=1,
        content_hash="hash-1",
    )
    
    assert "queued" in status or "applied" in status


def test_document_apply_patch_rejects_malformed_snapshot_token_no_colon():
    """Test that malformed tokens without colons are rejected."""
    tool, _ = _patch_tool("Alpha Beta")
    
    with pytest.raises(InvalidSnapshotTokenError, match="tab_id:version_id"):
        tool.run(
            snapshot_token="no-colon-here",
            content="Gamma",
            target_span={"start_line": 0, "end_line": 0},
            document_version="digest-1",
            version_id=1,
            content_hash="hash-1",
        )


def test_document_apply_patch_rejects_snapshot_token_empty_tab_id():
    """Test that tokens with empty tab_id are rejected."""
    tool, _ = _patch_tool("Alpha Beta")
    
    with pytest.raises(InvalidSnapshotTokenError, match="tab_id"):
        tool.run(
            snapshot_token=":1",
            content="Gamma",
            target_span={"start_line": 0, "end_line": 0},
            document_version="digest-1",
            version_id=1,
            content_hash="hash-1",
        )


def test_document_apply_patch_rejects_snapshot_token_empty_version_id():
    """Test that tokens with empty version_id are rejected."""
    tool, _ = _patch_tool("Alpha Beta")
    
    with pytest.raises(InvalidSnapshotTokenError, match="version_id"):
        tool.run(
            snapshot_token="my-tab:",
            content="Gamma",
            target_span={"start_line": 0, "end_line": 0},
            document_version="digest-1",
            version_id=1,
            content_hash="hash-1",
        )


def test_document_apply_patch_accepts_none_snapshot_token():
    """Test that None snapshot_token is handled gracefully."""
    tool, bridge = _patch_tool("Alpha Beta")
    
    # Should work without snapshot_token using traditional fields
    status = _run_with_meta(
        tool,
        content="Gamma",
        target_span={"start_line": 0, "end_line": 0},
    )
    
    assert "queued" in status or "applied" in status


def test_document_apply_patch_auto_fills_target_span_from_suggested_span():
    """Test that target_span is auto-filled from suggested_span when not provided."""
    tool, bridge = _patch_tool("Alpha\nBeta\nGamma\n")
    
    status = tool.run(
        suggested_span={"start_line": 1, "end_line": 1},
        content="New Beta\n",
        document_version="digest-1",
        version_id=1,
        content_hash="hash-1",
    )
    
    assert "queued" in status or "applied" in status
    payload = bridge.calls[-1]
    range_payload = payload["ranges"][0]
    # Should have used the suggested_span to derive the range
    assert range_payload["start"] == len("Alpha\n")
    assert range_payload["end"] == len("Alpha\nBeta\n")
