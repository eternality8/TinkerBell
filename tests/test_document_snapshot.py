"""Workstream 1 regression tests for windowed snapshots."""

from __future__ import annotations

from copy import deepcopy

from tinkerbell.ai.tools.document_snapshot import DocumentSnapshotTool
from tinkerbell.editor.document_model import DocumentMetadata, DocumentState, SelectionRange
from tinkerbell.services.bridge import DocumentBridge


class _EditorStub:
    def __init__(self, text: str, selection: tuple[int, int]) -> None:
        self.state = DocumentState(
            text=text,
            metadata=DocumentMetadata(language="markdown"),
        )
        self._selection = SelectionRange(*selection)

    def to_document(self) -> DocumentState:  # pragma: no cover - trivial getter
        return deepcopy(self.state)

    def load_document(self, document: DocumentState) -> None:  # pragma: no cover - unused in tests
        self.state = deepcopy(document)

    def apply_ai_edit(self, directive, *, preserve_selection: bool = False):  # pragma: no cover - unused in tests
        return self.state

    def apply_patch_result(self, result, selection_hint=None, *, preserve_selection: bool = False):  # pragma: no cover - unused in tests
        return self.state

    def selection_span(self) -> tuple[int, int]:
        return (self._selection.start, self._selection.end)

    def selection_range(self) -> SelectionRange:
        return SelectionRange(self._selection.start, self._selection.end)



class _SnapshotProviderStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.snapshot = {
            "text": "Hello world",
            "text_range": {"start": 0, "end": 5},
            "window": {"start": 0, "end": 5},
            "document_id": "doc-stub",
            "version": "base",
            "length": 11,
        }

    def generate_snapshot(self, **kwargs):
        self.calls.append(kwargs)
        return dict(self.snapshot)

    def get_last_diff_summary(self, tab_id=None):  # pragma: no cover - returns static diff
        return "Δ0"

    def get_last_snapshot_version(self, tab_id=None):  # pragma: no cover - returns static version
        return "base"


def test_bridge_windowed_snapshot_returns_manifest_cache_hits():
    editor = _EditorStub(text="alpha beta gamma delta epsilon zeta", selection=(12, 17))
    bridge = DocumentBridge(editor=editor)

    window = {"kind": "range", "start": 12, "end": 22, "max_chars": 16}
    first = bridge.generate_snapshot(window=window, chunk_profile="prose")
    second = bridge.generate_snapshot(window=window, chunk_profile="prose")

    span = first["text_range"]["end"] - first["text_range"]["start"]
    assert span == 10
    manifest = first.get("chunk_manifest")
    assert manifest and manifest["cache_hit"] is False
    assert manifest["window"]["start"] == window["start"]
    assert manifest["window"]["end"] == window["end"]
    assert second["chunk_manifest"]["cache_hit"] is True


def test_document_snapshot_tool_defaults_to_document_window():
    provider = _SnapshotProviderStub()
    tool = DocumentSnapshotTool(provider=provider)

    snapshot = tool.run()

    assert snapshot["version"] == "base"
    assert provider.calls, "provider should receive a windowed call"
    window_arg = provider.calls[-1].get("window")
    assert isinstance(window_arg, dict)
    assert window_arg.get("kind") == "document"


def test_document_snapshot_tool_respects_explicit_window_and_include_text_flag():
    provider = _SnapshotProviderStub()
    tool = DocumentSnapshotTool(provider=provider)

    tool.run(window="document", include_text=False)

    window_arg = provider.calls[-1].get("window")
    assert window_arg == "document"
    assert provider.calls[-1].get("include_text") is False


def test_document_snapshot_tool_accepts_positional_request_mapping():
    provider = _SnapshotProviderStub()
    tool = DocumentSnapshotTool(provider=provider)

    tool.run({"window": {"kind": "range", "start": 0, "end": 360}, "document_id": "doc-alt"})

    call = provider.calls[-1]
    assert call["window"]["kind"] == "range"
    assert call["window"]["start"] == 0
    assert call["window"]["end"] == 360
    assert call.get("tab_id") == "doc-alt"


def test_document_snapshot_tool_parses_string_request_payload():
    provider = _SnapshotProviderStub()
    tool = DocumentSnapshotTool(provider=provider)

    request = "{'window': {'kind': 'range', 'start': 10, 'end': 20}, 'include_text': False}"
    tool.run(request)

    call = provider.calls[-1]
    assert call["window"]["start"] == 10
    assert call["window"]["end"] == 20
    assert call["include_text"] is False


# -----------------------------------------------------------------------
# WS1.1.8: Integration tests for snapshot_token and suggested_span
# -----------------------------------------------------------------------


def test_document_snapshot_tool_emits_snapshot_token():
    """Test that snapshot_token is emitted in the correct format."""
    provider = _SnapshotProviderStub()
    provider.snapshot["version_id"] = 42
    provider.snapshot["tab_id"] = "my-tab"
    tool = DocumentSnapshotTool(provider=provider)

    snapshot = tool.run()

    assert "snapshot_token" in snapshot
    assert snapshot["snapshot_token"] == "my-tab:42"


def test_document_snapshot_tool_uses_document_id_for_snapshot_token_when_no_tab_id():
    """Test that document_id is used as fallback for snapshot_token."""
    provider = _SnapshotProviderStub()
    provider.snapshot["version_id"] = 1
    provider.snapshot["document_id"] = "doc-stub"
    # Ensure no tab_id is present
    provider.snapshot.pop("tab_id", None)
    tool = DocumentSnapshotTool(provider=provider)

    snapshot = tool.run()

    assert "snapshot_token" in snapshot
    assert snapshot["snapshot_token"] == "doc-stub:1"


def test_document_snapshot_tool_emits_suggested_span():
    """Test that suggested_span is derived from text_range."""
    provider = _SnapshotProviderStub()
    provider.snapshot["text_range"] = {"start": 10, "end": 50}
    provider.snapshot["line_start_offsets"] = [0, 15, 30, 45, 60]
    tool = DocumentSnapshotTool(provider=provider)

    snapshot = tool.run()

    assert "suggested_span" in snapshot
    span = snapshot["suggested_span"]
    assert span["start_line"] == 0  # offset 10 is in line 0 (0-14)
    assert span["end_line"] == 3    # offset 49 (end-1) is in line 3 (45-59)


def test_document_snapshot_tool_suggested_span_handles_full_document():
    """Test that suggested_span covers full document when text_range spans all."""
    provider = _SnapshotProviderStub()
    provider.snapshot["text"] = "Line 1\nLine 2\nLine 3\n"
    provider.snapshot["text_range"] = {"start": 0, "end": 21}
    provider.snapshot["line_start_offsets"] = [0, 7, 14, 21]
    tool = DocumentSnapshotTool(provider=provider)

    snapshot = tool.run()

    assert "suggested_span" in snapshot
    span = snapshot["suggested_span"]
    assert span["start_line"] == 0
    assert span["end_line"] == 2  # Last line with content


def test_document_snapshot_tool_no_suggested_span_without_line_offsets():
    """Test that suggested_span is not emitted without line_start_offsets."""
    provider = _SnapshotProviderStub()
    provider.snapshot["text_range"] = {"start": 0, "end": 100}
    provider.snapshot.pop("line_start_offsets", None)
    provider.snapshot.pop("line_offsets", None)
    tool = DocumentSnapshotTool(provider=provider)

    snapshot = tool.run()

    assert "suggested_span" not in snapshot


def test_document_snapshot_tool_version_flow_end_to_end():
    """Test complete flow: snapshot returns version info for downstream tools."""
    editor = _EditorStub(text="Alpha\nBeta\nGamma\n", selection=(0, 5))
    bridge = DocumentBridge(editor=editor)

    # Get snapshot
    snapshot = bridge.generate_snapshot(window={"kind": "document"})

    # Verify all required version fields are present
    assert "version_id" in snapshot or "version" in snapshot
    assert "document_id" in snapshot or "tab_id" in snapshot
    assert "text_range" in snapshot
    assert "line_start_offsets" in snapshot or "line_offsets" in snapshot


# -----------------------------------------------------------------------
# WS3 4.4.x: Ignored keys tests
# -----------------------------------------------------------------------


def test_document_snapshot_tool_returns_ignored_keys():
    """WS3 4.4.1: Unrecognized request keys are tracked and returned."""
    provider = _SnapshotProviderStub()
    tool = DocumentSnapshotTool(provider=provider)

    # Pass some unrecognized keys
    snapshot = tool.run({"unknown_param": "value", "another_unknown": 123, "delta_only": False})

    assert "ignored_keys" in snapshot
    assert "another_unknown" in snapshot["ignored_keys"]
    assert "unknown_param" in snapshot["ignored_keys"]
    assert len(snapshot["ignored_keys"]) == 2
    assert "warning" in snapshot
    assert "another_unknown" in snapshot["warning"]


def test_document_snapshot_tool_no_ignored_keys_for_valid_request():
    """WS3 4.4.1: No ignored_keys when all request keys are recognized."""
    provider = _SnapshotProviderStub()
    tool = DocumentSnapshotTool(provider=provider)

    # Pass only recognized keys
    snapshot = tool.run({"delta_only": True, "include_text": True, "window": "document"})

    assert "ignored_keys" not in snapshot
    assert "warning" not in snapshot
