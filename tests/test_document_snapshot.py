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
