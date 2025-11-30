"""WorkspaceBridgeRouter metadata tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import pytest

from tinkerbell.services.bridge import DocumentBridge
from tinkerbell.services.bridge_router import WorkspaceBridgeRouter

from tests.test_bridge import RecordingEditor, _make_diff


@dataclass
class _StubTab:
    id: str
    bridge: DocumentBridge
    title: str = "Untitled"

    def document(self):  # pragma: no cover - compatibility shim
        return self.bridge.editor.to_document()


class _StubWorkspace:
    def __init__(self, tabs: Sequence[_StubTab]):
        self._tabs: Dict[str, _StubTab] = {tab.id: tab for tab in tabs}
        self._order: List[str] = [tab.id for tab in tabs]
        self.active_tab_id: str | None = self._order[0] if self._order else None

    def iter_tabs(self):
        return list(self._tabs.values())

    def serialize_tabs(self) -> list[dict[str, Any]]:
        return [{"id": tab.id, "title": tab.title} for tab in self._tabs.values()]

    def active_bridge(self) -> DocumentBridge:
        if self.active_tab_id is None:
            raise RuntimeError("No active tab")
        return self._tabs[self.active_tab_id].bridge

    def get_tab(self, tab_id: str) -> _StubTab:
        return self._tabs[tab_id]

    def require_active_tab(self) -> _StubTab:
        if self.active_tab_id is None:
            raise KeyError("No active tab")
        return self._tabs[self.active_tab_id]


def test_router_propagates_failure_metadata_with_tab_id() -> None:
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    bridge.set_tab_context(tab_id="tab-1")
    tab = _StubTab(id="tab-1", bridge=bridge, title="Doc 1")
    workspace = _StubWorkspace([tab])
    router = WorkspaceBridgeRouter(workspace)

    failures: list[Mapping[str, Any] | None] = []

    def _listener(_directive, _message, metadata: Mapping[str, Any] | None = None) -> None:
        failures.append(metadata)

    router.add_failure_listener(_listener)

    snapshot = router.generate_snapshot(tab_id="tab-1")
    diff = _make_diff("HELLO WORLD", "HELLO BRAVE WORLD")

    with pytest.raises(RuntimeError):
        router.queue_edit(
            {
                "action": "patch",
                "diff": diff,
                "document_version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
            },
            tab_id="tab-1",
        )

    assert failures, "expected router to emit failure metadata"
    metadata = failures[-1]
    assert metadata is not None
    assert metadata["tab_id"] == "tab-1"
    assert metadata["status"] == "conflict"
    assert metadata["cause"] == DocumentBridge.CAUSE_HASH_MISMATCH

    cached = router.get_last_failure_metadata("tab-1")
    assert cached is not None
    assert cached["tab_id"] == "tab-1"
    assert cached["status"] == metadata["status"]


class TestRouterToolContextProviderProtocol:
    """Tests verifying WorkspaceBridgeRouter implements ToolContextProvider protocol.
    
    These tests ensure the router can be used as a context_provider for ToolDispatcher,
    which requires get_document_content, set_document_content, get_active_tab_id, etc.
    """

    def test_get_document_content_returns_tab_text(self) -> None:
        """Router should return document text for a valid tab_id."""
        editor = RecordingEditor()
        # RecordingEditor uses DocumentState, update via load_document
        from tinkerbell.editor.document_model import DocumentState
        editor.load_document(DocumentState(text="Hello, World!"))
        bridge = DocumentBridge(editor=editor)
        tab = _StubTab(id="tab-1", bridge=bridge, title="Test Doc")
        workspace = _StubWorkspace([tab])
        router = WorkspaceBridgeRouter(workspace)

        content = router.get_document_content("tab-1")
        assert content == "Hello, World!"

    def test_get_document_content_returns_none_for_missing_tab(self) -> None:
        """Router should return None for non-existent tab_id."""
        editor = RecordingEditor()
        bridge = DocumentBridge(editor=editor)
        tab = _StubTab(id="tab-1", bridge=bridge)
        workspace = _StubWorkspace([tab])
        router = WorkspaceBridgeRouter(workspace)

        content = router.get_document_content("nonexistent-tab")
        assert content is None

    def test_get_tab_content_alias(self) -> None:
        """get_tab_content should work the same as get_document_content."""
        editor = RecordingEditor()
        from tinkerbell.editor.document_model import DocumentState
        editor.load_document(DocumentState(text="Tab content here"))
        bridge = DocumentBridge(editor=editor)
        tab = _StubTab(id="tab-1", bridge=bridge)
        workspace = _StubWorkspace([tab])
        router = WorkspaceBridgeRouter(workspace)

        content = router.get_tab_content("tab-1")
        assert content == "Tab content here"

    def test_get_active_tab_id(self) -> None:
        """Router should expose get_active_tab_id for ToolContextProvider."""
        editor = RecordingEditor()
        bridge = DocumentBridge(editor=editor)
        tab = _StubTab(id="tab-1", bridge=bridge)
        workspace = _StubWorkspace([tab])
        router = WorkspaceBridgeRouter(workspace)

        # _StubWorkspace sets first tab as active
        assert router.get_active_tab_id() == "tab-1"

    def test_set_document_content_updates_tab(self) -> None:
        """Router should update document text for a valid tab_id."""
        editor = RecordingEditor()
        from tinkerbell.editor.document_model import DocumentState
        editor.load_document(DocumentState(text="Original content"))
        bridge = DocumentBridge(editor=editor)
        bridge.set_tab_context(tab_id="tab-1")
        tab = _StubTab(id="tab-1", bridge=bridge)
        workspace = _StubWorkspace([tab])
        router = WorkspaceBridgeRouter(workspace)

        router.set_document_content("tab-1", "Updated content")

        assert router.get_document_content("tab-1") == "Updated content"

    def test_router_implements_tool_context_provider_protocol(self) -> None:
        """Router should have all methods required by ToolContextProvider."""
        editor = RecordingEditor()
        bridge = DocumentBridge(editor=editor)
        tab = _StubTab(id="tab-1", bridge=bridge)
        workspace = _StubWorkspace([tab])
        router = WorkspaceBridgeRouter(workspace)

        # These methods are required by ToolContextProvider protocol
        assert hasattr(router, "get_active_tab_id")
        assert callable(router.get_active_tab_id)
        assert hasattr(router, "get_document_content")
        assert callable(router.get_document_content)
        assert hasattr(router, "set_document_content")
        assert callable(router.set_document_content)
        assert hasattr(router, "get_version_token")
        assert callable(router.get_version_token)

    def test_router_implements_tab_listing_provider_protocol(self) -> None:
        """Router should have all methods required by TabListingProvider."""
        editor = RecordingEditor()
        bridge = DocumentBridge(editor=editor)
        tab = _StubTab(id="tab-1", bridge=bridge)
        workspace = _StubWorkspace([tab])
        router = WorkspaceBridgeRouter(workspace)

        # These methods are required by TabListingProvider protocol
        assert hasattr(router, "list_tabs")
        assert callable(router.list_tabs)
        assert hasattr(router, "active_tab_id")
        assert callable(router.active_tab_id)


class TestNoDocumentSnapshot:
    """Tests for snapshot generation when no documents are open."""

    def test_generate_snapshot_with_no_tabs_returns_empty_snapshot(self) -> None:
        """When no tabs are open, generate_snapshot returns a no_document snapshot."""
        workspace = _StubWorkspace([])  # Empty workspace
        router = WorkspaceBridgeRouter(workspace)

        snapshot = router.generate_snapshot()

        assert snapshot["no_document"] is True
        assert snapshot["tab_id"] is None
        assert snapshot["text"] == ""
        assert snapshot["version"] is None
        assert snapshot["length"] == 0
        assert snapshot["document_id"] == ""

    def test_generate_snapshot_with_no_tabs_includes_open_documents(self) -> None:
        """When include_open_documents=True, empty snapshot includes empty tabs list."""
        workspace = _StubWorkspace([])
        router = WorkspaceBridgeRouter(workspace)

        snapshot = router.generate_snapshot(include_open_documents=True)

        assert snapshot["no_document"] is True
        assert snapshot["open_tabs"] == []
        assert snapshot["active_tab_id"] is None

    def test_generate_snapshot_with_specific_tab_id_still_works(self) -> None:
        """Specifying a tab_id should still work even with no active tab."""
        editor = RecordingEditor()
        from tinkerbell.editor.document_model import DocumentState
        editor.load_document(DocumentState(text="Some content"))
        bridge = DocumentBridge(editor=editor)
        bridge.set_tab_context(tab_id="tab-1")
        tab = _StubTab(id="tab-1", bridge=bridge, title="Test Doc")
        workspace = _StubWorkspace([tab])
        workspace.active_tab_id = None  # No active tab, but tab exists
        router = WorkspaceBridgeRouter(workspace)

        snapshot = router.generate_snapshot(tab_id="tab-1")

        assert "no_document" not in snapshot or snapshot.get("no_document") is not True
        assert snapshot["text"] == "Some content"

    def test_list_tabs_with_no_tabs_returns_empty_list(self) -> None:
        """list_tabs should work even with no tabs open."""
        workspace = _StubWorkspace([])
        router = WorkspaceBridgeRouter(workspace)

        tabs = router.list_tabs()

        assert tabs == []

    def test_active_tab_id_with_no_tabs_returns_none(self) -> None:
        """active_tab_id should return None when no tabs are open."""
        workspace = _StubWorkspace([])
        router = WorkspaceBridgeRouter(workspace)

        assert router.active_tab_id() is None
        assert router.get_active_tab_id() is None

    def test_getattr_raises_attribute_error_with_no_tabs(self) -> None:
        """Accessing undefined attributes should raise AttributeError when no tabs."""
        workspace = _StubWorkspace([])
        router = WorkspaceBridgeRouter(workspace)

        # Should raise AttributeError, not RuntimeError
        with pytest.raises(AttributeError, match="no active tab"):
            _ = router.some_undefined_attribute

    def test_hasattr_returns_false_for_undefined_attributes_no_tabs(self) -> None:
        """hasattr should return False for undefined attributes when no tabs."""
        workspace = _StubWorkspace([])
        router = WorkspaceBridgeRouter(workspace)

        # hasattr should return False, not raise an exception
        assert hasattr(router, "list_tabs") is True  # Explicitly defined
        assert hasattr(router, "active_tab_id") is True  # Explicitly defined
        assert hasattr(router, "get_document_text") is False  # Not defined, no tab fallback
        assert hasattr(router, "get_tab_content")
        assert callable(router.get_tab_content)