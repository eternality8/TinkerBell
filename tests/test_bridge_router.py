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