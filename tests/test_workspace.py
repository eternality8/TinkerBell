"""Unit tests for the workspace/tab management layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinkerbell.ai.memory.cache_bus import DocumentCacheBus, DocumentCacheEvent, DocumentClosedEvent
from tinkerbell.editor.document_model import DocumentMetadata, DocumentState
from tinkerbell.editor.workspace import DocumentWorkspace
from tinkerbell.services.bridge import DocumentBridge


class _StubEditor:
    """Minimal editor adapter that avoids requiring a real QApplication."""

    def __init__(self) -> None:
        self._document = DocumentState()

    def load_document(self, document: DocumentState) -> None:
        self._document = document

    def to_document(self) -> DocumentState:
        return self._document

    def request_snapshot(self, *, delta_only: bool = False) -> dict:
        return self._document.snapshot(delta_only=delta_only)

    def add_snapshot_listener(self, _listener) -> None:  # pragma: no cover - unused in tests
        return None

    def add_text_listener(self, _listener) -> None:  # pragma: no cover - unused in tests
        return None

    def add_selection_listener(self, _listener) -> None:  # pragma: no cover - unused in tests
        return None

    def apply_selection(self, selection) -> None:
        self._document.selection = selection

    def set_preview_mode(self, _enabled: bool) -> None:
        return None

    def toggle_preview(self) -> None:
        return None

    def apply_theme(self, _theme) -> None:
        return None

    def apply_ai_edit(self, directive) -> DocumentState:  # pragma: no cover - not exercised
        self._document.update_text(directive.content)
        return self._document

    def apply_patch_result(self, result, selection_hint=None) -> DocumentState:  # pragma: no cover - not exercised
        self._document.update_text(result.text)
        return self._document


def _make_workspace() -> DocumentWorkspace:
    return DocumentWorkspace(editor_factory=lambda: _StubEditor())  # type: ignore[arg-type]


def test_workspace_creates_tabs_and_tracks_active() -> None:
    workspace = _make_workspace()
    first = workspace.create_tab()
    second = workspace.create_tab()

    assert workspace.tab_count() == 2
    assert workspace.active_tab_id == second.id

    workspace.set_active_tab(first.id)
    assert workspace.active_tab_id == first.id

    closed = workspace.close_tab(first.id)
    assert closed.id == first.id
    assert workspace.tab_count() == 1
    assert workspace.active_tab_id == second.id


def test_workspace_notifies_listeners_on_active_change() -> None:
    workspace = _make_workspace()
    tab_a = workspace.create_tab()
    tab_b = workspace.create_tab()

    observed: list[str | None] = []

    def listener(tab) -> None:
        observed.append(tab.id if tab is not None else None)

    workspace.add_active_listener(listener)
    workspace.set_active_tab(tab_a.id)
    workspace.set_active_tab(tab_b.id)
    workspace.close_tab(tab_b.id)

    assert observed[0] == tab_a.id
    assert observed[1] == tab_b.id
    assert observed[-1] == tab_a.id  # after closing tab_b, tab_a becomes active again


def test_workspace_finds_tabs_by_path(tmp_path: Path) -> None:
    a_path = tmp_path / "alpha.md"
    b_path = tmp_path / "bravo.md"
    state_a = DocumentState(text="# A", metadata=DocumentMetadata(path=a_path))
    state_b = DocumentState(text="# B", metadata=DocumentMetadata(path=b_path))

    workspace = _make_workspace()
    tab_a = workspace.create_tab(document=state_a)
    workspace.create_tab(document=state_b)

    found = workspace.find_tab_by_path(a_path)
    assert found is tab_a


def test_workspace_serializes_tab_metadata(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    state = DocumentState(text="hi", metadata=DocumentMetadata(path=path))

    workspace = _make_workspace()
    tab = workspace.create_tab(document=state)
    tab.document().dirty = True
    tab.update_title()

    payload = workspace.serialize_tabs()
    assert payload == [
        {
            "tab_id": tab.id,
            "title": tab.title,
            "dirty": True,
            "language": "markdown",
            "path": str(path),
            "created_at": tab.created_at.isoformat(),
        }
    ]


def test_workspace_serialized_state_includes_active_and_counter() -> None:
    workspace = _make_workspace()
    tab = workspace.create_tab()
    state = workspace.serialize_state()

    assert state["active_tab_id"] == tab.id
    assert isinstance(state["untitled_counter"], int)
    assert state["open_tabs"][0]["tab_id"] == tab.id


def test_workspace_accepts_custom_tab_metadata() -> None:
    workspace = _make_workspace()
    custom_id = "tab-123"
    tab = workspace.create_tab(tab_id=custom_id, untitled_index=5)

    assert tab.id == custom_id
    assert tab.untitled_index == 5


def test_workspace_get_tab_accepts_path(tmp_path: Path) -> None:
    path = tmp_path / "story.txt"
    state = DocumentState(text="hello", metadata=DocumentMetadata(path=path))

    workspace = _make_workspace()
    tab = workspace.create_tab(document=state)

    resolved = workspace.get_tab(str(path))
    assert resolved is tab


def test_workspace_errors_when_switching_to_unknown_tab() -> None:
    workspace = _make_workspace()
    workspace.create_tab()
    with pytest.raises(KeyError):
        workspace.set_active_tab("missing")


def test_workspace_close_tab_publishes_closed_event() -> None:
    bus = DocumentCacheBus()
    closed_events: list[DocumentClosedEvent] = []

    def on_closed(event: DocumentCacheEvent) -> None:
        assert isinstance(event, DocumentClosedEvent)
        closed_events.append(event)

    bus.subscribe(DocumentClosedEvent, on_closed)

    workspace = DocumentWorkspace(
        editor_factory=lambda: _StubEditor(),  # type: ignore[arg-type]
        bridge_factory=lambda editor: DocumentBridge(editor=editor, cache_bus=bus),
    )

    tab = workspace.create_tab()
    workspace.close_tab(tab.id)

    assert closed_events
    assert closed_events[-1].document_id == tab.document().document_id
