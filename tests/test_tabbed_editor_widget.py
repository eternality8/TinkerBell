"""Tests for the TabbedEditorWidget wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinkerbell.editor.document_model import DocumentMetadata, DocumentState, SelectionRange
from tinkerbell.editor.tabbed_editor import TabbedEditorWidget

pytestmark = pytest.mark.usefixtures("qtbot")


def _build_state(text: str, path: Path | None = None) -> DocumentState:
    metadata = DocumentMetadata(path=path)
    return DocumentState(text=text, metadata=metadata)


def test_tabbed_editor_emits_snapshot_metadata(tmp_path: Path) -> None:
    widget = TabbedEditorWidget()
    snapshots: list[dict] = []

    widget.add_snapshot_listener(lambda snapshot: snapshots.append(snapshot))

    payload = widget.request_snapshot()

    assert payload["tab_id"] == widget.active_tab_id()
    assert payload["active_tab_id"] == widget.active_tab_id()
    assert isinstance(payload["open_tabs"], list)
    assert payload["tab_title"]

    # listeners get called as well
    assert snapshots
    assert snapshots[-1]["tab_id"] == payload["tab_id"]


def test_tabbed_editor_can_create_and_switch_tabs(tmp_path: Path) -> None:
    widget = TabbedEditorWidget()
    first_id = widget.active_tab_id()
    assert first_id is not None

    doc_path = tmp_path / "notes.md"
    second = widget.create_tab(document=_build_state("# second", doc_path))
    assert widget.active_tab_id() == second.id

    widget.focus_tab(first_id)
    assert widget.active_tab_id() == first_id


def test_tabbed_editor_closing_tab_updates_workspace(tmp_path: Path) -> None:
    widget = TabbedEditorWidget()
    first_id = widget.active_tab_id()
    assert first_id is not None
    second = widget.create_tab(document=_build_state("content"))

    widget.close_tab(second.id)

    assert widget.active_tab_id() == first_id
    assert len(widget.workspace.serialize_tabs()) == 1


def test_tabbed_editor_close_handler_can_intercept(tmp_path: Path) -> None:
    widget = TabbedEditorWidget()
    first_id = widget.active_tab_id()
    assert first_id is not None
    second = widget.create_tab(document=_build_state("content"))

    handled: list[str] = []

    def _handler(tab_id: str) -> bool:
        handled.append(tab_id)
        widget.close_tab(tab_id)
        return True

    widget.set_tab_close_handler(_handler)
    widget.request_tab_close(second.id)

    assert handled == [second.id]
    assert widget.workspace.tab_count() == 1


def test_tabbed_editor_snapshot_excludes_selection_metadata() -> None:
    widget = TabbedEditorWidget()

    snapshot = widget.request_snapshot()

    assert "selection" not in snapshot
    assert "selection_summary" not in snapshot


def test_tabbed_editor_diff_overlay_is_routable() -> None:
    widget = TabbedEditorWidget()
    tab_id = widget.active_tab_id()
    assert tab_id is not None

    widget.show_diff_overlay("@@", spans=[(0, 1)], summary="Δ", source="tool", tab_id=tab_id)

    tab = widget.workspace.get_tab(tab_id)
    assert tab.editor.diff_overlay is not None

    widget.clear_diff_overlay(tab_id=tab_id)

    assert tab.editor.diff_overlay is None


def test_tabbed_editor_notifies_selection_listeners() -> None:
    widget = TabbedEditorWidget()
    tab_id = widget.active_tab_id()
    assert tab_id is not None

    editor = widget.active_editor()
    editor.set_text("alpha\nbeta")

    captured: list[tuple[str, tuple[int, int], int, int]] = []

    def _listener(tab: str, selection: SelectionRange, line: int, column: int) -> None:
        captured.append((tab, selection.as_tuple(), line, column))

    widget.add_selection_listener(_listener)
    editor._set_selection(SelectionRange(6, 6))

    assert captured
    assert captured[-1] == (tab_id, (6, 6), 2, 1)
