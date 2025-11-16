"""Tests for the TabbedEditorWidget wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinkerbell.editor.document_model import DocumentMetadata, DocumentState, SelectionRange
from tinkerbell.editor.tabbed_editor import TabbedEditorWidget

pytestmark = pytest.mark.usefixtures("qtbot")


def _build_state(text: str, path: Path | None = None) -> DocumentState:
    metadata = DocumentMetadata(path=path)
    return DocumentState(text=text, metadata=metadata, selection=SelectionRange(0, 0))


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


def test_selection_listeners_fire_on_tab_switch() -> None:
    widget = TabbedEditorWidget()
    second = widget.create_tab(document=_build_state("other"))

    observed: list[SelectionRange] = []
    widget.add_selection_listener(lambda selection: observed.append(selection))

    widget.focus_tab(second.id)
    assert observed
    assert isinstance(observed[-1], SelectionRange)


def test_tabbed_editor_diff_overlay_is_routable() -> None:
    widget = TabbedEditorWidget()
    tab_id = widget.active_tab_id()
    assert tab_id is not None

    widget.show_diff_overlay("@@", spans=[(0, 1)], summary="Δ", source="tool", tab_id=tab_id)

    tab = widget.workspace.get_tab(tab_id)
    assert tab.editor.diff_overlay is not None

    widget.clear_diff_overlay(tab_id=tab_id)

    assert tab.editor.diff_overlay is None
