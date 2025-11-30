"""Editor widget tests covering logical behaviors in headless mode."""

from typing import cast

import pytest

from tinkerbell.ui.presentation.chat.message_model import EditDirective
from tinkerbell.core.ranges import TextRange
from tinkerbell.editor.document_model import DocumentState, SelectionRange
from tinkerbell.editor.editor_widget import EditorWidget
from tinkerbell.editor.patches import PatchResult
from tinkerbell.services.bridge import DocumentBridge


@pytest.fixture(autouse=True)
def _ensure_qapp(qapp):  # pragma: no cover - pytest-qt provides the fixture
    """Guarantee a running QApplication when PySide6 is installed."""

    return qapp


def test_editor_widget_snapshot_roundtrip():
    widget = EditorWidget()
    doc = DocumentState(text="sample")
    widget.load_document(doc)
    snapshot = widget.request_snapshot()
    assert snapshot["text"] == "sample"


def test_editor_widget_applies_ai_edit_insert_and_replace():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="hello"))

    insert = EditDirective(action="insert", target_range=(5, 5), content=" world")
    widget.apply_ai_edit(insert)
    assert widget.to_document().text == "hello world"

    replace = EditDirective(action="replace", target_range=(0, 5), content="hi")
    widget.apply_ai_edit(replace)
    assert widget.to_document().text == "hi world"


def test_ai_edits_collapse_selection_after_application():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="hello"))

    insert = EditDirective(action="insert", target_range=(5, 5), content=" world")
    widget.apply_ai_edit(insert)
    selection = widget.selection_range()
    assert selection.start == selection.end == len("hello world")

    replace = EditDirective(action="replace", target_range=(0, 5), content="hi")
    widget.apply_ai_edit(replace)
    selection = widget.selection_range()
    assert selection.start == selection.end == len("hi")


def test_ai_edits_preserve_selection_when_requested():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="hello world"))
    widget._set_selection(SelectionRange(2, 4))

    insert = EditDirective(action="insert", target_range=(5, 5), content=" brave")
    widget.apply_ai_edit(insert, preserve_selection=True)

    selection = widget.selection_range()
    assert selection.start == 2
    assert selection.end == 4


def test_editor_widget_rejects_zero_length_replace():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="alpha beta"))

    directive = EditDirective(action="replace", target_range=(3, 3), content="-")

    with pytest.raises(ValueError, match="non-empty"):
        widget.apply_ai_edit(directive)


def test_editor_widget_preview_snapshot_contains_html():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="# Title"))
    widget.set_preview_mode(True)
    snapshot = widget.request_snapshot()
    assert snapshot["preview_enabled"] is True
    assert "tb-markdown-preview" in snapshot["preview"]
    assert "<h1" in snapshot["preview"]


def test_editor_widget_undo_redo_roundtrip():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="one"))
    widget.set_text("two")
    widget.insert_text(" three", position=3)
    widget.undo()
    assert widget.to_document().text == "two"
    widget.redo()
    assert widget.to_document().text == "two three"


def test_editor_widget_selection_updates_document_state():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="content"))
    widget._set_selection(SelectionRange(1, 4))
    selection = widget.selection_range()
    assert selection.start == 1
    assert selection.end == 4


def test_editor_widget_selection_listener_reports_line_column():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="alpha\nbeta"))

    captured: list[tuple[tuple[int, int], int, int]] = []

    def _listener(selection: SelectionRange, line: int, column: int) -> None:
        captured.append((selection.as_tuple(), line, column))

    widget.add_selection_listener(_listener)
    widget._set_selection(SelectionRange(6, 6))

    assert captured
    assert captured[-1] == ((6, 6), 2, 1)


def test_editor_widget_accepts_text_range_selection_inputs():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="content"))

    widget._set_selection(TextRange(2, 5))

    selection = widget.selection_range()
    assert selection.start == 2
    assert selection.end == 5


def test_editor_widget_diff_overlay_tracks_state():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="overlay text"))

    state = widget.show_diff_overlay("@@ diff @@", spans=[(0, 7)], summary="Δ", source="test")

    assert state.summary == "Δ"
    assert widget.diff_overlay is not None

    widget.clear_diff_overlay()

    assert widget.diff_overlay is None


def test_patch_result_collapses_selection_to_span_end():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="hello world"))

    result = PatchResult(text="hello brave world", spans=((6, 11),), summary="patch: +5")
    widget.apply_patch_result(result)

    selection = widget.selection_range()
    assert selection.start == selection.end == 11


def test_patch_result_preserves_selection_when_requested():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="hello world"))
    widget._set_selection(SelectionRange(1, 3))

    result = PatchResult(text="HELLO world", spans=((0, 5),), summary="patch: +0")
    widget.apply_patch_result(result, preserve_selection=True)

    selection = widget.selection_range()
    assert selection.start == 1
    assert selection.end == 3


def test_patch_result_uses_selection_hint_when_spans_missing():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="hello"))

    result = PatchResult(text="HELLO", spans=(), summary="patch: +0")
    widget.apply_patch_result(result, selection_hint=(2, 2))

    selection = widget.selection_range()
    assert selection.start == selection.end == 2


def test_undo_redo_preserves_text_range_history():
    widget = EditorWidget()
    widget.load_document(DocumentState(text="alpha beta"))
    widget._set_selection(TextRange(1, 4))

    widget.set_text("alpha beta!")
    widget._set_selection(TextRange(0, 0))

    widget.undo()
    selection = widget.selection_range()
    assert selection.start == selection.end == len("alpha beta")

    widget.redo()
    selection = widget.selection_range()
    assert selection.start == selection.end == len("alpha beta!")


def test_editor_widget_readonly_state():
    """Test that readonly state can be set and queried."""
    widget = EditorWidget()
    widget.load_document(DocumentState(text="hello"))
    
    # Initially not readonly
    assert not widget.is_readonly()
    
    # Set readonly
    widget.set_readonly(True)
    assert widget.is_readonly()
    
    # Unset readonly
    widget.set_readonly(False)
    assert not widget.is_readonly()
    
    # Programmatic edits still work when readonly (headless mode)
    widget.set_readonly(True)
    widget.set_text("world")
    assert widget.to_document().text == "world"
    assert widget.is_readonly()  # Still readonly
