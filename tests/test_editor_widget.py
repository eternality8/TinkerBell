"""Editor widget tests covering logical behaviors in headless mode."""

import pytest

from tinkerbell.chat.message_model import EditDirective
from tinkerbell.editor.document_model import DocumentState, SelectionRange
from tinkerbell.editor.editor_widget import EditorWidget
from tinkerbell.editor.patches import PatchResult


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
    selection = widget.to_document().selection
    assert selection.start == selection.end == len("hello world")

    replace = EditDirective(action="replace", target_range=(0, 5), content="hi")
    widget.apply_ai_edit(replace)
    selection = widget.to_document().selection
    assert selection.start == selection.end == len("hi")


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
    doc = DocumentState(text="content", selection=SelectionRange(0, 0))
    widget.load_document(doc)
    widget.apply_selection(SelectionRange(1, 4))
    assert widget.to_document().selection.start == 1
    assert widget.to_document().selection.end == 4


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

    selection = widget.to_document().selection
    assert selection.start == selection.end == 11
