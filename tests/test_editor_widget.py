"""Editor widget tests."""

from tinkerbell.editor.document_model import DocumentState
from tinkerbell.editor.editor_widget import EditorWidget


def test_editor_widget_snapshot_roundtrip():
    widget = EditorWidget()
    doc = DocumentState(text="sample")
    widget.load_document(doc)
    snapshot = widget.request_snapshot()
    assert snapshot["text"] == "sample"
