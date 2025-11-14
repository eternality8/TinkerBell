"""Bridge tests."""

from tinkerbell.editor.document_model import DocumentState
from tinkerbell.services.bridge import DocumentBridge


class DummyEditor:
    def __init__(self) -> None:
        self.state = DocumentState(text="hello")

    def load_document(self, document: DocumentState) -> None:
        self.state = document

    def to_document(self) -> DocumentState:
        return self.state


def test_generate_snapshot():
    bridge = DocumentBridge(editor=DummyEditor())
    snapshot = bridge.generate_snapshot()
    assert snapshot["language"] == "markdown"
