"""Document bridge connecting AI directives to the editor widget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..chat.message_model import EditDirective
from ..editor.document_model import DocumentState


class EditorAdapter(Protocol):
    """Minimal interface consumed by the bridge."""

    def load_document(self, document: DocumentState) -> None:
        ...

    def to_document(self) -> DocumentState:
        ...


@dataclass(slots=True)
class DocumentBridge:
    """Orchestrates safe document snapshots and queued edits."""

    editor: EditorAdapter

    def generate_snapshot(self, *, delta_only: bool = False) -> dict:
        """Return a document snapshot for agent consumption."""

        return self.editor.to_document().snapshot(delta_only=delta_only)

    def queue_edit(self, directive: EditDirective) -> None:
        """Apply the provided directive through the editor adapter."""

        del directive
        # TODO: implement edit routing

