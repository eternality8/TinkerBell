"""Wrapper around the text editor widget used in the main window."""

from __future__ import annotations

from typing import Optional, Protocol

from .document_model import DocumentState, SelectionRange

class QWidget:  # pragma: no cover - placeholder base class
    """Runtime placeholder avoiding PySide6 dependency during scaffolding."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple shim
        pass


class SnapshotListener(Protocol):
    """Protocol for callbacks interested in document snapshots."""

    def __call__(self, snapshot: dict) -> None:  # pragma: no cover - protocol
        ...


class EditorWidget(QWidget):
    """High-level widget orchestrating the text editor component."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        self._state = DocumentState()
        self._snapshot_listeners: list[SnapshotListener] = []

    def load_document(self, document: DocumentState) -> None:
        """Load a new document state into the widget."""

        self._state = document

    def to_document(self) -> DocumentState:
        """Return the current document representation."""

        return self._state

    def apply_selection(self, selection: SelectionRange) -> None:
        """Apply an external selection update to the widget."""

        self._state.selection = selection

    def request_snapshot(self, delta_only: bool = False) -> dict:
        """Notify listeners with the current document snapshot."""

        snapshot = self._state.snapshot(delta_only=delta_only)
        for listener in self._snapshot_listeners:
            listener(snapshot)
        return snapshot

    def add_snapshot_listener(self, listener: SnapshotListener) -> None:
        """Register a callback invoked whenever a snapshot is requested."""

        self._snapshot_listeners.append(listener)


# Syntax package -----------------------------------------------------------------
