"""Main window scaffolding for TinkerBell."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from PySide6.QtWidgets import QMainWindow, QWidget
else:  # pragma: no cover - runtime fallback when PySide6 is absent

    class QMainWindow:  # type: ignore
        """Fallback placeholder when PySide6 is unavailable."""


    class QWidget:  # type: ignore
        """Fallback placeholder when PySide6 is unavailable."""


@dataclass(slots=True)
class WindowContext:
    """Shared context passed to the main window when constructing the UI."""

    settings: Optional[object] = None
    ai_controller: Optional[object] = None


class MainWindow(QMainWindow):
    """Primary application window hosting the editor and chat splitter."""

    def __init__(self, context: WindowContext):  # noqa: D401 - doc inherited
        super().__init__()
        self._context = context
        self._initialize_ui()

    def _initialize_ui(self) -> None:
        """Set up menus, toolbars, splitter layout, and status widgets."""

        raise NotImplementedError("Main window UI not implemented yet.")

    def open_document(self, path: str) -> None:
        """Open the provided document path inside the editor widget."""

        raise NotImplementedError("open_document not implemented yet.")

    def save_document(self) -> None:
        """Persist the current document to disk."""

        raise NotImplementedError("save_document not implemented yet.")
