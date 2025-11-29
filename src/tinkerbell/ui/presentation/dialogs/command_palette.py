"""Lightweight command palette for triggering window actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Mapping, Sequence

from ...models.actions import WindowAction

try:  # pragma: no cover - optional Qt dependency
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QDialog,
        QHBoxLayout,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QVBoxLayout,
    )

    _QT_AVAILABLE = True
except Exception:  # pragma: no cover - headless fallback
    Qt = None  # type: ignore[assignment]
    QDialog = None  # type: ignore[assignment]
    QHBoxLayout = None  # type: ignore[assignment]
    QLineEdit = None  # type: ignore[assignment]
    QListWidget = None  # type: ignore[assignment]
    QListWidgetItem = None  # type: ignore[assignment]
    QVBoxLayout = None  # type: ignore[assignment]
    _QT_AVAILABLE = False


@dataclass(slots=True)
class PaletteCommand:
    """Lightweight descriptor for palette entries."""

    command_id: str
    label: str
    detail: str
    shortcut: str | None
    callback: Callable[[], Any] | None

    def matches(self, query: str) -> bool:
        if not query:
            return True
        haystack = " ".join(
            part
            for part in (self.label, self.detail or "", self.shortcut or "")
            if part
        ).casefold()
        return all(token in haystack for token in query.split())


def build_palette_commands(
    actions: Mapping[str, WindowAction],
    *,
    exclude: Iterable[str] | None = None,
) -> list[PaletteCommand]:
    """Derive palette entries from window actions (headless friendly)."""

    excluded = {name.strip().lower() for name in (exclude or []) if name}
    entries: list[PaletteCommand] = []
    for name, action in actions.items():
        if name.strip().lower() in excluded:
            continue
        label = action.text or name.replace("_", " ").title()
        detail = action.status_tip or ""
        entry = PaletteCommand(
            command_id=name,
            label=label,
            detail=detail,
            shortcut=action.shortcut,
            callback=action.callback,
        )
        entries.append(entry)
    entries.sort(key=lambda entry: entry.label.casefold())
    return entries


class CommandPaletteDialog:
    """Simple searchable palette built on top of :class:`WindowAction`."""

    def __init__(
        self,
        *,
        parent: Any | None = None,
        enable_qt: bool | None = None,
    ) -> None:
        self._parent = parent
        if enable_qt is None:
            self._qt_enabled = bool(_QT_AVAILABLE)
        else:
            self._qt_enabled = bool(enable_qt and _QT_AVAILABLE)
        self._dialog: Any | None = None
        self._search_input: Any | None = None
        self._list_widget: Any | None = None
        self._entries: list[PaletteCommand] = []
        self._filtered: list[PaletteCommand] = []
        if self._qt_enabled:
            self._build_dialog()

    def set_entries(self, entries: Sequence[PaletteCommand]) -> None:
        self._entries = list(entries)
        self._filter_commands("")

    def show(self) -> None:
        if self._dialog is None:
            return
        self._filter_commands("")
        self._dialog.show()
        self._dialog.raise_()  # type: ignore[attr-defined]
        self._dialog.activateWindow()  # type: ignore[attr-defined]
        if self._search_input is not None:
            self._search_input.setFocus()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_dialog(self) -> None:
        if not self._qt_enabled or QDialog is None or QVBoxLayout is None:
            return
        dialog = QDialog(self._parent)
        dialog.setWindowTitle("Command Palette")
        dialog.setObjectName("tb-command-palette")
        dialog.setModal(False)
        dialog.setWindowFlag(Qt.WindowType.Tool, True)
        layout = QVBoxLayout(dialog)
        search_input = QLineEdit()
        search_input.setPlaceholderText("Search commands…")
        search_input.textChanged.connect(self._handle_query_changed)  # type: ignore[attr-defined]
        layout.addWidget(search_input)
        list_widget = QListWidget()
        list_widget.itemActivated.connect(self._handle_item_activated)  # type: ignore[attr-defined]
        layout.addWidget(list_widget)
        self._dialog = dialog
        self._search_input = search_input
        self._list_widget = list_widget

    def _handle_query_changed(self, text: str) -> None:
        query = text.casefold().strip()
        self._filter_commands(query)

    def _handle_item_activated(self, item: Any) -> None:
        if item is None:
            return
        command = item.data(Qt.ItemDataRole.UserRole) if Qt is not None else None
        if isinstance(command, PaletteCommand):
            self._invoke_command(command)

    def _filter_commands(self, query: str) -> None:
        self._filtered = [entry for entry in self._entries if entry.matches(query)]
        if not self._filtered and self._entries and query:
            self._filtered = [self._entries[0]]
        self._render_entries()

    def _render_entries(self) -> None:
        if self._list_widget is None:
            return
        self._list_widget.clear()
        for entry in self._filtered:
            label = entry.label
            if entry.shortcut:
                label = f"{label}    ({entry.shortcut})"
            item = QListWidgetItem(label)
            if entry.detail:
                item.setToolTip(entry.detail)
            if Qt is not None:
                item.setData(Qt.ItemDataRole.UserRole, entry)
            self._list_widget.addItem(item)
        if self._list_widget.count():
            self._list_widget.setCurrentRow(0)

    def _invoke_command(self, entry: PaletteCommand) -> None:
        callback = entry.callback
        if callable(callback):
            try:
                callback()
            except Exception:  # pragma: no cover - defensive guard
                pass
        if self._dialog is not None:
            self._dialog.close()


__all__ = ["CommandPaletteDialog", "PaletteCommand", "build_palette_commands"]
