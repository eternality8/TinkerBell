"""Validation errors dialog for displaying error lists."""

from __future__ import annotations

from typing import Iterable

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)

__all__ = [
    "ValidationErrorsDialog",
    "show_validation_errors",
]


class ValidationErrorsDialog(QDialog):
    """Dialog presenting a list of validation errors to the user."""

    def __init__(self, errors: Iterable[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Validation Errors")
        self.setModal(True)

        layout = QVBoxLayout(self)
        header = QLabel("Please resolve the following issues:")
        header.setWordWrap(True)
        layout.addWidget(header)

        self._list = QListWidget()
        for error in errors:
            if not error:
                continue
            QListWidgetItem(error, self._list)
        if not self._list.count():
            QListWidgetItem("No validation errors reported.", self._list)
        layout.addWidget(self._list)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


def show_validation_errors(parent: QWidget | None = None, errors: Iterable[str] = ()) -> None:
    """Display the validation errors dialog if errors were supplied."""

    errors = list(errors)
    if not errors:
        return
    dialog = ValidationErrorsDialog(errors, parent)
    dialog.exec()
