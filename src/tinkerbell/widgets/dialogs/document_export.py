"""Document export/save dialog with preview and token budgets."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...services.telemetry import count_text_tokens
from .common import DEFAULT_FILE_FILTER, PREVIEW_CHAR_LIMIT

__all__ = [
    "DocumentExportDialog",
]


class DocumentExportDialog(QDialog):
    """Save dialog showing document preview and token stats."""

    def __init__(
        self,
        *,
        parent: QWidget | None = None,
        caption: str = "Save Document",
        start_dir: Path | str | None = None,
        file_filter: str | None = None,
        default_suffix: str = "md",
        document_text: str | None = None,
        token_budget: int | None = None,
        preview_char_limit: int = PREVIEW_CHAR_LIMIT,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(caption)
        self.setModal(True)

        self._start_dir = Path(start_dir).expanduser() if start_dir else None
        self._file_filter = file_filter or DEFAULT_FILE_FILTER
        self._default_suffix = default_suffix.lstrip(".")
        self._token_budget = token_budget
        self._preview_char_limit = max(512, preview_char_limit)
        self._document_text = document_text or ""

        layout = QVBoxLayout(self)

        self._path_input = QLineEdit()
        self._path_input.setObjectName("save_path_input")
        self._path_input.setPlaceholderText("Choose where to save the document…")
        self._path_input.textChanged.connect(self._update_accept_state)

        browse_button = QPushButton("Browse…")
        browse_button.setObjectName("save_browse_button")
        browse_button.clicked.connect(self._browse_for_path)

        path_row = QHBoxLayout()
        path_row.addWidget(self._path_input, 1)
        path_row.addWidget(browse_button, 0)
        layout.addLayout(path_row)

        stats_row = QHBoxLayout()
        self._document_tokens = count_text_tokens(self._document_text) if self._document_text else 0
        doc_label = QLabel(f"Document: {self._document_tokens:,} tokens")
        doc_label.setObjectName("document_stats_label")
        self._preview_stats_label = QLabel(self._preview_label_text())
        self._preview_stats_label.setObjectName("preview_stats_label")
        stats_row.addWidget(doc_label)
        stats_row.addWidget(self._preview_stats_label)
        layout.addLayout(stats_row)

        self._budget_bar = QProgressBar()
        self._budget_bar.setObjectName("save_budget_bar")
        self._budget_bar.setRange(0, 100)
        self._budget_bar.setVisible(token_budget is not None)
        layout.addWidget(self._budget_bar)
        self._update_budget_bar()

        self._preview = QPlainTextEdit()
        self._preview.setObjectName("save_preview")
        self._preview.setReadOnly(True)
        layout.addWidget(self._preview, 1)

        self._apply_preview_text(self._active_preview_text())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        self._update_accept_state()

    def selected_path(self) -> Path | None:
        text = self._path_input.text().strip()
        if not text:
            return None
        path = Path(text)
        if self._default_suffix and not path.suffix:
            path = path.with_suffix(f".{self._default_suffix}")
        return path

    def _preview_label_text(self) -> str:
        return f"Document preview: {self._current_preview_tokens():,} tokens"

    def _active_preview_text(self) -> str:
        return self._document_text

    def _current_preview_tokens(self) -> int:
        return self._document_tokens

    def _apply_preview_text(self, text: str) -> None:
        preview = text[: self._preview_char_limit]
        if not preview and self._document_text:
            preview = self._document_text[: self._preview_char_limit]
        self._preview.setPlainText(preview)

    def _browse_for_path(self) -> None:
        start_dir = str(self._start_dir) if self._start_dir else ""
        path, _ = QFileDialog.getSaveFileName(self, self.windowTitle(), start_dir, self._file_filter)
        if not path:
            return
        self._path_input.setText(path)

    def _update_budget_bar(self) -> None:
        if self._token_budget is None or self._token_budget <= 0:
            self._budget_bar.setVisible(False)
            return
        preview_tokens = self._current_preview_tokens()
        percent = min(100, int((preview_tokens / self._token_budget) * 100)) if preview_tokens else 0
        self._budget_bar.setVisible(True)
        self._budget_bar.setValue(percent)
        self._budget_bar.setFormat(
            f"{preview_tokens:,} preview tokens · {percent}% of budget {self._token_budget:,}"
        )

    def _refresh_preview_stats(self) -> None:
        self._preview_stats_label.setText(self._preview_label_text())
        self._update_budget_bar()

    def _on_accept(self) -> None:
        if self.selected_path() is None:
            self.reject()
            return
        self.accept()

    def _update_accept_state(self) -> None:
        if self._save_button is not None:
            self._save_button.setEnabled(bool(self._path_input.text().strip()))
