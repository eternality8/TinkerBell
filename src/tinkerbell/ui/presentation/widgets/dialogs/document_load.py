"""Document load/open dialog with preview and token budgets."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
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

from tinkerbell.services.telemetry import count_text_tokens
from .common import (
    DEFAULT_FILE_FILTER,
    PREVIEW_CHAR_LIMIT,
    humanize_bytes,
    language_from_suffix,
)
from .sample_document import SampleDocument

__all__ = [
    "DocumentLoadDialog",
]


class DocumentLoadDialog(QDialog):
    """Custom open/import dialog with token budgets and preview support."""

    def __init__(
        self,
        *,
        parent: QWidget | None = None,
        caption: str = "Open Document",
        start_dir: Path | str | None = None,
        file_filter: str | None = None,
        token_budget: int | None = None,
        sample_documents: Sequence[SampleDocument] | None = None,
        preview_char_limit: int = PREVIEW_CHAR_LIMIT,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(caption)
        self.setModal(True)

        self._start_dir = Path(start_dir).expanduser() if start_dir else None
        self._file_filter = file_filter or DEFAULT_FILE_FILTER
        self._token_budget = token_budget
        self._preview_char_limit = max(512, preview_char_limit)
        self._sample_documents = tuple(sample_documents or ())
        self._selected_path: Path | None = None
        self._suppress_path_updates = False

        layout = QVBoxLayout(self)

        self._path_input = QLineEdit()
        self._path_input.setObjectName("document_path_input")
        self._path_input.setPlaceholderText("Select a file to open…")
        self._path_input.textChanged.connect(self._handle_path_changed)

        browse_button = QPushButton("Browse…")
        browse_button.setObjectName("document_browse_button")
        browse_button.clicked.connect(self._browse_for_path)

        path_row = QHBoxLayout()
        path_row.addWidget(self._path_input, 1)
        path_row.addWidget(browse_button, 0)
        layout.addLayout(path_row)

        if self._sample_documents:
            self._sample_combo = QComboBox()
            self._sample_combo.setObjectName("sample_document_combo")
            self._sample_combo.addItem("Load sample…", None)
            for sample in self._sample_documents:
                tooltip = sample.description or f"Sample {sample.language} file"
                self._sample_combo.addItem(sample.name, sample)
                index = self._sample_combo.count() - 1
                self._sample_combo.setItemData(index, tooltip, Qt.ItemDataRole.ToolTipRole)
            self._sample_combo.currentIndexChanged.connect(self._handle_sample_selected)
            layout.addWidget(self._sample_combo)
        else:
            self._sample_combo = None

        stats_row = QHBoxLayout()
        self._size_label = QLabel("Size: –")
        self._size_label.setObjectName("document_size_label")
        self._token_label = QLabel("Tokens: –")
        self._token_label.setObjectName("document_token_label")
        self._language_label = QLabel("Language: –")
        self._language_label.setObjectName("document_language_label")
        stats_row.addWidget(self._size_label)
        stats_row.addWidget(self._token_label)
        stats_row.addWidget(self._language_label)
        layout.addLayout(stats_row)

        self._budget_bar = QProgressBar()
        self._budget_bar.setObjectName("token_budget_bar")
        self._budget_bar.setRange(0, 100)
        self._budget_bar.setValue(0)
        self._budget_bar.setFormat("%p% of token budget")
        self._budget_bar.setVisible(token_budget is not None)
        layout.addWidget(self._budget_bar)

        preview_label = QLabel(f"Preview (first {self._preview_char_limit:,} chars)")
        preview_label.setObjectName("document_preview_label")
        layout.addWidget(preview_label)

        self._preview = QPlainTextEdit()
        self._preview.setObjectName("document_preview")
        self._preview.setReadOnly(True)
        self._preview.setPlaceholderText("Select a document to see a quick preview.")
        layout.addWidget(self._preview, 1)

        self._status_label = QLabel()
        self._status_label.setObjectName("document_preview_status")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Open | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._accept_button = buttons.button(QDialogButtonBox.StandardButton.Open)
        self._update_accept_state()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def selected_path(self) -> Path | None:
        return self._selected_path

    def preview_text(self) -> str:
        return self._preview.toPlainText()

    def select_path(self, path: Path) -> None:
        resolved = path.expanduser()
        self._suppress_path_updates = True
        try:
            self._path_input.setText(str(resolved))
        finally:
            self._suppress_path_updates = False
        self._load_preview_from_path(resolved)

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------
    def _handle_path_changed(self, text: str) -> None:
        if self._suppress_path_updates:
            return
        trimmed = text.strip()
        if not trimmed:
            self._selected_path = None
            self._clear_preview()
            self._update_accept_state()
            return
        path = Path(trimmed)
        if path.exists() and path.is_file():
            self._load_preview_from_path(path)
        else:
            self._selected_path = None
            self._set_status("Path does not exist or is not a file.")
            self._clear_preview()
            self._update_accept_state()

    def _browse_for_path(self) -> None:
        start_dir = str(self._start_dir) if self._start_dir else ""
        path, _ = QFileDialog.getOpenFileName(self, self.windowTitle(), start_dir, self._file_filter)
        if not path:
            return
        self.select_path(Path(path))

    def _handle_sample_selected(self, index: int) -> None:
        if self._sample_combo is None:
            return
        sample = self._sample_combo.itemData(index)
        if not isinstance(sample, SampleDocument):
            return
        self.select_path(sample.path)

    def _on_accept(self) -> None:
        if self._selected_path is None:
            self.reject()
            return
        self.accept()

    # ------------------------------------------------------------------
    # Preview helpers
    # ------------------------------------------------------------------
    def _load_preview_from_path(self, path: Path) -> None:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            self._selected_path = None
            self._clear_preview()
            self._set_status(f"Unable to read file: {exc}")
            self._update_accept_state()
            return

        self._selected_path = path
        self._set_status(f"Loaded preview from {path.name}")
        self._apply_preview(path=path, text=text)
        self._update_accept_state()

    def _apply_preview(self, *, path: Path, text: str) -> None:
        preview = text[: self._preview_char_limit]
        self._preview.setPlainText(preview)
        size_label = "Size: –"
        try:
            size_label = f"Size: {humanize_bytes(path.stat().st_size)}"
        except OSError:
            pass
        self._size_label.setText(size_label)
        token_count = count_text_tokens(text)
        self._token_label.setText(f"Tokens: {token_count:,}")
        language = language_from_suffix(path.suffix) or "plain"
        self._language_label.setText(f"Language: {language}")
        self._update_budget_bar(token_count)

    def _clear_preview(self) -> None:
        self._preview.clear()
        self._size_label.setText("Size: –")
        self._token_label.setText("Tokens: –")
        self._language_label.setText("Language: –")
        self._update_budget_bar(0)

    def _update_budget_bar(self, token_count: int) -> None:
        if self._token_budget is None or self._token_budget <= 0:
            self._budget_bar.setVisible(False)
            return
        self._budget_bar.setVisible(True)
        percent = min(100, int((token_count / self._token_budget) * 100)) if token_count else 0
        self._budget_bar.setValue(max(0, percent))
        self._budget_bar.setFormat(f"{token_count:,} tokens · {percent}% of budget {self._token_budget:,}")

    def _set_status(self, message: str) -> None:
        self._status_label.setText(message)

    def _update_accept_state(self) -> None:
        if self._accept_button is not None:
            self._accept_button.setEnabled(self._selected_path is not None)
