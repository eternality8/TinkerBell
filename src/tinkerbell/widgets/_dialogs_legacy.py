"""Dialogs for file operations and settings."""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlparse

import httpx

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..services.settings import (
    ContextPolicySettings,
    DEFAULT_EMBEDDING_MODE,
    EMBEDDING_MODE_CHOICES,
    Settings,
)
from ..services.telemetry import count_text_tokens
from ..theme import theme_manager

if TYPE_CHECKING:  # pragma: no cover - type-checking only
    from ..ui.embedding_controller import EmbeddingValidationResult

__all__ = [
    "DEFAULT_FILE_FILTER",
    "SampleDocument",
    "discover_sample_documents",
    "SettingsDialog",
    "SettingsDialogResult",
    "ValidationErrorsDialog",
    "test_ai_api_settings",
    "open_file_dialog",
    "save_file_dialog",
    "show_settings_dialog",
    "show_validation_errors",
]

DEFAULT_FILE_FILTER = "Markdown / Text (*.md *.markdown *.mdx *.txt *.json *.yaml *.yml);;All Files (*)"
_MODEL_SUGGESTIONS = ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "o4-mini")
_EMBEDDING_MODE_LABELS: Mapping[str, str] = {
    "disabled": "Disabled",
    "same-api": "Same API as Chat Model",
    "custom-api": "Separate OpenAI-Compatible API",
    "local": "Local (SentenceTransformers)",
}
_HINT_COLORS = {
    "info": "#6a737d",
    "success": "#1a7f37",
    "warning": "#b08800",
    "error": "#d73a49",
}
_SAMPLE_LANG_MAP = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".txt": "text",
}
_DEFAULT_SAMPLE_LIMIT = 12
_PREVIEW_CHAR_LIMIT = 3200


@dataclass(slots=True)
class SampleDocument:
    """Metadata describing a built-in sample document shipped with the app."""

    name: str
    path: Path
    language: str = "markdown"
    description: str | None = None


def discover_sample_documents(limit: int | None = _DEFAULT_SAMPLE_LIMIT) -> tuple[SampleDocument, ...]:
    """Return available large-file samples bundled with the repository."""

    root = _project_root()
    search_dirs = [root / "test_data", root / "assets" / "sample_docs"]
    entries: list[SampleDocument] = []
    seen: set[Path] = set()

    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        try:
            files = sorted(directory.iterdir(), key=lambda item: item.name.lower())
        except OSError:
            continue
        for candidate in files:
            if not candidate.is_file():
                continue
            language = _language_from_suffix(candidate.suffix)
            if language is None:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            size_hint = _humanize_bytes(candidate.stat().st_size)
            label = f"{candidate.name} ({size_hint})"
            description = f"Sample {language} document"
            entries.append(SampleDocument(name=label, path=resolved, language=language, description=description))
            seen.add(resolved)
            if limit is not None and len(entries) >= limit:
                return tuple(entries)
    return tuple(entries)


def _project_root() -> Path:
    current = Path(__file__).resolve()
    parents = list(current.parents)
    for parent in parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return parents[-1] if parents else current.parent


def _language_from_suffix(suffix: str) -> str | None:
    if not suffix:
        return None
    return _SAMPLE_LANG_MAP.get(suffix.lower())


def _humanize_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    count = float(max(0, size))
    unit_index = 0
    while count >= 1024 and unit_index < len(units) - 1:
        count /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(count)} {units[unit_index]}"
    return f"{count:.1f} {units[unit_index]}"


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
        preview_char_limit: int = _PREVIEW_CHAR_LIMIT,
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
            size_label = f"Size: {_humanize_bytes(path.stat().st_size)}"
        except OSError:
            pass
        self._size_label.setText(size_label)
        token_count = count_text_tokens(text)
        self._token_label.setText(f"Tokens: {token_count:,}")
        language = _language_from_suffix(path.suffix) or "plain"
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
        preview_char_limit: int = _PREVIEW_CHAR_LIMIT,
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


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a set of settings."""

    ok: bool
    message: str = ""


SettingsValidator = Callable[[Settings], ValidationResult | tuple[bool, str] | bool]
SettingsTester = Callable[[Settings], ValidationResult | tuple[bool, str] | bool]


@dataclass(slots=True)
class SettingsDialogResult:
    """Outcome returned when the settings dialog closes."""

    accepted: bool
    settings: Settings
    validated: bool = False
    validation_message: str | None = None
    api_tested: bool = False
    api_test_message: str | None = None
    embedding_tested: bool = False
    embedding_test_message: str | None = None


def open_file_dialog(
    parent: QWidget | None = None,
    *,
    caption: str = "Open Document",
    start_dir: Path | str | None = None,
    file_filter: str | None = None,
    token_budget: int | None = None,
    enable_samples: bool = True,
    sample_documents: Sequence[SampleDocument] | None = None,
    preview_char_limit: int = _PREVIEW_CHAR_LIMIT,
) -> Path | None:
    """Display the enhanced open/import dialog with previews and token budgets."""

    samples: Sequence[SampleDocument] | None = sample_documents
    if enable_samples and samples is None:
        samples = discover_sample_documents()

    dialog = DocumentLoadDialog(
        parent=parent,
        caption=caption,
        start_dir=start_dir,
        file_filter=file_filter,
        token_budget=token_budget,
        sample_documents=samples,
        preview_char_limit=preview_char_limit,
    )
    accepted = dialog.exec() == int(QDialog.DialogCode.Accepted)
    if not accepted:
        return None
    return dialog.selected_path()


def save_file_dialog(
    parent: QWidget | None = None,
    *,
    caption: str = "Save Document",
    start_dir: Path | str | None = None,
    file_filter: str | None = None,
    default_suffix: str = "md",
    document_text: str | None = None,
    token_budget: int | None = None,
    preview_char_limit: int = _PREVIEW_CHAR_LIMIT,
) -> Path | None:
    """Display the enhanced save/export dialog with document previews."""

    dialog = DocumentExportDialog(
        parent=parent,
        caption=caption,
        start_dir=start_dir,
        file_filter=file_filter,
        default_suffix=default_suffix,
        document_text=document_text,
        token_budget=token_budget,
        preview_char_limit=preview_char_limit,
    )
    accepted = dialog.exec() == int(QDialog.DialogCode.Accepted)
    if not accepted:
        return None
    selected = dialog.selected_path()
    if selected is None:
        return None
    resolved = Path(selected)
    if default_suffix and not resolved.suffix:
        resolved = resolved.with_suffix(f".{default_suffix}")
    return resolved


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


class SettingsDialog(QDialog):
    """Modal dialog allowing users to configure API connectivity settings."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        parent: QWidget | None = None,
        available_models: Sequence[str] | None = None,
        validator: SettingsValidator | None = None,
        api_tester: SettingsTester | None = None,
        embedding_tester: SettingsTester | None = None,
        show_toasts: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Settings")
        self.setModal(True)

        self._original = settings or Settings()
        self._result = self._original
        self._validator = validator
        self._api_tester = api_tester
        self._last_validation = ValidationResult(ok=False, message="")
        self._last_api_test = ValidationResult(ok=False, message="")
        self._model_choices = tuple(available_models or _MODEL_SUGGESTIONS)
        self._toast_enabled = show_toasts
        self._field_errors: dict[str, str] = {}
        self._active_toasts: list[QMessageBox] = []
        self._save_button: QPushButton | None = None
        self._embedding_tester = embedding_tester
        self._last_embedding_test = ValidationResult(ok=False, message="")

        metadata = dict(getattr(self._original, "metadata", {}) or {})
        self._metadata_snapshot = metadata
        self._embedding_api_metadata = dict(metadata.get("embedding_api") or {})
        raw_mode = str(metadata.get("embedding_mode") or DEFAULT_EMBEDDING_MODE)
        self._initial_embedding_mode = raw_mode if raw_mode in EMBEDDING_MODE_CHOICES else DEFAULT_EMBEDDING_MODE

        self._base_url_input = QLineEdit(self._original.base_url)
        self._base_url_input.setObjectName("base_url_input")
        self._base_url_input.setPlaceholderText("https://api.openai.com/v1")
        self._base_url_hint = QLabel("HTTPS endpoint, e.g., https://api.openai.com/v1")
        self._base_url_hint.setObjectName("base_url_hint")
        self._prepare_hint_label(self._base_url_hint)
        self._base_url_input.textChanged.connect(self._update_base_url_hint)
        self._api_key_input = QLineEdit(self._original.api_key)
        self._api_key_input.setObjectName("api_key_input")
        self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._model_combo = QComboBox()
        self._model_combo.setObjectName("model_combo")
        self._model_combo.setEditable(True)
        seen = set()
        for model in self._model_choices:
            if model in seen:
                continue
            seen.add(model)
            self._model_combo.addItem(model)
        if self._original.model:
            index = self._model_combo.findText(self._original.model)
            if index >= 0:
                self._model_combo.setCurrentIndex(index)
            else:
                self._model_combo.setEditText(self._original.model)
        self._temperature_input = QDoubleSpinBox()
        self._temperature_input.setObjectName("temperature_input")
        self._temperature_input.setRange(0.0, 2.0)
        self._temperature_input.setSingleStep(0.05)
        self._temperature_input.setDecimals(2)
        temp_default = float(getattr(self._original, "temperature", 0.2) or 0.0)
        temp_default = max(0.0, min(temp_default, 2.0))
        self._temperature_input.setValue(temp_default)
        self._temperature_input.setToolTip("Sampling temperature for chat completions.")
        self._temperature_hint = QLabel("0 = deterministic, 1+ trades accuracy for creativity.")
        self._temperature_hint.setObjectName("temperature_hint")
        self._prepare_hint_label(self._temperature_hint)
        self._temperature_input.valueChanged.connect(self._update_temperature_hint)
        self._organization_input = QLineEdit(self._original.organization or "")
        self._organization_input.setObjectName("organization_input")
        self._theme_combo = QComboBox()
        self._theme_combo.setObjectName("theme_combo")
        self._theme_combo.setEditable(False)
        self._populate_theme_combo(self._original.theme)
        # Embedding model input (shown only for API modes)
        embedding_model_default = getattr(self._original, "embedding_model_name", "text-embedding-3-large") or "text-embedding-3-large"
        self._embedding_model_input = QLineEdit(embedding_model_default)
        self._embedding_model_input.setObjectName("embedding_model_input")
        self._embedding_model_input.textChanged.connect(self._validate_embedding_fields)
        self._embedding_model_hint = QLabel("Model name for the embedding API.")
        self._embedding_model_hint.setObjectName("embedding_model_hint")
        self._prepare_hint_label(self._embedding_model_hint)
        # Embedding mode combo (single dropdown for all modes)
        self._embedding_mode_combo = QComboBox()
        self._embedding_mode_combo.setObjectName("embedding_mode_combo")
        for mode in EMBEDDING_MODE_CHOICES:
            label = _EMBEDDING_MODE_LABELS.get(mode, mode.replace("-", " ").title())
            self._embedding_mode_combo.addItem(label, mode)
        mode_index = self._embedding_mode_combo.findData(self._initial_embedding_mode)
        self._embedding_mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 0)
        self._embedding_mode_combo.currentIndexChanged.connect(self._handle_embedding_mode_changed)
        self._embedding_mode_hint = QLabel("Choose how embeddings are generated for semantic search.")
        self._embedding_mode_hint.setObjectName("embedding_mode_hint")
        self._prepare_hint_label(self._embedding_mode_hint)
        # Mode-specific settings panels
        self._embedding_mode_stack = QStackedWidget()
        self._embedding_mode_stack.setObjectName("embedding_mode_stack")
        self._mode_index_map: dict[str, int] = {}
        disabled_panel = self._build_disabled_panel()
        same_panel = self._build_same_api_panel()
        custom_panel = self._build_custom_api_panel()
        local_panel = self._build_local_embedding_panel()
        for index, (mode, panel) in enumerate(
            (
                ("disabled", disabled_panel),
                ("same-api", same_panel),
                ("custom-api", custom_panel),
                ("local", local_panel),
            )
        ):
            self._mode_index_map[mode] = index
            self._embedding_mode_stack.addWidget(panel)
        self._debug_checkbox = QCheckBox("Enable debug logging")
        self._debug_checkbox.setObjectName("debug_logging_checkbox")
        self._debug_checkbox.setChecked(self._original.debug_logging)
        self._event_log_checkbox = QCheckBox("Capture per-chat event logs (JSONL)")
        self._event_log_checkbox.setObjectName("debug_event_logging_checkbox")
        self._event_log_checkbox.setToolTip(
            "Writes turn-by-turn JSONL files when debug logging is enabled for diagnosing tool issues."
        )
        self._event_log_checkbox.setChecked(bool(getattr(self._original, "debug_event_logging", False)))
        self._tool_panel_checkbox = QCheckBox("Show tool activity panel (debug view)")
        self._tool_panel_checkbox.setObjectName("tool_activity_checkbox")
        self._tool_panel_checkbox.setChecked(
            bool(getattr(self._original, "show_tool_activity_panel", False))
        )
        self._max_tool_iterations_input = QSpinBox()
        self._max_tool_iterations_input.setObjectName("max_tool_iterations_input")
        self._max_tool_iterations_input.setRange(1, 200)
        self._max_tool_iterations_input.setValue(
            max(1, int(getattr(self._original, "max_tool_iterations", 8) or 8))
        )
        self._max_tool_iterations_input.setSuffix(" loops")
        self._max_tool_iterations_input.setToolTip(
            "Maximum times the agent may invoke tools before returning a response."
        )
        self._max_context_tokens_input = QSpinBox()
        self._max_context_tokens_input.setObjectName("max_context_tokens_input")
        self._max_context_tokens_input.setRange(32_000, 512_000)
        self._max_context_tokens_input.setSingleStep(1_000)
        context_default = int(getattr(self._original, "max_context_tokens", 128_000) or 128_000)
        context_default = max(32_000, min(context_default, 512_000))
        self._max_context_tokens_input.setValue(context_default)
        self._max_context_tokens_input.setSuffix(" tokens")
        self._max_context_tokens_input.setToolTip(
            "Total prompt tokens (system + history + user) allowed before reserving response space."
        )
        self._max_context_tokens_input.valueChanged.connect(self._update_reserve_hint)
        self._response_token_reserve_input = QSpinBox()
        self._response_token_reserve_input.setObjectName("response_token_reserve_input")
        self._response_token_reserve_input.setRange(4_000, 64_000)
        self._response_token_reserve_input.setSingleStep(500)
        reserve_default = int(getattr(self._original, "response_token_reserve", 16_000) or 16_000)
        reserve_default = max(4_000, min(reserve_default, 64_000))
        self._response_token_reserve_input.setValue(reserve_default)
        self._response_token_reserve_input.setSuffix(" tokens")
        self._response_token_reserve_input.setToolTip(
            "Tokens held back for the assistant's reply so streaming never truncates early."
        )
        self._reserve_hint = QLabel("Reserve must stay below the max context tokens.")
        self._reserve_hint.setObjectName("response_reserve_hint")
        self._prepare_hint_label(self._reserve_hint)
        self._response_token_reserve_input.valueChanged.connect(self._update_reserve_hint)
        self._request_timeout_input = QDoubleSpinBox()
        self._request_timeout_input.setObjectName("request_timeout_input")
        self._request_timeout_input.setRange(5.0, 600.0)
        self._request_timeout_input.setSingleStep(5.0)
        self._request_timeout_input.setSuffix(" s")
        self._request_timeout_input.setDecimals(1)
        timeout_value = float(getattr(self._original, "request_timeout", 90.0) or 90.0)
        self._request_timeout_input.setValue(max(5.0, min(timeout_value, 600.0)))
        self._request_timeout_input.setToolTip(
            "Maximum seconds to wait for AI responses before timing out."
        )
        self._timeout_hint = QLabel("Applies to chat + tool turns; keep between 5–300 seconds.")
        self._timeout_hint.setObjectName("request_timeout_hint")
        self._prepare_hint_label(self._timeout_hint)
        self._request_timeout_input.valueChanged.connect(self._update_timeout_hint)
        policy_original = getattr(self._original, "context_policy", ContextPolicySettings())
        self._context_policy_enabled = QCheckBox("Enable context budget policy")
        self._context_policy_enabled.setObjectName("context_policy_enabled_checkbox")
        self._context_policy_enabled.setChecked(bool(getattr(policy_original, "enabled", False)))
        self._context_policy_dry_run = QCheckBox("Dry run only (log decisions, do not block)")
        self._context_policy_dry_run.setObjectName("context_policy_dry_run_checkbox")
        self._context_policy_dry_run.setChecked(bool(getattr(policy_original, "dry_run", True)))
        self._context_prompt_override_toggle = QCheckBox("Custom prompt budget")
        self._context_prompt_override_toggle.setObjectName("context_policy_prompt_override_toggle")
        has_prompt_override = getattr(policy_original, "prompt_budget_override", None)
        self._context_prompt_override_toggle.setChecked(has_prompt_override is not None)
        self._context_prompt_override_input = QSpinBox()
        self._context_prompt_override_input.setObjectName("context_policy_prompt_override_input")
        self._context_prompt_override_input.setRange(32_000, 512_000)
        self._context_prompt_override_input.setSingleStep(1_000)
        self._context_prompt_override_input.setSuffix(" tokens")
        self._context_prompt_override_input.setValue(
            int(has_prompt_override or getattr(self._original, "max_context_tokens", 128_000))
        )
        self._context_prompt_override_input.setEnabled(has_prompt_override is not None)
        self._context_reserve_override_toggle = QCheckBox("Custom response reserve")
        self._context_reserve_override_toggle.setObjectName("context_policy_reserve_override_toggle")
        has_reserve_override = getattr(policy_original, "response_reserve_override", None)
        self._context_reserve_override_toggle.setChecked(has_reserve_override is not None)
        self._context_reserve_override_input = QSpinBox()
        self._context_reserve_override_input.setObjectName("context_policy_reserve_override_input")
        self._context_reserve_override_input.setRange(2_000, 64_000)
        self._context_reserve_override_input.setSingleStep(500)
        self._context_reserve_override_input.setSuffix(" tokens")
        reserve_seed = has_reserve_override or getattr(self._original, "response_token_reserve", 16_000)
        self._context_reserve_override_input.setValue(int(reserve_seed))
        self._context_reserve_override_input.setEnabled(has_reserve_override is not None)
        self._context_policy_hint = QLabel("Context policy runs budgeting checks before calling the model.")
        self._context_policy_hint.setObjectName("context_policy_hint_label")
        self._prepare_hint_label(self._context_policy_hint)
        self._context_policy_enabled.toggled.connect(self._update_context_policy_hint)
        self._context_policy_dry_run.toggled.connect(self._update_context_policy_hint)
        self._context_prompt_override_toggle.toggled.connect(self._toggle_context_prompt_override)
        self._context_reserve_override_toggle.toggled.connect(self._toggle_context_reserve_override)

        # Build grouped form sections for the tabbed layout
        base_url_container = QWidget()
        base_url_layout = QVBoxLayout(base_url_container)
        base_url_layout.setContentsMargins(0, 0, 0, 0)
        base_url_layout.setSpacing(2)
        base_url_layout.addWidget(self._base_url_input)
        base_url_layout.addWidget(self._base_url_hint)

        api_container = QWidget()
        api_layout = QVBoxLayout(api_container)
        api_layout.setContentsMargins(0, 0, 0, 0)
        api_layout.addWidget(self._api_key_input)
        show_checkbox = QCheckBox("Show API key")
        show_checkbox.setObjectName("show_api_checkbox")
        show_checkbox.toggled.connect(self._toggle_api_visibility)
        api_layout.addWidget(show_checkbox)

        temperature_container = QWidget()
        temperature_layout = QVBoxLayout(temperature_container)
        temperature_layout.setContentsMargins(0, 0, 0, 0)
        temperature_layout.setSpacing(2)
        temperature_layout.addWidget(self._temperature_input)
        temperature_layout.addWidget(self._temperature_hint)

        embedding_mode_container = QWidget()
        embedding_mode_layout = QVBoxLayout(embedding_mode_container)
        embedding_mode_layout.setContentsMargins(0, 0, 0, 0)
        embedding_mode_layout.setSpacing(2)
        embedding_mode_layout.addWidget(self._embedding_mode_combo)
        embedding_mode_layout.addWidget(self._embedding_mode_hint)

        self._embedding_model_container = QWidget()
        embedding_model_layout = QVBoxLayout(self._embedding_model_container)
        embedding_model_layout.setContentsMargins(0, 0, 0, 0)
        embedding_model_layout.setSpacing(2)
        embedding_model_layout.addWidget(self._embedding_model_input)
        embedding_model_layout.addWidget(self._embedding_model_hint)

        embedding_mode_stack_container = QWidget()
        embedding_mode_stack_layout = QVBoxLayout(embedding_mode_stack_container)
        embedding_mode_stack_layout.setContentsMargins(0, 0, 0, 0)
        embedding_mode_stack_layout.setSpacing(4)
        embedding_mode_stack_layout.addWidget(self._embedding_mode_stack)

        reserve_container = QWidget()
        reserve_layout = QVBoxLayout(reserve_container)
        reserve_layout.setContentsMargins(0, 0, 0, 0)
        reserve_layout.setSpacing(2)
        reserve_layout.addWidget(self._response_token_reserve_input)
        reserve_layout.addWidget(self._reserve_hint)

        timeout_container = QWidget()
        timeout_layout = QVBoxLayout(timeout_container)
        timeout_layout.setContentsMargins(0, 0, 0, 0)
        timeout_layout.setSpacing(2)
        timeout_layout.addWidget(self._request_timeout_input)
        timeout_layout.addWidget(self._timeout_hint)

        policy_container = QWidget()
        policy_layout = QVBoxLayout(policy_container)
        policy_layout.setContentsMargins(0, 0, 0, 0)
        policy_layout.setSpacing(4)
        policy_layout.addWidget(self._context_policy_enabled)
        policy_layout.addWidget(self._context_policy_dry_run)
        prompt_row = QHBoxLayout()
        prompt_row.addWidget(self._context_prompt_override_toggle)
        prompt_row.addWidget(self._context_prompt_override_input)
        policy_layout.addLayout(prompt_row)
        reserve_row = QHBoxLayout()
        reserve_row.addWidget(self._context_reserve_override_toggle)
        reserve_row.addWidget(self._context_reserve_override_input)
        policy_layout.addLayout(reserve_row)
        policy_layout.addWidget(self._context_policy_hint)

        def _build_form_tab(rows: list[tuple[str, QWidget | Any]]) -> QWidget:
            tab = QWidget()
            form = QFormLayout(tab)
            form.setContentsMargins(0, 0, 0, 0)
            form.setSpacing(8)
            for label, widget in rows:
                form.addRow(label, widget)
            return tab

        general_tab = _build_form_tab(
            [
                ("Base URL", base_url_container),
                ("API Key", api_container),
                ("Model", self._model_combo),
                ("Temperature", temperature_container),
                ("Organization", self._organization_input),
                ("Theme", self._theme_combo),
            ]
        )

        embedding_tab = _build_form_tab(
            [
                ("Mode", embedding_mode_container),
                ("Embedding Model", self._embedding_model_container),
                ("Settings", embedding_mode_stack_container),
            ]
        )

        features_tab = _build_form_tab(
            [
                ("Debug", self._debug_checkbox),
                ("Event Logs", self._event_log_checkbox),
                ("Tool Traces", self._tool_panel_checkbox),
            ]
        )

        runtime_tab = _build_form_tab(
            [
                ("Max Tool Iterations", self._max_tool_iterations_input),
                ("Max Context Tokens", self._max_context_tokens_input),
                ("Response Token Reserve", reserve_container),
                ("AI Timeout", timeout_container),
            ]
        )

        policy_tab = _build_form_tab([("Context Budget Policy", policy_container)])

        tab_widget = QTabWidget()
        tab_widget.setObjectName("settings_tab_widget")
        tab_widget.addTab(general_tab, "General")
        tab_widget.addTab(embedding_tab, "Embeddings")
        tab_widget.addTab(features_tab, "Features")
        tab_widget.addTab(runtime_tab, "Runtime")
        tab_widget.addTab(policy_tab, "Policy")

        layout = QVBoxLayout(self)
        layout.addWidget(tab_widget, 1)

        validation_row = QHBoxLayout()
        self._validation_label = QLabel()
        self._validation_label.setObjectName("validation_label")
        self._validation_label.setWordWrap(True)
        self._validation_label.setVisible(False)
        self._validation_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        validation_row.addWidget(self._validation_label, 1)

        self._validate_button = QPushButton("Validate Key")
        self._validate_button.setObjectName("validate_button")
        self._validate_button.clicked.connect(self._run_validation)
        self._validate_button.setEnabled(self._validator is not None)
        validation_row.addWidget(self._validate_button, 0)

        self._test_button = QPushButton("Test Connection")
        self._test_button.setObjectName("test_button")
        self._test_button.clicked.connect(self._run_api_test)
        self._test_button.setEnabled(self._api_tester is not None)
        validation_row.addWidget(self._test_button, 0)
        self._embedding_test_button = QPushButton("Test Embeddings")
        self._embedding_test_button.setObjectName("embedding_test_button")
        self._embedding_test_button.clicked.connect(self._run_embedding_test)
        self._embedding_test_button.setEnabled(self._embedding_tester is not None)
        validation_row.addWidget(self._embedding_test_button, 0)
        layout.addLayout(validation_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._save_button = buttons.button(QDialogButtonBox.StandardButton.Save)

        self._update_base_url_hint()
        self._update_reserve_hint()
        self._update_timeout_hint()
        self._update_temperature_hint()
        self._update_context_policy_hint()
        self._handle_embedding_mode_changed()
        self._validate_embedding_fields()
        self._update_buttons_state()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def settings(self) -> Settings:
        """Return the settings captured by the dialog."""

        return self._result

    @property
    def validated(self) -> bool:
        """Indicate whether the last validation run succeeded."""

        return self._last_validation.ok

    @property
    def last_validation(self) -> ValidationResult:
        """Return the status of the most recent validation attempt."""

        return self._last_validation

    @property
    def api_tested(self) -> bool:
        """Indicate whether the last API test succeeded."""

        return self._last_api_test.ok

    @property
    def last_api_test(self) -> ValidationResult:
        """Return the status of the most recent API test."""

        return self._last_api_test

    @property
    def embedding_tested(self) -> bool:
        """Indicate whether the last embedding test succeeded."""

        return self._last_embedding_test.ok

    @property
    def last_embedding_test(self) -> ValidationResult:
        """Return the status of the most recent embedding test."""

        return self._last_embedding_test

    def gather_settings(self) -> Settings:
        """Assemble a `Settings` instance from the current form widgets."""

        base_url = self._base_url_input.text().strip() or self._original.base_url
        api_key = self._api_key_input.text().strip()
        model = self._model_combo.currentText().strip() or self._original.model
        organization = self._organization_input.text().strip() or None
        theme = self._current_theme_name()
        temperature = float(self._temperature_input.value())
        embedding_mode = self._embedding_mode_value()
        # Derive embedding_backend from mode for backwards compatibility
        if embedding_mode == "disabled":
            embedding_backend = "disabled"
        elif embedding_mode == "local":
            embedding_backend = "sentence-transformers"
        else:
            # Both same-api and custom-api use langchain for flexibility
            embedding_backend = "langchain"
        embedding_model = self._embedding_model_input.text().strip() or self._original.embedding_model_name
        debug_logging = self._debug_checkbox.isChecked()
        debug_event_logging = self._event_log_checkbox.isChecked()
        show_tool_activity_panel = self._tool_panel_checkbox.isChecked()
        max_tool_iterations = int(self._max_tool_iterations_input.value())
        max_context_tokens = int(self._max_context_tokens_input.value())
        response_token_reserve = int(self._response_token_reserve_input.value())
        request_timeout = float(self._request_timeout_input.value())
        context_policy = self._gather_context_policy_settings()
        metadata = self._build_embedding_metadata()
        return replace(
            self._original,
            base_url=base_url,
            api_key=api_key,
            model=model,
            organization=organization,
            theme=theme,
            temperature=temperature,
            embedding_backend=embedding_backend,
            embedding_model_name=embedding_model,
            debug_logging=debug_logging,
            debug_event_logging=debug_event_logging,
            show_tool_activity_panel=show_tool_activity_panel,
            max_tool_iterations=max_tool_iterations,
            max_context_tokens=max_context_tokens,
            response_token_reserve=response_token_reserve,
            request_timeout=request_timeout,
            context_policy=context_policy,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_embedding_metadata(self) -> dict[str, Any]:
        metadata = dict(self._metadata_snapshot)
        mode = self._embedding_mode_value()
        metadata["embedding_mode"] = mode
        if mode == "custom-api":
            metadata["embedding_api"] = self._collect_custom_api_metadata()
        elif "embedding_api" in metadata:
            metadata["embedding_api"] = dict(self._embedding_api_metadata)
        if mode == "local":
            for key, value in self._collect_local_embedding_metadata().items():
                if value in (None, ""):
                    metadata.pop(key, None)
                else:
                    metadata[key] = value
        return metadata

    def _build_disabled_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label = QLabel("Embeddings disabled. Semantic search will fall back to regex and outline matching.")
        self._prepare_hint_label(label)
        layout.addWidget(label)
        return container

    def _build_same_api_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        label = QLabel("Reuse the chat API credentials for embedding calls.")
        self._prepare_hint_label(label)
        layout.addWidget(label)
        return container

    def _build_custom_api_panel(self) -> QWidget:
        container = QWidget()
        layout = QFormLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        base_url = self._embedding_api_metadata.get("base_url", "") or ""
        self._custom_api_base_url_input = QLineEdit(str(base_url))
        self._custom_api_base_url_input.setObjectName("embedding_custom_base_url_input")
        self._custom_api_base_url_input.setPlaceholderText("https://api.example.com/v1")
        self._custom_api_base_url_input.textChanged.connect(self._validate_embedding_fields)
        layout.addRow("Endpoint", self._custom_api_base_url_input)

        api_key = self._embedding_api_metadata.get("api_key", "") or ""
        hint = self._embedding_api_metadata.get("api_key_hint", "") or ""
        self._custom_api_key_input = QLineEdit(str(api_key))
        self._custom_api_key_input.setObjectName("embedding_custom_api_key_input")
        self._custom_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        if not api_key and hint:
            self._custom_api_key_input.setPlaceholderText(f"Stored secret ({hint})")
        self._custom_api_key_input.textChanged.connect(self._validate_embedding_fields)
        api_key_container = QWidget()
        api_key_layout = QVBoxLayout(api_key_container)
        api_key_layout.setContentsMargins(0, 0, 0, 0)
        api_key_layout.setSpacing(2)
        api_key_layout.addWidget(self._custom_api_key_input)
        show_checkbox = QCheckBox("Show custom API key")
        show_checkbox.setObjectName("embedding_custom_show_key_checkbox")
        show_checkbox.toggled.connect(
            lambda checked: self._custom_api_key_input.setEchoMode(
                QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
            )
        )
        api_key_layout.addWidget(show_checkbox)
        layout.addRow("API Key", api_key_container)

        organization = self._embedding_api_metadata.get("organization") or ""
        self._custom_api_org_input = QLineEdit(str(organization))
        self._custom_api_org_input.setObjectName("embedding_custom_organization_input")
        self._custom_api_org_input.setPlaceholderText("Optional organization / tenant")
        layout.addRow("Organization", self._custom_api_org_input)

        timeout_value = float(self._embedding_api_metadata.get("request_timeout") or self._original.request_timeout)
        self._custom_api_timeout_input = QDoubleSpinBox()
        self._custom_api_timeout_input.setObjectName("embedding_custom_timeout_input")
        self._custom_api_timeout_input.setRange(5.0, 600.0)
        self._custom_api_timeout_input.setSuffix(" s")
        self._custom_api_timeout_input.setSingleStep(5.0)
        self._custom_api_timeout_input.setValue(timeout_value)
        layout.addRow("Timeout", self._custom_api_timeout_input)

        retries_value = int(self._embedding_api_metadata.get("max_retries") or self._original.max_retries)
        self._custom_api_max_retries_input = QSpinBox()
        self._custom_api_max_retries_input.setObjectName("embedding_custom_max_retries_input")
        self._custom_api_max_retries_input.setRange(0, 10)
        self._custom_api_max_retries_input.setValue(retries_value)
        layout.addRow("Max Retries", self._custom_api_max_retries_input)

        retry_min = float(self._embedding_api_metadata.get("retry_min_seconds") or self._original.retry_min_seconds)
        self._custom_api_retry_min_input = QDoubleSpinBox()
        self._custom_api_retry_min_input.setObjectName("embedding_custom_retry_min_input")
        self._custom_api_retry_min_input.setRange(0.1, 30.0)
        self._custom_api_retry_min_input.setSingleStep(0.1)
        self._custom_api_retry_min_input.setValue(retry_min)
        layout.addRow("Retry Min", self._custom_api_retry_min_input)

        retry_max = float(self._embedding_api_metadata.get("retry_max_seconds") or self._original.retry_max_seconds)
        self._custom_api_retry_max_input = QDoubleSpinBox()
        self._custom_api_retry_max_input.setObjectName("embedding_custom_retry_max_input")
        self._custom_api_retry_max_input.setRange(0.1, 120.0)
        self._custom_api_retry_max_input.setSingleStep(0.5)
        self._custom_api_retry_max_input.setValue(retry_max)
        layout.addRow("Retry Max", self._custom_api_retry_max_input)

        headers_payload = self._embedding_api_metadata.get("default_headers") or {}
        headers_text = json.dumps(headers_payload, indent=2) if headers_payload else ""
        self._custom_api_headers_input = QPlainTextEdit(headers_text)
        self._custom_api_headers_input.setObjectName("embedding_custom_headers_input")
        self._custom_api_headers_input.setPlaceholderText('{"X-Request-ID": "abc-123"}')
        self._custom_api_headers_input.textChanged.connect(self._validate_embedding_fields)
        layout.addRow("Headers (JSON)", self._custom_api_headers_input)

        return container

    def _build_local_embedding_panel(self) -> QWidget:
        container = QWidget()
        layout = QFormLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        model_path = self._metadata_snapshot.get("st_model_path", "") or ""
        self._local_model_path_input = QLineEdit(str(model_path))
        self._local_model_path_input.setObjectName("embedding_local_model_path_input")
        self._local_model_path_input.setPlaceholderText("SentenceTransformers model path or HF repo")
        self._local_model_path_input.textChanged.connect(self._validate_embedding_fields)
        model_container = QWidget()
        model_layout = QHBoxLayout(model_container)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(4)
        model_layout.addWidget(self._local_model_path_input, 1)
        browse_model = QPushButton("Browse…")
        browse_model.setObjectName("embedding_local_model_browse_button")
        browse_model.clicked.connect(self._browse_local_model_path)
        model_layout.addWidget(browse_model, 0)
        layout.addRow("Model Path", model_container)

        self._local_device_combo = QComboBox()
        self._local_device_combo.setObjectName("embedding_local_device_combo")
        self._local_device_combo.setEditable(True)
        for choice in ("auto", "cpu", "cuda:0", "cuda:1", "mps"):
            self._local_device_combo.addItem(choice)
        device_value = str(self._metadata_snapshot.get("st_device") or "auto")
        index = self._local_device_combo.findText(device_value)
        if index >= 0:
            self._local_device_combo.setCurrentIndex(index)
        else:
            self._local_device_combo.setEditText(device_value)
        self._local_device_combo.currentTextChanged.connect(self._validate_embedding_fields)
        layout.addRow("Device", self._local_device_combo)

        self._local_dtype_combo = QComboBox()
        self._local_dtype_combo.setObjectName("embedding_local_dtype_combo")
        self._local_dtype_combo.setEditable(True)
        for choice in ("", "float32", "float16", "bfloat16", "int8"):
            label = "Default" if not choice else choice
            self._local_dtype_combo.addItem(label, choice)
        dtype_value = str(self._metadata_snapshot.get("st_dtype") or "")
        dtype_index = self._local_dtype_combo.findData(dtype_value)
        if dtype_index >= 0:
            self._local_dtype_combo.setCurrentIndex(dtype_index)
        else:
            self._local_dtype_combo.setEditText(dtype_value)
        self._local_dtype_combo.currentTextChanged.connect(self._validate_embedding_fields)
        layout.addRow("Torch DType", self._local_dtype_combo)

        cache_dir = self._metadata_snapshot.get("st_cache_dir", "") or ""
        self._local_cache_dir_input = QLineEdit(str(cache_dir))
        self._local_cache_dir_input.setObjectName("embedding_local_cache_dir_input")
        self._local_cache_dir_input.setPlaceholderText("Optional cache directory")
        cache_container = QWidget()
        cache_layout = QHBoxLayout(cache_container)
        cache_layout.setContentsMargins(0, 0, 0, 0)
        cache_layout.setSpacing(4)
        cache_layout.addWidget(self._local_cache_dir_input, 1)
        browse_cache = QPushButton("Browse…")
        browse_cache.setObjectName("embedding_local_cache_browse_button")
        browse_cache.clicked.connect(self._browse_local_cache_dir)
        cache_layout.addWidget(browse_cache, 0)
        layout.addRow("Cache Dir", cache_container)

        batch_override = self._metadata_snapshot.get("st_batch_size")
        batch_value = int(batch_override) if isinstance(batch_override, int) else 8
        self._local_batch_size_input = QSpinBox()
        self._local_batch_size_input.setObjectName("embedding_local_batch_size_input")
        self._local_batch_size_input.setRange(1, 256)
        self._local_batch_size_input.setValue(batch_value)
        self._local_batch_size_input.valueChanged.connect(self._validate_embedding_fields)
        layout.addRow("Batch Size", self._local_batch_size_input)

        return container

    def _collect_custom_api_metadata(self) -> dict[str, Any]:
        payload = dict(self._embedding_api_metadata)
        base_url = self._custom_api_base_url_input.text().strip()
        normalized_url, _ = self._normalize_custom_api_base_url(base_url) if base_url else (None, None)
        if normalized_url:
            payload["base_url"] = normalized_url
        else:
            payload.pop("base_url", None)

        organization = self._custom_api_org_input.text().strip()
        if organization:
            payload["organization"] = organization
        else:
            payload.pop("organization", None)

        api_key = self._custom_api_key_input.text().strip()
        existing_plaintext = str(self._embedding_api_metadata.get("api_key") or "").strip()
        existing_ciphertext = self._embedding_api_metadata.get("api_key_ciphertext")
        existing_hint = self._embedding_api_metadata.get("api_key_hint")

        if api_key:
            payload["api_key"] = api_key
            payload.pop("api_key_ciphertext", None)
            payload.pop("api_key_hint", None)
        elif existing_plaintext or existing_ciphertext:
            if existing_plaintext:
                payload["api_key"] = existing_plaintext
            else:
                payload.pop("api_key", None)
            if existing_ciphertext:
                payload["api_key_ciphertext"] = existing_ciphertext
            else:
                payload.pop("api_key_ciphertext", None)
            if existing_hint is not None:
                payload["api_key_hint"] = existing_hint
            else:
                payload.pop("api_key_hint", None)
        else:
            payload["api_key"] = ""
            payload.pop("api_key_ciphertext", None)
            payload.pop("api_key_hint", None)

        payload["request_timeout"] = float(self._custom_api_timeout_input.value())
        payload["max_retries"] = int(self._custom_api_max_retries_input.value())
        payload["retry_min_seconds"] = float(self._custom_api_retry_min_input.value())
        payload["retry_max_seconds"] = float(self._custom_api_retry_max_input.value())

        headers_text = self._custom_api_headers_input.toPlainText().strip()
        if headers_text:
            try:
                headers_obj = json.loads(headers_text)
            except json.JSONDecodeError:
                headers_obj = None
            if isinstance(headers_obj, Mapping):
                payload["default_headers"] = dict(headers_obj)
        else:
            payload.pop("default_headers", None)

        return payload

    def _collect_local_embedding_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        payload["st_model_path"] = self._local_model_path_input.text().strip()
        device_text = self._local_device_combo.currentText().strip()
        payload["st_device"] = device_text if device_text and device_text.lower() != "auto" else ""
        dtype_data = self._local_dtype_combo.currentData()
        dtype_text = str(dtype_data).strip() if isinstance(dtype_data, str) else ""
        if not dtype_text:
            text_value = self._local_dtype_combo.currentText().strip()
            dtype_text = "" if text_value.lower() == "default" else text_value
        payload["st_dtype"] = dtype_text
        payload["st_cache_dir"] = self._local_cache_dir_input.text().strip()
        payload["st_batch_size"] = int(self._local_batch_size_input.value())
        return payload

    def _browse_local_model_path(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Model Directory")
        if selected:
            self._local_model_path_input.setText(selected)

    def _browse_local_cache_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Cache Directory")
        if selected:
            self._local_cache_dir_input.setText(selected)

    def _toggle_api_visibility(self, checked: bool) -> None:
        self._api_key_input.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )

    def _on_accept(self) -> None:
        self._result = self.gather_settings()
        self.accept()

    def _run_validation(self) -> None:
        if self._validator is None:
            self._last_validation = ValidationResult(
                ok=True, message="No validator configured; settings saved locally."
            )
        else:
            candidate = self.gather_settings()
            raw_result = self._validator(candidate)
            self._last_validation = _coerce_validation_result(raw_result)
        self._display_status(self._last_validation, fallback="Validation completed.")

    def _run_api_test(self) -> None:
        if self._api_tester is None:
            self._last_api_test = ValidationResult(
                ok=False, message="No API tester configured; cannot run connection test."
            )
        else:
            candidate = self.gather_settings()
            raw_result = self._api_tester(candidate)
            self._last_api_test = _coerce_validation_result(raw_result)
        self._display_status(self._last_api_test, fallback="API test completed.", toast=True)

    def _run_embedding_test(self) -> None:
        if self._embedding_tester is None:
            self._last_embedding_test = ValidationResult(
                ok=False, message="No embeddings tester configured; cannot run test."
            )
        else:
            candidate = self.gather_settings()
            raw_result = self._embedding_tester(candidate)
            self._last_embedding_test = _coerce_validation_result(raw_result)
        self._display_status(self._last_embedding_test, fallback="Embedding test completed.", toast=True)

    def _display_status(self, result: ValidationResult, *, fallback: str, toast: bool = False) -> None:
        self._validation_label.setVisible(True)
        self._validation_label.setText(result.message or fallback)
        color = "#22863a" if result.ok else "#d73a49"
        self._validation_label.setStyleSheet(f"color: {color};")
        if toast and (result.message or fallback):
            self._show_toast(result.message or fallback, success=result.ok)

    def _update_base_url_hint(self) -> None:
        text = (self._base_url_input.text() or "").strip()
        level = "info"
        message = "HTTPS endpoint, e.g., https://api.openai.com/v1"
        if text:
            parsed = urlparse(text)
            scheme = (parsed.scheme or "").lower()
            if scheme not in {"https", "http"}:
                level = "error"
                message = "Base URL must start with https://"
            elif scheme != "https":
                level = "warning"
                message = "Use https:// endpoints to protect API keys."
            elif not parsed.netloc:
                level = "error"
                message = "Enter a hostname, e.g., api.openai.com."
            else:
                host = parsed.netloc
                message = f"Connecting to {host}"
                level = "success"
        self._set_hint(self._base_url_hint, message, level)
        self._set_field_error("base_url", message if level == "error" else None)

    def _update_reserve_hint(self) -> None:
        reserve = int(self._response_token_reserve_input.value())
        max_tokens = int(self._max_context_tokens_input.value())
        level = "info"
        message = "Reserve must stay below the max context tokens."
        if reserve >= max_tokens:
            level = "error"
            message = (
                f"Reserve ({reserve:,}) must be lower than max context ({max_tokens:,})."
            )
        elif reserve >= int(max_tokens * 0.85):
            level = "warning"
            remaining = max_tokens - reserve
            message = f"Only {remaining:,} prompt tokens remain; consider lowering reserve."
        self._set_hint(self._reserve_hint, message, level)
        self._set_field_error("token_reserve", message if level == "error" else None)

    def _update_timeout_hint(self) -> None:
        timeout = float(self._request_timeout_input.value())
        level = "info"
        message = "Applies to chat + tool turns; keep between 5–300 seconds."
        if timeout < 5.0:
            level = "error"
            message = "Timeout must be at least 5 seconds."
        elif timeout > 300.0:
            level = "warning"
            message = "Large (>300s) timeouts can stall the UI if the API hangs."
        self._set_hint(self._timeout_hint, message, level)
        self._set_field_error("timeout", message if level == "error" else None)

    def _update_temperature_hint(self) -> None:
        temperature = float(self._temperature_input.value())
        level = "info"
        message = "0 = deterministic, higher values increase variety."
        if temperature >= 1.2:
            level = "warning"
            message = "High (>1.2) temps boost creativity but risk tangents."
        elif temperature >= 0.6:
            level = "success"
            message = "0.6–1.1 is a balanced creative range for drafting."
        elif temperature <= 0.1:
            level = "info"
            message = "Near-zero temps keep responses predictable."
        self._set_hint(self._temperature_hint, message, level)

    def _update_context_policy_hint(self) -> None:
        enabled = self._context_policy_enabled.isChecked()
        dry_run = self._context_policy_dry_run.isChecked()
        if not enabled:
            level = "info"
            message = "Disabled – controller uses legacy prompt budgeting."
        elif dry_run:
            level = "warning"
            message = "Dry run logs decisions without blocking oversized prompts."
        else:
            level = "success"
            message = "Enabled – controller will block requests exceeding the policy."
        self._set_hint(self._context_policy_hint, message, level)

    def _toggle_context_prompt_override(self, checked: bool) -> None:
        self._context_prompt_override_input.setEnabled(checked)

    def _toggle_context_reserve_override(self, checked: bool) -> None:
        self._context_reserve_override_input.setEnabled(checked)

    def _gather_context_policy_settings(self) -> ContextPolicySettings:
        original = getattr(self._original, "context_policy", ContextPolicySettings())
        prompt_override = (
            int(self._context_prompt_override_input.value())
            if self._context_prompt_override_toggle.isChecked()
            else None
        )
        reserve_override = (
            int(self._context_reserve_override_input.value())
            if self._context_reserve_override_toggle.isChecked()
            else None
        )
        return replace(
            original,
            enabled=self._context_policy_enabled.isChecked(),
            dry_run=self._context_policy_dry_run.isChecked(),
            prompt_budget_override=prompt_override,
            response_reserve_override=reserve_override,
        )

    def _prepare_hint_label(self, label: QLabel) -> None:
        label.setWordWrap(True)
        label.setContentsMargins(0, 0, 0, 0)
        self._set_hint(label, label.text(), "info")

    def _set_hint(self, label: QLabel, text: str, level: str = "info") -> None:
        label.setText(text)
        color = _HINT_COLORS.get(level, _HINT_COLORS["info"])
        label.setStyleSheet(f"color: {color}; font-size: 11px;")

    def _set_field_error(self, field: str, message: str | None) -> None:
        if message:
            self._field_errors[field] = message
        else:
            self._field_errors.pop(field, None)
        self._update_buttons_state()

    def _update_buttons_state(self) -> None:
        has_errors = bool(self._field_errors)
        if self._save_button is not None:
            self._save_button.setEnabled(not has_errors)
        if self._validate_button is not None:
            self._validate_button.setEnabled(self._validator is not None and not has_errors)
        if self._test_button is not None:
            self._test_button.setEnabled(self._api_tester is not None and not has_errors)
        if self._embedding_test_button is not None:
            self._embedding_test_button.setEnabled(self._embedding_tester is not None and not has_errors)

    def _show_toast(self, message: str, *, success: bool) -> None:
        if not self._toast_enabled or not message:
            return
        box = QMessageBox(self)
        box.setWindowTitle("AI Settings")
        icon = QMessageBox.Icon.Information if success else QMessageBox.Icon.Warning
        box.setIcon(icon)
        box.setText(message)
        box.setStandardButtons(QMessageBox.StandardButton.Close)
        close_button = box.button(QMessageBox.StandardButton.Close)
        if close_button is not None:
            close_button.clicked.connect(box.close)
        box.setModal(False)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        box.show()

        self._active_toasts.append(box)

        def _cleanup() -> None:
            if box in self._active_toasts:
                self._active_toasts.remove(box)

        box.destroyed.connect(lambda *_: _cleanup())
        QTimer.singleShot(3500, box.close)

    def _embedding_mode_value(self) -> str:
        index = self._embedding_mode_combo.currentIndex()
        value = self._embedding_mode_combo.itemData(index)
        return str(value) if isinstance(value, str) and value else self._initial_embedding_mode

    def _handle_embedding_mode_changed(self) -> None:
        mode = self._embedding_mode_value()
        stack_index = self._mode_index_map.get(mode, 0)
        self._embedding_mode_stack.setCurrentIndex(stack_index)
        # Update hints based on mode
        if mode == "disabled":
            hint = "Semantic search will fall back to regex and outline matching."
        elif mode == "custom-api":
            hint = "Configure a dedicated endpoint and API key just for embeddings."
        elif mode == "local":
            hint = "Run embeddings locally via SentenceTransformers."
        else:
            hint = "Uses your chat API credentials for embeddings."
        self._set_hint(self._embedding_mode_hint, hint)
        # Show/hide model input based on mode (only needed for API modes)
        show_model = mode in ("same-api", "custom-api")
        self._embedding_model_container.setVisible(show_model)
        self._validate_embedding_fields()

    def _validate_embedding_fields(self) -> None:
        mode = self._embedding_mode_value()
        model = self._embedding_model_input.text().strip()
        # Model only required for API modes
        if mode in ("disabled", "local"):
            self._set_field_error("embedding_model_name", None)
        elif not model:
            self._set_field_error("embedding_model_name", "Embedding model is required for API modes.")
        else:
            self._set_field_error("embedding_model_name", None)
        self._validate_custom_api_fields(mode)
        self._validate_local_fields(mode)

    def _validate_custom_api_fields(self, mode: str) -> None:
        if mode != "custom-api":
            self._set_field_error("embedding_custom_base_url", None)
            self._set_field_error("embedding_custom_api_key", None)
            self._set_field_error("embedding_custom_headers", None)
            return
        base_url = self._custom_api_base_url_input.text().strip()
        api_key = self._custom_api_key_input.text().strip() or self._embedding_api_metadata.get("api_key", "")
        normalized_url, url_error = self._normalize_custom_api_base_url(base_url) if base_url else (None, None)
        if url_error:
            self._set_field_error("embedding_custom_base_url", url_error)
        else:
            if normalized_url and normalized_url != base_url:
                self._custom_api_base_url_input.blockSignals(True)
                self._custom_api_base_url_input.setText(normalized_url)
                self._custom_api_base_url_input.blockSignals(False)
            self._set_field_error("embedding_custom_base_url", None)
        if not api_key:
            self._set_field_error("embedding_custom_api_key", "API key is required for custom embeddings.")
        else:
            self._set_field_error("embedding_custom_api_key", None)
        headers_text = self._custom_api_headers_input.toPlainText().strip()
        if headers_text:
            try:
                headers_obj = json.loads(headers_text)
                valid_headers = isinstance(headers_obj, Mapping)
            except json.JSONDecodeError:
                valid_headers = False
            if not valid_headers:
                self._set_field_error("embedding_custom_headers", "Custom headers must be valid JSON.")
            else:
                self._set_field_error("embedding_custom_headers", None)
        else:
            self._set_field_error("embedding_custom_headers", None)

    def _validate_local_fields(self, mode: str) -> None:
        if mode != "local":
            self._set_field_error("embedding_local_model_path", None)
            return
        path = self._local_model_path_input.text().strip()
        if not path:
            self._set_field_error("embedding_local_model_path", "Local embeddings require a model path or identifier.")
        else:
            self._set_field_error("embedding_local_model_path", None)

    def _normalize_custom_api_base_url(self, raw_value: str) -> tuple[str | None, str | None]:
        trimmed = raw_value.strip()
        if not trimmed:
            return None, "Base URL is required for custom embeddings."
        candidate = trimmed
        if "://" not in candidate:
            candidate = f"https://{candidate.lstrip('/')}"
        parsed = urlparse(candidate)
        scheme = (parsed.scheme or "").lower()
        if scheme not in {"http", "https"} or not parsed.netloc:
            return None, "Provide a valid http(s) endpoint for embeddings."
        sanitized = parsed._replace(fragment="", query="")
        return sanitized.geturl(), None

    def _populate_theme_combo(self, active_theme: str | None) -> None:
        available = theme_manager.available()
        self._theme_combo.clear()
        seen: set[str] = set()
        for theme in available:
            key = (theme.name or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            label = theme.title or theme.name
            index = self._theme_combo.count()
            self._theme_combo.addItem(label, theme.name)
            if theme.description:
                self._theme_combo.setItemData(index, theme.description, Qt.ItemDataRole.ToolTipRole)
        active_key = (active_theme or "").strip()
        if active_key and active_key.lower() not in seen:
            index = self._theme_combo.count()
            self._theme_combo.addItem(active_key, active_key)
            self._theme_combo.setItemData(index, "Custom theme", Qt.ItemDataRole.ToolTipRole)
        if self._theme_combo.count() == 0:
            self._theme_combo.addItem("Default", "default")
        target_index = (
            self._theme_combo.findData(active_key, Qt.ItemDataRole.UserRole)
            if active_key
            else 0
        )
        self._theme_combo.setCurrentIndex(target_index if target_index >= 0 else 0)

    def _current_theme_name(self) -> str:
        data = self._theme_combo.currentData(Qt.ItemDataRole.UserRole)
        if isinstance(data, str) and data.strip():
            return data.strip()
        text = self._theme_combo.currentText().strip()
        return text or self._original.theme


def show_settings_dialog(
    parent: QWidget | None = None,
    *,
    settings: Settings | None = None,
    models: Sequence[str] | None = None,
    validator: SettingsValidator | None = None,
    api_tester: SettingsTester | None = None,
    embedding_tester: SettingsTester | None = None,
) -> SettingsDialogResult:
    """Display the settings dialog and return the captured values."""

    tester = api_tester if api_tester is not None else test_ai_api_settings
    embedding_test = embedding_tester if embedding_tester is not None else test_embedding_settings
    dialog = SettingsDialog(
        settings=settings,
        parent=parent,
        available_models=models,
        validator=validator,
        api_tester=tester,
        embedding_tester=embedding_test,
    )
    accepted = dialog.exec() == int(QDialog.DialogCode.Accepted)
    last_message = dialog.last_validation.message or None
    return SettingsDialogResult(
        accepted=accepted,
        settings=dialog.settings(),
        validated=dialog.validated,
        validation_message=last_message,
        api_tested=dialog.api_tested,
        api_test_message=dialog.last_api_test.message or None,
        embedding_tested=dialog.embedding_tested,
        embedding_test_message=dialog.last_embedding_test.message or None,
    )


def _coerce_validation_result(result: ValidationResult | tuple[bool, str] | bool) -> ValidationResult:
    if isinstance(result, ValidationResult):
        return result
    if isinstance(result, tuple):
        ok, message = result
        return ValidationResult(bool(ok), str(message))
    return ValidationResult(bool(result), "")


def test_ai_api_settings(settings: Settings) -> ValidationResult:
    """Attempt to reach the AI API using the provided settings."""

    base_url = (settings.base_url or "").strip().rstrip("/")
    api_key = (settings.api_key or "").strip()
    if not base_url:
        return ValidationResult(False, "Base URL is required to test the AI API.")
    if not api_key:
        return ValidationResult(False, "API key is required to test the AI API.")

    endpoint = f"{base_url}/models"
    headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"}
    if settings.organization:
        headers["OpenAI-Organization"] = settings.organization
    if settings.default_headers:
        headers.update(settings.default_headers)

    timeout = settings.request_timeout or 30.0

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            response = client.get(endpoint)
    except httpx.TimeoutException:
        return ValidationResult(False, "Connection to the AI API timed out.")
    except httpx.HTTPError as exc:
        return ValidationResult(False, f"Failed to reach AI API: {exc}")

    if response.is_success:
        return ValidationResult(True, "AI API connection succeeded.")

    snippet = (response.text or "").strip()
    if len(snippet) > 120:
        snippet = f"{snippet[:117]}..."
    reason = snippet or response.reason_phrase or "Unknown error"
    return ValidationResult(False, f"API responded with {response.status_code}: {reason}")


def test_embedding_settings(settings: Settings) -> ValidationResult:
    """Exercise the embedding backend using the selected mode and metadata."""

    try:
        from ..ui.embedding_controller import EmbeddingController
    except Exception as exc:  # pragma: no cover - optional dependency guard
        return ValidationResult(False, f"Embedding controller unavailable: {exc}")

    metadata = getattr(settings, "metadata", {}) or {}
    cache_root = (Path(tempfile.gettempdir()) / "tinkerbell-embedding-test").resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    controller: EmbeddingController | None = None
    try:
        previous_loop = asyncio.get_event_loop_policy().get_event_loop() if asyncio.get_event_loop_policy() else None
    except RuntimeError:
        previous_loop = None
    try:
        controller = EmbeddingController(
            status_bar=None,
            cache_root_resolver=lambda: cache_root,
            outline_worker_resolver=lambda: None,
            async_loop_resolver=lambda: None,
            background_task_runner=lambda coro: None,
            phase3_outline_enabled=True,
        )
        mode = controller._resolve_embedding_mode(metadata)
        backend = controller._normalize_backend(getattr(settings, "embedding_backend", "auto"), mode)
        if backend == "disabled":
            return ValidationResult(False, "Embeddings are disabled; pick a backend before testing.")
        embedding_settings = controller._build_embedding_settings(settings, metadata, mode)
        model_name = getattr(settings, "embedding_model_name", "") or "text-embedding-3-large"
        provider, state = controller._build_embedding_provider(backend, model_name, embedding_settings, metadata, mode)
        if provider is None:
            return ValidationResult(False, state.error or "Unable to build embedding provider.")
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        def _run_validation_with_new_loop(
            *, restore_loop: asyncio.AbstractEventLoop | None,
        ) -> "EmbeddingValidationResult":
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            validation_coro = controller._validator.validate(provider, mode=mode)
            task = loop.create_task(validation_coro)
            try:
                return loop.run_until_complete(task)
            finally:
                if not task.done():
                    task.cancel()
                    try:
                        loop.run_until_complete(task)
                    except Exception:
                        pass
                loop.close()
                if restore_loop is not None:
                    try:
                        asyncio.set_event_loop(restore_loop)
                    except Exception:
                        pass

        def _run_validation_in_thread() -> "EmbeddingValidationResult":
            result_holder: list["EmbeddingValidationResult"] = []
            error_holder: list[BaseException] = []
            finished = threading.Event()

            def _worker() -> None:
                try:
                    result_holder.append(_run_validation_with_new_loop(restore_loop=None))
                except BaseException as exc:  # pragma: no cover - propagates to caller
                    error_holder.append(exc)
                finally:
                    finished.set()

            thread = threading.Thread(target=_worker, name="embedding-validation-test", daemon=True)
            thread.start()
            finished.wait()
            if error_holder:
                raise error_holder[0]
            return result_holder[0]

        if running_loop is not None:
            result = _run_validation_in_thread()
        else:
            result = _run_validation_with_new_loop(restore_loop=previous_loop)
        if result.status == "ready":
            return ValidationResult(True, result.detail or "Embeddings validated successfully.")
        return ValidationResult(False, result.error or "Embedding validation failed.")
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        return ValidationResult(False, f"Embedding test failed: {exc}")
    finally:
        try:
            if controller is not None:
                controller._dispose_embedding_resource()
        except Exception:
            pass

