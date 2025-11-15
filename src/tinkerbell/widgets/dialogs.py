"""Dialogs for file operations and settings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

import httpx

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..services.settings import Settings

__all__ = [
    "DEFAULT_FILE_FILTER",
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


def open_file_dialog(
    parent: QWidget | None = None,
    *,
    caption: str = "Open Document",
    start_dir: Path | str | None = None,
    file_filter: str | None = None,
) -> Path | None:
    """Show a native open-file dialog with project-specific defaults."""

    selected_filter = file_filter or DEFAULT_FILE_FILTER
    directory = str(start_dir) if start_dir else ""
    path, _ = QFileDialog.getOpenFileName(parent, caption, directory, selected_filter)
    return Path(path) if path else None


def save_file_dialog(
    parent: QWidget | None = None,
    *,
    caption: str = "Save Document",
    start_dir: Path | str | None = None,
    file_filter: str | None = None,
    default_suffix: str = "md",
) -> Path | None:
    """Show a native save-file dialog with sensible defaults."""

    selected_filter = file_filter or DEFAULT_FILE_FILTER
    directory = str(start_dir) if start_dir else ""
    path, _ = QFileDialog.getSaveFileName(parent, caption, directory, selected_filter)
    if not path:
        return None
    resolved = Path(path)
    if not resolved.suffix and default_suffix:
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

        self._base_url_input = QLineEdit(self._original.base_url)
        self._base_url_input.setObjectName("base_url_input")
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
        self._organization_input = QLineEdit(self._original.organization or "")
        self._organization_input.setObjectName("organization_input")
        self._theme_input = QLineEdit(self._original.theme)
        self._theme_input.setObjectName("theme_input")
        self._debug_checkbox = QCheckBox("Enable debug logging")
        self._debug_checkbox.setObjectName("debug_logging_checkbox")
        self._debug_checkbox.setChecked(self._original.debug_logging)
        self._tool_panel_checkbox = QCheckBox("Show tool activity panel (debug view)")
        self._tool_panel_checkbox.setObjectName("tool_activity_checkbox")
        self._tool_panel_checkbox.setChecked(
            bool(getattr(self._original, "show_tool_activity_panel", False))
        )
        self._max_tool_iterations_input = QSpinBox()
        self._max_tool_iterations_input.setObjectName("max_tool_iterations_input")
        self._max_tool_iterations_input.setRange(1, 25)
        self._max_tool_iterations_input.setValue(
            max(1, int(getattr(self._original, "max_tool_iterations", 8) or 8))
        )
        self._max_tool_iterations_input.setSuffix(" loops")
        self._max_tool_iterations_input.setToolTip(
            "Maximum times the agent may invoke tools before returning a response."
        )

        form_layout = QFormLayout()
        form_layout.addRow("Base URL", self._base_url_input)

        api_container = QWidget()
        api_layout = QVBoxLayout(api_container)
        api_layout.setContentsMargins(0, 0, 0, 0)
        api_layout.addWidget(self._api_key_input)
        show_checkbox = QCheckBox("Show API key")
        show_checkbox.setObjectName("show_api_checkbox")
        show_checkbox.toggled.connect(self._toggle_api_visibility)
        api_layout.addWidget(show_checkbox)
        form_layout.addRow("API Key", api_container)

        form_layout.addRow("Model", self._model_combo)
        form_layout.addRow("Organization", self._organization_input)
        form_layout.addRow("Theme", self._theme_input)
        form_layout.addRow("Debug", self._debug_checkbox)
        form_layout.addRow("Tool Traces", self._tool_panel_checkbox)
        form_layout.addRow("Max Tool Iterations", self._max_tool_iterations_input)

        layout = QVBoxLayout(self)
        layout.addLayout(form_layout)

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
        layout.addLayout(validation_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

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

    def gather_settings(self) -> Settings:
        """Assemble a `Settings` instance from the current form widgets."""

        base_url = self._base_url_input.text().strip() or self._original.base_url
        api_key = self._api_key_input.text().strip()
        model = self._model_combo.currentText().strip() or self._original.model
        organization = self._organization_input.text().strip() or None
        theme = self._theme_input.text().strip() or self._original.theme
        debug_logging = self._debug_checkbox.isChecked()
        show_tool_activity_panel = self._tool_panel_checkbox.isChecked()
        max_tool_iterations = int(self._max_tool_iterations_input.value())
        return replace(
            self._original,
            base_url=base_url,
            api_key=api_key,
            model=model,
            organization=organization,
            theme=theme,
            debug_logging=debug_logging,
            show_tool_activity_panel=show_tool_activity_panel,
            max_tool_iterations=max_tool_iterations,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
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
        self._display_status(self._last_api_test, fallback="API test completed.")

    def _display_status(self, result: ValidationResult, *, fallback: str) -> None:
        self._validation_label.setVisible(True)
        self._validation_label.setText(result.message or fallback)
        color = "#22863a" if result.ok else "#d73a49"
        self._validation_label.setStyleSheet(f"color: {color};")


def show_settings_dialog(
    parent: QWidget | None = None,
    *,
    settings: Settings | None = None,
    models: Sequence[str] | None = None,
    validator: SettingsValidator | None = None,
    api_tester: SettingsTester | None = None,
) -> SettingsDialogResult:
    """Display the settings dialog and return the captured values."""

    tester = api_tester if api_tester is not None else test_ai_api_settings
    dialog = SettingsDialog(
        settings=settings,
        parent=parent,
        available_models=models,
        validator=validator,
        api_tester=tester,
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

