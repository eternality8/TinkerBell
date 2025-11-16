"""Tests for the widgets.dialogs module."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
)

from tinkerbell.services.settings import Settings
from tinkerbell.widgets import dialogs
from tinkerbell.widgets.dialogs import (
    DocumentExportDialog,
    DocumentLoadDialog,
    SampleDocument,
    SettingsDialog,
    ValidationResult,
)


@pytest.fixture
def dialog_settings() -> Settings:
    return Settings(
        base_url="https://api.openai.com/v1",
        api_key="original-key",
        model="gpt-4o-mini",
        theme="light",
        organization="legacy",
        request_timeout=75.0,
    )


def test_settings_dialog_gather_settings_reflects_changes(qtbot, dialog_settings: Settings) -> None:
    dialog = SettingsDialog(settings=dialog_settings, show_toasts=False)
    qtbot.addWidget(dialog)

    base_input = dialog.findChild(QLineEdit, "base_url_input")
    api_input = dialog.findChild(QLineEdit, "api_key_input")
    model_combo = dialog.findChild(QComboBox, "model_combo")
    organization_input = dialog.findChild(QLineEdit, "organization_input")
    theme_input = dialog.findChild(QLineEdit, "theme_input")
    tool_checkbox = dialog.findChild(QCheckBox, "tool_activity_checkbox")
    timeout_input = dialog.findChild(QDoubleSpinBox, "request_timeout_input")
    context_input = dialog.findChild(QSpinBox, "max_context_tokens_input")
    reserve_input = dialog.findChild(QSpinBox, "response_token_reserve_input")

    assert base_input is not None
    assert api_input is not None
    assert model_combo is not None
    assert organization_input is not None
    assert theme_input is not None
    assert tool_checkbox is not None
    assert timeout_input is not None
    assert context_input is not None
    assert reserve_input is not None

    base_input.setText("https://example.com/v2")
    api_input.setText("new-key")
    model_combo.setEditText("gpt-custom")
    organization_input.setText("acme")
    theme_input.setText("dracula")
    tool_checkbox.setChecked(True)
    timeout_input.setValue(42.5)
    context_input.setValue(256_000)
    reserve_input.setValue(20_000)

    updated = dialog.gather_settings()

    assert updated.base_url == "https://example.com/v2"
    assert updated.api_key == "new-key"
    assert updated.model == "gpt-custom"
    assert updated.organization == "acme"
    assert updated.theme == "dracula"
    assert updated.show_tool_activity_panel is True
    assert updated.request_timeout == pytest.approx(42.5)
    assert updated.max_context_tokens == 256_000
    assert updated.response_token_reserve == 20_000


def test_settings_dialog_validation_uses_validator(qtbot, dialog_settings: Settings) -> None:
    def validator(settings: Settings) -> ValidationResult:
        if settings.api_key:
            return ValidationResult(ok=True, message="API key looks valid")
        return ValidationResult(ok=False, message="Missing API key")

    dialog = SettingsDialog(settings=dialog_settings, validator=validator, show_toasts=False)
    qtbot.addWidget(dialog)

    validate_button = dialog.findChild(QPushButton, "validate_button")
    api_input = dialog.findChild(QLineEdit, "api_key_input")
    assert validate_button is not None
    assert api_input is not None

    qtbot.mouseClick(validate_button, Qt.MouseButton.LeftButton)
    assert dialog.validated is True
    assert "valid" in (dialog.last_validation.message or "").lower()

    api_input.clear()
    qtbot.mouseClick(validate_button, Qt.MouseButton.LeftButton)
    assert dialog.validated is False
    assert "missing" in (dialog.last_validation.message or "").lower()


def test_settings_dialog_test_button_runs_api_tester(qtbot, dialog_settings: Settings) -> None:
    captured: dict[str, Settings] = {}

    def tester(settings: Settings) -> ValidationResult:
        captured["settings"] = settings
        return ValidationResult(ok=True, message="API reachable")

    dialog = SettingsDialog(settings=dialog_settings, api_tester=tester, show_toasts=False)
    qtbot.addWidget(dialog)

    test_button = dialog.findChild(QPushButton, "test_button")
    assert test_button is not None

    qtbot.mouseClick(test_button, Qt.MouseButton.LeftButton)
    assert captured["settings"].api_key == dialog_settings.api_key
    assert dialog.api_tested is True
    assert "api" in (dialog.last_api_test.message or "").lower()


def test_open_file_dialog_returns_path_via_custom_dialog(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDialog:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._path = Path("C:/tmp/notes.md")

        def exec(self) -> int:
            return int(dialogs.QDialog.DialogCode.Accepted)

        def selected_path(self) -> Path:
            return self._path

    monkeypatch.setattr(dialogs, "DocumentLoadDialog", FakeDialog)

    result = dialogs.open_file_dialog(token_budget=1000)

    assert result == Path("C:/tmp/notes.md")


def test_save_file_dialog_appends_default_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSaveDialog:
        def __init__(self, **kwargs):
            self._path = Path("C:/tmp/new-note")

        def exec(self) -> int:
            return int(dialogs.QDialog.DialogCode.Accepted)

        def selected_path(self) -> Path:
            return self._path

    monkeypatch.setattr(dialogs, "DocumentExportDialog", FakeSaveDialog)

    result = dialogs.save_file_dialog(default_suffix="md")

    assert result == Path("C:/tmp/new-note.md")


def test_document_load_dialog_preview_updates(qtbot, tmp_path: Path) -> None:
    target = tmp_path / "sample.md"
    target.write_text("Hello world", encoding="utf-8")

    dialog = DocumentLoadDialog(parent=None, sample_documents=(), token_budget=2000)
    qtbot.addWidget(dialog)

    dialog.select_path(target)

    assert dialog.selected_path() == target
    preview_text = dialog.preview_text()
    assert "Hello" in preview_text
    token_label = dialog.findChild(QLabel, "document_token_label")
    assert token_label is not None
    assert "Tokens" in token_label.text()


def test_document_export_dialog_preview_modes(qtbot) -> None:
    dialog = DocumentExportDialog(
        parent=None,
        document_text="alpha beta gamma",
        selection_text="alpha beta",
        token_budget=1000,
    )
    qtbot.addWidget(dialog)

    stats_label = dialog.findChild(QLabel, "preview_stats_label")
    assert stats_label is not None
    assert "Selection" in stats_label.text()

    mode_combo = dialog.findChild(QComboBox, "preview_mode_combo")
    assert mode_combo is not None
    mode_combo.setCurrentIndex(1)
    assert "Document" in stats_label.text()


def test_discover_sample_documents_returns_entries() -> None:
    samples = dialogs.discover_sample_documents()
    assert isinstance(samples, tuple)
    assert samples
    assert all(isinstance(entry, SampleDocument) for entry in samples)


def test_test_ai_api_settings_success(monkeypatch: pytest.MonkeyPatch, dialog_settings: Settings) -> None:
    called: dict[str, object] = {}

    class FakeResponse:
        def __init__(self, status_code: int = 200, text: str = "ok", reason: str = "OK") -> None:
            self.status_code = status_code
            self.text = text
            self.reason_phrase = reason

        @property
        def is_success(self) -> bool:
            return self.status_code < 400

    class FakeClient:
        def __init__(self, *_, **kwargs) -> None:
            called["headers"] = kwargs.get("headers", {})
            called["timeout"] = kwargs.get("timeout")

        def __enter__(self):  # noqa: D401 - context manager helper
            return self

        def __exit__(self, *_) -> None:
            return None

        def get(self, endpoint: str) -> FakeResponse:
            called["endpoint"] = endpoint
            return FakeResponse()

    monkeypatch.setattr(dialogs.httpx, "Client", FakeClient)

    result = dialogs.test_ai_api_settings(dialog_settings)

    assert result.ok is True
    assert str(called["endpoint"]).endswith("/models")
    headers_obj = called["headers"]
    assert isinstance(headers_obj, dict)
    assert str(headers_obj.get("Authorization", "")).startswith("Bearer ")
    assert called["timeout"] == pytest.approx(dialog_settings.request_timeout)


def test_test_ai_api_settings_failure(monkeypatch: pytest.MonkeyPatch, dialog_settings: Settings) -> None:
    class FakeResponse:
        def __init__(self) -> None:
            self.status_code = 401
            self.text = "unauthorized"
            self.reason_phrase = "Unauthorized"

        @property
        def is_success(self) -> bool:
            return False

    class FakeClient:
        def __init__(self, *_, **__) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_) -> None:
            return None

        def get(self, endpoint: str) -> FakeResponse:
            return FakeResponse()

    monkeypatch.setattr(dialogs.httpx, "Client", FakeClient)

    result = dialogs.test_ai_api_settings(dialog_settings)

    assert result.ok is False
    assert "401" in result.message


def test_settings_dialog_reserve_validation_blocks_save(qtbot, dialog_settings: Settings) -> None:
    dialog = SettingsDialog(settings=dialog_settings, show_toasts=False)
    qtbot.addWidget(dialog)

    context_input = dialog.findChild(QSpinBox, "max_context_tokens_input")
    reserve_input = dialog.findChild(QSpinBox, "response_token_reserve_input")
    button_box = dialog.findChild(QDialogButtonBox)
    assert context_input is not None
    assert reserve_input is not None
    assert button_box is not None

    save_button = button_box.button(QDialogButtonBox.StandardButton.Save)
    assert save_button is not None
    assert save_button.isEnabled()

    context_input.setValue(40_000)
    reserve_input.setValue(50_000)
    assert save_button.isEnabled() is False

    reserve_input.setValue(context_input.value() - 4_000)
    assert save_button.isEnabled() is True


def test_settings_dialog_base_url_hint_shows_errors(qtbot, dialog_settings: Settings) -> None:
    dialog = SettingsDialog(settings=dialog_settings, show_toasts=False)
    qtbot.addWidget(dialog)

    base_input = dialog.findChild(QLineEdit, "base_url_input")
    base_hint = dialog.findChild(QLabel, "base_url_hint")
    button_box = dialog.findChild(QDialogButtonBox)

    assert base_input is not None
    assert base_hint is not None
    assert button_box is not None

    save_button = button_box.button(QDialogButtonBox.StandardButton.Save)
    assert save_button is not None

    base_input.setText("ftp://example.com/v1")
    assert "https" in base_hint.text().lower()
    assert save_button.isEnabled() is False

    base_input.setText("https://example.com/v1")
    assert "example.com" in base_hint.text().lower()
    assert save_button.isEnabled() is True