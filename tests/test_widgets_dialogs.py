"""Tests for the widgets.dialogs module."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QLineEdit, QPushButton

from tinkerbell.services.settings import Settings
from tinkerbell.widgets import dialogs
from tinkerbell.widgets.dialogs import SettingsDialog, ValidationResult


@pytest.fixture
def dialog_settings() -> Settings:
    return Settings(
        base_url="https://api.openai.com/v1",
        api_key="original-key",
        model="gpt-4o-mini",
        theme="light",
        organization="legacy",
        telemetry_opt_in=False,
    )


def test_settings_dialog_gather_settings_reflects_changes(qtbot, dialog_settings: Settings) -> None:
    dialog = SettingsDialog(settings=dialog_settings)
    qtbot.addWidget(dialog)

    base_input = dialog.findChild(QLineEdit, "base_url_input")
    api_input = dialog.findChild(QLineEdit, "api_key_input")
    model_combo = dialog.findChild(QComboBox, "model_combo")
    organization_input = dialog.findChild(QLineEdit, "organization_input")
    theme_input = dialog.findChild(QLineEdit, "theme_input")
    telemetry_checkbox = dialog.findChild(QCheckBox, "telemetry_checkbox")

    assert base_input and api_input and model_combo and organization_input and theme_input
    assert telemetry_checkbox

    base_input.setText("https://example.com/v2")
    api_input.setText("new-key")
    model_combo.setEditText("gpt-custom")
    organization_input.setText("acme")
    theme_input.setText("dracula")
    telemetry_checkbox.setChecked(True)

    updated = dialog.gather_settings()

    assert updated.base_url == "https://example.com/v2"
    assert updated.api_key == "new-key"
    assert updated.model == "gpt-custom"
    assert updated.organization == "acme"
    assert updated.theme == "dracula"
    assert updated.telemetry_opt_in is True


def test_settings_dialog_validation_uses_validator(qtbot, dialog_settings: Settings) -> None:
    def validator(settings: Settings) -> ValidationResult:
        if settings.api_key:
            return ValidationResult(ok=True, message="API key looks valid")
        return ValidationResult(ok=False, message="Missing API key")

    dialog = SettingsDialog(settings=dialog_settings, validator=validator)
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


def test_open_file_dialog_wraps_qfiledialog(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_get_open_file_name(parent, caption, directory, filters):
        captured["args"] = (parent, caption, directory, filters)
        return ("C:/tmp/notes.md", filters)

    monkeypatch.setattr(dialogs.QFileDialog, "getOpenFileName", staticmethod(fake_get_open_file_name))

    result = dialogs.open_file_dialog(start_dir=Path("docs"))

    assert result == Path("C:/tmp/notes.md")
    assert captured["args"][1] == "Open Document"
    assert captured["args"][2].endswith("docs")
    assert isinstance(captured["args"][3], str)


def test_save_file_dialog_appends_default_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_save_file_name(parent, caption, directory, filters):
        return ("C:/tmp/new-note", filters)

    monkeypatch.setattr(dialogs.QFileDialog, "getSaveFileName", staticmethod(fake_get_save_file_name))

    result = dialogs.save_file_dialog(default_suffix="md")

    assert result == Path("C:/tmp/new-note.md")