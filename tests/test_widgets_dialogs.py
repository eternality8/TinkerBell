"""Tests for the widgets.dialogs module."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLineEdit, QPushButton, QSpinBox

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
        request_timeout=75.0,
    )


def test_settings_dialog_gather_settings_reflects_changes(qtbot, dialog_settings: Settings) -> None:
    dialog = SettingsDialog(settings=dialog_settings)
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


def test_settings_dialog_test_button_runs_api_tester(qtbot, dialog_settings: Settings) -> None:
    captured: dict[str, Settings] = {}

    def tester(settings: Settings) -> ValidationResult:
        captured["settings"] = settings
        return ValidationResult(ok=True, message="API reachable")

    dialog = SettingsDialog(settings=dialog_settings, api_tester=tester)
    qtbot.addWidget(dialog)

    test_button = dialog.findChild(QPushButton, "test_button")
    assert test_button is not None

    qtbot.mouseClick(test_button, Qt.MouseButton.LeftButton)
    assert captured["settings"].api_key == dialog_settings.api_key
    assert dialog.api_tested is True
    assert "api" in (dialog.last_api_test.message or "").lower()


def test_open_file_dialog_wraps_qfiledialog(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_get_open_file_name(parent, caption, directory, filters):
        captured["args"] = (parent, caption, directory, filters)
        return ("C:/tmp/notes.md", filters)

    monkeypatch.setattr(dialogs.QFileDialog, "getOpenFileName", staticmethod(fake_get_open_file_name))

    result = dialogs.open_file_dialog(start_dir=Path("docs"))

    assert result == Path("C:/tmp/notes.md")
    args = captured["args"]
    assert isinstance(args, tuple)
    assert args[1] == "Open Document"
    assert str(args[2]).endswith("docs")
    assert isinstance(args[3], str)


def test_save_file_dialog_appends_default_suffix(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get_save_file_name(parent, caption, directory, filters):
        return ("C:/tmp/new-note", filters)

    monkeypatch.setattr(dialogs.QFileDialog, "getSaveFileName", staticmethod(fake_get_save_file_name))

    result = dialogs.save_file_dialog(default_suffix="md")

    assert result == Path("C:/tmp/new-note.md")


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