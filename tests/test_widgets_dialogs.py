"""Tests for the widgets.dialogs module."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
)

from tinkerbell.services.settings import DEFAULT_EMBEDDING_MODE, Settings
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
    theme_combo = dialog.findChild(QComboBox, "theme_combo")
    temperature_input = dialog.findChild(QDoubleSpinBox, "temperature_input")
    tool_checkbox = dialog.findChild(QCheckBox, "tool_activity_checkbox")
    timeout_input = dialog.findChild(QDoubleSpinBox, "request_timeout_input")
    context_input = dialog.findChild(QSpinBox, "max_context_tokens_input")
    reserve_input = dialog.findChild(QSpinBox, "response_token_reserve_input")
    policy_enable = dialog.findChild(QCheckBox, "context_policy_enabled_checkbox")
    policy_dry_run = dialog.findChild(QCheckBox, "context_policy_dry_run_checkbox")
    prompt_toggle = dialog.findChild(QCheckBox, "context_policy_prompt_override_toggle")
    prompt_input = dialog.findChild(QSpinBox, "context_policy_prompt_override_input")
    reserve_toggle = dialog.findChild(QCheckBox, "context_policy_reserve_override_toggle")
    reserve_override_input = dialog.findChild(QSpinBox, "context_policy_reserve_override_input")

    assert base_input is not None
    assert api_input is not None
    assert model_combo is not None
    assert organization_input is not None
    assert theme_combo is not None
    assert temperature_input is not None
    assert tool_checkbox is not None
    assert timeout_input is not None
    assert context_input is not None
    assert reserve_input is not None
    assert policy_enable is not None
    assert policy_dry_run is not None
    assert prompt_toggle is not None
    assert prompt_input is not None
    assert reserve_toggle is not None
    assert reserve_override_input is not None

    base_input.setText("https://example.com/v2")
    api_input.setText("new-key")
    model_combo.setEditText("gpt-custom")
    organization_input.setText("acme")
    theme_combo.setCurrentIndex(0)
    selected_theme = theme_combo.currentData(Qt.ItemDataRole.UserRole) or theme_combo.currentText()
    temperature_input.setValue(0.85)
    tool_checkbox.setChecked(True)
    timeout_input.setValue(42.5)
    context_input.setValue(256_000)
    reserve_input.setValue(20_000)
    policy_enable.setChecked(True)
    policy_dry_run.setChecked(False)
    prompt_toggle.setChecked(True)
    prompt_input.setValue(100_000)
    reserve_toggle.setChecked(True)
    reserve_override_input.setValue(12_000)

    updated = dialog.gather_settings()

    assert updated.base_url == "https://example.com/v2"
    assert updated.api_key == "new-key"
    assert updated.model == "gpt-custom"
    assert updated.organization == "acme"
    assert updated.theme == selected_theme
    assert updated.temperature == pytest.approx(0.85)
    assert updated.show_tool_activity_panel is True
    assert updated.request_timeout == pytest.approx(42.5)
    assert updated.max_context_tokens == 256_000
    assert updated.response_token_reserve == 20_000
    assert updated.context_policy.enabled is True
    assert updated.context_policy.dry_run is False
    assert updated.context_policy.prompt_budget_override == 100_000
    assert updated.context_policy.response_reserve_override == 12_000
    assert updated.metadata.get("embedding_mode") == DEFAULT_EMBEDDING_MODE


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


def test_settings_dialog_custom_mode_updates_metadata(qtbot, dialog_settings: Settings) -> None:
    dialog = SettingsDialog(settings=dialog_settings, show_toasts=False)
    qtbot.addWidget(dialog)

    mode_combo = dialog.findChild(QComboBox, "embedding_mode_combo")
    custom_index = mode_combo.findData("custom-api")
    assert custom_index >= 0
    mode_combo.setCurrentIndex(custom_index)

    base_input = dialog.findChild(QLineEdit, "embedding_custom_base_url_input")
    key_input = dialog.findChild(QLineEdit, "embedding_custom_api_key_input")
    headers_input = dialog.findChild(QPlainTextEdit, "embedding_custom_headers_input")
    assert base_input is not None
    assert key_input is not None
    assert headers_input is not None

    base_input.setText("https://embeddings.example/v1")
    key_input.setText("embed-key")
    headers_input.setPlainText('{"X-Test": "1"}')

    updated = dialog.gather_settings()

    assert updated.metadata.get("embedding_mode") == "custom-api"
    api_metadata = updated.metadata.get("embedding_api", {})
    assert api_metadata.get("base_url") == "https://embeddings.example/v1"
    assert api_metadata.get("api_key") == "embed-key"
    assert api_metadata.get("default_headers", {}).get("X-Test") == "1"


def test_settings_dialog_custom_mode_preserves_stored_key(qtbot, dialog_settings: Settings) -> None:
    stored_settings = replace(
        dialog_settings,
        metadata={
            "embedding_mode": "custom-api",
            "embedding_api": {
                "base_url": "https://embeddings.example/v1",
                "api_key": "stored-secret",
                "api_key_hint": "st****et",
            },
        },
    )

    dialog = SettingsDialog(settings=stored_settings, show_toasts=False)
    qtbot.addWidget(dialog)

    mode_combo = dialog.findChild(QComboBox, "embedding_mode_combo")
    custom_index = mode_combo.findData("custom-api")
    assert custom_index >= 0
    mode_combo.setCurrentIndex(custom_index)

    base_input = dialog.findChild(QLineEdit, "embedding_custom_base_url_input")
    key_input = dialog.findChild(QLineEdit, "embedding_custom_api_key_input")
    assert base_input is not None
    assert key_input is not None

    base_input.setText("https://embeddings.example/v2")
    key_input.clear()
    assert key_input.text().strip() == ""

    updated = dialog.gather_settings()

    api_metadata = updated.metadata.get("embedding_api", {})
    assert api_metadata.get("api_key") == "stored-secret"
    assert api_metadata.get("api_key_hint") == "st****et"
    assert api_metadata.get("base_url") == "https://embeddings.example/v2"


def test_settings_dialog_custom_mode_embedding_test_reuses_stored_key(
    qtbot, dialog_settings: Settings
) -> None:
    stored_settings = replace(
        dialog_settings,
        metadata={
            "embedding_mode": "custom-api",
            "embedding_api": {
                "base_url": "https://embeddings.example/v1",
                "api_key": "stored-secret",
            },
        },
    )

    captured: dict[str, Settings] = {}

    def embedding_tester(settings: Settings) -> ValidationResult:
        captured["settings"] = settings
        return ValidationResult(ok=True, message="Embeddings reachable")

    dialog = SettingsDialog(settings=stored_settings, embedding_tester=embedding_tester, show_toasts=False)
    qtbot.addWidget(dialog)

    key_input = dialog.findChild(QLineEdit, "embedding_custom_api_key_input")
    assert key_input is not None
    key_input.clear()

    test_button = dialog.findChild(QPushButton, "embedding_test_button")
    assert test_button is not None

    qtbot.mouseClick(test_button, Qt.MouseButton.LeftButton)

    api_metadata = captured["settings"].metadata.get("embedding_api", {})
    assert api_metadata.get("api_key") == "stored-secret"


def test_settings_dialog_custom_mode_auto_prefixes_scheme(qtbot, dialog_settings: Settings) -> None:
    dialog = SettingsDialog(
        settings=dialog_settings,
        embedding_tester=lambda s: ValidationResult(True, "ok"),
        show_toasts=False,
    )
    qtbot.addWidget(dialog)

    mode_combo = dialog.findChild(QComboBox, "embedding_mode_combo")
    key_input = dialog.findChild(QLineEdit, "embedding_custom_api_key_input")
    base_input = dialog.findChild(QLineEdit, "embedding_custom_base_url_input")
    assert mode_combo is not None
    assert key_input is not None
    assert base_input is not None

    custom_index = mode_combo.findData("custom-api")
    assert custom_index >= 0
    mode_combo.setCurrentIndex(custom_index)

    key_input.setText("secret")
    base_input.setText("neko:8666/v1")
    QApplication.processEvents()

    assert base_input.text() == "https://neko:8666/v1"
    test_button = dialog.findChild(QPushButton, "embedding_test_button")
    assert test_button is not None
    assert test_button.isEnabled() is True


def test_settings_dialog_custom_mode_accepts_http_endpoints(qtbot, dialog_settings: Settings) -> None:
    dialog = SettingsDialog(settings=dialog_settings, show_toasts=False)
    qtbot.addWidget(dialog)

    mode_combo = dialog.findChild(QComboBox, "embedding_mode_combo")
    key_input = dialog.findChild(QLineEdit, "embedding_custom_api_key_input")
    base_input = dialog.findChild(QLineEdit, "embedding_custom_base_url_input")
    assert mode_combo is not None
    assert key_input is not None
    assert base_input is not None

    custom_index = mode_combo.findData("custom-api")
    assert custom_index >= 0
    mode_combo.setCurrentIndex(custom_index)

    key_input.setText("secret")
    base_input.setText("http://localhost:8666/v1")
    QApplication.processEvents()

    updated = dialog.gather_settings()
    api_metadata = updated.metadata.get("embedding_api", {})
    assert api_metadata.get("base_url") == "http://localhost:8666/v1"


def test_settings_dialog_local_mode_requires_model_path(qtbot, dialog_settings: Settings) -> None:
    dialog = SettingsDialog(settings=dialog_settings, show_toasts=False)
    qtbot.addWidget(dialog)

    mode_combo = dialog.findChild(QComboBox, "embedding_mode_combo")
    local_index = mode_combo.findData("local")
    assert local_index >= 0
    mode_combo.setCurrentIndex(local_index)

    button_box = dialog.findChild(QDialogButtonBox)
    assert button_box is not None
    save_button = button_box.button(QDialogButtonBox.StandardButton.Save)
    assert save_button is not None

    model_input = dialog.findChild(QLineEdit, "embedding_local_model_path_input")
    assert model_input is not None
    model_input.clear()
    QApplication.processEvents()
    assert save_button.isEnabled() is False

    model_input.setText("sentence-transformers/all-MiniLM-L6-v2")
    QApplication.processEvents()
    assert save_button.isEnabled() is True


def test_settings_dialog_embedding_test_button_uses_tester(qtbot, dialog_settings: Settings) -> None:
    captured: dict[str, Settings] = {}

    def embedding_tester(settings: Settings) -> ValidationResult:
        captured["settings"] = settings
        return ValidationResult(ok=True, message="Embeddings reachable")

    dialog = SettingsDialog(settings=dialog_settings, embedding_tester=embedding_tester, show_toasts=False)
    qtbot.addWidget(dialog)

    test_button = dialog.findChild(QPushButton, "embedding_test_button")
    assert test_button is not None

    qtbot.mouseClick(test_button, Qt.MouseButton.LeftButton)

    assert captured["settings"].api_key == dialog_settings.api_key
    assert dialog.embedding_tested is True
    assert "embedding" in (dialog.last_embedding_test.message or "").lower()


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


def test_document_export_dialog_displays_document_preview(qtbot) -> None:
    dialog = DocumentExportDialog(
        parent=None,
        document_text="alpha beta gamma",
        token_budget=1000,
    )
    qtbot.addWidget(dialog)

    stats_label = dialog.findChild(QLabel, "preview_stats_label")
    assert stats_label is not None
    assert "Document preview" in stats_label.text()

    preview_widget = dialog.findChild(QPlainTextEdit, "save_preview")
    assert preview_widget is not None
    assert "alpha beta" in preview_widget.toPlainText()

    assert dialog.findChild(QComboBox, "preview_mode_combo") is None


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