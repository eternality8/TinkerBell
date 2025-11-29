"""Settings dialog for AI and application configuration."""

from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence
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
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...services.settings import (
    ContextPolicySettings,
    DEFAULT_EMBEDDING_MODE,
    EMBEDDING_MODE_CHOICES,
    Settings,
)
from ...theme import theme_manager

if TYPE_CHECKING:
    from ...ui.embedding_controller import EmbeddingValidationResult

__all__ = [
    "SettingsDialog",
    "SettingsDialogResult",
    "ValidationResult",
    "SettingsValidator",
    "SettingsTester",
    "show_settings_dialog",
    "test_ai_api_settings",
    "test_embedding_settings",
]

# Constants
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


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a set of settings."""

    ok: bool
    message: str = ""


SettingsValidator = Callable[[Settings], "ValidationResult | tuple[bool, str] | bool"]
SettingsTester = Callable[[Settings], "ValidationResult | tuple[bool, str] | bool"]


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


def _coerce_validation_result(result: ValidationResult | tuple[bool, str] | bool) -> ValidationResult:
    if isinstance(result, ValidationResult):
        return result
    if isinstance(result, tuple):
        ok, message = result
        return ValidationResult(bool(ok), str(message))
    return ValidationResult(bool(result), "")


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
        self._validate_button: QPushButton | None = None
        self._test_button: QPushButton | None = None
        self._embedding_test_button: QPushButton | None = None

        metadata = dict(getattr(self._original, "metadata", {}) or {})
        self._metadata_snapshot = metadata
        self._embedding_api_metadata = dict(metadata.get("embedding_api") or {})
        raw_mode = str(metadata.get("embedding_mode") or DEFAULT_EMBEDDING_MODE)
        self._initial_embedding_mode = raw_mode if raw_mode in EMBEDDING_MODE_CHOICES else DEFAULT_EMBEDDING_MODE

        self._init_general_widgets()
        self._init_embedding_widgets()
        self._init_feature_widgets()
        self._init_runtime_widgets()
        self._init_policy_widgets()
        self._build_layout()

    # Continued in next section...

    def _init_general_widgets(self) -> None:
        """Initialize general settings widgets."""
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
        self._temperature_input.setValue(max(0.0, min(temp_default, 2.0)))
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

    def _init_embedding_widgets(self) -> None:
        """Initialize embedding-related widgets."""
        embedding_model_default = getattr(self._original, "embedding_model_name", "text-embedding-3-large") or "text-embedding-3-large"
        self._embedding_model_input = QLineEdit(embedding_model_default)
        self._embedding_model_input.setObjectName("embedding_model_input")
        self._embedding_model_input.textChanged.connect(self._validate_embedding_fields)
        self._embedding_model_hint = QLabel("Model name for the embedding API.")
        self._embedding_model_hint.setObjectName("embedding_model_hint")
        self._prepare_hint_label(self._embedding_model_hint)

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

        self._embedding_mode_stack = QStackedWidget()
        self._embedding_mode_stack.setObjectName("embedding_mode_stack")
        self._mode_index_map: dict[str, int] = {}
        for index, (mode, panel) in enumerate([
            ("disabled", self._build_disabled_panel()),
            ("same-api", self._build_same_api_panel()),
            ("custom-api", self._build_custom_api_panel()),
            ("local", self._build_local_embedding_panel()),
        ]):
            self._mode_index_map[mode] = index
            self._embedding_mode_stack.addWidget(panel)

    def _init_feature_widgets(self) -> None:
        """Initialize feature toggle widgets."""
        self._debug_checkbox = QCheckBox("Enable debug logging")
        self._debug_checkbox.setObjectName("debug_logging_checkbox")
        self._debug_checkbox.setChecked(self._original.debug_logging)

        self._event_log_checkbox = QCheckBox("Capture per-chat event logs (JSONL)")
        self._event_log_checkbox.setObjectName("debug_event_logging_checkbox")
        self._event_log_checkbox.setToolTip("Writes turn-by-turn JSONL files when debug logging is enabled.")
        self._event_log_checkbox.setChecked(bool(getattr(self._original, "debug_event_logging", False)))

        self._tool_panel_checkbox = QCheckBox("Show tool activity panel (debug view)")
        self._tool_panel_checkbox.setObjectName("tool_activity_checkbox")
        self._tool_panel_checkbox.setChecked(bool(getattr(self._original, "show_tool_activity_panel", False)))

    def _init_runtime_widgets(self) -> None:
        """Initialize runtime configuration widgets."""
        self._max_tool_iterations_input = QSpinBox()
        self._max_tool_iterations_input.setObjectName("max_tool_iterations_input")
        self._max_tool_iterations_input.setRange(1, 200)
        self._max_tool_iterations_input.setValue(max(1, int(getattr(self._original, "max_tool_iterations", 8) or 8)))
        self._max_tool_iterations_input.setSuffix(" loops")

        self._max_context_tokens_input = QSpinBox()
        self._max_context_tokens_input.setObjectName("max_context_tokens_input")
        self._max_context_tokens_input.setRange(32_000, 512_000)
        self._max_context_tokens_input.setSingleStep(1_000)
        context_default = int(getattr(self._original, "max_context_tokens", 128_000) or 128_000)
        self._max_context_tokens_input.setValue(max(32_000, min(context_default, 512_000)))
        self._max_context_tokens_input.setSuffix(" tokens")
        self._max_context_tokens_input.valueChanged.connect(self._update_reserve_hint)

        self._response_token_reserve_input = QSpinBox()
        self._response_token_reserve_input.setObjectName("response_token_reserve_input")
        self._response_token_reserve_input.setRange(4_000, 64_000)
        self._response_token_reserve_input.setSingleStep(500)
        reserve_default = int(getattr(self._original, "response_token_reserve", 16_000) or 16_000)
        self._response_token_reserve_input.setValue(max(4_000, min(reserve_default, 64_000)))
        self._response_token_reserve_input.setSuffix(" tokens")
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
        self._timeout_hint = QLabel("Applies to chat + tool turns; keep between 5–300 seconds.")
        self._timeout_hint.setObjectName("request_timeout_hint")
        self._prepare_hint_label(self._timeout_hint)
        self._request_timeout_input.valueChanged.connect(self._update_timeout_hint)

    def _init_policy_widgets(self) -> None:
        """Initialize context policy widgets."""
        policy_original = getattr(self._original, "context_policy", ContextPolicySettings())

        self._context_policy_enabled = QCheckBox("Enable context budget policy")
        self._context_policy_enabled.setObjectName("context_policy_enabled_checkbox")
        self._context_policy_enabled.setChecked(bool(getattr(policy_original, "enabled", False)))

        self._context_policy_dry_run = QCheckBox("Dry run only (log decisions, do not block)")
        self._context_policy_dry_run.setObjectName("context_policy_dry_run_checkbox")
        self._context_policy_dry_run.setChecked(bool(getattr(policy_original, "dry_run", True)))

        has_prompt_override = getattr(policy_original, "prompt_budget_override", None)
        self._context_prompt_override_toggle = QCheckBox("Custom prompt budget")
        self._context_prompt_override_toggle.setObjectName("context_policy_prompt_override_toggle")
        self._context_prompt_override_toggle.setChecked(has_prompt_override is not None)

        self._context_prompt_override_input = QSpinBox()
        self._context_prompt_override_input.setObjectName("context_policy_prompt_override_input")
        self._context_prompt_override_input.setRange(32_000, 512_000)
        self._context_prompt_override_input.setSingleStep(1_000)
        self._context_prompt_override_input.setSuffix(" tokens")
        self._context_prompt_override_input.setValue(int(has_prompt_override or 128_000))
        self._context_prompt_override_input.setEnabled(has_prompt_override is not None)

        has_reserve_override = getattr(policy_original, "response_reserve_override", None)
        self._context_reserve_override_toggle = QCheckBox("Custom response reserve")
        self._context_reserve_override_toggle.setObjectName("context_policy_reserve_override_toggle")
        self._context_reserve_override_toggle.setChecked(has_reserve_override is not None)

        self._context_reserve_override_input = QSpinBox()
        self._context_reserve_override_input.setObjectName("context_policy_reserve_override_input")
        self._context_reserve_override_input.setRange(2_000, 64_000)
        self._context_reserve_override_input.setSingleStep(500)
        self._context_reserve_override_input.setSuffix(" tokens")
        self._context_reserve_override_input.setValue(int(has_reserve_override or 16_000))
        self._context_reserve_override_input.setEnabled(has_reserve_override is not None)

        self._context_policy_hint = QLabel("Context policy runs budgeting checks before calling the model.")
        self._context_policy_hint.setObjectName("context_policy_hint_label")
        self._prepare_hint_label(self._context_policy_hint)

        self._context_policy_enabled.toggled.connect(self._update_context_policy_hint)
        self._context_policy_dry_run.toggled.connect(self._update_context_policy_hint)
        self._context_prompt_override_toggle.toggled.connect(self._toggle_context_prompt_override)
        self._context_reserve_override_toggle.toggled.connect(self._toggle_context_reserve_override)

    def _build_layout(self) -> None:
        """Build the dialog layout with tabs."""
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

        general_tab = _build_form_tab([
            ("Base URL", base_url_container),
            ("API Key", api_container),
            ("Model", self._model_combo),
            ("Temperature", temperature_container),
            ("Organization", self._organization_input),
            ("Theme", self._theme_combo),
        ])

        embedding_tab = _build_form_tab([
            ("Mode", embedding_mode_container),
            ("Embedding Model", self._embedding_model_container),
            ("Settings", embedding_mode_stack_container),
        ])

        features_tab = _build_form_tab([
            ("Debug", self._debug_checkbox),
            ("Event Logs", self._event_log_checkbox),
            ("Tool Traces", self._tool_panel_checkbox),
        ])

        runtime_tab = _build_form_tab([
            ("Max Tool Iterations", self._max_tool_iterations_input),
            ("Max Context Tokens", self._max_context_tokens_input),
            ("Response Token Reserve", reserve_container),
            ("AI Timeout", timeout_container),
        ])

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
        self._validation_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
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
    # Panel builders
    # ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Standalone functions
# ------------------------------------------------------------------
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
