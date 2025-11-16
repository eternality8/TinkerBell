"""Tests for the settings persistence layer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tinkerbell.services.settings import DebugSettings, SecretVault, Settings, SettingsStore


def test_load_returns_defaults_when_file_missing(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json", vault=SecretVault(key_path=tmp_path / "key"))

    settings = store.load()

    assert settings == Settings()


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    original = Settings(
        base_url="https://example.com/v1",
        api_key="super-secret",
        model="gpt-4.1-mini",
        theme="dark",
        organization="acme",
        autosave_interval=45.0,
        default_headers={"X-Test": "1"},
        metadata={"env": "dev"},
        recent_files=["notes.md"],
        unsaved_snapshot={"text": "draft", "language": "markdown", "selection": [0, 5]},
        unsaved_snapshots={"/tmp/demo.md": {"text": "draft", "language": "markdown", "selection": [0, 5]}},
        max_tool_iterations=12,
    )

    store.save(original)
    reloaded = SettingsStore(path).load()

    assert reloaded == original


def test_load_legacy_plaintext_api_key(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text(
        json.dumps(
            {
                "base_url": "https://old",
                "api_key": "plain-key",
                "model": "gpt-3.5",
            }
        ),
        encoding="utf-8",
    )

    loaded = SettingsStore(target).load()

    assert loaded.api_key == "plain-key"
    assert loaded.base_url == "https://old"


def test_env_overrides_take_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    SettingsStore(path).save(Settings(base_url="https://local", api_key="abc"))
    monkeypatch.setenv("TINKERBELL_BASE_URL", "https://env-base")
    monkeypatch.setenv("TINKERBELL_API_KEY", "env-key")

    overridden = SettingsStore(path).load()

    assert overridden.base_url == "https://env-base"
    assert overridden.api_key == "env-key"


def test_bool_env_overrides_enable_debug_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    SettingsStore(path).save(Settings())
    monkeypatch.setenv("TINKERBELL_DEBUG_LOGGING", "true")

    overridden = SettingsStore(path).load()

    assert overridden.debug_logging is True


def test_debug_settings_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    original = Settings(debug=DebugSettings(token_logging_enabled=True, token_log_limit=321))

    store.save(original)
    loaded = store.load()

    assert loaded.debug.token_logging_enabled is True
    assert loaded.debug.token_log_limit == 321


def test_load_applies_cli_overrides(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    store.save(Settings(base_url="https://ui", request_timeout=45.0))

    loaded = store.load(overrides={"base_url": "https://cli", "request_timeout": 30.5})

    assert loaded.base_url == "https://cli"
    assert loaded.request_timeout == pytest.approx(30.5)


def test_env_overrides_take_priority_over_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    store.save(Settings(base_url="https://ui"))

    monkeypatch.setenv("TINKERBELL_BASE_URL", "https://env")

    loaded = store.load(overrides={"base_url": "https://cli"})

    assert loaded.base_url == "https://env"


def test_secret_vault_forced_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("tinkerbell.services.settings._dpapi_supported", lambda: False)
    monkeypatch.setenv("TINKERBELL_SECRET_BACKEND", "dpapi")
    vault = SecretVault(key_path=tmp_path / "settings.key")

    assert vault.strategy == "fernet"
    token = vault.encrypt("super-secret")
    assert token.startswith("fernet:")
    assert vault.decrypt(token) == "super-secret"
