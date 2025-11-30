"""Tests for the settings persistence layer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tinkerbell.services.settings import DebugSettings, SecretVault, Settings, SettingsStore


def test_load_returns_defaults_when_file_missing(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json", vault=SecretVault(key_path=tmp_path / "key"))

    settings = store.load()

    expected = Settings()
    expected.metadata["embedding_mode"] = "same-api"
    # Backend is derived from mode: same-api -> langchain
    expected = replace(expected, embedding_backend="langchain")
    assert settings == expected


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

    expected_metadata = dict(original.metadata)
    expected_metadata["embedding_mode"] = "same-api"
    # Backend is derived from mode: same-api -> langchain
    expected = replace(original, metadata=expected_metadata, embedding_backend="langchain")

    assert reloaded == expected


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


def test_bool_env_overrides_enable_debug_event_logging(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    SettingsStore(path).save(Settings())
    monkeypatch.setenv("TINKERBELL_DEBUG_EVENT_LOGGING", "1")

    overridden = SettingsStore(path).load()

    assert overridden.debug_event_logging is True


def test_float_env_override_sets_temperature(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    SettingsStore(path).save(Settings())
    monkeypatch.setenv("TINKERBELL_TEMPERATURE", "0.95")

    overridden = SettingsStore(path).load()

    assert overridden.temperature == pytest.approx(0.95)


def test_float_env_override_sets_tool_timeout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    SettingsStore(path).save(Settings())
    monkeypatch.setenv("TINKERBELL_TOOL_TIMEOUT", "180.0")

    overridden = SettingsStore(path).load()

    assert overridden.tool_timeout == pytest.approx(180.0)


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


def test_embedding_mode_migration_disables_backend(tmp_path: Path) -> None:
    """When loading old settings without embedding_mode, embeddings are disabled."""
    target = tmp_path / "settings.json"
    # Old settings file with no embedding_mode
    target.write_text(json.dumps({"embedding_backend": "langchain", "version": 2}), encoding="utf-8")

    settings = SettingsStore(target).load()

    # Missing embedding_mode causes both mode and backend to be set to disabled
    assert settings.embedding_backend == "disabled"
    assert settings.metadata.get("embedding_mode") == "disabled"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["version"] == 3
    assert payload.get("embedding_backend") == "disabled"
    assert payload.get("metadata", {}).get("embedding_mode") == "disabled"


def test_local_mode_forces_sentence_transformers(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text(
        json.dumps(
            {
                "embedding_backend": "auto",
                "metadata": {"embedding_mode": "local"},
                "version": 3,
            }
        ),
        encoding="utf-8",
    )

    settings = SettingsStore(target).load()

    assert settings.embedding_backend == "sentence-transformers"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload.get("embedding_backend") == "sentence-transformers"


def test_embedding_api_secret_encrypted_and_roundtripped(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    secret_value = "secret-token-value"
    original = Settings(metadata={"embedding_api": {"api_key": secret_value, "base_url": "https://example"}})

    store.save(original)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    embedding_api = persisted.get("metadata", {}).get("embedding_api", {})
    assert "api_key" not in embedding_api or not embedding_api.get("api_key")
    assert embedding_api.get("api_key_ciphertext")
    assert embedding_api.get("api_key_hint", "").startswith("se")

    reloaded = store.load()

    api_metadata = reloaded.metadata.get("embedding_api", {})
    assert api_metadata.get("api_key") == secret_value


def test_custom_api_mode_forces_remote_backend(tmp_path: Path) -> None:
    """custom-api mode derives langchain backend regardless of stored value."""
    target = tmp_path / "settings.json"
    target.write_text(
        json.dumps(
            {
                "embedding_backend": "sentence-transformers",
                "metadata": {"embedding_mode": "custom-api"},
                "version": 3,
            }
        ),
        encoding="utf-8",
    )

    settings = SettingsStore(target).load()

    # custom-api mode -> langchain backend
    assert settings.embedding_backend == "langchain"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload.get("embedding_backend") == "langchain"
    assert payload.get("metadata", {}).get("embedding_mode") == "custom-api"


def test_same_api_mode_resets_unknown_backend(tmp_path: Path) -> None:
    """same-api mode derives langchain backend regardless of stored value."""
    target = tmp_path / "settings.json"
    target.write_text(
        json.dumps(
            {
                "embedding_backend": "bogus-backend",
                "metadata": {"embedding_mode": "same-api"},
                "version": 3,
            }
        ),
        encoding="utf-8",
    )

    settings = SettingsStore(target).load()

    # same-api mode -> langchain backend
    assert settings.embedding_backend == "langchain"
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload.get("embedding_backend") == "langchain"


def test_local_dtype_default_label_migrates(tmp_path: Path) -> None:
    target = tmp_path / "settings.json"
    target.write_text(
        json.dumps(
            {
                "metadata": {
                    "embedding_mode": "local",
                    "st_dtype": "Default",
                },
                "version": 3,
            }
        ),
        encoding="utf-8",
    )

    settings = SettingsStore(target).load()

    assert settings.metadata.get("st_dtype") == ""
    persisted = json.loads(target.read_text(encoding="utf-8"))
    assert persisted.get("metadata", {}).get("st_dtype") == ""
