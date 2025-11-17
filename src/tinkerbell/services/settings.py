"""Settings dataclasses and persistence helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping

from cryptography.fernet import Fernet, InvalidToken

__all__ = ["Settings", "SettingsStore", "SecretVault", "DebugSettings", "ContextPolicySettings"]

LOGGER = logging.getLogger(__name__)
_SETTINGS_DIR = Path.home() / ".tinkerbell"
_DEFAULT_SETTINGS_PATH = _SETTINGS_DIR / "settings.json"
_SETTINGS_VERSION = 2
_ENV_OVERRIDES: Mapping[str, str] = {
    "TINKERBELL_API_KEY": "api_key",
    "TINKERBELL_BASE_URL": "base_url",
    "TINKERBELL_MODEL": "model",
    "TINKERBELL_EMBEDDING_BACKEND": "embedding_backend",
    "TINKERBELL_EMBEDDING_MODEL": "embedding_model_name",
    "TINKERBELL_THEME": "theme",
    "TINKERBELL_ORGANIZATION": "organization",
}
_BOOL_ENV_OVERRIDES: Mapping[str, str] = {
    "TINKERBELL_DEBUG_LOGGING": "debug_logging",
    "TINKERBELL_TOOL_ACTIVITY_PANEL": "show_tool_activity_panel",
    "TINKERBELL_PHASE3_OUTLINE_TOOLS": "phase3_outline_tools",
}
_FLOAT_ENV_OVERRIDES: Mapping[str, str] = {
    "TINKERBELL_REQUEST_TIMEOUT": "request_timeout",
}
_TRUE_VALUES = {"1", "true", "yes", "on", "debug"}
_API_KEY_FIELD = "api_key_ciphertext"
_SECRET_BACKEND_ENV = "TINKERBELL_SECRET_BACKEND"


@dataclass(slots=True)
class DebugSettings:
    """Debug/diagnostic toggles grouped to match the AI v2 plan."""

    token_logging_enabled: bool = False
    token_log_limit: int = 200


@dataclass(slots=True)
class ContextPolicySettings:
    """Context budget policy configuration surfaced in the settings UI."""

    enabled: bool = True
    dry_run: bool = False
    prompt_budget_override: int | None = None
    response_reserve_override: int | None = None
    emergency_buffer: int = 2_000


@dataclass(slots=True)
class Settings:
    """User-configurable settings persisted between sessions."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    theme: str = "default"
    organization: str | None = None
    request_timeout: float = 90.0
    max_retries: int = 3
    retry_min_seconds: float = 0.5
    retry_max_seconds: float = 6.0
    max_tool_iterations: int = 8
    max_context_tokens: int = 128_000
    response_token_reserve: int = 16_000
    embedding_backend: str = "auto"
    embedding_model_name: str = "text-embedding-3-large"
    autosave_interval: float = 60.0
    default_headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    recent_files: list[str] = field(default_factory=list)
    last_open_file: str | None = None
    unsaved_snapshot: dict[str, Any] | None = None
    unsaved_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    untitled_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    open_tabs: list[dict[str, Any]] = field(default_factory=list)
    active_tab_id: str | None = None
    next_untitled_index: int = 1
    font_family: str = "JetBrains Mono"
    font_size: int = 13
    window_geometry: str | None = None
    debug_logging: bool = False
    show_tool_activity_panel: bool = False
    phase3_outline_tools: bool = False
    debug: DebugSettings = field(default_factory=DebugSettings)
    context_policy: ContextPolicySettings = field(default_factory=ContextPolicySettings)


class SecretProvider(ABC):
    """Interface for encrypting and decrypting sensitive strings."""

    name: str = "unknown"

    @abstractmethod
    def encrypt(self, secret: str) -> str:
        """Return an encoded representation of ``secret`` suitable for storage."""

    @abstractmethod
    def decrypt(self, token: str) -> str:
        """Return the plaintext representation of ``token``."""


class WindowsSecretProvider(SecretProvider):
    """Secret provider backed by the Windows DPAPI interfaces."""

    name = "dpapi"

    def encrypt(self, secret: str) -> str:
        raw = _dpapi_protect(secret.encode("utf-8"))
        return base64.urlsafe_b64encode(raw).decode("ascii")

    def decrypt(self, token: str) -> str:
        payload = base64.urlsafe_b64decode(token.encode("ascii"))
        return _dpapi_unprotect(payload).decode("utf-8")


class FernetSecretProvider(SecretProvider):
    """Secret provider that uses a symmetric Fernet key stored on disk."""

    name = "fernet"

    def __init__(self, key_path: Path | None = None) -> None:
        self._key_path = key_path or (_SETTINGS_DIR / "settings.key")
        self._fernet: Fernet | None = None

    def encrypt(self, secret: str) -> str:
        token = self._get_fernet().encrypt(secret.encode("utf-8"))
        return token.decode("ascii")

    def decrypt(self, token: str) -> str:
        raw = self._get_fernet().decrypt(token.encode("ascii"))
        return raw.decode("utf-8")

    def _get_fernet(self) -> Fernet:
        if self._fernet is None:
            key = self._load_or_create_key()
            self._fernet = Fernet(key)
        return self._fernet

    def _load_or_create_key(self) -> bytes:
        path = self._key_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return path.read_bytes().strip()
        key = Fernet.generate_key()
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_bytes(key)
        if os.name != "nt":  # pragma: no cover - depends on OS
            os.chmod(tmp_path, 0o600)
        tmp_path.replace(path)
        return key


class SettingsStore:
    """Persistence adapter for :class:`Settings`."""

    def __init__(self, path: Path | None = None, *, vault: SecretVault | None = None) -> None:
        self._path = path or _DEFAULT_SETTINGS_PATH
        key_path = self._path.with_suffix(".key")
        self._vault = vault or SecretVault(key_path=key_path)

    @property
    def path(self) -> Path:
        """Return the resolved path backing this store."""

        return self._path

    @property
    def vault(self) -> SecretVault:
        """Return the secret vault managing API key encryption."""

        return self._vault

    def load(self, *, overrides: Mapping[str, Any] | None = None) -> Settings:
        """Load settings from disk, applying CLI/environment overrides when present."""

        payload = self._read_payload()
        settings = Settings()
        needs_migration = False

        if payload:
            plaintext_key, migrated = self._decrypt_api_key(
                payload.pop(_API_KEY_FIELD, None), payload.pop("api_key", None)
            )
            needs_migration = migrated
            data = _filter_fields(payload)
            debug_payload = data.get("debug")
            if isinstance(debug_payload, Mapping):
                try:
                    data["debug"] = DebugSettings(**debug_payload)
                except TypeError:
                    data["debug"] = DebugSettings()
            policy_payload = data.get("context_policy")
            if isinstance(policy_payload, Mapping):
                try:
                    data["context_policy"] = ContextPolicySettings(**policy_payload)
                except TypeError:
                    data["context_policy"] = ContextPolicySettings()
            try:
                settings = Settings(**data)
            except TypeError as exc:
                LOGGER.warning("Settings payload contained unexpected data: %s", exc)
                settings = Settings()
            if plaintext_key:
                settings = replace(settings, api_key=plaintext_key)

        version_mismatch = bool(payload) and payload.get("version") != _SETTINGS_VERSION
        if needs_migration or version_mismatch:
            try:
                self.save(settings)
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning("Failed to migrate settings payload: %s", exc)

        if overrides:
            settings = self._apply_overrides(settings, overrides, source="CLI")

        settings = self._apply_env_overrides(settings)
        return settings

    def save(self, settings: Settings) -> Path:
        """Persist settings to disk with atomic file writes."""

        payload = self._serialize(settings)
        body = json.dumps(payload, indent=2, sort_keys=True)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(body, encoding="utf-8")
        tmp_path.replace(self._path)
        return self._path

    def _serialize(self, settings: Settings) -> Dict[str, Any]:
        data = asdict(settings)
        api_key = data.pop("api_key", "") or ""
        ciphertext = self._encrypt_api_key(api_key)
        if ciphertext:
            data[_API_KEY_FIELD] = ciphertext
        data["version"] = _SETTINGS_VERSION
        data["secret_backend"] = self._vault.strategy
        return data

    def _read_payload(self) -> Dict[str, Any]:
        if not self._path.exists():
            return {}
        try:
            text = self._path.read_text(encoding="utf-8")
            return json.loads(text)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as exc:
            LOGGER.warning("Settings file %s is not valid JSON: %s", self._path, exc)
            return {}

    def _apply_overrides(
        self,
        settings: Settings,
        overrides: Mapping[str, Any],
        *,
        source: str = "runtime",
    ) -> Settings:
        allowed = {field.name for field in fields(Settings)}
        filtered: Dict[str, Any] = {}
        for key, value in overrides.items():
            if key not in allowed or value is None:
                continue
            filtered[key] = value
        if filtered:
            LOGGER.debug(
                "Applying %s settings overrides: %s", source, sorted(filtered)
            )
            settings = replace(settings, **filtered)
        return settings

    def _apply_env_overrides(self, settings: Settings) -> Settings:
        overrides: Dict[str, Any] = {}
        for env_name, field_name in _ENV_OVERRIDES.items():
            value = os.environ.get(env_name)
            if value is not None:
                overrides[field_name] = value
        for env_name, field_name in _BOOL_ENV_OVERRIDES.items():
            value = os.environ.get(env_name)
            if value is not None:
                overrides[field_name] = value.strip().lower() in _TRUE_VALUES
        for env_name, field_name in _FLOAT_ENV_OVERRIDES.items():
            value = os.environ.get(env_name)
            if value is None:
                continue
            try:
                overrides[field_name] = float(value)
            except ValueError:
                LOGGER.warning(
                    "Environment override %s=%s is not a valid float", env_name, value
                )
        if overrides:
            settings = self._apply_overrides(settings, overrides, source="environment")
        return settings

    def _encrypt_api_key(self, api_key: str) -> str | None:
        if not api_key:
            return None
        try:
            token = self._vault.encrypt(api_key)
            LOGGER.debug("API key encrypted via %s backend", self._vault.strategy)
            return token
        except Exception as exc:  # pragma: no cover - extremely rare
            LOGGER.warning("Failed to encrypt API key: %s", exc)
            return None

    def _decrypt_api_key(
        self, ciphertext: str | None, legacy_plaintext: str | None
    ) -> tuple[str, bool]:
        if ciphertext:
            try:
                return self._vault.decrypt(ciphertext), False
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning("Unable to decrypt API key: %s", exc)
                return "", False
        if legacy_plaintext:
            LOGGER.info("Detected legacy plaintext API key; migrating to encrypted storage.")
            return legacy_plaintext, True
        return "", False


class SecretVault:
    """Encrypts and decrypts sensitive strings for settings persistence."""

    def __init__(
        self,
        *,
        key_path: Path | None = None,
        provider: SecretProvider | None = None,
    ) -> None:
        self._key_path = key_path or (_SETTINGS_DIR / "settings.key")
        self._fernet_provider = FernetSecretProvider(self._key_path)
        self._windows_provider = WindowsSecretProvider() if _dpapi_supported() else None
        self._provider = provider or self._auto_detect_provider()

    @property
    def strategy(self) -> str:
        return self._provider.name

    def encrypt(self, secret: str) -> str:
        if not secret:
            return ""
        payload = self._provider.encrypt(secret)
        return f"{self._provider.name}:{payload}"

    def decrypt(self, token: str | None) -> str:
        if not token:
            return ""
        prefix, payload = self._split_token(token)
        provider = self._provider_for_prefix(prefix)
        if provider is None:
            LOGGER.warning("Unknown secret token prefix %s; returning ciphertext.", prefix)
            return token
        try:
            return provider.decrypt(payload)
        except InvalidToken as exc:  # pragma: no cover - indicates tampering
            raise ValueError("Invalid Fernet token") from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ValueError(f"Unable to decrypt secret via {provider.name}") from exc

    def _provider_for_prefix(self, prefix: str | None) -> SecretProvider | None:
        if prefix == WindowsSecretProvider.name and self._windows_provider is not None:
            return self._windows_provider
        if prefix == FernetSecretProvider.name:
            return self._fernet_provider
        if prefix is None:
            return self._provider
        return None

    def _auto_detect_provider(self) -> SecretProvider:
        forced = os.environ.get(_SECRET_BACKEND_ENV, "").strip().lower()
        if forced == WindowsSecretProvider.name and self._windows_provider is None:
            LOGGER.warning(
                "DPAPI secret backend requested but unavailable; falling back to Fernet."
            )
            forced = FernetSecretProvider.name
        if forced == FernetSecretProvider.name:
            return self._fernet_provider
        if forced == WindowsSecretProvider.name and self._windows_provider is not None:
            return self._windows_provider
        if self._windows_provider is not None:
            return self._windows_provider
        return self._fernet_provider

    @staticmethod
    def _split_token(token: str) -> tuple[str | None, str]:
        if ":" not in token:
            return None, token
        prefix, payload = token.split(":", 1)
        return (prefix or None), payload


def _filter_fields(payload: Mapping[str, Any]) -> Dict[str, Any]:
    allowed = {field.name for field in fields(Settings)} - {"api_key"}
    result: Dict[str, Any] = {}
    for key, value in payload.items():
        if key not in allowed:
            continue
        result[key] = value
    return result


def _dpapi_supported() -> bool:
    return sys.platform.startswith("win") and hasattr(sys, "getwindowsversion")


def _dpapi_protect(data: bytes) -> bytes:
    if not _dpapi_supported():  # pragma: no cover - depends on OS
        raise RuntimeError("DPAPI is only available on Windows")
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]

    def _to_blob(buffer: bytes) -> DATA_BLOB:
        blob = DATA_BLOB()
        blob.cbData = len(buffer)
        c_buffer = ctypes.create_string_buffer(buffer)
        blob.pbData = ctypes.cast(c_buffer, ctypes.POINTER(ctypes.c_char))
        blob._buffer = c_buffer  # type: ignore[attr-defined]
        return blob

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32

    input_blob = _to_blob(data)
    output_blob = DATA_BLOB()
    if not crypt32.CryptProtectData(
        ctypes.byref(input_blob), None, None, None, None, 0, ctypes.byref(output_blob)
    ):
        raise OSError("CryptProtectData failed")
    try:
        pointer = ctypes.cast(output_blob.pbData, ctypes.POINTER(ctypes.c_char))
        return ctypes.string_at(pointer, output_blob.cbData)
    finally:
        kernel32.LocalFree(output_blob.pbData)


def _dpapi_unprotect(data: bytes) -> bytes:
    if not _dpapi_supported():  # pragma: no cover - depends on OS
        raise RuntimeError("DPAPI is only available on Windows")
    import ctypes
    from ctypes import wintypes

    class DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_char)),
        ]

    def _to_blob(buffer: bytes) -> DATA_BLOB:
        blob = DATA_BLOB()
        blob.cbData = len(buffer)
        c_buffer = ctypes.create_string_buffer(buffer)
        blob.pbData = ctypes.cast(c_buffer, ctypes.POINTER(ctypes.c_char))
        blob._buffer = c_buffer  # type: ignore[attr-defined]
        return blob

    crypt32 = ctypes.windll.crypt32
    kernel32 = ctypes.windll.kernel32

    input_blob = _to_blob(data)
    output_blob = DATA_BLOB()
    if not crypt32.CryptUnprotectData(
        ctypes.byref(input_blob), None, None, None, None, 0, ctypes.byref(output_blob)
    ):
        raise OSError("CryptUnprotectData failed")
    try:
        pointer = ctypes.cast(output_blob.pbData, ctypes.POINTER(ctypes.c_char))
        return ctypes.string_at(pointer, output_blob.cbData)
    finally:
        kernel32.LocalFree(output_blob.pbData)

