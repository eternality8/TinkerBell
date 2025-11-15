"""Settings dataclasses and persistence helpers."""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping

from cryptography.fernet import Fernet, InvalidToken

__all__ = ["Settings", "SettingsStore", "SecretVault"]

LOGGER = logging.getLogger(__name__)
_SETTINGS_DIR = Path.home() / ".tinkerbell"
_DEFAULT_SETTINGS_PATH = _SETTINGS_DIR / "settings.json"
_ENV_OVERRIDES: Mapping[str, str] = {
    "TINKERBELL_API_KEY": "api_key",
    "TINKERBELL_BASE_URL": "base_url",
    "TINKERBELL_MODEL": "model",
    "TINKERBELL_THEME": "theme",
    "TINKERBELL_ORGANIZATION": "organization",
}
_BOOL_ENV_OVERRIDES: Mapping[str, str] = {
    "TINKERBELL_DEBUG_LOGGING": "debug_logging",
}
_TRUE_VALUES = {"1", "true", "yes", "on", "debug"}
_API_KEY_FIELD = "api_key_ciphertext"


@dataclass(slots=True)
class Settings:
    """User-configurable settings persisted between sessions."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    theme: str = "default"
    organization: str | None = None
    request_timeout: float = 30.0
    max_retries: int = 3
    retry_min_seconds: float = 0.5
    retry_max_seconds: float = 6.0
    autosave_interval: float = 60.0
    default_headers: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
    recent_files: list[str] = field(default_factory=list)
    last_open_file: str | None = None
    unsaved_snapshot: dict[str, Any] | None = None
    unsaved_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    font_family: str = "JetBrains Mono"
    font_size: int = 13
    window_geometry: str | None = None
    debug_logging: bool = False


class SettingsStore:
    """Persistence adapter for :class:`Settings`."""

    def __init__(self, path: Path | None = None, *, vault: SecretVault | None = None) -> None:
        self._path = path or _DEFAULT_SETTINGS_PATH
        key_path = self._path.with_suffix(".key")
        self._vault = vault or SecretVault(key_path=key_path)

    def load(self) -> Settings:
        """Load settings from disk, applying environment overrides when present."""

        payload = self._read_payload()
        settings = Settings()

        if payload:
            plaintext_key = self._decrypt_api_key(
                payload.pop(_API_KEY_FIELD, None), payload.pop("api_key", None)
            )
            data = _filter_fields(payload)
            try:
                settings = Settings(**data)
            except TypeError as exc:
                LOGGER.warning("Settings payload contained unexpected data: %s", exc)
                settings = Settings()
            if plaintext_key:
                settings = replace(settings, api_key=plaintext_key)

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
        data["version"] = 1
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
        if overrides:
            LOGGER.debug("Applying settings overrides from environment: %s", sorted(overrides))
            settings = replace(settings, **overrides)
        return settings

    def _encrypt_api_key(self, api_key: str) -> str | None:
        if not api_key:
            return None
        try:
            return self._vault.encrypt(api_key)
        except Exception as exc:  # pragma: no cover - extremely rare
            LOGGER.warning("Failed to encrypt API key: %s", exc)
            return None

    def _decrypt_api_key(self, ciphertext: str | None, legacy_plaintext: str | None) -> str:
        if ciphertext:
            try:
                return self._vault.decrypt(ciphertext)
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning("Unable to decrypt API key: %s", exc)
                return ""
        return legacy_plaintext or ""


class SecretVault:
    """Encrypts and decrypts sensitive strings for settings persistence."""

    def __init__(self, *, key_path: Path | None = None) -> None:
        self._key_path = key_path or (_SETTINGS_DIR / "settings.key")
        self._fernet: Fernet | None = None

    def encrypt(self, secret: str) -> str:
        if not secret:
            return ""
        if _dpapi_supported():
            protected = _dpapi_protect(secret.encode("utf-8"))
            encoded = base64.urlsafe_b64encode(protected).decode("ascii")
            return f"dpapi:{encoded}"
        token = self._get_fernet().encrypt(secret.encode("utf-8")).decode("ascii")
        return f"fernet:{token}"

    def decrypt(self, token: str | None) -> str:
        if not token:
            return ""
        if token.startswith("dpapi:") and _dpapi_supported():
            payload = token.split(":", 1)[1].encode("ascii")
            raw = base64.urlsafe_b64decode(payload)
            return _dpapi_unprotect(raw).decode("utf-8")
        if token.startswith("fernet:"):
            payload = token.split(":", 1)[1].encode("ascii")
            try:
                return self._get_fernet().decrypt(payload).decode("utf-8")
            except InvalidToken as exc:  # pragma: no cover - indicates tampering
                raise ValueError("Invalid Fernet token") from exc
        return token

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
        if os.name != "nt":
            os.chmod(tmp_path, 0o600)
        tmp_path.replace(path)
        return key


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

