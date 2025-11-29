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
from typing import Any, Dict, Mapping, MutableMapping, Literal

from cryptography.fernet import Fernet, InvalidToken

__all__ = [
    "Settings",
    "SettingsStore",
    "SecretVault",
    "DebugSettings",
    "ContextPolicySettings",
    "EMBEDDING_MODE_CHOICES",
    "DEFAULT_EMBEDDING_MODE",
    "redact_secret",
    "redact_metadata",
]

LOGGER = logging.getLogger(__name__)
_SETTINGS_DIR = Path.home() / ".tinkerbell"
_DEFAULT_SETTINGS_PATH = _SETTINGS_DIR / "settings.json"
_SETTINGS_VERSION = 3
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
    "TINKERBELL_DEBUG_EVENT_LOGGING": "debug_event_logging",
    "TINKERBELL_TOOL_ACTIVITY_PANEL": "show_tool_activity_panel",
}
_FLOAT_ENV_OVERRIDES: Mapping[str, str] = {
    "TINKERBELL_REQUEST_TIMEOUT": "request_timeout",
    "TINKERBELL_TEMPERATURE": "temperature",
}
_INT_ENV_OVERRIDES: Mapping[str, str] = {}
_TRUE_VALUES = {"1", "true", "yes", "on", "debug"}
_API_KEY_FIELD = "api_key_ciphertext"
_SECRET_BACKEND_ENV = "TINKERBELL_SECRET_BACKEND"
DEFAULT_EMBEDDING_MODE = "same-api"
EMBEDDING_MODE_CHOICES: tuple[str, ...] = ("disabled", "same-api", "custom-api", "local")
_LOCAL_EMBEDDING_BACKEND = "sentence-transformers"
EmbeddingMode = Literal["disabled", "same-api", "custom-api", "local"]


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
    temperature: float = 0.2
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
    metadata: dict[str, Any] = field(default_factory=dict)
    recent_files: list[str] = field(default_factory=list)
    last_open_file: str | None = None
    unsaved_snapshot: dict[str, Any] | None = None
    unsaved_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    untitled_snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)
    open_tabs: list[dict[str, Any]] | None = None  # None = never saved, [] = explicitly empty
    active_tab_id: str | None = None
    next_untitled_index: int = 1
    font_family: str = "JetBrains Mono"
    font_size: int = 13
    window_geometry: str | None = None
    debug_logging: bool = False
    debug_event_logging: bool = False
    show_tool_activity_panel: bool = False
    chunk_profile: str = "auto"
    chunk_overlap_chars: int = 256
    chunk_max_inline_tokens: int = 1_800
    chunk_iterator_limit: int = 4
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
        tab_payload = payload.get("open_tabs") if payload else None
        # Handle both None (never saved) and null (explicitly saved as null in JSON)
        if tab_payload is None:
            tab_payload = []
        tab_ids_from_disk = [t.get("tab_id", "?") for t in tab_payload if isinstance(t, dict)]
        LOGGER.debug(
            "Settings loaded from %s: %d tabs (ids=%s), active_tab_id=%s",
            self._path,
            len(tab_payload),
            tab_ids_from_disk,
            payload.get("active_tab_id") if payload else None,
        )
        settings = Settings()
        needs_migration = False

        if payload:
            plaintext_key, migrated = self._decrypt_api_key(
                payload.pop(_API_KEY_FIELD, None), payload.pop("api_key", None)
            )
            needs_migration = migrated
            data = _filter_fields(payload)
            metadata_payload = data.get("metadata")
            (
                metadata,
                metadata_migrated,
                embedding_mode_was_missing,
            ) = self._normalize_metadata(metadata_payload)
            data["metadata"] = metadata
            needs_migration = needs_migration or metadata_migrated
            # If embedding_mode was missing from old settings, disable embeddings to preserve old behavior
            if embedding_mode_was_missing and data.get("embedding_backend") != "disabled":
                data["embedding_backend"] = "disabled"
                metadata["embedding_mode"] = "disabled"
                data["metadata"] = metadata
                needs_migration = True
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

            settings, policy_migrated = self._apply_embedding_policy(settings)
            if policy_migrated:
                needs_migration = True

        version_mismatch = bool(payload) and payload.get("version") != _SETTINGS_VERSION
        if needs_migration or version_mismatch:
            try:
                self.save(settings)
            except Exception as exc:  # pragma: no cover - defensive guard
                LOGGER.warning("Failed to migrate settings payload: %s", exc)

        if overrides:
            settings = self._apply_overrides(settings, overrides, source="CLI")

        settings = self._apply_env_overrides(settings)
        settings, _ = self._apply_embedding_policy(settings)
        return settings

    def save(self, settings: Settings) -> Path:
        """Persist settings to disk with atomic file writes."""

        payload = self._serialize(settings)
        body = json.dumps(payload, indent=2, sort_keys=True)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(body, encoding="utf-8")
        tmp_path.replace(self._path)
        open_tabs = settings.open_tabs or []
        tab_ids = [t.get("tab_id", "?") for t in open_tabs]
        LOGGER.debug(
            "Settings saved to %s: %d tabs (ids=%s), active_tab_id=%s",
            self._path,
            len(open_tabs),
            tab_ids,
            settings.active_tab_id,
        )
        return self._path

    def _serialize(self, settings: Settings) -> Dict[str, Any]:
        data = asdict(settings)
        api_key = data.pop("api_key", "") or ""
        ciphertext = self._encrypt_api_key(api_key)
        if ciphertext:
            data[_API_KEY_FIELD] = ciphertext
        data["metadata"] = self._prepare_metadata_for_storage(data.get("metadata"))
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
        metadata_override = filtered.get("metadata")
        if isinstance(metadata_override, Mapping):
            merged_metadata = dict(settings.metadata or {})
            merged_metadata.update(metadata_override)
            filtered["metadata"] = merged_metadata
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
        for env_name, field_name in _INT_ENV_OVERRIDES.items():
            value = os.environ.get(env_name)
            if value is None:
                continue
            try:
                overrides[field_name] = int(value, 10)
            except ValueError:
                LOGGER.warning(
                    "Environment override %s=%s is not a valid integer",
                    env_name,
                    value,
                )
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

    def _apply_embedding_policy(self, settings: Settings) -> tuple[Settings, bool]:
        """Derive embedding_backend from embedding_mode for backwards compatibility."""
        original_metadata = getattr(settings, "metadata", {}) or {}
        metadata = dict(original_metadata)
        raw_mode = metadata.get("embedding_mode")
        mode, mode_changed, _ = _normalize_embedding_mode_value(raw_mode)
        if mode_changed or raw_mode != mode:
            metadata["embedding_mode"] = mode
        # Derive backend from mode:
        # - disabled -> disabled
        # - local -> sentence-transformers
        # - same-api / custom-api -> langchain (flexible for any OpenAI-compatible API)
        if mode == "disabled":
            desired_backend = "disabled"
        elif mode == "local":
            desired_backend = _LOCAL_EMBEDDING_BACKEND
        else:
            desired_backend = "langchain"
        current_backend = getattr(settings, "embedding_backend", "auto")
        backend_changed = current_backend != desired_backend
        mutated = backend_changed or metadata != original_metadata
        if not mutated:
            return settings, False
        updates: Dict[str, Any] = {}
        if metadata != original_metadata:
            updates["metadata"] = metadata
        if backend_changed:
            updates["embedding_backend"] = desired_backend
        return replace(settings, **updates), True

    def _normalize_metadata(self, payload: Any) -> tuple[dict[str, Any], bool, bool]:
        if isinstance(payload, Mapping):
            metadata = dict(payload)
        else:
            metadata = {}
            if payload not in (None, ""):
                LOGGER.debug("Ignoring non-mapping metadata payload of type %s", type(payload))
        raw_mode = metadata.get("embedding_mode")
        mode, mode_changed, mode_missing = _normalize_embedding_mode_value(raw_mode)
        metadata["embedding_mode"] = mode
        migrated = mode_changed
        embedding_api_payload = metadata.get("embedding_api")
        if isinstance(embedding_api_payload, Mapping):
            normalized_api, api_migrated = self._normalize_embedding_api_metadata(embedding_api_payload)
            metadata["embedding_api"] = normalized_api
            migrated = migrated or api_migrated

        dtype_value = metadata.get("st_dtype")
        if dtype_value is not None:
            normalized_dtype = str(dtype_value).strip()
            if normalized_dtype.lower() == "default":
                normalized_dtype = ""
            if normalized_dtype != dtype_value:
                metadata["st_dtype"] = normalized_dtype
                migrated = True
        return metadata, migrated, mode_missing

    def _normalize_embedding_api_metadata(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        metadata = dict(payload)
        migrated = False
        plaintext = metadata.get("api_key") or ""
        ciphertext = metadata.get("api_key_ciphertext") or ""
        if plaintext and not ciphertext:
            token = self._encrypt_secret_value(plaintext, field_name="embedding API key")
            if token:
                metadata["api_key_ciphertext"] = token
            hint = metadata.get("api_key_hint")
            if not hint:
                metadata["api_key_hint"] = redact_secret(plaintext)
            migrated = True
        elif not plaintext:
            metadata.setdefault("api_key", "")
        if metadata.get("api_key_ciphertext"):
            if not metadata.get("api_key"):
                try:
                    metadata["api_key"] = self._vault.decrypt(metadata["api_key_ciphertext"])
                except Exception as exc:  # pragma: no cover - defensive guard
                    LOGGER.warning("Unable to decrypt embedding API key: %s", exc)
                    metadata["api_key"] = ""
                    metadata["api_key_ciphertext"] = ""
                    metadata.setdefault("api_key_hint", "")
                    migrated = True
        plaintext = metadata.get("api_key") or ""
        hint = metadata.get("api_key_hint")
        if plaintext and not hint:
            metadata["api_key_hint"] = redact_secret(plaintext)
            migrated = True
        return metadata, migrated

    def _prepare_metadata_for_storage(self, metadata: Any) -> dict[str, Any]:
        if not isinstance(metadata, Mapping):
            return {"embedding_mode": DEFAULT_EMBEDDING_MODE}
        prepared: dict[str, Any] = {}
        for key, value in metadata.items():
            if key == "embedding_api" and isinstance(value, Mapping):
                serialized_api = self._serialize_embedding_api_metadata(value)
                if serialized_api:
                    prepared[key] = serialized_api
            else:
                prepared[key] = value
        prepared.setdefault("embedding_mode", DEFAULT_EMBEDDING_MODE)
        return prepared

    def _serialize_embedding_api_metadata(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(metadata)
        plaintext = payload.pop("api_key", "") or ""
        ciphertext = payload.get("api_key_ciphertext") or ""
        hint = payload.get("api_key_hint") or ""
        if plaintext:
            token = self._encrypt_secret_value(plaintext, field_name="embedding API key")
            if token:
                payload["api_key_ciphertext"] = token
                ciphertext = token
            if not hint:
                hint = redact_secret(plaintext)
        if ciphertext:
            payload["api_key_ciphertext"] = ciphertext
        if hint:
            payload["api_key_hint"] = hint
        elif ciphertext:
            payload.setdefault("api_key_hint", "")
        return payload

    def _encrypt_secret_value(self, secret: str, *, field_name: str) -> str | None:
        if not secret:
            return None
        try:
            token = self._vault.encrypt(secret)
            LOGGER.debug("%s encrypted via %s backend", field_name, self._vault.strategy)
            return token
        except Exception as exc:  # pragma: no cover - extremely rare
            LOGGER.warning("Failed to encrypt %s: %s", field_name, exc)
            return None

    def _encrypt_api_key(self, api_key: str) -> str | None:
        return self._encrypt_secret_value(api_key, field_name="API key")

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


def _normalize_embedding_mode_value(value: Any) -> tuple[EmbeddingMode, bool, bool]:
    if value is None:
        return DEFAULT_EMBEDDING_MODE, True, True
    normalized = str(value).strip().lower()
    if not normalized:
        return DEFAULT_EMBEDDING_MODE, True, True
    if normalized in EMBEDDING_MODE_CHOICES:
        return normalized, False, False
    LOGGER.warning(
        "Unknown embedding_mode '%s'; defaulting to %s.",
        value,
        DEFAULT_EMBEDDING_MODE,
    )
    return DEFAULT_EMBEDDING_MODE, True, False


def redact_secret(value: str) -> str:
    stripped = (value or "").strip()
    if not stripped:
        return ""
    if len(stripped) <= 4:
        return "*" * len(stripped)
    return f"{stripped[:2]}{'*' * (len(stripped) - 4)}{stripped[-2:]}"


def redact_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    redacted: dict[str, Any] = {}
    for key, value in metadata.items():
        if key == "embedding_api" and isinstance(value, Mapping):
            api_meta = dict(value)
            raw_key = api_meta.get("api_key")
            if isinstance(raw_key, str):
                api_meta["api_key"] = redact_secret(raw_key)
            redacted[key] = api_meta
        else:
            redacted[key] = value
    return redacted


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

