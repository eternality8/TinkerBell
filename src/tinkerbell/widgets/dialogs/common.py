"""Shared constants and utilities for dialog modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from PySide6.QtWidgets import QFileDialog, QWidget

__all__ = [
    "DEFAULT_FILE_FILTER",
    "MODEL_SUGGESTIONS",
    "EMBEDDING_MODE_LABELS",
    "HINT_COLORS",
    "SAMPLE_LANG_MAP",
    "DEFAULT_SAMPLE_LIMIT",
    "PREVIEW_CHAR_LIMIT",
    "ValidationResult",
    "SettingsValidator",
    "SettingsTester",
    "humanize_bytes",
    "language_from_suffix",
    "project_root",
]

DEFAULT_FILE_FILTER = "Markdown / Text (*.md *.markdown *.mdx *.txt *.json *.yaml *.yml);;All Files (*)"
MODEL_SUGGESTIONS = ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "o4-mini")
EMBEDDING_MODE_LABELS: Mapping[str, str] = {
    "disabled": "Disabled",
    "same-api": "Same API as Chat Model",
    "custom-api": "Separate OpenAI-Compatible API",
    "local": "Local (SentenceTransformers)",
}
HINT_COLORS = {
    "info": "#6a737d",
    "success": "#1a7f37",
    "warning": "#b08800",
    "error": "#d73a49",
}
SAMPLE_LANG_MAP = {
    ".md": "markdown",
    ".markdown": "markdown",
    ".mdx": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".txt": "text",
}
DEFAULT_SAMPLE_LIMIT = 12
PREVIEW_CHAR_LIMIT = 3200


@dataclass(slots=True)
class ValidationResult:
    """Outcome of validating a set of settings."""

    ok: bool
    message: str = ""


SettingsValidator = Callable[["Settings"], "ValidationResult | tuple[bool, str] | bool"]
SettingsTester = Callable[["Settings"], "ValidationResult | tuple[bool, str] | bool"]

# Import Settings type for annotations
from ...services.settings import Settings  # noqa: E402


def humanize_bytes(size: int) -> str:
    """Convert byte count to human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    count = float(max(0, size))
    unit_index = 0
    while count >= 1024 and unit_index < len(units) - 1:
        count /= 1024
        unit_index += 1
    if unit_index == 0:
        return f"{int(count)} {units[unit_index]}"
    return f"{count:.1f} {units[unit_index]}"


def language_from_suffix(suffix: str) -> str | None:
    """Map file suffix to language identifier."""
    if not suffix:
        return None
    return SAMPLE_LANG_MAP.get(suffix.lower())


def project_root() -> Path:
    """Find the project root directory by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    parents = list(current.parents)
    for parent in parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return parents[-1] if parents else current.parent


def coerce_validation_result(result: ValidationResult | tuple[bool, str] | bool) -> ValidationResult:
    """Convert various result formats to ValidationResult."""
    if isinstance(result, ValidationResult):
        return result
    if isinstance(result, tuple):
        ok, message = result
        return ValidationResult(bool(ok), str(message))
    return ValidationResult(bool(result), "")
