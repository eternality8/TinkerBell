"""Settings dataclasses and persistence stubs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    """User-configurable settings persisted between sessions."""

    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    theme: str = "default"


class SettingsStore:
    """Persistence adapter for settings data."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path.home() / ".tinkerbell-settings.json"

    def load(self) -> Settings:
        """Load settings from disk (stub)."""

        return Settings()

    def save(self, settings: Settings) -> None:
        """Persist settings to disk (stub)."""

        del settings
        # TODO: write JSON file

