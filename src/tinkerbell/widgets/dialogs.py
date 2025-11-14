"""Dialogs for file operations and settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SettingsDialogResult:
    """Outcome returned when the settings dialog closes."""

    accepted: bool
    api_key: str


def show_settings_dialog() -> SettingsDialogResult:
    """Display the settings dialog (stub)."""

    return SettingsDialogResult(accepted=False, api_key="")

