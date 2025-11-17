"""Compatibility helpers that route editor theming through the new module."""

from __future__ import annotations

from ...theme import Theme, load_theme as _load_theme, theme_manager


DEFAULT_THEME = _load_theme()


def load_theme(theme: Theme | str | None = None) -> Theme:
    """Resolve ``theme`` to a :class:`Theme` instance via the shared registry."""

    return _load_theme(theme)


def available_themes() -> list[str]:
    """List bundled theme names for legacy call sites."""

    return theme_manager.available_names()
