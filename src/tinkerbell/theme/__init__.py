"""Theme module consolidating palette data and registry helpers."""

from .models import ColorTuple, Theme, normalize_color
from .manager import (
    ThemeManager,
    available_themes,
    build_default_dark_theme,
    build_light_theme,
    load_theme,
    theme_manager,
)

__all__ = [
    "ColorTuple",
    "Theme",
    "ThemeManager",
    "available_themes",
    "build_default_dark_theme",
    "build_light_theme",
    "load_theme",
    "normalize_color",
    "theme_manager",
]
