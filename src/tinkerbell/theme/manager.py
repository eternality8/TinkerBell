"""Theme registry, serialization helpers, and Qt integration utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, cast

from .models import ColorTuple, Theme

_DEFAULT_DARK_PALETTE: Dict[str, ColorTuple] = {
    "background": (26, 26, 27),
    "surface": (32, 32, 33),
    "surface_alt": (45, 45, 48),
    "border": (60, 60, 60),
    "foreground": (235, 235, 235),
    "text_muted": (150, 150, 150),
    "accent": (97, 175, 239),
    "accent_alt": (198, 120, 221),
    "selection": (38, 79, 120),
    "selection_foreground": (255, 255, 255),
    "selection_inactive": (55, 55, 60),
    "preview_background": (26, 26, 27),
    "preview_foreground": (235, 235, 235),
    "preview_code_background": (31, 31, 31),
    "preview_code_border": (63, 63, 63),
    "editor_background": (30, 30, 30),
    "editor_foreground": (212, 212, 212),
    "editor_gutter": (45, 45, 48),
    "editor_line_highlight": (42, 45, 46),
    "editor_placeholder": (117, 117, 117),
    "diff_added": (20, 70, 32),
    "diff_removed": (90, 26, 26),
    "diff_highlight": (255, 243, 196),
    "diff_highlight_foreground": (32, 33, 36),
    "status_info": (129, 201, 149),
    "status_warning": (246, 173, 85),
    "status_error": (245, 101, 101),
    "status_muted": (120, 120, 120),
    "link": (108, 199, 255),
}

_DEFAULT_LIGHT_PALETTE: Dict[str, ColorTuple] = {
    "background": (248, 248, 248),
    "surface": (255, 255, 255),
    "surface_alt": (240, 240, 240),
    "border": (202, 202, 204),
    "foreground": (32, 33, 36),
    "text_muted": (90, 92, 97),
    "accent": (45, 123, 246),
    "accent_alt": (163, 21, 81),
    "selection": (181, 215, 255),
    "selection_foreground": (32, 33, 36),
    "selection_inactive": (220, 220, 220),
    "preview_background": (255, 255, 255),
    "preview_foreground": (32, 33, 36),
    "preview_code_background": (247, 247, 247),
    "preview_code_border": (222, 222, 226),
    "editor_background": (255, 255, 255),
    "editor_foreground": (33, 37, 41),
    "editor_gutter": (240, 240, 240),
    "editor_line_highlight": (235, 240, 255),
    "editor_placeholder": (150, 150, 150),
    "diff_added": (212, 237, 218),
    "diff_removed": (248, 215, 218),
    "diff_highlight": (255, 244, 197),
    "diff_highlight_foreground": (33, 37, 41),
    "status_info": (0, 123, 255),
    "status_warning": (255, 193, 7),
    "status_error": (220, 53, 69),
    "status_muted": (120, 124, 130),
    "link": (0, 106, 166),
}


def build_default_dark_theme() -> Theme:
    return Theme(
        name="default",
        title="TinkerBell Dark",
        description="High contrast theme designed for dim environments.",
        palette=_DEFAULT_DARK_PALETTE,
        metadata={"qt_style": "Fusion", "appearance": "dark"},
    )


def build_light_theme() -> Theme:
    return Theme(
        name="daylight",
        title="Daylight",
        description="A bright theme suited for well-lit rooms.",
        palette=_DEFAULT_LIGHT_PALETTE,
        metadata={"qt_style": "Fusion", "appearance": "light"},
    )


class ThemeManager:
    """Registry that resolves and serializes themes across the application."""

    def __init__(self, themes: Iterable[Theme] | None = None, *, default_name: str = "default") -> None:
        self._themes: Dict[str, Theme] = {}
        self._default_name = default_name.lower()
        if themes:
            for theme in themes:
                self.register(theme)
        if not self._themes:
            self.register(build_default_dark_theme())
        if self._default_name not in self._themes:
            self._default_name = next(iter(self._themes))

    def register(self, theme: Theme, *, overwrite: bool = True) -> None:
        key = theme.name.lower()
        if not overwrite and key in self._themes:
            raise ValueError(f"Theme '{theme.name}' already registered")
        self._themes[key] = theme

    def available(self) -> List[Theme]:
        return [self._themes[name] for name in sorted(self._themes.keys())]

    def available_names(self) -> List[str]:
        return [theme.name for theme in self.available()]

    def resolve(self, theme: Theme | str | None = None) -> Theme:
        if isinstance(theme, Theme):
            return theme
        key = (theme or self._default_name).strip().lower()
        return self._themes.get(key) or self._themes[self._default_name]

    def default(self) -> Theme:
        return self._themes[self._default_name]

    def set_default(self, theme_name: str) -> None:
        key = theme_name.strip().lower()
        if key not in self._themes:
            raise KeyError(f"Unknown theme '{theme_name}'")
        self._default_name = key

    def export_theme(self, theme: Theme | str | None, destination: str | Path, *, indent: int = 2) -> Path:
        resolved = self.resolve(theme)
        path = Path(destination)
        payload = resolved.to_dict()
        path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
        return path

    def import_theme(self, source: str | Path, *, activate: bool = False) -> Theme:
        path = Path(source)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("Theme file must contain a JSON object")
        theme = Theme.from_dict(payload)
        self.register(theme)
        if activate:
            self.set_default(theme.name)
        return theme

    def apply_to_application(self, theme: Theme | str | None = None, *, app: Any | None = None) -> Theme:
        resolved = self.resolve(theme)
        try:  # pragma: no cover - Qt optional in CI
            from PySide6.QtGui import QColor, QPalette  # type: ignore
            from PySide6.QtWidgets import QApplication  # type: ignore
        except Exception:  # pragma: no cover - headless fallback
            return resolved

        palette_cls = cast(Any, QPalette)
        qt_app: Any = app if app is not None else QApplication.instance()
        if qt_app is None:
            return resolved

        palette = palette_cls()

        def _set(role: Any, key: str, fallback: ColorTuple) -> None:
            rgb = resolved.color(key, fallback)
            try:
                palette.setColor(role, QColor(*rgb))
            except Exception:  # pragma: no cover - defensive guard
                pass

        _set(palette_cls.Window, "background", (30, 30, 30))
        _set(palette_cls.WindowText, "foreground", (235, 235, 235))
        _set(palette_cls.Base, "surface", (26, 26, 27))
        _set(palette_cls.AlternateBase, "surface_alt", (45, 45, 48))
        _set(palette_cls.Text, "foreground", (235, 235, 235))
        _set(palette_cls.Button, "surface", (26, 26, 27))
        _set(palette_cls.ButtonText, "foreground", (235, 235, 235))
        _set(palette_cls.Highlight, "selection", (38, 79, 120))
        _set(palette_cls.HighlightedText, "selection_foreground", (255, 255, 255))
        _set(palette_cls.Link, "link", (108, 199, 255))

        setter = getattr(qt_app, "setPalette", None)
        if callable(setter):
            try:
                setter(palette)
            except Exception:  # pragma: no cover - defensive guard
                pass

        style_name = resolved.metadata.get("qt_style")
        style_setter = getattr(qt_app, "setStyle", None)
        if style_name and callable(style_setter):
            try:
                style_setter(style_name)
            except Exception:  # pragma: no cover - Qt style may be missing
                pass

        return resolved


_BUILTIN_THEMES = [build_default_dark_theme(), build_light_theme()]

theme_manager = ThemeManager(_BUILTIN_THEMES)


def load_theme(theme: Theme | str | None = None) -> Theme:
    return theme_manager.resolve(theme)


def available_themes() -> List[str]:
    return theme_manager.available_names()


__all__ = [
    "ThemeManager",
    "available_themes",
    "build_default_dark_theme",
    "build_light_theme",
    "load_theme",
    "theme_manager",
]
