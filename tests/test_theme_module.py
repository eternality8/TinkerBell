"""Unit tests for the shared theme module."""

from __future__ import annotations

from pathlib import Path

from tinkerbell.ui.theme import Theme, ThemeManager, build_default_dark_theme


def test_theme_serialization_round_trip() -> None:
    palette = {
        "background": (1, 2, 3),
        "foreground": "#ffffff",
        "accent": "10, 20, 30",
    }
    original = Theme(name="custom", title="Custom", palette=palette, metadata={"qt_style": "Fusion"})

    payload = original.to_dict()
    assert payload["palette"]["background"] == [1, 2, 3]
    assert payload["metadata"]["qt_style"] == "Fusion"

    restored = Theme.from_dict(payload)
    assert restored.name == "custom"
    assert restored.palette["foreground"] == (255, 255, 255)
    assert restored.palette["accent"] == (10, 20, 30)


def test_theme_manager_resolve_and_list() -> None:
    base = Theme(name="default", title="Default", palette={"background": (0, 0, 0)})
    manager = ThemeManager([base])

    sunrise = Theme(name="sunrise", title="Sunrise", palette={"background": (255, 255, 255)})
    manager.register(sunrise)

    resolved = manager.resolve("sunrise")
    assert resolved is sunrise
    assert set(manager.available_names()) >= {"default", "sunrise"}


def test_theme_manager_export_and_import(tmp_path: Path) -> None:
    manager = ThemeManager([Theme(name="default", title="Default", palette={"background": (0, 0, 0)})])
    export_path = tmp_path / "theme.json"

    manager.export_theme("default", export_path)

    other_manager = ThemeManager([Theme(name="fallback", title="Fallback", palette={"background": (10, 10, 10)})])
    imported = other_manager.import_theme(export_path)

    assert imported.name == "default"
    assert other_manager.resolve("default").palette["background"] == (0, 0, 0)


def test_default_dark_theme_exposes_diff_highlight_foreground() -> None:
    theme = build_default_dark_theme()
    assert theme.palette["diff_highlight_foreground"] == (32, 33, 36)
