"""Theme definitions applied to editor widgets and previews."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(slots=True)
class Theme:
    """Simple theme descriptor for syntax highlighting."""

    name: str
    palette: Dict[str, Tuple[int, int, int]]


DEFAULT_THEME = Theme(name="default", palette={"background": (30, 30, 30), "foreground": (235, 235, 235)})


def load_theme(name: str = "default") -> Theme:
    """Return the requested theme; defaults to dark palette."""

    return DEFAULT_THEME if name == "default" else DEFAULT_THEME


def available_themes() -> list[str]:
    """List bundled theme names."""

    return ["default"]
