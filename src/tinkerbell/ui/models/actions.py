"""UI action and layout data structures used by the main window."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple


@dataclass(slots=True)
class WindowAction:
    """Represents a high-level action exposed through menus/toolbars."""

    name: str
    text: str
    shortcut: Optional[str] = None
    status_tip: Optional[str] = None
    callback: Optional[Callable[[], Any]] = None

    def trigger(self) -> None:
        """Invoke the registered callback, if available."""

        if self.callback is not None:
            self.callback()


@dataclass(slots=True)
class MenuSpec:
    """Declarative menu definition used for headless + Qt builds."""

    name: str
    title: str
    actions: Tuple[str, ...]


@dataclass(slots=True)
class ToolbarSpec:
    """Declarative toolbar definition mirroring the cleanup plan contract."""

    name: str
    actions: Tuple[str, ...]


@dataclass(slots=True)
class SplitterState:
    """Simple structure describing the editor/chat splitter layout."""

    editor: Any
    chat_panel: Any
    orientation: str = "horizontal"
    stretch_factors: Tuple[int, int] = (3, 2)


__all__ = [
    "WindowAction",
    "MenuSpec",
    "ToolbarSpec",
    "SplitterState",
]
