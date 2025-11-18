"""UI package holding the desktop application's widgets and controllers."""

from .main_window import (
    MainWindow,
    WindowAction,
    WindowContext,
    MenuSpec,
    ToolbarSpec,
    SplitterState,
)

__all__ = [
    "MainWindow",
    "WindowAction",
    "WindowContext",
    "MenuSpec",
    "ToolbarSpec",
    "SplitterState",
]
