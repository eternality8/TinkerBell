"""UI package holding the desktop application's widgets and controllers."""

from .main_window import MainWindow
from .models.actions import MenuSpec, SplitterState, ToolbarSpec, WindowAction
from .models.window_state import OutlineStatusInfo, WindowContext

__all__ = [
    "MainWindow",
    "WindowAction",
    "MenuSpec",
    "ToolbarSpec",
    "SplitterState",
    "WindowContext",
    "OutlineStatusInfo",
]
