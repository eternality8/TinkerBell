"""UI package holding the desktop application's widgets and controllers."""

from .bootstrap import create_application, create_application_headless
from .events import EventBus
from .models.actions import MenuSpec, SplitterState, ToolbarSpec, WindowAction
from .models.window_state import OutlineStatusInfo, WindowContext
from .presentation.main_window import ThinMainWindow

# Backwards compatibility alias
MainWindow = ThinMainWindow

__all__ = [
    # Bootstrap
    "create_application",
    "create_application_headless",
    # Event Bus
    "EventBus",
    # Main Window
    "MainWindow",
    "ThinMainWindow",
    # Actions & Models
    "WindowAction",
    "MenuSpec",
    "ToolbarSpec",
    "SplitterState",
    "WindowContext",
    "OutlineStatusInfo",
]
