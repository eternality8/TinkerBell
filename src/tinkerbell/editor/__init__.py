"""Editor package containing document models and widgets."""

from importlib import import_module
from typing import Any

from . import document_model, editor_widget

__all__ = ["document_model", "editor_widget"]


def __getattr__(name: str) -> Any:
	if name == "tabbed_editor":
		module = import_module(f"{__name__}.{name}")
		globals()[name] = module
		return module
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
