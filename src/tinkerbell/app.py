"""Application bootstrap module.

This module is responsible for configuring logging, loading persisted settings,
creating the Qt application instance (with qasync), and launching the main
window. All functions are scaffolds that will be filled in future milestones.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .services.settings import Settings


def configure_logging(debug: bool = False) -> None:
    """Configure structured logging for the application."""

    raise NotImplementedError("Logging configuration not implemented yet.")


def load_settings(path: Optional[Path] = None) -> Settings:
    """Load persisted settings or fall back to defaults."""

    raise NotImplementedError("Settings loader not implemented yet.")


def create_qapp(settings: Settings):
    """Create a qasync-powered QApplication instance.

    Returns the Qt application instance; annotated as Any to avoid a hard
    dependency on PySide6 during early scaffolding.
    """

    raise NotImplementedError("Qt application factory not implemented yet.")


def main() -> None:
    """Entry point invoked by the Poetry script."""

    raise NotImplementedError("Main entry point not implemented yet.")
