"""Structured logging helpers for the TinkerBell application."""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

__all__ = ["setup_logging", "get_logger", "get_log_path"]

_DEFAULT_LOG_DIR = Path.home() / ".tinkerbell" / "logs"
_NOISY_LOGGERS: tuple[str, ...] = ("asyncio", "qasync", "httpx", "langchain", "openai")
_CONFIGURED = False
_LOG_PATH: Path | None = None


def setup_logging(
    level: int = logging.INFO,
    *,
    log_dir: Path | str | None = None,
    console: bool = True,
    max_bytes: int = 1_000_000,
    backup_count: int = 3,
    force: bool = False,
) -> Path:
    """Configure root logging with rotating file + optional console handlers."""

    global _CONFIGURED, _LOG_PATH
    if _CONFIGURED and not force and _LOG_PATH is not None:
        return _LOG_PATH

    target_dir = _resolve_log_dir(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = target_dir / "tinkerbell.log"

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)
    logging.captureWarnings(True)
    _tune_external_loggers(level)

    _CONFIGURED = True
    _LOG_PATH = log_path
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Return a module-specific logger."""

    return logging.getLogger(name)


def get_log_path() -> Path | None:
    """Return the currently configured log file if available."""

    return _LOG_PATH


def _resolve_log_dir(log_dir: Path | str | None) -> Path:
    env_override = os.environ.get("TINKERBELL_LOG_DIR")
    return Path(log_dir or env_override or _DEFAULT_LOG_DIR).expanduser()


def _tune_external_loggers(root_level: int) -> None:
    quiet_level = logging.WARNING if root_level < logging.WARNING else root_level
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(quiet_level)


