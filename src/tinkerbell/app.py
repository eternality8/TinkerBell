"""Application bootstrap helpers for the TinkerBell desktop app."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

from .main_window import MainWindow, WindowContext
from .services.settings import Settings, SettingsStore
from .utils import logging as logging_utils

_TRUE_VALUES = {"1", "true", "yes", "on", "debug"}
_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class QtRuntime:
    """Container returned by :func:`create_qapp`."""

    app: Any
    loop: asyncio.AbstractEventLoop


def configure_logging(debug: bool = False) -> None:
    """Configure structured logging for the application."""

    level = logging.DEBUG if debug else logging.INFO
    logging_utils.setup_logging(level)
    _LOGGER.debug("Logging configured (level=%s)", logging.getLevelName(level))
    _install_qt_message_handler()


def load_settings(path: Optional[Path] = None) -> Settings:
    """Load persisted settings or fall back to defaults."""

    store = SettingsStore(path)
    try:
        settings = store.load()
    except Exception as exc:  # pragma: no cover - defensive path
        store_path = getattr(store, "_path", path)
        _LOGGER.warning("Failed to load settings from %s: %s", store_path, exc)
        settings = Settings()
    return settings


def create_qapp(settings: Settings) -> QtRuntime:
    """Create a qasync-powered QApplication instance."""

    try:  # Local import to avoid mandatory PySide6 dependency at import time.
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - depends on desktop stack
        raise RuntimeError(
            "PySide6 must be installed to launch the TinkerBell UI."
        ) from exc

    try:
        from qasync import QEventLoop
    except ImportError as exc:  # pragma: no cover - depends on env setup
        raise RuntimeError("qasync is required to run the async Qt event loop.") from exc

    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = cast(Any, QApplication.instance() or QApplication(sys.argv))
    app.setApplicationName("TinkerBell")
    app.setApplicationDisplayName("TinkerBell")

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    try:
        app.aboutToQuit.connect(loop.stop)  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - in case of mock QApplication
        pass

    theme = (getattr(settings, "theme", "") or "").lower()
    if theme == "dark":
        try:
            app.setStyle("Fusion")
        except Exception:  # pragma: no cover - style availability varies
            _LOGGER.debug("Fusion style unavailable; continuing with default theme.")

    return QtRuntime(app=app, loop=loop)


def main() -> None:
    """Entry point invoked by the `tinkerbell` console script."""

    debug = _env_flag("TINKERBELL_DEBUG", default=False)
    configure_logging(debug)

    settings_path = os.environ.get("TINKERBELL_SETTINGS_PATH")
    resolved_path = Path(settings_path).expanduser() if settings_path else None
    settings = load_settings(resolved_path)

    _warmup_vector_store()

    runtime = create_qapp(settings)
    window = MainWindow(WindowContext(settings=settings, ai_controller=None))
    window.show()

    loop = runtime.loop
    try:
        loop.run_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown path
        _LOGGER.info("Shutdown requested by user.")
    finally:
        loop.close()


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _install_qt_message_handler() -> None:
    """Redirect Qt warnings to the Python logging stack when available."""

    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:  # pragma: no cover - PySide6 optional during tests
        return

    level_map = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def _handler(mode, context, message):  # type: ignore[no-untyped-def]
        del context
        level = level_map.get(mode, logging.INFO)
        logging.getLogger("PySide6").log(level, message)

    qInstallMessageHandler(_handler)


def _warmup_vector_store() -> None:
    """Eagerly import FAISS so the first agent request has lower latency."""

    try:
        import faiss  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency during tests
        _LOGGER.debug("faiss not available; skipping warmup.")
        return
    except Exception as exc:  # pragma: no cover - defensive logging
        _LOGGER.warning("Unexpected FAISS import error: %s", exc)
        return

    try:
        version = getattr(faiss, "__version__", "unknown")
        _LOGGER.debug("FAISS loaded (version=%s).", version)
    except Exception:  # pragma: no cover - extremely defensive
        _LOGGER.debug("FAISS import succeeded.")
