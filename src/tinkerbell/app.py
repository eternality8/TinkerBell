"""Application bootstrap helpers for the TinkerBell desktop app."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, TextIO, cast, get_args, get_origin, get_type_hints

from .ai.agents.executor import AIController
from .ai.client import AIClient, ClientSettings
from .main_window import MainWindow, WindowContext
from .services.settings import Settings, SettingsStore
from .utils import logging as logging_utils

_TRUE_VALUES = {"1", "true", "yes", "on", "debug"}
_LOGGER = logging.getLogger(__name__)
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}


@dataclass(slots=True)
class QtRuntime:
    """Container returned by :func:`create_qapp`."""

    app: Any
    loop: asyncio.AbstractEventLoop


def configure_logging(debug: bool = False, *, force: bool = False) -> None:
    """Configure structured logging for the application."""

    level = logging.DEBUG if debug else logging.INFO
    logging_utils.setup_logging(level, force=force)
    _LOGGER.debug("Logging configured (level=%s)", logging.getLevelName(level))
    _install_qt_message_handler()


def load_settings(
    path: Optional[Path] = None,
    *,
    store: SettingsStore | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Load persisted settings or fall back to defaults."""

    active_store = store or SettingsStore(path)
    try:
        settings = active_store.load(overrides=overrides)
    except Exception as exc:  # pragma: no cover - defensive path
        store_path = getattr(active_store, "_path", path)
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


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point invoked by the `tinkerbell` console script."""

    args, passthrough = _parse_cli_args(argv)
    _rewrite_sys_argv(passthrough)

    debug = _env_flag("TINKERBELL_DEBUG", default=False)
    configure_logging(debug)

    settings_path = args.settings_path or os.environ.get("TINKERBELL_SETTINGS_PATH")
    resolved_path = Path(settings_path).expanduser() if settings_path else None
    settings_store = SettingsStore(resolved_path)
    try:
        cli_overrides = _coerce_cli_overrides(args.overrides or [])
    except ValueError as exc:
        print(f"Invalid --set override: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    overrides_mapping: Dict[str, Any] | None = cli_overrides or None
    settings = load_settings(resolved_path, store=settings_store, overrides=overrides_mapping)

    if args.dump_settings:
        _dump_settings(settings, settings_store, overrides=cli_overrides)
        return

    if settings.debug_logging and not debug:
        configure_logging(True, force=True)
        debug = True

    _warmup_vector_store()

    ai_controller = _build_ai_controller(settings, debug_logging=debug)

    runtime = create_qapp(settings)
    window = MainWindow(
        WindowContext(settings=settings, ai_controller=ai_controller, settings_store=settings_store)
    )
    window.show()

    loop = runtime.loop
    try:
        loop.run_forever()
    except KeyboardInterrupt:  # pragma: no cover - manual shutdown path
        _LOGGER.info("Shutdown requested by user.")
    finally:
        with contextlib.suppress(RuntimeError):
            loop.run_until_complete(_shutdown_ai_controller(ai_controller))
        _drain_event_loop(loop)
        loop.close()


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUE_VALUES


def _drain_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel outstanding tasks and shutdown async machinery before closing."""

    if loop.is_closed():
        return

    async def _cleanup() -> None:
        current_task = None
        with contextlib.suppress(RuntimeError):
            current_task = asyncio.current_task(loop=loop)

        tasks = [
            task
            for task in asyncio.all_tasks(loop)
            if not task.done() and task is not current_task
        ]
        if tasks:
            _LOGGER.debug("Canceling %s pending asyncio task(s) before shutdown.", len(tasks))
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        shutdown_steps = [
            getattr(loop, "shutdown_asyncgens", None),
            getattr(loop, "shutdown_default_executor", None),
        ]
        for step in shutdown_steps:
            if step is None:
                continue
            with contextlib.suppress(RuntimeError, NotImplementedError):
                await step()

    try:
        loop.run_until_complete(_cleanup())
    except RuntimeError as exc:  # pragma: no cover - defensive guard
        _LOGGER.debug("Unable to drain asyncio loop: %s", exc)


async def _shutdown_ai_controller(controller: AIController | None) -> None:
    """Close the AI controller to release background network resources."""

    if controller is None:
        return

    close = getattr(controller, "aclose", None)
    if close is None:
        return

    try:
        await close()
    except Exception as exc:  # pragma: no cover - defensive logging
        _LOGGER.debug("AI controller shutdown failed: %s", exc)


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


def _build_ai_controller(settings: Settings, *, debug_logging: bool = False) -> AIController | None:
    """Construct the AI controller using the current settings, if possible."""

    try:
        client_settings = ClientSettings(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
            organization=settings.organization,
            request_timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            retry_min_seconds=settings.retry_min_seconds,
            retry_max_seconds=settings.retry_max_seconds,
            default_headers=settings.default_headers,
            metadata=settings.metadata,
            debug_logging=debug_logging or settings.debug_logging,
        )
        client = AIClient(client_settings)
    except Exception as exc:  # pragma: no cover - dependency/config errors
        _LOGGER.warning("AI controller unavailable: %s", exc)
        return None

    limit = _resolve_max_tool_iterations(settings)
    context_tokens = getattr(settings, "max_context_tokens", 128_000)
    response_reserve = getattr(settings, "response_token_reserve", 16_000)
    debug_settings = getattr(settings, "debug", None)
    telemetry_enabled = bool(getattr(debug_settings, "token_logging_enabled", False))
    telemetry_limit = getattr(debug_settings, "token_log_limit", 200)
    try:
        telemetry_limit = int(telemetry_limit)
    except (TypeError, ValueError):
        telemetry_limit = 200
    return AIController(
        client=client,
        max_tool_iterations=limit,
        max_context_tokens=context_tokens,
        response_token_reserve=response_reserve,
        telemetry_enabled=telemetry_enabled,
        telemetry_limit=telemetry_limit,
    )


def _resolve_max_tool_iterations(settings: Settings | None) -> int:
    """Clamp the configured iteration limit into a safe operating range."""

    raw_value = getattr(settings, "max_tool_iterations", 8) if settings else 8
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = 8
    return max(1, min(value, 50))


def _parse_cli_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        prog="tinkerbell",
        add_help=True,
        description="Launch the TinkerBell desktop editor or inspect its configuration.",
    )
    parser.add_argument(
        "--dump-settings",
        action="store_true",
        help="Print the effective settings payload (with secrets redacted) and exit.",
    )
    parser.add_argument(
        "--settings-path",
        metavar="PATH",
        help="Override the default ~/.tinkerbell/settings.json path.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Override persisted settings before launch (repeatable).",
    )
    return parser.parse_known_args(argv)


def _rewrite_sys_argv(passthrough: Sequence[str]) -> None:
    program = sys.argv[0] if sys.argv else "tinkerbell"
    sys.argv = [program, *passthrough]


def _coerce_cli_overrides(items: Sequence[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if not items:
        return overrides

    fields = Settings.__dataclass_fields__  # type: ignore[attr-defined]
    type_hints = get_type_hints(Settings)
    for entry in items:
        if "=" not in entry:
            raise ValueError(f"Override '{entry}' must use KEY=VALUE syntax.")
        key, raw_value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Override is missing a field name.")
        if key not in fields:
            raise ValueError(f"Unknown setting '{key}'.")
        annotation = type_hints.get(key, fields[key].type)
        overrides[key] = _coerce_value(annotation, raw_value.strip())
    return overrides


def _coerce_value(annotation: Any, raw_value: str) -> Any:
    target = _resolve_annotation(annotation)
    normalized = raw_value.strip()

    if target is str or target is Any:
        return normalized
    if target is bool:
        return _parse_bool(normalized)
    if target is int:
        return int(normalized, 10)
    if target is float:
        return float(normalized)
    if target is type(None) or normalized.lower() in {"none", "null"}:
        return None
    if is_dataclass(target):
        try:
            payload = json.loads(normalized or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Dataclass overrides must be valid JSON") from exc
        if isinstance(target, type):
            return target(**payload)
        raise ValueError("Dataclass override target is not instantiable")
    if target is list:
        try:
            return json.loads(normalized or "[]")
        except json.JSONDecodeError as exc:
            raise ValueError("List overrides must be valid JSON arrays") from exc
    if target is dict:
        try:
            return json.loads(normalized or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Dict overrides must be valid JSON objects") from exc
    return normalized


def _resolve_annotation(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin is None:
        return annotation
    if origin in {list, dict}:
        return origin
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if not args:
        return origin
    return args[0]


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(f"Cannot coerce '{value}' to a boolean.")


def _dump_settings(
    settings: Settings,
    store: SettingsStore,
    *,
    overrides: Mapping[str, Any],
    stream: TextIO | None = None,
) -> None:
    destination = stream or sys.stdout
    payload = asdict(settings)
    api_key = payload.get("api_key", "")
    if isinstance(api_key, str):
        payload["api_key"] = _redact_secret(api_key)
    metadata = {
        "path": str(store.path),
        "secret_backend": getattr(store.vault, "strategy", "unknown"),
        "cli_overrides": sorted(overrides.keys()),
        "environment_variables": _active_env_overrides(),
    }
    output = {"settings": payload, "meta": metadata}
    json.dump(output, destination, indent=2)
    destination.write("\n")


def _redact_secret(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    if len(stripped) <= 4:
        return "*" * len(stripped)
    return f"{stripped[:2]}{'*' * (len(stripped) - 4)}{stripped[-2:]}"


def _active_env_overrides() -> list[str]:
    return sorted(name for name in os.environ if name.startswith("TINKERBELL_"))
