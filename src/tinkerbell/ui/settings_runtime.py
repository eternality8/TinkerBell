"""Helpers for applying settings, themes, and AI runtime configuration."""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Mapping

from ..services.settings import Settings
from ..theme import load_theme, theme_manager
from ..utils import logging as logging_utils
from .models.window_state import WindowContext

if False:  # pragma: no cover - imported for type checking only
    from ..ai.orchestration import AIOrchestrator
    from ..editor.tabbed_editor import TabbedEditorWidget
    from ..services.telemetry import TelemetryController
    from .embedding_controller import EmbeddingController


_LOGGER = logging.getLogger(__name__)


class SettingsRuntime:
    """Encapsulates theme, logging, and AI orchestrator updates."""

    def __init__(
        self,
        *,
        context: WindowContext,
        editor: Any,
        telemetry_controller: Any,
        embedding_controller: Any,
        register_default_ai_tools: Callable[[], None],
        ai_task_getter: Callable[[], Any | None],
        ai_task_setter: Callable[[Any | None], None],
        ai_stream_state_setter: Callable[[bool], None],
        initial_settings: Settings | None,
    ) -> None:
        self._context = context
        self._editor = editor
        self._telemetry_controller = telemetry_controller
        self._embedding_controller = embedding_controller
        self._register_default_ai_tools = register_default_ai_tools
        self._get_ai_task = ai_task_getter
        self._set_ai_task = ai_task_setter
        self._set_ai_stream_state = ai_stream_state_setter
        self._active_theme: str | None = None
        self._active_theme_request: str | None = None
        self._debug_logging_enabled = bool(getattr(initial_settings, "debug_logging", False)) if initial_settings else False
        self._ai_settings_signature: tuple[Any, ...] | None = self._compute_ai_settings_signature(initial_settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def ai_client_signature(self) -> tuple[Any, ...] | None:
        """Legacy alias for ai_settings_signature."""
        return self._ai_settings_signature

    @ai_client_signature.setter
    def ai_client_signature(self, value: tuple[Any, ...] | None) -> None:
        self._ai_settings_signature = value

    def apply_runtime_settings(
        self,
        settings: Settings,
        *,
        chat_panel_handler: Callable[[Settings], None],
    ) -> None:
        """Apply runtime settings changes."""
        chat_panel_handler(settings)
        self._apply_debug_logging_setting(settings)
        self.apply_theme_setting(settings)
        self._embedding_controller.refresh_runtime(settings)
        self._refresh_ai_runtime(settings)

    def apply_theme_setting(self, settings: Settings) -> None:
        """Apply theme setting."""
        requested_name = (getattr(settings, "theme", "") or "default").strip() or "default"
        normalized_request = requested_name.lower()
        if normalized_request == self._active_theme_request:
            return
        self._active_theme_request = normalized_request
        theme = load_theme(requested_name)
        self._active_theme = theme.name
        try:
            apply_theme = getattr(self._editor, "apply_theme", None)
            if callable(apply_theme):
                apply_theme(theme)
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Unable to apply editor theme %s: %s", theme.name, exc)
        theme_manager.apply_to_application(theme)

    def build_ai_client_from_settings(self, settings: Settings):
        """Build an AIClient from settings."""
        try:
            from ..ai.client import AIClient, ClientSettings
        except Exception as exc:  # pragma: no cover - dependency guard
            _LOGGER.warning("AI client components unavailable: %s", exc)
            return None

        client_settings = ClientSettings(
            base_url=settings.base_url,
            api_key=settings.api_key,
            model=settings.model,
            organization=settings.organization,
            request_timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            retry_min_seconds=settings.retry_min_seconds,
            retry_max_seconds=settings.retry_max_seconds,
            default_headers=settings.default_headers or None,
            metadata=settings.metadata or None,
            debug_logging=bool(getattr(settings, "debug_logging", False)),
        )
        try:
            return AIClient(client_settings)
        except Exception as exc:  # pragma: no cover - dependency guard
            _LOGGER.warning("Failed to build AI client: %s", exc)
            return None

    def build_ai_orchestrator_from_settings(self, settings: Settings):
        """Build an AIOrchestrator from settings."""
        client = self.build_ai_client_from_settings(settings)
        if client is None:
            return None
        try:
            from ..ai.orchestration import AIOrchestrator, OrchestratorConfig
        except Exception as exc:  # pragma: no cover - dependency guard
            _LOGGER.warning("AI orchestrator unavailable: %s", exc)
            return None

        try:
            config = OrchestratorConfig(
                max_iterations=self._resolve_max_tool_iterations(settings),
                max_context_tokens=getattr(settings, "max_context_tokens", 128_000),
                response_token_reserve=getattr(settings, "response_token_reserve", 16_000),
                temperature=getattr(settings, "temperature", 0.2),
                streaming_enabled=True,
            )
            return AIOrchestrator(client=client, config=config)
        except Exception as exc:
            _LOGGER.warning("Failed to initialize AI orchestrator: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _apply_debug_logging_setting(self, settings: Settings) -> None:
        new_debug = bool(getattr(settings, "debug_logging", False))
        if new_debug != self._debug_logging_enabled:
            self._update_logging_configuration(new_debug)
            self._debug_logging_enabled = new_debug
        self._update_ai_debug_logging(new_debug)

    def _update_logging_configuration(self, debug_enabled: bool) -> None:
        level = logging.DEBUG if debug_enabled else logging.INFO
        try:
            logging_utils.setup_logging(level, force=True)
            _LOGGER.debug("Runtime logging level updated to %s", logging.getLevelName(level))
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.warning("Unable to update logging configuration: %s", exc)

    def _update_ai_debug_logging(self, debug_enabled: bool) -> None:
        orchestrator = self._context.ai_orchestrator
        if orchestrator is None:
            return
        client = getattr(orchestrator, "client", None)
        if client is None:
            return
        client_settings = getattr(client, "settings", None)
        if client_settings is None:
            return
        try:
            client_settings.debug_logging = debug_enabled
            _LOGGER.debug("AI client debug logging set to %s", debug_enabled)
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Unable to update AI client debug flag: %s", exc)

    def _refresh_ai_runtime(self, settings: Settings) -> None:
        """Refresh AI orchestrator when settings change."""
        if not self._ai_settings_ready(settings):
            self._disable_ai_orchestrator()
            return

        signature = self._compute_ai_settings_signature(settings)
        orchestrator = self._context.ai_orchestrator

        if orchestrator is None:
            # No orchestrator - create new one
            orchestrator = self.build_ai_orchestrator_from_settings(settings)
            if orchestrator is None:
                return
            self._context.ai_orchestrator = orchestrator
            self._ai_settings_signature = signature
            self._register_default_ai_tools()
        elif signature != self._ai_settings_signature:
            # Settings changed - update client or rebuild
            client = self.build_ai_client_from_settings(settings)
            if client is None:
                return
            orchestrator.update_client(client)
            # Update config if needed
            from ..ai.orchestration import OrchestratorConfig
            new_config = OrchestratorConfig(
                max_iterations=self._resolve_max_tool_iterations(settings),
                max_context_tokens=getattr(settings, "max_context_tokens", 128_000),
                response_token_reserve=getattr(settings, "response_token_reserve", 16_000),
                temperature=getattr(settings, "temperature", 0.2),
                streaming_enabled=True,
            )
            orchestrator.set_config(new_config)
            self._ai_settings_signature = signature

        self._update_ai_debug_logging(bool(getattr(settings, "debug_logging", False)))

    def _ai_settings_ready(self, settings: Settings) -> bool:
        return bool(
            (settings.api_key or "").strip()
            and (settings.base_url or "").strip()
            and (settings.model or "").strip()
        )

    def _compute_ai_settings_signature(self, settings: Settings | None) -> tuple[Any, ...] | None:
        if settings is None:
            return None
        headers = tuple(sorted((settings.default_headers or {}).items()))
        metadata = _metadata_signature(settings.metadata)
        return (
            settings.base_url,
            settings.api_key,
            settings.model,
            settings.organization,
            settings.request_timeout,
            settings.max_retries,
            settings.retry_min_seconds,
            settings.retry_max_seconds,
            headers,
            metadata,
            # Include orchestrator-specific settings in signature
            getattr(settings, "max_context_tokens", 128_000),
            getattr(settings, "response_token_reserve", 16_000),
            getattr(settings, "temperature", 0.2),
            getattr(settings, "max_tool_iterations", 8),
        )

    def _resolve_max_tool_iterations(self, settings: Settings | None) -> int:
        raw = getattr(settings, "max_tool_iterations", 8) if settings else 8
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 8
        return max(1, min(value, 200))

    def _disable_ai_orchestrator(self) -> None:
        orchestrator = self._context.ai_orchestrator
        if orchestrator is None and self._ai_settings_signature is None:
            return
        task = self._get_ai_task()
        if task and not getattr(task, "done", lambda: True)():
            try:
                task.cancel()
            except Exception:  # pragma: no cover - defensive guard
                pass
        self._set_ai_task(None)
        self._set_ai_stream_state(False)
        self._context.ai_orchestrator = None
        self._ai_settings_signature = None
        _LOGGER.info("AI orchestrator disabled until settings are completed.")


__all__ = ["SettingsRuntime"]


def _metadata_signature(metadata: Any) -> tuple[tuple[str, str], ...]:
    if not isinstance(metadata, Mapping):
        return ()
    entries: list[tuple[str, str]] = []
    for key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
        entries.append((str(key), _stable_metadata_value(value)))
    return tuple(entries)


def _stable_metadata_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return repr(value)
