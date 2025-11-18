"""Helpers for applying settings, themes, and AI runtime configuration."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from ..ai.ai_types import SubagentRuntimeConfig
from ..services.settings import Settings
from ..theme import load_theme, theme_manager
from ..utils import logging as logging_utils
from .models.window_state import WindowContext

if False:  # pragma: no cover - imported for type checking only
    from ..ai.orchestration import AIController
    from ..editor.tabbed_editor import TabbedEditorWidget
    from ..services.telemetry import TelemetryController
    from .embedding_controller import EmbeddingController


_LOGGER = logging.getLogger(__name__)


class SettingsRuntime:
    """Encapsulates theme, logging, and AI controller updates."""

    def __init__(
        self,
        *,
        context: WindowContext,
        editor: Any,
        telemetry_controller: Any,
        embedding_controller: Any,
        register_default_ai_tools: Callable[[], None],
        outline_tool_provider: Callable[[], Any | None],
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
        self._outline_tool_provider = outline_tool_provider
        self._get_ai_task = ai_task_getter
        self._set_ai_task = ai_task_setter
        self._set_ai_stream_state = ai_stream_state_setter
        self._active_theme: str | None = None
        self._active_theme_request: str | None = None
        self._debug_logging_enabled = bool(getattr(initial_settings, "debug_logging", False)) if initial_settings else False
        self._ai_client_signature: tuple[Any, ...] | None = self._ai_settings_signature(initial_settings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def ai_client_signature(self) -> tuple[Any, ...] | None:
        return self._ai_client_signature

    @ai_client_signature.setter
    def ai_client_signature(self, value: tuple[Any, ...] | None) -> None:
        self._ai_client_signature = value

    def apply_runtime_settings(
        self,
        settings: Settings,
        *,
        chat_panel_handler: Callable[[Settings], None],
        phase3_handler: Callable[[Settings], None],
        plot_scaffolding_handler: Callable[[Settings], None],
    ) -> None:
        chat_panel_handler(settings)
        phase3_handler(settings)
        plot_scaffolding_handler(settings)
        self._apply_debug_logging_setting(settings)
        self.apply_theme_setting(settings)
        self._embedding_controller.refresh_runtime(settings)
        self._refresh_ai_runtime(settings)

    def apply_theme_setting(self, settings: Settings) -> None:
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
        try:
            from ..ai.client import AIClient, ClientSettings  # type: ignore
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

    def build_ai_controller_from_settings(self, settings: Settings):
        client = self.build_ai_client_from_settings(settings)
        if client is None:
            return None
        try:
            from ..ai.orchestration import AIController
        except Exception as exc:  # pragma: no cover - dependency guard
            _LOGGER.warning("AI controller unavailable: %s", exc)
            return None

        try:
            limit = self._resolve_max_tool_iterations(settings)
            policy = self._build_context_budget_policy(settings)
            return AIController(
                client=client,
                max_tool_iterations=limit,
                max_context_tokens=getattr(settings, "max_context_tokens", 128_000),
                response_token_reserve=getattr(settings, "response_token_reserve", 16_000),
                budget_policy=policy,
                subagent_config=self._build_subagent_runtime_config(settings),
            )
        except Exception as exc:
            _LOGGER.warning("Failed to initialize AI controller: %s", exc)
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
        controller = self._context.ai_controller
        if controller is None:
            return
        client = getattr(controller, "client", None)
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
        new_subagent_flag = bool(getattr(settings, "enable_subagents", False))
        telemetry_setter = getattr(self._telemetry_controller, "set_subagent_enabled", None)
        if callable(telemetry_setter):
            telemetry_setter(new_subagent_flag)
        if not self._ai_settings_ready(settings):
            self._disable_ai_controller()
            return

        signature = self._ai_settings_signature(settings)
        controller = self._context.ai_controller
        iteration_limit = self._resolve_max_tool_iterations(settings)

        if controller is None:
            controller = self.build_ai_controller_from_settings(settings)
            if controller is None:
                return
            self._context.ai_controller = controller
            self._ai_client_signature = signature
            self._register_default_ai_tools()
        elif signature != self._ai_client_signature:
            client = self.build_ai_client_from_settings(settings)
            if client is None:
                return
            controller.update_client(client)
            self._ai_client_signature = signature

        if controller is not None:
            self._apply_max_tool_iterations(controller, iteration_limit)
            self._apply_context_window_settings(controller, settings)
            self._apply_context_policy_settings(controller, settings)
            self._apply_subagent_runtime_config(controller, settings)

        self._update_ai_debug_logging(bool(getattr(settings, "debug_logging", False)))

    def _ai_settings_ready(self, settings: Settings) -> bool:
        return bool((settings.api_key or "").strip() and (settings.base_url or "").strip() and (settings.model or "").strip())

    def _ai_settings_signature(self, settings: Settings | None) -> tuple[Any, ...] | None:
        if settings is None:
            return None
        headers = tuple(sorted((settings.default_headers or {}).items()))
        metadata = tuple(sorted((settings.metadata or {}).items()))
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
        )

    def _apply_max_tool_iterations(self, controller: Any, limit: int) -> None:
        setter = getattr(controller, "set_max_tool_iterations", None)
        if not callable(setter):
            return
        try:
            setter(limit)
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Unable to update max tool iterations: %s", exc)

    def _apply_context_window_settings(self, controller: Any, settings: Settings) -> None:
        configurator = getattr(controller, "configure_context_window", None)
        if not callable(configurator):
            return
        try:
            configurator(
                max_context_tokens=getattr(settings, "max_context_tokens", 128_000),
                response_token_reserve=getattr(settings, "response_token_reserve", 16_000),
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Unable to update context window settings: %s", exc)

    def _apply_context_policy_settings(self, controller: Any, settings: Settings) -> None:
        policy = self._build_context_budget_policy(settings)
        configurator = getattr(controller, "configure_budget_policy", None)
        if callable(configurator):
            try:
                configurator(policy)
            except Exception as exc:  # pragma: no cover - defensive guard
                _LOGGER.debug("Unable to update context budget policy: %s", exc)
        outline_tool = self._outline_tool_provider()
        if outline_tool is not None:
            outline_tool.budget_policy = policy

    def _build_context_budget_policy(self, settings: Settings):
        try:
            from ..ai.services.context_policy import ContextBudgetPolicy  # type: ignore
        except Exception as exc:  # pragma: no cover - dependency guard
            _LOGGER.debug("Context budget policy unavailable: %s", exc)
            return None

        policy_settings = getattr(settings, "context_policy", None)
        max_context = getattr(settings, "max_context_tokens", 128_000)
        reserve = getattr(settings, "response_token_reserve", 16_000)
        model_name = getattr(settings, "model", None)
        return ContextBudgetPolicy.from_settings(
            policy_settings,
            model_name=model_name,
            max_context_tokens=max_context,
            response_token_reserve=reserve,
        )

    def _build_subagent_runtime_config(self, settings: Settings) -> SubagentRuntimeConfig:
        enabled = bool(getattr(settings, "enable_subagents", False))
        return SubagentRuntimeConfig(
            enabled=enabled,
            plot_scaffolding_enabled=bool(getattr(settings, "enable_plot_scaffolding", False)),
        )

    def _resolve_max_tool_iterations(self, settings: Settings | None) -> int:
        raw = getattr(settings, "max_tool_iterations", 8) if settings else 8
        try:
            value = int(raw)
        except (TypeError, ValueError):  # pragma: no cover - validated via tests
            value = 8
        return max(1, min(value, 50))

    def _apply_subagent_runtime_config(self, controller: Any, settings: Settings) -> None:
        configurator = getattr(controller, "configure_subagents", None)
        if not callable(configurator):
            return
        config = self._build_subagent_runtime_config(settings)
        try:
            configurator(config)
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Unable to update subagent runtime config: %s", exc)

    def _disable_ai_controller(self) -> None:
        controller = self._context.ai_controller
        if controller is None and self._ai_client_signature is None:
            return
        task = self._get_ai_task()
        if task and not getattr(task, "done", lambda: True)():
            try:
                task.cancel()
            except Exception:  # pragma: no cover - defensive guard
                pass
        self._set_ai_task(None)
        self._set_ai_stream_state(False)
        self._context.ai_controller = None
        self._ai_client_signature = None
        _LOGGER.info("AI controller disabled until settings are completed.")


__all__ = ["SettingsRuntime"]
