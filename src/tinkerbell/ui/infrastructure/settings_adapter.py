"""Settings adapter for applying runtime configuration and emitting events.

This module extracts settings application logic from SettingsRuntime into
a clean adapter that emits events through the EventBus.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any, Callable, Mapping, TYPE_CHECKING

from ..events import EventBus, SettingsChanged
from ..models.window_state import WindowContext

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ...ai.client import AIClient
    from ...ai.orchestration import AIOrchestrator
    from ...services.settings import Settings
    from ...editor.tabbed_editor import TabbedEditorWidget

_LOGGER = logging.getLogger(__name__)


class SettingsAdapter:
    """Adapter for applying settings changes and managing runtime configuration.

    This adapter encapsulates theme, logging, and AI orchestrator updates,
    emitting SettingsChanged events when settings are applied. It follows
    the infrastructure layer pattern of wrapping external systems while
    providing event-based communication.

    Attributes:
        _context: The shared window context containing orchestrator reference.
        _event_bus: The event bus for publishing SettingsChanged events.
        _editor: The tabbed editor widget for theme application.
        _active_theme: The name of the currently active theme.
        _active_theme_request: The normalized theme request string.
        _debug_logging_enabled: Whether debug logging is currently enabled.
        _ai_settings_signature: Signature for detecting AI settings changes.
    """

    __slots__ = (
        "_context",
        "_event_bus",
        "_editor",
        "_active_theme",
        "_active_theme_request",
        "_debug_logging_enabled",
        "_ai_settings_signature",
    )

    def __init__(
        self,
        context: WindowContext,
        event_bus: EventBus,
        *,
        editor: Any | None = None,
        initial_settings: Settings | None = None,
    ) -> None:
        """Initialize the settings adapter.

        Args:
            context: The shared window context.
            event_bus: The event bus for publishing events.
            editor: Optional editor widget for theme application.
            initial_settings: Optional initial settings for state initialization.
        """
        self._context = context
        self._event_bus = event_bus
        self._editor = editor
        self._active_theme: str | None = None
        self._active_theme_request: str | None = None
        self._debug_logging_enabled = (
            bool(getattr(initial_settings, "debug_logging", False))
            if initial_settings
            else False
        )
        self._ai_settings_signature: tuple[Any, ...] | None = (
            self._compute_ai_settings_signature(initial_settings)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def ai_settings_signature(self) -> tuple[Any, ...] | None:
        """Return the current AI settings signature for change detection."""
        return self._ai_settings_signature

    def set_editor(self, editor: Any) -> None:
        """Set the editor widget for theme application.

        Args:
            editor: The tabbed editor widget.
        """
        self._editor = editor

    def apply_settings(
        self,
        settings: Settings,
        *,
        chat_panel_handler: Callable[[Settings], None] | None = None,
        embedding_controller: Any | None = None,
        register_ai_tools: Callable[[], None] | None = None,
    ) -> None:
        """Apply runtime settings changes and emit SettingsChanged event.

        This method applies all settings changes including theme, logging,
        embeddings, and AI configuration, then emits a SettingsChanged event.

        Args:
            settings: The settings to apply.
            chat_panel_handler: Optional callback to update chat panel settings.
            embedding_controller: Optional embedding controller for refresh.
            register_ai_tools: Optional callback to register AI tools.
        """
        if chat_panel_handler:
            chat_panel_handler(settings)

        self.apply_debug_logging(settings)
        self.apply_theme(settings)

        if embedding_controller:
            try:
                embedding_controller.refresh_runtime(settings)
            except Exception as exc:  # pragma: no cover - defensive guard
                _LOGGER.debug("Unable to refresh embedding runtime: %s", exc)

        self._refresh_ai_runtime(settings, register_ai_tools)

        # Emit settings changed event
        self._emit_settings_changed(settings)

    def apply_theme(self, settings: Settings) -> None:
        """Apply theme setting to the editor and application.

        Args:
            settings: The settings containing the theme configuration.
        """
        try:
            from tinkerbell.ui.theme import load_theme, theme_manager
        except ImportError:  # pragma: no cover - dependency guard
            _LOGGER.debug("Theme module not available")
            return

        requested_name = (getattr(settings, "theme", "") or "default").strip() or "default"
        normalized_request = requested_name.lower()

        if normalized_request == self._active_theme_request:
            return

        self._active_theme_request = normalized_request
        theme = load_theme(requested_name)
        self._active_theme = theme.name

        # Apply to editor if available
        if self._editor is not None:
            try:
                apply_theme = getattr(self._editor, "apply_theme", None)
                if callable(apply_theme):
                    apply_theme(theme)
            except Exception as exc:  # pragma: no cover - defensive guard
                _LOGGER.debug("Unable to apply editor theme %s: %s", theme.name, exc)

        # Apply to application
        theme_manager.apply_to_application(theme)

    def apply_debug_logging(self, settings: Settings) -> None:
        """Apply debug logging setting.

        Args:
            settings: The settings containing the debug logging configuration.
        """
        new_debug = bool(getattr(settings, "debug_logging", False))

        if new_debug != self._debug_logging_enabled:
            self._update_logging_configuration(new_debug)
            self._debug_logging_enabled = new_debug

        self._update_ai_debug_logging(new_debug)

    def build_ai_client(self, settings: Settings) -> AIClient | None:
        """Build an AIClient from settings.

        Args:
            settings: The settings containing AI client configuration.

        Returns:
            A configured AIClient instance, or None if creation failed.
        """
        try:
            from ...ai.client import AIClient, ClientSettings
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
            debug_logging=bool(getattr(settings, "debug_logging", False)),
        )

        try:
            return AIClient(client_settings)
        except Exception as exc:  # pragma: no cover - dependency guard
            _LOGGER.warning("Failed to build AI client: %s", exc)
            return None

    def build_ai_orchestrator(self, settings: Settings) -> AIOrchestrator | None:
        """Build an AIOrchestrator from settings.

        Args:
            settings: The settings containing AI orchestrator configuration.

        Returns:
            A configured AIOrchestrator instance, or None if creation failed.
        """
        client = self.build_ai_client(settings)
        if client is None:
            return None

        try:
            from ...ai.orchestration import AIOrchestrator, OrchestratorConfig
        except Exception as exc:  # pragma: no cover - dependency guard
            _LOGGER.warning("AI orchestrator unavailable: %s", exc)
            return None

        # Build analysis provider
        analysis_provider = self._build_analysis_provider()

        try:
            config = OrchestratorConfig(
                max_iterations=self._resolve_max_tool_iterations(settings),
                max_context_tokens=getattr(settings, "max_context_tokens", 128_000),
                response_token_reserve=getattr(settings, "response_token_reserve", 16_000),
                temperature=getattr(settings, "temperature", 0.2),
                streaming_enabled=True,
                tool_timeout=getattr(settings, "tool_timeout", 120.0),
            )
            return AIOrchestrator(
                client=client,
                config=config,
                analysis_provider=analysis_provider,
            )
        except Exception as exc:
            _LOGGER.warning("Failed to initialize AI orchestrator: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit_settings_changed(self, settings: Settings) -> None:
        """Emit a SettingsChanged event with current settings.

        Args:
            settings: The settings that were applied.
        """
        try:
            # Convert settings to dict for the event
            settings_dict = asdict(settings)
            # Redact sensitive information
            if "api_key" in settings_dict:
                settings_dict["api_key"] = "***" if settings_dict["api_key"] else ""
            self._event_bus.publish(SettingsChanged(settings=settings_dict))
        except Exception as exc:  # pragma: no cover - defensive guard
            _LOGGER.debug("Unable to emit SettingsChanged event: %s", exc)

    def _build_analysis_provider(self) -> Any | None:
        """Build an analysis provider for preflight analysis."""
        try:
            from ...ai.analysis import AnalysisAgent
            from ...ai.orchestration import AnalysisAgentAdapter
        except ImportError:
            _LOGGER.debug("Analysis components not available")
            return None

        try:
            agent = AnalysisAgent()
            return AnalysisAgentAdapter(agent)
        except Exception as exc:
            _LOGGER.debug("Failed to create analysis provider: %s", exc)
            return None

    def _update_logging_configuration(self, debug_enabled: bool) -> None:
        """Update the logging configuration.

        Args:
            debug_enabled: Whether debug logging should be enabled.
        """
        try:
            from ...utils import logging as logging_utils
        except ImportError:  # pragma: no cover - dependency guard
            _LOGGER.debug("Logging utils not available")
            return

        level = logging.DEBUG if debug_enabled else logging.INFO
        try:
            logging_utils.setup_logging(level, force=True)
            _LOGGER.debug("Runtime logging level updated to %s", logging.getLevelName(level))
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.warning("Unable to update logging configuration: %s", exc)

    def _update_ai_debug_logging(self, debug_enabled: bool) -> None:
        """Update AI client debug logging flag.

        Args:
            debug_enabled: Whether debug logging should be enabled.
        """
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

    def _refresh_ai_runtime(
        self,
        settings: Settings,
        register_ai_tools: Callable[[], None] | None = None,
    ) -> None:
        """Refresh AI orchestrator when settings change.

        Args:
            settings: The current settings.
            register_ai_tools: Optional callback to register AI tools.
        """
        if not self._ai_settings_ready(settings):
            self._disable_ai_orchestrator()
            return

        signature = self._compute_ai_settings_signature(settings)
        orchestrator = self._context.ai_orchestrator

        if orchestrator is None:
            # No orchestrator - create new one
            orchestrator = self.build_ai_orchestrator(settings)
            if orchestrator is None:
                return
            self._context.ai_orchestrator = orchestrator
            self._ai_settings_signature = signature
            if register_ai_tools:
                register_ai_tools()
        elif signature != self._ai_settings_signature:
            # Settings changed - update client or rebuild
            client = self.build_ai_client(settings)
            if client is None:
                return
            orchestrator.update_client(client)

            # Update config if needed
            try:
                from ...ai.orchestration import OrchestratorConfig

                new_config = OrchestratorConfig(
                    max_iterations=self._resolve_max_tool_iterations(settings),
                    max_context_tokens=getattr(settings, "max_context_tokens", 128_000),
                    response_token_reserve=getattr(settings, "response_token_reserve", 16_000),
                    temperature=getattr(settings, "temperature", 0.2),
                    streaming_enabled=True,
                    tool_timeout=getattr(settings, "tool_timeout", 120.0),
                )
                orchestrator.set_config(new_config)
            except Exception as exc:  # pragma: no cover - defensive guard
                _LOGGER.debug("Unable to update orchestrator config: %s", exc)

            self._ai_settings_signature = signature

        self._update_ai_debug_logging(bool(getattr(settings, "debug_logging", False)))

    def _ai_settings_ready(self, settings: Settings) -> bool:
        """Check if AI settings are complete enough to create an orchestrator.

        Args:
            settings: The settings to check.

        Returns:
            True if settings are ready for AI orchestrator creation.
        """
        return bool(
            (settings.api_key or "").strip()
            and (settings.base_url or "").strip()
            and (settings.model or "").strip()
        )

    def _compute_ai_settings_signature(
        self, settings: Settings | None
    ) -> tuple[Any, ...] | None:
        """Compute a signature for detecting AI settings changes.

        Args:
            settings: The settings to compute signature for.

        Returns:
            A tuple signature, or None if settings is None.
        """
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
        """Resolve max tool iterations from settings.

        Args:
            settings: The settings containing max_tool_iterations.

        Returns:
            The resolved max tool iterations value (1-200).
        """
        raw = getattr(settings, "max_tool_iterations", 8) if settings else 8
        try:
            value = int(raw)
        except (TypeError, ValueError):
            value = 8
        return max(1, min(value, 200))

    def _disable_ai_orchestrator(self) -> None:
        """Disable the AI orchestrator when settings are incomplete."""
        orchestrator = self._context.ai_orchestrator
        if orchestrator is None and self._ai_settings_signature is None:
            return

        self._context.ai_orchestrator = None
        self._ai_settings_signature = None
        _LOGGER.info("AI orchestrator disabled until settings are completed.")


def _metadata_signature(metadata: Any) -> tuple[tuple[str, str], ...]:
    """Compute a stable signature for metadata.

    Args:
        metadata: The metadata mapping to compute signature for.

    Returns:
        A tuple of (key, value) pairs representing the metadata signature.
    """
    if not isinstance(metadata, Mapping):
        return ()

    entries: list[tuple[str, str]] = []
    for key, value in sorted(metadata.items(), key=lambda item: str(item[0])):
        entries.append((str(key), _stable_metadata_value(value)))
    return tuple(entries)


def _stable_metadata_value(value: Any) -> str:
    """Convert a metadata value to a stable string representation.

    Args:
        value: The value to convert.

    Returns:
        A stable string representation of the value.
    """
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except TypeError:
        return repr(value)


__all__ = ["SettingsAdapter"]
