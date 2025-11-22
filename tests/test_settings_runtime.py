"""Unit tests for the SettingsRuntime helper."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from tinkerbell.services.settings import Settings
from tinkerbell.services.unsaved_cache import UnsavedCache
from tinkerbell.ui.models.window_state import WindowContext
from tinkerbell.ui.settings_runtime import SettingsRuntime


class _EditorStub:
    def __init__(self) -> None:
        self.applied_theme = None

    def apply_theme(self, theme) -> None:
        self.applied_theme = theme


class _TelemetryStub:
    def __init__(self) -> None:
        self.flags: list[bool] = []

    def set_subagent_enabled(self, flag: bool) -> None:
        self.flags.append(flag)


class _EmbeddingStub:
    def __init__(self) -> None:
        self.runtime_settings: Settings | None = None

    def refresh_runtime(self, settings: Settings) -> None:
        self.runtime_settings = settings


class _ControllerStub:
    def __init__(self) -> None:
        self.max_iterations: int | None = None
        self.context_window: dict[str, int] | None = None
        self.policy = None
        self.subagent_config = None
        self.updated_clients: list[object] = []
        self.client = SimpleNamespace(settings=SimpleNamespace(debug_logging=False))
        self.event_logging_enabled = False

    def set_max_tool_iterations(self, value: int) -> None:
        self.max_iterations = value

    def configure_context_window(self, **kwargs: int) -> None:
        self.context_window = dict(kwargs)

    def configure_budget_policy(self, policy) -> None:
        self.policy = policy

    def configure_subagents(self, config) -> None:
        self.subagent_config = config

    def update_client(self, client) -> None:
        self.updated_clients.append(client)

    def configure_debug_event_logging(self, *, enabled: bool | None = None, event_log_dir=None) -> None:
        if enabled is not None:
            self.event_logging_enabled = bool(enabled)


class _FakeTask:
    def __init__(self) -> None:
        self.cancelled = False

    def done(self) -> bool:
        return False

    def cancel(self) -> None:
        self.cancelled = True


def _runtime_bundle(initial_settings: Settings | None = None):
    initial = initial_settings or Settings(api_key="init-key", base_url="https://api", model="gpt-4o-mini")
    context = WindowContext(settings=initial, unsaved_cache=UnsavedCache())
    editor = _EditorStub()
    telemetry = _TelemetryStub()
    embedding = _EmbeddingStub()
    outline_tool = SimpleNamespace(budget_policy=None)
    register_state = {"count": 0}
    ai_state = {"task": None, "streaming": False}

    runtime = SettingsRuntime(
        context=context,
        editor=editor,
        telemetry_controller=telemetry,
        embedding_controller=embedding,
        register_default_ai_tools=lambda: register_state.__setitem__("count", register_state["count"] + 1),
        outline_tool_provider=lambda: outline_tool,
        ai_task_getter=lambda: ai_state["task"],
        ai_task_setter=lambda task: ai_state.__setitem__("task", task),
        ai_stream_state_setter=lambda active: ai_state.__setitem__("streaming", active),
        initial_settings=initial,
    )

    return {
        "runtime": runtime,
        "context": context,
        "editor": editor,
        "telemetry": telemetry,
        "embedding": embedding,
        "outline_tool": outline_tool,
        "register_state": register_state,
        "ai_state": ai_state,
    }


def test_apply_theme_setting_updates_editor_and_application(monkeypatch) -> None:
    bundle = _runtime_bundle(Settings(theme="default"))
    runtime = bundle["runtime"]
    editor = bundle["editor"]

    def _fake_load(name: str):
        return SimpleNamespace(name=f"{name}-theme")

    applied_to_app: dict[str, str] = {}
    monkeypatch.setattr("tinkerbell.ui.settings_runtime.load_theme", _fake_load)
    monkeypatch.setattr(
        "tinkerbell.ui.settings_runtime.theme_manager.apply_to_application",
        lambda theme: applied_to_app.setdefault("theme", theme.name),
    )

    runtime.apply_theme_setting(Settings(theme="midnight"))

    assert editor.applied_theme is not None
    assert editor.applied_theme.name == "midnight-theme"
    assert applied_to_app["theme"] == "midnight-theme"


def test_apply_runtime_settings_creates_controller_and_refreshes_runtime(monkeypatch) -> None:
    settings = Settings(
        api_key="live",
        base_url="https://api",
        model="gpt-4o-mini",
        enable_subagents=True,
        enable_plot_scaffolding=True,
        max_context_tokens=4096,
        response_token_reserve=512,
        max_tool_iterations=5,
    )
    bundle = _runtime_bundle(settings)
    runtime = bundle["runtime"]
    context = bundle["context"]
    telemetry = bundle["telemetry"]
    embedding = bundle["embedding"]
    outline_tool = bundle["outline_tool"]

    controller = _ControllerStub()
    monkeypatch.setattr(runtime, "build_ai_controller_from_settings", lambda cfg: controller)
    monkeypatch.setattr(runtime, "_build_context_budget_policy", lambda cfg: "policy")
    monkeypatch.setattr(runtime, "_build_subagent_runtime_config", lambda cfg: "subagent-config")

    handlers: dict[str, bool] = {}
    runtime.apply_runtime_settings(
        settings,
        chat_panel_handler=lambda s: handlers.setdefault("chat", s is settings),
        outline_handler=lambda s: handlers.setdefault("outline", True),
        phase3_handler=lambda s: handlers.setdefault("phase3", True),
        plot_scaffolding_handler=lambda s: handlers.setdefault("plot", True),
        safe_edit_handler=lambda s: handlers.setdefault("safe", True),
    )

    assert handlers == {"chat": True, "outline": True, "phase3": True, "plot": True, "safe": True}
    assert context.ai_controller is controller
    assert bundle["register_state"]["count"] == 1
    assert telemetry.flags[-1] is True
    assert embedding.runtime_settings is settings
    assert controller.max_iterations == 5
    assert controller.context_window == {
        "max_context_tokens": 4096,
        "response_token_reserve": 512,
    }
    assert controller.policy == "policy"
    assert controller.subagent_config == "subagent-config"
    assert outline_tool.budget_policy == "policy"


def test_apply_runtime_settings_disables_controller_without_credentials() -> None:
    valid = Settings(api_key="live", base_url="https://api", model="gpt-4o-mini")
    bundle = _runtime_bundle(valid)
    runtime = bundle["runtime"]
    context = bundle["context"]
    embedding = bundle["embedding"]
    telemetry = bundle["telemetry"]
    ai_state = bundle["ai_state"]

    controller = _ControllerStub()
    context.ai_controller = controller
    runtime.ai_client_signature = ("sig",)
    task = _FakeTask()
    ai_state["task"] = task

    updated = Settings(api_key="", base_url="https://api", model="gpt-4o-mini")
    runtime.apply_runtime_settings(
        updated,
        chat_panel_handler=lambda s: None,
        outline_handler=lambda s: None,
        phase3_handler=lambda s: None,
        plot_scaffolding_handler=lambda s: None,
        safe_edit_handler=lambda s: None,
    )

    assert context.ai_controller is None
    assert runtime.ai_client_signature is None
    assert ai_state["task"] is None
    assert ai_state["streaming"] is False


def test_event_logging_toggle_updates_controller() -> None:
    base = Settings(api_key="live", base_url="https://api", model="gpt-4o-mini", debug_event_logging=False)
    bundle = _runtime_bundle(base)
    runtime = bundle["runtime"]
    context = bundle["context"]
    controller = _ControllerStub()
    context.ai_controller = controller
    runtime.ai_client_signature = runtime._ai_settings_signature(base)

    updated = replace(base, debug_event_logging=True)
    runtime.apply_runtime_settings(
        updated,
        chat_panel_handler=lambda s: None,
        outline_handler=lambda s: None,
        phase3_handler=lambda s: None,
        plot_scaffolding_handler=lambda s: None,
        safe_edit_handler=lambda s: None,
    )

    assert controller.event_logging_enabled is True


def test_safe_edit_handler_invoked() -> None:
    base = Settings(api_key="live", base_url="https://api", model="gpt-4o-mini")
    bundle = _runtime_bundle(base)
    runtime = bundle["runtime"]
    invoked: list[Settings] = []

    runtime.apply_runtime_settings(
        base,
        chat_panel_handler=lambda s: None,
        outline_handler=lambda s: None,
        phase3_handler=lambda s: None,
        plot_scaffolding_handler=lambda s: None,
        safe_edit_handler=lambda s: invoked.append(s),
    )

    assert invoked == [base]
