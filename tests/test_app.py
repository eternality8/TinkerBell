"""Tests covering the application bootstrap helpers."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any, cast

import pytest

from tinkerbell import app
from tinkerbell.services.settings import Settings, SettingsStore


class _StubAIClient:
    def __init__(self, settings: Any):
        self.settings = settings


@pytest.fixture(autouse=True)
def _stub_ai_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "AIClient", _StubAIClient)


def test_build_ai_controller_uses_settings_iteration_limit() -> None:
    settings = Settings(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
        max_tool_iterations=25,
    )

    controller = app._build_ai_controller(settings)

    assert controller is not None
    assert controller.max_tool_iterations == 25
    assert controller.graph["metadata"]["max_iterations"] == 25


def test_build_ai_controller_clamps_iteration_limit() -> None:
    settings = Settings(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
        max_tool_iterations=0,
    )

    controller = app._build_ai_controller(settings)

    assert controller is not None
    assert controller.max_tool_iterations == 1

    high_settings = Settings(
        base_url="https://api.example.com/v1",
        api_key="test-key",
        model="gpt-4o-mini",
        max_tool_iterations=500,
    )

    high_controller = app._build_ai_controller(high_settings)

    assert high_controller is not None
    assert high_controller.max_tool_iterations == 200


def test_drain_event_loop_cancels_pending_tasks() -> None:
    loop = asyncio.new_event_loop()

    cancellation_flag = {"called": False}

    async def pending() -> None:
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:  # pragma: no cover - cancellation path exercised
            cancellation_flag["called"] = True
            raise

    loop.create_task(pending())

    try:
        app._drain_event_loop(loop)
        assert cancellation_flag["called"] is True
    finally:
        loop.close()


def test_drain_event_loop_ignores_closed_loop() -> None:
    loop = asyncio.new_event_loop()
    loop.close()

    app._drain_event_loop(loop)


def test_drain_event_loop_skips_current_task(monkeypatch: pytest.MonkeyPatch) -> None:
    loop = asyncio.new_event_loop()

    real_all_tasks = asyncio.all_tasks
    recorded: dict[str, asyncio.Task[Any] | None] = {"current": None}

    def fake_all_tasks(target_loop: asyncio.AbstractEventLoop | None = None):
        tasks = set(real_all_tasks(target_loop))
        try:
            current = asyncio.current_task(loop=target_loop)
        except RuntimeError:
            current = None
        recorded["current"] = current
        if current is not None:
            tasks.add(current)
        return tasks

    monkeypatch.setattr(app.asyncio, "all_tasks", fake_all_tasks)

    try:
        app._drain_event_loop(loop)
        current = recorded["current"]
        if current is not None:
            assert not current.cancelled()
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_shutdown_ai_controller_closes_controller() -> None:
    class _StubController:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    controller = _StubController()

    await app._shutdown_ai_controller(cast(app.AIController, controller))

    assert controller.closed is True


@pytest.mark.asyncio
async def test_shutdown_ai_controller_ignores_missing_method() -> None:
    class _StubController:
        pass

    controller = _StubController()

    await app._shutdown_ai_controller(cast(app.AIController, controller))


def test_coerce_cli_overrides_casts_types() -> None:
    overrides = app._coerce_cli_overrides(
        [
            "base_url=https://cli",
            "debug_logging=true",
            "max_tool_iterations=12",
            "request_timeout=42.25",
        ]
    )

    assert overrides["base_url"] == "https://cli"
    assert overrides["debug_logging"] is True
    assert overrides["max_tool_iterations"] == 12
    assert overrides["request_timeout"] == pytest.approx(42.25)


def test_coerce_cli_overrides_rejects_unknown_field() -> None:
    with pytest.raises(ValueError):
        app._coerce_cli_overrides(["not_a_setting=value"])


def test_dump_settings_redacts_api_key(tmp_path: Path) -> None:
    settings = Settings(api_key="super-secret", base_url="https://example.com")
    store = SettingsStore(tmp_path / "settings.json")
    buffer = io.StringIO()

    app._dump_settings(settings, store, overrides={"base_url": "https://cli"}, stream=buffer)

    payload = json.loads(buffer.getvalue())
    redacted = payload["settings"]["api_key"]
    assert "super-secret" not in redacted
    assert payload["meta"]["secret_backend"] == store.vault.strategy
    assert "base_url" in payload["meta"]["cli_overrides"]


def test_choose_qt_event_loop_uses_selector_on_windows() -> None:
    class _Default:
        pass

    class _Selector:
        pass

    selected = app._choose_qt_event_loop_class(_Default, _Selector, platform="nt")

    assert selected is _Selector


def test_choose_qt_event_loop_uses_default_elsewhere() -> None:
    class _Default:
        pass

    class _Selector:
        pass

    selected = app._choose_qt_event_loop_class(_Default, _Selector, platform="posix")

    assert selected is _Default
