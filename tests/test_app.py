"""Tests covering the application bootstrap helpers."""

from __future__ import annotations

from typing import Any

import pytest

from tinkerbell import app
from tinkerbell.services.settings import Settings


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
    assert high_controller.max_tool_iterations == 50
