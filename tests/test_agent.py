"""Tests for the AI controller façade."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterable, List, cast

from tinkerbell.ai.agents.executor import AIController
from tinkerbell.ai.client import AIClient, AIStreamEvent


class _StubClient:
    def __init__(self, events: Iterable[AIStreamEvent]):
        self._events: List[AIStreamEvent] = list(events)
        self.calls: list[dict[str, Any]] = []
        self.settings: Any = SimpleNamespace(debug_logging=False)

    async def stream_chat(self, **kwargs: Any) -> AsyncIterator[AIStreamEvent]:
        self.calls.append(kwargs)
        for event in self._events:
            yield event


def test_ai_controller_collects_streamed_response(sample_snapshot):
    stub_client = _StubClient(
        [
            AIStreamEvent(type="content.delta", content="Hello"),
            AIStreamEvent(type="content.delta", content=" world"),
            AIStreamEvent(type="content.done", content="!"),
        ]
    )
    controller = AIController(client=cast(AIClient, stub_client))

    async def run() -> dict:
        return await controller.run_chat("hi", sample_snapshot, metadata={"tab": "notes"})

    result = asyncio.run(run())
    assert result["prompt"] == "hi"
    assert result["response"] == "Hello world!"

    graph = result["graph"]
    assert graph["entry"] == "ingest"
    assert graph["metadata"]["max_iterations"] == 8
    assert [node["name"] for node in graph["nodes"][:5]] == [
        "ingest",
        "planner",
        "tool_router",
        "guard",
        "respond",
    ]
    assert graph["tools"] == []
    assert stub_client.calls[0]["metadata"] == {"tab": "notes"}


def test_ai_controller_tracks_registered_tools(sample_snapshot):
    stub_client = _StubClient([AIStreamEvent(type="content.done", content="done")])
    controller = AIController(client=cast(AIClient, stub_client))

    controller.register_tool("snapshot", object(), description="doc snapshot")

    assert controller.available_tools() == ("snapshot",)
    graph = controller.graph
    assert graph["metadata"]["tooling"]["registered"] == 1
    assert graph["tools"][0]["name"] == "snapshot"
    assert any(node["name"] == "tool:snapshot" for node in graph["nodes"])

    async def run() -> dict:
        return await controller.run_chat("hi", sample_snapshot)

    result = asyncio.run(run())
    assert result["tool_calls"] == []


def test_ai_controller_logs_response_when_debug_enabled(sample_snapshot, caplog):
    stub_client = _StubClient(
        [
            AIStreamEvent(type="content.delta", content="final"),
            AIStreamEvent(type="content.delta", content=" answer"),
        ]
    )
    stub_client.settings = SimpleNamespace(debug_logging=True)
    controller = AIController(client=cast(AIClient, stub_client))

    async def run() -> dict:
        return await controller.run_chat("Explain", sample_snapshot)

    with caplog.at_level("DEBUG"):
        result = asyncio.run(run())

    assert result["response"] == "final answer"
    assert any("AI response text" in record.message and "final answer" in record.message for record in caplog.records)
