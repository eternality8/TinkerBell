"""Tests for the AI controller façade."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterable, List, cast

from tinkerbell.ai.agents.executor import AIController
from tinkerbell.ai.client import AIClient, AIStreamEvent


class _StubClient:
    def __init__(self, events: Iterable[Any]):
        payload = list(events)
        self._batches: List[List[AIStreamEvent]] = []
        if payload and isinstance(payload[0], AIStreamEvent):
            self._batches.append(list(cast(Iterable[AIStreamEvent], payload)))
        else:
            for batch in cast(Iterable[Iterable[AIStreamEvent]], payload):
                self._batches.append(list(cast(Iterable[AIStreamEvent], batch)))
        if not self._batches:
            self._batches = [[]]
        self.calls: list[dict[str, Any]] = []
        self.settings: Any = SimpleNamespace(debug_logging=False)

    async def stream_chat(self, **kwargs: Any) -> AsyncIterator[AIStreamEvent]:
        self.calls.append(kwargs)
        batch_index = min(len(self.calls) - 1, len(self._batches) - 1)
        for event in self._batches[batch_index]:
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


def test_ai_controller_updates_max_tool_iterations():
    stub_client = _StubClient([AIStreamEvent(type="content.done", content="done")])
    controller = AIController(client=cast(AIClient, stub_client), max_tool_iterations=3)

    assert controller.graph["metadata"]["max_iterations"] == 3

    controller.set_max_tool_iterations(5)

    assert controller.graph["metadata"]["max_iterations"] == 5


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


def test_ai_controller_executes_tool_and_continues(sample_snapshot):
    first_turn = [
        AIStreamEvent(type="content.delta", content="Working"),
        AIStreamEvent(
            type="tool_calls.function.arguments.done",
            tool_name="snapshot",
            tool_index=0,
            tool_arguments='{"delta_only": true}',
            parsed={"delta_only": True},
            tool_call_id="call-1",
        ),
    ]
    second_turn = [
        AIStreamEvent(type="content.delta", content="All set"),
        AIStreamEvent(type="content.done", content="All set"),
    ]
    stub_client = _StubClient([first_turn, second_turn])
    controller = AIController(client=cast(AIClient, stub_client))

    calls: list[Any] = []

    class _SnapshotTool:
        def run(self, delta_only: bool = False) -> dict[str, Any]:
            calls.append(delta_only)
            return {"delta_only": delta_only}

    controller.register_tool("snapshot", _SnapshotTool())

    async def run() -> dict:
        return await controller.run_chat("use tool", sample_snapshot)

    result = asyncio.run(run())

    assert result["response"] == "All set"
    assert calls == [True]
    assert result["tool_calls"][0]["name"] == "snapshot"
    assert result["tool_calls"][0]["resolved_arguments"] == {"delta_only": True}


def test_ai_controller_suggest_followups_returns_parsed_json():
    stub_client = _StubClient(
        [
            [
                AIStreamEvent(type="content.delta", content='["Outline next steps",'),
                AIStreamEvent(type="content.delta", content='"Review tone"]'),
            ]
        ]
    )
    controller = AIController(client=cast(AIClient, stub_client))

    async def run() -> list[str]:
        history = [{"role": "user", "content": "Summarize the intro"}]
        return await controller.suggest_followups(history, max_suggestions=2)

    suggestions = asyncio.run(run())

    assert suggestions == ["Outline next steps", "Review tone"]
    assert stub_client.calls
    payload = stub_client.calls[-1]
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["role"] == "user"
