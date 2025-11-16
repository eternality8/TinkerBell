"""Tests for the AI controller façade."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterable, List, cast

import pytest

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


def test_ai_controller_limits_history_and_reserves_completion_tokens(sample_snapshot):
    stub_client = _StubClient([AIStreamEvent(type="content.done", content="ok")])
    controller = AIController(
        client=cast(AIClient, stub_client),
        max_context_tokens=40_000,
        response_token_reserve=16_000,
    )

    long_history = [
        {
            "role": "user" if idx % 2 == 0 else "assistant",
            "content": f"message-{idx} " + ("#" * 2000),
        }
        for idx in range(80)
    ]

    async def run() -> None:
        await controller.run_chat("Trim please", sample_snapshot, history=long_history)

    asyncio.run(run())

    payload = stub_client.calls[0]
    assert payload.get("max_completion_tokens") == 16_000
    sent_messages = payload["messages"]
    history_sent = sent_messages[1:-1]
    assert history_sent  # history should not be empty
    assert len(history_sent) < len(long_history)
    assert len(history_sent) < 60  # token budget trims further than the window limit
    assert history_sent[-1]["content"].startswith("message-79")


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


def test_ai_controller_emits_result_events_with_fallback_ids(sample_snapshot):
    first_turn = [
        AIStreamEvent(
            type="tool_calls.function.arguments.delta",
            tool_name="snapshot",
            tool_index=0,
            arguments_delta='{"delta_only": true',
        ),
        AIStreamEvent(
            type="tool_calls.function.arguments.done",
            tool_name="snapshot",
            tool_index=0,
            tool_arguments='{"delta_only": true}',
            parsed={"delta_only": True},
        ),
    ]
    second_turn = [
        AIStreamEvent(type="content.delta", content="All done"),
        AIStreamEvent(type="content.done", content="All done"),
    ]
    stub_client = _StubClient([first_turn, second_turn])
    controller = AIController(client=cast(AIClient, stub_client))

    recorded: list[AIStreamEvent] = []

    def _on_event(event: AIStreamEvent) -> None:
        recorded.append(event)

    class _SnapshotTool:
        def run(self, delta_only: bool = False) -> str:
            return f"delta_only={delta_only}"  # pragma: no cover - deterministic

    controller.register_tool("snapshot", _SnapshotTool())

    async def run() -> dict:
        return await controller.run_chat("use tool", sample_snapshot, on_event=_on_event)

    asyncio.run(run())

    result_event = next(evt for evt in recorded if evt.type == "tool_calls.result")
    assert result_event.tool_call_id == "snapshot:0"
    assert result_event.tool_index == 0


def test_ai_controller_includes_history_before_latest_prompt(sample_snapshot):
    stub_client = _StubClient([AIStreamEvent(type="content.done", content="ready")])
    controller = AIController(client=cast(AIClient, stub_client))

    history = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier reply"},
    ]

    async def run() -> dict:
        return await controller.run_chat("New topic", sample_snapshot, history=history)

    asyncio.run(run())

    messages = stub_client.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1:3] == history
    assert messages[-1]["role"] == "user"
    assert "New topic" in messages[-1]["content"]


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


def test_ai_controller_prompts_until_document_edit_runs(sample_snapshot):
    diff_turn = [
        AIStreamEvent(
            type="tool_calls.function.arguments.done",
            tool_name="diff_builder",
            tool_index=0,
            tool_arguments='{"original":"hello","updated":"HELLO"}',
            parsed={"original": "hello", "updated": "HELLO"},
        )
    ]
    idle_turn = [AIStreamEvent(type="content.done", content="Working on it")]
    edit_turn = [
        AIStreamEvent(
            type="tool_calls.function.arguments.done",
            tool_name="document_edit",
            tool_index=0,
            tool_arguments='{"action":"patch","diff":"diff","document_version":"digest"}',
            parsed={"action": "patch", "diff": "diff", "document_version": "digest"},
        )
    ]
    final_turn = [AIStreamEvent(type="content.done", content="Applied!")]

    stub_client = _StubClient([diff_turn, idle_turn, edit_turn, final_turn])
    controller = AIController(client=cast(AIClient, stub_client))

    diff_calls: list[tuple[str, str]] = []
    edit_calls: list[dict[str, Any]] = []

    class _DiffTool:
        def run(self, original: str, updated: str) -> str:
            diff_calls.append((original, updated))
            return "--- a/doc\n+++ b/doc\n@@ -1 +1 @@\n-old\n+new\n"

    class _EditTool:
        def run(self, action: str, diff: str, document_version: str) -> str:
            edit_calls.append({"action": action, "diff": diff, "version": document_version})
            return "applied"

    controller.register_tool("diff_builder", _DiffTool())
    controller.register_tool("document_edit", _EditTool())

    async def run() -> dict:
        return await controller.run_chat("please edit", sample_snapshot)

    result = asyncio.run(run())

    assert len(diff_calls) == 1
    assert len(edit_calls) == 1
    assert len(result["tool_calls"]) == 2
    assert len(stub_client.calls) == 4
    reminder_payload = stub_client.calls[2]
    assert reminder_payload["messages"][-1]["role"] == "system"
    assert "document_edit" in reminder_payload["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_ai_controller_aclose_cancels_active_task(sample_snapshot) -> None:
    pending = asyncio.Event()

    class _BlockingClient:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(debug_logging=False)
            self.closed = False

        async def stream_chat(self, **kwargs: Any) -> AsyncIterator[AIStreamEvent]:
            del kwargs
            yield AIStreamEvent(type="content.delta", content="hi")
            await pending.wait()

        async def aclose(self) -> None:
            self.closed = True
            pending.set()

    stub_client = _BlockingClient()
    controller = AIController(client=cast(AIClient, stub_client))

    run_task = asyncio.create_task(controller.run_chat("hello", sample_snapshot))
    await asyncio.sleep(0)

    await controller.aclose()

    assert stub_client.closed is True
    assert run_task.cancelled()
