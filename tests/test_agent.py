"""Tests for the AI controller façade."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterable, List, cast

import pytest

import tinkerbell.ai.agents.executor as agent_executor
from tinkerbell.ai.agents.executor import AIController
from tinkerbell.ai.client import AIClient, AIStreamEvent
from tinkerbell.ai.services.context_policy import BudgetDecision, ContextBudgetPolicy
from tinkerbell.ai import prompts
from tinkerbell.services.settings import ContextPolicySettings


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


class _StubBudgetPolicy:
    def __init__(self, verdicts: Iterable[str]):
        self.verdicts = list(verdicts)
        self.enabled = True
        self.dry_run = False
        self.prompt_budget = 1_000
        self.response_reserve = 0
        self.emergency_buffer = 0
        self.model_name = "stub"
        self._last: BudgetDecision | None = None
        self.history: list[str] = []

    def tokens_available(
        self,
        *,
        prompt_tokens: int,
        response_reserve: int | None = None,
        pending_tool_tokens: int = 0,
        document_id: str | None = None,
        **_kwargs: Any,
    ) -> BudgetDecision:
        verdict = self.verdicts.pop(0) if self.verdicts else "ok"
        decision = BudgetDecision(
            verdict=verdict,  # type: ignore[arg-type]
            reason="forced" if verdict != "ok" else "within-budget",
            prompt_tokens=prompt_tokens + pending_tool_tokens,
            prompt_budget=self.prompt_budget,
            response_reserve=response_reserve or self.response_reserve,
            pending_tool_tokens=pending_tool_tokens,
            deficit=0,
            dry_run=False,
            model=self.model_name,
            document_id=document_id,
        )
        self._last = decision
        self.history.append(decision.verdict)
        return decision

    def record_usage(self, *_args, **_kwargs) -> None:  # pragma: no cover - no-op
        return

    def status_snapshot(self) -> dict[str, object]:  # pragma: no cover - simple helper
        verdict = self._last.verdict if self._last else None
        return {"verdict": verdict}


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
    assert graph["entry"] == "plan"
    assert graph["metadata"]["max_iterations"] == 8
    assert [node["name"] for node in graph["nodes"][:5]] == [
        "plan",
        "select_tool",
        "tool_executor",
        "safety_validator",
        "response_builder",
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


def test_ai_controller_records_context_usage_events(sample_snapshot):
    stub_client = _StubClient([AIStreamEvent(type="content.done", content="ok")])
    stub_client.settings.model = "fake-model"
    controller = AIController(client=cast(AIClient, stub_client), telemetry_enabled=True, telemetry_limit=10)

    snapshot = dict(sample_snapshot)
    snapshot.update(
        {
            "embedding_backend": "langchain",
            "embedding_model": "deepseek-embedding",
            "embedding_status": "ready",
            "embedding_detail": "LangChain/DeepSeek",
        }
    )

    async def run() -> dict:
        return await controller.run_chat("track", snapshot)

    result = asyncio.run(run())
    assert result["response"] == "ok"

    events = controller.get_recent_context_events(limit=1)
    assert len(events) == 1
    event = events[0]
    assert event.model == "fake-model"
    assert event.prompt_tokens > 0
    assert event.embedding_backend == "langchain"
    assert event.embedding_model == "deepseek-embedding"
    assert event.embedding_status == "ready"
    assert event.embedding_detail == "LangChain/DeepSeek"


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
    tool_trace = result["tool_calls"][0]
    assert tool_trace["name"] == "snapshot"
    assert tool_trace["resolved_arguments"] == {"delta_only": True}
    assert tool_trace["status"] == "ok"
    assert tool_trace["tokens_used"] >= 0
    assert tool_trace["duration_ms"] >= 0
    assert tool_trace["diff_summary"] is None
    assert "started_at" in tool_trace


def test_ai_controller_injects_outline_guardrail_hint(sample_snapshot):
    first_turn = [
        AIStreamEvent(
            type="tool_calls.function.arguments.done",
            tool_name="document_outline",
            tool_index=0,
            tool_arguments="{}",
            parsed={},
        )
    ]
    second_turn = [AIStreamEvent(type="content.done", content="ok")]
    stub_client = _StubClient([first_turn, second_turn])
    controller = AIController(client=cast(AIClient, stub_client))

    class _OutlineTool:
        def run(self) -> dict[str, Any]:
            return {
                "status": "pending",
                "reason": "outline_pending",
                "document_id": "doc-guardrail",
                "outline_available": False,
                "retry_after_ms": 1200,
                "guardrails": [
                    {
                        "type": "huge_document",
                        "message": "Document exceeds safe outline size; only top-level sections returned.",
                        "action": "Work in chunks",
                    }
                ],
            }

    controller.register_tool("document_outline", _OutlineTool())

    async def run() -> dict:
        return await controller.run_chat("plan", sample_snapshot)

    asyncio.run(run())

    assert len(stub_client.calls) >= 2
    second_messages = stub_client.calls[1]["messages"]
    hint_messages = [
        msg
        for msg in second_messages
        if msg.get("role") == "system" and isinstance(msg.get("content"), str) and "Guardrail hint" in msg.get("content", "")
    ]
    assert hint_messages, second_messages
    assert any("DocumentOutlineTool" in msg["content"] for msg in hint_messages)
    assert any("still building" in msg["content"].lower() for msg in hint_messages)


def test_ai_controller_injects_retrieval_guardrail_hint(sample_snapshot):
    first_turn = [
        AIStreamEvent(
            type="tool_calls.function.arguments.done",
            tool_name="document_find_sections",
            tool_index=0,
            tool_arguments='{"query": "intro"}',
            parsed={"query": "intro"},
        )
    ]
    second_turn = [AIStreamEvent(type="content.done", content="done")]
    stub_client = _StubClient([first_turn, second_turn])
    controller = AIController(client=cast(AIClient, stub_client))

    class _RetrievalTool:
        def run(self, **_kwargs: Any) -> dict[str, Any]:
            return {
                "status": "offline_fallback",
                "document_id": "doc-retrieval",
                "query": "intro",
                "strategy": "fallback",
                "offline_mode": True,
                "fallback_reason": "embedding_unavailable",
                "pointers": [],
            }

    controller.register_tool("document_find_sections", _RetrievalTool())

    async def run() -> dict:
        return await controller.run_chat("find intro", sample_snapshot)

    asyncio.run(run())

    assert len(stub_client.calls) >= 2
    second_messages = stub_client.calls[1]["messages"]
    hint_messages = [
        msg
        for msg in second_messages
        if msg.get("role") == "system" and isinstance(msg.get("content"), str) and "Guardrail hint" in msg.get("content", "")
    ]
    assert hint_messages, second_messages
    assert any("DocumentFindSectionsTool" in msg["content"] for msg in hint_messages)
    assert any("embeddings" in msg["content"].lower() for msg in hint_messages)


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


def test_ai_controller_emits_budget_decision(monkeypatch, sample_snapshot):
    stub_client = _StubClient([AIStreamEvent(type="content.done", content="ready")])
    policy = ContextBudgetPolicy.from_settings(
        ContextPolicySettings(enabled=True, dry_run=True, prompt_budget_override=1_500),
        model_name="stub",
        max_context_tokens=2_000,
        response_token_reserve=200,
    )
    captured: list[tuple[str, dict[str, object] | None]] = []

    def _emit(name: str, payload: dict[str, object] | None = None) -> None:
        captured.append((name, payload))

    monkeypatch.setattr(agent_executor.telemetry_service, "emit", _emit)

    controller = AIController(client=cast(AIClient, stub_client), budget_policy=policy)

    async def run() -> None:
        await controller.run_chat("hello", sample_snapshot)

    asyncio.run(run())
    assert captured
    name, payload = captured[-1]
    assert name == "context_budget_decision"
    assert payload and payload.get("verdict") == "ok"


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


def test_outline_routing_hint_flags_large_documents(sample_snapshot):
    stub_client = _StubClient([AIStreamEvent(type="content.done", content="ok")])
    controller = AIController(client=cast(AIClient, stub_client))

    snapshot = dict(sample_snapshot)
    snapshot["text"] = "# Heading\n" + ("body\n" * (prompts.LARGE_DOC_CHAR_THRESHOLD // 5 + 10))
    snapshot["document_id"] = "doc-large"

    hint = controller._outline_routing_hint("Please outline this document", snapshot)

    assert hint is not None
    assert "DocumentOutlineTool" in hint
    assert "large" in hint.lower()


def test_outline_routing_hint_promotes_retrieval_requests(sample_snapshot):
    stub_client = _StubClient([AIStreamEvent(type="content.done", content="ok")])
    controller = AIController(client=cast(AIClient, stub_client))

    snapshot = dict(sample_snapshot)
    snapshot["text"] = "Short"
    snapshot["document_id"] = "doc-retrieval"

    hint = controller._outline_routing_hint("Can you find section about safety policies?", snapshot)

    assert hint is not None
    assert "DocumentFindSectionsTool" in hint


def test_outline_routing_hint_tracks_outline_digest(sample_snapshot):
    stub_client = _StubClient([AIStreamEvent(type="content.done", content="ok")])
    controller = AIController(client=cast(AIClient, stub_client))

    snapshot = dict(sample_snapshot)
    snapshot["document_id"] = "doc-digest"
    snapshot.pop("text", None)
    snapshot["outline_digest"] = "deadbeefcafefeed"

    first = controller._outline_routing_hint("", snapshot)
    assert first is not None
    assert "Outline digest updated" in first

    repeat = controller._outline_routing_hint("", snapshot)
    assert repeat is not None
    assert "matches your prior fetch" in repeat


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
    diff_trace, edit_trace = result["tool_calls"]
    assert diff_trace["status"] == "ok"
    assert diff_trace["diff_summary"] == "+1/-1 lines across 1 hunk(s)"
    assert edit_trace["status"] == "ok"
    assert edit_trace["diff_summary"] == "+0/-0 lines across 1 hunk(s)"
    assert len(stub_client.calls) == 4
    reminder_payload = stub_client.calls[2]
    assert reminder_payload["messages"][-1]["role"] == "system"
    assert "document_edit" in reminder_payload["messages"][-1]["content"]


def test_ai_controller_compacts_tool_output_with_pointer(sample_snapshot):
    first_turn = [
        AIStreamEvent(
            type="tool_calls.function.arguments.done",
            tool_name="snapshot",
            tool_index=0,
            tool_arguments="{}",
            parsed={},
        )
    ]
    final_turn = [AIStreamEvent(type="content.done", content="complete")]
    stub_client = _StubClient([first_turn, final_turn])
    policy = _StubBudgetPolicy(["ok", "needs_summary", "needs_summary", "ok"])
    controller = AIController(client=cast(AIClient, stub_client), budget_policy=cast(ContextBudgetPolicy, policy))

    class _VerboseTool:
        def run(self) -> str:
            return "\n".join(f"line {idx}: lorem ipsum" for idx in range(200))

    controller.register_tool("snapshot", _VerboseTool())

    async def run() -> dict:
        return await controller.run_chat("use tool", sample_snapshot)

    result = asyncio.run(run())
    tool_record = result["tool_calls"][0]
    pointer = tool_record.get("pointer")
    assert pointer is not None, policy.history
    assert pointer["metadata"]["tool_name"] == "snapshot"
    assert pointer["display_text"]
    assert pointer["rehydrate_instructions"].startswith("Re-run snapshot")
    assert tool_record["result"].startswith("line 0")


def test_ai_controller_respects_non_summarizable_tool(sample_snapshot):
    first_turn = [
        AIStreamEvent(
            type="tool_calls.function.arguments.done",
            tool_name="critical_tool",
            tool_index=0,
            tool_arguments="{}",
            parsed={},
        )
    ]
    final_turn = [AIStreamEvent(type="content.done", content="complete")]
    stub_client = _StubClient([first_turn, final_turn])
    policy = _StubBudgetPolicy(["ok", "needs_summary"])
    controller = AIController(client=cast(AIClient, stub_client), budget_policy=cast(ContextBudgetPolicy, policy))

    class _CriticalTool:
        summarizable = False

        def run(self) -> str:
            return "critical status: applied patch"

    controller.register_tool("critical_tool", _CriticalTool())

    async def run() -> dict:
        return await controller.run_chat("use tool", sample_snapshot)

    result = asyncio.run(run())
    tool_record = result["tool_calls"][0]
    assert "pointer" not in tool_record
    assert tool_record["result"].startswith("critical status")


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
