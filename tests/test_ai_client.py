"""Tests for the OpenAI-compatible AI client."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, cast

import pytest

from openai import AsyncOpenAI

from tinkerbell.ai.client import (
    AIClient,
    AIStreamEvent,
    ClientSettings,
    ApproxByteCounter,
    TokenCounterRegistry,
)


@dataclass
class _FakeEvent:
    """Simple structure emulating ChatCompletionStreamEvent attributes."""

    type: str
    delta: str | None = None
    content: str | None = None
    name: str | None = None
    index: int | None = None
    arguments: str | None = None
    arguments_delta: str | None = None
    parsed_arguments: Any | None = None
    refusal: str | None = None


class _FakeStream:
    def __init__(self, events: Iterable[_FakeEvent]):
        self._events = list(events)
        self._iterator = iter(self._events)

    def __aiter__(self) -> "_FakeStream":
        return self

    async def __anext__(self) -> _FakeEvent:
        try:
            return next(self._iterator)
        except StopIteration as exc:  # pragma: no cover - exhaust iterator
            raise StopAsyncIteration from exc


class _FakeStreamContext:
    def __init__(self, events: Iterable[_FakeEvent]):
        self._events = list(events)

    async def __aenter__(self) -> _FakeStream:
        return _FakeStream(self._events)

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeCompletions:
    def __init__(self, events: Iterable[_FakeEvent]):
        self._events = list(events)
        self.calls: list[dict[str, Any]] = []

    def stream(self, **kwargs: Any) -> _FakeStreamContext:
        self.calls.append(kwargs)
        return _FakeStreamContext(self._events)


class _FakeModels:
    def __init__(self, payload: list[SimpleNamespace]):
        self._payload = payload
        self.calls = 0

    async def list(self) -> SimpleNamespace:
        self.calls += 1
        return SimpleNamespace(data=self._payload)


def _make_client(events: Iterable[_FakeEvent]) -> SimpleNamespace:
    completions = _FakeCompletions(events)
    models = _FakeModels([SimpleNamespace(id="test-model")])
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat, models=models)


@pytest.mark.asyncio
async def test_list_models_caches_results() -> None:
    payload = [SimpleNamespace(id="gpt-4o"), SimpleNamespace(id="gpt-4o-mini")]
    fake_models = _FakeModels(payload)
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions([])),
        models=fake_models,
    )
    client = AIClient(
        ClientSettings(base_url="http://local", api_key="test", model="stub"),
        client=cast(AsyncOpenAI, fake_client),
    )

    first = await client.list_models()
    second = await client.list_models()

    assert first == ["gpt-4o", "gpt-4o-mini"]
    assert second == first  # cached result
    assert fake_models.calls == 1


@pytest.mark.asyncio
async def test_stream_chat_normalizes_delta_and_tool_events() -> None:
    events = [
        _FakeEvent(type="content.delta", delta="Hello"),
        _FakeEvent(
            type="tool_calls.function.arguments.delta",
            name="DocumentEdit",
            index=0,
            arguments="{\"text\": \"",
            arguments_delta="{\"text\": \"",
            parsed_arguments={"text": ""},
        ),
        _FakeEvent(
            type="tool_calls.function.arguments.done",
            name="DocumentEdit",
            index=0,
            arguments="{\"text\": \"value\"}",
            parsed_arguments={"text": "value"},
        ),
        _FakeEvent(type="content.done", content="Hello world"),
    ]
    fake_client = _make_client(events)
    client = AIClient(
        ClientSettings(
            base_url="http://local",
            api_key="test",
            model="gpt-4o-mini",
            max_retries=1,
        ),
        client=cast(AsyncOpenAI, fake_client),
    )

    collected: list[AIStreamEvent] = []
    async for event in client.stream_chat(
        messages=[{"role": "user", "content": "Hi"}],
    ):
        collected.append(event)

    assert [event.type for event in collected] == [
        "content.delta",
        "tool_calls.function.arguments.delta",
        "tool_calls.function.arguments.done",
        "content.done",
    ]
    assert fake_client.chat.completions.calls[0]["messages"][0]["role"] == "user"


@pytest.mark.asyncio
async def test_stream_chat_requires_messages() -> None:
    fake_client = _make_client([])
    client = AIClient(
        ClientSettings(base_url="http://local", api_key="test", model="gpt-4o"),
        client=cast(AsyncOpenAI, fake_client),
    )

    generator = client.stream_chat(messages=[])
    with pytest.raises(ValueError):
        await generator.__anext__()


@pytest.mark.asyncio
async def test_debug_logging_captures_prompt_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    events = [_FakeEvent(type="content.done", content="done")]
    fake_client = _make_client(events)
    client = AIClient(
        ClientSettings(base_url="http://local", api_key="test", model="debug", debug_logging=True),
        client=cast(AsyncOpenAI, fake_client),
    )
    captured: dict[str, Any] = {}

    async def _noop_handler(event: AIStreamEvent) -> None:  # pragma: no cover - helper
        del event

    def _capture(payload: Any) -> None:
        captured["payload"] = payload

    monkeypatch.setattr(client, "_log_prompt_payload", _capture)

    async for _event in client.stream_chat(messages=[{"role": "user", "content": "Hello"}]):
        await _noop_handler(_event)

    assert "payload" in captured
    assert captured["payload"]["messages"][0]["content"] == "Hello"


@pytest.mark.asyncio
async def test_aclose_closes_underlying_client() -> None:
    class _StubAsyncOpenAI:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    stub = _StubAsyncOpenAI()
    client = AIClient(
        ClientSettings(base_url="http://local", api_key="test", model="stub-model"),
        client=cast(AsyncOpenAI, stub),
    )

    await client.aclose()

    assert stub.closed is True


class _FakeCounter:
    def __init__(self, multiplier: int = 1, *, model_name: str = "fake-model") -> None:
        self.multiplier = multiplier
        self.model_name: str | None = model_name

    def count(self, text: str) -> int:
        return len(text) * self.multiplier

    def estimate(self, text: str) -> int:
        return len(text)


def test_token_counter_registry_handles_missing_models() -> None:
    registry = TokenCounterRegistry(fallback=ApproxByteCounter())
    registry.register("fake", _FakeCounter(multiplier=2, model_name="fake"))

    precise = registry.count("fake", "abc")
    approximate = registry.count("missing", "abc")

    assert precise == 6
    assert approximate == registry.estimate("abc")


@pytest.mark.asyncio
async def test_ai_client_count_tokens_falls_back_when_counter_breaks() -> None:
    class _BrokenCounter:
        def __init__(self) -> None:
            self.model_name: str | None = "stub"

        def count(self, text: str) -> int:
            raise RuntimeError("boom")

        def estimate(self, text: str) -> int:
            return len(text) + 5

    registry = TokenCounterRegistry(fallback=ApproxByteCounter())
    registry.register("stub-model", _BrokenCounter())

    client = AIClient(
        ClientSettings(base_url="http://local", api_key="test", model="stub-model"),
        client=cast(AsyncOpenAI, _make_client([])),
        token_registry=registry,
    )

    assert client.count_tokens("hello") == len("hello") + 5