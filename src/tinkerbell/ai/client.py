"""Async AI client wrapper built around OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterable, List, Mapping, MutableMapping, Sequence, cast

import httpx
from openai import AsyncOpenAI, APIConnectionError, APIError, APIStatusError, RateLimitError
from openai.lib.streaming.chat import ChatCompletionStreamEvent
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ClientSettings:
    """Subset of settings required to configure the AI client."""

    base_url: str
    api_key: str
    model: str
    organization: str | None = None
    request_timeout: float | None = 30.0
    max_retries: int = 3
    retry_min_seconds: float = 0.5
    retry_max_seconds: float = 6.0
    default_headers: Mapping[str, str] | None = None
    metadata: Mapping[str, str] | None = None


@dataclass(slots=True)
class AIStreamEvent:
    """Normalized representation of streaming deltas."""

    type: str
    content: str | None = None
    parsed: Any | None = None
    tool_name: str | None = None
    tool_index: int | None = None
    tool_arguments: str | None = None
    arguments_delta: str | None = None


class AIClient:
    """Async client providing streaming helpers with retry semantics."""

    def __init__(self, settings: ClientSettings, *, client: AsyncOpenAI | None = None) -> None:
        self._settings = settings
        self._client = client or self._build_client(settings)
        self._models_cache: List[str] | None = None
        self._models_lock = asyncio.Lock()

    @property
    def settings(self) -> ClientSettings:
        return self._settings

    async def stream_chat(
        self,
        messages: Iterable[Mapping[str, Any] | ChatCompletionMessageParam],
        *,
        tools: Iterable[ChatCompletionToolParam] | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
        response_format: type[Any] | ResponseFormat | None = None,
        temperature: float | None = 0.2,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        metadata: Mapping[str, str] | None = None,
        **extra_params: Any,
    ) -> AsyncIterator[AIStreamEvent]:
        """Stream chat completions for the provided messages."""

        payload = self._build_chat_payload(
            messages=self._coerce_messages(messages),
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            metadata=metadata,
            extra_params=extra_params,
        )
        message_count = len(payload["messages"])
        LOGGER.debug(
            "Starting streamed chat completion via %s with %s message(s)",
            self._settings.model,
            message_count,
        )

        async for attempt in self._retrying():
            with attempt:
                async with self._client.chat.completions.stream(**payload) as stream:
                    async for event in stream:
                        normalized = self._normalize_stream_event(event)
                        if normalized is not None:
                            yield normalized
                break

    async def list_models(self, *, force_refresh: bool = False) -> List[str]:
        """Return a list of supported model identifiers."""

        if self._models_cache is not None and not force_refresh:
            return list(self._models_cache)

        async with self._models_lock:
            if self._models_cache is not None and not force_refresh:
                return list(self._models_cache)

            response = await self._client.models.list()
            models = [item.id for item in response.data if getattr(item, "id", None)]
            self._models_cache = models
            return list(models)

    def _build_client(self, settings: ClientSettings) -> AsyncOpenAI:
        headers = dict(settings.default_headers) if settings.default_headers else None
        return AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            organization=settings.organization,
            timeout=settings.request_timeout,
            default_headers=headers,
        )

    def _retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(max(1, self._settings.max_retries)),
            wait=wait_exponential(
                multiplier=self._settings.retry_min_seconds,
                max=self._settings.retry_max_seconds,
            ),
            retry=retry_if_exception_type(
                (
                    APIError,
                    APIStatusError,
                    APIConnectionError,
                    RateLimitError,
                    httpx.TimeoutException,
                )
            ),
        )

    def _coerce_messages(
        self, messages: Iterable[Mapping[str, Any] | ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        normalized: List[ChatCompletionMessageParam] = []
        for message in messages:
            if isinstance(message, MutableMapping):
                normalized.append(cast(ChatCompletionMessageParam, dict(message)))
            else:
                try:
                    normalized.append(cast(ChatCompletionMessageParam, dict(message)))
                except TypeError as exc:  # pragma: no cover - defensive guard
                    raise TypeError("Messages must be mapping-like objects") from exc
        if not normalized:
            raise ValueError("At least one message is required to start a chat")
        return normalized

    def _build_chat_payload(
        self,
        *,
        messages: Sequence[ChatCompletionMessageParam],
        tools: Iterable[ChatCompletionToolParam] | None,
        tool_choice: ChatCompletionToolChoiceOptionParam | None,
        response_format: type[Any] | ResponseFormat | None,
        temperature: float | None,
        max_completion_tokens: int | None,
        max_tokens: int | None,
        metadata: Mapping[str, str] | None,
        extra_params: Mapping[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self._settings.model,
            "messages": list(messages),
        }

        merged_metadata = self._merge_metadata(metadata)
        if merged_metadata:
            payload["metadata"] = merged_metadata
        if tools:
            payload["tools"] = list(tools)
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format
        if temperature is not None:
            payload["temperature"] = temperature
        if max_completion_tokens is not None:
            payload["max_completion_tokens"] = max_completion_tokens
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_params:
            payload.update(extra_params)

        return payload

    def _merge_metadata(self, runtime_metadata: Mapping[str, str] | None) -> Dict[str, str] | None:
        combined: Dict[str, str] = {}
        if self._settings.metadata:
            combined.update(self._settings.metadata)
        if runtime_metadata:
            combined.update(runtime_metadata)
        return combined or None

    def _normalize_stream_event(self, event: ChatCompletionStreamEvent[Any]) -> AIStreamEvent | None:
        event_type = getattr(event, "type", None)
        if event_type is None:
            return None

        if event_type == "chunk":
            return None
        if event_type == "content.delta":
            delta_text = getattr(event, "delta", None)
            if delta_text:
                return AIStreamEvent(type=event_type, content=str(delta_text))
            return None
        if event_type == "content.done":
            return AIStreamEvent(
                type=event_type,
                content=getattr(event, "content", None),
                parsed=getattr(event, "parsed", None),
            )
        if event_type == "refusal.delta":
            return AIStreamEvent(type=event_type, content=getattr(event, "delta", None))
        if event_type == "refusal.done":
            return AIStreamEvent(type=event_type, content=getattr(event, "refusal", None))
        if event_type == "tool_calls.function.arguments.delta":
            return AIStreamEvent(
                type=event_type,
                tool_name=getattr(event, "name", None),
                tool_index=getattr(event, "index", None),
                tool_arguments=getattr(event, "arguments", None),
                arguments_delta=getattr(event, "arguments_delta", None),
                parsed=getattr(event, "parsed_arguments", None),
            )
        if event_type == "tool_calls.function.arguments.done":
            return AIStreamEvent(
                type=event_type,
                tool_name=getattr(event, "name", None),
                tool_index=getattr(event, "index", None),
                tool_arguments=getattr(event, "arguments", None),
                parsed=getattr(event, "parsed_arguments", None),
            )
        return None

