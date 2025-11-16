"""Async AI client wrapper built around OpenAI-compatible endpoints."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
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

try:  # pragma: no cover - optional dependency used when installed
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional fallback when package missing
    tiktoken = None

from .ai_types import TokenCounterProtocol

LOGGER = logging.getLogger(__name__)
_DEFAULT_BYTES_PER_TOKEN = 4
_TIKTOKEN_WARNING_EMITTED = False


class ApproxByteCounter(TokenCounterProtocol):
    """Deterministic fallback counter that estimates tokens via byte length."""

    def __init__(self, *, model_name: str | None = None, charset: str = "utf-8", bytes_per_token: int = _DEFAULT_BYTES_PER_TOKEN) -> None:
        self.model_name = model_name
        self._charset = charset
        self._bytes_per_token = max(1, int(bytes_per_token))

    def count(self, text: str) -> int:
        return self.estimate(text)

    def estimate(self, text: str) -> int:
        if not text:
            return 0
        data = text.encode(self._charset, errors="ignore")
        return max(1, math.ceil(len(data) / self._bytes_per_token))


class TiktokenCounter(TokenCounterProtocol):
    """Token counter backed by OpenAI's tiktoken package."""

    def __init__(self, model_name: str, *, encoding_name: str | None = None) -> None:
        if not model_name:
            raise ValueError("model_name is required for TiktokenCounter")
        if tiktoken is None:  # pragma: no cover - depends on optional dependency
            raise RuntimeError("tiktoken is not installed")
        self.model_name = model_name
        self._encoding = self._load_encoding(model_name, encoding_name)
        self._fallback = ApproxByteCounter(model_name=model_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self._encoding.encode(text))
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("tiktoken encode failed; falling back to approximation", exc_info=True)
            return self._fallback.estimate(text)

    def estimate(self, text: str) -> int:
        return self._fallback.estimate(text)

    def _load_encoding(self, model_name: str, encoding_name: str | None):
        token_module = cast(Any, tiktoken)
        try:
            if encoding_name:
                return token_module.get_encoding(encoding_name)
            return token_module.encoding_for_model(model_name)
        except Exception:  # pragma: no cover - falls back to default encoding
            LOGGER.debug("Falling back to cl100k_base encoding for model %s", model_name)
            return token_module.get_encoding("cl100k_base")


class TokenCounterRegistry:
    """Registry maintaining tokenizer implementations per model."""

    _shared: TokenCounterRegistry | None = None

    def __init__(self, *, fallback: TokenCounterProtocol | None = None) -> None:
        self._fallback = fallback or ApproxByteCounter()
        self._counters: Dict[str, TokenCounterProtocol] = {}

    @classmethod
    def global_instance(cls) -> "TokenCounterRegistry":
        if cls._shared is None:
            cls._shared = TokenCounterRegistry()
        return cls._shared

    def register(self, model_name: str, counter: TokenCounterProtocol) -> None:
        key = self._normalize_key(model_name)
        if not key:
            raise ValueError("model_name is required for token counter registration")
        self._counters[key] = counter

    def unregister(self, model_name: str) -> None:
        key = self._normalize_key(model_name)
        if key in self._counters:
            self._counters.pop(key)

    def has(self, model_name: str | None) -> bool:
        key = self._normalize_key(model_name)
        return bool(key and key in self._counters)

    def get(self, model_name: str | None = None) -> TokenCounterProtocol:
        key = self._normalize_key(model_name)
        if key and key in self._counters:
            return self._counters[key]
        return self._fallback

    def count(self, model_name: str | None, text: str) -> int:
        counter = self.get(model_name)
        try:
            return counter.count(text)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Token counter failed; falling back to estimate", exc_info=True)
            return counter.estimate(text)

    def estimate(self, text: str) -> int:
        return self._fallback.estimate(text)

    @staticmethod
    def _normalize_key(model_name: str | None) -> str:
        return (model_name or "").strip().lower()


def _log_tiktoken_warning_once() -> None:
    global _TIKTOKEN_WARNING_EMITTED
    if _TIKTOKEN_WARNING_EMITTED:
        return
    _TIKTOKEN_WARNING_EMITTED = True
    LOGGER.warning(
        "tiktoken is not installed; using approximate byte counter for token estimates. Install the optional "
        "[ai_tokenizers] dependency group for exact counts."
    )


@dataclass(slots=True)
class ClientSettings:
    """Subset of settings required to configure the AI client."""

    base_url: str
    api_key: str
    model: str
    organization: str | None = None
    request_timeout: float | None = 90.0
    max_retries: int = 3
    retry_min_seconds: float = 0.5
    retry_max_seconds: float = 6.0
    default_headers: Mapping[str, str] | None = None
    metadata: Mapping[str, str] | None = None
    debug_logging: bool = False


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
    tool_call_id: str | None = None


class AIClient:
    """Async client providing streaming helpers with retry semantics."""

    def __init__(
        self,
        settings: ClientSettings,
        *,
        client: AsyncOpenAI | None = None,
        token_registry: TokenCounterRegistry | None = None,
    ) -> None:
        self._settings = settings
        self._client = client or self._build_client(settings)
        self._models_cache: List[str] | None = None
        self._models_lock = asyncio.Lock()
        self._token_registry = token_registry or TokenCounterRegistry.global_instance()
        self._register_default_token_counter()

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
        if self._settings.debug_logging:
            self._log_prompt_payload(payload)

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

    def get_token_counter(self, model: str | None = None) -> TokenCounterProtocol:
        model_name = model or self._settings.model
        try:
            return self._token_registry.get(model_name)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Token registry returned no counter; using fallback")
            return ApproxByteCounter(model_name=model_name)

    def count_tokens(self, text: str, *, model: str | None = None, estimate_only: bool = False) -> int:
        if not text:
            return 0
        counter = self.get_token_counter(model)
        if estimate_only:
            return counter.estimate(text)
        try:
            return counter.count(text)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("count_tokens failed; falling back to estimate", exc_info=True)
            return counter.estimate(text)

    def _register_default_token_counter(self) -> None:
        model_name = (self._settings.model or "").strip()
        if not model_name or self._token_registry.has(model_name):
            return
        counter = self._build_token_counter(model_name)
        if counter is None:
            return
        try:
            self._token_registry.register(model_name, counter)
        except ValueError:
            LOGGER.debug("Unable to register token counter for model %s", model_name)

    def _build_token_counter(self, model_name: str) -> TokenCounterProtocol | None:
        if not model_name:
            return None
        if tiktoken is None:
            _log_tiktoken_warning_once()
            return ApproxByteCounter(model_name=model_name)
        try:
            return TiktokenCounter(model_name)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.debug("Failed to initialize tiktoken counter for %s: %s", model_name, exc)
            return ApproxByteCounter(model_name=model_name)

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
                tool_call_id=getattr(event, "id", None)
                or getattr(event, "tool_call_id", None),
            )
        if event_type == "tool_calls.function.arguments.done":
            return AIStreamEvent(
                type=event_type,
                tool_name=getattr(event, "name", None),
                tool_index=getattr(event, "index", None),
                tool_arguments=getattr(event, "arguments", None),
                parsed=getattr(event, "parsed_arguments", None),
                tool_call_id=getattr(event, "id", None)
                or getattr(event, "tool_call_id", None),
            )
        return None

    def _log_prompt_payload(self, payload: Mapping[str, Any]) -> None:
        try:
            serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            LOGGER.debug("AI prompt payload (unserializable): %s", payload)
        else:
            LOGGER.debug("AI prompt payload:\n%s", serialized)

    async def aclose(self) -> None:
        """Close the underlying OpenAI client to release network resources."""

        close = getattr(self._client, "close", None)
        if close is None:
            return
        try:
            result = close()
        except Exception as exc:  # pragma: no cover - defensive guard
            LOGGER.debug("AI client close failed to start: %s", exc)
            return
        if inspect.isawaitable(result):
            await result

