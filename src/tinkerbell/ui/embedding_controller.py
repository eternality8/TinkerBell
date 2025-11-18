"""Embedding runtime controller extracted from the main window implementation."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Mapping, cast

from ..ai.memory import (
    DocumentEmbeddingIndex,
    EmbeddingProvider,
    LangChainEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from ..services.settings import Settings
from ..widgets.status_bar import StatusBar

LOGGER = logging.getLogger(__name__)

OPENAI_API_BASE_URL = "https://api.openai.com/v1"


@dataclass(slots=True)
class EmbeddingRuntimeState:
    """Bookkeeping structure describing the active embedding backend."""

    backend: str = "disabled"
    model: str | None = None
    provider_label: str | None = None
    status: str = "disabled"
    detail: str | None = None
    error: str | None = None

    @property
    def label(self) -> str:
        if self.status in {"disabled", "unavailable"}:
            return "Off" if self.status == "disabled" else "Unavailable"
        if self.status == "error":
            return "Error"
        return self.provider_label or self.backend.title()

    def as_snapshot(self) -> dict[str, Any]:
        return {
            "embedding_backend": self.backend if self.backend != "disabled" else None,
            "embedding_model": self.model,
            "embedding_status": self.status,
            "embedding_detail": self.detail or self.error,
        }


@dataclass(slots=True, frozen=True)
class _LangChainProviderTemplate:
    """Describes heuristics for provider-specific LangChain wiring."""

    family: str
    match_keywords: tuple[str, ...]
    default_base_url: str | None = None
    api_key_env: str | None = None
    tokenizer: str | None = None
    default_headers: Mapping[str, str] | None = None
    dimensions: Mapping[str, int] = field(default_factory=dict)
    extra_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def matches(self, model_name: str, *, forced_family: str | None = None) -> bool:
        if forced_family:
            return forced_family == self.family
        if not self.match_keywords:
            return True
        lowered = model_name.lower()
        return any(keyword in lowered for keyword in self.match_keywords)

    def dimension_for(self, model_name: str) -> int | None:
        lookup = self.dimensions or {}
        if not lookup:
            return None
        return lookup.get(model_name.lower())


_DEFAULT_LANGCHAIN_PROVIDER = _LangChainProviderTemplate(
    family="openai",
    match_keywords=("text-embedding", "openai", "ada"),
    default_base_url=OPENAI_API_BASE_URL,
    tokenizer="cl100k_base",
    dimensions={
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    },
)

_LANGCHAIN_PROVIDER_TEMPLATES: tuple[_LangChainProviderTemplate, ...] = (
    _DEFAULT_LANGCHAIN_PROVIDER,
    _LangChainProviderTemplate(
        family="deepseek",
        match_keywords=("deepseek",),
        default_base_url="https://api.deepseek.com/v1",
        api_key_env="DEEPSEEK_API_KEY",
        tokenizer="cl100k_base",
        dimensions={"deepseek-embedding": 1536},
    ),
    _LangChainProviderTemplate(
        family="glm",
        match_keywords=("glm", "zhipu"),
        default_base_url="https://open.bigmodel.cn/api/paas/v4",
        api_key_env="GLM_API_KEY",
        tokenizer="cl100k_base",
        dimensions={"glm-4-embed": 1024},
    ),
    _LangChainProviderTemplate(
        family="moonshot",
        match_keywords=("moonshot", "kimi"),
        default_base_url="https://api.moonshot.cn/v1",
        api_key_env="MOONSHOT_API_KEY",
        tokenizer="cl100k_base",
    ),
)

_LANGCHAIN_PROVIDER_ENV_VARS: tuple[str, ...] = tuple(
    sorted({template.api_key_env for template in _LANGCHAIN_PROVIDER_TEMPLATES if template.api_key_env})
)


class EmbeddingController:
    """Owns embedding runtime state shared across the UI."""

    def __init__(
        self,
        *,
        status_bar: StatusBar | None,
        cache_root_resolver: Callable[[], Path],
        outline_worker_resolver: Callable[[], Any | None],
        async_loop_resolver: Callable[[], asyncio.AbstractEventLoop | None],
        background_task_runner: Callable[[Callable[[], Coroutine[Any, Any, Any]]], None],
        phase3_outline_enabled: bool,
    ) -> None:
        self._status_bar = status_bar
        self._cache_root_resolver = cache_root_resolver
        self._outline_worker_resolver = outline_worker_resolver
        self._resolve_async_loop = async_loop_resolver
        self._run_background_task = background_task_runner
        self._phase3_outline_enabled = phase3_outline_enabled
        self._embedding_index: DocumentEmbeddingIndex | None = None
        self._embedding_state = EmbeddingRuntimeState()
        self._embedding_signature: tuple[Any, ...] | None = None
        self._embedding_snapshot_metadata: dict[str, Any] = {}
        self._embedding_resource: Any | None = None

    # ------------------------------------------------------------------
    # Life-cycle hooks
    # ------------------------------------------------------------------
    def set_phase3_outline_enabled(self, enabled: bool) -> None:
        if self._phase3_outline_enabled == enabled:
            return
        self._phase3_outline_enabled = enabled
        if not enabled:
            self._teardown_embedding_runtime(reason="Phase 3 tools disabled", hide_status=True)

    def refresh_runtime(self, settings: Settings | None) -> None:
        if settings is None or not self._phase3_outline_enabled:
            self._teardown_embedding_runtime(
                reason="Phase 3 tools disabled",
                hide_status=not self._phase3_outline_enabled,
            )
            return

        backend = (getattr(settings, "embedding_backend", "auto") or "auto").strip().lower()
        if backend in {"", "auto"}:
            backend = "openai"
        if backend == "disabled":
            self._teardown_embedding_runtime(reason="Embeddings disabled in settings")
            return

        model_name = (getattr(settings, "embedding_model_name", "") or "").strip() or "text-embedding-3-large"
        signature = self._embedding_settings_signature(settings, backend, model_name)
        if signature == self._embedding_signature and self._embedding_index is not None:
            self._set_embedding_state(self._embedding_state)
            return

        self._teardown_embedding_runtime(reason="Reinitializing embedding backend", keep_status=True)

        provider, provider_state = self._build_embedding_provider(backend, model_name, settings)
        if provider is None:
            self._set_embedding_state(provider_state)
            return

        cache_root = self._resolve_embedding_cache_root()
        index = self._create_embedding_index(provider, storage_dir=cache_root)
        if index is None:
            error_state = EmbeddingRuntimeState(
                backend=backend,
                model=model_name,
                provider_label=provider_state.provider_label,
                status="error",
                error=f"Unable to initialize embedding cache under {cache_root}",
            )
            self._set_embedding_state(error_state)
            return

        self._embedding_index = index
        self._embedding_signature = signature
        self._set_embedding_state(provider_state)
        self.propagate_index_to_worker()

    # ------------------------------------------------------------------
    # Public utility helpers
    # ------------------------------------------------------------------
    def resolve_index(self) -> DocumentEmbeddingIndex | None:
        return self._embedding_index

    def apply_snapshot_metadata(self, snapshot: dict[str, Any]) -> None:
        if not self._embedding_snapshot_metadata:
            for field in ("embedding_backend", "embedding_model", "embedding_status", "embedding_detail"):
                snapshot.pop(field, None)
            return
        snapshot.update(self._embedding_snapshot_metadata)

    def propagate_index_to_worker(self) -> None:
        worker = self._outline_worker_resolver()
        if worker is None:
            return
        updater = getattr(worker, "update_embedding_index", None)
        if callable(updater):
            try:
                updater(self._embedding_index)
            except Exception:
                LOGGER.debug("Failed to propagate embedding index to worker", exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_embedding_cache_root(self) -> Path:
        return self._cache_root_resolver()

    def _embedding_settings_signature(self, settings: Settings, backend: str, model: str) -> tuple[Any, ...]:
        metadata = getattr(settings, "metadata", {}) or {}
        class_override = metadata.get("langchain_embeddings_class")
        kwargs_payload = metadata.get("langchain_embeddings_kwargs")
        provider_hint = metadata.get("langchain_provider_family")
        env_class = os.environ.get("TINKERBELL_LANGCHAIN_EMBEDDINGS_CLASS")
        env_kwargs = os.environ.get("TINKERBELL_LANGCHAIN_EMBEDDINGS_KWARGS")
        env_provider_family = os.environ.get("TINKERBELL_LANGCHAIN_PROVIDER_FAMILY")
        provider_env_secrets = tuple(os.environ.get(var) for var in _LANGCHAIN_PROVIDER_ENV_VARS)
        return (
            backend,
            model,
            getattr(settings, "base_url", None),
            getattr(settings, "api_key", None),
            getattr(settings, "organization", None),
            class_override,
            kwargs_payload,
            env_class,
            env_kwargs,
            provider_hint,
            env_provider_family,
            provider_env_secrets,
        )

    def _build_embedding_provider(
        self,
        backend: str,
        model_name: str,
        settings: Settings,
    ) -> tuple[EmbeddingProvider | None, EmbeddingRuntimeState]:
        if backend == "openai":
            return self._build_openai_embedding_provider(model_name, settings)
        if backend == "langchain":
            return self._build_langchain_embedding_provider(model_name, settings)
        state = EmbeddingRuntimeState(
            backend=backend,
            model=model_name,
            status="error",
            error=f"Unknown backend '{backend}'",
        )
        return None, state

    def _build_openai_embedding_provider(
        self,
        model_name: str,
        settings: Settings,
    ) -> tuple[EmbeddingProvider | None, EmbeddingRuntimeState]:
        state = EmbeddingRuntimeState(backend="openai", model=model_name, provider_label="OpenAI", status="error")
        api_key = (getattr(settings, "api_key", "") or "").strip()
        base_url = (getattr(settings, "base_url", "") or "").strip() or OPENAI_API_BASE_URL
        if not api_key:
            state.error = "API key required for OpenAI embeddings"
            return None, state
        try:
            from openai import AsyncOpenAI
        except Exception as exc:  # pragma: no cover - optional dependency
            state.error = f"openai package unavailable: {exc}"
            return None, state

        try:
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                organization=getattr(settings, "organization", None),
                timeout=getattr(settings, "request_timeout", 90.0),
            )
            provider = OpenAIEmbeddingProvider(client=client, model=model_name)
        except Exception as exc:
            state.error = f"Failed to initialize OpenAI embeddings: {exc}"
            return None, state

        state.status = "ready"
        state.detail = model_name
        self._embedding_resource = client
        return provider, state

    def _build_langchain_embedding_provider(
        self,
        model_name: str,
        settings: Settings,
    ) -> tuple[EmbeddingProvider | None, EmbeddingRuntimeState]:
        state = EmbeddingRuntimeState(backend="langchain", model=model_name, provider_label="LangChain")
        try:
            embeddings, provider_template = self._instantiate_langchain_embeddings(model_name, settings)
        except Exception as exc:
            state.status = "error"
            state.error = str(exc)
            return None, state
        provider = LangChainEmbeddingProvider(embeddings=embeddings)
        label = getattr(embeddings, "__class__", type(embeddings)).__name__
        template_label = provider_template.family.title() if provider_template else label
        state.provider_label = f"LangChain/{template_label}"
        model_detail = getattr(embeddings, "model", None) or getattr(embeddings, "model_name", None)
        state.detail = str(model_detail or model_name)
        state.status = "ready"
        return provider, state

    def _instantiate_langchain_embeddings(self, model_name: str, settings: Settings) -> tuple[Any, _LangChainProviderTemplate]:
        metadata = getattr(settings, "metadata", {}) or {}
        class_override = metadata.get("langchain_embeddings_class") or os.environ.get("TINKERBELL_LANGCHAIN_EMBEDDINGS_CLASS")
        raw_kwargs = metadata.get("langchain_embeddings_kwargs") or os.environ.get("TINKERBELL_LANGCHAIN_EMBEDDINGS_KWARGS")
        extra_kwargs = self._parse_langchain_kwargs(raw_kwargs)
        provider_template = self._detect_langchain_provider_template(model_name, metadata)
        params = dict(extra_kwargs)
        params.setdefault("model", model_name)
        self._apply_langchain_provider_template(params, provider_template, model_name, settings, metadata)
        if class_override:
            module_name, _, attr_name = class_override.rpartition(".")
            if not module_name or not attr_name:
                raise RuntimeError(
                    "LangChain embeddings class override must include module path, e.g., 'langchain_openai.OpenAIEmbeddings'",
                )
            module = importlib.import_module(module_name)
            factory = getattr(module, attr_name)
            embeddings = factory(**params)
            return embeddings, provider_template
        try:
            from langchain_openai import OpenAIEmbeddings  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "langchain-openai package is required for the LangChain embedding backend. "
                "Install it or set metadata['langchain_embeddings_class'] to a custom implementation.",
            ) from exc

        embeddings = OpenAIEmbeddings(**params)
        return embeddings, provider_template

    def _detect_langchain_provider_template(
        self,
        model_name: str,
        metadata: Mapping[str, Any],
    ) -> _LangChainProviderTemplate:
        forced = str(
            metadata.get("langchain_provider_family")
            or os.environ.get("TINKERBELL_LANGCHAIN_PROVIDER_FAMILY")
            or ""
        ).strip().lower()
        forced_family = forced or None
        for template in _LANGCHAIN_PROVIDER_TEMPLATES:
            if template.matches(model_name, forced_family=forced_family):
                return template
        return _DEFAULT_LANGCHAIN_PROVIDER

    def _apply_langchain_provider_template(
        self,
        params: dict[str, Any],
        template: _LangChainProviderTemplate,
        model_name: str,
        settings: Settings,
        metadata: Mapping[str, Any],
    ) -> None:
        metadata_base_override = str(metadata.get(f"{template.family}_base_url") or "").strip()
        configured_base_url = metadata_base_override or (getattr(settings, "base_url", "") or "").strip()
        if template.family != "openai" and template.default_base_url:
            if self._should_override_base_url(configured_base_url, template.default_base_url):
                params.setdefault("base_url", template.default_base_url)
            elif configured_base_url:
                params.setdefault("base_url", configured_base_url)
            else:
                params.setdefault("base_url", template.default_base_url)
        else:
            if configured_base_url:
                params.setdefault("base_url", configured_base_url)
            elif template.default_base_url:
                params.setdefault("base_url", template.default_base_url)

        metadata_key = metadata.get(f"{template.family}_api_key")
        api_key = str(metadata_key).strip() if isinstance(metadata_key, str) else ""
        if not api_key:
            api_key = (getattr(settings, "api_key", "") or "").strip()
        if not api_key and template.api_key_env:
            api_key = (os.environ.get(template.api_key_env, "") or "").strip()
        if api_key:
            params.setdefault("api_key", api_key)

        organization = getattr(settings, "organization", None)
        if organization:
            params.setdefault("organization", organization)

        headers: dict[str, str] = {}
        if template.default_headers:
            headers.update(template.default_headers)
        user_headers = getattr(settings, "default_headers", None) or {}
        headers.update({str(k): str(v) for k, v in user_headers.items()})
        if headers:
            params.setdefault("default_headers", headers)

        if template.tokenizer:
            params.setdefault("tiktoken_model_name", template.tokenizer)

        dimension_override = template.dimension_for(model_name)
        if dimension_override:
            params.setdefault("dimensions", dimension_override)

        for key, value in (template.extra_kwargs or {}).items():
            params.setdefault(key, value)

    @staticmethod
    def _should_override_base_url(current: str, candidate: str) -> bool:
        normalized_current = (current or "").strip().lower()
        normalized_candidate = (candidate or "").strip().lower()
        if not normalized_candidate:
            return False
        if not normalized_current:
            return True
        openai_default = OPENAI_API_BASE_URL.lower()
        return normalized_current == openai_default and normalized_candidate != openai_default

    @staticmethod
    def _parse_langchain_kwargs(payload: object) -> dict[str, Any]:
        if not payload:
            return {}
        if isinstance(payload, Mapping):
            return dict(payload)
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return {}
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return {}
            return data if isinstance(data, dict) else {}
        return {}

    def _create_embedding_index(self, provider: EmbeddingProvider, *, storage_dir: Path) -> DocumentEmbeddingIndex | None:
        try:
            loop = self._resolve_async_loop()
            if loop is not None:
                return DocumentEmbeddingIndex(storage_dir=storage_dir, provider=provider, loop=loop)
            return DocumentEmbeddingIndex(storage_dir=storage_dir, provider=provider)
        except Exception as exc:
            LOGGER.warning("Failed to initialize embedding index: %s", exc)
            return None

    def _teardown_embedding_runtime(
        self,
        *,
        reason: str,
        hide_status: bool = False,
        keep_status: bool = False,
    ) -> None:
        if self._embedding_index is not None:
            self._close_embedding_index(self._embedding_index)
        self._embedding_index = None
        self._embedding_signature = None
        self._embedding_snapshot_metadata = {}
        self.propagate_index_to_worker()
        self._dispose_embedding_resource()
        if keep_status:
            return
        fallback_backend = "disabled" if self._phase3_outline_enabled else "unavailable"
        state = EmbeddingRuntimeState(
            backend=fallback_backend,
            model=None,
            status=fallback_backend if fallback_backend != "unavailable" else "unavailable",
            detail=reason,
        )
        self._set_embedding_state(state, hide_status=hide_status)

    def _close_embedding_index(self, index: DocumentEmbeddingIndex) -> None:
        async_close = cast(Callable[[], Coroutine[Any, Any, Any]], index.aclose)
        self._run_background_task(async_close)

    def _dispose_embedding_resource(self) -> None:
        resource = self._embedding_resource
        self._embedding_resource = None
        if resource is None:
            return
        close = getattr(resource, "aclose", None)
        if callable(close):
            try:
                async_close = cast(Callable[[], Coroutine[Any, Any, Any]], close)
                self._run_background_task(async_close)
            except Exception:
                LOGGER.debug("Failed to dispose async embedding resource", exc_info=True)
            return
        close = getattr(resource, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

    def _set_embedding_state(self, state: EmbeddingRuntimeState, *, hide_status: bool = False) -> None:
        self._embedding_state = state
        snapshot = {k: v for k, v in state.as_snapshot().items() if v not in (None, "")}
        self._embedding_snapshot_metadata = snapshot
        label = "" if hide_status else state.label
        detail = state.detail or state.error or ""
        try:
            if self._status_bar is not None:
                self._status_bar.set_embedding_status(label, detail=detail)
        except Exception:  # pragma: no cover - defensive guard
            pass


__all__ = ["EmbeddingController", "EmbeddingRuntimeState", "OPENAI_API_BASE_URL"]
