"""Embedding runtime controller extracted from the main window implementation."""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
from typing import Any, Callable, Coroutine, Mapping, Sequence, cast

from ..ai.memory import (
    DocumentEmbeddingIndex,
    EmbeddingProvider,
    LangChainEmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
)
from ..services.settings import DEFAULT_EMBEDDING_MODE, EMBEDDING_MODE_CHOICES, Settings
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
    mode: str = DEFAULT_EMBEDDING_MODE

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
            "embedding_mode": self.mode,
        }


@dataclass(slots=True)
class EmbeddingValidationResult:
    status: str
    detail: str | None = None
    error: str | None = None


class EmbeddingValidator:
    """Lightweight helper that validates embedding providers asynchronously."""

    def __init__(self, *, sample_text: str = "TinkerBell embeddings validation ping", timeout: float = 15.0) -> None:
        self._sample_text = sample_text
        self._timeout = timeout

    async def validate(self, provider: EmbeddingProvider, *, mode: str) -> EmbeddingValidationResult:
        async def _run() -> None:
            if mode == "local":
                await provider.embed_documents([self._sample_text])
            else:
                await provider.embed_query(self._sample_text)

        try:
            await asyncio.wait_for(_run(), timeout=self._timeout)
        except Exception as exc:
            LOGGER.warning("Embedding validation failed (%s backend): %s", mode, exc)
            return EmbeddingValidationResult(status="error", error=f"Validation failed: {exc}")
        return EmbeddingValidationResult(status="ready", detail="Validated")


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
        self._validator = EmbeddingValidator()

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

        metadata = getattr(settings, "metadata", {}) or {}
        mode = self._resolve_embedding_mode(metadata)
        backend = self._normalize_backend(getattr(settings, "embedding_backend", "auto"), mode)
        if backend == "disabled":
            self._teardown_embedding_runtime(reason="Embeddings disabled in settings")
            return

        model_name = (getattr(settings, "embedding_model_name", "") or "").strip() or "text-embedding-3-large"
        embedding_settings = self._build_embedding_settings(settings, metadata, mode)
        signature = self._embedding_settings_signature(embedding_settings, backend, model_name, metadata, mode)
        if signature == self._embedding_signature and self._embedding_index is not None:
            self._set_embedding_state(self._embedding_state)
            return

        self._teardown_embedding_runtime(reason="Reinitializing embedding backend", keep_status=True)

        provider, provider_state = self._build_embedding_provider(
            backend,
            model_name,
            embedding_settings,
            metadata,
            mode,
        )
        if provider is None:
            self._set_embedding_state(provider_state)
            return

        cache_root = self._resolve_embedding_cache_root()
        index = self._create_embedding_index(
            provider,
            storage_dir=cache_root,
            mode=mode,
            provider_label=provider_state.provider_label,
        )
        if index is None:
            error_state = EmbeddingRuntimeState(
                backend=backend,
                model=model_name,
                provider_label=provider_state.provider_label,
                status="error",
                error=f"Unable to initialize embedding cache under {cache_root}",
                mode=mode,
            )
            self._set_embedding_state(error_state)
            return

        self._embedding_index = index
        self._embedding_signature = signature
        self._set_embedding_state(provider_state)
        self.propagate_index_to_worker()
        self._schedule_validation(provider, mode=mode, signature=signature)

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

    def _resolve_embedding_mode(self, metadata: Mapping[str, Any]) -> str:
        raw_value = str(metadata.get("embedding_mode") or DEFAULT_EMBEDDING_MODE).strip().lower()
        if raw_value in EMBEDDING_MODE_CHOICES:
            return raw_value
        return DEFAULT_EMBEDDING_MODE

    def _normalize_backend(self, backend: str | None, mode: str) -> str:
        normalized = (backend or "auto").strip().lower()
        if normalized in {"", "auto"}:
            normalized = "openai"
        if mode == "local":
            return "sentence-transformers"
        if normalized == "sentence-transformers" and mode != "local":
            return "openai"
        return normalized

    def _build_embedding_settings(self, settings: Settings, metadata: Mapping[str, Any], mode: str) -> Settings:
        if mode != "custom-api":
            return settings
        payload = metadata.get("embedding_api")
        if not isinstance(payload, Mapping):
            return settings
        overrides: dict[str, Any] = {}
        for field_name in (
            "base_url",
            "api_key",
            "organization",
            "request_timeout",
            "max_retries",
            "retry_min_seconds",
            "retry_max_seconds",
        ):
            value = payload.get(field_name)
            if value in (None, ""):
                continue
            try:
                if field_name in {"request_timeout", "retry_min_seconds", "retry_max_seconds"}:
                    overrides[field_name] = float(value)
                elif field_name == "max_retries":
                    overrides[field_name] = int(value)
                else:
                    overrides[field_name] = value
            except (TypeError, ValueError):
                continue
        headers = payload.get("default_headers")
        if isinstance(headers, Mapping) and headers:
            normalized_headers = {str(k): str(v) for k, v in headers.items()}
            overrides["default_headers"] = normalized_headers
        if not overrides:
            return settings
        return replace(settings, **overrides)

    def _embedding_settings_signature(
        self,
        settings: Settings,
        backend: str,
        model: str,
        metadata: Mapping[str, Any],
        mode: str,
    ) -> tuple[Any, ...]:
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
            mode,
            self._signature_for_embedding_api(metadata),
            self._signature_for_sentence_transformers(metadata),
        )

    @staticmethod
    def _signature_for_embedding_api(metadata: Mapping[str, Any]) -> str | None:
        payload = metadata.get("embedding_api")
        if not isinstance(payload, Mapping):
            return None
        filtered: dict[str, Any] = {}
        secret = str(payload.get("api_key") or "").strip()
        if secret:
            filtered["api_key_fingerprint"] = hashlib.sha256(secret.encode("utf-8")).hexdigest()
        for key in ("base_url", "organization", "request_timeout", "max_retries", "retry_min_seconds", "retry_max_seconds"):
            value = payload.get(key)
            if value not in (None, ""):
                filtered[key] = value
        headers = payload.get("default_headers")
        if isinstance(headers, Mapping) and headers:
            filtered["default_headers"] = {str(k): str(v) for k, v in sorted(headers.items(), key=lambda item: str(item[0]))}
        if not filtered:
            return None
        return json.dumps(filtered, sort_keys=True, default=str)

    @staticmethod
    def _signature_for_sentence_transformers(metadata: Mapping[str, Any]) -> str | None:
        keys = ("st_model_path", "st_device", "st_dtype", "st_cache_dir", "st_batch_size")
        filtered = {key: metadata.get(key) for key in keys if metadata.get(key) not in (None, "")}
        if not filtered:
            return None
        return json.dumps(filtered, sort_keys=True, default=str)

    def _build_embedding_provider(
        self,
        backend: str,
        model_name: str,
        settings: Settings,
        metadata: Mapping[str, Any],
        mode: str,
    ) -> tuple[EmbeddingProvider | None, EmbeddingRuntimeState]:
        if backend == "openai":
            return self._build_openai_embedding_provider(model_name, settings, mode)
        if backend == "langchain":
            return self._build_langchain_embedding_provider(model_name, settings, metadata, mode)
        if backend == "sentence-transformers":
            return self._build_sentence_transformer_provider(model_name, metadata, mode)
        state = EmbeddingRuntimeState(
            backend=backend,
            model=model_name,
            status="error",
            error=f"Unknown backend '{backend}'",
            mode=mode,
        )
        return None, state

    def _build_openai_embedding_provider(
        self,
        model_name: str,
        settings: Settings,
        mode: str,
    ) -> tuple[EmbeddingProvider | None, EmbeddingRuntimeState]:
        label_suffix = "Shared" if mode == "same-api" else "Custom API"
        state = EmbeddingRuntimeState(
            backend="openai",
            model=model_name,
            provider_label=f"OpenAI/{label_suffix}",
            status="error",
            mode=mode,
        )
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
        metadata: Mapping[str, Any],
        mode: str,
    ) -> tuple[EmbeddingProvider | None, EmbeddingRuntimeState]:
        state = EmbeddingRuntimeState(backend="langchain", model=model_name, provider_label="LangChain", mode=mode)
        try:
            embeddings, provider_template = self._instantiate_langchain_embeddings(model_name, settings, metadata)
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

    def _build_sentence_transformer_provider(
        self,
        model_name: str,
        metadata: Mapping[str, Any],
        mode: str,
    ) -> tuple[EmbeddingProvider | None, EmbeddingRuntimeState]:
        state = EmbeddingRuntimeState(
            backend="sentence-transformers",
            model=model_name,
            provider_label="SentenceTransformers",
            mode=mode,
        )
        target_model = str(metadata.get("st_model_path") or model_name).strip()
        if not target_model:
            state.status = "error"
            state.error = "metadata.st_model_path is required for local embeddings"
            return None, state
        batch_override = metadata.get("st_batch_size")
        try:
            batch_size = max(1, int(batch_override)) if batch_override is not None else 8
        except (TypeError, ValueError):
            state.status = "error"
            state.error = "metadata.st_batch_size must be an integer"
            return None, state
        cache_dir = str(metadata.get("st_cache_dir") or "").strip() or None
        raw_device = str(metadata.get("st_device") or "auto").strip()
        device_arg = None if raw_device in {"", "auto"} else raw_device
        dtype_name = str(metadata.get("st_dtype") or "").strip()
        model_kwargs: dict[str, Any] | None = None
        if dtype_name:
            try:
                torch_module = importlib.import_module("torch")
            except Exception as exc:
                state.status = "error"
                state.error = f"torch is required when st_dtype is set: {exc}"
                return None, state
            dtype_value = getattr(torch_module, dtype_name, None)
            if dtype_value is None:
                state.status = "error"
                state.error = f"Unknown st_dtype '{dtype_name}'"
                return None, state
            model_kwargs = {"dtype": dtype_value}
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            state.status = "error"
            state.error = (
                "sentence-transformers is required for local embeddings. Install the 'embeddings' extra."
                f" ({exc})"
            )
            return None, state

        try:
            model = SentenceTransformer(
                target_model,
                device=device_arg,
                cache_folder=cache_dir,
                model_kwargs=model_kwargs,
            )
        except Exception as exc:
            state.status = "error"
            state.error = f"Failed to load SentenceTransformer model: {exc}"
            return None, state

        async def _encode_batch(texts: Sequence[str]) -> list[list[float]]:
            inputs = list(texts)
            if not inputs:
                return []
            vectors = await asyncio.to_thread(
                model.encode,
                inputs,
                batch_size=batch_size,
                convert_to_numpy=True,
                device=device_arg,
            )
            return self._coerce_local_vectors(vectors)

        async def _encode_query(text: str) -> list[float]:
            batch = await _encode_batch([text])
            return batch[0]

        provider = LocalEmbeddingProvider(
            embed_batch=_encode_batch,
            embed_query=_encode_query,
            name=f"sentence-transformers:{Path(target_model).name or target_model}",
            max_batch_size=batch_size,
        )
        self._embedding_resource = model
        state.status = "ready"
        state.detail = Path(target_model).name or target_model
        return provider, state

    def _instantiate_langchain_embeddings(
        self,
        model_name: str,
        settings: Settings,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[Any, _LangChainProviderTemplate]:
        metadata_payload = metadata or getattr(settings, "metadata", {}) or {}
        class_override = metadata_payload.get("langchain_embeddings_class") or os.environ.get(
            "TINKERBELL_LANGCHAIN_EMBEDDINGS_CLASS"
        )
        raw_kwargs = metadata_payload.get("langchain_embeddings_kwargs") or os.environ.get(
            "TINKERBELL_LANGCHAIN_EMBEDDINGS_KWARGS"
        )
        extra_kwargs = self._parse_langchain_kwargs(raw_kwargs)
        provider_template = self._detect_langchain_provider_template(model_name, metadata_payload)
        params = dict(extra_kwargs)
        params.setdefault("model", model_name)
        self._apply_langchain_provider_template(params, provider_template, model_name, settings, metadata_payload)
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

    @staticmethod
    def _coerce_local_vectors(batch: Any) -> list[list[float]]:
        payload = batch.tolist() if hasattr(batch, "tolist") else batch
        if not isinstance(payload, Sequence):
            raise TypeError("Embedding batch must be a sequence")
        normalized: list[list[float]] = []
        for vector in payload:
            candidate = vector.tolist() if hasattr(vector, "tolist") else vector
            if not isinstance(candidate, Sequence):
                raise TypeError("Embedding vector must be a sequence")
            normalized.append([float(component) for component in candidate])
        return normalized

    def _create_embedding_index(
        self,
        provider: EmbeddingProvider,
        *,
        storage_dir: Path,
        mode: str | None = None,
        provider_label: str | None = None,
    ) -> DocumentEmbeddingIndex | None:
        try:
            loop = self._resolve_async_loop()
            if loop is not None:
                return DocumentEmbeddingIndex(
                    storage_dir=storage_dir,
                    provider=provider,
                    loop=loop,
                    mode=mode,
                    provider_label=provider_label,
                    activity_callback=self._handle_embedding_activity,
                )
            return DocumentEmbeddingIndex(
                storage_dir=storage_dir,
                provider=provider,
                mode=mode,
                provider_label=provider_label,
                activity_callback=self._handle_embedding_activity,
            )
        except Exception as exc:
            LOGGER.warning("Failed to initialize embedding index: %s", exc)
            return None

    def _schedule_validation(self, provider: EmbeddingProvider, *, mode: str, signature: tuple[Any, ...]) -> None:
        async def _run() -> None:
            result = await self._validator.validate(provider, mode=mode)
            if self._embedding_signature != signature:
                return
            self._apply_validation_result(result)

        self._run_background_task(_run)

    def _apply_validation_result(self, result: EmbeddingValidationResult) -> None:
        current = self._embedding_state
        if current.backend in {"disabled", "unavailable"}:
            return
        success = result.status == "ready"
        detail = current.detail
        error = current.error
        status = current.status
        if success:
            status = "ready"
            detail = result.detail or detail or "Validated"
            error = None
        else:
            status = "error"
            error = result.error or "Embedding validation failed"
        updated = EmbeddingRuntimeState(
            backend=current.backend,
            model=current.model,
            provider_label=current.provider_label,
            status=status,
            detail=detail,
            error=error,
            mode=current.mode,
        )
        self._set_embedding_state(updated)

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
        self._handle_embedding_activity(False, None)
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

    def _handle_embedding_activity(self, active: bool, detail: str | None) -> None:
        status_bar = self._status_bar
        if status_bar is None:
            return
        setter = getattr(status_bar, "set_embedding_processing", None)
        if not callable(setter):
            return
        try:
            setter(active, detail=detail)
        except Exception:
            LOGGER.debug("Failed to update embedding activity indicator", exc_info=True)

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


__all__ = ["EmbeddingController", "EmbeddingRuntimeState", "EmbeddingValidator", "OPENAI_API_BASE_URL"]
