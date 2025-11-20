"""Tests for the embedding controller's SentenceTransformers wiring."""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tinkerbell.ai.memory.embeddings import LocalEmbeddingProvider
from tinkerbell.ui.embedding_controller import EmbeddingController


def _build_controller(tmp_path: Path) -> EmbeddingController:
    return EmbeddingController(
        status_bar=None,
        cache_root_resolver=lambda: tmp_path,
        outline_worker_resolver=lambda: None,
        async_loop_resolver=lambda: asyncio.get_event_loop(),
        background_task_runner=lambda task_factory: None,
        phase3_outline_enabled=True,
    )


@pytest.mark.asyncio
async def test_sentence_transformer_provider_builds_local_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _build_controller(tmp_path)
    created_models: list[SimpleNamespace] = []

    class _FakeSentenceTransformer:
        def __init__(self, model_id: str, *, device=None, cache_folder=None, model_kwargs=None) -> None:
            self.model_id = model_id
            self.kwargs = {
                "device": device,
                "cache_folder": cache_folder,
                "model_kwargs": model_kwargs,
            }
            created_models.append(self)

        def encode(self, inputs, *, batch_size: int | None = None, convert_to_numpy: bool = True, device=None):
            return [[float(len(text))] for text in inputs]

    fake_module = SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    real_import_module = importlib.import_module

    def _fake_import_module(name: str, package: str | None = None):
        if name == "torch":
            return SimpleNamespace(float16="torch-float16")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    metadata = {
        "st_model_path": str(tmp_path / "models" / "mini-model"),
        "st_batch_size": 4,
        "st_device": "cpu",
        "st_cache_dir": str(tmp_path / "cache"),
        "st_dtype": "float16",
    }

    provider, state = controller._build_sentence_transformer_provider("text-embedding-3-large", metadata, "local")

    assert isinstance(provider, LocalEmbeddingProvider)
    assert state.status == "ready"
    assert state.detail == "mini-model"
    assert controller._embedding_resource is created_models[0]
    model_kwargs = created_models[0].kwargs
    assert model_kwargs["device"] == "cpu"
    assert Path(model_kwargs["cache_folder"]).name == "cache"
    assert model_kwargs["model_kwargs"] == {"torch_dtype": "torch-float16"}

    vectors = await provider.embed_documents(["alpha", "delta foxtrot"])
    assert vectors == [[5.0], [13.0]]


def test_sentence_transformer_provider_requires_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    controller = _build_controller(tmp_path)
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "sentence_transformers":
            raise ImportError("module missing")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    provider, state = controller._build_sentence_transformer_provider(
        "text-embedding-3-large",
        {"st_model_path": "hf/model"},
        "local",
    )

    assert provider is None
    assert state.status == "error"
    assert "sentence-transformers" in (state.error or "").lower()


def test_sentence_transformer_provider_rejects_unknown_dtype(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _build_controller(tmp_path)
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda *args, **kwargs: None))
    real_import_module = importlib.import_module

    def _fake_import_module(name: str, package: str | None = None):
        if name == "torch":
            return SimpleNamespace()
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    provider, state = controller._build_sentence_transformer_provider(
        "text-embedding-3-large",
        {"st_model_path": "hf/model", "st_dtype": "float128"},
        "local",
    )

    assert provider is None
    assert state.status == "error"
    assert "Unknown st_dtype" in (state.error or "")
