"""Tests for the OutlineRuntime helper."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tinkerbell.ui.outline_runtime import OutlineRuntime


@pytest.fixture()
def stub_worker_cls(monkeypatch: pytest.MonkeyPatch) -> type:
    """Provide a stub OutlineBuilderWorker implementation for tests."""

    class _StubWorker:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.loop = kwargs["loop"]
            self.memory = {"outline": "available"}

        def aclose(self) -> Any:
            async def _close() -> None:
                return None

            return _close()

    monkeypatch.setattr("tinkerbell.ui.outline_runtime.OutlineBuilderWorker", _StubWorker)
    return _StubWorker


def test_outline_runtime_starts_worker_when_loop_running(tmp_path: Path, stub_worker_cls: type) -> None:
    propagations: list[str] = []

    def _propagate() -> None:
        propagations.append("called")

    loop = SimpleNamespace(is_running=lambda: True)
    runtime = OutlineRuntime(
        document_provider=lambda _doc_id: None,
        storage_root=tmp_path,
        loop_resolver=lambda: cast(asyncio.AbstractEventLoop, loop),
        index_propagator=_propagate,
    )

    worker = runtime.ensure_started()

    assert isinstance(worker, stub_worker_cls)
    assert worker.kwargs["storage_dir"] == tmp_path
    assert propagations == ["called"]
    assert runtime.outline_memory() == {"outline": "available"}


def test_outline_runtime_starts_even_when_loop_idle(tmp_path: Path, stub_worker_cls: type) -> None:
    propagations: list[str] = []

    class _IdleLoop:
        def is_running(self) -> bool:
            return False

        def is_closed(self) -> bool:
            return False

        def call_soon(self, callback: Any, *args: Any) -> None:
            raise AssertionError("call_soon should not be used when loop is idle")

    loop = _IdleLoop()
    runtime = OutlineRuntime(
        document_provider=lambda _doc_id: None,
        storage_root=tmp_path,
        loop_resolver=lambda: cast(asyncio.AbstractEventLoop, loop),
        index_propagator=lambda: propagations.append("called"),
    )

    worker = runtime.ensure_started()

    assert isinstance(worker, stub_worker_cls)
    assert runtime.worker() is worker
    assert runtime.outline_memory() == {"outline": "available"}
    assert propagations == ["called"]


def test_outline_runtime_shutdown_handles_running_and_idle_loops(tmp_path: Path) -> None:
    close_calls: list[str] = []

    async def _close() -> None:
        close_calls.append("closed")

    runtime = OutlineRuntime(
        document_provider=lambda _doc_id: None,
        storage_root=tmp_path,
        loop_resolver=lambda: None,
        index_propagator=None,
    )

    tasks: list[Any] = []

    class _RunningLoop:
        def is_running(self) -> bool:
            return True

        def create_task(self, coro: Any) -> None:
            tasks.append(coro)

    runtime._worker = cast(Any, SimpleNamespace(loop=_RunningLoop(), aclose=lambda: _close(), memory=None))
    runtime.shutdown()

    assert tasks, "Expected running loop to schedule close coroutine"
    assert runtime.worker() is None
    asyncio.run(tasks.pop())
    assert close_calls == ["closed"]

    run_calls: list[Any] = []

    class _IdleLoop:
        def is_running(self) -> bool:
            return False

        def run_until_complete(self, coro: Any) -> None:
            run_calls.append(coro)
            asyncio.run(coro)

    runtime._worker = cast(Any, SimpleNamespace(loop=_IdleLoop(), aclose=lambda: _close(), memory=None))
    runtime.shutdown()

    assert run_calls, "Expected idle loop to run close coroutine synchronously"
    assert close_calls == ["closed", "closed"]
