"""Runtime helper that encapsulates the optional outline worker lifecycle."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..ai.services import OutlineBuilderWorker
from ..ai.memory.buffers import DocumentSummaryMemory

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OutlineRuntime:
    """Starts, tracks, and shuts down the OutlineBuilderWorker safely."""

    document_provider: Callable[[str], Any]
    storage_root: Path
    loop_resolver: Callable[[], asyncio.AbstractEventLoop | None]
    index_propagator: Callable[[], None] | None = None

    _worker: OutlineBuilderWorker | None = None

    def ensure_started(self) -> OutlineBuilderWorker | None:
        loop = self.loop_resolver()
        if loop is None:
            return None
        if self._worker is not None:
            return self._worker
        if loop.is_running():
            return self._start_worker(loop)
        try:
            loop.call_soon(self._start_worker, loop)
        except RuntimeError:  # pragma: no cover - loop may be closed in tests
            return None
        return None

    def _start_worker(self, loop: asyncio.AbstractEventLoop) -> OutlineBuilderWorker | None:
        if self._worker is not None:
            return self._worker
        try:
            worker = OutlineBuilderWorker(
                document_provider=self.document_provider,
                storage_dir=self.storage_root,
                loop=loop,
            )
        except Exception:  # pragma: no cover - optional feature
            LOGGER.debug("Outline worker unavailable; continuing without outlines.", exc_info=True)
            return None
        self._worker = worker
        if self.index_propagator is not None:
            try:
                self.index_propagator()
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.debug("Outline worker propagation callback failed", exc_info=True)
        return worker

    def outline_memory(self) -> DocumentSummaryMemory | None:
        worker = self._worker
        if worker is None:
            return None
        return getattr(worker, "memory", None)

    def worker(self) -> OutlineBuilderWorker | None:
        return self._worker

    def shutdown(self) -> None:
        worker = self._worker
        if worker is None:
            return
        self._worker = None
        close_coro = worker.aclose()
        loop = worker.loop
        if loop.is_running():
            loop.create_task(close_coro)
            return
        try:
            loop.run_until_complete(close_coro)
        except RuntimeError:
            asyncio.run(close_coro)


__all__ = ["OutlineRuntime"]
