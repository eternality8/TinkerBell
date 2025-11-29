"""Outline store for managing document outline state.

This module provides the OutlineStore domain manager, which handles
the lifecycle of the OutlineBuilderWorker for generating document
outlines and summaries.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ..events import EventBus
    from ...ai.services import OutlineBuilderWorker
    from ...ai.memory.buffers import DocumentSummaryMemory

LOGGER = logging.getLogger(__name__)


class OutlineStore:
    """Domain manager for document outline runtime.

    The OutlineStore manages the lifecycle of the OutlineBuilderWorker,
    which generates document outlines and summaries. It provides a clean
    interface for starting, accessing, and shutting down the worker.

    Attributes:
        worker: The active OutlineBuilderWorker, or None if not started.
        memory: The DocumentSummaryMemory from the worker, or None.
    """

    __slots__ = (
        "_event_bus",
        "_document_provider",
        "_storage_root",
        "_loop_resolver",
        "_worker",
        "_index_propagator",
    )

    def __init__(
        self,
        document_provider: Callable[[str], Any],
        storage_root: Path,
        loop_resolver: Callable[[], asyncio.AbstractEventLoop | None],
        event_bus: EventBus,
        *,
        index_propagator: Callable[[], None] | None = None,
    ) -> None:
        """Initialize the outline store.

        Args:
            document_provider: Callable that returns a document by ID.
            storage_root: Root directory for outline storage.
            loop_resolver: Callable that returns the async event loop.
            event_bus: Event bus for publishing state change events.
            index_propagator: Optional callback to propagate embedding index
                to the worker after startup.
        """
        self._event_bus = event_bus
        self._document_provider = document_provider
        self._storage_root = storage_root
        self._loop_resolver = loop_resolver
        self._index_propagator = index_propagator
        self._worker: OutlineBuilderWorker | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def worker(self) -> OutlineBuilderWorker | None:
        """Get the active OutlineBuilderWorker, or None if not started."""
        return self._worker

    @property
    def memory(self) -> DocumentSummaryMemory | None:
        """Get the DocumentSummaryMemory from the worker, or None."""
        worker = self._worker
        if worker is None:
            return None
        return getattr(worker, "memory", None)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def ensure_started(self) -> OutlineBuilderWorker | None:
        """Ensure the outline worker is started and return it.

        This method lazily initializes the OutlineBuilderWorker. It will
        return the existing worker if already started, or create a new
        one if needed.

        Returns:
            The OutlineBuilderWorker instance, or None if startup failed
            (e.g., no event loop available or loop is closed).
        """
        loop = self._loop_resolver()
        if loop is None:
            LOGGER.warning(
                "Outline worker not started; no asyncio event loop was resolved."
            )
            return None

        # Check if loop is closed
        loop_is_closed = getattr(loop, "is_closed", None)
        try:
            if callable(loop_is_closed) and loop_is_closed():
                LOGGER.warning(
                    "Outline worker not started; resolved event loop is closed."
                )
                return None
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Outline worker could not verify loop state", exc_info=True)

        if self._worker is not None:
            return self._worker

        worker = self._start_worker(loop)
        if worker is None:
            LOGGER.warning(
                "Outline worker startup failed; see debug logs for details."
            )
        return worker

    def shutdown(self) -> None:
        """Shutdown the outline worker and release resources.

        This method safely closes the OutlineBuilderWorker, handling both
        running and idle event loops. Call this during application shutdown.
        """
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

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _start_worker(
        self,
        loop: asyncio.AbstractEventLoop,
    ) -> OutlineBuilderWorker | None:
        """Start the outline worker with the given event loop.

        Args:
            loop: The asyncio event loop to use.

        Returns:
            The started OutlineBuilderWorker, or None if startup failed.
        """
        if self._worker is not None:
            return self._worker

        try:
            from ...ai.services import OutlineBuilderWorker

            worker = OutlineBuilderWorker(
                document_provider=self._document_provider,
                storage_dir=self._storage_root,
                loop=loop,
            )
        except Exception as exc:  # pragma: no cover - optional feature
            LOGGER.warning(
                "Outline worker unavailable; continuing without outlines: %s",
                exc,
            )
            LOGGER.debug("Outline worker startup traceback", exc_info=True)
            return None

        self._worker = worker

        # Call the index propagator if provided
        if self._index_propagator is not None:
            try:
                self._index_propagator()
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.debug(
                    "Outline worker propagation callback failed",
                    exc_info=True,
                )

        return worker


__all__ = ["OutlineStore"]
