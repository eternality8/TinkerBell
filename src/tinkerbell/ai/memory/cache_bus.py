"""Document cache invalidation bus used by agent tools and editor bridges."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Iterable, List, MutableMapping, Type
import logging
import weakref

_LOGGER = logging.getLogger(__name__)


class DocumentCacheEvent:
    """Base class for cache invalidation events."""

    __slots__ = ("document_id", "source")

    def __init__(self, document_id: str, *, source: str | None = None) -> None:
        self.document_id = document_id
        self.source = source


class DocumentChangedEvent(DocumentCacheEvent):
    """Published after the editor mutates a document."""

    __slots__ = ("version_id", "content_hash", "edited_ranges")

    def __init__(
        self,
        *,
        document_id: str,
        version_id: int,
        content_hash: str,
        edited_ranges: Iterable[tuple[int, int]] | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(document_id, source=source)
        spans = tuple(tuple(span) for span in (edited_ranges or ()))
        self.version_id = int(version_id)
        self.content_hash = content_hash
        self.edited_ranges: tuple[tuple[int, int], ...] = spans


class DocumentClosedEvent(DocumentCacheEvent):
    """Published when a tab/document is closed and caches should cleanup."""

    __slots__ = ("version_id", "content_hash", "reason")

    def __init__(
        self,
        *,
        document_id: str,
        version_id: int | None = None,
        content_hash: str | None = None,
        reason: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(document_id, source=source)
        self.version_id = version_id
        self.content_hash = content_hash
        self.reason = reason


Subscriber = Callable[[DocumentCacheEvent], None]


@dataclass(slots=True)
class _Subscriber:
    event_type: Type[DocumentCacheEvent]
    identity_func: Any
    identity_target: object | weakref.ReferenceType[Any] | None
    identity_uses_ref: bool
    strong_handler: Callable[[DocumentCacheEvent], None] | None
    weak_ref: Callable[[], Callable[[DocumentCacheEvent], None] | None] | None = None

    def resolve(self) -> Callable[[DocumentCacheEvent], None] | None:
        if self.weak_ref is not None:
            return self.weak_ref()
        return self.strong_handler

    def matches(self, handler: Callable[[DocumentCacheEvent], None]) -> bool:
        func = getattr(handler, "__func__", handler)
        if func is not self.identity_func:
            return False
        bound = getattr(handler, "__self__", None)
        if not self.identity_uses_ref:
            return self.identity_target is bound
        target_ref = self.identity_target
        if isinstance(target_ref, weakref.ReferenceType):
            return target_ref() is bound
        return target_ref is bound


class DocumentCacheBus:
    """Synchronous pub/sub bus for cache invalidation events."""

    def __init__(self) -> None:
        self._subscribers: MutableMapping[Type[DocumentCacheEvent], List[_Subscriber]] = {}
        self._lock = RLock()

    def subscribe(
        self,
        event_type: Type[DocumentCacheEvent],
        handler: Subscriber,
        *,
        weak: bool = False,
    ) -> None:
        callback, weak_ref = self._wrap_handler(handler, weak=weak)
        func = getattr(handler, "__func__", handler)
        bound = getattr(handler, "__self__", None)
        identity_target: object | weakref.ReferenceType[Any] | None = bound
        identity_uses_ref = False
        if weak and bound is not None:
            try:
                identity_target = weakref.ref(bound)
                identity_uses_ref = True
            except TypeError:
                identity_target = bound
        subscriber = _Subscriber(
            event_type=event_type,
            identity_func=func,
            identity_target=identity_target,
            identity_uses_ref=identity_uses_ref,
            strong_handler=callback,
            weak_ref=weak_ref,
        )
        with self._lock:
            self._subscribers.setdefault(event_type, []).append(subscriber)

    def unsubscribe(self, event_type: Type[DocumentCacheEvent], handler: Subscriber) -> None:
        with self._lock:
            subscribers = self._subscribers.get(event_type)
            if not subscribers:
                return
            subscribers[:] = [sub for sub in subscribers if not sub.matches(handler)]
            if not subscribers:
                self._subscribers.pop(event_type, None)

    def publish(self, event: DocumentCacheEvent) -> None:
        to_invoke: list[Callable[[DocumentCacheEvent], None]] = []
        with self._lock:
            for event_type, subscribers in list(self._subscribers.items()):
                if not isinstance(event, event_type):
                    continue
                stale: list[_Subscriber] = []
                for subscriber in list(subscribers):
                    callback = subscriber.resolve()
                    if callback is None:
                        stale.append(subscriber)
                        continue
                    to_invoke.append(callback)
                for target in stale:
                    subscribers.remove(target)
                if not subscribers:
                    self._subscribers.pop(event_type, None)
        for callback in to_invoke:
            try:
                callback(event)
            except Exception:  # pragma: no cover - subscriber isolation
                _LOGGER.exception("Document cache subscriber failed")

    @staticmethod
    def _wrap_handler(
        handler: Subscriber,
        *,
        weak: bool,
    ) -> tuple[Callable[[DocumentCacheEvent], None] | None, Callable[[], Callable[[DocumentCacheEvent], None] | None] | None]:
        if not weak:
            return handler, None
        weak_callback: Callable[[], Callable[[DocumentCacheEvent], None] | None] | None
        try:
            weak_callback = weakref.WeakMethod(handler)  # type: ignore[arg-type]
        except TypeError:
            try:
                weak_callback = weakref.ref(handler)  # type: ignore[arg-type]
            except TypeError:
                weak_callback = None
        if weak_callback is None:
            return handler, None
        return None, weak_callback

_GLOBAL_BUS: DocumentCacheBus | None = None


def get_document_cache_bus() -> DocumentCacheBus:
    global _GLOBAL_BUS
    if _GLOBAL_BUS is None:
        _GLOBAL_BUS = DocumentCacheBus()
    return _GLOBAL_BUS


def set_document_cache_bus(bus: DocumentCacheBus | None) -> DocumentCacheBus:
    global _GLOBAL_BUS
    _GLOBAL_BUS = bus
    if _GLOBAL_BUS is None:
        _GLOBAL_BUS = DocumentCacheBus()
    return _GLOBAL_BUS


class _BaseCacheObserver:
    __slots__ = ("name", "bus", "events")

    def __init__(self, name: str, *, bus: DocumentCacheBus | None = None) -> None:
        self.name = name
        self.bus = bus or get_document_cache_bus()
        self.events: list[DocumentCacheEvent] = []
        self.bus.subscribe(DocumentChangedEvent, self._handle_changed, weak=True)  # type: ignore[arg-type]
        self.bus.subscribe(DocumentClosedEvent, self._handle_closed, weak=True)  # type: ignore[arg-type]

    def _handle_changed(self, event: DocumentChangedEvent) -> None:
        self.events.append(event)
        _LOGGER.debug("%s invalidating %s (v%s)", self.name, event.document_id, event.version_id)

    def _handle_closed(self, event: DocumentClosedEvent) -> None:
        self.events.append(event)
        _LOGGER.debug("%s closing %s", self.name, event.document_id)


class ChunkCacheSubscriber(_BaseCacheObserver):
    def __init__(self, *, bus: DocumentCacheBus | None = None) -> None:
        super().__init__("ChunkCache", bus=bus)


class OutlineCacheSubscriber(_BaseCacheObserver):
    def __init__(self, *, bus: DocumentCacheBus | None = None) -> None:
        super().__init__("OutlineCache", bus=bus)


class EmbeddingCacheSubscriber(_BaseCacheObserver):
    def __init__(self, *, bus: DocumentCacheBus | None = None) -> None:
        super().__init__("EmbeddingCache", bus=bus)


__all__ = [
    "DocumentCacheBus",
    "DocumentCacheEvent",
    "DocumentChangedEvent",
    "DocumentClosedEvent",
    "ChunkCacheSubscriber",
    "OutlineCacheSubscriber",
    "EmbeddingCacheSubscriber",
    "get_document_cache_bus",
    "set_document_cache_bus",
]
