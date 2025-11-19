"""TTL cache for analysis advice."""

from __future__ import annotations

import time
from threading import Lock

from .models import AnalysisAdvice


class AnalysisCache:
    """Thread-safe TTL cache keyed by immutable tuples."""

    def __init__(self, ttl_seconds: float = 120.0) -> None:
        self._ttl = max(1.0, float(ttl_seconds))
        self._entries: dict[tuple[object, ...], tuple[float, AnalysisAdvice]] = {}
        self._lock = Lock()

    @property
    def ttl_seconds(self) -> float:
        return self._ttl

    def get(self, key: tuple[object, ...]) -> AnalysisAdvice | None:
        now = time.monotonic()
        with self._lock:
            payload = self._entries.get(key)
            if not payload:
                return None
            expires_at, advice = payload
            if now > expires_at:
                del self._entries[key]
                return None
            return advice

    def set(self, key: tuple[object, ...], advice: AnalysisAdvice) -> None:
        expires_at = time.monotonic() + self._ttl
        with self._lock:
            self._entries[key] = (expires_at, advice)

    def invalidate(self, key: tuple[object, ...]) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def invalidate_document(self, document_id: str) -> None:
        with self._lock:
            doomed = [key for key in self._entries if key and key[0] == document_id]
            for key in doomed:
                del self._entries[key]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
