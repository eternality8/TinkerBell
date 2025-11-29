"""Document cache service for the orchestration pipeline.

This module provides a caching layer for document snapshots to avoid
redundant reads and maintain consistency during turn execution.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from ..transaction import DocumentSnapshot

__all__ = [
    "DocumentCache",
    "CacheEntry",
    "CacheConfig",
    "CacheStats",
]

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Cache Configuration
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class CacheConfig:
    """Configuration for the document cache.

    Attributes:
        max_entries: Maximum number of documents to cache.
        ttl_seconds: Time-to-live for cache entries in seconds (0 = no expiry).
        track_stats: Whether to track cache statistics.
    """

    max_entries: int = 100
    ttl_seconds: float = 300.0  # 5 minutes default
    track_stats: bool = True


# -----------------------------------------------------------------------------
# Cache Entry
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class CacheEntry:
    """A cached document snapshot with metadata.

    Attributes:
        snapshot: The cached document snapshot.
        created_at: When the entry was created.
        accessed_at: When the entry was last accessed.
        access_count: Number of times the entry has been accessed.
    """

    snapshot: DocumentSnapshot
    created_at: float = field(default_factory=time.monotonic)
    accessed_at: float = field(default_factory=time.monotonic)
    access_count: int = 0

    def touch(self) -> None:
        """Update access time and count."""
        self.accessed_at = time.monotonic()
        self.access_count += 1

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if the entry has expired.

        Args:
            ttl_seconds: Time-to-live in seconds. 0 means no expiry.

        Returns:
            True if the entry is expired.
        """
        if ttl_seconds <= 0:
            return False
        age = time.monotonic() - self.created_at
        return age > ttl_seconds


# -----------------------------------------------------------------------------
# Cache Statistics
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class CacheStats:
    """Statistics for cache operations.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        evictions: Number of entries evicted due to size limits.
        expirations: Number of entries expired due to TTL.
        invalidations: Number of explicit invalidations.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    invalidations: int = 0

    @property
    def total_requests(self) -> int:
        """Total number of get requests."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        total = self.total_requests
        if total == 0:
            return 0.0
        return self.hits / total

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for logging/telemetry."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "invalidations": self.invalidations,
            "total_requests": self.total_requests,
            "hit_rate": self.hit_rate,
        }

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.invalidations = 0


# -----------------------------------------------------------------------------
# Document Cache
# -----------------------------------------------------------------------------


class DocumentCache:
    """Thread-safe LRU cache for document snapshots.

    Provides caching for DocumentSnapshot objects with:
    - LRU eviction when max entries is reached
    - Optional TTL-based expiration
    - Thread-safe operations
    - Cache statistics tracking

    Example:
        >>> cache = DocumentCache()
        >>> cache.set("doc-1", snapshot)
        >>> cached = cache.get("doc-1")
        >>> if cached is not None:
        ...     print(f"Cache hit! Content: {cached.content[:50]}...")
    """

    def __init__(self, config: CacheConfig | None = None) -> None:
        """Initialize the document cache.

        Args:
            config: Optional cache configuration.
        """
        self._config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats() if self._config.track_stats else None

    @property
    def config(self) -> CacheConfig:
        """The cache configuration."""
        return self._config

    @property
    def stats(self) -> CacheStats | None:
        """Cache statistics (None if tracking disabled)."""
        return self._stats

    def get(self, doc_id: str) -> DocumentSnapshot | None:
        """Get a document snapshot from the cache.

        Args:
            doc_id: The document identifier.

        Returns:
            The cached snapshot, or None if not found or expired.
        """
        with self._lock:
            entry = self._cache.get(doc_id)

            if entry is None:
                if self._stats:
                    self._stats.misses += 1
                return None

            # Check expiration
            if entry.is_expired(self._config.ttl_seconds):
                del self._cache[doc_id]
                if self._stats:
                    self._stats.expirations += 1
                    self._stats.misses += 1
                LOGGER.debug("Cache entry expired for %s", doc_id)
                return None

            # Update access and move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(doc_id)

            if self._stats:
                self._stats.hits += 1

            return entry.snapshot

    def set(
        self,
        doc_id: str,
        snapshot: DocumentSnapshot,
    ) -> None:
        """Store a document snapshot in the cache.

        If the cache is full, the least recently used entry is evicted.

        Args:
            doc_id: The document identifier.
            snapshot: The document snapshot to cache.
        """
        with self._lock:
            # If already exists, update it
            if doc_id in self._cache:
                self._cache[doc_id] = CacheEntry(snapshot=snapshot)
                self._cache.move_to_end(doc_id)
                return

            # Evict if at capacity
            while len(self._cache) >= self._config.max_entries:
                evicted_id, _ = self._cache.popitem(last=False)
                if self._stats:
                    self._stats.evictions += 1
                LOGGER.debug("Evicted cache entry for %s", evicted_id)

            # Add new entry
            self._cache[doc_id] = CacheEntry(snapshot=snapshot)

    def invalidate(self, doc_id: str) -> bool:
        """Remove a document from the cache.

        Args:
            doc_id: The document identifier to invalidate.

        Returns:
            True if the entry was found and removed, False otherwise.
        """
        with self._lock:
            if doc_id in self._cache:
                del self._cache[doc_id]
                if self._stats:
                    self._stats.invalidations += 1
                LOGGER.debug("Invalidated cache entry for %s", doc_id)
                return True
            return False

    def invalidate_all(self) -> int:
        """Remove all entries from the cache.

        Returns:
            The number of entries that were removed.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            if self._stats:
                self._stats.invalidations += count
            LOGGER.debug("Invalidated all %d cache entries", count)
            return count

    def contains(self, doc_id: str) -> bool:
        """Check if a document is in the cache (without updating access).

        Note: Does not check expiration or update access time.

        Args:
            doc_id: The document identifier.

        Returns:
            True if the document is in the cache.
        """
        with self._lock:
            return doc_id in self._cache

    def size(self) -> int:
        """Get the current number of entries in the cache.

        Returns:
            The number of cached entries.
        """
        with self._lock:
            return len(self._cache)

    def keys(self) -> list[str]:
        """Get all document IDs currently in the cache.

        Returns:
            List of document IDs.
        """
        with self._lock:
            return list(self._cache.keys())

    def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.

        Returns:
            The number of entries that were removed.
        """
        if self._config.ttl_seconds <= 0:
            return 0

        with self._lock:
            expired_ids = [
                doc_id
                for doc_id, entry in self._cache.items()
                if entry.is_expired(self._config.ttl_seconds)
            ]

            for doc_id in expired_ids:
                del self._cache[doc_id]
                if self._stats:
                    self._stats.expirations += 1

            if expired_ids:
                LOGGER.debug("Cleaned up %d expired cache entries", len(expired_ids))

            return len(expired_ids)

    def get_or_load(
        self,
        doc_id: str,
        loader: Callable[[], DocumentSnapshot | None],
    ) -> DocumentSnapshot | None:
        """Get from cache or load using the provided function.

        This is a convenience method that implements the cache-aside pattern.

        Args:
            doc_id: The document identifier.
            loader: Function to call if not in cache. Should return a
                DocumentSnapshot or None.

        Returns:
            The cached or freshly loaded snapshot, or None if loader returns None.
        """
        # Try cache first
        cached = self.get(doc_id)
        if cached is not None:
            return cached

        # Load from source
        snapshot = loader()
        if snapshot is not None:
            self.set(doc_id, snapshot)

        return snapshot
