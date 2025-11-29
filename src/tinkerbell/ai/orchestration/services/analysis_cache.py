"""Analysis cache service for the orchestration pipeline.

This module provides a caching layer for analysis results to avoid
redundant preflight analysis operations during turn execution.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from ..pipeline.analyze import AnalysisAdvice
from ..transaction import DocumentSnapshot

__all__ = [
    "AnalysisCache",
    "AnalysisCacheConfig",
    "AnalysisCacheEntry",
    "AnalysisCacheStats",
    "compute_snapshot_hash",
]

LOGGER = logging.getLogger(__name__)


# TypeVar for advice implementations
A = TypeVar("A", bound=AnalysisAdvice)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def compute_snapshot_hash(snapshot: DocumentSnapshot) -> str:
    """Compute a hash for a document snapshot.
    
    The hash is based on content and version to identify unique states.
    
    Args:
        snapshot: The document snapshot to hash.
        
    Returns:
        A hex string hash of the snapshot.
    """
    hasher = hashlib.sha256()
    hasher.update(snapshot.tab_id.encode("utf-8"))
    hasher.update(snapshot.content.encode("utf-8"))
    if snapshot.version_token is not None:
        hasher.update(snapshot.version_token.encode("utf-8"))
    return hasher.hexdigest()[:16]  # Use first 16 chars for brevity


# -----------------------------------------------------------------------------
# Cache Configuration
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class AnalysisCacheConfig:
    """Configuration for the analysis cache.

    Attributes:
        max_entries: Maximum number of analyses to cache.
        ttl_seconds: Time-to-live for cache entries in seconds (0 = no expiry).
        track_stats: Whether to track cache statistics.
    """

    max_entries: int = 50
    ttl_seconds: float = 600.0  # 10 minutes default (analysis less frequent)
    track_stats: bool = True


# -----------------------------------------------------------------------------
# Cache Entry
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class AnalysisCacheEntry(Generic[A]):
    """A cached analysis result with metadata.

    Attributes:
        advice: The cached analysis advice.
        snapshot_hash: Hash of the snapshot that was analyzed.
        created_at: When the entry was created.
        accessed_at: When the entry was last accessed.
        access_count: Number of times the entry has been accessed.
    """

    advice: A
    snapshot_hash: str
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
class AnalysisCacheStats:
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
# Analysis Cache
# -----------------------------------------------------------------------------


class AnalysisCache(Generic[A]):
    """Thread-safe LRU cache for analysis results.

    Provides caching for AnalysisAdvice objects with:
    - LRU eviction when max entries is reached
    - Optional TTL-based expiration
    - Thread-safe operations
    - Cache statistics tracking
    - Hash-based lookup by snapshot content

    The cache is keyed by snapshot hash, allowing efficient lookup
    when the same document state is analyzed multiple times.

    Example:
        >>> cache: AnalysisCache[MyAdvice] = AnalysisCache()
        >>> hash_key = compute_snapshot_hash(snapshot)
        >>> cache.set(hash_key, advice)
        >>> cached = cache.get(hash_key)
        >>> if cached is not None:
        ...     print(f"Cache hit! Using cached analysis.")
    """

    def __init__(self, config: AnalysisCacheConfig | None = None) -> None:
        """Initialize the analysis cache.

        Args:
            config: Optional cache configuration.
        """
        self._config = config or AnalysisCacheConfig()
        self._cache: OrderedDict[str, AnalysisCacheEntry[A]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = AnalysisCacheStats() if self._config.track_stats else None

    @property
    def config(self) -> AnalysisCacheConfig:
        """The cache configuration."""
        return self._config

    @property
    def stats(self) -> AnalysisCacheStats | None:
        """Cache statistics (None if tracking disabled)."""
        return self._stats

    def get(self, snapshot_hash: str) -> A | None:
        """Get an analysis result from the cache.

        Args:
            snapshot_hash: The hash of the document snapshot.

        Returns:
            The cached advice, or None if not found or expired.
        """
        with self._lock:
            entry = self._cache.get(snapshot_hash)

            if entry is None:
                if self._stats:
                    self._stats.misses += 1
                return None

            # Check expiration
            if entry.is_expired(self._config.ttl_seconds):
                del self._cache[snapshot_hash]
                if self._stats:
                    self._stats.expirations += 1
                    self._stats.misses += 1
                LOGGER.debug("Cache entry expired for hash %s", snapshot_hash[:8])
                return None

            # Update access and move to end (most recently used)
            entry.touch()
            self._cache.move_to_end(snapshot_hash)

            if self._stats:
                self._stats.hits += 1

            return entry.advice

    def set(self, snapshot_hash: str, advice: A) -> None:
        """Store an analysis result in the cache.

        If the cache is full, the least recently used entry is evicted.

        Args:
            snapshot_hash: The hash of the document snapshot.
            advice: The analysis advice to cache.
        """
        with self._lock:
            # If already exists, update it
            if snapshot_hash in self._cache:
                self._cache[snapshot_hash] = AnalysisCacheEntry(
                    advice=advice,
                    snapshot_hash=snapshot_hash,
                )
                self._cache.move_to_end(snapshot_hash)
                return

            # Evict if at capacity
            while len(self._cache) >= self._config.max_entries:
                evicted_hash, _ = self._cache.popitem(last=False)
                if self._stats:
                    self._stats.evictions += 1
                LOGGER.debug("Evicted cache entry for hash %s", evicted_hash[:8])

            # Add new entry
            self._cache[snapshot_hash] = AnalysisCacheEntry(
                advice=advice,
                snapshot_hash=snapshot_hash,
            )

    def invalidate(self, snapshot_hash: str) -> bool:
        """Remove an analysis from the cache.

        Args:
            snapshot_hash: The snapshot hash to invalidate.

        Returns:
            True if the entry was found and removed, False otherwise.
        """
        with self._lock:
            if snapshot_hash in self._cache:
                del self._cache[snapshot_hash]
                if self._stats:
                    self._stats.invalidations += 1
                LOGGER.debug("Invalidated cache entry for hash %s", snapshot_hash[:8])
                return True
            return False

    def invalidate_for_document(self, doc_id: str) -> int:
        """Remove all analyses for a specific document.
        
        This is useful when a document changes and all cached analyses
        for it should be invalidated.

        Args:
            doc_id: The document identifier.

        Returns:
            The number of entries that were removed.
        """
        with self._lock:
            # Find all entries for this document
            to_remove = [
                hash_key
                for hash_key, entry in self._cache.items()
                if entry.advice.document_id == doc_id
            ]
            
            for hash_key in to_remove:
                del self._cache[hash_key]
                if self._stats:
                    self._stats.invalidations += 1
            
            if to_remove:
                LOGGER.debug(
                    "Invalidated %d cache entries for document %s",
                    len(to_remove),
                    doc_id,
                )
            
            return len(to_remove)

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

    def contains(self, snapshot_hash: str) -> bool:
        """Check if an analysis is in the cache (without updating access).

        Note: Does not check expiration or update access time.

        Args:
            snapshot_hash: The snapshot hash.

        Returns:
            True if the analysis is in the cache.
        """
        with self._lock:
            return snapshot_hash in self._cache

    def size(self) -> int:
        """Get the current number of entries in the cache.

        Returns:
            The number of cached entries.
        """
        with self._lock:
            return len(self._cache)

    def keys(self) -> list[str]:
        """Get all snapshot hashes currently in the cache.

        Returns:
            List of snapshot hashes.
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
            expired_hashes = [
                hash_key
                for hash_key, entry in self._cache.items()
                if entry.is_expired(self._config.ttl_seconds)
            ]

            for hash_key in expired_hashes:
                del self._cache[hash_key]
                if self._stats:
                    self._stats.expirations += 1

            if expired_hashes:
                LOGGER.debug("Cleaned up %d expired cache entries", len(expired_hashes))

            return len(expired_hashes)

    def get_or_analyze(
        self,
        snapshot: DocumentSnapshot,
        analyzer: Callable[[DocumentSnapshot], A | None],
    ) -> A | None:
        """Get from cache or run analysis.

        This is a convenience method that implements the cache-aside pattern.
        It computes the snapshot hash, checks the cache, and runs the analyzer
        if needed.

        Args:
            snapshot: The document snapshot to analyze.
            analyzer: Function to call if not in cache. Should return
                AnalysisAdvice or None.

        Returns:
            The cached or freshly computed advice, or None if analyzer returns None.
        """
        snapshot_hash = compute_snapshot_hash(snapshot)
        
        # Try cache first
        cached = self.get(snapshot_hash)
        if cached is not None:
            return cached

        # Run analysis
        advice = analyzer(snapshot)
        if advice is not None:
            self.set(snapshot_hash, advice)

        return advice
