"""Tests for the document cache service.

This module tests the DocumentCache class that provides caching
for document snapshots.
"""

from __future__ import annotations

import time
import threading
import pytest
from datetime import datetime, timezone

from tinkerbell.ai.orchestration.services.document_cache import (
    DocumentCache,
    CacheConfig,
    CacheEntry,
    CacheStats,
)
from tinkerbell.ai.orchestration.transaction import DocumentSnapshot


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def snapshot() -> DocumentSnapshot:
    """Create a test document snapshot."""
    return DocumentSnapshot(
        tab_id="tab-1",
        content="Test document content",
        version_token="v1",
    )


@pytest.fixture
def cache() -> DocumentCache:
    """Create a cache with default config."""
    return DocumentCache()


@pytest.fixture
def small_cache() -> DocumentCache:
    """Create a cache with small capacity for eviction tests."""
    return DocumentCache(CacheConfig(max_entries=3))


def make_snapshot(tab_id: str, content: str = "content") -> DocumentSnapshot:
    """Helper to create snapshots."""
    return DocumentSnapshot(
        tab_id=tab_id,
        content=content,
        version_token="v1",
    )


# =============================================================================
# CacheConfig Tests
# =============================================================================


class TestCacheConfig:
    """Tests for CacheConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = CacheConfig()
        
        assert config.max_entries == 100
        assert config.ttl_seconds == 300.0
        assert config.track_stats is True

    def test_custom_values(self):
        """Should accept custom values."""
        config = CacheConfig(
            max_entries=50,
            ttl_seconds=60.0,
            track_stats=False,
        )
        
        assert config.max_entries == 50
        assert config.ttl_seconds == 60.0
        assert config.track_stats is False

    def test_frozen(self):
        """Config should be frozen."""
        config = CacheConfig()
        
        with pytest.raises(AttributeError):
            config.max_entries = 10  # type: ignore


# =============================================================================
# CacheEntry Tests
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""

    def test_creation(self, snapshot: DocumentSnapshot):
        """Should create entry with snapshot."""
        entry = CacheEntry(snapshot=snapshot)
        
        assert entry.snapshot is snapshot
        assert entry.access_count == 0
        assert entry.created_at > 0
        assert entry.accessed_at > 0

    def test_touch_updates_access(self, snapshot: DocumentSnapshot):
        """touch() should update access time and count."""
        entry = CacheEntry(snapshot=snapshot)
        initial_accessed = entry.accessed_at
        
        time.sleep(0.01)  # Small delay
        entry.touch()
        
        assert entry.access_count == 1
        assert entry.accessed_at > initial_accessed

    def test_touch_multiple_times(self, snapshot: DocumentSnapshot):
        """Multiple touch() calls should increment count."""
        entry = CacheEntry(snapshot=snapshot)
        
        entry.touch()
        entry.touch()
        entry.touch()
        
        assert entry.access_count == 3

    def test_is_expired_with_ttl(self, snapshot: DocumentSnapshot):
        """is_expired should check TTL correctly."""
        entry = CacheEntry(snapshot=snapshot)
        
        # Should not be expired with 10 second TTL
        assert entry.is_expired(10.0) is False
        
        # Should be expired with 0.001 second TTL after small delay
        time.sleep(0.01)
        assert entry.is_expired(0.001) is True

    def test_is_expired_no_ttl(self, snapshot: DocumentSnapshot):
        """is_expired should return False when TTL is 0."""
        entry = CacheEntry(snapshot=snapshot)
        
        # TTL of 0 means no expiry
        assert entry.is_expired(0) is False
        assert entry.is_expired(-1) is False


# =============================================================================
# CacheStats Tests
# =============================================================================


class TestCacheStats:
    """Tests for CacheStats dataclass."""

    def test_initial_values(self):
        """Should start with all zeros."""
        stats = CacheStats()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.expirations == 0
        assert stats.invalidations == 0

    def test_total_requests(self):
        """total_requests should be hits + misses."""
        stats = CacheStats(hits=10, misses=5)
        
        assert stats.total_requests == 15

    def test_hit_rate_with_requests(self):
        """hit_rate should calculate correctly."""
        stats = CacheStats(hits=75, misses=25)
        
        assert stats.hit_rate == 0.75

    def test_hit_rate_no_requests(self):
        """hit_rate should return 0.0 with no requests."""
        stats = CacheStats()
        
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        stats = CacheStats(
            hits=10,
            misses=5,
            evictions=2,
            expirations=1,
            invalidations=3,
        )
        
        result = stats.to_dict()
        
        assert result["hits"] == 10
        assert result["misses"] == 5
        assert result["evictions"] == 2
        assert result["expirations"] == 1
        assert result["invalidations"] == 3
        assert result["total_requests"] == 15
        assert result["hit_rate"] == pytest.approx(0.667, rel=0.01)

    def test_reset(self):
        """reset should clear all statistics."""
        stats = CacheStats(
            hits=10,
            misses=5,
            evictions=2,
            expirations=1,
            invalidations=3,
        )
        
        stats.reset()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.expirations == 0
        assert stats.invalidations == 0


# =============================================================================
# DocumentCache Basic Operations
# =============================================================================


class TestDocumentCacheBasic:
    """Tests for basic cache operations."""

    def test_get_missing_returns_none(self, cache: DocumentCache):
        """get() should return None for missing entries."""
        result = cache.get("nonexistent")
        
        assert result is None

    def test_set_and_get(self, cache: DocumentCache, snapshot: DocumentSnapshot):
        """set() and get() should work together."""
        cache.set("doc-1", snapshot)
        
        result = cache.get("doc-1")
        
        assert result is snapshot

    def test_set_overwrites_existing(self, cache: DocumentCache):
        """set() should overwrite existing entries."""
        snapshot1 = make_snapshot("tab-1", "content 1")
        snapshot2 = make_snapshot("tab-1", "content 2")
        
        cache.set("doc-1", snapshot1)
        cache.set("doc-1", snapshot2)
        
        result = cache.get("doc-1")
        assert result is snapshot2

    def test_invalidate_removes_entry(
        self, cache: DocumentCache, snapshot: DocumentSnapshot
    ):
        """invalidate() should remove entry."""
        cache.set("doc-1", snapshot)
        
        removed = cache.invalidate("doc-1")
        
        assert removed is True
        assert cache.get("doc-1") is None

    def test_invalidate_missing_returns_false(self, cache: DocumentCache):
        """invalidate() should return False for missing entries."""
        removed = cache.invalidate("nonexistent")
        
        assert removed is False

    def test_invalidate_all(self, cache: DocumentCache):
        """invalidate_all() should clear all entries."""
        cache.set("doc-1", make_snapshot("tab-1"))
        cache.set("doc-2", make_snapshot("tab-2"))
        cache.set("doc-3", make_snapshot("tab-3"))
        
        count = cache.invalidate_all()
        
        assert count == 3
        assert cache.size() == 0

    def test_contains(self, cache: DocumentCache, snapshot: DocumentSnapshot):
        """contains() should check presence."""
        cache.set("doc-1", snapshot)
        
        assert cache.contains("doc-1") is True
        assert cache.contains("doc-2") is False

    def test_size(self, cache: DocumentCache):
        """size() should return entry count."""
        assert cache.size() == 0
        
        cache.set("doc-1", make_snapshot("tab-1"))
        assert cache.size() == 1
        
        cache.set("doc-2", make_snapshot("tab-2"))
        assert cache.size() == 2
        
        cache.invalidate("doc-1")
        assert cache.size() == 1

    def test_keys(self, cache: DocumentCache):
        """keys() should return all document IDs."""
        cache.set("doc-1", make_snapshot("tab-1"))
        cache.set("doc-2", make_snapshot("tab-2"))
        cache.set("doc-3", make_snapshot("tab-3"))
        
        keys = cache.keys()
        
        assert set(keys) == {"doc-1", "doc-2", "doc-3"}


# =============================================================================
# LRU Eviction Tests
# =============================================================================


class TestDocumentCacheLRU:
    """Tests for LRU eviction behavior."""

    def test_evicts_oldest_when_full(self, small_cache: DocumentCache):
        """Should evict LRU entry when at capacity."""
        small_cache.set("doc-1", make_snapshot("tab-1"))
        small_cache.set("doc-2", make_snapshot("tab-2"))
        small_cache.set("doc-3", make_snapshot("tab-3"))
        
        # This should evict doc-1
        small_cache.set("doc-4", make_snapshot("tab-4"))
        
        assert small_cache.size() == 3
        assert small_cache.get("doc-1") is None
        assert small_cache.get("doc-4") is not None

    def test_access_updates_lru_order(self, small_cache: DocumentCache):
        """Accessing an entry should move it to MRU position."""
        small_cache.set("doc-1", make_snapshot("tab-1"))
        small_cache.set("doc-2", make_snapshot("tab-2"))
        small_cache.set("doc-3", make_snapshot("tab-3"))
        
        # Access doc-1, making it MRU
        small_cache.get("doc-1")
        
        # Add new entry - should evict doc-2 (now LRU)
        small_cache.set("doc-4", make_snapshot("tab-4"))
        
        assert small_cache.get("doc-1") is not None  # Still there
        assert small_cache.get("doc-2") is None      # Evicted
        assert small_cache.get("doc-3") is not None  # Still there
        assert small_cache.get("doc-4") is not None  # Added

    def test_eviction_tracks_stats(self, small_cache: DocumentCache):
        """Evictions should be tracked in stats."""
        small_cache.set("doc-1", make_snapshot("tab-1"))
        small_cache.set("doc-2", make_snapshot("tab-2"))
        small_cache.set("doc-3", make_snapshot("tab-3"))
        small_cache.set("doc-4", make_snapshot("tab-4"))  # Evicts doc-1
        small_cache.set("doc-5", make_snapshot("tab-5"))  # Evicts doc-2
        
        assert small_cache.stats is not None
        assert small_cache.stats.evictions == 2


# =============================================================================
# TTL Expiration Tests
# =============================================================================


class TestDocumentCacheTTL:
    """Tests for TTL expiration behavior."""

    def test_expired_entry_returns_none(self):
        """Expired entries should return None on get()."""
        cache = DocumentCache(CacheConfig(ttl_seconds=0.01))
        cache.set("doc-1", make_snapshot("tab-1"))
        
        time.sleep(0.02)  # Wait for expiry
        
        result = cache.get("doc-1")
        assert result is None

    def test_expired_entry_removed_from_cache(self):
        """Expired entries should be removed on get()."""
        cache = DocumentCache(CacheConfig(ttl_seconds=0.01))
        cache.set("doc-1", make_snapshot("tab-1"))
        
        time.sleep(0.02)
        cache.get("doc-1")
        
        assert cache.contains("doc-1") is False

    def test_expiration_tracks_stats(self):
        """Expirations should be tracked in stats."""
        cache = DocumentCache(CacheConfig(ttl_seconds=0.01))
        cache.set("doc-1", make_snapshot("tab-1"))
        
        time.sleep(0.02)
        cache.get("doc-1")
        
        assert cache.stats is not None
        assert cache.stats.expirations == 1

    def test_cleanup_expired(self):
        """cleanup_expired() should remove all expired entries."""
        cache = DocumentCache(CacheConfig(ttl_seconds=0.01))
        cache.set("doc-1", make_snapshot("tab-1"))
        cache.set("doc-2", make_snapshot("tab-2"))
        
        time.sleep(0.02)
        
        count = cache.cleanup_expired()
        
        assert count == 2
        assert cache.size() == 0

    def test_cleanup_expired_with_no_ttl(self):
        """cleanup_expired() with TTL=0 should do nothing."""
        cache = DocumentCache(CacheConfig(ttl_seconds=0))
        cache.set("doc-1", make_snapshot("tab-1"))
        
        count = cache.cleanup_expired()
        
        assert count == 0
        assert cache.size() == 1

    def test_no_expiry_when_ttl_zero(self):
        """TTL of 0 should mean no expiry."""
        cache = DocumentCache(CacheConfig(ttl_seconds=0))
        cache.set("doc-1", make_snapshot("tab-1"))
        
        # Entry should never expire
        result = cache.get("doc-1")
        assert result is not None


# =============================================================================
# Statistics Tests
# =============================================================================


class TestDocumentCacheStats:
    """Tests for cache statistics tracking."""

    def test_hit_tracked(self, cache: DocumentCache, snapshot: DocumentSnapshot):
        """Cache hits should be tracked."""
        cache.set("doc-1", snapshot)
        cache.get("doc-1")
        
        assert cache.stats is not None
        assert cache.stats.hits == 1

    def test_miss_tracked(self, cache: DocumentCache):
        """Cache misses should be tracked."""
        cache.get("nonexistent")
        
        assert cache.stats is not None
        assert cache.stats.misses == 1

    def test_invalidation_tracked(
        self, cache: DocumentCache, snapshot: DocumentSnapshot
    ):
        """Invalidations should be tracked."""
        cache.set("doc-1", snapshot)
        cache.invalidate("doc-1")
        
        assert cache.stats is not None
        assert cache.stats.invalidations == 1

    def test_stats_disabled(self, snapshot: DocumentSnapshot):
        """Stats should be None when disabled."""
        cache = DocumentCache(CacheConfig(track_stats=False))
        
        assert cache.stats is None
        
        # Operations should still work
        cache.set("doc-1", snapshot)
        result = cache.get("doc-1")
        assert result is snapshot


# =============================================================================
# get_or_load Tests
# =============================================================================


class TestDocumentCacheGetOrLoad:
    """Tests for the get_or_load convenience method."""

    def test_returns_cached_value(
        self, cache: DocumentCache, snapshot: DocumentSnapshot
    ):
        """get_or_load should return cached value if present."""
        cache.set("doc-1", snapshot)
        
        loader_called = False
        def loader():
            nonlocal loader_called
            loader_called = True
            return make_snapshot("tab-new")
        
        result = cache.get_or_load("doc-1", loader)
        
        assert result is snapshot
        assert loader_called is False

    def test_loads_and_caches_on_miss(self, cache: DocumentCache):
        """get_or_load should call loader and cache result on miss."""
        loaded_snapshot = make_snapshot("tab-loaded")
        
        def loader():
            return loaded_snapshot
        
        result = cache.get_or_load("doc-1", loader)
        
        assert result is loaded_snapshot
        assert cache.get("doc-1") is loaded_snapshot

    def test_returns_none_when_loader_returns_none(self, cache: DocumentCache):
        """get_or_load should return None if loader returns None."""
        def loader():
            return None
        
        result = cache.get_or_load("doc-1", loader)
        
        assert result is None
        assert cache.contains("doc-1") is False


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestDocumentCacheThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_access(self, cache: DocumentCache):
        """Cache should handle concurrent access safely."""
        results: list[DocumentSnapshot | None] = []
        errors: list[Exception] = []
        
        def writer(doc_id: str):
            try:
                for i in range(100):
                    cache.set(doc_id, make_snapshot(f"tab-{doc_id}-{i}"))
            except Exception as e:
                errors.append(e)
        
        def reader(doc_id: str):
            try:
                for _ in range(100):
                    result = cache.get(doc_id)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=writer, args=("doc-1",)),
            threading.Thread(target=writer, args=("doc-2",)),
            threading.Thread(target=reader, args=("doc-1",)),
            threading.Thread(target=reader, args=("doc-2",)),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        # Results should be either None or valid snapshots
        for r in results:
            if r is not None:
                assert isinstance(r, DocumentSnapshot)

    def test_concurrent_eviction(self):
        """Eviction should be thread-safe."""
        cache = DocumentCache(CacheConfig(max_entries=10))
        errors: list[Exception] = []
        
        def writer(prefix: str):
            try:
                for i in range(50):
                    cache.set(f"{prefix}-{i}", make_snapshot(f"tab-{prefix}-{i}"))
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=writer, args=(f"thread-{i}",))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert cache.size() <= 10  # Should respect max_entries
