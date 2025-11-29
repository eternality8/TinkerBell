"""Tests for the analysis cache service.

This module tests the AnalysisCache class that provides caching
for analysis results.
"""

from __future__ import annotations

import time
import threading
import pytest
from dataclasses import dataclass
from typing import Sequence, Any

from tinkerbell.ai.orchestration.services.analysis_cache import (
    AnalysisCache,
    AnalysisCacheConfig,
    AnalysisCacheEntry,
    AnalysisCacheStats,
    compute_snapshot_hash,
)
from tinkerbell.ai.orchestration.transaction import DocumentSnapshot


# =============================================================================
# Mock AnalysisAdvice Implementation
# =============================================================================


@dataclass(slots=True, frozen=True)
class MockAdvice:
    """Mock implementation of AnalysisAdvice protocol."""
    
    document_id: str = "doc-1"
    chunk_profile: str = "auto"
    required_tools: tuple[str, ...] = ()
    optional_tools: tuple[str, ...] = ()
    must_refresh_outline: bool = False
    plot_state_status: str | None = None
    concordance_status: str | None = None
    warnings: tuple[Any, ...] = ()


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
def advice() -> MockAdvice:
    """Create a test analysis advice."""
    return MockAdvice(document_id="doc-1")


@pytest.fixture
def cache() -> AnalysisCache[MockAdvice]:
    """Create a cache with default config."""
    return AnalysisCache()


@pytest.fixture
def small_cache() -> AnalysisCache[MockAdvice]:
    """Create a cache with small capacity for eviction tests."""
    return AnalysisCache(AnalysisCacheConfig(max_entries=3))


def make_snapshot(tab_id: str, content: str = "content") -> DocumentSnapshot:
    """Helper to create snapshots."""
    return DocumentSnapshot(
        tab_id=tab_id,
        content=content,
        version_token="v1",
    )


def make_advice(doc_id: str = "doc-1") -> MockAdvice:
    """Helper to create advice."""
    return MockAdvice(document_id=doc_id)


# =============================================================================
# compute_snapshot_hash Tests
# =============================================================================


class TestComputeSnapshotHash:
    """Tests for the compute_snapshot_hash function."""

    def test_same_content_same_hash(self):
        """Same content should produce same hash."""
        snap1 = make_snapshot("tab-1", "content")
        snap2 = make_snapshot("tab-1", "content")
        
        assert compute_snapshot_hash(snap1) == compute_snapshot_hash(snap2)

    def test_different_content_different_hash(self):
        """Different content should produce different hash."""
        snap1 = make_snapshot("tab-1", "content 1")
        snap2 = make_snapshot("tab-1", "content 2")
        
        assert compute_snapshot_hash(snap1) != compute_snapshot_hash(snap2)

    def test_different_tab_id_different_hash(self):
        """Different tab ID should produce different hash."""
        snap1 = make_snapshot("tab-1", "content")
        snap2 = make_snapshot("tab-2", "content")
        
        assert compute_snapshot_hash(snap1) != compute_snapshot_hash(snap2)

    def test_hash_is_deterministic(self, snapshot: DocumentSnapshot):
        """Hash should be deterministic."""
        hash1 = compute_snapshot_hash(snapshot)
        hash2 = compute_snapshot_hash(snapshot)
        
        assert hash1 == hash2

    def test_hash_format(self, snapshot: DocumentSnapshot):
        """Hash should be a hex string of expected length."""
        result = compute_snapshot_hash(snapshot)
        
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)


# =============================================================================
# AnalysisCacheConfig Tests
# =============================================================================


class TestAnalysisCacheConfig:
    """Tests for AnalysisCacheConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = AnalysisCacheConfig()
        
        assert config.max_entries == 50
        assert config.ttl_seconds == 600.0
        assert config.track_stats is True

    def test_custom_values(self):
        """Should accept custom values."""
        config = AnalysisCacheConfig(
            max_entries=25,
            ttl_seconds=120.0,
            track_stats=False,
        )
        
        assert config.max_entries == 25
        assert config.ttl_seconds == 120.0
        assert config.track_stats is False

    def test_frozen(self):
        """Config should be frozen."""
        config = AnalysisCacheConfig()
        
        with pytest.raises(AttributeError):
            config.max_entries = 10  # type: ignore


# =============================================================================
# AnalysisCacheEntry Tests
# =============================================================================


class TestAnalysisCacheEntry:
    """Tests for AnalysisCacheEntry dataclass."""

    def test_creation(self, advice: MockAdvice):
        """Should create entry with advice."""
        entry: AnalysisCacheEntry[MockAdvice] = AnalysisCacheEntry(
            advice=advice,
            snapshot_hash="abc123",
        )
        
        assert entry.advice is advice
        assert entry.snapshot_hash == "abc123"
        assert entry.access_count == 0
        assert entry.created_at > 0
        assert entry.accessed_at > 0

    def test_touch_updates_access(self, advice: MockAdvice):
        """touch() should update access time and count."""
        entry: AnalysisCacheEntry[MockAdvice] = AnalysisCacheEntry(
            advice=advice,
            snapshot_hash="abc123",
        )
        initial_accessed = entry.accessed_at
        
        time.sleep(0.01)
        entry.touch()
        
        assert entry.access_count == 1
        assert entry.accessed_at > initial_accessed

    def test_is_expired_with_ttl(self, advice: MockAdvice):
        """is_expired should check TTL correctly."""
        entry: AnalysisCacheEntry[MockAdvice] = AnalysisCacheEntry(
            advice=advice,
            snapshot_hash="abc123",
        )
        
        # Should not be expired with 10 second TTL
        assert entry.is_expired(10.0) is False
        
        # Should be expired with tiny TTL after delay
        time.sleep(0.01)
        assert entry.is_expired(0.001) is True

    def test_is_expired_no_ttl(self, advice: MockAdvice):
        """is_expired should return False when TTL is 0."""
        entry: AnalysisCacheEntry[MockAdvice] = AnalysisCacheEntry(
            advice=advice,
            snapshot_hash="abc123",
        )
        
        assert entry.is_expired(0) is False
        assert entry.is_expired(-1) is False


# =============================================================================
# AnalysisCacheStats Tests
# =============================================================================


class TestAnalysisCacheStats:
    """Tests for AnalysisCacheStats dataclass."""

    def test_initial_values(self):
        """Should start with all zeros."""
        stats = AnalysisCacheStats()
        
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0
        assert stats.expirations == 0
        assert stats.invalidations == 0

    def test_total_requests(self):
        """total_requests should be hits + misses."""
        stats = AnalysisCacheStats(hits=10, misses=5)
        
        assert stats.total_requests == 15

    def test_hit_rate_with_requests(self):
        """hit_rate should calculate correctly."""
        stats = AnalysisCacheStats(hits=75, misses=25)
        
        assert stats.hit_rate == 0.75

    def test_hit_rate_no_requests(self):
        """hit_rate should return 0.0 with no requests."""
        stats = AnalysisCacheStats()
        
        assert stats.hit_rate == 0.0

    def test_to_dict(self):
        """to_dict should serialize all fields."""
        stats = AnalysisCacheStats(
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
        stats = AnalysisCacheStats(
            hits=10,
            misses=5,
            evictions=2,
            expirations=1,
            invalidations=3,
        )
        
        stats.reset()
        
        assert stats.hits == 0
        assert stats.misses == 0


# =============================================================================
# AnalysisCache Basic Operations
# =============================================================================


class TestAnalysisCacheBasic:
    """Tests for basic cache operations."""

    def test_get_missing_returns_none(self, cache: AnalysisCache[MockAdvice]):
        """get() should return None for missing entries."""
        result = cache.get("nonexistent")
        
        assert result is None

    def test_set_and_get(self, cache: AnalysisCache[MockAdvice], advice: MockAdvice):
        """set() and get() should work together."""
        cache.set("hash-1", advice)
        
        result = cache.get("hash-1")
        
        assert result is advice

    def test_set_overwrites_existing(self, cache: AnalysisCache[MockAdvice]):
        """set() should overwrite existing entries."""
        advice1 = MockAdvice(document_id="doc-1")
        advice2 = MockAdvice(document_id="doc-2")
        
        cache.set("hash-1", advice1)
        cache.set("hash-1", advice2)
        
        result = cache.get("hash-1")
        assert result is advice2

    def test_invalidate_removes_entry(
        self, cache: AnalysisCache[MockAdvice], advice: MockAdvice
    ):
        """invalidate() should remove entry."""
        cache.set("hash-1", advice)
        
        removed = cache.invalidate("hash-1")
        
        assert removed is True
        assert cache.get("hash-1") is None

    def test_invalidate_missing_returns_false(self, cache: AnalysisCache[MockAdvice]):
        """invalidate() should return False for missing entries."""
        removed = cache.invalidate("nonexistent")
        
        assert removed is False

    def test_invalidate_for_document(self, cache: AnalysisCache[MockAdvice]):
        """invalidate_for_document() should remove all entries for a doc."""
        cache.set("hash-1", MockAdvice(document_id="doc-1"))
        cache.set("hash-2", MockAdvice(document_id="doc-1"))
        cache.set("hash-3", MockAdvice(document_id="doc-2"))
        
        count = cache.invalidate_for_document("doc-1")
        
        assert count == 2
        assert cache.get("hash-1") is None
        assert cache.get("hash-2") is None
        assert cache.get("hash-3") is not None

    def test_invalidate_all(self, cache: AnalysisCache[MockAdvice]):
        """invalidate_all() should clear all entries."""
        cache.set("hash-1", make_advice("doc-1"))
        cache.set("hash-2", make_advice("doc-2"))
        cache.set("hash-3", make_advice("doc-3"))
        
        count = cache.invalidate_all()
        
        assert count == 3
        assert cache.size() == 0

    def test_contains(
        self, cache: AnalysisCache[MockAdvice], advice: MockAdvice
    ):
        """contains() should check presence."""
        cache.set("hash-1", advice)
        
        assert cache.contains("hash-1") is True
        assert cache.contains("hash-2") is False

    def test_size(self, cache: AnalysisCache[MockAdvice]):
        """size() should return entry count."""
        assert cache.size() == 0
        
        cache.set("hash-1", make_advice())
        assert cache.size() == 1
        
        cache.set("hash-2", make_advice())
        assert cache.size() == 2
        
        cache.invalidate("hash-1")
        assert cache.size() == 1

    def test_keys(self, cache: AnalysisCache[MockAdvice]):
        """keys() should return all snapshot hashes."""
        cache.set("hash-1", make_advice())
        cache.set("hash-2", make_advice())
        cache.set("hash-3", make_advice())
        
        keys = cache.keys()
        
        assert set(keys) == {"hash-1", "hash-2", "hash-3"}


# =============================================================================
# LRU Eviction Tests
# =============================================================================


class TestAnalysisCacheLRU:
    """Tests for LRU eviction behavior."""

    def test_evicts_oldest_when_full(self, small_cache: AnalysisCache[MockAdvice]):
        """Should evict LRU entry when at capacity."""
        small_cache.set("hash-1", make_advice())
        small_cache.set("hash-2", make_advice())
        small_cache.set("hash-3", make_advice())
        
        # This should evict hash-1
        small_cache.set("hash-4", make_advice())
        
        assert small_cache.size() == 3
        assert small_cache.get("hash-1") is None
        assert small_cache.get("hash-4") is not None

    def test_access_updates_lru_order(self, small_cache: AnalysisCache[MockAdvice]):
        """Accessing an entry should move it to MRU position."""
        small_cache.set("hash-1", make_advice())
        small_cache.set("hash-2", make_advice())
        small_cache.set("hash-3", make_advice())
        
        # Access hash-1, making it MRU
        small_cache.get("hash-1")
        
        # Add new entry - should evict hash-2 (now LRU)
        small_cache.set("hash-4", make_advice())
        
        assert small_cache.get("hash-1") is not None  # Still there
        assert small_cache.get("hash-2") is None      # Evicted
        assert small_cache.get("hash-3") is not None  # Still there
        assert small_cache.get("hash-4") is not None  # Added

    def test_eviction_tracks_stats(self, small_cache: AnalysisCache[MockAdvice]):
        """Evictions should be tracked in stats."""
        small_cache.set("hash-1", make_advice())
        small_cache.set("hash-2", make_advice())
        small_cache.set("hash-3", make_advice())
        small_cache.set("hash-4", make_advice())  # Evicts hash-1
        small_cache.set("hash-5", make_advice())  # Evicts hash-2
        
        assert small_cache.stats is not None
        assert small_cache.stats.evictions == 2


# =============================================================================
# TTL Expiration Tests
# =============================================================================


class TestAnalysisCacheTTL:
    """Tests for TTL expiration behavior."""

    def test_expired_entry_returns_none(self):
        """Expired entries should return None on get()."""
        cache: AnalysisCache[MockAdvice] = AnalysisCache(
            AnalysisCacheConfig(ttl_seconds=0.01)
        )
        cache.set("hash-1", make_advice())
        
        time.sleep(0.02)
        
        result = cache.get("hash-1")
        assert result is None

    def test_expired_entry_removed_from_cache(self):
        """Expired entries should be removed on get()."""
        cache: AnalysisCache[MockAdvice] = AnalysisCache(
            AnalysisCacheConfig(ttl_seconds=0.01)
        )
        cache.set("hash-1", make_advice())
        
        time.sleep(0.02)
        cache.get("hash-1")
        
        assert cache.contains("hash-1") is False

    def test_expiration_tracks_stats(self):
        """Expirations should be tracked in stats."""
        cache: AnalysisCache[MockAdvice] = AnalysisCache(
            AnalysisCacheConfig(ttl_seconds=0.01)
        )
        cache.set("hash-1", make_advice())
        
        time.sleep(0.02)
        cache.get("hash-1")
        
        assert cache.stats is not None
        assert cache.stats.expirations == 1

    def test_cleanup_expired(self):
        """cleanup_expired() should remove all expired entries."""
        cache: AnalysisCache[MockAdvice] = AnalysisCache(
            AnalysisCacheConfig(ttl_seconds=0.01)
        )
        cache.set("hash-1", make_advice())
        cache.set("hash-2", make_advice())
        
        time.sleep(0.02)
        
        count = cache.cleanup_expired()
        
        assert count == 2
        assert cache.size() == 0

    def test_no_expiry_when_ttl_zero(self):
        """TTL of 0 should mean no expiry."""
        cache: AnalysisCache[MockAdvice] = AnalysisCache(
            AnalysisCacheConfig(ttl_seconds=0)
        )
        cache.set("hash-1", make_advice())
        
        result = cache.get("hash-1")
        assert result is not None


# =============================================================================
# Statistics Tests
# =============================================================================


class TestAnalysisCacheStatsTracking:
    """Tests for cache statistics tracking."""

    def test_hit_tracked(
        self, cache: AnalysisCache[MockAdvice], advice: MockAdvice
    ):
        """Cache hits should be tracked."""
        cache.set("hash-1", advice)
        cache.get("hash-1")
        
        assert cache.stats is not None
        assert cache.stats.hits == 1

    def test_miss_tracked(self, cache: AnalysisCache[MockAdvice]):
        """Cache misses should be tracked."""
        cache.get("nonexistent")
        
        assert cache.stats is not None
        assert cache.stats.misses == 1

    def test_invalidation_tracked(
        self, cache: AnalysisCache[MockAdvice], advice: MockAdvice
    ):
        """Invalidations should be tracked."""
        cache.set("hash-1", advice)
        cache.invalidate("hash-1")
        
        assert cache.stats is not None
        assert cache.stats.invalidations == 1

    def test_stats_disabled(self, advice: MockAdvice):
        """Stats should be None when disabled."""
        cache: AnalysisCache[MockAdvice] = AnalysisCache(
            AnalysisCacheConfig(track_stats=False)
        )
        
        assert cache.stats is None
        
        # Operations should still work
        cache.set("hash-1", advice)
        result = cache.get("hash-1")
        assert result is advice


# =============================================================================
# get_or_analyze Tests
# =============================================================================


class TestAnalysisCacheGetOrAnalyze:
    """Tests for the get_or_analyze convenience method."""

    def test_returns_cached_value(
        self, cache: AnalysisCache[MockAdvice], snapshot: DocumentSnapshot
    ):
        """get_or_analyze should return cached value if present."""
        cached_advice = make_advice("cached-doc")
        snapshot_hash = compute_snapshot_hash(snapshot)
        cache.set(snapshot_hash, cached_advice)
        
        analyzer_called = False
        def analyzer(snap: DocumentSnapshot) -> MockAdvice:
            nonlocal analyzer_called
            analyzer_called = True
            return make_advice("fresh-doc")
        
        result = cache.get_or_analyze(snapshot, analyzer)
        
        assert result is cached_advice
        assert analyzer_called is False

    def test_analyzes_and_caches_on_miss(
        self, cache: AnalysisCache[MockAdvice], snapshot: DocumentSnapshot
    ):
        """get_or_analyze should call analyzer and cache result on miss."""
        fresh_advice = make_advice("fresh-doc")
        
        def analyzer(snap: DocumentSnapshot) -> MockAdvice:
            return fresh_advice
        
        result = cache.get_or_analyze(snapshot, analyzer)
        
        assert result is fresh_advice
        
        # Should now be cached
        snapshot_hash = compute_snapshot_hash(snapshot)
        assert cache.get(snapshot_hash) is fresh_advice

    def test_returns_none_when_analyzer_returns_none(
        self, cache: AnalysisCache[MockAdvice], snapshot: DocumentSnapshot
    ):
        """get_or_analyze should return None if analyzer returns None."""
        def analyzer(snap: DocumentSnapshot) -> MockAdvice | None:
            return None
        
        result = cache.get_or_analyze(snapshot, analyzer)
        
        assert result is None
        
        # Should not be cached
        snapshot_hash = compute_snapshot_hash(snapshot)
        assert cache.contains(snapshot_hash) is False


# =============================================================================
# Thread Safety Tests
# =============================================================================


class TestAnalysisCacheThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_access(self, cache: AnalysisCache[MockAdvice]):
        """Cache should handle concurrent access safely."""
        results: list[MockAdvice | None] = []
        errors: list[Exception] = []
        
        def writer(hash_prefix: str):
            try:
                for i in range(100):
                    cache.set(f"{hash_prefix}-{i}", make_advice(f"doc-{hash_prefix}-{i}"))
            except Exception as e:
                errors.append(e)
        
        def reader(hash_prefix: str):
            try:
                for i in range(100):
                    result = cache.get(f"{hash_prefix}-{i}")
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=writer, args=("w1",)),
            threading.Thread(target=writer, args=("w2",)),
            threading.Thread(target=reader, args=("w1",)),
            threading.Thread(target=reader, args=("w2",)),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0

    def test_concurrent_eviction(self):
        """Eviction should be thread-safe."""
        cache: AnalysisCache[MockAdvice] = AnalysisCache(
            AnalysisCacheConfig(max_entries=10)
        )
        errors: list[Exception] = []
        
        def writer(prefix: str):
            try:
                for i in range(50):
                    cache.set(f"{prefix}-{i}", make_advice(f"doc-{prefix}-{i}"))
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
        assert cache.size() <= 10
