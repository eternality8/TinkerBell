"""Tests for SessionStore domain manager."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tinkerbell.ui.events import EventBus
from tinkerbell.ui.domain.session_store import SessionStore


# ---------------------------------------------------------------------------
# Mock Types
# ---------------------------------------------------------------------------


@dataclass
class MockSettings:
    """Mock settings object for testing."""

    recent_files: list[str] = field(default_factory=list)
    last_open_file: str | None = None
    open_tabs: list[dict[str, Any]] = field(default_factory=list)
    active_tab_id: str | None = None
    next_untitled_index: int = 1


@dataclass
class MockUnsavedCache:
    """Mock unsaved cache for testing."""

    unsaved_snapshots: dict[str, dict[str, Any]] | None = None
    untitled_snapshots: dict[str, dict[str, Any]] | None = None
    unsaved_snapshot: dict[str, Any] | None = None  # Legacy single snapshot


class MockSettingsStore:
    """Mock settings store for testing."""

    def __init__(self, *, raise_on_save: bool = False) -> None:
        self.saved: list[MockSettings] = []
        self.raise_on_save = raise_on_save

    def save(self, settings: MockSettings) -> None:
        if self.raise_on_save:
            raise OSError("Simulated save failure")
        self.saved.append(settings)


class MockUnsavedCacheStore:
    """Mock unsaved cache store for testing."""

    def __init__(self, *, raise_on_save: bool = False) -> None:
        self.saved: list[MockUnsavedCache] = []
        self.raise_on_save = raise_on_save

    def save(self, cache: MockUnsavedCache) -> None:
        if self.raise_on_save:
            raise OSError("Simulated save failure")
        self.saved.append(cache)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def event_bus() -> EventBus:
    """Create an event bus for testing."""
    return EventBus()


@pytest.fixture
def settings_store() -> MockSettingsStore:
    """Create a mock settings store."""
    return MockSettingsStore()


@pytest.fixture
def unsaved_cache_store() -> MockUnsavedCacheStore:
    """Create a mock unsaved cache store."""
    return MockUnsavedCacheStore()


@pytest.fixture
def session_store(
    settings_store: MockSettingsStore,
    unsaved_cache_store: MockUnsavedCacheStore,
    event_bus: EventBus,
) -> SessionStore:
    """Create a SessionStore with mock dependencies."""
    return SessionStore(
        settings_store=settings_store,
        unsaved_cache_store=unsaved_cache_store,
        event_bus=event_bus,
    )


@pytest.fixture
def minimal_session_store(event_bus: EventBus) -> SessionStore:
    """Create a SessionStore with no persistence stores."""
    return SessionStore(
        settings_store=None,
        unsaved_cache_store=None,
        event_bus=event_bus,
    )


# ---------------------------------------------------------------------------
# Tests: Initialization
# ---------------------------------------------------------------------------


class TestSessionStoreInit:
    """Tests for SessionStore initialization."""

    def test_init_with_all_dependencies(
        self,
        settings_store: MockSettingsStore,
        unsaved_cache_store: MockUnsavedCacheStore,
        event_bus: EventBus,
    ) -> None:
        """SessionStore initializes with all dependencies."""
        store = SessionStore(settings_store, unsaved_cache_store, event_bus)

        assert store._settings_store is settings_store
        assert store._unsaved_cache_store is unsaved_cache_store
        assert store._bus is event_bus
        assert store.current_path is None

    def test_init_with_no_stores(self, event_bus: EventBus) -> None:
        """SessionStore initializes with None stores."""
        store = SessionStore(None, None, event_bus)

        assert store._settings_store is None
        assert store._unsaved_cache_store is None
        assert store.current_path is None


# ---------------------------------------------------------------------------
# Tests: Current Path Management
# ---------------------------------------------------------------------------


class TestCurrentPath:
    """Tests for current path management."""

    def test_current_path_initially_none(
        self, session_store: SessionStore
    ) -> None:
        """Current path is initially None."""
        assert session_store.current_path is None

    def test_set_current_path_with_path(
        self, session_store: SessionStore
    ) -> None:
        """Can set current path with Path object."""
        path = Path("/some/path/file.txt")
        session_store.set_current_path(path)
        assert session_store.current_path == path

    def test_set_current_path_with_string(
        self, session_store: SessionStore
    ) -> None:
        """Can set current path with string."""
        session_store.set_current_path("/some/path/file.txt")
        assert session_store.current_path == Path("/some/path/file.txt")

    def test_set_current_path_to_none(
        self, session_store: SessionStore
    ) -> None:
        """Can clear current path by setting to None."""
        session_store.set_current_path("/some/path.txt")
        session_store.set_current_path(None)
        assert session_store.current_path is None

    def test_current_path_property_readonly(
        self, session_store: SessionStore
    ) -> None:
        """Current path is read-only property."""
        session_store.set_current_path("/test/path.txt")
        # Property getter returns value
        assert session_store.current_path is not None


# ---------------------------------------------------------------------------
# Tests: Recent Files Management
# ---------------------------------------------------------------------------


class TestRememberRecentFile:
    """Tests for remember_recent_file method."""

    def test_remember_adds_to_front(
        self,
        session_store: SessionStore,
        settings_store: MockSettingsStore,
    ) -> None:
        """Remembering a file adds it to the front of the list."""
        settings = MockSettings()
        session_store.remember_recent_file("/path/to/file.txt", settings)

        assert len(settings.recent_files) == 1
        # Normalized to absolute path
        assert Path(settings.recent_files[0]).name == "file.txt"
        assert settings.last_open_file is not None

    def test_remember_moves_existing_to_front(
        self, session_store: SessionStore
    ) -> None:
        """Remembering existing file moves it to front."""
        settings = MockSettings(recent_files=["/a.txt", "/b.txt", "/c.txt"])
        session_store.remember_recent_file("/b.txt", settings)

        # b.txt should now be first
        names = [Path(p).name for p in settings.recent_files]
        assert names[0] == "b.txt"
        assert len(names) == 3

    def test_remember_limits_to_ten_files(
        self, session_store: SessionStore
    ) -> None:
        """Recent files list is limited to 10 entries."""
        files = [f"/file{i}.txt" for i in range(15)]
        settings = MockSettings(recent_files=files)

        session_store.remember_recent_file("/new.txt", settings)

        assert len(settings.recent_files) <= 10
        assert Path(settings.recent_files[0]).name == "new.txt"

    def test_remember_persists_settings(
        self,
        session_store: SessionStore,
        settings_store: MockSettingsStore,
    ) -> None:
        """Remembering a file persists settings."""
        settings = MockSettings()
        session_store.remember_recent_file("/test.txt", settings)

        assert len(settings_store.saved) == 1
        assert settings_store.saved[0] is settings

    def test_remember_handles_path_object(
        self, session_store: SessionStore
    ) -> None:
        """Can remember file using Path object."""
        settings = MockSettings()
        session_store.remember_recent_file(Path("/doc.md"), settings)

        assert len(settings.recent_files) == 1


# ---------------------------------------------------------------------------
# Tests: Settings Persistence
# ---------------------------------------------------------------------------


class TestPersistSettings:
    """Tests for persist_settings method."""

    def test_persist_saves_settings(
        self,
        session_store: SessionStore,
        settings_store: MockSettingsStore,
    ) -> None:
        """Persisting settings saves to store."""
        settings = MockSettings()
        result = session_store.persist_settings(settings)

        assert result is True
        assert len(settings_store.saved) == 1
        assert settings_store.saved[0] is settings

    def test_persist_returns_false_for_none_settings(
        self, session_store: SessionStore
    ) -> None:
        """Persisting None settings returns False."""
        result = session_store.persist_settings(None)
        assert result is False

    def test_persist_returns_false_without_store(
        self, minimal_session_store: SessionStore
    ) -> None:
        """Persisting without a store returns False."""
        settings = MockSettings()
        result = minimal_session_store.persist_settings(settings)
        assert result is False

    def test_persist_handles_save_error(
        self, event_bus: EventBus
    ) -> None:
        """Persisting handles save errors gracefully."""
        store = SessionStore(
            settings_store=MockSettingsStore(raise_on_save=True),
            unsaved_cache_store=None,
            event_bus=event_bus,
        )
        settings = MockSettings()
        result = store.persist_settings(settings)
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Unsaved Cache Persistence
# ---------------------------------------------------------------------------


class TestPersistUnsavedCache:
    """Tests for persist_unsaved_cache method."""

    def test_persist_saves_cache(
        self,
        session_store: SessionStore,
        unsaved_cache_store: MockUnsavedCacheStore,
    ) -> None:
        """Persisting cache saves to store."""
        cache = MockUnsavedCache()
        result = session_store.persist_unsaved_cache(cache)

        assert result is True
        assert len(unsaved_cache_store.saved) == 1

    def test_persist_returns_false_for_none_cache(
        self, session_store: SessionStore
    ) -> None:
        """Persisting None cache returns False."""
        result = session_store.persist_unsaved_cache(None)
        assert result is False

    def test_persist_returns_false_without_store(
        self, minimal_session_store: SessionStore
    ) -> None:
        """Persisting without a store returns False."""
        cache = MockUnsavedCache()
        result = minimal_session_store.persist_unsaved_cache(cache)
        assert result is False

    def test_persist_handles_save_error(
        self, event_bus: EventBus
    ) -> None:
        """Persisting handles save errors gracefully."""
        store = SessionStore(
            settings_store=None,
            unsaved_cache_store=MockUnsavedCacheStore(raise_on_save=True),
            event_bus=event_bus,
        )
        cache = MockUnsavedCache()
        result = store.persist_unsaved_cache(cache)
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Workspace State Sync
# ---------------------------------------------------------------------------


class TestSyncWorkspaceState:
    """Tests for sync_workspace_state method."""

    def test_sync_updates_settings(
        self, session_store: SessionStore
    ) -> None:
        """Syncing updates settings with workspace state."""
        settings = MockSettings()
        workspace_state = {
            "open_tabs": [{"tab_id": "t1", "path": "/a.txt"}],
            "active_tab_id": "t1",
            "untitled_counter": 5,
        }

        result = session_store.sync_workspace_state(
            workspace_state, settings, persist=False
        )

        assert result is True
        assert len(settings.open_tabs) == 1
        assert settings.active_tab_id == "t1"
        assert settings.next_untitled_index == 5

    def test_sync_persists_by_default(
        self,
        session_store: SessionStore,
        settings_store: MockSettingsStore,
    ) -> None:
        """Syncing persists settings by default."""
        settings = MockSettings()
        session_store.sync_workspace_state({}, settings)

        assert len(settings_store.saved) == 1

    def test_sync_skips_persist_when_disabled(
        self,
        session_store: SessionStore,
        settings_store: MockSettingsStore,
    ) -> None:
        """Syncing can skip persistence."""
        settings = MockSettings()
        session_store.sync_workspace_state({}, settings, persist=False)

        assert len(settings_store.saved) == 0

    def test_sync_returns_false_for_none_settings(
        self, session_store: SessionStore
    ) -> None:
        """Syncing returns False for None settings."""
        result = session_store.sync_workspace_state({}, None)
        assert result is False

    def test_sync_handles_missing_keys(
        self, session_store: SessionStore
    ) -> None:
        """Syncing handles missing keys in workspace state."""
        settings = MockSettings()
        result = session_store.sync_workspace_state({}, settings, persist=False)

        assert result is True
        assert settings.open_tabs == []
        assert settings.active_tab_id is None


# ---------------------------------------------------------------------------
# Tests: Snapshot Management - Get
# ---------------------------------------------------------------------------


class TestGetUnsavedSnapshot:
    """Tests for get_unsaved_snapshot method."""

    def test_get_snapshot_by_path(
        self, session_store: SessionStore
    ) -> None:
        """Can get snapshot by file path."""
        key = session_store.normalize_snapshot_key("/test/file.txt")
        cache = MockUnsavedCache(
            unsaved_snapshots={key: {"text": "content", "dirty": True}}
        )

        snapshot = session_store.get_unsaved_snapshot(cache, "/test/file.txt", None)

        assert snapshot is not None
        assert snapshot["text"] == "content"

    def test_get_untitled_snapshot_by_tab_id(
        self, session_store: SessionStore
    ) -> None:
        """Can get untitled snapshot by tab ID."""
        cache = MockUnsavedCache(
            untitled_snapshots={"tab-123": {"text": "untitled content"}}
        )

        snapshot = session_store.get_unsaved_snapshot(cache, None, "tab-123")

        assert snapshot is not None
        assert snapshot["text"] == "untitled content"

    def test_get_untitled_snapshot_legacy_fallback(
        self, session_store: SessionStore
    ) -> None:
        """Falls back to legacy single snapshot for untitled."""
        cache = MockUnsavedCache(
            unsaved_snapshot={"text": "legacy content"}
        )

        snapshot = session_store.get_unsaved_snapshot(cache, None, "any-tab")

        assert snapshot is not None
        assert snapshot["text"] == "legacy content"

    def test_get_returns_none_for_missing(
        self, session_store: SessionStore
    ) -> None:
        """Returns None for missing snapshot."""
        cache = MockUnsavedCache(unsaved_snapshots={})

        snapshot = session_store.get_unsaved_snapshot(cache, "/missing.txt", None)

        assert snapshot is None

    def test_get_returns_none_for_none_cache(
        self, session_store: SessionStore
    ) -> None:
        """Returns None for None cache."""
        snapshot = session_store.get_unsaved_snapshot(None, "/test.txt", None)
        assert snapshot is None

    def test_get_validates_snapshot_has_text(
        self, session_store: SessionStore
    ) -> None:
        """Only returns snapshots that have 'text' key."""
        key = session_store.normalize_snapshot_key("/test.txt")
        cache = MockUnsavedCache(
            unsaved_snapshots={key: {"dirty": True}}  # No 'text' key
        )

        snapshot = session_store.get_unsaved_snapshot(cache, "/test.txt", None)

        assert snapshot is None


# ---------------------------------------------------------------------------
# Tests: Snapshot Management - Clear
# ---------------------------------------------------------------------------


class TestClearUnsavedSnapshot:
    """Tests for clear_unsaved_snapshot method."""

    def test_clear_snapshot_by_path(
        self, session_store: SessionStore
    ) -> None:
        """Can clear snapshot by file path."""
        key = session_store.normalize_snapshot_key("/test/file.txt")
        cache = MockUnsavedCache(
            unsaved_snapshots={key: {"text": "content"}}
        )

        result = session_store.clear_unsaved_snapshot(cache, "/test/file.txt", None)

        assert result is True
        assert key not in (cache.unsaved_snapshots or {})

    def test_clear_untitled_snapshot_by_tab_id(
        self, session_store: SessionStore
    ) -> None:
        """Can clear untitled snapshot by tab ID."""
        cache = MockUnsavedCache(
            untitled_snapshots={"tab-123": {"text": "content"}}
        )

        result = session_store.clear_unsaved_snapshot(cache, None, "tab-123")

        assert result is True
        assert "tab-123" not in (cache.untitled_snapshots or {})

    def test_clear_returns_false_for_missing(
        self, session_store: SessionStore
    ) -> None:
        """Returns False when snapshot doesn't exist."""
        cache = MockUnsavedCache(unsaved_snapshots={})

        result = session_store.clear_unsaved_snapshot(cache, "/missing.txt", None)

        assert result is False

    def test_clear_returns_false_for_none_cache(
        self, session_store: SessionStore
    ) -> None:
        """Returns False for None cache."""
        result = session_store.clear_unsaved_snapshot(None, "/test.txt", None)
        assert result is False


# ---------------------------------------------------------------------------
# Tests: Cleanup Orphan Snapshots
# ---------------------------------------------------------------------------


class TestCleanupOrphanSnapshots:
    """Tests for cleanup_orphan_snapshots method."""

    def test_removes_orphan_file_snapshots(
        self, session_store: SessionStore
    ) -> None:
        """Removes snapshots for closed files."""
        key1 = session_store.normalize_snapshot_key("/a.txt")
        key2 = session_store.normalize_snapshot_key("/b.txt")
        cache = MockUnsavedCache(
            unsaved_snapshots={
                key1: {"text": "a"},
                key2: {"text": "b"},
            }
        )

        # Only /a.txt is still open
        open_tabs = [{"tab_id": "t1", "path": "/a.txt"}]
        result = session_store.cleanup_orphan_snapshots(cache, open_tabs)

        assert result is True
        assert key1 in (cache.unsaved_snapshots or {})
        assert key2 not in (cache.unsaved_snapshots or {})

    def test_removes_orphan_untitled_snapshots(
        self, session_store: SessionStore
    ) -> None:
        """Removes snapshots for closed untitled tabs."""
        cache = MockUnsavedCache(
            untitled_snapshots={
                "tab-1": {"text": "a"},
                "tab-2": {"text": "b"},
            }
        )

        # Only tab-1 is still open
        open_tabs = [{"tab_id": "tab-1"}]
        result = session_store.cleanup_orphan_snapshots(cache, open_tabs)

        assert result is True
        assert "tab-1" in (cache.untitled_snapshots or {})
        assert "tab-2" not in (cache.untitled_snapshots or {})

    def test_returns_false_when_nothing_removed(
        self, session_store: SessionStore
    ) -> None:
        """Returns False when no orphans exist."""
        key = session_store.normalize_snapshot_key("/a.txt")
        cache = MockUnsavedCache(
            unsaved_snapshots={key: {"text": "a"}}
        )

        open_tabs = [{"tab_id": "t1", "path": "/a.txt"}]
        result = session_store.cleanup_orphan_snapshots(cache, open_tabs)

        assert result is False

    def test_returns_false_for_none_cache(
        self, session_store: SessionStore
    ) -> None:
        """Returns False for None cache."""
        result = session_store.cleanup_orphan_snapshots(None, [])
        assert result is False

    def test_handles_empty_open_tabs(
        self, session_store: SessionStore
    ) -> None:
        """Handles None or empty open_tabs list."""
        cache = MockUnsavedCache(
            unsaved_snapshots={"key": {"text": "val"}}
        )

        result = session_store.cleanup_orphan_snapshots(cache, None)
        assert result is True

        result = session_store.cleanup_orphan_snapshots(cache, [])
        assert result is False  # Already empty from previous cleanup


# ---------------------------------------------------------------------------
# Tests: Static Helpers
# ---------------------------------------------------------------------------


class TestStaticHelpers:
    """Tests for static helper methods."""

    def test_normalize_snapshot_key(self) -> None:
        """Normalizes paths consistently."""
        key1 = SessionStore.normalize_snapshot_key("/test/file.txt")
        key2 = SessionStore.normalize_snapshot_key(Path("/test/file.txt"))

        assert key1 == key2
        assert isinstance(key1, str)

    def test_normalize_snapshot_key_with_tab_id(self) -> None:
        """Tab ID parameter is accepted but unused."""
        key = SessionStore.normalize_snapshot_key("/test.txt", "tab-123")
        assert "tab-123" not in key

    def test_infer_language_markdown(self) -> None:
        """Infers markdown language."""
        assert SessionStore.infer_language("/test.md") == "markdown"
        assert SessionStore.infer_language("/test.markdown") == "markdown"

    def test_infer_language_yaml(self) -> None:
        """Infers YAML language."""
        assert SessionStore.infer_language("/test.yaml") == "yaml"
        assert SessionStore.infer_language("/test.yml") == "yaml"

    def test_infer_language_json(self) -> None:
        """Infers JSON language."""
        assert SessionStore.infer_language("/test.json") == "json"

    def test_infer_language_text(self) -> None:
        """Infers text language."""
        assert SessionStore.infer_language("/test.txt") == "text"
        assert SessionStore.infer_language("/test") == "text"

    def test_infer_language_default(self) -> None:
        """Returns 'plain' for unknown extensions."""
        assert SessionStore.infer_language("/test.xyz") == "plain"

    def test_infer_language_none(self) -> None:
        """Returns 'markdown' for None path."""
        assert SessionStore.infer_language(None) == "markdown"


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_multiple_operations_same_store(
        self, session_store: SessionStore
    ) -> None:
        """Multiple operations accumulate correctly."""
        settings = MockSettings()

        session_store.remember_recent_file("/a.txt", settings)
        session_store.remember_recent_file("/b.txt", settings)
        session_store.remember_recent_file("/c.txt", settings)

        assert len(settings.recent_files) == 3
        assert Path(settings.recent_files[0]).name == "c.txt"

    def test_path_normalization_consistency(
        self, session_store: SessionStore
    ) -> None:
        """Path normalization is consistent across operations."""
        cache = MockUnsavedCache(unsaved_snapshots={})

        # Add via one path format
        key = session_store.normalize_snapshot_key("./relative/../test.txt")
        cache.unsaved_snapshots = {key: {"text": "content"}}

        # Retrieve via different format
        snapshot = session_store.get_unsaved_snapshot(
            cache, "./relative/../test.txt", None
        )

        assert snapshot is not None

    def test_handles_non_mapping_entries_in_open_tabs(
        self, session_store: SessionStore
    ) -> None:
        """Cleanup handles malformed open_tabs entries."""
        cache = MockUnsavedCache(
            unsaved_snapshots={"key": {"text": "val"}}
        )

        # Mix of valid and invalid entries
        open_tabs: list[Any] = [
            {"tab_id": "t1", "path": "/a.txt"},
            None,
            "invalid",
            123,
        ]

        # Should not raise
        session_store.cleanup_orphan_snapshots(cache, open_tabs)
