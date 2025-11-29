"""Session store domain service.

Manages persistence of workspace state, settings, and unsaved document
snapshots. This is the domain layer abstraction for session persistence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..events import EventBus

if TYPE_CHECKING:  # pragma: no cover
    from ...services.settings import Settings, SettingsStore
    from ...services.unsaved_cache import UnsavedCache, UnsavedCacheStore

LOGGER = logging.getLogger(__name__)


class SessionStore:
    """Domain manager for session persistence.

    Handles saving and loading of workspace state, settings, and unsaved
    document snapshots. This manager does not own any file dialogs or
    UI concerns - it focuses purely on persistence operations.
    """

    def __init__(
        self,
        settings_store: SettingsStore | None,
        unsaved_cache_store: UnsavedCacheStore | None,
        event_bus: EventBus,
    ) -> None:
        """Initialize the session store.

        Args:
            settings_store: Store for persisting settings, or None.
            unsaved_cache_store: Store for unsaved document snapshots, or None.
            event_bus: The event bus for publishing events.
        """
        self._settings_store = settings_store
        self._unsaved_cache_store = unsaved_cache_store
        self._bus = event_bus
        self._current_path: Path | None = None

    # ------------------------------------------------------------------
    # Current Path Management
    # ------------------------------------------------------------------

    @property
    def current_path(self) -> Path | None:
        """Get the current document path."""
        return self._current_path

    def set_current_path(self, path: Path | str | None) -> None:
        """Set the current document path.

        Args:
            path: The path to set, or None to clear.
        """
        if path is None:
            self._current_path = None
        else:
            self._current_path = Path(path)

        LOGGER.debug("SessionStore.set_current_path: %s", self._current_path)

    # ------------------------------------------------------------------
    # Recent Files Management
    # ------------------------------------------------------------------

    def remember_recent_file(self, path: Path | str, settings: Settings) -> None:
        """Add a file to the recent files list.

        Updates the settings object and persists it.

        Args:
            path: The file path to remember.
            settings: The settings object to update.
        """
        normalized = str(Path(path).expanduser().resolve())

        # Build updated list with this file at front
        updated: list[str] = [normalized]
        for existing in settings.recent_files:
            candidate = str(Path(existing).expanduser().resolve())
            if candidate == normalized:
                continue
            updated.append(existing)
            if len(updated) >= 10:
                break

        settings.recent_files = updated
        settings.last_open_file = normalized

        LOGGER.debug(
            "SessionStore.remember_recent_file: %s, total=%d",
            normalized,
            len(updated),
        )

        self.persist_settings(settings)

    # ------------------------------------------------------------------
    # Settings Persistence
    # ------------------------------------------------------------------

    def persist_settings(self, settings: Settings | None) -> bool:
        """Persist settings to storage.

        Args:
            settings: The settings to persist, or None.

        Returns:
            True if settings were persisted successfully.
        """
        if settings is None:
            LOGGER.debug("SessionStore.persist_settings: skipping - no settings")
            return False

        if self._settings_store is None:
            LOGGER.debug("SessionStore.persist_settings: skipping - no store")
            return False

        try:
            self._settings_store.save(settings)
            LOGGER.debug("SessionStore.persist_settings: saved successfully")
            return True
        except Exception as exc:
            LOGGER.warning("SessionStore.persist_settings: failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Unsaved Cache Persistence
    # ------------------------------------------------------------------

    def persist_unsaved_cache(self, cache: UnsavedCache | None) -> bool:
        """Persist unsaved document cache to storage.

        Args:
            cache: The cache to persist, or None.

        Returns:
            True if cache was persisted successfully.
        """
        if cache is None:
            return False

        if self._unsaved_cache_store is None:
            return False

        try:
            self._unsaved_cache_store.save(cache)
            LOGGER.debug("SessionStore.persist_unsaved_cache: saved successfully")
            return True
        except Exception as exc:
            LOGGER.debug("SessionStore.persist_unsaved_cache: failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Workspace State Sync
    # ------------------------------------------------------------------

    def sync_workspace_state(
        self,
        workspace_state: dict[str, Any],
        settings: Settings | None,
        *,
        persist: bool = True,
    ) -> bool:
        """Sync workspace state to settings.

        Updates the settings object with current workspace state and
        optionally persists it.

        Args:
            workspace_state: Serialized workspace state dict.
            settings: The settings object to update.
            persist: Whether to persist settings after update.

        Returns:
            True if state was synced (and persisted if requested).
        """
        if settings is None:
            LOGGER.debug("SessionStore.sync_workspace_state: skipping - no settings")
            return False

        # Update settings with workspace state
        settings.open_tabs = workspace_state.get("open_tabs", [])
        settings.active_tab_id = workspace_state.get("active_tab_id")

        untitled_counter = workspace_state.get("untitled_counter")
        if isinstance(untitled_counter, int):
            settings.next_untitled_index = untitled_counter

        tab_ids = [t.get("tab_id", "?") for t in settings.open_tabs]
        LOGGER.debug(
            "SessionStore.sync_workspace_state: %d tabs (ids=%s), active=%s",
            len(settings.open_tabs),
            tab_ids,
            settings.active_tab_id,
        )

        if persist:
            return self.persist_settings(settings)

        return True

    # ------------------------------------------------------------------
    # Snapshot Management
    # ------------------------------------------------------------------

    def get_unsaved_snapshot(
        self,
        cache: UnsavedCache | None,
        path: Path | str | None,
        tab_id: str | None,
    ) -> dict[str, Any] | None:
        """Get an unsaved document snapshot.

        Args:
            cache: The unsaved cache to search.
            path: The file path, or None for untitled.
            tab_id: The tab ID for untitled documents.

        Returns:
            The snapshot dict if found, or None.
        """
        if cache is None:
            return None

        if path is not None:
            # Look up by path
            key = self.normalize_snapshot_key(path, tab_id)
            snapshots = cache.unsaved_snapshots or {}
            snapshot = snapshots.get(key)
            if isinstance(snapshot, dict) and "text" in snapshot:
                return snapshot
        else:
            # Look up untitled by tab_id
            if tab_id and cache.untitled_snapshots:
                snapshot = cache.untitled_snapshots.get(tab_id)
                if isinstance(snapshot, dict) and "text" in snapshot:
                    return snapshot

            # Fall back to legacy unsaved_snapshot
            snapshot = cache.unsaved_snapshot
            if isinstance(snapshot, dict) and "text" in snapshot:
                return snapshot

        return None

    def clear_unsaved_snapshot(
        self,
        cache: UnsavedCache | None,
        path: Path | str | None,
        tab_id: str | None,
    ) -> bool:
        """Clear an unsaved document snapshot.

        Args:
            cache: The unsaved cache to modify.
            path: The file path, or None for untitled.
            tab_id: The tab ID for untitled documents.

        Returns:
            True if a snapshot was cleared.
        """
        if cache is None:
            return False

        cleared = False

        if path is not None:
            key = self.normalize_snapshot_key(path, tab_id)
            snapshots = cache.unsaved_snapshots or {}
            if key in snapshots:
                del snapshots[key]
                cache.unsaved_snapshots = snapshots
                cleared = True
        else:
            if tab_id and cache.untitled_snapshots and tab_id in cache.untitled_snapshots:
                untitled = dict(cache.untitled_snapshots)
                del untitled[tab_id]
                cache.untitled_snapshots = untitled
                cleared = True

        return cleared

    def cleanup_orphan_snapshots(
        self,
        cache: UnsavedCache | None,
        open_tabs: list[dict[str, Any]] | None,
    ) -> bool:
        """Remove snapshots for tabs that are no longer open.

        Args:
            cache: The unsaved cache to clean.
            open_tabs: List of currently open tab entries.

        Returns:
            True if any snapshots were removed.
        """
        if cache is None:
            return False

        entries = [e for e in (open_tabs or []) if isinstance(e, Mapping)]

        # Collect active paths and tab IDs
        active_paths: set[str] = set()
        active_tab_ids: set[str] = set()

        for entry in entries:
            tab_id = entry.get("tab_id")
            if tab_id:
                active_tab_ids.add(str(tab_id))

            path_value = entry.get("path")
            if path_value:
                normalized = self._normalize_path(path_value)
                if normalized:
                    active_paths.add(normalized)

        # Clean file snapshots
        removed_file = False
        snapshots = dict(cache.unsaved_snapshots or {})
        for key in list(snapshots):
            if key not in active_paths:
                snapshots.pop(key, None)
                removed_file = True
        if removed_file:
            cache.unsaved_snapshots = snapshots

        # Clean untitled snapshots
        removed_untitled = False
        untitled = dict(cache.untitled_snapshots or {})
        for tid in list(untitled):
            if tid not in active_tab_ids:
                untitled.pop(tid, None)
                removed_untitled = True
        if removed_untitled:
            cache.untitled_snapshots = untitled

        return removed_file or removed_untitled

    # ------------------------------------------------------------------
    # Static Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_snapshot_key(path: Path | str, tab_id: str | None = None) -> str:
        """Normalize a path for use as a snapshot key.

        Args:
            path: The file path.
            tab_id: Optional tab ID (unused, for interface compatibility).

        Returns:
            Normalized path string.
        """
        return str(Path(path).expanduser().resolve())

    @staticmethod
    def _normalize_path(value: Any) -> str | None:
        """Normalize a path value to string or None."""
        try:
            return str(Path(value).expanduser().resolve())
        except Exception:
            try:
                return str(Path(str(value)).expanduser().resolve())
            except Exception:
                return None

    @staticmethod
    def infer_language(path: Path | str | None) -> str:
        """Infer document language from file extension.

        Args:
            path: The file path, or None.

        Returns:
            Language identifier string.
        """
        if path is None:
            return "markdown"

        suffix = Path(path).suffix.lower()

        if suffix in {".md", ".markdown"}:
            return "markdown"
        if suffix in {".yaml", ".yml"}:
            return "yaml"
        if suffix == ".json":
            return "json"
        if suffix in {".txt", ""}:
            return "text"

        return "plain"


__all__ = ["SessionStore"]
