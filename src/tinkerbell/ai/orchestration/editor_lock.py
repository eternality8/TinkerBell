"""Editor Lock System for AI turn coordination.

Provides locking mechanism to prevent user edits during AI operations,
with visual feedback and clean cancellation support.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Protocol, Sequence

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Lock State
# -----------------------------------------------------------------------------


class LockState(Enum):
    """State of the editor lock."""

    UNLOCKED = auto()
    LOCKING = auto()  # Transition state
    LOCKED = auto()
    UNLOCKING = auto()  # Transition state


class LockReason(Enum):
    """Reason for acquiring the lock."""

    AI_TURN = auto()
    REVIEW_PENDING = auto()
    TRANSACTION = auto()


# -----------------------------------------------------------------------------
# Lock Event Callbacks
# -----------------------------------------------------------------------------


class LockStateListener(Protocol):
    """Callback for lock state changes."""

    def __call__(self, state: LockState, reason: LockReason | None) -> None:
        """Called when lock state changes."""
        ...


class LockStatusUpdater(Protocol):
    """Callback for updating status bar with lock info."""

    def __call__(self, message: str, is_locked: bool) -> None:
        """Update status bar with lock message."""
        ...


# -----------------------------------------------------------------------------
# Tab Lock Protocol
# -----------------------------------------------------------------------------


class LockableTab(Protocol):
    """Protocol for tabs that can be locked."""

    @property
    def id(self) -> str:
        """Tab identifier."""
        ...

    def set_readonly(self, readonly: bool) -> None:
        """Set the tab's read-only state."""
        ...

    def is_readonly(self) -> bool:
        """Check if the tab is read-only."""
        ...


class TabProvider(Protocol):
    """Protocol for accessing tabs."""

    def get_all_tabs(self) -> Sequence[LockableTab]:
        """Get all open tabs."""
        ...

    def get_active_tab(self) -> LockableTab | None:
        """Get the currently active tab."""
        ...


# -----------------------------------------------------------------------------
# Editor Lock Manager
# -----------------------------------------------------------------------------


@dataclass
class LockSession:
    """Represents an active lock session."""

    session_id: str
    reason: LockReason
    locked_tabs: set[str] = field(default_factory=set)
    acquired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    timeout_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EditorLockManager:
    """Manages editor locking during AI operations.

    The lock manager coordinates read-only state across all editor tabs
    during AI turns to prevent concurrent user edits. It provides:
    - Lock acquisition with reason tracking
    - Visual feedback via status updates
    - Clean cancellation with state restoration
    - Timeout support for safety
    """

    def __init__(
        self,
        *,
        tab_provider: TabProvider | None = None,
        status_updater: LockStatusUpdater | None = None,
        on_state_change: LockStateListener | None = None,
    ) -> None:
        """Initialize the lock manager.

        Args:
            tab_provider: Provider for accessing tabs to lock.
            status_updater: Callback for status bar updates.
            on_state_change: Callback for lock state changes.
        """
        self._tab_provider = tab_provider
        self._status_updater = status_updater
        self._on_state_change = on_state_change

        self._lock = threading.RLock()
        self._state = LockState.UNLOCKED
        self._active_session: LockSession | None = None
        self._session_counter = 0
        self._previous_tab_states: dict[str, bool] = {}  # tab_id -> was_readonly

    # ------------------------------------------------------------------
    # Public State
    # ------------------------------------------------------------------

    @property
    def state(self) -> LockState:
        """Current lock state."""
        with self._lock:
            return self._state

    @property
    def is_locked(self) -> bool:
        """Check if the editor is currently locked."""
        with self._lock:
            return self._state == LockState.LOCKED

    @property
    def active_session(self) -> LockSession | None:
        """Get the active lock session, if any."""
        with self._lock:
            return self._active_session

    # ------------------------------------------------------------------
    # Lock Acquisition
    # ------------------------------------------------------------------

    def acquire(
        self,
        reason: LockReason = LockReason.AI_TURN,
        *,
        timeout_seconds: float | None = 300.0,
        metadata: dict[str, Any] | None = None,
    ) -> LockSession | None:
        """Acquire the editor lock.

        Args:
            reason: Why the lock is being acquired.
            timeout_seconds: Auto-release timeout (None for no timeout).
            metadata: Additional metadata for the session.

        Returns:
            LockSession if acquired, None if already locked.
        """
        with self._lock:
            if self._state != LockState.UNLOCKED:
                LOGGER.warning(
                    "Cannot acquire lock: current state is %s",
                    self._state.name,
                )
                return None

            # Transition to locking
            self._set_state(LockState.LOCKING, reason)

            # Create session
            self._session_counter += 1
            session = LockSession(
                session_id=f"lock-{self._session_counter}",
                reason=reason,
                timeout_seconds=timeout_seconds,
                metadata=metadata or {},
            )
            self._active_session = session

            # Lock all tabs
            self._lock_all_tabs(session)

            # Transition to locked
            self._set_state(LockState.LOCKED, reason)
            self._update_status(f"Editor locked: {self._reason_message(reason)}", True)

            LOGGER.info(
                "Lock acquired: session=%s, reason=%s, tabs=%d",
                session.session_id,
                reason.name,
                len(session.locked_tabs),
            )

            return session

    def release(self, session_id: str | None = None) -> bool:
        """Release the editor lock.

        Args:
            session_id: Optional session ID to verify ownership.

        Returns:
            True if released, False if not locked or wrong session.
        """
        with self._lock:
            if self._state != LockState.LOCKED:
                LOGGER.debug("Cannot release: not locked (state=%s)", self._state.name)
                return False

            if session_id and self._active_session and self._active_session.session_id != session_id:
                LOGGER.warning(
                    "Cannot release: session mismatch (expected=%s, got=%s)",
                    self._active_session.session_id,
                    session_id,
                )
                return False

            # Transition to unlocking
            reason = self._active_session.reason if self._active_session else None
            self._set_state(LockState.UNLOCKING, reason)

            # Unlock all tabs
            self._unlock_all_tabs()

            # Clear session
            old_session = self._active_session
            self._active_session = None

            # Transition to unlocked
            self._set_state(LockState.UNLOCKED, None)
            self._update_status("Editor unlocked", False)

            LOGGER.info(
                "Lock released: session=%s",
                old_session.session_id if old_session else "none",
            )

            return True

    def force_release(self) -> bool:
        """Force release the lock regardless of state.

        Use this for error recovery or cancellation.

        Returns:
            True if any lock was released.
        """
        with self._lock:
            was_locked = self._state in (LockState.LOCKED, LockState.LOCKING)

            # Restore all tab states
            self._unlock_all_tabs()
            self._previous_tab_states.clear()

            # Clear session
            self._active_session = None

            # Force to unlocked
            self._state = LockState.UNLOCKED
            self._notify_state_change(LockState.UNLOCKED, None)
            self._update_status("Editor unlocked (forced)", False)

            if was_locked:
                LOGGER.warning("Lock force-released")

            return was_locked

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(self) -> bool:
        """Cancel the current AI operation and release the lock.

        This is the user-triggered cancel action.

        Returns:
            True if cancelled, False if nothing to cancel.
        """
        with self._lock:
            if self._state == LockState.UNLOCKED:
                return False

            LOGGER.info("User cancelled AI operation")
            return self.force_release()

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "EditorLockManager":
        """Context manager entry - acquire lock."""
        self.acquire()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - release lock."""
        self.release()

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _lock_all_tabs(self, session: LockSession) -> None:
        """Lock all tabs and remember their previous state."""
        if not self._tab_provider:
            return

        for tab in self._tab_provider.get_all_tabs():
            try:
                # Remember previous state
                was_readonly = tab.is_readonly()
                self._previous_tab_states[tab.id] = was_readonly

                # Lock the tab
                if not was_readonly:
                    tab.set_readonly(True)
                    session.locked_tabs.add(tab.id)
            except Exception:
                LOGGER.debug("Failed to lock tab %s", tab.id, exc_info=True)

    def _unlock_all_tabs(self) -> None:
        """Unlock all tabs, restoring their previous state."""
        if not self._tab_provider:
            return

        for tab in self._tab_provider.get_all_tabs():
            try:
                # Restore previous state
                was_readonly = self._previous_tab_states.get(tab.id, False)
                if not was_readonly:
                    tab.set_readonly(False)
            except Exception:
                LOGGER.debug("Failed to unlock tab %s", tab.id, exc_info=True)

        self._previous_tab_states.clear()

    def _set_state(self, new_state: LockState, reason: LockReason | None) -> None:
        """Update state and notify listeners."""
        self._state = new_state
        self._notify_state_change(new_state, reason)

    def _notify_state_change(self, state: LockState, reason: LockReason | None) -> None:
        """Notify state change listener."""
        if self._on_state_change:
            try:
                self._on_state_change(state, reason)
            except Exception:
                LOGGER.debug("State change listener failed", exc_info=True)

    def _update_status(self, message: str, is_locked: bool) -> None:
        """Update status bar."""
        if self._status_updater:
            try:
                self._status_updater(message, is_locked)
            except Exception:
                LOGGER.debug("Status updater failed", exc_info=True)

    @staticmethod
    def _reason_message(reason: LockReason) -> str:
        """Get human-readable message for lock reason."""
        messages = {
            LockReason.AI_TURN: "AI is working...",
            LockReason.REVIEW_PENDING: "Pending AI review",
            LockReason.TRANSACTION: "Applying changes...",
        }
        return messages.get(reason, "Processing...")


# -----------------------------------------------------------------------------
# Global Instance
# -----------------------------------------------------------------------------

_global_lock_manager: EditorLockManager | None = None


def get_lock_manager() -> EditorLockManager:
    """Get the global lock manager instance."""
    global _global_lock_manager
    if _global_lock_manager is None:
        _global_lock_manager = EditorLockManager()
    return _global_lock_manager


def set_lock_manager(manager: EditorLockManager | None) -> None:
    """Set the global lock manager instance."""
    global _global_lock_manager
    _global_lock_manager = manager


def reset_lock_manager() -> None:
    """Reset the global lock manager (for testing)."""
    global _global_lock_manager
    if _global_lock_manager:
        _global_lock_manager.force_release()
    _global_lock_manager = None
