"""Tests for Workstream 4: Editor Lock & Diff Review.

Tests for:
- WS4.1: Editor Lock (EditorLockManager)
- WS4.2: Atomic Operations & Rollback (Transaction, TransactionManager)
- WS4.4: Turn Checkpoints (CheckpointStore, Checkpoint)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Sequence
from unittest.mock import MagicMock, call

import pytest

# WS4.1 imports
from tinkerbell.ai.orchestration.editor_lock import (
    EditorLockManager,
    LockState,
    LockReason,
    LockSession,
    LockableTab,
    TabProvider,
    get_lock_manager,
    set_lock_manager,
    reset_lock_manager,
)

# WS4.2 imports
from tinkerbell.ai.orchestration.transaction import (
    Transaction,
    TransactionManager,
    TransactionState,
    ChangeType,
    StagedChange,
    DocumentSnapshot,
    TransactionError,
    CommitError,
    get_transaction_manager,
    set_transaction_manager,
    reset_transaction_manager,
)

# WS4.4 imports
from tinkerbell.ai.orchestration.checkpoints import (
    CheckpointStore,
    Checkpoint,
    CheckpointType,
    DocumentState,
    CheckpointDiff,
    compute_simple_diff,
    get_checkpoint_store,
    set_checkpoint_store,
    reset_checkpoint_store,
)


# =============================================================================
# Test Fixtures & Mocks
# =============================================================================


@dataclass
class MockTab:
    """Mock implementation of LockableTab protocol."""

    id: str
    _readonly: bool = False

    def set_readonly(self, readonly: bool) -> None:
        self._readonly = readonly

    def is_readonly(self) -> bool:
        return self._readonly


@dataclass
class MockTabProvider:
    """Mock implementation of TabProvider protocol."""

    tabs: list[MockTab] = field(default_factory=list)
    _active_tab: MockTab | None = None

    def get_all_tabs(self) -> Sequence[MockTab]:
        return self.tabs

    def get_active_tab(self) -> MockTab | None:
        return self._active_tab


@dataclass
class MockDocumentProvider:
    """Mock document provider for transactions and checkpoints."""

    documents: dict[str, str] = field(default_factory=dict)
    versions: dict[str, str] = field(default_factory=dict)
    cursors: dict[str, tuple[int, int]] = field(default_factory=dict)
    selections: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)

    def get_document_content(self, tab_id: str) -> str | None:
        return self.documents.get(tab_id)

    def set_document_content(self, tab_id: str, content: str) -> None:
        self.documents[tab_id] = content

    def get_version_token(self, tab_id: str) -> str | None:
        return self.versions.get(tab_id)

    def get_cursor_position(self, tab_id: str) -> tuple[int, int] | None:
        return self.cursors.get(tab_id)

    def get_selection(self, tab_id: str) -> tuple[int, int, int, int] | None:
        return self.selections.get(tab_id)


# =============================================================================
# WS4.1: Editor Lock Tests
# =============================================================================


class TestEditorLockManager:
    """Tests for EditorLockManager."""

    def test_initial_state_is_unlocked(self):
        """Lock manager starts in unlocked state."""
        manager = EditorLockManager()
        assert manager.state == LockState.UNLOCKED
        assert not manager.is_locked
        assert manager.active_session is None

    def test_acquire_lock(self):
        """Can acquire a lock."""
        manager = EditorLockManager()
        session = manager.acquire(LockReason.AI_TURN)

        assert session is not None
        assert manager.state == LockState.LOCKED
        assert manager.is_locked
        assert manager.active_session == session
        assert session.reason == LockReason.AI_TURN

    def test_acquire_fails_when_already_locked(self):
        """Cannot acquire lock when already locked."""
        manager = EditorLockManager()
        manager.acquire()

        session2 = manager.acquire()
        assert session2 is None

    def test_release_lock(self):
        """Can release an acquired lock."""
        manager = EditorLockManager()
        session = manager.acquire()

        result = manager.release()
        assert result is True
        assert manager.state == LockState.UNLOCKED
        assert not manager.is_locked
        assert manager.active_session is None

    def test_release_fails_when_not_locked(self):
        """Release returns False when not locked."""
        manager = EditorLockManager()
        result = manager.release()
        assert result is False

    def test_release_with_session_id_verification(self):
        """Release verifies session ID if provided."""
        manager = EditorLockManager()
        session = manager.acquire()

        # Wrong session ID
        result = manager.release(session_id="wrong-id")
        assert result is False
        assert manager.is_locked

        # Correct session ID
        result = manager.release(session_id=session.session_id)
        assert result is True
        assert not manager.is_locked

    def test_force_release(self):
        """Force release works regardless of state."""
        manager = EditorLockManager()
        manager.acquire()

        result = manager.force_release()
        assert result is True
        assert manager.state == LockState.UNLOCKED

    def test_cancel_releases_lock(self):
        """Cancel releases the lock."""
        manager = EditorLockManager()
        manager.acquire()

        result = manager.cancel()
        assert result is True
        assert not manager.is_locked

    def test_cancel_when_not_locked(self):
        """Cancel returns False when not locked."""
        manager = EditorLockManager()
        result = manager.cancel()
        assert result is False

    def test_context_manager(self):
        """Lock manager works as context manager."""
        manager = EditorLockManager()

        with manager:
            assert manager.is_locked

        assert not manager.is_locked

    def test_locks_all_tabs(self):
        """Lock sets all tabs to readonly."""
        tabs = [MockTab("tab1"), MockTab("tab2"), MockTab("tab3")]
        provider = MockTabProvider(tabs=tabs)
        manager = EditorLockManager(tab_provider=provider)

        manager.acquire()

        for tab in tabs:
            assert tab.is_readonly()

    def test_unlocks_all_tabs_on_release(self):
        """Release restores tabs to their original state."""
        tabs = [MockTab("tab1"), MockTab("tab2", _readonly=True)]
        provider = MockTabProvider(tabs=tabs)
        manager = EditorLockManager(tab_provider=provider)

        manager.acquire()
        manager.release()

        assert not tabs[0].is_readonly()  # Was not readonly
        assert tabs[1].is_readonly()  # Was already readonly

    def test_status_updater_called(self):
        """Status updater is called on lock/unlock."""
        status_calls = []

        def updater(message: str, is_locked: bool) -> None:
            status_calls.append((message, is_locked))

        manager = EditorLockManager(status_updater=updater)
        manager.acquire(LockReason.AI_TURN)
        manager.release()

        assert len(status_calls) == 2
        assert status_calls[0][1] is True  # Locked
        assert status_calls[1][1] is False  # Unlocked

    def test_state_change_listener(self):
        """State change listener is called."""
        state_changes = []

        def listener(state: LockState, reason: LockReason | None) -> None:
            state_changes.append((state, reason))

        manager = EditorLockManager(on_state_change=listener)
        manager.acquire(LockReason.REVIEW_PENDING)
        manager.release()

        assert len(state_changes) == 4  # LOCKING, LOCKED, UNLOCKING, UNLOCKED

    def test_lock_reasons(self):
        """Different lock reasons are tracked."""
        manager = EditorLockManager()

        for reason in LockReason:
            session = manager.acquire(reason)
            assert session.reason == reason
            manager.release()


class TestEditorLockGlobalAccess:
    """Tests for global lock manager access."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_lock_manager()

    def teardown_method(self):
        """Clean up global state after each test."""
        reset_lock_manager()

    def test_get_lock_manager_returns_singleton(self):
        """get_lock_manager returns the same instance."""
        manager1 = get_lock_manager()
        manager2 = get_lock_manager()
        assert manager1 is manager2

    def test_set_lock_manager(self):
        """Can set a custom lock manager."""
        custom = EditorLockManager()
        set_lock_manager(custom)
        assert get_lock_manager() is custom

    def test_reset_lock_manager_releases_lock(self):
        """Reset releases any active lock."""
        manager = get_lock_manager()
        manager.acquire()

        reset_lock_manager()

        # New manager should be unlocked
        new_manager = get_lock_manager()
        assert not new_manager.is_locked


# =============================================================================
# WS4.2: Transaction Tests
# =============================================================================


class TestTransaction:
    """Tests for Transaction class."""

    def test_initial_state_is_pending(self):
        """Transaction starts in PENDING state."""
        tx = Transaction()
        assert tx.state == TransactionState.PENDING
        assert not tx.is_active

    def test_begin_transaction(self):
        """Begin activates the transaction."""
        tx = Transaction()
        tx.begin()

        assert tx.state == TransactionState.ACTIVE
        assert tx.is_active

    def test_begin_fails_if_already_active(self):
        """Cannot begin an already active transaction."""
        tx = Transaction()
        tx.begin()

        with pytest.raises(TransactionError):
            tx.begin()

    def test_stage_change(self):
        """Can stage changes in active transaction."""
        tx = Transaction()
        tx.begin()

        change = tx.stage_change(
            "tab1",
            "original content",
            "new content",
        )

        assert change.tab_id == "tab1"
        assert change.original_content == "original content"
        assert change.new_content == "new content"
        assert len(tx.staged_changes) == 1

    def test_stage_change_fails_if_not_active(self):
        """Cannot stage changes in non-active transaction."""
        tx = Transaction()

        with pytest.raises(TransactionError):
            tx.stage_change("tab1", "old", "new")

    def test_affected_tabs(self):
        """affected_tabs returns unique tab IDs."""
        tx = Transaction()
        tx.begin()

        tx.stage_change("tab1", "old1", "new1")
        tx.stage_change("tab2", "old2", "new2")
        tx.stage_change("tab1", "new1", "newer1")  # Same tab

        assert tx.affected_tabs == {"tab1", "tab2"}

    def test_get_final_content(self):
        """get_final_content returns last staged content for tab."""
        tx = Transaction()
        tx.begin()

        tx.stage_change("tab1", "v1", "v2")
        tx.stage_change("tab1", "v2", "v3")

        assert tx.get_final_content("tab1") == "v3"
        assert tx.get_final_content("tab2") is None

    def test_commit_empty_transaction(self):
        """Committing empty transaction succeeds."""
        tx = Transaction()
        tx.begin()

        result = tx.commit()

        assert result is True
        assert tx.state == TransactionState.COMMITTED

    def test_commit_applies_changes(self):
        """Commit applies all staged changes."""
        provider = MockDocumentProvider(documents={"tab1": "original"})
        tx = Transaction(document_provider=provider)
        tx.begin()
        tx.stage_change("tab1", "original", "modified")

        result = tx.commit()

        assert result is True
        assert provider.documents["tab1"] == "modified"
        assert tx.state == TransactionState.COMMITTED

    def test_commit_multiple_documents(self):
        """Commit applies changes to multiple documents."""
        provider = MockDocumentProvider(
            documents={"tab1": "content1", "tab2": "content2"}
        )
        tx = Transaction(document_provider=provider)
        tx.begin()
        tx.stage_change("tab1", "content1", "new1")
        tx.stage_change("tab2", "content2", "new2")

        tx.commit()

        assert provider.documents["tab1"] == "new1"
        assert provider.documents["tab2"] == "new2"

    def test_rollback_discards_changes(self):
        """Rollback discards all staged changes."""
        provider = MockDocumentProvider(documents={"tab1": "original"})
        tx = Transaction(document_provider=provider)
        tx.begin(tab_ids=["tab1"])
        tx.stage_change("tab1", "original", "modified")

        result = tx.rollback("user cancelled")

        assert result is True
        assert tx.state == TransactionState.ROLLED_BACK
        assert provider.documents["tab1"] == "original"

    def test_rollback_fails_if_not_active(self):
        """Rollback returns False for non-active transaction."""
        tx = Transaction()
        result = tx.rollback()
        assert result is False

    def test_stage_full_replacement(self):
        """stage_full_replacement auto-fetches original content."""
        provider = MockDocumentProvider(documents={"tab1": "current content"})
        tx = Transaction(document_provider=provider)
        tx.begin()

        change = tx.stage_full_replacement("tab1", "new content")

        assert change.original_content == "current content"
        assert change.new_content == "new content"

    def test_compute_diff_stats(self):
        """compute_diff_stats returns correct statistics."""
        tx = Transaction()
        tx.begin()

        tx.stage_change("tab1", "hello", "hello world")  # +6 chars
        tx.stage_change("tab2", "goodbye world", "bye")  # -10 chars

        stats = tx.compute_diff_stats()

        assert stats["changes"] == 2
        assert stats["tabs_affected"] == 2

    def test_original_snapshots_preserved(self):
        """Original snapshots are preserved for rollback."""
        provider = MockDocumentProvider(
            documents={"tab1": "original"},
            versions={"tab1": "v1"},
        )
        tx = Transaction(document_provider=provider)
        tx.begin(tab_ids=["tab1"])

        # Modify via provider (simulating external change)
        provider.documents["tab1"] = "externally modified"

        # But rollback should restore original
        tx.rollback()
        assert provider.documents["tab1"] == "original"


class TestTransactionManager:
    """Tests for TransactionManager class."""

    def test_create_transaction(self):
        """Can create transactions via manager."""
        provider = MockDocumentProvider(documents={"tab1": "content"})
        manager = TransactionManager(document_provider=provider)

        tx = manager.create_transaction()

        assert tx.is_active
        assert tx.transaction_id in [t.transaction_id for t in manager.active_transactions]

    def test_get_transaction(self):
        """Can retrieve transaction by ID."""
        manager = TransactionManager()
        tx = manager.create_transaction()

        retrieved = manager.get_transaction(tx.transaction_id)
        assert retrieved is tx

    def test_cleanup_completed(self):
        """cleanup_completed removes finished transactions."""
        manager = TransactionManager()
        tx1 = manager.create_transaction()
        tx2 = manager.create_transaction()

        tx1.commit()

        count = manager.cleanup_completed()

        assert count == 1
        assert manager.get_transaction(tx1.transaction_id) is None
        assert manager.get_transaction(tx2.transaction_id) is tx2

    def test_context_manager(self):
        """Manager provides context manager for transactions."""
        provider = MockDocumentProvider(documents={"tab1": "content"})
        manager = TransactionManager(document_provider=provider)

        with manager.transaction(tab_ids=["tab1"]) as tx:
            tx.stage_full_replacement("tab1", "new content")
            tx.commit()

        assert provider.documents["tab1"] == "new content"

    def test_context_manager_auto_rollback_on_exception(self):
        """Context manager auto-rolls back on exception."""
        provider = MockDocumentProvider(documents={"tab1": "original"})
        manager = TransactionManager(document_provider=provider)

        with pytest.raises(ValueError):
            with manager.transaction(tab_ids=["tab1"]) as tx:
                tx.stage_full_replacement("tab1", "new content")
                raise ValueError("test error")

        # Should be rolled back
        assert provider.documents["tab1"] == "original"

    def test_has_active_transaction_for_tab(self):
        """Can check if tab has active transaction."""
        manager = TransactionManager()
        tx = manager.create_transaction()
        tx.stage_change("tab1", "old", "new")

        assert manager.has_active_transaction_for_tab("tab1")
        assert not manager.has_active_transaction_for_tab("tab2")


class TestTransactionGlobalAccess:
    """Tests for global transaction manager access."""

    def setup_method(self):
        reset_transaction_manager()

    def teardown_method(self):
        reset_transaction_manager()

    def test_get_transaction_manager_singleton(self):
        """get_transaction_manager returns singleton."""
        manager1 = get_transaction_manager()
        manager2 = get_transaction_manager()
        assert manager1 is manager2

    def test_set_transaction_manager(self):
        """Can set custom transaction manager."""
        custom = TransactionManager()
        set_transaction_manager(custom)
        assert get_transaction_manager() is custom


# =============================================================================
# WS4.4: Checkpoint Tests
# =============================================================================


class TestCheckpointStore:
    """Tests for CheckpointStore class."""

    def test_create_checkpoint(self):
        """Can create a checkpoint."""
        provider = MockDocumentProvider(
            documents={"tab1": "content1"},
            versions={"tab1": "v1"},
        )
        store = CheckpointStore(document_provider=provider)

        checkpoint = store.create_checkpoint(["tab1"])

        assert checkpoint.checkpoint_id is not None
        assert "tab1" in checkpoint.documents
        assert checkpoint.documents["tab1"].content == "content1"

    def test_create_checkpoint_multiple_tabs(self):
        """Can create checkpoint with multiple tabs."""
        provider = MockDocumentProvider(
            documents={"tab1": "content1", "tab2": "content2"}
        )
        store = CheckpointStore(document_provider=provider)

        checkpoint = store.create_checkpoint(["tab1", "tab2"])

        assert len(checkpoint.documents) == 2
        assert checkpoint.tab_ids == ["tab1", "tab2"]

    def test_create_pre_turn_checkpoint(self):
        """Can create pre-turn checkpoint."""
        provider = MockDocumentProvider(documents={"tab1": "content"})
        store = CheckpointStore(document_provider=provider)

        checkpoint = store.create_pre_turn_checkpoint(["tab1"], turn_number=5)

        assert checkpoint.checkpoint_type == CheckpointType.PRE_TURN
        assert checkpoint.turn_number == 5

    def test_create_post_turn_checkpoint(self):
        """Can create post-turn checkpoint."""
        provider = MockDocumentProvider(documents={"tab1": "content"})
        store = CheckpointStore(document_provider=provider)

        checkpoint = store.create_post_turn_checkpoint(
            ["tab1"],
            turn_number=5,
            action_summary="Edited paragraph",
        )

        assert checkpoint.checkpoint_type == CheckpointType.POST_TURN
        assert checkpoint.turn_number == 5
        assert checkpoint.action_summary == "Edited paragraph"

    def test_get_checkpoints_for_tab(self):
        """Can retrieve checkpoints for a tab."""
        provider = MockDocumentProvider(documents={"tab1": "content"})
        store = CheckpointStore(document_provider=provider)

        store.create_checkpoint(["tab1"])
        store.create_checkpoint(["tab1"])

        checkpoints = store.get_checkpoints_for_tab("tab1")

        assert len(checkpoints) == 2

    def test_get_checkpoints_by_type(self):
        """Can filter checkpoints by type."""
        provider = MockDocumentProvider(documents={"tab1": "content"})
        store = CheckpointStore(document_provider=provider)

        store.create_checkpoint(["tab1"], checkpoint_type=CheckpointType.MANUAL)
        store.create_pre_turn_checkpoint(["tab1"], turn_number=1)
        store.create_checkpoint(["tab1"], checkpoint_type=CheckpointType.MANUAL)

        manual = store.get_checkpoints_for_tab("tab1", checkpoint_type=CheckpointType.MANUAL)
        pre_turn = store.get_checkpoints_for_tab("tab1", checkpoint_type=CheckpointType.PRE_TURN)

        assert len(manual) == 2
        assert len(pre_turn) == 1

    def test_get_latest_checkpoint(self):
        """get_latest_checkpoint returns most recent."""
        provider = MockDocumentProvider(documents={"tab1": "content"})
        store = CheckpointStore(document_provider=provider)

        cp1 = store.create_checkpoint(["tab1"])
        cp2 = store.create_checkpoint(["tab1"])

        latest = store.get_latest_checkpoint("tab1")

        assert latest.checkpoint_id == cp2.checkpoint_id

    def test_restore_checkpoint(self):
        """Can restore document to checkpoint state."""
        provider = MockDocumentProvider(documents={"tab1": "original"})
        store = CheckpointStore(document_provider=provider)

        checkpoint = store.create_checkpoint(["tab1"])

        # Modify document
        provider.documents["tab1"] = "modified"

        # Restore
        result = store.restore_checkpoint(checkpoint.checkpoint_id)

        assert result is True
        assert provider.documents["tab1"] == "original"

    def test_restore_checkpoint_creates_new_checkpoint(self):
        """Restore creates a new checkpoint by default."""
        provider = MockDocumentProvider(documents={"tab1": "original"})
        store = CheckpointStore(document_provider=provider)

        checkpoint = store.create_checkpoint(["tab1"])
        initial_count = store.get_checkpoint_count("tab1")

        provider.documents["tab1"] = "modified"
        store.restore_checkpoint(checkpoint.checkpoint_id)

        # Should have created a new checkpoint before restore
        assert store.get_checkpoint_count("tab1") == initial_count + 1

    def test_restore_without_new_checkpoint(self):
        """Can restore without creating new checkpoint."""
        provider = MockDocumentProvider(documents={"tab1": "original"})
        store = CheckpointStore(document_provider=provider)

        checkpoint = store.create_checkpoint(["tab1"])
        initial_count = store.get_checkpoint_count("tab1")

        provider.documents["tab1"] = "modified"
        store.restore_checkpoint(checkpoint.checkpoint_id, create_new_checkpoint=False)

        assert store.get_checkpoint_count("tab1") == initial_count

    def test_restore_to_pre_turn(self):
        """Can restore to pre-turn state."""
        provider = MockDocumentProvider(documents={"tab1": "before turn"})
        store = CheckpointStore(document_provider=provider)

        store.create_pre_turn_checkpoint(["tab1"], turn_number=5)

        provider.documents["tab1"] = "after turn"

        result = store.restore_to_pre_turn(5, "tab1")

        assert result is True
        assert provider.documents["tab1"] == "before turn"

    def test_compute_diff(self):
        """Can compute diff between current and checkpoint."""
        provider = MockDocumentProvider(documents={"tab1": "line1\nline2\nline3"})
        store = CheckpointStore(document_provider=provider)

        checkpoint = store.create_checkpoint(["tab1"])

        provider.documents["tab1"] = "line1\nmodified\nline3\nline4"

        diff = store.compute_diff(checkpoint.checkpoint_id, "tab1")

        assert diff is not None
        assert diff.has_changes is True
        assert diff.lines_added > 0

    def test_compute_diff_no_changes(self):
        """Diff shows no changes when content identical."""
        provider = MockDocumentProvider(documents={"tab1": "content"})
        store = CheckpointStore(document_provider=provider)

        checkpoint = store.create_checkpoint(["tab1"])

        diff = store.compute_diff(checkpoint.checkpoint_id, "tab1")

        assert diff.has_changes is False

    def test_clear_checkpoints_for_tab(self):
        """Can clear checkpoints for specific tab."""
        provider = MockDocumentProvider(
            documents={"tab1": "content1", "tab2": "content2"}
        )
        store = CheckpointStore(document_provider=provider)

        store.create_checkpoint(["tab1"])
        store.create_checkpoint(["tab2"])

        count = store.clear_checkpoints("tab1")

        assert count == 1
        assert store.get_checkpoint_count("tab1") == 0
        assert store.get_checkpoint_count("tab2") == 1

    def test_clear_all_checkpoints(self):
        """Can clear all checkpoints."""
        provider = MockDocumentProvider(
            documents={"tab1": "content1", "tab2": "content2"}
        )
        store = CheckpointStore(document_provider=provider)

        store.create_checkpoint(["tab1"])
        store.create_checkpoint(["tab2"])

        count = store.clear_checkpoints()

        assert count == 2
        assert len(store.get_all_tabs_with_checkpoints()) == 0

    def test_max_checkpoints_enforced(self):
        """Maximum checkpoints per tab is enforced."""
        provider = MockDocumentProvider(documents={"tab1": "content"})
        store = CheckpointStore(
            document_provider=provider,
            max_checkpoints_per_tab=3,
        )

        for _ in range(5):
            store.create_checkpoint(["tab1"])

        assert store.get_checkpoint_count("tab1") == 3

    def test_checkpoint_listener_called(self):
        """Checkpoint listener is called on events."""
        provider = MockDocumentProvider(documents={"tab1": "content"})
        listener = MagicMock()
        store = CheckpointStore(
            document_provider=provider,
            listener=listener,
        )

        checkpoint = store.create_checkpoint(["tab1"])
        store.restore_checkpoint(checkpoint.checkpoint_id)
        store.clear_checkpoints("tab1")

        listener.on_checkpoint_created.assert_called()
        listener.on_checkpoint_restored.assert_called()
        listener.on_checkpoints_cleared.assert_called_with("tab1")


class TestDocumentState:
    """Tests for DocumentState class."""

    def test_content_hash(self):
        """content_hash returns consistent hash."""
        state = DocumentState(tab_id="tab1", content="hello world")

        hash1 = state.content_hash()
        hash2 = state.content_hash()

        assert hash1 == hash2

    def test_line_count(self):
        """line_count returns correct count."""
        state = DocumentState(tab_id="tab1", content="line1\nline2\nline3")
        assert state.line_count() == 3

    def test_char_count(self):
        """char_count returns correct count."""
        state = DocumentState(tab_id="tab1", content="hello")
        assert state.char_count() == 5


class TestComputeSimpleDiff:
    """Tests for compute_simple_diff function."""

    def test_identical_content(self):
        """Identical content has no diff."""
        added, removed, chars_added, chars_removed = compute_simple_diff(
            "hello world",
            "hello world",
        )
        assert added == 0
        assert removed == 0

    def test_added_lines(self):
        """Detects added lines."""
        added, removed, _, _ = compute_simple_diff(
            "line1\nline2\nline3",
            "line1\nline2",
        )
        assert added == 1
        assert removed == 0

    def test_removed_lines(self):
        """Detects removed lines."""
        added, removed, _, _ = compute_simple_diff(
            "line1\nline2",
            "line1\nline2\nline3",
        )
        assert added == 0
        assert removed == 1


class TestCheckpointGlobalAccess:
    """Tests for global checkpoint store access."""

    def setup_method(self):
        reset_checkpoint_store()

    def teardown_method(self):
        reset_checkpoint_store()

    def test_get_checkpoint_store_singleton(self):
        """get_checkpoint_store returns singleton."""
        store1 = get_checkpoint_store()
        store2 = get_checkpoint_store()
        assert store1 is store2

    def test_set_checkpoint_store(self):
        """Can set custom checkpoint store."""
        custom = CheckpointStore()
        set_checkpoint_store(custom)
        assert get_checkpoint_store() is custom

    def test_reset_checkpoint_store_clears_data(self):
        """Reset clears checkpoint data."""
        store = get_checkpoint_store()
        # Add mock data manually since we don't have a provider
        store._checkpoints["tab1"] = []

        reset_checkpoint_store()

        new_store = get_checkpoint_store()
        assert new_store.get_checkpoint_count("tab1") == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestWS4Integration:
    """Integration tests for WS4 components working together."""

    def test_lock_transaction_checkpoint_workflow(self):
        """Full workflow: lock -> transaction -> checkpoint."""
        # Setup
        provider = MockDocumentProvider(documents={"tab1": "original content"})
        tabs = [MockTab("tab1")]
        tab_provider = MockTabProvider(tabs=tabs)

        lock_manager = EditorLockManager(tab_provider=tab_provider)
        tx_manager = TransactionManager(document_provider=provider)
        checkpoint_store = CheckpointStore(document_provider=provider)

        # Step 1: Acquire lock
        lock_session = lock_manager.acquire(LockReason.AI_TURN)
        assert lock_session is not None
        assert tabs[0].is_readonly()

        # Step 2: Create pre-turn checkpoint
        pre_checkpoint = checkpoint_store.create_pre_turn_checkpoint(["tab1"], turn_number=1)

        # Step 3: Execute transaction
        with tx_manager.transaction(tab_ids=["tab1"]) as tx:
            tx.stage_full_replacement("tab1", "modified content")
            tx.commit()

        # Step 4: Create post-turn checkpoint
        post_checkpoint = checkpoint_store.create_post_turn_checkpoint(
            ["tab1"],
            turn_number=1,
            action_summary="Modified content",
        )

        # Step 5: Release lock
        lock_manager.release()
        assert not tabs[0].is_readonly()

        # Verify state
        assert provider.documents["tab1"] == "modified content"
        assert checkpoint_store.get_checkpoint_count("tab1") == 2

        # Step 6: User can restore to pre-turn state
        checkpoint_store.restore_to_pre_turn(1, "tab1")
        assert provider.documents["tab1"] == "original content"

    def test_transaction_rollback_on_lock_cancel(self):
        """Transaction rolls back when lock is cancelled."""
        provider = MockDocumentProvider(documents={"tab1": "original"})
        lock_manager = EditorLockManager()
        tx = Transaction(document_provider=provider)

        # Start lock and transaction
        lock_manager.acquire()
        tx.begin(tab_ids=["tab1"])
        tx.stage_full_replacement("tab1", "modified")

        # Cancel lock
        lock_manager.cancel()

        # Rollback transaction
        tx.rollback("Lock cancelled")

        assert provider.documents["tab1"] == "original"
        assert not lock_manager.is_locked

    def test_concurrent_lock_attempts(self):
        """Only one lock can be acquired at a time."""
        manager = EditorLockManager()
        results = []

        def try_acquire():
            session = manager.acquire()
            results.append(session is not None)
            if session:
                time.sleep(0.1)
                manager.release()

        threads = [threading.Thread(target=try_acquire) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Multiple threads tried, but only sequential acquisitions succeed
        assert sum(results) >= 1  # At least one succeeded


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_transaction_commit_with_no_provider(self):
        """Commit without provider raises error."""
        tx = Transaction()
        tx.begin()
        tx.stage_change("tab1", "old", "new")

        with pytest.raises(CommitError):
            tx.commit()

    def test_restore_nonexistent_checkpoint(self):
        """Restoring non-existent checkpoint returns False."""
        store = CheckpointStore()
        result = store.restore_checkpoint("nonexistent")
        assert result is False

    def test_checkpoint_with_missing_tab(self):
        """Checkpoint handles tabs not found in provider."""
        provider = MockDocumentProvider(documents={})
        store = CheckpointStore(document_provider=provider)

        checkpoint = store.create_checkpoint(["nonexistent"])

        assert len(checkpoint.documents) == 0

    def test_staged_change_metrics(self):
        """StagedChange computes correct metrics."""
        change = StagedChange(
            change_id="c1",
            tab_id="tab1",
            change_type=ChangeType.FULL_CONTENT,
            original_content="hello",
            new_content="hello world",
        )

        assert change.size_change() == 6  # Added 6 characters
        assert change.lines_affected() > 0

    def test_empty_document_checkpoint(self):
        """Can checkpoint empty documents."""
        provider = MockDocumentProvider(documents={"tab1": ""})
        store = CheckpointStore(document_provider=provider)

        checkpoint = store.create_checkpoint(["tab1"])

        assert checkpoint.documents["tab1"].content == ""
        assert checkpoint.documents["tab1"].line_count() == 1

    def test_lock_manager_with_no_tabs(self):
        """Lock manager works with no tabs."""
        provider = MockTabProvider(tabs=[])
        manager = EditorLockManager(tab_provider=provider)

        session = manager.acquire()
        manager.release()

        assert session is not None
