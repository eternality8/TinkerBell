"""Transaction System for Atomic Document Operations.

Provides staged changes with atomic commit/rollback semantics,
supporting multi-document transactions during AI operations.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Transaction State
# -----------------------------------------------------------------------------


class TransactionState(Enum):
    """State of a transaction."""

    PENDING = auto()  # Not started
    ACTIVE = auto()  # Changes being staged
    COMMITTED = auto()  # Changes applied
    ROLLED_BACK = auto()  # Changes discarded
    FAILED = auto()  # Error during commit


class ChangeType(Enum):
    """Type of staged change."""

    FULL_CONTENT = auto()  # Complete document replacement
    INSERT = auto()  # Lines inserted
    DELETE = auto()  # Lines deleted
    REPLACE = auto()  # Lines replaced


# -----------------------------------------------------------------------------
# Staged Change
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class StagedChange:
    """Represents a single staged change to a document.

    Attributes:
        change_id: Unique identifier for this change.
        tab_id: Target document tab.
        change_type: Type of change.
        original_content: Original document content before change.
        new_content: New document content after change.
        metadata: Additional change metadata (line ranges, etc.).
        timestamp: When the change was staged.
    """

    change_id: str
    tab_id: str
    change_type: ChangeType
    original_content: str
    new_content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def lines_affected(self) -> int:
        """Calculate approximate number of lines affected."""
        orig_lines = self.original_content.count("\n") + 1
        new_lines = self.new_content.count("\n") + 1
        return abs(new_lines - orig_lines) + max(orig_lines, new_lines)

    def size_change(self) -> int:
        """Calculate size change in characters."""
        return len(self.new_content) - len(self.original_content)


# -----------------------------------------------------------------------------
# Document State Snapshot
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class DocumentSnapshot:
    """Immutable snapshot of a document's state.

    Attributes:
        tab_id: Document tab identifier.
        content: Full document content.
        version_token: Version token at snapshot time.
        timestamp: When snapshot was taken.
    """

    tab_id: str
    content: str
    version_token: str | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def content_hash(self) -> int:
        """Get content hash for quick comparison."""
        return hash(self.content)


# -----------------------------------------------------------------------------
# Document Provider Protocol
# -----------------------------------------------------------------------------


class TransactionDocumentProvider(Protocol):
    """Protocol for document access in transactions."""

    def get_document_content(self, tab_id: str) -> str | None:
        """Get the current content of a document."""
        ...

    def set_document_content(self, tab_id: str, content: str) -> None:
        """Set the content of a document."""
        ...

    def get_version_token(self, tab_id: str) -> str | None:
        """Get the current version token for a document."""
        ...


class TransactionListener(Protocol):
    """Callback for transaction events."""

    def on_transaction_started(self, transaction_id: str, tab_ids: Sequence[str]) -> None:
        """Called when a transaction starts."""
        ...

    def on_change_staged(self, transaction_id: str, change: StagedChange) -> None:
        """Called when a change is staged."""
        ...

    def on_transaction_committed(self, transaction_id: str, changes: Sequence[StagedChange]) -> None:
        """Called when a transaction commits."""
        ...

    def on_transaction_rolled_back(self, transaction_id: str, reason: str | None) -> None:
        """Called when a transaction rolls back."""
        ...


# -----------------------------------------------------------------------------
# Transaction
# -----------------------------------------------------------------------------


@dataclass
class Transaction:
    """Represents a multi-document transaction.

    A transaction collects changes to one or more documents and provides
    atomic commit/rollback semantics. Changes are staged in memory and
    only applied when commit() is called.

    Example:
        tx = Transaction(doc_provider=provider)
        tx.begin()
        tx.stage_change("tab1", "old content", "new content")
        tx.stage_change("tab2", "old2", "new2")
        tx.commit()  # All changes applied atomically
    """

    transaction_id: str = field(default_factory=lambda: f"tx-{uuid.uuid4().hex[:12]}")
    state: TransactionState = TransactionState.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    committed_at: datetime | None = None
    rolled_back_at: datetime | None = None

    # Document access
    document_provider: TransactionDocumentProvider | None = None
    listener: TransactionListener | None = None

    # Staged changes (in order)
    staged_changes: list[StagedChange] = field(default_factory=list)

    # Snapshots of original document states
    original_snapshots: dict[str, DocumentSnapshot] = field(default_factory=dict)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal lock for thread safety
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _change_counter: int = field(default=0, repr=False)

    # ------------------------------------------------------------------
    # Transaction Lifecycle
    # ------------------------------------------------------------------

    def begin(self, tab_ids: Sequence[str] | None = None) -> "Transaction":
        """Begin the transaction.

        Args:
            tab_ids: Optional list of tab IDs to pre-snapshot.

        Returns:
            Self for chaining.

        Raises:
            TransactionError: If transaction already started.
        """
        with self._lock:
            if self.state != TransactionState.PENDING:
                raise TransactionError(
                    f"Cannot begin: transaction is {self.state.name}",
                    transaction_id=self.transaction_id,
                )

            self.state = TransactionState.ACTIVE
            self.staged_changes.clear()
            self.original_snapshots.clear()

            # Pre-snapshot requested tabs
            if tab_ids and self.document_provider:
                for tab_id in tab_ids:
                    self._snapshot_tab(tab_id)

            # Notify listener
            if self.listener:
                try:
                    self.listener.on_transaction_started(
                        self.transaction_id,
                        list(tab_ids or []),
                    )
                except Exception:
                    LOGGER.debug("Listener failed on transaction start", exc_info=True)

            LOGGER.debug(
                "Transaction %s started (pre-snapshotted %d tabs)",
                self.transaction_id,
                len(self.original_snapshots),
            )

            return self

    def commit(self) -> bool:
        """Commit all staged changes atomically.

        Returns:
            True if committed successfully.

        Raises:
            TransactionError: If transaction not active.
            CommitError: If any change fails to apply.
        """
        with self._lock:
            if self.state != TransactionState.ACTIVE:
                raise TransactionError(
                    f"Cannot commit: transaction is {self.state.name}",
                    transaction_id=self.transaction_id,
                )

            if not self.staged_changes:
                # Nothing to commit
                self.state = TransactionState.COMMITTED
                self.committed_at = datetime.now(timezone.utc)
                LOGGER.debug("Transaction %s committed (no changes)", self.transaction_id)
                return True

            applied_changes: list[StagedChange] = []
            try:
                # Apply all changes
                for change in self.staged_changes:
                    self._apply_change(change)
                    applied_changes.append(change)

                self.state = TransactionState.COMMITTED
                self.committed_at = datetime.now(timezone.utc)

                # Notify listener
                if self.listener:
                    try:
                        self.listener.on_transaction_committed(
                            self.transaction_id,
                            self.staged_changes,
                        )
                    except Exception:
                        LOGGER.debug("Listener failed on commit", exc_info=True)

                LOGGER.info(
                    "Transaction %s committed (%d changes across %d documents)",
                    self.transaction_id,
                    len(self.staged_changes),
                    len(self.affected_tabs),
                )
                return True

            except Exception as exc:
                # Rollback applied changes
                LOGGER.error(
                    "Transaction %s commit failed, rolling back %d applied changes: %s",
                    self.transaction_id,
                    len(applied_changes),
                    exc,
                )
                self._rollback_changes(applied_changes)
                self.state = TransactionState.FAILED

                raise CommitError(
                    f"Commit failed: {exc}",
                    transaction_id=self.transaction_id,
                    failed_change=self.staged_changes[len(applied_changes)] if len(applied_changes) < len(self.staged_changes) else None,
                ) from exc

    def rollback(self, reason: str | None = None) -> bool:
        """Rollback the transaction, discarding all staged changes.

        Args:
            reason: Optional reason for rollback.

        Returns:
            True if rolled back successfully.
        """
        with self._lock:
            if self.state not in (TransactionState.ACTIVE, TransactionState.FAILED):
                LOGGER.debug(
                    "Cannot rollback: transaction %s is %s",
                    self.transaction_id,
                    self.state.name,
                )
                return False

            # Restore original snapshots
            self._restore_snapshots()

            self.state = TransactionState.ROLLED_BACK
            self.rolled_back_at = datetime.now(timezone.utc)

            # Notify listener
            if self.listener:
                try:
                    self.listener.on_transaction_rolled_back(
                        self.transaction_id,
                        reason,
                    )
                except Exception:
                    LOGGER.debug("Listener failed on rollback", exc_info=True)

            LOGGER.info(
                "Transaction %s rolled back: %s",
                self.transaction_id,
                reason or "no reason provided",
            )

            return True

    # ------------------------------------------------------------------
    # Staging Changes
    # ------------------------------------------------------------------

    def stage_change(
        self,
        tab_id: str,
        original_content: str,
        new_content: str,
        *,
        change_type: ChangeType = ChangeType.FULL_CONTENT,
        metadata: dict[str, Any] | None = None,
    ) -> StagedChange:
        """Stage a change for later commit.

        Args:
            tab_id: Target document tab.
            original_content: Content before change.
            new_content: Content after change.
            change_type: Type of change.
            metadata: Additional metadata.

        Returns:
            The staged change.

        Raises:
            TransactionError: If transaction not active.
        """
        with self._lock:
            if self.state != TransactionState.ACTIVE:
                raise TransactionError(
                    f"Cannot stage change: transaction is {self.state.name}",
                    transaction_id=self.transaction_id,
                )

            # Ensure we have a snapshot
            if tab_id not in self.original_snapshots:
                self._snapshot_tab(tab_id, original_content)

            # Create the change
            self._change_counter += 1
            change = StagedChange(
                change_id=f"{self.transaction_id}-{self._change_counter}",
                tab_id=tab_id,
                change_type=change_type,
                original_content=original_content,
                new_content=new_content,
                metadata=metadata or {},
            )

            self.staged_changes.append(change)

            # Notify listener
            if self.listener:
                try:
                    self.listener.on_change_staged(self.transaction_id, change)
                except Exception:
                    LOGGER.debug("Listener failed on change staged", exc_info=True)

            LOGGER.debug(
                "Staged change %s for tab %s (%+d chars)",
                change.change_id,
                tab_id,
                change.size_change(),
            )

            return change

    def stage_full_replacement(
        self,
        tab_id: str,
        new_content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> StagedChange:
        """Stage a full document replacement.

        Automatically fetches the current content as original.

        Args:
            tab_id: Target document tab.
            new_content: New content to set.
            metadata: Additional metadata.

        Returns:
            The staged change.

        Raises:
            TransactionError: If transaction not active or document not found.
        """
        with self._lock:
            if not self.document_provider:
                raise TransactionError(
                    "No document provider available",
                    transaction_id=self.transaction_id,
                )

            original_content = self.document_provider.get_document_content(tab_id)
            if original_content is None:
                raise TransactionError(
                    f"Document not found: {tab_id}",
                    transaction_id=self.transaction_id,
                )

            return self.stage_change(
                tab_id,
                original_content,
                new_content,
                change_type=ChangeType.FULL_CONTENT,
                metadata=metadata,
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """Check if transaction is active."""
        return self.state == TransactionState.ACTIVE

    @property
    def affected_tabs(self) -> set[str]:
        """Get set of tab IDs affected by staged changes."""
        return {change.tab_id for change in self.staged_changes}

    @property
    def change_count(self) -> int:
        """Get number of staged changes."""
        return len(self.staged_changes)

    def get_changes_for_tab(self, tab_id: str) -> list[StagedChange]:
        """Get all staged changes for a specific tab."""
        return [c for c in self.staged_changes if c.tab_id == tab_id]

    def get_final_content(self, tab_id: str) -> str | None:
        """Get the final content for a tab after all staged changes.

        Returns the new_content from the last change for this tab,
        or None if no changes staged for this tab.
        """
        changes = self.get_changes_for_tab(tab_id)
        if not changes:
            return None
        return changes[-1].new_content

    def get_original_content(self, tab_id: str) -> str | None:
        """Get the original content for a tab before any changes."""
        snapshot = self.original_snapshots.get(tab_id)
        return snapshot.content if snapshot else None

    def compute_diff_stats(self) -> dict[str, Any]:
        """Compute statistics about staged changes."""
        total_added = 0
        total_removed = 0
        total_lines_affected = 0

        for change in self.staged_changes:
            size_change = change.size_change()
            if size_change > 0:
                total_added += size_change
            else:
                total_removed += abs(size_change)
            total_lines_affected += change.lines_affected()

        return {
            "changes": len(self.staged_changes),
            "tabs_affected": len(self.affected_tabs),
            "chars_added": total_added,
            "chars_removed": total_removed,
            "lines_affected": total_lines_affected,
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _snapshot_tab(self, tab_id: str, content: str | None = None) -> None:
        """Create a snapshot of a tab's current state."""
        if tab_id in self.original_snapshots:
            return

        if content is None:
            if not self.document_provider:
                return
            content = self.document_provider.get_document_content(tab_id)
            if content is None:
                return

        version_token = None
        if self.document_provider:
            version_token = self.document_provider.get_version_token(tab_id)

        self.original_snapshots[tab_id] = DocumentSnapshot(
            tab_id=tab_id,
            content=content,
            version_token=version_token,
        )

    def _apply_change(self, change: StagedChange) -> None:
        """Apply a single change to the document."""
        if not self.document_provider:
            raise CommitError(
                "No document provider available",
                transaction_id=self.transaction_id,
            )

        self.document_provider.set_document_content(
            change.tab_id,
            change.new_content,
        )

    def _rollback_changes(self, changes: list[StagedChange]) -> None:
        """Rollback a list of applied changes."""
        if not self.document_provider:
            return

        # Restore from snapshots
        affected_tabs = {c.tab_id for c in changes}
        for tab_id in affected_tabs:
            snapshot = self.original_snapshots.get(tab_id)
            if snapshot:
                try:
                    self.document_provider.set_document_content(
                        tab_id,
                        snapshot.content,
                    )
                except Exception:
                    LOGGER.error(
                        "Failed to restore tab %s during rollback",
                        tab_id,
                        exc_info=True,
                    )

    def _restore_snapshots(self) -> None:
        """Restore all tabs to their original snapshots."""
        if not self.document_provider:
            return

        for tab_id, snapshot in self.original_snapshots.items():
            try:
                self.document_provider.set_document_content(
                    tab_id,
                    snapshot.content,
                )
            except Exception:
                LOGGER.error(
                    "Failed to restore tab %s during rollback",
                    tab_id,
                    exc_info=True,
                )


# -----------------------------------------------------------------------------
# Transaction Errors
# -----------------------------------------------------------------------------


class TransactionError(Exception):
    """Base error for transaction operations."""

    def __init__(
        self,
        message: str,
        *,
        transaction_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.transaction_id = transaction_id


class CommitError(TransactionError):
    """Error during transaction commit."""

    def __init__(
        self,
        message: str,
        *,
        transaction_id: str | None = None,
        failed_change: StagedChange | None = None,
    ) -> None:
        super().__init__(message, transaction_id=transaction_id)
        self.failed_change = failed_change


# -----------------------------------------------------------------------------
# Transaction Manager
# -----------------------------------------------------------------------------


class TransactionManager:
    """Manages transactions across the application.

    Provides a central point for creating and tracking transactions,
    with support for transaction isolation and conflict detection.
    """

    def __init__(
        self,
        *,
        document_provider: TransactionDocumentProvider | None = None,
        listener: TransactionListener | None = None,
    ) -> None:
        """Initialize the transaction manager.

        Args:
            document_provider: Provider for document access.
            listener: Global transaction listener.
        """
        self._document_provider = document_provider
        self._listener = listener
        self._lock = threading.RLock()
        self._active_transactions: dict[str, Transaction] = {}

    # ------------------------------------------------------------------
    # Transaction Creation
    # ------------------------------------------------------------------

    def create_transaction(
        self,
        *,
        tab_ids: Sequence[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Transaction:
        """Create a new transaction.

        Args:
            tab_ids: Optional tab IDs to pre-snapshot.
            metadata: Transaction metadata.

        Returns:
            A new Transaction instance.
        """
        tx = Transaction(
            document_provider=self._document_provider,
            listener=self._listener,
            metadata=metadata or {},
        )

        with self._lock:
            self._active_transactions[tx.transaction_id] = tx

        # Begin the transaction
        tx.begin(tab_ids)

        return tx

    def get_transaction(self, transaction_id: str) -> Transaction | None:
        """Get a transaction by ID."""
        with self._lock:
            return self._active_transactions.get(transaction_id)

    def cleanup_completed(self) -> int:
        """Remove completed transactions from tracking.

        Returns:
            Number of transactions cleaned up.
        """
        with self._lock:
            completed = [
                tx_id
                for tx_id, tx in self._active_transactions.items()
                if tx.state in (TransactionState.COMMITTED, TransactionState.ROLLED_BACK)
            ]
            for tx_id in completed:
                del self._active_transactions[tx_id]
            return len(completed)

    # ------------------------------------------------------------------
    # Context Manager
    # ------------------------------------------------------------------

    @contextmanager
    def transaction(
        self,
        *,
        tab_ids: Sequence[str] | None = None,
        auto_rollback: bool = True,
    ) -> Iterator[Transaction]:
        """Context manager for transactions.

        Args:
            tab_ids: Optional tab IDs to pre-snapshot.
            auto_rollback: If True, rollback on exception.

        Yields:
            Active transaction.

        Example:
            with tx_manager.transaction(tab_ids=["tab1"]) as tx:
                tx.stage_change("tab1", old, new)
                tx.commit()
        """
        tx = self.create_transaction(tab_ids=tab_ids)
        try:
            yield tx
        except Exception:
            if auto_rollback and tx.state == TransactionState.ACTIVE:
                tx.rollback("Exception during transaction")
            raise
        finally:
            self.cleanup_completed()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def active_transactions(self) -> list[Transaction]:
        """Get all active transactions."""
        with self._lock:
            return [
                tx
                for tx in self._active_transactions.values()
                if tx.state == TransactionState.ACTIVE
            ]

    def has_active_transaction_for_tab(self, tab_id: str) -> bool:
        """Check if any active transaction affects a tab."""
        return any(
            tab_id in tx.affected_tabs
            for tx in self.active_transactions
        )


# -----------------------------------------------------------------------------
# Global Instance
# -----------------------------------------------------------------------------

_global_transaction_manager: TransactionManager | None = None


def get_transaction_manager() -> TransactionManager:
    """Get the global transaction manager instance."""
    global _global_transaction_manager
    if _global_transaction_manager is None:
        _global_transaction_manager = TransactionManager()
    return _global_transaction_manager


def set_transaction_manager(manager: TransactionManager | None) -> None:
    """Set the global transaction manager instance."""
    global _global_transaction_manager
    _global_transaction_manager = manager


def reset_transaction_manager() -> None:
    """Reset the global transaction manager (for testing)."""
    global _global_transaction_manager
    _global_transaction_manager = None


__all__ = [
    # Enums
    "TransactionState",
    "ChangeType",
    # Data classes
    "StagedChange",
    "DocumentSnapshot",
    # Protocols
    "TransactionDocumentProvider",
    "TransactionListener",
    # Transaction
    "Transaction",
    "TransactionManager",
    # Errors
    "TransactionError",
    "CommitError",
    # Global access
    "get_transaction_manager",
    "set_transaction_manager",
    "reset_transaction_manager",
]
