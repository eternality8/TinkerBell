"""Turn Checkpoints System for Document State Management.

Provides checkpoint creation, storage, and restoration for document states,
enabling users to review and rollback AI changes.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Checkpoint Types
# -----------------------------------------------------------------------------


class CheckpointType(Enum):
    """Type of checkpoint."""

    PRE_TURN = auto()  # Before AI turn
    POST_TURN = auto()  # After AI turn (committed)
    MANUAL = auto()  # User-created checkpoint
    AUTO = auto()  # Auto-save checkpoint


# -----------------------------------------------------------------------------
# Checkpoint Data
# -----------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class DocumentState:
    """Immutable snapshot of a single document's state.

    Attributes:
        tab_id: Document tab identifier.
        content: Full document content.
        version_token: Version token at snapshot time.
        cursor_position: Cursor position (line, column).
        selection: Selection range if any.
    """

    tab_id: str
    content: str
    version_token: str | None = None
    cursor_position: tuple[int, int] | None = None
    selection: tuple[int, int, int, int] | None = None  # start_line, start_col, end_line, end_col

    def content_hash(self) -> int:
        """Get content hash for quick comparison."""
        return hash(self.content)

    def line_count(self) -> int:
        """Get number of lines in the document."""
        return self.content.count("\n") + 1

    def char_count(self) -> int:
        """Get number of characters in the document."""
        return len(self.content)


@dataclass(slots=True)
class Checkpoint:
    """A checkpoint representing a point-in-time state of documents.

    Attributes:
        checkpoint_id: Unique identifier.
        turn_number: AI turn number (if applicable).
        checkpoint_type: Type of checkpoint.
        timestamp: When the checkpoint was created.
        documents: Map of tab_id -> DocumentState.
        action_summary: Brief description of changes that led to this state.
        metadata: Additional checkpoint metadata.
    """

    checkpoint_id: str
    turn_number: int | None
    checkpoint_type: CheckpointType
    timestamp: datetime
    documents: dict[str, DocumentState]
    action_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tab_ids(self) -> list[str]:
        """Get list of tab IDs in this checkpoint."""
        return list(self.documents.keys())

    def get_document(self, tab_id: str) -> DocumentState | None:
        """Get document state for a specific tab."""
        return self.documents.get(tab_id)

    def total_size(self) -> int:
        """Get total size of all documents in characters."""
        return sum(doc.char_count() for doc in self.documents.values())


# -----------------------------------------------------------------------------
# Document Provider Protocol
# -----------------------------------------------------------------------------


class CheckpointDocumentProvider(Protocol):
    """Protocol for document access in checkpoints."""

    def get_document_content(self, tab_id: str) -> str | None:
        """Get the current content of a document."""
        ...

    def set_document_content(self, tab_id: str, content: str) -> None:
        """Set the content of a document."""
        ...

    def get_version_token(self, tab_id: str) -> str | None:
        """Get the current version token for a document."""
        ...

    def get_cursor_position(self, tab_id: str) -> tuple[int, int] | None:
        """Get the current cursor position (line, column)."""
        ...

    def get_selection(self, tab_id: str) -> tuple[int, int, int, int] | None:
        """Get the current selection range."""
        ...


class CheckpointListener(Protocol):
    """Callback for checkpoint events."""

    def on_checkpoint_created(self, checkpoint: Checkpoint) -> None:
        """Called when a checkpoint is created."""
        ...

    def on_checkpoint_restored(self, checkpoint: Checkpoint) -> None:
        """Called when a checkpoint is restored."""
        ...

    def on_checkpoints_cleared(self, tab_id: str | None) -> None:
        """Called when checkpoints are cleared."""
        ...


# -----------------------------------------------------------------------------
# Diff Computation
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class CheckpointDiff:
    """Represents the difference between current state and a checkpoint.

    Attributes:
        checkpoint_id: The checkpoint being compared against.
        tab_id: The document tab.
        has_changes: Whether there are any changes.
        lines_added: Number of lines added.
        lines_removed: Number of lines removed.
        chars_added: Number of characters added.
        chars_removed: Number of characters removed.
        current_content: Current document content.
        checkpoint_content: Content at checkpoint.
    """

    checkpoint_id: str
    tab_id: str
    has_changes: bool
    lines_added: int = 0
    lines_removed: int = 0
    chars_added: int = 0
    chars_removed: int = 0
    current_content: str = ""
    checkpoint_content: str = ""


def compute_simple_diff(current: str, checkpoint: str) -> tuple[int, int, int, int]:
    """Compute simple diff statistics between two texts.

    Returns:
        Tuple of (lines_added, lines_removed, chars_added, chars_removed).
    """
    current_lines = current.split("\n")
    checkpoint_lines = checkpoint.split("\n")

    current_set = set(current_lines)
    checkpoint_set = set(checkpoint_lines)

    lines_added = len(current_set - checkpoint_set)
    lines_removed = len(checkpoint_set - current_set)

    chars_added = max(0, len(current) - len(checkpoint))
    chars_removed = max(0, len(checkpoint) - len(current))

    return lines_added, lines_removed, chars_added, chars_removed


# -----------------------------------------------------------------------------
# Checkpoint Store
# -----------------------------------------------------------------------------


class CheckpointStore:
    """Stores and manages checkpoints for documents.

    Checkpoints are stored per-document (tab_id) and are session-scoped,
    meaning they are cleared when the document is closed.

    The store supports:
    - Creating checkpoints (pre/post turn, manual, auto)
    - Restoring documents to checkpoint states
    - Computing diffs from current state
    - Querying checkpoint history
    """

    def __init__(
        self,
        *,
        document_provider: CheckpointDocumentProvider | None = None,
        listener: CheckpointListener | None = None,
        max_checkpoints_per_tab: int = 50,
    ) -> None:
        """Initialize the checkpoint store.

        Args:
            document_provider: Provider for document access.
            listener: Checkpoint event listener.
            max_checkpoints_per_tab: Maximum checkpoints to keep per document.
        """
        self._document_provider = document_provider
        self._listener = listener
        self._max_per_tab = max_checkpoints_per_tab

        self._lock = threading.RLock()
        # tab_id -> list of checkpoints (ordered by timestamp, newest last)
        self._checkpoints: dict[str, list[Checkpoint]] = {}
        self._checkpoint_counter = 0

    # ------------------------------------------------------------------
    # Checkpoint Creation
    # ------------------------------------------------------------------

    def create_checkpoint(
        self,
        tab_ids: Sequence[str],
        *,
        checkpoint_type: CheckpointType = CheckpointType.MANUAL,
        turn_number: int | None = None,
        action_summary: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Create a checkpoint for one or more documents.

        Args:
            tab_ids: Tab IDs to include in the checkpoint.
            checkpoint_type: Type of checkpoint.
            turn_number: AI turn number (if applicable).
            action_summary: Brief description of the action.
            metadata: Additional metadata.

        Returns:
            The created checkpoint.
        """
        with self._lock:
            self._checkpoint_counter += 1
            checkpoint_id = f"ckpt-{self._checkpoint_counter}-{uuid.uuid4().hex[:8]}"

            # Capture document states
            documents: dict[str, DocumentState] = {}
            for tab_id in tab_ids:
                state = self._capture_document_state(tab_id)
                if state:
                    documents[tab_id] = state

            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                turn_number=turn_number,
                checkpoint_type=checkpoint_type,
                timestamp=datetime.now(timezone.utc),
                documents=documents,
                action_summary=action_summary,
                metadata=metadata or {},
            )

            # Store checkpoint for each tab
            for tab_id in documents:
                if tab_id not in self._checkpoints:
                    self._checkpoints[tab_id] = []

                self._checkpoints[tab_id].append(checkpoint)

                # Enforce max limit
                while len(self._checkpoints[tab_id]) > self._max_per_tab:
                    self._checkpoints[tab_id].pop(0)

            # Notify listener
            if self._listener:
                try:
                    self._listener.on_checkpoint_created(checkpoint)
                except Exception:
                    LOGGER.debug("Listener failed on checkpoint created", exc_info=True)

            LOGGER.debug(
                "Created checkpoint %s: type=%s, turn=%s, tabs=%d",
                checkpoint_id,
                checkpoint_type.name,
                turn_number,
                len(documents),
            )

            return checkpoint

    def create_pre_turn_checkpoint(
        self,
        tab_ids: Sequence[str],
        turn_number: int,
    ) -> Checkpoint:
        """Create a checkpoint before an AI turn.

        Args:
            tab_ids: Tab IDs to checkpoint.
            turn_number: The upcoming turn number.

        Returns:
            The created checkpoint.
        """
        return self.create_checkpoint(
            tab_ids,
            checkpoint_type=CheckpointType.PRE_TURN,
            turn_number=turn_number,
            action_summary=f"Before AI turn {turn_number}",
        )

    def create_post_turn_checkpoint(
        self,
        tab_ids: Sequence[str],
        turn_number: int,
        action_summary: str = "",
    ) -> Checkpoint:
        """Create a checkpoint after an AI turn.

        Args:
            tab_ids: Tab IDs to checkpoint.
            turn_number: The completed turn number.
            action_summary: Summary of what the turn did.

        Returns:
            The created checkpoint.
        """
        return self.create_checkpoint(
            tab_ids,
            checkpoint_type=CheckpointType.POST_TURN,
            turn_number=turn_number,
            action_summary=action_summary or f"After AI turn {turn_number}",
        )

    # ------------------------------------------------------------------
    # Checkpoint Restoration
    # ------------------------------------------------------------------

    def restore_checkpoint(
        self,
        checkpoint_id: str,
        *,
        tab_ids: Sequence[str] | None = None,
        create_new_checkpoint: bool = True,
    ) -> bool:
        """Restore documents to a checkpoint state.

        Args:
            checkpoint_id: ID of checkpoint to restore.
            tab_ids: Specific tabs to restore (None = all in checkpoint).
            create_new_checkpoint: If True, create a new checkpoint before restoring.

        Returns:
            True if restored successfully.
        """
        with self._lock:
            # Find the checkpoint
            checkpoint = self._find_checkpoint(checkpoint_id)
            if not checkpoint:
                LOGGER.warning("Checkpoint not found: %s", checkpoint_id)
                return False

            if not self._document_provider:
                LOGGER.error("No document provider available for restoration")
                return False

            # Determine which tabs to restore
            tabs_to_restore = (
                [tid for tid in tab_ids if tid in checkpoint.documents]
                if tab_ids
                else list(checkpoint.documents.keys())
            )

            if not tabs_to_restore:
                LOGGER.warning("No matching tabs to restore in checkpoint %s", checkpoint_id)
                return False

            # Create a new checkpoint before restoring (non-destructive)
            if create_new_checkpoint:
                self.create_checkpoint(
                    tabs_to_restore,
                    checkpoint_type=CheckpointType.MANUAL,
                    action_summary=f"Before restore to {checkpoint_id}",
                )

            # Restore each tab
            for tab_id in tabs_to_restore:
                doc_state = checkpoint.documents[tab_id]
                try:
                    self._document_provider.set_document_content(
                        tab_id,
                        doc_state.content,
                    )
                except Exception:
                    LOGGER.error(
                        "Failed to restore tab %s from checkpoint %s",
                        tab_id,
                        checkpoint_id,
                        exc_info=True,
                    )
                    return False

            # Notify listener
            if self._listener:
                try:
                    self._listener.on_checkpoint_restored(checkpoint)
                except Exception:
                    LOGGER.debug("Listener failed on checkpoint restored", exc_info=True)

            LOGGER.info(
                "Restored %d tabs from checkpoint %s",
                len(tabs_to_restore),
                checkpoint_id,
            )

            return True

    def restore_to_pre_turn(self, turn_number: int, tab_id: str) -> bool:
        """Restore a document to its state before a specific turn.

        Args:
            turn_number: The turn to restore before.
            tab_id: The tab to restore.

        Returns:
            True if restored successfully.
        """
        with self._lock:
            checkpoints = self._checkpoints.get(tab_id, [])

            # Find the pre-turn checkpoint
            for checkpoint in reversed(checkpoints):
                if (
                    checkpoint.checkpoint_type == CheckpointType.PRE_TURN
                    and checkpoint.turn_number == turn_number
                ):
                    return self.restore_checkpoint(checkpoint.checkpoint_id, tab_ids=[tab_id])

            LOGGER.warning(
                "Pre-turn checkpoint not found: turn=%d, tab=%s",
                turn_number,
                tab_id,
            )
            return False

    # ------------------------------------------------------------------
    # Diff Computation
    # ------------------------------------------------------------------

    def compute_diff(
        self,
        checkpoint_id: str,
        tab_id: str,
    ) -> CheckpointDiff | None:
        """Compute diff between current document state and a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to compare against.
            tab_id: Tab to compute diff for.

        Returns:
            CheckpointDiff or None if checkpoint not found.
        """
        with self._lock:
            checkpoint = self._find_checkpoint(checkpoint_id)
            if not checkpoint or tab_id not in checkpoint.documents:
                return None

            if not self._document_provider:
                return None

            checkpoint_content = checkpoint.documents[tab_id].content
            current_content = self._document_provider.get_document_content(tab_id)
            if current_content is None:
                return None

            has_changes = current_content != checkpoint_content
            lines_added, lines_removed, chars_added, chars_removed = (
                compute_simple_diff(current_content, checkpoint_content)
                if has_changes
                else (0, 0, 0, 0)
            )

            return CheckpointDiff(
                checkpoint_id=checkpoint_id,
                tab_id=tab_id,
                has_changes=has_changes,
                lines_added=lines_added,
                lines_removed=lines_removed,
                chars_added=chars_added,
                chars_removed=chars_removed,
                current_content=current_content,
                checkpoint_content=checkpoint_content,
            )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_checkpoints_for_tab(
        self,
        tab_id: str,
        *,
        checkpoint_type: CheckpointType | None = None,
        limit: int | None = None,
    ) -> list[Checkpoint]:
        """Get checkpoints for a specific tab.

        Args:
            tab_id: Tab to get checkpoints for.
            checkpoint_type: Optional filter by type.
            limit: Maximum number to return (newest first).

        Returns:
            List of checkpoints (newest first).
        """
        with self._lock:
            checkpoints = list(reversed(self._checkpoints.get(tab_id, [])))

            if checkpoint_type is not None:
                checkpoints = [c for c in checkpoints if c.checkpoint_type == checkpoint_type]

            if limit is not None:
                checkpoints = checkpoints[:limit]

            return checkpoints

    def get_latest_checkpoint(
        self,
        tab_id: str,
        *,
        checkpoint_type: CheckpointType | None = None,
    ) -> Checkpoint | None:
        """Get the most recent checkpoint for a tab.

        Args:
            tab_id: Tab to get checkpoint for.
            checkpoint_type: Optional filter by type.

        Returns:
            The latest checkpoint or None.
        """
        checkpoints = self.get_checkpoints_for_tab(
            tab_id,
            checkpoint_type=checkpoint_type,
            limit=1,
        )
        return checkpoints[0] if checkpoints else None

    def get_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Get a specific checkpoint by ID."""
        with self._lock:
            return self._find_checkpoint(checkpoint_id)

    def get_checkpoint_count(self, tab_id: str) -> int:
        """Get the number of checkpoints for a tab."""
        with self._lock:
            return len(self._checkpoints.get(tab_id, []))

    def get_all_tabs_with_checkpoints(self) -> list[str]:
        """Get list of all tabs that have checkpoints."""
        with self._lock:
            return [tab_id for tab_id, cps in self._checkpoints.items() if cps]

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def clear_checkpoints(self, tab_id: str | None = None) -> int:
        """Clear checkpoints.

        Args:
            tab_id: Specific tab to clear (None = clear all).

        Returns:
            Number of checkpoints cleared.
        """
        with self._lock:
            if tab_id:
                count = len(self._checkpoints.get(tab_id, []))
                self._checkpoints.pop(tab_id, None)
            else:
                count = sum(len(cps) for cps in self._checkpoints.values())
                self._checkpoints.clear()

            # Notify listener
            if self._listener:
                try:
                    self._listener.on_checkpoints_cleared(tab_id)
                except Exception:
                    LOGGER.debug("Listener failed on checkpoints cleared", exc_info=True)

            LOGGER.debug("Cleared %d checkpoints (tab=%s)", count, tab_id or "all")
            return count

    def prune_old_checkpoints(
        self,
        *,
        older_than_hours: float = 24,
    ) -> int:
        """Remove checkpoints older than a threshold.

        Args:
            older_than_hours: Age threshold in hours.

        Returns:
            Number of checkpoints pruned.
        """
        from datetime import timedelta

        threshold = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
        pruned = 0

        with self._lock:
            for tab_id in list(self._checkpoints.keys()):
                original = self._checkpoints[tab_id]
                filtered = [c for c in original if c.timestamp >= threshold]
                pruned += len(original) - len(filtered)
                self._checkpoints[tab_id] = filtered

        if pruned > 0:
            LOGGER.debug("Pruned %d old checkpoints", pruned)

        return pruned

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _capture_document_state(self, tab_id: str) -> DocumentState | None:
        """Capture the current state of a document."""
        if not self._document_provider:
            return None

        content = self._document_provider.get_document_content(tab_id)
        if content is None:
            return None

        return DocumentState(
            tab_id=tab_id,
            content=content,
            version_token=self._document_provider.get_version_token(tab_id),
            cursor_position=self._document_provider.get_cursor_position(tab_id),
            selection=self._document_provider.get_selection(tab_id),
        )

    def _find_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        """Find a checkpoint by ID across all tabs."""
        for checkpoints in self._checkpoints.values():
            for checkpoint in checkpoints:
                if checkpoint.checkpoint_id == checkpoint_id:
                    return checkpoint
        return None


# -----------------------------------------------------------------------------
# Global Instance
# -----------------------------------------------------------------------------

_global_checkpoint_store: CheckpointStore | None = None


def get_checkpoint_store() -> CheckpointStore:
    """Get the global checkpoint store instance."""
    global _global_checkpoint_store
    if _global_checkpoint_store is None:
        _global_checkpoint_store = CheckpointStore()
    return _global_checkpoint_store


def set_checkpoint_store(store: CheckpointStore | None) -> None:
    """Set the global checkpoint store instance."""
    global _global_checkpoint_store
    _global_checkpoint_store = store


def reset_checkpoint_store() -> None:
    """Reset the global checkpoint store (for testing)."""
    global _global_checkpoint_store
    if _global_checkpoint_store:
        _global_checkpoint_store.clear_checkpoints()
    _global_checkpoint_store = None


__all__ = [
    # Enums
    "CheckpointType",
    # Data classes
    "DocumentState",
    "Checkpoint",
    "CheckpointDiff",
    # Protocols
    "CheckpointDocumentProvider",
    "CheckpointListener",
    # Store
    "CheckpointStore",
    # Helpers
    "compute_simple_diff",
    # Global access
    "get_checkpoint_store",
    "set_checkpoint_store",
    "reset_checkpoint_store",
]
