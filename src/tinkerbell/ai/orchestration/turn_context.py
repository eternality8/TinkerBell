"""Turn context for AI operations.

This module provides a context object that captures state at the start of an AI turn
and carries it through all tool calls within that turn. This prevents race conditions
where the user might switch tabs while the AI is processing.

The TurnContext is created at the start of each AI turn and passed to the tool
dispatcher, which uses it to resolve tab IDs and other context-dependent values.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


@dataclass(slots=True)
class TurnContext:
    """Context captured at the start of an AI turn.
    
    This context is "pinned" at turn start and remains stable throughout
    the turn, even if the user switches tabs or makes other UI changes.
    
    Attributes:
        turn_id: Unique identifier for this turn.
        pinned_tab_id: The active tab ID at turn start, used as default for tools.
        document_id: The document ID at turn start (for tracking).
        started_at: Timestamp when the turn started.
        metadata: Additional turn metadata (extensible).
    """
    
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pinned_tab_id: str | None = None
    document_id: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any] | None,
        *,
        turn_id: str | None = None,
    ) -> "TurnContext":
        """Create a TurnContext from a document snapshot.
        
        This is the primary factory method for creating turn contexts.
        It extracts the tab_id and document_id from the snapshot provided
        at the start of the AI turn.
        
        Args:
            snapshot: Document snapshot from the start of the turn.
            turn_id: Optional explicit turn ID (generates one if not provided).
            
        Returns:
            A new TurnContext with pinned values from the snapshot.
        """
        snapshot = snapshot or {}
        
        # Extract tab_id - try multiple possible keys
        pinned_tab_id = (
            snapshot.get("tab_id")
            or snapshot.get("tabId")
            or _extract_tab_id_from_version(snapshot.get("version"))
        )
        
        # Extract document_id
        document_id = snapshot.get("document_id") or snapshot.get("documentId")
        
        return cls(
            turn_id=turn_id or str(uuid.uuid4()),
            pinned_tab_id=pinned_tab_id,
            document_id=document_id,
            metadata={
                "snapshot_version": snapshot.get("version"),
                "snapshot_path": snapshot.get("path"),
            },
        )
    
    def with_tab_id(self, tab_id: str) -> "TurnContext":
        """Return a copy with a different pinned tab ID.
        
        Useful for operations that explicitly target a different tab
        while still preserving other turn context.
        
        Args:
            tab_id: The new tab ID to pin.
            
        Returns:
            A new TurnContext with the updated tab ID.
        """
        return TurnContext(
            turn_id=self.turn_id,
            pinned_tab_id=tab_id,
            document_id=self.document_id,
            started_at=self.started_at,
            metadata=dict(self.metadata),
        )
    
    def pin_tab_if_empty(self, tab_id: str) -> bool:
        """Pin a tab ID if none is currently pinned.
        
        This is used when a turn starts without any open documents, and
        the first document created during the turn should become the
        pinned tab for subsequent operations.
        
        Args:
            tab_id: The tab ID to pin.
            
        Returns:
            True if the tab was pinned, False if already pinned.
        """
        if self.pinned_tab_id is None:
            self.pinned_tab_id = tab_id
            return True
        return False
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize the turn context for logging/debugging."""
        return {
            "turn_id": self.turn_id,
            "pinned_tab_id": self.pinned_tab_id,
            "document_id": self.document_id,
            "started_at": self.started_at.isoformat(),
            "metadata": dict(self.metadata),
        }


def _extract_tab_id_from_version(version: str | None) -> str | None:
    """Extract tab_id from a version token string.
    
    Version tokens have the format: "tab_id:hash:version_id"
    
    Args:
        version: Version token string.
        
    Returns:
        The tab_id portion, or None if not parseable.
    """
    if not version or not isinstance(version, str):
        return None
    parts = version.split(":")
    if len(parts) >= 1 and parts[0]:
        return parts[0]
    return None


__all__ = ["TurnContext"]
