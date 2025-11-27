"""Version token system for AI tool operations.

This module provides per-tab version tracking to ensure write operations
reference the latest document state and prevent stale edits.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Protocol


class VersionProvider(Protocol):
    """Protocol for document sources that provide version information."""

    @property
    def document_id(self) -> str:
        """Unique identifier for the document."""
        ...

    @property
    def version_id(self) -> int:
        """Monotonically increasing version number."""
        ...

    @property
    def content_hash(self) -> str:
        """SHA-1 hash of the document content."""
        ...


@dataclass(slots=True, frozen=True)
class VersionToken:
    """Immutable representation of a document version state.

    The version token encapsulates document identity and current version,
    used to validate write operations against stale state.

    Attributes:
        tab_id: Identifier for the editor tab containing the document.
        document_id: Unique identifier for the document.
        version_id: Monotonically increasing version number (starts at 1).
        content_hash: SHA-1 hash of the document content for integrity checking.
    """

    tab_id: str
    document_id: str
    version_id: int
    content_hash: str

    def to_string(self) -> str:
        """Serialize the token to a compact string format.

        Format: `{tab_id}:{document_id}:{version_id}:{content_hash}`
        """
        return f"{self.tab_id}:{self.document_id}:{self.version_id}:{self.content_hash}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize the token to a dictionary."""
        return {
            "tab_id": self.tab_id,
            "document_id": self.document_id,
            "version_id": self.version_id,
            "content_hash": self.content_hash,
        }

    def matches(self, other: "VersionToken") -> bool:
        """Check if this token matches another (same document and version)."""
        return (
            self.tab_id == other.tab_id
            and self.document_id == other.document_id
            and self.version_id == other.version_id
            and self.content_hash == other.content_hash
        )

    def is_stale_compared_to(self, current: "VersionToken") -> bool:
        """Check if this token is stale compared to a current version.

        Returns True if the document IDs match but the version or content
        hash differs, indicating the document has changed since this token
        was issued.
        """
        if self.tab_id != current.tab_id or self.document_id != current.document_id:
            return False  # Different documents, not comparable
        return self.version_id < current.version_id or self.content_hash != current.content_hash

    @classmethod
    def from_string(cls, token_str: str) -> "VersionToken":
        """Parse a version token from its string representation.

        Args:
            token_str: Token string in format `tab_id:document_id:version_id:content_hash`

        Returns:
            Parsed VersionToken instance.

        Raises:
            ValueError: If the token string is malformed.
        """
        if not token_str or not isinstance(token_str, str):
            raise ValueError("Version token must be a non-empty string")

        parts = token_str.strip().split(":")
        if len(parts) < 4:
            raise ValueError(
                f"Invalid version token format: expected 'tab_id:document_id:version_id:content_hash', "
                f"got '{token_str}'"
            )

        tab_id = parts[0]
        document_id = parts[1]
        try:
            version_id = int(parts[2])
        except ValueError as exc:
            raise ValueError(f"Invalid version_id in token: '{parts[2]}' is not an integer") from exc
        content_hash = ":".join(parts[3:])  # Content hash may contain colons (unlikely but safe)

        if not tab_id:
            raise ValueError("Version token tab_id cannot be empty")
        if not document_id:
            raise ValueError("Version token document_id cannot be empty")
        if version_id < 1:
            raise ValueError(f"Version token version_id must be >= 1, got {version_id}")
        if not content_hash:
            raise ValueError("Version token content_hash cannot be empty")

        return cls(
            tab_id=tab_id,
            document_id=document_id,
            version_id=version_id,
            content_hash=content_hash,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VersionToken":
        """Create a VersionToken from a dictionary.

        Args:
            data: Dictionary with keys: tab_id, document_id, version_id, content_hash

        Returns:
            VersionToken instance.

        Raises:
            ValueError: If required keys are missing or values are invalid.
        """
        if not isinstance(data, Mapping):
            raise ValueError("Version token data must be a mapping")

        tab_id = data.get("tab_id")
        document_id = data.get("document_id")
        version_id = data.get("version_id")
        content_hash = data.get("content_hash")

        if not tab_id:
            raise ValueError("Version token requires 'tab_id'")
        if not document_id:
            raise ValueError("Version token requires 'document_id'")
        if version_id is None:
            raise ValueError("Version token requires 'version_id'")
        if not content_hash:
            raise ValueError("Version token requires 'content_hash'")

        try:
            version_int = int(version_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Version token version_id must be an integer, got {type(version_id).__name__}") from exc

        return cls(
            tab_id=str(tab_id),
            document_id=str(document_id),
            version_id=version_int,
            content_hash=str(content_hash),
        )


@dataclass(slots=True)
class TabVersionState:
    """Internal state tracking for a single tab's version history."""

    tab_id: str
    document_id: str
    version_id: int = 1
    content_hash: str = ""
    is_dirty: bool = False

    def increment(self, content_hash: str) -> int:
        """Increment the version and update the content hash.

        Returns:
            The new version_id after incrementing.
        """
        self.version_id += 1
        self.content_hash = content_hash
        self.is_dirty = True
        return self.version_id

    def reset(self, document_id: str, content_hash: str) -> None:
        """Reset version tracking for a new or reloaded document.

        Called when a document is loaded from disk or a new document is created.
        """
        self.document_id = document_id
        self.version_id = 1
        self.content_hash = content_hash
        self.is_dirty = False

    def to_token(self) -> VersionToken:
        """Create a VersionToken from the current state."""
        return VersionToken(
            tab_id=self.tab_id,
            document_id=self.document_id,
            version_id=self.version_id,
            content_hash=self.content_hash,
        )


class VersionManager:
    """Centralized version management for all editor tabs.

    Thread-safe manager that tracks document versions across tabs and validates
    version tokens for write operations.

    Example usage:
        >>> manager = VersionManager()
        >>> token = manager.register_tab("tab-1", "doc-abc", "a1b2c3...")
        >>> manager.validate_token(token)  # Returns True
        >>> manager.increment_version("tab-1", "new-hash")
        >>> manager.validate_token(token)  # Raises VersionMismatchError
    """

    def __init__(self) -> None:
        self._tabs: MutableMapping[str, TabVersionState] = {}
        self._lock = threading.RLock()

    def register_tab(
        self,
        tab_id: str,
        document_id: str,
        content_hash: str,
    ) -> VersionToken:
        """Register a new tab or update an existing tab's document.

        Args:
            tab_id: Unique identifier for the tab.
            document_id: Unique identifier for the document.
            content_hash: SHA-1 hash of the document content.

        Returns:
            VersionToken representing the current document state.
        """
        with self._lock:
            existing = self._tabs.get(tab_id)
            if existing is not None:
                # Tab exists, reset for new document
                existing.reset(document_id, content_hash)
                return existing.to_token()

            # Create new tab state
            state = TabVersionState(
                tab_id=tab_id,
                document_id=document_id,
                version_id=1,
                content_hash=content_hash,
            )
            self._tabs[tab_id] = state
            return state.to_token()

    def unregister_tab(self, tab_id: str) -> bool:
        """Remove a tab from version tracking.

        Args:
            tab_id: Identifier of the tab to remove.

        Returns:
            True if the tab was removed, False if it didn't exist.
        """
        with self._lock:
            if tab_id in self._tabs:
                del self._tabs[tab_id]
                return True
            return False

    def get_current_token(self, tab_id: str) -> VersionToken | None:
        """Get the current version token for a tab.

        Args:
            tab_id: Identifier of the tab.

        Returns:
            Current VersionToken, or None if the tab is not registered.
        """
        with self._lock:
            state = self._tabs.get(tab_id)
            if state is None:
                return None
            return state.to_token()

    def increment_version(self, tab_id: str, content_hash: str) -> VersionToken:
        """Increment the version for a tab after a successful edit.

        Args:
            tab_id: Identifier of the tab that was edited.
            content_hash: New SHA-1 hash of the document content.

        Returns:
            New VersionToken after incrementing.

        Raises:
            KeyError: If the tab is not registered.
        """
        with self._lock:
            state = self._tabs.get(tab_id)
            if state is None:
                raise KeyError(f"Tab '{tab_id}' is not registered")
            state.increment(content_hash)
            return state.to_token()

    def validate_token(self, token: VersionToken | str) -> bool:
        """Validate that a version token matches the current document state.

        Args:
            token: VersionToken instance or token string to validate.

        Returns:
            True if the token is valid and current.

        Raises:
            VersionMismatchError: If the token is stale or invalid.
            KeyError: If the tab is not registered.
        """
        if isinstance(token, str):
            token = VersionToken.from_string(token)

        with self._lock:
            state = self._tabs.get(token.tab_id)
            if state is None:
                raise KeyError(f"Tab '{token.tab_id}' is not registered")

            current = state.to_token()

            if token.document_id != current.document_id:
                raise VersionMismatchError(
                    message="Document has been replaced since the token was issued",
                    your_version=token,
                    current_version=current,
                    suggestion="Fetch a new snapshot with read_document to get the current state.",
                )

            if token.version_id < current.version_id:
                raise VersionMismatchError(
                    message=f"Document has been edited: your version {token.version_id} < current {current.version_id}",
                    your_version=token,
                    current_version=current,
                    suggestion="Fetch a new snapshot with read_document and rebase your changes.",
                )

            if token.content_hash != current.content_hash:
                raise VersionMismatchError(
                    message="Document content has changed (hash mismatch)",
                    your_version=token,
                    current_version=current,
                    suggestion="Fetch a new snapshot with read_document to see the current content.",
                )

            return True

    def reset_on_reload(self, tab_id: str, document_id: str, content_hash: str) -> VersionToken:
        """Reset version tracking when a document is reloaded from disk.

        This resets the version_id to 1, indicating a fresh document state.

        Args:
            tab_id: Identifier of the tab.
            document_id: Unique identifier for the document.
            content_hash: SHA-1 hash of the reloaded content.

        Returns:
            New VersionToken with version_id = 1.
        """
        with self._lock:
            state = self._tabs.get(tab_id)
            if state is None:
                return self.register_tab(tab_id, document_id, content_hash)
            state.reset(document_id, content_hash)
            return state.to_token()

    def list_tabs(self) -> list[str]:
        """Return a list of all registered tab IDs."""
        with self._lock:
            return list(self._tabs.keys())

    def clear(self) -> None:
        """Remove all tab registrations (for testing or shutdown)."""
        with self._lock:
            self._tabs.clear()


@dataclass
class VersionMismatchError(Exception):
    """Exception raised when an operation references a stale document version.

    Attributes:
        message: Human-readable description of the mismatch.
        your_version: The stale version token provided by the caller.
        current_version: The actual current version token.
        suggestion: Actionable guidance for the AI to recover.
    """

    message: str
    your_version: VersionToken | None = None
    current_version: VersionToken | None = None
    suggestion: str = ""

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary for JSON tool responses."""
        result: dict[str, Any] = {
            "error": "version_mismatch",
            "message": self.message,
        }
        if self.your_version is not None:
            result["your_version"] = self.your_version.to_dict()
        if self.current_version is not None:
            result["current_version"] = self.current_version.to_dict()
        if self.suggestion:
            result["suggestion"] = self.suggestion
        return result


def compute_content_hash(text: str) -> str:
    """Compute a SHA-1 hash of the document text.

    Args:
        text: Document content to hash.

    Returns:
        Hexadecimal SHA-1 hash string.
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# Module-level singleton for convenient access (optional usage)
_default_manager: VersionManager | None = None


def get_version_manager() -> VersionManager:
    """Get or create the default global VersionManager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = VersionManager()
    return _default_manager


def reset_version_manager() -> None:
    """Reset the global VersionManager (primarily for testing)."""
    global _default_manager
    if _default_manager is not None:
        _default_manager.clear()
    _default_manager = None


__all__ = [
    "VersionToken",
    "VersionManager",
    "VersionMismatchError",
    "TabVersionState",
    "VersionProvider",
    "compute_content_hash",
    "get_version_manager",
    "reset_version_manager",
]
