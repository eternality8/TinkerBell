"""Tests for the version token system."""

from __future__ import annotations

import pytest

from tinkerbell.ai.tools.version import (
    VersionToken,
    VersionManager,
    VersionMismatchError,
    TabVersionState,
    compute_content_hash,
    get_version_manager,
    reset_version_manager,
)


class TestVersionToken:
    """Tests for VersionToken dataclass."""

    def test_create_token(self) -> None:
        """Test basic token creation."""
        token = VersionToken(
            tab_id="tab-1",
            document_id="doc-abc",
            version_id=1,
            content_hash="abc123",
        )
        assert token.tab_id == "tab-1"
        assert token.document_id == "doc-abc"
        assert token.version_id == 1
        assert token.content_hash == "abc123"

    def test_to_string(self) -> None:
        """Test token serialization to compact external format."""
        token = VersionToken(
            tab_id="t1",
            document_id="doc-abc",
            version_id=5,
            content_hash="deadbeef1234",
        )
        # New compact format: tab_id:short_hash:version
        assert token.to_string() == "t1:dead:5"

    def test_to_string_format(self) -> None:
        """Test that to_string produces the expected short format."""
        token = VersionToken(
            tab_id="t42",
            document_id="doc-xyz",
            version_id=42,
            content_hash="fedcba9876543210",
        )
        result = token.to_string()
        # Format: {tab_id}:{4 chars from hash}:{version}
        assert result == "t42:fedc:42"
        # Should be compact (under 15 chars typically)
        assert len(result) < 20

    def test_to_dict(self) -> None:
        """Test token serialization to dictionary."""
        token = VersionToken(
            tab_id="tab-1",
            document_id="doc-abc",
            version_id=5,
            content_hash="deadbeef",
        )
        result = token.to_dict()
        assert result == {
            "tab_id": "tab-1",
            "document_id": "doc-abc",
            "version_id": 5,
            "content_hash": "deadbeef",
        }

    def test_from_string_valid(self) -> None:
        """Test parsing valid token string."""
        token = VersionToken.from_string("tab-1:doc-abc:5:deadbeef")
        assert token.tab_id == "tab-1"
        assert token.document_id == "doc-abc"
        assert token.version_id == 5
        assert token.content_hash == "deadbeef"

    def test_from_string_with_colons_in_hash(self) -> None:
        """Test parsing token where hash contains colons."""
        token = VersionToken.from_string("tab-1:doc-abc:5:hash:with:colons")
        assert token.content_hash == "hash:with:colons"

    def test_from_string_invalid_format(self) -> None:
        """Test parsing invalid token string."""
        with pytest.raises(ValueError, match="Invalid version token format"):
            VersionToken.from_string("invalid-token")

    def test_from_string_invalid_version_id(self) -> None:
        """Test parsing token with non-integer version."""
        with pytest.raises(ValueError, match="not an integer"):
            VersionToken.from_string("tab-1:doc-abc:not-a-number:hash")

    def test_from_string_empty_components(self) -> None:
        """Test parsing token with empty components."""
        with pytest.raises(ValueError, match="tab_id cannot be empty"):
            VersionToken.from_string(":doc-abc:1:hash")

    def test_from_string_version_below_one(self) -> None:
        """Test parsing token with version < 1."""
        with pytest.raises(ValueError, match="version_id must be >= 1"):
            VersionToken.from_string("tab-1:doc-abc:0:hash")

    def test_from_dict_valid(self) -> None:
        """Test creating token from dictionary."""
        data = {
            "tab_id": "tab-1",
            "document_id": "doc-abc",
            "version_id": 3,
            "content_hash": "xyz789",
        }
        token = VersionToken.from_dict(data)
        assert token.tab_id == "tab-1"
        assert token.version_id == 3

    def test_from_dict_missing_keys(self) -> None:
        """Test creating token from dictionary with missing keys."""
        with pytest.raises(ValueError, match="requires 'tab_id'"):
            VersionToken.from_dict({"document_id": "doc", "version_id": 1, "content_hash": "h"})

    def test_matches(self) -> None:
        """Test token matching."""
        token1 = VersionToken("tab-1", "doc-abc", 1, "hash")
        token2 = VersionToken("tab-1", "doc-abc", 1, "hash")
        token3 = VersionToken("tab-1", "doc-abc", 2, "hash")

        assert token1.matches(token2)
        assert not token1.matches(token3)

    def test_is_stale_compared_to(self) -> None:
        """Test staleness detection."""
        old_token = VersionToken("tab-1", "doc-abc", 1, "oldhash")
        new_token = VersionToken("tab-1", "doc-abc", 2, "newhash")
        same_token = VersionToken("tab-1", "doc-abc", 1, "oldhash")
        different_doc = VersionToken("tab-1", "doc-xyz", 5, "hash")

        assert old_token.is_stale_compared_to(new_token)
        assert not old_token.is_stale_compared_to(same_token)
        assert not old_token.is_stale_compared_to(different_doc)  # Different docs not comparable


class TestTabVersionState:
    """Tests for TabVersionState internal class."""

    def test_increment(self) -> None:
        """Test version increment."""
        state = TabVersionState(
            tab_id="tab-1",
            document_id="doc-abc",
            version_id=1,
            content_hash="hash1",
        )

        new_version = state.increment("hash2")

        assert new_version == 2
        assert state.version_id == 2
        assert state.content_hash == "hash2"
        assert state.is_dirty is True

    def test_reset(self) -> None:
        """Test version reset."""
        state = TabVersionState(
            tab_id="tab-1",
            document_id="doc-old",
            version_id=5,
            content_hash="oldhash",
            is_dirty=True,
        )

        state.reset("doc-new", "newhash")

        assert state.document_id == "doc-new"
        assert state.version_id == 1
        assert state.content_hash == "newhash"
        assert state.is_dirty is False

    def test_to_token(self) -> None:
        """Test converting state to token."""
        state = TabVersionState(
            tab_id="tab-1",
            document_id="doc-abc",
            version_id=3,
            content_hash="hash123",
        )

        token = state.to_token()

        assert isinstance(token, VersionToken)
        assert token.tab_id == "tab-1"
        assert token.version_id == 3


class TestVersionManager:
    """Tests for VersionManager class."""

    @pytest.fixture
    def manager(self) -> VersionManager:
        """Create a fresh VersionManager for each test."""
        return VersionManager()

    def test_register_tab(self, manager: VersionManager) -> None:
        """Test registering a new tab."""
        token = manager.register_tab("tab-1", "doc-abc", "hash123")

        assert token.tab_id == "tab-1"
        assert token.document_id == "doc-abc"
        assert token.version_id == 1
        assert token.content_hash == "hash123"

    def test_register_tab_updates_existing(self, manager: VersionManager) -> None:
        """Test that registering an existing tab resets it."""
        manager.register_tab("tab-1", "doc-old", "oldhash")
        token = manager.register_tab("tab-1", "doc-new", "newhash")

        assert token.document_id == "doc-new"
        assert token.version_id == 1  # Reset to 1

    def test_unregister_tab(self, manager: VersionManager) -> None:
        """Test removing a tab."""
        manager.register_tab("tab-1", "doc-abc", "hash")

        assert manager.unregister_tab("tab-1") is True
        assert manager.unregister_tab("tab-1") is False  # Already removed

    def test_get_current_token(self, manager: VersionManager) -> None:
        """Test getting current token."""
        manager.register_tab("tab-1", "doc-abc", "hash")

        token = manager.get_current_token("tab-1")
        assert token is not None
        assert token.tab_id == "tab-1"

        assert manager.get_current_token("nonexistent") is None

    def test_increment_version(self, manager: VersionManager) -> None:
        """Test incrementing version after edit."""
        manager.register_tab("tab-1", "doc-abc", "hash1")

        new_token = manager.increment_version("tab-1", "hash2")

        assert new_token.version_id == 2
        assert new_token.content_hash == "hash2"

    def test_increment_version_unregistered_tab(self, manager: VersionManager) -> None:
        """Test incrementing version for unregistered tab."""
        with pytest.raises(KeyError, match="not registered"):
            manager.increment_version("nonexistent", "hash")

    def test_validate_token_valid(self, manager: VersionManager) -> None:
        """Test validating a current token."""
        token = manager.register_tab("tab-1", "doc-abc", "hash123")

        assert manager.validate_token(token) is True
        assert manager.validate_token(token.to_string()) is True  # String form

    def test_validate_token_stale_version(self, manager: VersionManager) -> None:
        """Test validating a stale token."""
        token = manager.register_tab("tab-1", "doc-abc", "hash1")
        manager.increment_version("tab-1", "hash2")

        with pytest.raises(VersionMismatchError, match="has been edited"):
            manager.validate_token(token)

    def test_validate_token_hash_mismatch(self, manager: VersionManager) -> None:
        """Test validating token with wrong hash."""
        manager.register_tab("tab-1", "doc-abc", "correcthash")
        fake_token = VersionToken("tab-1", "doc-abc", 1, "wronghash")

        with pytest.raises(VersionMismatchError, match="hash mismatch"):
            manager.validate_token(fake_token)

    def test_validate_token_document_replaced(self, manager: VersionManager) -> None:
        """Test validating token after document replaced."""
        old_token = manager.register_tab("tab-1", "doc-old", "hash")
        manager.reset_on_reload("tab-1", "doc-new", "newhash")

        with pytest.raises(VersionMismatchError, match="has been replaced"):
            manager.validate_token(old_token)

    def test_validate_token_unregistered_tab(self, manager: VersionManager) -> None:
        """Test validating token for unregistered tab."""
        token = VersionToken("nonexistent", "doc", 1, "hash")

        with pytest.raises(KeyError, match="not registered"):
            manager.validate_token(token)

    def test_reset_on_reload(self, manager: VersionManager) -> None:
        """Test resetting version on document reload."""
        manager.register_tab("tab-1", "doc-abc", "hash1")
        manager.increment_version("tab-1", "hash2")
        manager.increment_version("tab-1", "hash3")

        token = manager.reset_on_reload("tab-1", "doc-abc", "freshhash")

        assert token.version_id == 1  # Reset to 1
        assert token.content_hash == "freshhash"

    def test_reset_on_reload_new_tab(self, manager: VersionManager) -> None:
        """Test reset_on_reload creates tab if not exists."""
        token = manager.reset_on_reload("new-tab", "doc-new", "hash")

        assert token.tab_id == "new-tab"
        assert token.version_id == 1

    def test_list_tabs(self, manager: VersionManager) -> None:
        """Test listing registered tabs."""
        manager.register_tab("tab-1", "doc-1", "h1")
        manager.register_tab("tab-2", "doc-2", "h2")
        manager.register_tab("tab-3", "doc-3", "h3")

        tabs = manager.list_tabs()

        assert set(tabs) == {"tab-1", "tab-2", "tab-3"}

    def test_clear(self, manager: VersionManager) -> None:
        """Test clearing all registrations."""
        manager.register_tab("tab-1", "doc-1", "h1")
        manager.register_tab("tab-2", "doc-2", "h2")

        manager.clear()

        assert manager.list_tabs() == []


class TestVersionMismatchError:
    """Tests for VersionMismatchError exception."""

    def test_create_error(self) -> None:
        """Test creating a version mismatch error."""
        your_token = VersionToken("tab-1", "doc", 1, "old")
        current_token = VersionToken("tab-1", "doc", 2, "new")

        error = VersionMismatchError(
            message="Document has changed",
            your_version=your_token,
            current_version=current_token,
            suggestion="Refresh and retry",
        )

        assert str(error) == "Document has changed"
        assert error.your_version == your_token
        assert error.current_version == current_token
        assert error.suggestion == "Refresh and retry"

    def test_to_dict(self) -> None:
        """Test error serialization."""
        your_token = VersionToken("tab-1", "doc", 1, "old")
        current_token = VersionToken("tab-1", "doc", 2, "new")

        error = VersionMismatchError(
            message="Document has changed",
            your_version=your_token,
            current_version=current_token,
            suggestion="Refresh",
        )

        result = error.to_dict()

        assert result["error"] == "version_mismatch"
        assert result["message"] == "Document has changed"
        assert result["your_version"]["version_id"] == 1
        assert result["current_version"]["version_id"] == 2
        assert result["suggestion"] == "Refresh"


class TestComputeContentHash:
    """Tests for hash computation."""

    def test_hash_empty_string(self) -> None:
        """Test hashing empty string."""
        result = compute_content_hash("")
        assert isinstance(result, str)
        assert len(result) == 40  # SHA-1 hex length

    def test_hash_deterministic(self) -> None:
        """Test that same content produces same hash."""
        content = "Hello, World!"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)
        assert hash1 == hash2

    def test_hash_different_content(self) -> None:
        """Test that different content produces different hash."""
        hash1 = compute_content_hash("Hello")
        hash2 = compute_content_hash("World")
        assert hash1 != hash2


class TestGlobalManager:
    """Tests for module-level singleton functions."""

    def test_get_version_manager_singleton(self) -> None:
        """Test that get_version_manager returns same instance."""
        reset_version_manager()

        manager1 = get_version_manager()
        manager2 = get_version_manager()

        assert manager1 is manager2

    def test_reset_version_manager(self) -> None:
        """Test resetting the global manager."""
        manager1 = get_version_manager()
        manager1.register_tab("tab-1", "doc", "hash")

        reset_version_manager()

        manager2 = get_version_manager()
        assert manager2.list_tabs() == []  # New manager is empty
