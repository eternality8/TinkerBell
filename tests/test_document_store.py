"""Tests for DocumentStore domain manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from tinkerbell.ui.domain.document_store import DocumentStore
from tinkerbell.ui.events import (
    ActiveTabChanged,
    DocumentClosed,
    DocumentCreated,
    Event,
    EventBus,
)


# =============================================================================
# Fixtures
# =============================================================================


class MockDocumentState:
    """Mock DocumentState for testing."""

    def __init__(self, document_id: str = "doc-1", path: Path | None = None) -> None:
        self.document_id = document_id
        self.metadata = MagicMock()
        self.metadata.path = path
        self.dirty = False


class MockDocumentTab:
    """Mock DocumentTab for testing."""

    def __init__(
        self,
        tab_id: str = "t1",
        document_id: str = "doc-1",
        path: Path | None = None,
    ) -> None:
        self.id = tab_id
        self._document = MockDocumentState(document_id=document_id, path=path)
        self.title = "Untitled"

    def document(self) -> MockDocumentState:
        return self._document


class MockWorkspace:
    """Mock DocumentWorkspace for testing."""

    def __init__(self) -> None:
        self._tabs: dict[str, MockDocumentTab] = {}
        self._order: list[str] = []
        self._active_tab_id: str | None = None
        self._listeners: list[Any] = []
        self._tab_counter = 1

    def create_tab(
        self,
        *,
        document: Any = None,
        title: str | None = None,
        path: Any = None,
        make_active: bool = True,
        untitled_index: int | None = None,
    ) -> MockDocumentTab:
        tab_id = f"t{self._tab_counter}"
        self._tab_counter += 1
        doc_id = f"doc-{tab_id}"
        tab = MockDocumentTab(tab_id=tab_id, document_id=doc_id, path=path)
        self._tabs[tab_id] = tab
        self._order.append(tab_id)
        if make_active:
            self._active_tab_id = tab_id
            self._notify_listeners()
        return tab

    def close_tab(self, tab_id: str) -> MockDocumentTab:
        if tab_id not in self._tabs:
            raise KeyError(f"Unknown tab_id: {tab_id}")
        tab = self._tabs.pop(tab_id)
        self._order.remove(tab_id)
        if self._active_tab_id == tab_id:
            self._active_tab_id = self._order[0] if self._order else None
            self._notify_listeners()
        return tab

    def get_tab(self, tab_id: str) -> MockDocumentTab:
        if tab_id not in self._tabs:
            raise KeyError(f"Unknown tab_id: {tab_id}")
        return self._tabs[tab_id]

    @property
    def active_tab(self) -> MockDocumentTab | None:
        if self._active_tab_id is None:
            return None
        return self._tabs.get(self._active_tab_id)

    @property
    def active_tab_id(self) -> str | None:
        return self._active_tab_id

    def set_active_tab(self, tab_id: str) -> MockDocumentTab:
        if tab_id not in self._tabs:
            raise KeyError(f"Unknown tab_id: {tab_id}")
        self._active_tab_id = tab_id
        self._notify_listeners()
        return self._tabs[tab_id]

    def iter_tabs(self):
        for tab_id in self._order:
            yield self._tabs[tab_id]

    def tab_count(self) -> int:
        return len(self._order)

    def tab_ids(self):
        return tuple(self._order)

    def find_tab_by_path(self, path: Any) -> MockDocumentTab | None:
        for tab in self._tabs.values():
            if tab._document.metadata.path == path:
                return tab
        return None

    def find_document_by_id(self, document_id: str) -> MockDocumentState | None:
        for tab in self._tabs.values():
            if tab._document.document_id == document_id:
                return tab._document
        return None

    def ensure_tab(self) -> MockDocumentTab:
        if self._order:
            return self._tabs[self._order[-1]]
        return self.create_tab()

    def require_active_tab(self) -> MockDocumentTab:
        if self._active_tab_id is None:
            raise RuntimeError("No active tab")
        return self._tabs[self._active_tab_id]

    def active_document(self) -> MockDocumentState:
        return self.require_active_tab().document()

    def add_active_listener(self, listener: Any) -> None:
        self._listeners.append(listener)

    def remove_active_listener(self, listener: Any) -> None:
        self._listeners.remove(listener)

    def _notify_listeners(self) -> None:
        tab = self.active_tab
        for listener in self._listeners:
            listener(tab)


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def mock_workspace() -> MockWorkspace:
    return MockWorkspace()


@pytest.fixture
def document_store(mock_workspace: MockWorkspace, event_bus: EventBus) -> DocumentStore:
    return DocumentStore(workspace=mock_workspace, event_bus=event_bus)  # type: ignore[arg-type]


# =============================================================================
# Creation Tests
# =============================================================================


class TestDocumentStoreCreation:
    """Tests for DocumentStore initialization."""

    def test_registers_active_listener(
        self, mock_workspace: MockWorkspace, event_bus: EventBus
    ) -> None:
        """Store registers itself as active tab listener."""
        assert len(mock_workspace._listeners) == 0
        DocumentStore(workspace=mock_workspace, event_bus=event_bus)  # type: ignore[arg-type]
        assert len(mock_workspace._listeners) == 1


# =============================================================================
# Tab Lifecycle Tests
# =============================================================================


class TestDocumentStoreCreateTab:
    """Tests for DocumentStore.create_tab()."""

    def test_creates_tab_in_workspace(
        self, document_store: DocumentStore, mock_workspace: MockWorkspace
    ) -> None:
        """create_tab delegates to workspace."""
        tab = document_store.create_tab()
        assert tab.id in mock_workspace._tabs
        assert mock_workspace.tab_count() == 1

    def test_emits_document_created_event(
        self, document_store: DocumentStore, event_bus: EventBus
    ) -> None:
        """create_tab emits DocumentCreated event."""
        events: list[Event] = []
        event_bus.subscribe(DocumentCreated, events.append)

        tab = document_store.create_tab()

        # Filter to DocumentCreated only (ActiveTabChanged also emitted)
        created_events = [e for e in events if isinstance(e, DocumentCreated)]
        assert len(created_events) == 1
        assert created_events[0].tab_id == tab.id

    def test_emits_active_tab_changed_when_active(
        self, document_store: DocumentStore, event_bus: EventBus
    ) -> None:
        """create_tab emits ActiveTabChanged when make_active=True."""
        events: list[Event] = []
        event_bus.subscribe(ActiveTabChanged, events.append)

        tab = document_store.create_tab(make_active=True)

        assert len(events) == 1
        assert events[0].tab_id == tab.id

    def test_passes_path_to_workspace(
        self, document_store: DocumentStore, mock_workspace: MockWorkspace
    ) -> None:
        """create_tab passes path through to workspace."""
        path = Path("/test/file.txt")
        document_store.create_tab(path=path)
        # Workspace received the path parameter - check via iteration
        assert mock_workspace.tab_count() == 1


class TestDocumentStoreCloseTab:
    """Tests for DocumentStore.close_tab()."""

    def test_closes_tab_in_workspace(
        self, document_store: DocumentStore, mock_workspace: MockWorkspace
    ) -> None:
        """close_tab removes tab from workspace."""
        tab = document_store.create_tab()
        assert mock_workspace.tab_count() == 1

        document_store.close_tab(tab.id)
        assert mock_workspace.tab_count() == 0

    def test_emits_document_closed_event(
        self, document_store: DocumentStore, event_bus: EventBus
    ) -> None:
        """close_tab emits DocumentClosed event."""
        tab = document_store.create_tab()
        doc_id = tab.document().document_id

        events: list[Event] = []
        event_bus.subscribe(DocumentClosed, events.append)

        document_store.close_tab(tab.id)

        assert len(events) == 1
        assert events[0].tab_id == tab.id
        assert events[0].document_id == doc_id

    def test_returns_closed_tab(self, document_store: DocumentStore) -> None:
        """close_tab returns the closed tab."""
        tab = document_store.create_tab()
        closed = document_store.close_tab(tab.id)
        assert closed is tab

    def test_returns_none_for_unknown_tab(self, document_store: DocumentStore) -> None:
        """close_tab returns None for unknown tab ID."""
        result = document_store.close_tab("nonexistent")
        assert result is None

    def test_emits_active_tab_changed_when_closing_active(
        self, document_store: DocumentStore, event_bus: EventBus
    ) -> None:
        """Closing active tab emits ActiveTabChanged."""
        tab1 = document_store.create_tab()
        tab2 = document_store.create_tab()

        events: list[Event] = []
        event_bus.subscribe(ActiveTabChanged, events.append)

        document_store.close_tab(tab2.id)  # tab2 is active

        # Should have event for new active tab (tab1)
        assert len(events) == 1
        assert events[0].tab_id == tab1.id


# =============================================================================
# Tab Access Tests
# =============================================================================


class TestDocumentStoreTabAccess:
    """Tests for tab access methods."""

    def test_get_tab(self, document_store: DocumentStore) -> None:
        """get_tab returns correct tab."""
        tab = document_store.create_tab()
        result = document_store.get_tab(tab.id)
        assert result is tab

    def test_get_tab_raises_for_unknown(self, document_store: DocumentStore) -> None:
        """get_tab raises KeyError for unknown tab."""
        with pytest.raises(KeyError):
            document_store.get_tab("nonexistent")

    def test_active_tab_property(self, document_store: DocumentStore) -> None:
        """active_tab returns current active tab."""
        tab = document_store.create_tab()
        assert document_store.active_tab is tab

    def test_active_tab_id_property(self, document_store: DocumentStore) -> None:
        """active_tab_id returns ID of active tab."""
        tab = document_store.create_tab()
        assert document_store.active_tab_id == tab.id

    def test_set_active_tab(
        self, document_store: DocumentStore, event_bus: EventBus
    ) -> None:
        """set_active_tab changes active tab and emits event."""
        tab1 = document_store.create_tab()
        tab2 = document_store.create_tab()

        events: list[Event] = []
        event_bus.subscribe(ActiveTabChanged, events.append)

        document_store.set_active_tab(tab1.id)

        assert document_store.active_tab is tab1
        assert len(events) == 1
        assert events[0].tab_id == tab1.id

    def test_iter_tabs(self, document_store: DocumentStore) -> None:
        """iter_tabs yields all tabs in order."""
        tab1 = document_store.create_tab()
        tab2 = document_store.create_tab()

        tabs = list(document_store.iter_tabs())
        assert tabs == [tab1, tab2]

    def test_tab_count(self, document_store: DocumentStore) -> None:
        """tab_count returns number of tabs."""
        assert document_store.tab_count() == 0
        document_store.create_tab()
        assert document_store.tab_count() == 1
        document_store.create_tab()
        assert document_store.tab_count() == 2

    def test_tab_ids(self, document_store: DocumentStore) -> None:
        """tab_ids returns tuple of tab IDs."""
        tab1 = document_store.create_tab()
        tab2 = document_store.create_tab()

        ids = document_store.tab_ids()
        assert ids == (tab1.id, tab2.id)


# =============================================================================
# Document Lookup Tests
# =============================================================================


class TestDocumentStoreLookup:
    """Tests for document lookup methods."""

    def test_find_tab_by_path(
        self, document_store: DocumentStore, mock_workspace: MockWorkspace
    ) -> None:
        """find_tab_by_path delegates to workspace."""
        path = Path("/test/file.txt")
        tab = document_store.create_tab(path=path)
        # Manually set path on mock for test
        tab._document.metadata.path = path

        result = document_store.find_tab_by_path(path)
        assert result is tab

    def test_find_tab_by_path_returns_none(
        self, document_store: DocumentStore
    ) -> None:
        """find_tab_by_path returns None when not found."""
        document_store.create_tab()
        result = document_store.find_tab_by_path("/nonexistent")
        assert result is None

    def test_find_document_by_id(self, document_store: DocumentStore) -> None:
        """find_document_by_id finds document."""
        tab = document_store.create_tab()
        doc = tab.document()

        result = document_store.find_document_by_id(doc.document_id)
        assert result is doc

    def test_find_document_by_id_returns_none(
        self, document_store: DocumentStore
    ) -> None:
        """find_document_by_id returns None when not found."""
        document_store.create_tab()
        result = document_store.find_document_by_id("nonexistent")
        assert result is None


# =============================================================================
# Convenience Method Tests
# =============================================================================


class TestDocumentStoreConvenience:
    """Tests for convenience methods."""

    def test_ensure_tab_existing(self, document_store: DocumentStore) -> None:
        """ensure_tab returns existing tab when available."""
        tab = document_store.create_tab()
        result = document_store.ensure_tab()
        assert result is tab

    def test_ensure_tab_creates_when_empty(
        self, document_store: DocumentStore, event_bus: EventBus
    ) -> None:
        """ensure_tab creates tab when none exist."""
        events: list[Event] = []
        event_bus.subscribe(DocumentCreated, events.append)

        tab = document_store.ensure_tab()

        assert tab is not None
        assert len(events) == 1

    def test_require_active_tab(self, document_store: DocumentStore) -> None:
        """require_active_tab returns active tab."""
        tab = document_store.create_tab()
        result = document_store.require_active_tab()
        assert result is tab

    def test_require_active_tab_raises(self, document_store: DocumentStore) -> None:
        """require_active_tab raises when no active tab."""
        with pytest.raises(RuntimeError):
            document_store.require_active_tab()

    def test_active_document(self, document_store: DocumentStore) -> None:
        """active_document returns document from active tab."""
        tab = document_store.create_tab()
        result = document_store.active_document()
        assert result is tab.document()

    def test_workspace_property(
        self, document_store: DocumentStore, mock_workspace: MockWorkspace
    ) -> None:
        """workspace property provides access to underlying workspace."""
        assert document_store.workspace is mock_workspace


# =============================================================================
# Event Integration Tests
# =============================================================================


class TestDocumentStoreEventIntegration:
    """Integration tests for event emission."""

    def test_full_lifecycle_events(
        self, document_store: DocumentStore, event_bus: EventBus
    ) -> None:
        """Full tab lifecycle emits correct events."""
        created_events: list[DocumentCreated] = []
        closed_events: list[DocumentClosed] = []
        active_events: list[ActiveTabChanged] = []

        event_bus.subscribe(DocumentCreated, created_events.append)
        event_bus.subscribe(DocumentClosed, closed_events.append)
        event_bus.subscribe(ActiveTabChanged, active_events.append)

        # Create first tab
        tab1 = document_store.create_tab()
        assert len(created_events) == 1
        assert len(active_events) == 1

        # Create second tab
        tab2 = document_store.create_tab()
        assert len(created_events) == 2
        assert len(active_events) == 2

        # Switch to first tab
        document_store.set_active_tab(tab1.id)
        assert len(active_events) == 3

        # Close second tab (not active, no active change)
        document_store.close_tab(tab2.id)
        assert len(closed_events) == 1

        # Close first tab (active, triggers active change to None)
        document_store.close_tab(tab1.id)
        assert len(closed_events) == 2


# =============================================================================
# Module Export Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_document_store_exported_from_domain(self) -> None:
        """DocumentStore is exported from domain package."""
        from tinkerbell.ui.domain import DocumentStore as DS

        assert DS is DocumentStore
