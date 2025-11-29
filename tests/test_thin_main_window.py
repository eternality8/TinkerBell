"""Tests for the ThinMainWindow presentation component."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest

from tinkerbell.ui.events import (
    ActiveTabChanged,
    DocumentOpened,
    DocumentSaved,
    EventBus,
    StatusMessage,
    WindowTitleChanged,
)


def _ensure_qapp() -> None:
    """Create a minimal QApplication when PySide6 is available."""
    try:
        from PySide6.QtWidgets import QApplication
    except Exception:  # pragma: no cover - PySide6 optional in tests
        return
    if QApplication.instance() is None:  # pragma: no cover
        QApplication([])


# Ensure QApp exists before importing ThinMainWindow
_ensure_qapp()

from tinkerbell.ui.presentation.main_window import ThinMainWindow


class MockCoordinator:
    """Mock coordinator for testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._active_tab_id: str | None = None

    @property
    def active_tab_id(self) -> str | None:
        return self._active_tab_id

    def new_document(self) -> str:
        self.calls.append(("new_document", ()))
        return "tab-1"

    def open_document(self, path: Any = None) -> str | None:
        self.calls.append(("open_document", (path,)))
        return "tab-1"

    def save_document(self, path: Any = None) -> Any:
        self.calls.append(("save_document", (path,)))
        return None

    def save_document_as(self) -> Any:
        self.calls.append(("save_document_as", ()))
        return None

    def close_document(self, tab_id: str | None = None) -> bool:
        self.calls.append(("close_document", (tab_id,)))
        return True

    def revert_document(self) -> bool:
        self.calls.append(("revert_document", ()))
        return True

    def import_document(self, path: Any = None) -> str | None:
        self.calls.append(("import_document", (path,)))
        return None

    def refresh_snapshot(self, **kwargs: Any) -> Any:
        self.calls.append(("refresh_snapshot", ()))
        return None

    def accept_review(self) -> Any:
        self.calls.append(("accept_review", ()))
        return None

    def reject_review(self) -> Any:
        self.calls.append(("reject_review", ()))
        return None


def _make_window(
    *,
    event_bus: EventBus | None = None,
    coordinator: Any = None,
) -> ThinMainWindow:
    """Create a ThinMainWindow for testing."""
    if event_bus is None:
        event_bus = EventBus()
    if coordinator is None:
        coordinator = MockCoordinator()
    return ThinMainWindow(
        event_bus=event_bus,
        coordinator=coordinator,
        skip_widgets=True,
    )


def test_thin_main_window_initializes_without_widgets() -> None:
    """Test that ThinMainWindow can be created without widgets."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    assert window.editor is None
    assert window.chat_panel is None
    assert window.status_bar is None


def test_thin_main_window_subscribes_to_events() -> None:
    """Test that ThinMainWindow subscribes to relevant events."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    # Check that handlers are registered
    assert bus.handler_count(WindowTitleChanged) > 0
    assert bus.handler_count(DocumentOpened) > 0
    assert bus.handler_count(DocumentSaved) > 0
    assert bus.handler_count(ActiveTabChanged) > 0
    assert bus.handler_count(StatusMessage) > 0


def test_thin_main_window_unsubscribes_on_dispose() -> None:
    """Test that dispose() unsubscribes from events."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    # Store initial counts
    initial_count = bus.handler_count()

    # Dispose
    window.dispose()

    # Handlers should be removed
    assert bus.handler_count() < initial_count


def test_thin_main_window_handles_window_title_changed() -> None:
    """Test that WindowTitleChanged updates the title."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    # Track setWindowTitle calls
    title_set: list[str] = []
    window.setWindowTitle = lambda t: title_set.append(t)  # type: ignore[method-assign]

    # Publish event
    bus.publish(WindowTitleChanged(title="New Title"))

    assert "New Title" in title_set


def test_thin_main_window_tracks_last_status_message() -> None:
    """Test that StatusMessage is tracked."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    # Publish event
    bus.publish(StatusMessage(message="Test status"))

    assert window.last_status_message == "Test status"


def test_thin_main_window_delegates_new_document() -> None:
    """Test that new document action delegates to coordinator."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    window._handle_new_tab()

    assert ("new_document", ()) in coordinator.calls


def test_thin_main_window_delegates_open_document() -> None:
    """Test that open document action delegates to coordinator."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    window._handle_open()

    assert ("open_document", (None,)) in coordinator.calls


def test_thin_main_window_delegates_save_document() -> None:
    """Test that save document action delegates to coordinator."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    window._handle_save()

    assert ("save_document", (None,)) in coordinator.calls


def test_thin_main_window_delegates_save_as() -> None:
    """Test that save as action delegates to coordinator."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    window._handle_save_as()

    assert ("save_document_as", ()) in coordinator.calls


def test_thin_main_window_delegates_close_tab() -> None:
    """Test that close tab action delegates to coordinator."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    window._handle_close_tab()

    assert ("close_document", (None,)) in coordinator.calls


def test_thin_main_window_delegates_revert() -> None:
    """Test that revert action delegates to coordinator."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    window._handle_revert()

    assert ("revert_document", ()) in coordinator.calls


def test_thin_main_window_delegates_import() -> None:
    """Test that import action delegates to coordinator."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    window._handle_import()

    assert ("import_document", (None,)) in coordinator.calls


def test_thin_main_window_delegates_snapshot() -> None:
    """Test that snapshot action delegates to coordinator."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    window._handle_snapshot()

    assert ("refresh_snapshot", ()) in coordinator.calls


def test_thin_main_window_delegates_accept() -> None:
    """Test that accept action delegates to coordinator."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    window._handle_accept()

    assert ("accept_review", ()) in coordinator.calls


def test_thin_main_window_delegates_reject() -> None:
    """Test that reject action delegates to coordinator."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    window._handle_reject()

    assert ("reject_review", ()) in coordinator.calls


def test_thin_main_window_updates_title_on_document_events() -> None:
    """Test that document events trigger title updates."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    title_updates: list[str] = []
    window.setWindowTitle = lambda t: title_updates.append(t)  # type: ignore[method-assign]

    # Clear any initial calls
    title_updates.clear()

    # Publish document events
    bus.publish(DocumentOpened(tab_id="t1", document_id="d1", path="/test.txt"))
    bus.publish(DocumentSaved(tab_id="t1", document_id="d1", path="/test.txt"))
    bus.publish(ActiveTabChanged(tab_id="t1", document_id="d1"))

    # Each event should trigger a title update
    assert len(title_updates) == 3


def test_thin_main_window_actions_empty_without_chrome() -> None:
    """Test that actions dict is empty when chrome not assembled."""
    bus = EventBus()
    coordinator = MockCoordinator()

    window = ThinMainWindow(
        event_bus=bus,
        coordinator=coordinator,
        skip_widgets=True,
    )

    assert window.actions == {}
