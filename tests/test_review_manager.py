"""Tests for ReviewManager domain manager."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from tinkerbell.ui.domain.review_manager import ReviewManager
from tinkerbell.ui.events import (
    Event,
    EventBus,
    ReviewAccepted,
    ReviewRejected,
    ReviewStateChanged,
)
from tinkerbell.ui.models.review_models import PendingEdit, PendingReview, ReviewSession


# =============================================================================
# Fixtures
# =============================================================================


class MockDocumentState:
    """Mock DocumentState for testing."""

    def __init__(self, document_id: str = "doc-1") -> None:
        self.document_id = document_id
        self.text = "test content"


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def review_manager(event_bus: EventBus) -> ReviewManager:
    return ReviewManager(event_bus=event_bus)


@pytest.fixture
def mock_document() -> MockDocumentState:
    return MockDocumentState()


# =============================================================================
# Initialization Tests
# =============================================================================


class TestReviewManagerInit:
    """Tests for ReviewManager initialization."""

    def test_initial_state(self, review_manager: ReviewManager) -> None:
        """Manager starts with no pending review."""
        assert review_manager.pending_review is None
        assert review_manager.has_pending_review() is False
        assert review_manager.is_ready_for_review() is False


# =============================================================================
# Begin Review Tests
# =============================================================================


class TestReviewManagerBeginReview:
    """Tests for ReviewManager.begin_review()."""

    def test_creates_pending_review(self, review_manager: ReviewManager) -> None:
        """begin_review creates a PendingReview."""
        review = review_manager.begin_review("turn-1", "Test prompt")

        assert review is not None
        assert review.turn_id == "turn-1"
        assert review.prompt == "Test prompt"
        assert review_manager.has_pending_review() is True

    def test_emits_state_changed_event(
        self, review_manager: ReviewManager, event_bus: EventBus
    ) -> None:
        """begin_review emits ReviewStateChanged."""
        events: list[ReviewStateChanged] = []
        event_bus.subscribe(ReviewStateChanged, events.append)

        review_manager.begin_review("turn-1", "Test")

        assert len(events) == 1
        assert events[0].turn_id == "turn-1"
        assert events[0].ready is False

    def test_drops_existing_review(
        self, review_manager: ReviewManager
    ) -> None:
        """begin_review drops existing review."""
        review_manager.begin_review("turn-1", "First")
        review_manager.begin_review("turn-2", "Second")

        assert review_manager.pending_review.turn_id == "turn-2"

    def test_returns_pending_review(self, review_manager: ReviewManager) -> None:
        """begin_review returns the created review."""
        review = review_manager.begin_review("turn-1", "Test")
        assert review is review_manager.pending_review


# =============================================================================
# Record Edit Tests
# =============================================================================


class TestReviewManagerRecordEdit:
    """Tests for ReviewManager.record_edit()."""

    def test_creates_session_and_records_edit(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """record_edit creates session and adds edit."""
        review_manager.begin_review("turn-1", "Test")

        edit = PendingEdit(
            edit_id="e1",
            tab_id="tab-1",
            action="replace",
            range=(0, 10),
            diff="- old\n+ new",
        )
        review_manager.record_edit("tab-1", mock_document, edit)  # type: ignore[arg-type]

        session = review_manager.get_session("tab-1")
        assert session is not None
        assert len(session.edits) == 1
        assert session.edits[0].edit_id == "e1"

    def test_increments_total_edit_count(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """record_edit increments total edit count."""
        review_manager.begin_review("turn-1", "Test")

        edit1 = PendingEdit("e1", "tab-1", "replace", (0, 10), "diff")
        edit2 = PendingEdit("e2", "tab-1", "replace", (20, 30), "diff")
        review_manager.record_edit("tab-1", mock_document, edit1)  # type: ignore[arg-type]
        review_manager.record_edit("tab-1", mock_document, edit2)  # type: ignore[arg-type]

        assert review_manager.pending_review.total_edit_count == 2

    def test_no_op_without_pending_review(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """record_edit is no-op without pending review."""
        edit = PendingEdit("e1", "tab-1", "replace", (0, 10), "diff")
        review_manager.record_edit("tab-1", mock_document, edit)  # type: ignore[arg-type]
        # Should not raise


# =============================================================================
# Ensure Session Tests
# =============================================================================


class TestReviewManagerEnsureSession:
    """Tests for ReviewManager.ensure_session()."""

    def test_creates_new_session(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """ensure_session creates new session."""
        review_manager.begin_review("turn-1", "Test")

        session = review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]

        assert session is not None
        assert session.tab_id == "tab-1"
        assert session.document_id == "doc-1"

    def test_returns_existing_session(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """ensure_session returns existing session."""
        review_manager.begin_review("turn-1", "Test")

        session1 = review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]
        session2 = review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]

        assert session1 is session2

    def test_makes_deep_copy_of_snapshot(
        self, review_manager: ReviewManager
    ) -> None:
        """ensure_session makes deep copy of snapshot."""
        review_manager.begin_review("turn-1", "Test")

        original = MockDocumentState(document_id="doc-1")
        session = review_manager.ensure_session("tab-1", original)  # type: ignore[arg-type]

        # The snapshot should be a copy, not the same object
        assert session.snapshot is not original

    def test_returns_none_without_pending_review(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """ensure_session returns None without pending review."""
        session = review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]
        assert session is None


# =============================================================================
# Finalize Tests
# =============================================================================


class TestReviewManagerFinalize:
    """Tests for ReviewManager.finalize()."""

    def test_marks_ready_when_successful_with_edits(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """finalize marks ready when turn succeeded with edits."""
        review_manager.begin_review("turn-1", "Test")
        edit = PendingEdit("e1", "tab-1", "replace", (0, 10), "diff")
        review_manager.record_edit("tab-1", mock_document, edit)  # type: ignore[arg-type]

        review_manager.finalize(success=True)

        assert review_manager.pending_review.ready is True
        assert review_manager.is_ready_for_review() is True

    def test_drops_when_failed(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """finalize drops review when turn failed."""
        review_manager.begin_review("turn-1", "Test")
        edit = PendingEdit("e1", "tab-1", "replace", (0, 10), "diff")
        review_manager.record_edit("tab-1", mock_document, edit)  # type: ignore[arg-type]

        review_manager.finalize(success=False)

        assert review_manager.pending_review is None

    def test_drops_when_no_edits(
        self, review_manager: ReviewManager
    ) -> None:
        """finalize drops review when no edits."""
        review_manager.begin_review("turn-1", "Test")

        review_manager.finalize(success=True)

        assert review_manager.pending_review is None

    def test_emits_state_changed(
        self, review_manager: ReviewManager, event_bus: EventBus, mock_document: MockDocumentState
    ) -> None:
        """finalize emits ReviewStateChanged."""
        review_manager.begin_review("turn-1", "Test")
        edit = PendingEdit("e1", "tab-1", "replace", (0, 10), "diff")
        review_manager.record_edit("tab-1", mock_document, edit)  # type: ignore[arg-type]

        events: list[ReviewStateChanged] = []
        event_bus.subscribe(ReviewStateChanged, events.append)

        review_manager.finalize(success=True)

        assert len(events) == 1
        assert events[0].ready is True
        assert events[0].edit_count == 1


# =============================================================================
# Accept Tests
# =============================================================================


class TestReviewManagerAccept:
    """Tests for ReviewManager.accept()."""

    def test_returns_affected_tabs(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """accept returns list of affected tab IDs."""
        review_manager.begin_review("turn-1", "Test")
        review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]
        review_manager.ensure_session("tab-2", mock_document)  # type: ignore[arg-type]

        tabs = review_manager.accept()

        assert set(tabs) == {"tab-1", "tab-2"}

    def test_clears_pending_review(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """accept clears the pending review."""
        review_manager.begin_review("turn-1", "Test")
        review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]

        review_manager.accept()

        assert review_manager.pending_review is None

    def test_emits_review_accepted_event(
        self, review_manager: ReviewManager, event_bus: EventBus, mock_document: MockDocumentState
    ) -> None:
        """accept emits ReviewAccepted event."""
        review_manager.begin_review("turn-1", "Test")
        review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]

        events: list[ReviewAccepted] = []
        event_bus.subscribe(ReviewAccepted, events.append)

        review_manager.accept()

        assert len(events) == 1
        assert events[0].turn_id == "turn-1"
        assert "tab-1" in events[0].tabs

    def test_emits_state_changed_after_accept(
        self, review_manager: ReviewManager, event_bus: EventBus, mock_document: MockDocumentState
    ) -> None:
        """accept emits ReviewStateChanged after clearing."""
        review_manager.begin_review("turn-1", "Test")
        review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]

        events: list[ReviewStateChanged] = []
        event_bus.subscribe(ReviewStateChanged, events.append)

        review_manager.accept()

        assert len(events) == 1
        assert events[0].ready is False
        assert events[0].edit_count == 0

    def test_returns_empty_list_without_pending(
        self, review_manager: ReviewManager
    ) -> None:
        """accept returns empty list without pending review."""
        tabs = review_manager.accept()
        assert tabs == []


# =============================================================================
# Reject Tests
# =============================================================================


class TestReviewManagerReject:
    """Tests for ReviewManager.reject()."""

    def test_returns_snapshots_to_restore(
        self, review_manager: ReviewManager
    ) -> None:
        """reject returns dict of snapshots."""
        review_manager.begin_review("turn-1", "Test")
        doc1 = MockDocumentState(document_id="doc-1")
        doc2 = MockDocumentState(document_id="doc-2")
        review_manager.ensure_session("tab-1", doc1)  # type: ignore[arg-type]
        review_manager.ensure_session("tab-2", doc2)  # type: ignore[arg-type]

        snapshots = review_manager.reject()

        assert "tab-1" in snapshots
        assert "tab-2" in snapshots

    def test_clears_pending_review(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """reject clears the pending review."""
        review_manager.begin_review("turn-1", "Test")
        review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]

        review_manager.reject()

        assert review_manager.pending_review is None

    def test_emits_review_rejected_event(
        self, review_manager: ReviewManager, event_bus: EventBus, mock_document: MockDocumentState
    ) -> None:
        """reject emits ReviewRejected event."""
        review_manager.begin_review("turn-1", "Test")
        review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]

        events: list[ReviewRejected] = []
        event_bus.subscribe(ReviewRejected, events.append)

        review_manager.reject()

        assert len(events) == 1
        assert events[0].turn_id == "turn-1"
        assert "tab-1" in events[0].tabs

    def test_returns_empty_dict_without_pending(
        self, review_manager: ReviewManager
    ) -> None:
        """reject returns empty dict without pending review."""
        snapshots = review_manager.reject()
        assert snapshots == {}


# =============================================================================
# Drop Tests
# =============================================================================


class TestReviewManagerDrop:
    """Tests for ReviewManager.drop()."""

    def test_clears_pending_review(
        self, review_manager: ReviewManager
    ) -> None:
        """drop clears the pending review."""
        review_manager.begin_review("turn-1", "Test")

        review_manager.drop("test-reason")

        assert review_manager.pending_review is None

    def test_emits_state_changed(
        self, review_manager: ReviewManager, event_bus: EventBus
    ) -> None:
        """drop emits ReviewStateChanged."""
        review_manager.begin_review("turn-1", "Test")

        events: list[ReviewStateChanged] = []
        event_bus.subscribe(ReviewStateChanged, events.append)

        review_manager.drop("test-reason")

        assert len(events) == 1
        assert events[0].ready is False

    def test_no_op_without_pending(
        self, review_manager: ReviewManager, event_bus: EventBus
    ) -> None:
        """drop is no-op without pending review."""
        events: list[ReviewStateChanged] = []
        event_bus.subscribe(ReviewStateChanged, events.append)

        review_manager.drop("test")

        assert len(events) == 0


# =============================================================================
# Session Helper Tests
# =============================================================================


class TestReviewManagerSessionHelpers:
    """Tests for session helper methods."""

    def test_get_session(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """get_session returns session for tab."""
        review_manager.begin_review("turn-1", "Test")
        review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]

        session = review_manager.get_session("tab-1")
        assert session is not None
        assert session.tab_id == "tab-1"

    def test_get_session_returns_none(
        self, review_manager: ReviewManager
    ) -> None:
        """get_session returns None for missing tab."""
        review_manager.begin_review("turn-1", "Test")
        assert review_manager.get_session("nonexistent") is None

    def test_get_snapshot(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """get_snapshot returns document snapshot."""
        review_manager.begin_review("turn-1", "Test")
        review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]

        snapshot = review_manager.get_snapshot("tab-1")
        assert snapshot is not None

    def test_get_snapshot_returns_none(
        self, review_manager: ReviewManager
    ) -> None:
        """get_snapshot returns None for missing tab."""
        review_manager.begin_review("turn-1", "Test")
        assert review_manager.get_snapshot("nonexistent") is None

    def test_affected_tab_ids(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """affected_tab_ids returns all tab IDs."""
        review_manager.begin_review("turn-1", "Test")
        review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]
        review_manager.ensure_session("tab-2", mock_document)  # type: ignore[arg-type]

        ids = review_manager.affected_tab_ids()
        assert set(ids) == {"tab-1", "tab-2"}

    def test_affected_tab_ids_empty(
        self, review_manager: ReviewManager
    ) -> None:
        """affected_tab_ids returns empty tuple without review."""
        assert review_manager.affected_tab_ids() == ()

    def test_update_merged_spans(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """update_merged_spans sets spans on session."""
        review_manager.begin_review("turn-1", "Test")
        review_manager.ensure_session("tab-1", mock_document)  # type: ignore[arg-type]

        spans = ((0, 10), (20, 30))
        review_manager.update_merged_spans("tab-1", spans)

        session = review_manager.get_session("tab-1")
        assert session.merged_spans == spans


# =============================================================================
# Format Summary Tests
# =============================================================================


class TestReviewManagerFormatSummary:
    """Tests for format_summary()."""

    def test_format_empty(self, review_manager: ReviewManager) -> None:
        """Format with no pending review."""
        assert review_manager.format_summary() == "0 edits across 0 tabs"

    def test_format_single_edit(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """Format with single edit."""
        review_manager.begin_review("turn-1", "Test")
        edit = PendingEdit("e1", "tab-1", "replace", (0, 10), "diff")
        review_manager.record_edit("tab-1", mock_document, edit)  # type: ignore[arg-type]

        assert review_manager.format_summary() == "1 edit across 1 tab"

    def test_format_multiple_edits(
        self, review_manager: ReviewManager, mock_document: MockDocumentState
    ) -> None:
        """Format with multiple edits."""
        review_manager.begin_review("turn-1", "Test")
        review_manager.record_edit("tab-1", mock_document, PendingEdit("e1", "tab-1", "r", (0, 10), "d"))  # type: ignore[arg-type]
        review_manager.record_edit("tab-1", mock_document, PendingEdit("e2", "tab-1", "r", (20, 30), "d"))  # type: ignore[arg-type]
        review_manager.record_edit("tab-2", mock_document, PendingEdit("e3", "tab-2", "r", (0, 5), "d"))  # type: ignore[arg-type]

        assert review_manager.format_summary() == "3 edits across 2 tabs"


# =============================================================================
# Module Export Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_exported_from_domain(self) -> None:
        """ReviewManager is exported from domain package."""
        from tinkerbell.ui.domain import ReviewManager as RM

        assert RM is ReviewManager
