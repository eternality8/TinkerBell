"""Tests for review_models.py and ai_models.py."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tinkerbell.ui.models.ai_models import AITurnState, AITurnStatus
from tinkerbell.ui.models.review_models import PendingEdit, PendingReview, ReviewSession


# =============================================================================
# AITurnStatus Tests
# =============================================================================


class TestAITurnStatus:
    """Tests for AITurnStatus enum."""

    def test_status_values(self) -> None:
        """All expected status values exist."""
        assert AITurnStatus.PENDING.value == "pending"
        assert AITurnStatus.RUNNING.value == "running"
        assert AITurnStatus.COMPLETED.value == "completed"
        assert AITurnStatus.FAILED.value == "failed"
        assert AITurnStatus.CANCELED.value == "canceled"

    def test_status_count(self) -> None:
        """Exactly five status values."""
        assert len(AITurnStatus) == 5


# =============================================================================
# AITurnState Tests
# =============================================================================


class TestAITurnStateCreation:
    """Tests for AITurnState dataclass creation."""

    def test_minimal_creation(self) -> None:
        """Create with only required fields."""
        state = AITurnState(turn_id="turn-123", prompt="Hello")
        assert state.turn_id == "turn-123"
        assert state.prompt == "Hello"
        assert state.status == AITurnStatus.PENDING
        assert state.edit_count == 0
        assert state.error is None
        assert state.completed_at is None
        assert state.metadata == {}

    def test_full_creation(self) -> None:
        """Create with all fields."""
        now = datetime.now(timezone.utc)
        state = AITurnState(
            turn_id="turn-456",
            prompt="Test prompt",
            status=AITurnStatus.RUNNING,
            edit_count=5,
            error="Some error",
            created_at=now,
            completed_at=now,
            metadata={"key": "value"},
        )
        assert state.turn_id == "turn-456"
        assert state.prompt == "Test prompt"
        assert state.status == AITurnStatus.RUNNING
        assert state.edit_count == 5
        assert state.error == "Some error"
        assert state.created_at == now
        assert state.completed_at == now
        assert state.metadata == {"key": "value"}

    def test_created_at_default(self) -> None:
        """created_at defaults to current time."""
        before = datetime.now(timezone.utc)
        state = AITurnState(turn_id="t", prompt="p")
        after = datetime.now(timezone.utc)
        assert before <= state.created_at <= after


class TestAITurnStateProperties:
    """Tests for AITurnState property methods."""

    def test_is_running_true(self) -> None:
        """is_running returns True when RUNNING."""
        state = AITurnState(turn_id="t", prompt="p", status=AITurnStatus.RUNNING)
        assert state.is_running is True

    def test_is_running_false(self) -> None:
        """is_running returns False for other states."""
        for status in [
            AITurnStatus.PENDING,
            AITurnStatus.COMPLETED,
            AITurnStatus.FAILED,
            AITurnStatus.CANCELED,
        ]:
            state = AITurnState(turn_id="t", prompt="p", status=status)
            assert state.is_running is False, f"Expected False for {status}"

    def test_is_finished_terminal_states(self) -> None:
        """is_finished returns True for terminal states."""
        for status in [
            AITurnStatus.COMPLETED,
            AITurnStatus.FAILED,
            AITurnStatus.CANCELED,
        ]:
            state = AITurnState(turn_id="t", prompt="p", status=status)
            assert state.is_finished is True, f"Expected True for {status}"

    def test_is_finished_non_terminal_states(self) -> None:
        """is_finished returns False for non-terminal states."""
        for status in [AITurnStatus.PENDING, AITurnStatus.RUNNING]:
            state = AITurnState(turn_id="t", prompt="p", status=status)
            assert state.is_finished is False, f"Expected False for {status}"

    def test_is_successful_completed_only(self) -> None:
        """is_successful returns True only for COMPLETED."""
        state = AITurnState(turn_id="t", prompt="p", status=AITurnStatus.COMPLETED)
        assert state.is_successful is True

    def test_is_successful_false_for_others(self) -> None:
        """is_successful returns False for non-completed states."""
        for status in [
            AITurnStatus.PENDING,
            AITurnStatus.RUNNING,
            AITurnStatus.FAILED,
            AITurnStatus.CANCELED,
        ]:
            state = AITurnState(turn_id="t", prompt="p", status=status)
            assert state.is_successful is False, f"Expected False for {status}"


class TestAITurnStateTransitions:
    """Tests for AITurnState transition methods."""

    def test_mark_running(self) -> None:
        """mark_running sets status to RUNNING."""
        state = AITurnState(turn_id="t", prompt="p")
        state.mark_running()
        assert state.status == AITurnStatus.RUNNING

    def test_mark_completed(self) -> None:
        """mark_completed sets status and edit_count."""
        state = AITurnState(turn_id="t", prompt="p", status=AITurnStatus.RUNNING)
        before = datetime.now(timezone.utc)
        state.mark_completed(edit_count=3)
        after = datetime.now(timezone.utc)

        assert state.status == AITurnStatus.COMPLETED
        assert state.edit_count == 3
        assert state.completed_at is not None
        assert before <= state.completed_at <= after

    def test_mark_completed_default_edit_count(self) -> None:
        """mark_completed defaults to 0 edits."""
        state = AITurnState(turn_id="t", prompt="p", status=AITurnStatus.RUNNING)
        state.mark_completed()
        assert state.edit_count == 0

    def test_mark_failed(self) -> None:
        """mark_failed sets status and error."""
        state = AITurnState(turn_id="t", prompt="p", status=AITurnStatus.RUNNING)
        before = datetime.now(timezone.utc)
        state.mark_failed("Something went wrong")
        after = datetime.now(timezone.utc)

        assert state.status == AITurnStatus.FAILED
        assert state.error == "Something went wrong"
        assert state.completed_at is not None
        assert before <= state.completed_at <= after

    def test_mark_canceled(self) -> None:
        """mark_canceled sets status to CANCELED."""
        state = AITurnState(turn_id="t", prompt="p", status=AITurnStatus.RUNNING)
        before = datetime.now(timezone.utc)
        state.mark_canceled()
        after = datetime.now(timezone.utc)

        assert state.status == AITurnStatus.CANCELED
        assert state.completed_at is not None
        assert before <= state.completed_at <= after


# =============================================================================
# PendingEdit Tests
# =============================================================================


class TestPendingEditCreation:
    """Tests for PendingEdit dataclass creation."""

    def test_minimal_creation(self) -> None:
        """Create with required fields."""
        edit = PendingEdit(
            edit_id="edit-1",
            tab_id="tab-1",
            action="replace",
            range=(10, 20),
            diff="- old\n+ new",
        )
        assert edit.edit_id == "edit-1"
        assert edit.tab_id == "tab-1"
        assert edit.action == "replace"
        assert edit.range == (10, 20)
        assert edit.diff == "- old\n+ new"
        assert edit.spans == ()

    def test_full_creation(self) -> None:
        """Create with all fields."""
        now = datetime.now(timezone.utc)
        edit = PendingEdit(
            edit_id="edit-2",
            tab_id="tab-2",
            action="insert",
            range=(5, 5),
            diff="+ inserted",
            spans=((5, 15), (20, 30)),
            created_at=now,
        )
        assert edit.spans == ((5, 15), (20, 30))
        assert edit.created_at == now

    def test_created_at_default(self) -> None:
        """created_at defaults to current time."""
        before = datetime.now(timezone.utc)
        edit = PendingEdit(
            edit_id="e", tab_id="t", action="a", range=(0, 0), diff=""
        )
        after = datetime.now(timezone.utc)
        assert before <= edit.created_at <= after


# =============================================================================
# ReviewSession Tests
# =============================================================================


class TestReviewSessionCreation:
    """Tests for ReviewSession dataclass creation."""

    def test_minimal_creation(self) -> None:
        """Create with required fields."""
        # Use a placeholder for DocumentState since it's TYPE_CHECKING import
        session = ReviewSession(
            tab_id="tab-1",
            document_id="doc-1",
            snapshot=None,  # type: ignore[arg-type]
        )
        assert session.tab_id == "tab-1"
        assert session.document_id == "doc-1"
        assert session.snapshot is None
        assert session.edits == []
        assert session.previous_overlay is None
        assert session.merged_spans == ()

    def test_full_creation(self) -> None:
        """Create with all fields."""
        edit = PendingEdit(
            edit_id="e1", tab_id="tab-1", action="replace", range=(0, 10), diff="d"
        )
        session = ReviewSession(
            tab_id="tab-1",
            document_id="doc-1",
            snapshot=None,  # type: ignore[arg-type]
            edits=[edit],
            previous_overlay=None,
            merged_spans=((0, 10), (20, 30)),
        )
        assert len(session.edits) == 1
        assert session.merged_spans == ((0, 10), (20, 30))


# =============================================================================
# PendingReview Tests
# =============================================================================


class TestPendingReviewCreation:
    """Tests for PendingReview dataclass creation."""

    def test_minimal_creation(self) -> None:
        """Create with required fields."""
        review = PendingReview(turn_id="turn-1", prompt="Do something")
        assert review.turn_id == "turn-1"
        assert review.prompt == "Do something"
        assert review.sessions == {}
        assert review.ready is False
        assert review.total_edit_count == 0

    def test_full_creation(self) -> None:
        """Create with all fields."""
        now = datetime.now(timezone.utc)
        session = ReviewSession(
            tab_id="tab-1",
            document_id="doc-1",
            snapshot=None,  # type: ignore[arg-type]
        )
        review = PendingReview(
            turn_id="turn-2",
            prompt="Test prompt",
            sessions={"tab-1": session},
            ready=True,
            created_at=now,
            total_edit_count=5,
        )
        assert review.sessions == {"tab-1": session}
        assert review.ready is True
        assert review.created_at == now
        assert review.total_edit_count == 5


class TestPendingReviewMethods:
    """Tests for PendingReview methods."""

    def test_add_session(self) -> None:
        """add_session adds or replaces a session."""
        review = PendingReview(turn_id="t", prompt="p")
        session1 = ReviewSession(
            tab_id="tab-1", document_id="doc-1", snapshot=None  # type: ignore[arg-type]
        )
        review.add_session(session1)
        assert "tab-1" in review.sessions
        assert review.sessions["tab-1"] is session1

    def test_add_session_replaces_existing(self) -> None:
        """add_session replaces existing session for same tab."""
        session1 = ReviewSession(
            tab_id="tab-1", document_id="doc-1", snapshot=None  # type: ignore[arg-type]
        )
        session2 = ReviewSession(
            tab_id="tab-1", document_id="doc-2", snapshot=None  # type: ignore[arg-type]
        )
        review = PendingReview(
            turn_id="t", prompt="p", sessions={"tab-1": session1}
        )
        review.add_session(session2)
        assert review.sessions["tab-1"] is session2

    def test_get_session_existing(self) -> None:
        """get_session returns session for existing tab."""
        session = ReviewSession(
            tab_id="tab-1", document_id="doc-1", snapshot=None  # type: ignore[arg-type]
        )
        review = PendingReview(
            turn_id="t", prompt="p", sessions={"tab-1": session}
        )
        result = review.get_session("tab-1")
        assert result is session

    def test_get_session_missing(self) -> None:
        """get_session returns None for missing tab."""
        review = PendingReview(turn_id="t", prompt="p")
        result = review.get_session("nonexistent")
        assert result is None

    def test_affected_tab_count_empty(self) -> None:
        """affected_tab_count returns 0 with no sessions."""
        review = PendingReview(turn_id="t", prompt="p")
        assert review.affected_tab_count == 0

    def test_affected_tab_count_multiple(self) -> None:
        """affected_tab_count returns correct count."""
        session1 = ReviewSession(
            tab_id="tab-1", document_id="doc-1", snapshot=None  # type: ignore[arg-type]
        )
        session2 = ReviewSession(
            tab_id="tab-2", document_id="doc-2", snapshot=None  # type: ignore[arg-type]
        )
        review = PendingReview(
            turn_id="t",
            prompt="p",
            sessions={"tab-1": session1, "tab-2": session2},
        )
        assert review.affected_tab_count == 2

    def test_created_at_default(self) -> None:
        """created_at defaults to current time."""
        before = datetime.now(timezone.utc)
        review = PendingReview(turn_id="t", prompt="p")
        after = datetime.now(timezone.utc)
        assert before <= review.created_at <= after


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Test that modules export the expected symbols."""

    def test_ai_models_exports(self) -> None:
        """ai_models exports expected symbols."""
        from tinkerbell.ui.models import ai_models

        assert hasattr(ai_models, "AITurnStatus")
        assert hasattr(ai_models, "AITurnState")
        assert "AITurnStatus" in ai_models.__all__
        assert "AITurnState" in ai_models.__all__

    def test_review_models_exports(self) -> None:
        """review_models exports expected symbols."""
        from tinkerbell.ui.models import review_models

        assert hasattr(review_models, "PendingEdit")
        assert hasattr(review_models, "ReviewSession")
        assert hasattr(review_models, "PendingReview")
        assert "PendingEdit" in review_models.__all__
        assert "ReviewSession" in review_models.__all__
        assert "PendingReview" in review_models.__all__
