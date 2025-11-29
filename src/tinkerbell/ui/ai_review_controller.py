"""AI review controller extracted from the main window."""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

from ..chat.chat_panel import ChatPanel, ChatTurnSnapshot
from ..chat.message_model import EditDirective
from ..editor.document_model import DocumentState
from ..editor.editor_widget import DiffOverlayState
from ..editor.workspace import DocumentTab, DocumentWorkspace
from ..documents.ranges import TextRange
from ..widgets.status_bar import StatusBar

LOGGER = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class EditSummary:
    """Tracks a single edit applied during an AI turn."""

    directive: EditDirective
    diff: str
    spans: tuple[tuple[int, int], ...]
    range_hint: TextRange
    created_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        object.__setattr__(self, "range_hint", TextRange.from_value(self.range_hint, fallback=(0, 0)))


@dataclass(slots=True)
class PendingReviewSession:
    """Per-tab session data for a pending AI turn review."""

    tab_id: str
    document_id: str
    document_snapshot: DocumentState
    previous_overlay: DiffOverlayState | None = None
    applied_edits: list[EditSummary] = field(default_factory=list)
    merged_spans: tuple[tuple[int, int], ...] = ()
    last_overlay_label: str | None = None
    orphaned: bool = False
    latest_version_signature: str | None = None


@dataclass(slots=True)
class PendingTurnReview:
    """Envelope capturing chat + document state for an AI turn."""

    turn_id: str
    prompt: str
    prompt_metadata: dict[str, Any]
    chat_snapshot: ChatTurnSnapshot
    created_at: datetime = field(default_factory=_utcnow)
    tab_sessions: dict[str, PendingReviewSession] = field(default_factory=dict)
    total_edit_count: int = 0
    total_tabs_affected: int = 0
    ready_for_review: bool = False
    completed: bool = False


class AIReviewController:
    """Coordinates pending AI turn reviews outside the window class."""

    def __init__(
        self,
        *,
        status_bar: StatusBar | None,
        chat_panel: ChatPanel,
        workspace: DocumentWorkspace,
        clear_diff_overlay: Callable[[str | None], None],
        update_status: Callable[[str], None],
        post_assistant_notice: Callable[[str], None],
        accept_callback: Callable[[], None] | None = None,
        reject_callback: Callable[[], None] | None = None,
    ) -> None:
        self._status_bar = status_bar
        self._chat_panel = chat_panel
        self._workspace = workspace
        self._clear_diff_overlay = clear_diff_overlay
        self._update_status = update_status
        self._post_assistant_notice = post_assistant_notice
        self._accept_callback = accept_callback
        self._reject_callback = reject_callback
        self._pending_turn_review: PendingTurnReview | None = None
        self._pending_reviews_by_tab: dict[str, PendingReviewSession] = {}
        self._turn_sequence = 0
        self._last_review_summary: str | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def pending_turn_review(self) -> PendingTurnReview | None:
        return self._pending_turn_review

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    def begin_pending_turn_review(
        self,
        *,
        prompt: str,
        prompt_metadata: Mapping[str, Any] | None,
        chat_snapshot: ChatTurnSnapshot,
    ) -> PendingTurnReview:
        self._turn_sequence += 1
        turn_id = f"turn-{self._turn_sequence}"
        metadata_copy = dict(prompt_metadata or {})
        envelope = PendingTurnReview(
            turn_id=turn_id,
            prompt=prompt,
            prompt_metadata=metadata_copy,
            chat_snapshot=chat_snapshot,
        )
        self._pending_turn_review = envelope
        self._pending_reviews_by_tab = envelope.tab_sessions
        self.clear_review_controls()
        LOGGER.debug("Begin pending turn review turn_id=%s", turn_id)
        return envelope

    def drop_pending_turn_review(self, *, reason: str | None = None) -> None:
        if self._pending_turn_review is None:
            return
        turn_id = self._pending_turn_review.turn_id
        self._pending_turn_review = None
        self._pending_reviews_by_tab = {}
        self.clear_review_controls()
        if reason:
            LOGGER.debug("Dropped pending turn review turn_id=%s reason=%s", turn_id, reason)
        else:
            LOGGER.debug("Dropped pending turn review turn_id=%s", turn_id)

    def auto_accept_pending_review(self, *, reason: str = "new-turn") -> None:
        if self._pending_turn_review is None:
            return
        LOGGER.debug("Auto-accepting pending review turn_id=%s reason=%s", self._pending_turn_review.turn_id, reason)
        self.drop_pending_turn_review(reason=reason)

    def finalize_pending_turn_review(self, *, success: bool) -> None:
        turn = self._pending_turn_review
        LOGGER.debug(
            "finalize_pending_turn_review: success=%s turn=%s edit_count=%s",
            success,
            turn.turn_id if turn else None,
            turn.total_edit_count if turn else None,
        )
        if turn is None:
            return
        turn.completed = True
        if not success:
            self.drop_pending_turn_review(reason="turn-failed")
            return
        if turn.total_edit_count <= 0:
            self.drop_pending_turn_review(reason="no-edits")
            return
        turn.ready_for_review = True
        LOGGER.debug("finalize_pending_turn_review: showing review controls")
        self.show_review_controls()

    def abort_pending_review(
        self,
        *,
        reason: str,
        status: str | None = None,
        notice: str | None = None,
        restore_composer: bool = False,
        clear_overlays: bool = False,
    ) -> None:
        turn = self._pending_turn_review
        if turn is None:
            return
        if clear_overlays:
            self._clear_pending_review_overlays(turn)
        if restore_composer:
            self._restore_composer_prompt(turn)
        self.drop_pending_turn_review(reason=reason)
        if status:
            self._update_status(status)
        if notice:
            self._post_assistant_notice(notice)

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    def ensure_pending_review_session(
        self,
        *,
        tab_id: str,
        document_snapshot: DocumentState,
        tab: DocumentTab | None = None,
    ) -> PendingReviewSession | None:
        turn = self._pending_turn_review
        if turn is None:
            return None
        session = self._pending_reviews_by_tab.get(tab_id)
        if session is not None:
            return session
        actual_tab = tab
        if actual_tab is None:
            try:
                actual_tab = self._workspace.get_tab(tab_id)
            except KeyError:
                LOGGER.debug("Unable to create review session; tab %s missing", tab_id)
                return None
        snapshot_copy = deepcopy(document_snapshot)
        existing_overlay = actual_tab.editor.diff_overlay
        prior_overlay = deepcopy(existing_overlay) if existing_overlay is not None else None
        session = PendingReviewSession(
            tab_id=tab_id,
            document_id=document_snapshot.document_id,
            document_snapshot=snapshot_copy,
            previous_overlay=prior_overlay,
        )
        self._pending_reviews_by_tab[tab_id] = session
        turn.total_tabs_affected = len(self._pending_reviews_by_tab)
        return session

    def mark_pending_session_orphaned(self, tab_id: str, *, reason: str) -> None:
        if not tab_id:
            return
        session = self._pending_reviews_by_tab.get(tab_id)
        if session is None or session.orphaned:
            return
        session.orphaned = True
        LOGGER.debug("Marked tab %s orphaned during pending review (%s)", tab_id, reason)

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def show_review_controls(self) -> None:
        turn = self._pending_turn_review
        if turn is None or not turn.ready_for_review:
            self.clear_review_controls()
            return
        summary = self.format_review_summary(turn)
        self._last_review_summary = summary
        status_bar = self._status_bar
        if status_bar is None:
            return
        try:
            status_bar.set_review_state(
                summary,
                accept_callback=self._accept_callback,
                reject_callback=self._reject_callback,
            )
        except Exception:
            LOGGER.debug("Unable to display review controls", exc_info=True)

    def clear_review_controls(self) -> None:
        self._last_review_summary = None
        status_bar = self._status_bar
        if status_bar is None:
            return
        try:
            status_bar.clear_review_state()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def format_review_summary(turn: PendingTurnReview | None) -> str:
        if turn is None:
            return "0 edits across 0 tabs"
        edits = max(int(turn.total_edit_count), 0)
        tabs = max(int(turn.total_tabs_affected), 1)
        edit_label = "edit" if edits == 1 else "edits"
        tab_label = "tab" if tabs == 1 else "tabs"
        return f"{edits} {edit_label} across {tabs} {tab_label}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _restore_composer_prompt(self, turn: PendingTurnReview) -> None:
        snapshot = turn.chat_snapshot
        composer_text = turn.prompt
        composer_context = None
        if snapshot is not None:
            composer_text = snapshot.composer_text or composer_text
            composer_context = snapshot.composer_context
        context_copy = deepcopy(composer_context) if composer_context is not None else None
        try:
            self._chat_panel.set_composer_text(composer_text or "", context=context_copy)
        except Exception:
            LOGGER.debug("Unable to restore composer prompt", exc_info=True)

    def _clear_pending_review_overlays(self, turn: PendingTurnReview) -> None:
        for session in turn.tab_sessions.values():
            tab_id = session.tab_id
            if not tab_id:
                continue
            try:
                self._workspace.get_tab(tab_id)
            except KeyError:
                continue
            self._clear_diff_overlay(tab_id)


__all__ = [
    "AIReviewController",
    "EditSummary",
    "PendingReviewSession",
    "PendingTurnReview",
]
