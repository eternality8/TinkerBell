"""Manages diff overlays and review flows outside of :mod:`main_window`."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

from ..chat.chat_panel import ChatPanel
from ..chat.message_model import ToolTrace
from ..editor.document_model import DocumentState
from ..editor.editor_widget import DiffOverlayState
from ..editor.workspace import DocumentTab, DocumentWorkspace
from ..documents.ranges import TextRange
from .ai_review_controller import AIReviewController, PendingReviewSession

LOGGER = logging.getLogger(__name__)


class ReviewOverlayManager:
    """Owns diff overlay bookkeeping plus accept/reject orchestration."""

    def __init__(
        self,
        *,
        editor: Any,
        workspace: DocumentWorkspace,
        review_controller: AIReviewController,
        chat_panel: ChatPanel,
        status_updater: Callable[[str], None],
        notice_poster: Callable[[str], None],
        window_title_refresher: Callable[[DocumentState], None],
        autosave_updater: Callable[[DocumentState], None],
        sync_workspace_state: Callable[[], None],
    ) -> None:
        self._editor = editor
        self._workspace = workspace
        self._review_controller = review_controller
        self._chat_panel = chat_panel
        self._status_updater = status_updater
        self._notice_poster = notice_poster
        self._refresh_window_title = window_title_refresher
        self._update_autosave_indicator = autosave_updater
        self._sync_workspace_state = sync_workspace_state
        self._tabs_with_overlay: set[str] = set()

    # ------------------------------------------------------------------
    # Overlay lifecycle helpers
    # ------------------------------------------------------------------
    def apply_diff_overlay(
        self,
        trace: ToolTrace,
        *,
        document: DocumentState,
        range_hint: TextRange | Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
        tab_id: str | None = None,
        spans_override: tuple[tuple[int, int], ...] | None = None,
        label_override: str | None = None,
    ) -> None:
        target_id = tab_id or self.find_tab_id_for_document(document)
        if not target_id:
            return
        normalized_range = self._coerce_text_range(range_hint)
        metadata = trace.metadata if isinstance(trace.metadata, Mapping) else {}
        spans = spans_override if spans_override is not None else self.coerce_overlay_spans(
            metadata.get("spans"),
            fallback_range=normalized_range,
        )
        diff_payload = metadata.get("diff_preview") if isinstance(metadata, Mapping) else None
        label = label_override or str(diff_payload or trace.output_summary or trace.name)
        try:
            self._editor.show_diff_overlay(
                label,
                spans=spans,
                summary=trace.output_summary,
                source=trace.name,
                tab_id=target_id,
            )
        except Exception:  # pragma: no cover - defensive guard
            return
        self._tabs_with_overlay.add(target_id)

    def maybe_clear_diff_overlay(self, state: DocumentState) -> None:
        tab_id = self.find_tab_id_for_document(state)
        if not tab_id or tab_id not in self._tabs_with_overlay:
            return
        try:
            tab = self._workspace.get_tab(tab_id)
        except KeyError:
            self._tabs_with_overlay.discard(tab_id)
            return
        change_source = getattr(tab.editor, "last_change_source", "")
        if change_source != "user":
            return
        if self._review_controller.pending_turn_review is not None:
            self._review_controller.abort_pending_review(
                reason="manual-edit",
                status="AI edits discarded after manual edit",
                notice="Pending AI edits were cleared because you modified the document.",
                clear_overlays=True,
            )
        self.clear_diff_overlay(tab_id)

    def clear_diff_overlay(self, tab_id: str | None = None) -> None:
        target_id = tab_id or self._workspace.active_tab_id
        if not target_id or target_id not in self._tabs_with_overlay:
            return
        try:
            self._editor.clear_diff_overlay(tab_id=target_id)
        except KeyError:  # pragma: no cover - already closed
            pass
        self._tabs_with_overlay.discard(target_id)

    def discard_overlay(self, tab_id: str) -> None:
        self._tabs_with_overlay.discard(tab_id)

    def reset_overlays(self) -> None:
        self._tabs_with_overlay.clear()

    def overlay_tab_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._tabs_with_overlay))

    # ------------------------------------------------------------------
    # Accept / reject handling
    # ------------------------------------------------------------------
    def handle_accept_ai_changes(self) -> None:
        turn = self._review_controller.pending_turn_review
        if turn is None:
            self._status_updater("No AI edits pending review")
            return
        if not turn.ready_for_review:
            self._status_updater("AI turn still running – review not ready")
            return

        skipped_tabs: list[str] = []
        for session in list(turn.tab_sessions.values()):
            tab_id = session.tab_id
            if not tab_id:
                continue
            if session.orphaned:
                skipped_tabs.append(tab_id)
                continue
            try:
                self._workspace.get_tab(tab_id)
            except KeyError:
                session.orphaned = True
                skipped_tabs.append(tab_id)
                continue
            self.clear_diff_overlay(tab_id=tab_id)

        if skipped_tabs:
            LOGGER.debug(
                "Skipped clearing overlays for %s tab(s) during accept: %s",
                len(skipped_tabs),
                ", ".join(skipped_tabs),
            )

        summary = self._review_controller.format_review_summary(turn)
        notice = f"Accepted {summary}"
        if skipped_tabs:
            suffix = "tab" if len(skipped_tabs) == 1 else "tabs"
            notice = f"{notice} (skipped {len(skipped_tabs)} closed {suffix})"
        self._review_controller.drop_pending_turn_review(reason="accepted")
        self._status_updater(notice)
        self._notice_poster(notice)

    def handle_reject_ai_changes(self) -> None:
        turn = self._review_controller.pending_turn_review
        if not turn or not turn.ready_for_review:
            self._status_updater("No AI edits pending review")
            return
        sessions = list(turn.tab_sessions.values())
        if not sessions:
            self._status_updater("No AI edits pending review")
            self._review_controller.drop_pending_turn_review(reason="empty-review")
            return

        skipped_tabs: list[str] = []
        blocked_tabs: list[str] = []
        tabs_to_restore: list[tuple[DocumentTab, PendingReviewSession]] = []
        for session in sessions:
            if session.orphaned:
                skipped_tabs.append(session.tab_id)
                continue
            try:
                tab = self._workspace.get_tab(session.tab_id)
            except KeyError:
                session.orphaned = True
                skipped_tabs.append(session.tab_id)
                continue

            document = tab.document()
            display_name = tab.title or session.tab_id
            if document.document_id != session.document_id:
                blocked_tabs.append(f"{display_name} now points to a different document")
                continue
            if session.latest_version_signature:
                current_signature = document.version_signature()
                if current_signature != session.latest_version_signature:
                    blocked_tabs.append(f"{display_name} changed since the AI turn finished")
                    continue
            tabs_to_restore.append((tab, session))

        if blocked_tabs:
            detail = "; ".join(blocked_tabs)
            self._notice_poster(
                "Reject canceled because some tabs changed after the AI edits: " + detail
            )
            self._status_updater("Reject canceled; documents changed after AI edits")
            return

        for tab, session in tabs_to_restore:
            snapshot_copy = deepcopy(session.document_snapshot)
            tab.editor.load_document(snapshot_copy)
            tab.update_title()
            self._tabs_with_overlay.discard(tab.id)
            prior_overlay = session.previous_overlay
            if prior_overlay is not None:
                try:
                    tab.editor.show_diff_overlay(
                        prior_overlay.diff,
                        spans=prior_overlay.spans,
                        summary=prior_overlay.summary,
                        source=prior_overlay.source,
                    )
                    self._tabs_with_overlay.add(tab.id)
                except Exception:  # pragma: no cover - defensive
                    LOGGER.debug("Unable to restore previous overlay for tab %s", tab.id, exc_info=True)
            document = tab.document()
            if self._workspace.active_tab_id == tab.id:
                self._refresh_window_title(document)
            self._update_autosave_indicator(document)
            try:
                tab.bridge.generate_snapshot(delta_only=True)
            except Exception:  # pragma: no cover - snapshot best effort
                LOGGER.debug("Unable to refresh bridge snapshot for tab %s", tab.id, exc_info=True)

        if skipped_tabs:
            LOGGER.debug(
                "Skipped rolling back %s tab(s) during reject: %s",
                len(skipped_tabs),
                ", ".join(skipped_tabs),
            )

        snapshot = turn.chat_snapshot
        if snapshot is not None:
            try:
                self._chat_panel.restore_state(snapshot)
            except Exception:  # pragma: no cover - chat optional
                LOGGER.debug("Unable to restore chat snapshot during reject", exc_info=True)

        self._sync_workspace_state()
        summary = self._review_controller.format_review_summary(turn)
        notice = f"Rejected {summary}"
        if skipped_tabs:
            suffix = "tab" if len(skipped_tabs) == 1 else "tabs"
            notice = f"{notice} (skipped {len(skipped_tabs)} closed {suffix})"
        self._review_controller.drop_pending_turn_review(reason="rejected")
        self._status_updater(notice)
        self._notice_poster(notice)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def find_tab_id_for_document(self, document: DocumentState) -> str | None:
        document_id = getattr(document, "document_id", None)
        for tab in self._workspace.iter_tabs():
            try:
                tab_document = tab.document()
            except Exception:  # pragma: no cover - defensive guard
                continue
            if document_id and tab_document.document_id == document_id:
                return tab.id
        return self._workspace.active_tab_id

    @staticmethod
    def coerce_overlay_spans(
        raw_spans: Any,
        *,
        fallback_range: TextRange | Mapping[str, Any] | Sequence[int] | tuple[int, int] | None,
    ) -> tuple[tuple[int, int], ...]:
        spans: list[tuple[int, int]] = []
        if isinstance(raw_spans, Sequence):
            for entry in raw_spans:
                if not isinstance(entry, Sequence) or len(entry) != 2:
                    continue
                try:
                    start = int(entry[0])
                    end = int(entry[1])
                except (TypeError, ValueError):
                    continue
                if start == end:
                    continue
                if end < start:
                    start, end = end, start
                spans.append((start, end))
        fallback_tuple: tuple[int, int] | None = None
        if fallback_range is not None:
            normalized_range = ReviewOverlayManager._coerce_text_range(fallback_range)
            fallback_tuple = normalized_range.to_tuple() if normalized_range is not None else None
        if not spans and fallback_tuple is not None and fallback_tuple[0] != fallback_tuple[1]:
            start, end = fallback_tuple
            if end < start:
                start, end = end, start
            spans.append((start, end))
        return tuple(spans)

    @staticmethod
    def _coerce_text_range(range_hint: TextRange | Mapping[str, Any] | Sequence[int] | tuple[int, int] | None) -> TextRange | None:
        if range_hint is None:
            return None
        try:
            return TextRange.from_value(range_hint)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def merge_overlay_spans(
        existing: tuple[tuple[int, int], ...],
        new_spans: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        if not existing:
            return new_spans
        if not new_spans:
            return existing
        ordered = sorted(existing + new_spans, key=lambda span: span[0])
        merged: list[list[int]] = []
        for start, end in ordered:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
                continue
            merged[-1][1] = max(merged[-1][1], end)
        return tuple((start, end) for start, end in merged)


__all__ = ["ReviewOverlayManager"]
