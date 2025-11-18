"""Unit tests for :mod:`tinkerbell.ui.review_overlay_manager`."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

from tinkerbell.chat.message_model import ToolTrace
from tinkerbell.editor.document_model import DocumentMetadata, DocumentState
from tinkerbell.editor.editor_widget import DiffOverlayState
from tinkerbell.ui.ai_review_controller import PendingReviewSession, PendingTurnReview
from tinkerbell.ui.review_overlay_manager import ReviewOverlayManager


class _StubBridge:
    def __init__(self) -> None:
        self.snapshot_calls = 0

    def generate_snapshot(self, *, delta_only: bool = True) -> None:
        self.snapshot_calls += 1


class _StubWorkspace:
    def __init__(self, tabs: list[_StubTab], *, active_tab_id: str | None = None) -> None:
        self._tabs = {tab.id: tab for tab in tabs}
        self._active_tab_id = active_tab_id

    def get_tab(self, tab_id: str) -> "_StubTab":
        if tab_id not in self._tabs:
            raise KeyError(tab_id)
        return self._tabs[tab_id]

    def iter_tabs(self) -> tuple["_StubTab", ...]:
        return tuple(self._tabs.values())

    @property
    def active_tab_id(self) -> str | None:
        return self._active_tab_id

    @active_tab_id.setter
    def active_tab_id(self, tab_id: str | None) -> None:
        self._active_tab_id = tab_id

    def remove_tab(self, tab_id: str) -> None:
        self._tabs.pop(tab_id, None)


class _StubEditorFacade:
    def __init__(self, workspace: _StubWorkspace) -> None:
        self._workspace = workspace

    def show_diff_overlay(
        self,
        diff_text: str,
        *,
        spans: tuple[tuple[int, int], ...] | None = None,
        summary: str | None = None,
        source: str | None = None,
        tab_id: str | None = None,
    ) -> None:
        target_id = tab_id or self._workspace.active_tab_id
        tab = self._workspace.get_tab(target_id)
        tab.editor.show_diff_overlay(diff_text, spans=spans, summary=summary, source=source)

    def clear_diff_overlay(self, *, tab_id: str | None = None) -> None:
        target_id = tab_id or self._workspace.active_tab_id
        tab = self._workspace.get_tab(target_id)
        tab.editor.clear_diff_overlay()


class _TabEditor:
    def __init__(self, parent: "_StubTab") -> None:
        self._parent = parent
        self.diff_overlay: DiffOverlayState | None = None
        self.last_change_source = "ai"

    def to_document(self) -> DocumentState:
        return self._parent._document

    def load_document(self, document: DocumentState) -> None:
        self._parent._document = document

    def show_diff_overlay(
        self,
        diff_text: str,
        *,
        spans: tuple[tuple[int, int], ...] | None = None,
        summary: str | None = None,
        source: str | None = None,
    ) -> None:
        self.diff_overlay = DiffOverlayState(
            diff=diff_text,
            spans=tuple(spans or ()),
            summary=summary,
            source=source,
        )

    def clear_diff_overlay(self) -> None:
        self.diff_overlay = None


class _StubTab:
    def __init__(self, tab_id: str, document: DocumentState, *, title: str | None = None) -> None:
        self.id = tab_id
        self._document = document
        self.editor = _TabEditor(self)
        self.bridge = _StubBridge()
        self.title = title or tab_id

    def document(self) -> DocumentState:
        return self.editor.to_document()

    def update_title(self, fallback: str = "Untitled") -> None:  # noqa: D401 - signature parity
        del fallback
        # Titles are not material to these tests; keep the method for API parity.


class _StubChatPanel:
    def __init__(self) -> None:
        self.restored_snapshots: list[Any] = []

    def restore_state(self, snapshot: Any) -> None:
        self.restored_snapshots.append(snapshot)


class _StubReviewController:
    def __init__(self) -> None:
        self.pending_turn_review: PendingTurnReview | None = None
        self.summary_text = "0 edits"
        self.drop_reasons: list[str | None] = []
        self.abort_calls: list[dict[str, Any]] = []

    def format_review_summary(self, turn: PendingTurnReview) -> str:
        return self.summary_text

    def drop_pending_turn_review(self, *, reason: str | None = None) -> None:
        self.drop_reasons.append(reason)
        self.pending_turn_review = None

    def abort_pending_review(
        self,
        *,
        reason: str,
        status: str | None = None,
        notice: str | None = None,
        restore_composer: bool = False,
        clear_overlays: bool = False,
    ) -> None:
        self.abort_calls.append(
            {
                "reason": reason,
                "status": status,
                "notice": notice,
                "restore_composer": restore_composer,
                "clear_overlays": clear_overlays,
            }
        )
        self.pending_turn_review = None


class _Harness:
    def __init__(self, *tabs: _StubTab) -> None:
        self.workspace = _StubWorkspace(list(tabs), active_tab_id=tabs[0].id if tabs else None)
        self.editor = _StubEditorFacade(self.workspace)
        self.review = _StubReviewController()
        self.chat_panel = _StubChatPanel()
        self.status_messages: list[str] = []
        self.notices: list[str] = []
        self.window_title_docs: list[DocumentState] = []
        self.autosave_docs: list[DocumentState] = []
        self.sync_calls = 0
        self.manager = ReviewOverlayManager(
            editor=self.editor,
            workspace=self.workspace,
            review_controller=self.review,
            chat_panel=self.chat_panel,
            status_updater=self.status_messages.append,
            notice_poster=self.notices.append,
            window_title_refresher=lambda document: self.window_title_docs.append(document),
            autosave_updater=lambda document: self.autosave_docs.append(document),
            sync_workspace_state=self._sync_workspace_state,
        )

    def _sync_workspace_state(self) -> None:
        self.sync_calls += 1


def _make_document(text: str) -> DocumentState:
    return DocumentState(text=text, metadata=DocumentMetadata())


def test_apply_diff_overlay_tracks_tab_ids() -> None:
    doc = _make_document("Hello world")
    tab = _StubTab("tab-1", doc)
    harness = _Harness(tab)

    trace = ToolTrace(
        name="test-tool",
        input_summary="",
        output_summary="patched",
        metadata={"spans": [["bad"]], "diff_preview": "Δ preview"},
    )

    harness.manager.apply_diff_overlay(trace, document=doc, range_hint=(0, 5))

    overlay = tab.editor.diff_overlay
    assert overlay is not None
    assert overlay.diff == "Δ preview"
    assert overlay.spans == ((0, 5),)
    assert harness.manager.overlay_tab_ids() == ("tab-1",)


def test_manual_edit_abort_clears_overlay_and_turn() -> None:
    doc = _make_document("Draft text")
    tab = _StubTab("draft", doc)
    harness = _Harness(tab)

    trace = ToolTrace(name="edit", input_summary="", output_summary="diff", metadata={})
    harness.manager.apply_diff_overlay(trace, document=doc, range_hint=(0, len(doc.text)))
    harness.review.pending_turn_review = PendingTurnReview(
        turn_id="turn-1",
        prompt="prompt",
        prompt_metadata={},
        chat_snapshot=SimpleNamespace(),
    )

    tab.editor.last_change_source = "user"
    harness.manager.maybe_clear_diff_overlay(doc)

    assert harness.manager.overlay_tab_ids() == ()
    assert tab.editor.diff_overlay is None
    assert harness.review.abort_calls[-1]["reason"] == "manual-edit"
    assert harness.review.abort_calls[-1]["clear_overlays"] is True


def test_handle_accept_ai_changes_reports_skipped_tabs() -> None:
    base_doc = _make_document("Original")
    kept_tab = _StubTab("kept", deepcopy(base_doc))
    skipped_tab = _StubTab("skipped", deepcopy(base_doc))
    harness = _Harness(kept_tab, skipped_tab)
    harness.review.summary_text = "2 edits across 2 tabs"

    turn = PendingTurnReview(
        turn_id="turn-5",
        prompt="Summaries",
        prompt_metadata={},
        chat_snapshot=None,
    )
    turn.ready_for_review = True
    turn.tab_sessions = {
        kept_tab.id: PendingReviewSession(
            tab_id=kept_tab.id,
            document_id=kept_tab.document().document_id,
            document_snapshot=deepcopy(kept_tab.document()),
        ),
        skipped_tab.id: PendingReviewSession(
            tab_id=skipped_tab.id,
            document_id=skipped_tab.document().document_id,
            document_snapshot=deepcopy(skipped_tab.document()),
        ),
    }
    harness.review.pending_turn_review = turn

    trace = ToolTrace(name="edit", input_summary="", output_summary="diff", metadata={})
    harness.manager.apply_diff_overlay(trace, document=kept_tab.document(), range_hint=(0, 1), tab_id=kept_tab.id)
    harness.workspace.remove_tab(skipped_tab.id)

    harness.manager.handle_accept_ai_changes()

    assert harness.review.pending_turn_review is None
    assert harness.review.drop_reasons == ["accepted"]
    assert "Accepted 2 edits across 2 tabs" in harness.status_messages[-1]
    assert "skipped 1 closed tab" in harness.status_messages[-1]
    assert harness.notices[-1] == harness.status_messages[-1]
    assert harness.manager.overlay_tab_ids() == ()


def test_handle_reject_ai_changes_restores_documents_and_chat() -> None:
    original_primary = _make_document("Base primary")
    original_secondary = _make_document("Base secondary")
    primary_tab = _StubTab("primary", deepcopy(original_primary))
    secondary_tab = _StubTab("secondary", deepcopy(original_secondary))
    harness = _Harness(primary_tab, secondary_tab)
    harness.review.summary_text = "2 edits across 2 tabs"

    snapshot = SimpleNamespace(token="snapshot")
    turn = PendingTurnReview(
        turn_id="turn-7",
        prompt="Rewrite",
        prompt_metadata={},
        chat_snapshot=snapshot,
    )
    turn.ready_for_review = True

    def _make_session(tab: _StubTab, source_doc: DocumentState) -> PendingReviewSession:
        session = PendingReviewSession(
            tab_id=tab.id,
            document_id=source_doc.document_id,
            document_snapshot=deepcopy(source_doc),
            previous_overlay=DiffOverlayState(diff="old", spans=((0, 4),), summary="summary", source="tool"),
        )
        tab.editor.load_document(
            DocumentState(
                text=f"AI {source_doc.text}",
                metadata=DocumentMetadata(),
                document_id=source_doc.document_id,
            )
        )
        session.latest_version_signature = tab.document().version_signature()
        return session

    primary_session = _make_session(primary_tab, original_primary)
    secondary_session = _make_session(secondary_tab, original_secondary)
    turn.tab_sessions = {primary_tab.id: primary_session, secondary_tab.id: secondary_session}
    harness.review.pending_turn_review = turn

    harness.manager.handle_reject_ai_changes()

    assert harness.review.pending_turn_review is None
    assert harness.review.drop_reasons == ["rejected"]
    assert harness.status_messages[-1].startswith("Rejected 2 edits across 2 tabs")
    assert harness.notices[-1] == harness.status_messages[-1]
    assert harness.chat_panel.restored_snapshots == [snapshot]
    assert harness.sync_calls == 1

    assert primary_tab.document().text == original_primary.text
    assert secondary_tab.document().text == original_secondary.text
    assert primary_tab.editor.diff_overlay.diff == "old"
    assert secondary_tab.editor.diff_overlay.diff == "old"
    assert len(harness.autosave_docs) == 2
    assert harness.window_title_docs[-1].text == primary_tab.document().text
    assert primary_tab.bridge.snapshot_calls == 1
    assert secondary_tab.bridge.snapshot_calls == 1