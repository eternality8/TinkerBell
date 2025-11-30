"""Unit tests for the status bar context usage widget."""

from __future__ import annotations

from tinkerbell.ui.presentation.widgets.status_bar import ContextUsageWidget, StatusBar


def test_context_usage_widget_tracks_state_without_qt() -> None:
    widget = ContextUsageWidget()
    widget.install(None)
    widget.update("Prompt 10", totals="Σ Prompt 10", last_tool="diff_builder")

    assert widget.summary_text == "Prompt 10"
    assert widget.totals_text == "Σ Prompt 10"
    assert widget.last_tool == "diff_builder"


def test_status_bar_set_memory_usage_updates_context_widget() -> None:
    bar = StatusBar()
    bar.set_memory_usage("Prompt 5", totals="Σ Prompt 25", last_tool="document_edit")

    widget = bar.context_widget
    assert widget.summary_text == "Prompt 5"
    assert widget.last_tool == "document_edit"


def test_status_bar_autosave_state_tracks_detail() -> None:
    bar = StatusBar()

    assert bar.autosave_state == ("Saved", "")

    bar.set_autosave_state("Autosaved just now", detail="draft.md")

    assert bar.autosave_state == ("Autosaved just now", "draft.md")


def test_status_bar_outline_status_tracks_tooltip() -> None:
    bar = StatusBar()

    bar.set_outline_status("Fresh", tooltip="Last build 120 ms")

    assert bar.outline_state == ("Fresh", "Last build 120 ms")

    bar.set_outline_status("", tooltip=None)

    assert bar.outline_state == ("", "")


def test_status_bar_embedding_status_tracks_detail() -> None:
    bar = StatusBar()

    bar.set_embedding_status("LangChain", detail="text-embedding-3-large")

    assert bar.embedding_state == ("LangChain", "text-embedding-3-large")

    bar.set_embedding_status(None, detail=None)

    assert bar.embedding_state == ("", "")


def test_status_bar_embedding_processing_indicator_extends_label() -> None:
    bar = StatusBar()
    bar.set_embedding_status("Ready", detail="text-embedding-3-small")

    bar.set_embedding_processing(True, detail="doc-123")

    assert bar._embedding_processing is True  # noqa: SLF001 - verifying private state for indicator
    assert "Processing..." in bar._format_embedding_text()  # noqa: SLF001

    bar.set_embedding_processing(False)

    assert bar._embedding_processing is False  # noqa: SLF001


def test_status_bar_subagent_status_tracks_tooltip() -> None:
    bar = StatusBar()

    bar.set_subagent_status("Idle", detail="Waiting for jobs")

    assert bar.subagent_state == ("Idle", "Waiting for jobs")

    bar.set_subagent_status(None, detail=None)

    assert bar.subagent_state == ("", "")


def test_status_bar_chunk_flow_indicator_tracks_state() -> None:
    bar = StatusBar()

    bar.set_chunk_flow_state("Warning", detail="doc-1 fetched full snapshot")

    assert bar.chunk_flow_state == ("Warning", "doc-1 fetched full snapshot")

    bar.set_chunk_flow_state(None)

    assert bar.chunk_flow_state == ("", "")


def test_status_bar_review_controls_toggle_visibility() -> None:
    bar = StatusBar()

    assert bar.review_summary == ""
    assert bar.review_controls_visible is False

    bar.set_review_state("3 edits across 2 tabs")

    assert bar.review_summary == "3 edits across 2 tabs"
    assert bar.review_controls_visible is True

    bar.clear_review_state()

    assert bar.review_summary == ""
    assert bar.review_controls_visible is False


def test_status_bar_review_control_callbacks_fire() -> None:
    bar = StatusBar()
    calls: list[str] = []

    bar.set_review_state(
        "1 edit",
        accept_callback=lambda: calls.append("accept"),
        reject_callback=lambda: calls.append("reject"),
    )

    bar._review_controls.trigger_accept()
    bar._review_controls.trigger_reject()

    assert calls == ["accept", "reject"]


def test_status_bar_document_status_badge_tracks_severity() -> None:
    bar = StatusBar()

    bar.set_document_status_badge("Doc Ready", detail="profile auto", severity="info")

    assert bar.document_status_badge == ("Doc Ready", "profile auto")
    assert bar.document_status_severity == "info"

    bar.set_document_status_badge(None, detail=None)

    assert bar.document_status_badge == ("", "")
    assert bar.document_status_severity == ""


def test_status_bar_document_status_callback_invokes_indicator() -> None:
    bar = StatusBar()
    calls: list[str] = []

    bar.set_document_status_callback(lambda: calls.append("clicked"))
    indicator = bar._document_status_indicator  # noqa: SLF001 - intentional white-box access
    indicator.set_state("Ready", "detail", severity="normal")

    indicator._handle_clicked()  # noqa: SLF001 - simulate user interaction

    assert calls == ["clicked"]
