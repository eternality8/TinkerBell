"""Unit tests for the status bar context usage widget."""

from __future__ import annotations

from tinkerbell.widgets.status_bar import ContextUsageWidget, StatusBar


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


def test_status_bar_subagent_status_tracks_tooltip() -> None:
    bar = StatusBar()

    bar.set_subagent_status("Idle", detail="Waiting for jobs")

    assert bar.subagent_state == ("Idle", "Waiting for jobs")

    bar.set_subagent_status(None, detail=None)

    assert bar.subagent_state == ("", "")
