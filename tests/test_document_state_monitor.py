"""Unit tests for :mod:`tinkerbell.ui.document_state_monitor`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tinkerbell.ai.memory.cache_bus import DocumentCacheBus, DocumentChangedEvent
from tinkerbell.editor.document_model import DocumentMetadata, DocumentState
from tinkerbell.services.unsaved_cache import UnsavedCache
from tinkerbell.ui.document_state_monitor import DocumentStateMonitor


class _StubStatusBar:
    def __init__(self) -> None:
        self.states: list[tuple[str, str]] = []

    def set_autosave_state(self, status: str, *, detail: str) -> None:  # noqa: D401 - interface shim
        self.states.append((status, detail))


class _StubChatPanel:
    def __init__(self) -> None:
        self.suggestions: list[str] = []
        self.tool_traces: list[Any] = []

    def set_suggestions(self, suggestions: list[str]) -> None:
        self.suggestions = list(suggestions)

    def show_tool_trace(self, trace: Any) -> None:
        self.tool_traces.append(trace)


class _StubWorkspace:
    def __init__(self) -> None:
        self.documents: dict[str, DocumentState] = {}
        self.active_tab = SimpleNamespace(id="tab-1")
        self.active_tab_id = "tab-1"

    def register(self, document: DocumentState) -> None:
        self.documents[document.document_id] = document

    def find_document_by_id(self, document_id: str) -> DocumentState | None:
        return self.documents.get(document_id)


class _StubEditor:
    def __init__(self, document: DocumentState, workspace: _StubWorkspace) -> None:
        self._document = document
        self.workspace = workspace

    def to_document(self) -> DocumentState:
        return self._document


class _StubTab:
    def __init__(self, document: DocumentState) -> None:
        self._document = document

    def document(self) -> DocumentState:
        return self._document


def _make_document(
    *,
    text: str = "Hello world",
    path: Path | None = None,
) -> DocumentState:
    document = DocumentState(text=text, metadata=DocumentMetadata(path=path))
    document.dirty = True
    return document


def _make_monitor(document: DocumentState | None = None) -> tuple[DocumentStateMonitor, SimpleNamespace]:
    doc = document or _make_document()
    workspace = _StubWorkspace()
    workspace.register(doc)
    editor = _StubEditor(doc, workspace)
    chat_panel = _StubChatPanel()
    status_bar = _StubStatusBar()
    tracker = SimpleNamespace()
    tracker.cache = UnsavedCache()
    tracker.persisted = []
    tracker.sync_calls = []
    tracker.snapshots = []
    tracker.diff_events = []
    tracker.titles = []
    tracker.current_path = {"value": None}
    tracker.chat_panel = chat_panel
    tracker.status_bar = status_bar
    tracker.workspace = workspace
    tracker.editor = editor
    tracker.bus = DocumentCacheBus()

    def _persist(cache: Any) -> None:
        tracker.persisted.append(cache)

    def _sync_workspace(persist: bool) -> None:
        tracker.sync_calls.append(persist)

    def _set_path(path: Path | None) -> None:
        tracker.current_path["value"] = path

    def _get_path() -> Path | None:
        return tracker.current_path["value"]

    def _record_snapshot(snapshot: dict[str, Any]) -> None:
        tracker.snapshots.append(snapshot)

    def _clear_diff(state: DocumentState) -> None:
        tracker.diff_events.append(state)

    monitor = DocumentStateMonitor(
        editor=editor,
        workspace=workspace,
        chat_panel=chat_panel,
        status_bar=status_bar,
        unsaved_cache_provider=lambda: tracker.cache,
        unsaved_cache_persister=_persist,
        refresh_window_title=lambda state=None: tracker.titles.append(state),
        sync_workspace_state=_sync_workspace,
        current_path_getter=_get_path,
        current_path_setter=_set_path,
        last_snapshot_setter=_record_snapshot,
        active_document_provider=lambda: doc,
        maybe_clear_diff_overlay=_clear_diff,
        window_app_name="App",
        untitled_document_name="Untitled",
        untitled_snapshot_key="__untitled__",
        cache_bus_resolver=lambda: tracker.bus,
    )

    return monitor, tracker


def test_refresh_chat_suggestions_without_selection_context() -> None:
    doc = _make_document(text="Alpha beta")
    monitor, tracker = _make_monitor(document=doc)

    monitor.refresh_chat_suggestions()

    assert tracker.chat_panel.suggestions[:3] == [
        "Summarize the current document.",
        "Suggest improvements to the document structure.",
        "Highlight inconsistencies or missing sections.",
    ]


def test_update_autosave_indicator_tracks_saved_and_autosaved() -> None:
    doc = _make_document(path=Path("notes.md"))
    doc.dirty = False
    monitor, tracker = _make_monitor(document=doc)

    monitor.update_autosave_indicator(document=doc)
    assert tracker.status_bar.states[-1] == ("Saved", "notes.md")

    doc.dirty = True
    monitor.update_autosave_indicator(autosaved=True, document=doc)
    status, detail = tracker.status_bar.states[-1]
    assert status.startswith("Autosaved ")
    assert detail == "notes.md"


def test_handle_editor_snapshot_emits_single_timeline_event() -> None:
    doc = _make_document(path=Path("story.md"))
    monitor, tracker = _make_monitor(document=doc)
    snapshot = {"outline_digest": "abc123", "document_id": doc.document_id}

    monitor.handle_editor_snapshot(snapshot)
    monitor.handle_editor_snapshot(snapshot)

    assert len(tracker.chat_panel.tool_traces) == 1
    trace = tracker.chat_panel.tool_traces[0]
    assert trace.metadata["document_id"] == doc.document_id
    assert trace.metadata["outline_digest"] == "abc123"


def test_handle_editor_text_changed_persists_and_clears_snapshots() -> None:
    doc = _make_document(text="Draft text")
    monitor, tracker = _make_monitor(document=doc)

    monitor.handle_editor_text_changed("", doc)

    assert tracker.cache.untitled_snapshots["tab-1"]["text"] == "Draft text"
    assert tracker.sync_calls == [False]
    assert tracker.persisted[-1] is tracker.cache
    assert tracker.diff_events == [doc]

    doc.dirty = False
    monitor.handle_editor_text_changed("", doc)

    assert "tab-1" not in tracker.cache.untitled_snapshots
    assert tracker.diff_events[-1] is doc  # manual edit hook still invoked


def test_handle_editor_text_changed_publishes_cache_event() -> None:
    doc = _make_document(text="Outline me")
    monitor, tracker = _make_monitor(document=doc)
    events: list[DocumentChangedEvent] = []

    tracker.bus.subscribe(DocumentChangedEvent, lambda event: events.append(event))

    monitor.handle_editor_text_changed("", doc)

    assert len(events) == 1
    assert events[0].document_id == doc.document_id
    assert events[0].version_id == doc.version_id

    # No duplicate when version is unchanged
    monitor.handle_editor_text_changed("", doc)
    assert len(events) == 1

    doc.update_text(doc.text + "!")
    monitor.handle_editor_text_changed("", doc)

    assert len(events) == 2
    assert events[-1].version_id == doc.version_id


def test_handle_active_tab_changed_updates_path_and_title() -> None:
    doc = _make_document(path=Path("chapter.md"))
    doc.dirty = False
    monitor, tracker = _make_monitor(document=doc)
    tab = _StubTab(doc)

    monitor.handle_active_tab_changed(tab)  # type: ignore[arg-type]

    assert tracker.current_path["value"] == Path("chapter.md")
    assert tracker.titles[-1] is doc
    assert tracker.status_bar.states[-1] == ("Saved", "chapter.md")
    assert tracker.chat_panel.suggestions
    assert tracker.sync_calls[-1] is True

    monitor.handle_active_tab_changed(None)
    assert tracker.current_path["value"] is None