"""Tabbed editor container that orchestrates multiple editor instances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

from .document_model import DocumentState, SelectionRange
from .editor_widget import EditorWidget, SnapshotListener, TextChangeListener, QWidgetBase
from .syntax.themes import Theme, load_theme
from .workspace import DocumentTab, DocumentWorkspace
from ..services.bridge import DocumentBridge, Executor

try:  # pragma: no cover - PySide6 optional for tests
    from PySide6.QtWidgets import QTabWidget as _QtQTabWidget, QVBoxLayout as _QtQVBoxLayout

    QTabWidget = _QtQTabWidget
    QVBoxLayout = _QtQVBoxLayout
except Exception:  # pragma: no cover - runtime fallback
    QTabWidget = None
    QVBoxLayout = None


@dataclass(slots=True)
class TabSummary:
    """Lightweight view of a tab's display metadata."""

    tab_id: str
    title: str
    path: str | None
    dirty: bool


TabCreatedListener = Callable[[DocumentTab], None]
SelectionChangeListener = Callable[[str, SelectionRange, int, int], None]


class TabbedEditorWidget(QWidgetBase):
    """Container widget that wraps a :class:`DocumentWorkspace` with UI tabs."""

    def __init__(
        self,
        *,
        workspace: DocumentWorkspace | None = None,
        skip_default_tab: bool = False,
    ) -> None:
        super().__init__()
        self._snapshot_listeners: list[SnapshotListener] = []
        self._text_listeners: list[TextChangeListener] = []
        self._tab_widget: Any | None = None
        self._block_qt_signal = False
        self._initializing = True
        self._editor_lookup: Dict[int, str] = {}
        self._main_thread_executor: Executor | None = None
        self._tab_created_listeners: list[TabCreatedListener] = []
        self._selection_listeners: list[SelectionChangeListener] = []
        self._tab_close_handler: Callable[[str], bool | None] | None = None

        editor_factory = lambda: EditorWidget(parent=self)  # noqa: E731 - intentional lambda factory
        if workspace is None:
            workspace = DocumentWorkspace(editor_factory=editor_factory)
        self._workspace = workspace
        self._workspace.add_active_listener(self._handle_active_tab_changed)
        self._workspace.add_tab_created_listener(self._on_workspace_tab_created)

        self._build_ui()
        existing_tabs = list(self._workspace.iter_tabs())
        if not existing_tabs and not skip_default_tab:
            tab = self._workspace.create_tab()
            existing_tabs = [tab]
        for tab in existing_tabs:
            self._register_tab(tab)
        self._initializing = False
        self._sync_active_tab()

    # ------------------------------------------------------------------
    # Editor-like facade
    # ------------------------------------------------------------------
    def add_snapshot_listener(self, listener: SnapshotListener) -> None:
        self._snapshot_listeners.append(listener)

    def add_text_listener(self, listener: TextChangeListener) -> None:
        self._text_listeners.append(listener)

    def add_selection_listener(self, listener: SelectionChangeListener) -> None:
        self._selection_listeners.append(listener)

    def request_snapshot(self, *, tab_id: str | None = None, delta_only: bool = False) -> dict:
        tab = self._resolve_tab(tab_id)
        snapshot = tab.editor.request_snapshot(delta_only=delta_only)
        return self._augment_snapshot(tab, snapshot)

    def to_document(self, *, tab_id: str | None = None) -> DocumentState:
        return self._resolve_tab(tab_id).document()

    def load_document(self, document: DocumentState, *, tab_id: str | None = None) -> None:
        tab = self._resolve_tab(tab_id)
        tab.editor.load_document(document)
        tab.update_title()
        self._update_tab_label(tab)

    def set_text(self, text: str, *, mark_dirty: bool = True) -> None:
        self._workspace.active_editor().set_text(text, mark_dirty=mark_dirty)

    def insert_text(self, text: str, position: int | None = None) -> None:
        editor = self._workspace.active_editor()
        editor.insert_text(text, position=position)

    def replace_range(self, start: int, end: int, replacement: str) -> None:
        editor = self._workspace.active_editor()
        editor.replace_range(start, end, replacement)

    def set_preview_mode(self, enabled: bool) -> None:
        self._workspace.active_editor().set_preview_mode(enabled)

    def toggle_preview(self) -> None:
        self._workspace.active_editor().toggle_preview()

    def set_all_readonly(self, readonly: bool) -> None:
        """Set all tabs to read-only or editable state.
        
        Used during AI turns to prevent user edits while AI is working.
        """
        for tab in self._workspace.iter_tabs():
            tab.editor.set_readonly(readonly)

    def apply_theme(self, theme: Theme | str | None) -> Theme:
        resolved = load_theme(theme)
        for tab in self._workspace.iter_tabs():
            tab.editor.apply_theme(resolved)
        return resolved

    def show_diff_overlay(
        self,
        diff_text: str,
        *,
        spans: Sequence[tuple[int, int]] | None = None,
        summary: str | None = None,
        source: str | None = None,
        tab_id: str | None = None,
    ) -> None:
        tab = self._resolve_tab(tab_id)
        tab.editor.show_diff_overlay(diff_text, spans=spans, summary=summary, source=source)

    def clear_diff_overlay(self, *, tab_id: str | None = None) -> None:
        tab = self._resolve_tab(tab_id)
        tab.editor.clear_diff_overlay()

    # ------------------------------------------------------------------
    # Workspace helpers
    # ------------------------------------------------------------------
    @property
    def workspace(self) -> DocumentWorkspace:
        return self._workspace

    def active_tab_id(self) -> str | None:
        return self._workspace.active_tab_id

    def active_editor(self) -> EditorWidget:
        return self._workspace.active_editor()

    def active_bridge(self) -> DocumentBridge:
        return self._workspace.active_bridge()

    def tab_summaries(self) -> list[TabSummary]:
        summaries: list[TabSummary] = []
        for tab in self._workspace.iter_tabs():
            document = tab.document()
            summaries.append(
                TabSummary(
                    tab_id=tab.id,
                    title=tab.title,
                    path=str(document.metadata.path) if document.metadata.path else None,
                    dirty=document.dirty,
                )
            )
        return summaries

    def create_tab(
        self,
        *,
        document: DocumentState | None = None,
        path: Path | str | None = None,
        title: str | None = None,
        make_active: bool = True,
        tab_id: str | None = None,
        untitled_index: int | None = None,
    ) -> DocumentTab:
        tab = self._workspace.create_tab(
            document=document,
            path=path,
            title=title,
            make_active=make_active,
            tab_id=tab_id,
            untitled_index=untitled_index,
        )
        # Tab registration is handled by _on_workspace_tab_created listener
        return tab

    def close_tab(self, tab_id: str) -> DocumentTab | None:
        # Try to get tab from workspace - it may not exist if there's state inconsistency
        try:
            tab = self._workspace.get_tab(tab_id)
        except KeyError:
            # Tab not in workspace but may exist in Qt widget - clean up orphaned Qt tab
            self._remove_orphaned_qt_tab(tab_id)
            return None

        self._remove_tab_widget(tab)
        closed = self._workspace.close_tab(tab_id)
        self._editor_lookup.pop(id(tab.editor), None)
        return closed

    def _remove_orphaned_qt_tab(self, tab_id: str) -> None:
        """Remove a Qt tab that has no corresponding workspace entry."""
        if self._tab_widget is None:
            return
        # Find the editor by tab_id in our lookup (reverse lookup)
        editor_id = None
        for eid, tid in list(self._editor_lookup.items()):
            if tid == tab_id:
                editor_id = eid
                break
        if editor_id is not None:
            self._editor_lookup.pop(editor_id, None)
        # Try to find and remove the tab from Qt widget
        for i in range(self._tab_widget.count()):
            widget = self._tab_widget.widget(i)
            if widget is not None and self._tab_id_for_editor(widget) == tab_id:
                self._tab_widget.removeTab(i)
                break

    def close_active_tab(self) -> DocumentTab:
        return self.close_tab(self._workspace.require_active_tab().id)

    def focus_tab(self, tab_id: str) -> DocumentTab:
        tab = self._workspace.set_active_tab(tab_id)
        self._sync_active_tab()
        return tab

    def set_tab_close_handler(self, handler: Callable[[str], bool | None] | None) -> None:
        """Register a callback that can intercept tab-close requests."""

        self._tab_close_handler = handler

    def request_tab_close(self, tab_id: str) -> None:
        """Simulate a user-driven tab close (used by Qt + tests)."""

        if self._tab_close_handler is not None and self._tab_close_handler(tab_id):
            return
        self.close_tab(tab_id)

    def set_main_thread_executor(self, executor: Executor | None) -> None:
        self._main_thread_executor = executor
        for tab in self._workspace.iter_tabs():
            tab.bridge.set_main_thread_executor(executor)

    def add_tab_created_listener(self, listener: TabCreatedListener) -> None:
        self._tab_created_listeners.append(listener)

    def remove_tab_created_listener(self, listener: TabCreatedListener) -> None:
        try:
            self._tab_created_listeners.remove(listener)
        except ValueError:  # pragma: no cover - defensive guard
            pass

    # ------------------------------------------------------------------
    # UI wiring
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        if QTabWidget is None or QVBoxLayout is None:
            return
        self._tab_widget = QTabWidget(self)
        self._tab_widget.setTabsClosable(True)
        self._tab_widget.setMovable(True)
        try:  # pragma: no cover - Qt-specific wiring
            self._tab_widget.currentChanged.connect(self._handle_qt_tab_changed)
            self._tab_widget.tabCloseRequested.connect(self._handle_qt_tab_close_requested)
        except Exception:
            pass
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._tab_widget)

    def _handle_qt_tab_changed(self, index: int) -> None:  # pragma: no cover - Qt specific
        if self._block_qt_signal or self._tab_widget is None:
            return
        editor = self._tab_widget.widget(index)
        if editor is None:
            return
        tab_id = self._tab_id_for_editor(editor)
        if tab_id is None:
            return
        if tab_id not in self._workspace.tab_ids():
            return
        self._workspace.set_active_tab(tab_id)

    def _handle_qt_tab_close_requested(self, index: int) -> None:  # pragma: no cover - Qt specific
        if self._tab_widget is None:
            return
        editor = self._tab_widget.widget(index)
        if editor is None:
            return
        tab_id = self._tab_id_for_editor(editor)
        if tab_id is None:
            return
        self.request_tab_close(tab_id)

    def _sync_active_tab(self) -> None:
        tab = self._workspace.active_tab
        self._handle_active_tab_changed(tab)

    def _on_workspace_tab_created(self, tab: DocumentTab) -> None:
        """Handle a new tab created directly on the workspace."""
        # Avoid double-registration during __init__ which iterates existing tabs
        if self._initializing:
            return
        # Already registered?
        if id(tab.editor) in self._editor_lookup:
            return
        self._register_tab(tab)

    def _register_tab(self, tab: DocumentTab) -> None:
        self._editor_lookup[id(tab.editor)] = tab.id
        self._bind_editor_listeners(tab)
        self._insert_tab_widget(tab)
        if self._main_thread_executor is not None:
            tab.bridge.set_main_thread_executor(self._main_thread_executor)
        for listener in list(self._tab_created_listeners):
            listener(tab)

    def _bind_editor_listeners(self, tab: DocumentTab) -> None:
        tab.editor.add_snapshot_listener(lambda snapshot, tab_id=tab.id: self._emit_snapshot(tab_id, snapshot))
        tab.editor.add_text_listener(lambda text, state, tab_id=tab.id: self._emit_text(tab_id, text, state))
        tab.editor.add_selection_listener(
            lambda selection, line, column, tab_id=tab.id: self._emit_selection(tab_id, selection, line, column)
        )

    def _insert_tab_widget(self, tab: DocumentTab) -> None:
        if self._tab_widget is None:
            return
        index = self._tab_widget.addTab(tab.editor, tab.title)
        if self._workspace.active_tab_id == tab.id:
            self._block_qt_signal = True
            try:
                self._tab_widget.setCurrentIndex(index)
            finally:
                self._block_qt_signal = False

    def _remove_tab_widget(self, tab: DocumentTab) -> None:
        if self._tab_widget is None:
            return
        index = self._tab_widget.indexOf(tab.editor)
        if index >= 0:
            self._tab_widget.removeTab(index)

    def _handle_active_tab_changed(self, tab: DocumentTab | None) -> None:
        if self._initializing:
            return
        if self._tab_widget is not None and tab is not None:
            index = self._tab_widget.indexOf(tab.editor)
            if index >= 0 and self._tab_widget.currentIndex() != index:
                self._block_qt_signal = True
                try:
                    self._tab_widget.setCurrentIndex(index)
                finally:
                    self._block_qt_signal = False
        # Selection changes are no longer broadcast outside the editor widget.

    # ------------------------------------------------------------------
    # Event emitters
    # ------------------------------------------------------------------
    def _emit_snapshot(self, tab_id: str, snapshot: dict[str, Any]) -> None:
        tab = self._workspace.get_tab(tab_id)
        payload = self._augment_snapshot(tab, snapshot)
        for listener in list(self._snapshot_listeners):
            listener(payload)

    def _emit_text(self, tab_id: str, text: str, state: DocumentState) -> None:
        tab = self._workspace.get_tab(tab_id)
        tab.update_title()
        self._update_tab_label(tab)
        for listener in list(self._text_listeners):
            listener(text, state)

    def _emit_selection(self, tab_id: str, selection: SelectionRange, line: int, column: int) -> None:
        payload = SelectionRange(selection.start, selection.end)
        for listener in list(self._selection_listeners):
            listener(tab_id, payload, line, column)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _resolve_tab(self, tab_id: str | None) -> DocumentTab:
        if tab_id is None:
            return self._workspace.require_active_tab()
        return self._workspace.get_tab(tab_id)

    def _update_tab_label(self, tab: DocumentTab) -> None:
        if self._tab_widget is None:
            return
        index = self._tab_widget.indexOf(tab.editor)
        if index >= 0:
            self._tab_widget.setTabText(index, tab.title)

    def _tab_id_for_editor(self, editor: Any) -> str | None:
        return self._editor_lookup.get(id(editor))

    def _augment_snapshot(self, tab: DocumentTab, snapshot: dict[str, Any]) -> dict[str, Any]:
        payload = dict(snapshot)
        payload.setdefault("tab_id", tab.id)
        payload.setdefault("tab_title", tab.title)
        payload.setdefault("open_tabs", self._workspace.serialize_tabs())
        payload.setdefault("active_tab_id", self._workspace.active_tab_id)
        return payload

    def refresh_tab_title(self, tab_id: str | None = None) -> None:
        tab = self._resolve_tab(tab_id)
        tab.update_title()
        self._update_tab_label(tab)