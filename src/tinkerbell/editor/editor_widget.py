"""Editor widget implementation with Qt + headless fallbacks.

The widget intentionally keeps business logic (documents, selections, preview
generation) decoupled from the Qt presentation layer so tests can run in
headless environments. When PySide6 is available and a ``QApplication`` has
been instantiated, the widget wires a ``QPlainTextEdit`` + preview container;
otherwise an in-memory buffer is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, Sequence

from ..chat.message_model import EditDirective
from .document_model import DocumentState, SelectionRange
from .patches import PatchResult
from .syntax.markdown import MarkdownPreview, render_preview
from .syntax.themes import Theme, load_theme
Qt: Any = None
QTextCursor: Any = None
QApplication: Any = None
QLabel: Any = None
QPlainTextEdit: Any = None
QTextEdit: Any = None
QStackedWidget: Any = None
QVBoxLayout: Any = None
QWidgetBase: Any = None
QSizePolicy: Any = None
QColor: Any = None

try:  # pragma: no cover - PySide6 optional in CI
    from PySide6.QtCore import Qt as _QtCoreQt  # noqa: F401  (exported for future use)
    from PySide6.QtGui import QTextCursor as _QtTextCursor  # type: ignore[import-not-found]
    from PySide6.QtWidgets import (
        QApplication as _QtApplication,
        QLabel as _QtLabel,
        QPlainTextEdit as _QtPlainTextEdit,
        QTextEdit as _QtTextEdit,
        QSizePolicy as _QtSizePolicy,
        QStackedWidget as _QtStackedWidget,
        QVBoxLayout as _QtVBoxLayout,
        QWidget as _QtWidget,
    )
    from PySide6.QtGui import QColor as _QtColor

    Qt = _QtCoreQt
    QTextCursor = _QtTextCursor
    QApplication = _QtApplication
    QLabel = _QtLabel
    QPlainTextEdit = _QtPlainTextEdit
    QTextEdit = _QtTextEdit
    QStackedWidget = _QtStackedWidget
    QVBoxLayout = _QtVBoxLayout
    QWidgetBase = _QtWidget
    QSizePolicy = _QtSizePolicy
    QColor = _QtColor
except Exception:  # pragma: no cover - runtime fallback

    class _StubQWidget:  # type: ignore[too-many-ancestors]
        """Runtime fallback avoiding PySide6 dependency during tests."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401 - shim
            del args, kwargs

    QWidgetBase = _StubQWidget
    QTextEdit = None
    QColor = None


class SnapshotListener(Protocol):
    """Protocol for callbacks interested in document snapshots."""

    def __call__(self, snapshot: dict) -> None:
        ...


class TextChangeListener(Protocol):
    """Callback signature invoked when the editor text changes."""

    def __call__(self, text: str, state: DocumentState) -> None:
        ...


class SelectionListener(Protocol):
    """Callback signature invoked when the selection changes."""

    def __call__(self, selection: SelectionRange) -> None:
        ...


@dataclass(slots=True)
class _UndoEntry:
    """Represents a text snapshot for undo/redo bookkeeping."""

    text: str
    selection: SelectionRange


@dataclass(slots=True)
class DiffOverlayState:
    """Metadata describing a temporary diff highlight inside the editor."""

    diff: str
    spans: tuple[tuple[int, int], ...] = ()
    summary: str | None = None
    source: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class EditorWidget(QWidgetBase):
    """High-level widget orchestrating the text editor component."""

    MAX_HISTORY = 50

    def __init__(self, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._state = DocumentState()
        self._text_buffer: str = ""
        self._theme: Theme = load_theme()
        self._qt_editor: Any = None
        self._stack: Any = None
        self._preview_widget: Any = None
        self._preview_enabled = False
        self._preview_cache: Optional[MarkdownPreview] = None
        self._snapshot_listeners: list[SnapshotListener] = []
        self._text_listeners: list[TextChangeListener] = []
        self._selection_listeners: list[SelectionListener] = []
        self._undo_stack: list[_UndoEntry] = []
        self._redo_stack: list[_UndoEntry] = []
        self._diff_overlay: DiffOverlayState | None = None
        self._overlay_brush: Any | None = None
        self._overlay_text_brush: Any | None = None
        self._last_change_source: str = "init"

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction & helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Instantiate Qt widgets when a QApplication is available."""

        if QApplication is None:
            return
        if QPlainTextEdit is None or QStackedWidget is None or QVBoxLayout is None or QLabel is None:
            return
        try:
            if QApplication.instance() is None:
                # Headless mode – we still keep the logical bits working.
                return
        except Exception:  # pragma: no cover - defensive guard
            return

        self._qt_editor = QPlainTextEdit(self)
        self._qt_editor.textChanged.connect(self._handle_qt_text_changed)  # type: ignore[attr-defined]
        self._qt_editor.cursorPositionChanged.connect(  # type: ignore[attr-defined]
            self._handle_qt_selection_changed
        )

        self._preview_widget = QLabel(self)
        self._preview_widget.setObjectName("previewLabel")  # type: ignore[attr-defined]
        self._preview_widget.setWordWrap(True)  # type: ignore[attr-defined]
        if QSizePolicy is not None:
            try:
                policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                self._preview_widget.setSizePolicy(policy)
            except Exception:  # pragma: no cover - defensive guard
                pass

        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._qt_editor)
        self._stack.addWidget(self._preview_widget)

        layout = QVBoxLayout(self)
        layout.addWidget(self._stack)

    # ------------------------------------------------------------------
    # Document accessors
    # ------------------------------------------------------------------
    def load_document(self, document: DocumentState) -> None:
        """Load a new document state into the widget."""

        self._state = document
        self._text_buffer = document.text
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.clear_diff_overlay()
        if self._qt_editor is not None:
            self._qt_editor.blockSignals(True)
            self._qt_editor.setPlainText(document.text)
            self._qt_editor.blockSignals(False)
        self._refresh_preview(if_enabled=False)
        self._emit_text_changed()
        self._emit_selection_changed()
        self._mark_change_source("programmatic")

    def to_document(self) -> DocumentState:
        """Return the current document representation."""

        self._state.text = self._text_buffer
        return self._state

    def apply_selection(self, selection: SelectionRange) -> None:
        """Apply an external selection update to the widget."""

        clamped = self._clamp_range(selection.start, selection.end)
        self._state.selection = SelectionRange(*clamped)
        if self._qt_editor is not None and QTextCursor is not None:
            cursor = self._qt_editor.textCursor()
            cursor.setPosition(clamped[0])
            cursor.setPosition(clamped[1], QTextCursor.KeepAnchor)  # type: ignore[attr-defined]
            self._qt_editor.setTextCursor(cursor)
        self._emit_selection_changed()

    # ------------------------------------------------------------------
    # Editing helpers
    # ------------------------------------------------------------------
    def set_text(self, text: str, *, mark_dirty: bool = True) -> None:
        """Replace the entire document content with ``text``."""

        previous = self._text_buffer
        if previous == text:
            return
        self._push_undo_snapshot(previous)
        self._text_buffer = text
        if mark_dirty:
            self._state.update_text(text)
        else:
            self._state.text = text
        if self._qt_editor is not None:
            self._qt_editor.blockSignals(True)
            self._qt_editor.setPlainText(text)
            self._qt_editor.blockSignals(False)
        self._refresh_preview(if_enabled=True)
        self._emit_text_changed()
        self._mark_change_source("programmatic")

    def insert_text(self, text: str, position: Optional[int] = None) -> None:
        """Insert ``text`` at ``position`` or current selection start."""

        start = position if position is not None else self._state.selection.start
        start = max(0, min(start, len(self._text_buffer)))
        new_text = self._text_buffer[:start] + text + self._text_buffer[start:]
        selection = SelectionRange(start, start + len(text))
        self.set_text(new_text)
        self.apply_selection(selection)

    def replace_range(self, start: int, end: int, replacement: str) -> None:
        """Replace the slice ``[start:end]`` with ``replacement``."""

        begin, finish = self._clamp_range(start, end)
        new_text = self._text_buffer[:begin] + replacement + self._text_buffer[finish:]
        self.set_text(new_text)
        self.apply_selection(SelectionRange(begin, begin + len(replacement)))

    def apply_ai_edit(self, directive: EditDirective) -> DocumentState:
        """Apply an agent-issued edit directive (insert/replace/annotate)."""

        action = directive.action.lower()
        start, end = self._clamp_range(*directive.target_range)
        if action == "insert":
            self.insert_text(directive.content, position=start)
        elif action == "replace":
            if start == end:
                raise ValueError("Replace directives must target a non-empty range; use action='insert' for caret edits")
            self.replace_range(start, end, directive.content)
        elif action == "annotate":
            annotation = f"\n[AI Note]: {directive.content.strip()}\n"
            insert_at = end if end > start else len(self._text_buffer)
            self.insert_text(annotation, position=insert_at)
        else:
            raise ValueError(f"Unsupported directive action: {directive.action}")
        self._mark_change_source("programmatic")
        return self.to_document()

    def apply_patch_result(self, result: PatchResult, selection_hint: tuple[int, int] | None = None) -> DocumentState:
        """Apply a diff-based patch result while emitting a single undo snapshot."""

        previous = self._text_buffer
        if previous == result.text:
            return self.to_document()

        self._push_undo_snapshot(previous)
        self._text_buffer = result.text
        self._state.update_text(result.text)
        if self._qt_editor is not None:
            self._qt_editor.blockSignals(True)
            self._qt_editor.setPlainText(result.text)
            self._qt_editor.blockSignals(False)
        self._refresh_preview(if_enabled=True)
        self._emit_text_changed()

        spans = result.spans or ()
        if spans:
            start, end = spans[-1]
        elif selection_hint is not None:
            start, end = selection_hint
        else:
            start = end = len(result.text)

        self.apply_selection(SelectionRange(start, end))
        self._mark_change_source("programmatic")
        return self.to_document()

    # ------------------------------------------------------------------
    # Snapshot + listeners
    # ------------------------------------------------------------------
    def request_snapshot(self, delta_only: bool = False) -> dict:
        """Notify listeners with the current document snapshot."""

        snapshot = self._state.snapshot(delta_only=delta_only)
        snapshot["line_count"] = self.line_count
        snapshot["preview_enabled"] = self._preview_enabled
        if self._preview_enabled and self._preview_cache is not None:
            snapshot["preview"] = self._preview_cache.html
        for listener in list(self._snapshot_listeners):
            listener(snapshot)
        return snapshot

    def add_snapshot_listener(self, listener: SnapshotListener) -> None:
        """Register a callback invoked whenever a snapshot is requested."""

        self._snapshot_listeners.append(listener)

    def add_text_listener(self, listener: TextChangeListener) -> None:
        """Register a callback fired whenever the text buffer mutates."""

        self._text_listeners.append(listener)

    def add_selection_listener(self, listener: SelectionListener) -> None:
        """Register a callback fired whenever the selection range changes."""

        self._selection_listeners.append(listener)

    # ------------------------------------------------------------------
    # Preview & theming
    # ------------------------------------------------------------------
    def set_preview_mode(self, enabled: bool) -> None:
        """Enable or disable the Markdown preview pane."""

        if self._preview_enabled == enabled:
            return
        self._preview_enabled = enabled
        self._refresh_preview(if_enabled=True)
        if self._stack is not None:
            index = 1 if enabled else 0
            try:
                self._stack.setCurrentIndex(index)
            except Exception:  # pragma: no cover - defensive guard
                pass

    def toggle_preview(self) -> None:
        """Invert the preview toggle state."""

        self.set_preview_mode(not self._preview_enabled)

    def apply_theme(self, theme: Theme | str | None) -> Theme:
        """Load and apply a theme, returning the resolved :class:`Theme`."""

        self._theme = load_theme(theme)
        self._overlay_brush = None
        self._overlay_text_brush = None
        # Re-render preview markup so CSS colors stay in sync with the palette.
        self._refresh_preview(if_enabled=False)
        return self._theme

    # ------------------------------------------------------------------
    # Undo/redo support (headless-friendly)
    # ------------------------------------------------------------------
    def undo(self) -> None:
        """Restore the previous text snapshot if available."""

        if not self._undo_stack:
            return
        entry = self._undo_stack.pop()
        self._redo_stack.append(_UndoEntry(text=self._text_buffer, selection=self._state.selection))
        self._text_buffer = entry.text
        self._state.text = entry.text
        self.apply_selection(entry.selection)
        if self._qt_editor is not None:
            self._qt_editor.blockSignals(True)
            self._qt_editor.setPlainText(entry.text)
            self._qt_editor.blockSignals(False)
        self._emit_text_changed()

    def redo(self) -> None:
        """Reapply an undone text snapshot if available."""

        if not self._redo_stack:
            return
        entry = self._redo_stack.pop()
        self._undo_stack.append(_UndoEntry(text=self._text_buffer, selection=self._state.selection))
        self._text_buffer = entry.text
        self._state.text = entry.text
        self.apply_selection(entry.selection)
        if self._qt_editor is not None:
            self._qt_editor.blockSignals(True)
            self._qt_editor.setPlainText(entry.text)
            self._qt_editor.blockSignals(False)
        self._emit_text_changed()

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def line_count(self) -> int:
        """Return the current line count for status/preview metadata."""

        if not self._text_buffer:
            return 0
        return self._text_buffer.count("\n") + 1

    @property
    def preview_enabled(self) -> bool:
        """Expose the preview toggle state."""

        return self._preview_enabled

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _clamp_range(self, start: int, end: int) -> tuple[int, int]:
        length = len(self._text_buffer)
        start = max(0, min(start, length))
        end = max(0, min(end, length))
        if end < start:
            start, end = end, start
        return start, end

    def _push_undo_snapshot(self, previous_text: str) -> None:
        entry = _UndoEntry(text=previous_text, selection=self._state.selection)
        self._undo_stack.append(entry)
        if len(self._undo_stack) > self.MAX_HISTORY:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _refresh_preview(self, *, if_enabled: bool) -> None:
        should_render = self._preview_enabled or not if_enabled
        if should_render:
            self._preview_cache = render_preview(self._text_buffer, theme=self._theme)
        self._sync_preview_widget()

    def _sync_preview_widget(self) -> None:
        if self._preview_widget is None:
            return
        if not self._preview_enabled:
            try:
                self._preview_widget.clear()
            except Exception:  # pragma: no cover - defensive guard
                pass
            return
        if self._preview_cache is None:
            return
        try:
            self._preview_widget.setText(self._preview_cache.html)
        except Exception:  # pragma: no cover - defensive guard
            pass

    def _emit_text_changed(self) -> None:
        for listener in list(self._text_listeners):
            listener(self._text_buffer, self._state)

    def _emit_selection_changed(self) -> None:
        for listener in list(self._selection_listeners):
            listener(self._state.selection)

    # Qt callbacks -----------------------------------------------------
    def _handle_qt_text_changed(self) -> None:
        if self._qt_editor is None:
            return
        self._text_buffer = self._qt_editor.toPlainText()
        self._state.update_text(self._text_buffer)
        self._refresh_preview(if_enabled=True)
        self._emit_text_changed()
        self._mark_change_source("user")

    def _handle_qt_selection_changed(self) -> None:
        if self._qt_editor is None or QTextCursor is None:
            return
        cursor = self._qt_editor.textCursor()
        selection = SelectionRange(start=cursor.selectionStart(), end=cursor.selectionEnd())
        self._state.selection = selection
        self._emit_selection_changed()

    # ------------------------------------------------------------------
    # Diff overlay helpers
    # ------------------------------------------------------------------
    def show_diff_overlay(
        self,
        diff_text: str,
        *,
        spans: Sequence[tuple[int, int]] | None = None,
        summary: str | None = None,
        source: str | None = None,
    ) -> DiffOverlayState:
        normalized_spans = self._normalize_overlay_spans(spans)
        state = DiffOverlayState(
            diff=str(diff_text or ""),
            spans=normalized_spans,
            summary=summary,
            source=source,
        )
        self._diff_overlay = state
        self._apply_overlay_highlight()
        return state

    def clear_diff_overlay(self) -> None:
        if self._diff_overlay is None:
            return
        self._diff_overlay = None
        self._apply_overlay_highlight()

    @property
    def diff_overlay(self) -> DiffOverlayState | None:
        return self._diff_overlay

    @property
    def last_change_source(self) -> str:
        return self._last_change_source

    # ------------------------------------------------------------------
    # Overlay + change tracking internals
    # ------------------------------------------------------------------
    def _normalize_overlay_spans(self, spans: Sequence[tuple[int, int]] | None) -> tuple[tuple[int, int], ...]:
        if not spans:
            return ()
        normalized: list[tuple[int, int]] = []
        for start, end in spans:
            begin, finish = self._clamp_range(start, end)
            if begin == finish:
                continue
            normalized.append((begin, finish))
        return tuple(normalized)

    def _apply_overlay_highlight(self) -> None:
        if self._qt_editor is None or QTextCursor is None or QTextEdit is None:
            return
        overlay = self._diff_overlay
        selection_cls = getattr(QTextEdit, "ExtraSelection", None)
        if selection_cls is None:
            return
        selections: list[Any] = []
        if overlay and overlay.spans:
            for start, end in overlay.spans:
                cursor = self._qt_editor.textCursor()
                cursor.setPosition(start)
                cursor.setPosition(end, QTextCursor.KeepAnchor)
                selection = selection_cls()
                selection.cursor = cursor
                format_obj = selection.format
                color = self._overlay_color()
                text_color = self._overlay_text_color()
                if color is not None:
                    try:
                        format_obj.setBackground(color)
                    except Exception:
                        pass
                if text_color is not None:
                    try:
                        format_obj.setForeground(text_color)
                    except Exception:
                        pass
                selections.append(selection)
        try:
            self._qt_editor.setExtraSelections(selections)
        except Exception:  # pragma: no cover - defensive guard
            pass

    def _overlay_color(self) -> Any | None:
        if QColor is None:
            return None
        if self._overlay_brush is not None:
            return self._overlay_brush
        rgb = self._theme.color("diff_highlight", (255, 243, 196))
        try:
            self._overlay_brush = QColor(*rgb)
        except Exception:  # pragma: no cover - defensive guard
            self._overlay_brush = None
        return self._overlay_brush

    def _mark_change_source(self, source: str) -> None:
        self._last_change_source = source

    def _overlay_text_color(self) -> Any | None:
        if QColor is None:
            return None
        if self._overlay_text_brush is not None:
            return self._overlay_text_brush
        rgb = self._theme.color("diff_highlight_foreground", (32, 33, 36))
        try:
            self._overlay_text_brush = QColor(*rgb)
        except Exception:  # pragma: no cover - defensive guard
            self._overlay_text_brush = None
        return self._overlay_text_brush


# Syntax package -----------------------------------------------------------------
