"""Custom status bar implementation with optional Qt widgets."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

try:  # pragma: no cover - Qt imports are optional during tests
    from PySide6.QtWidgets import QApplication, QLabel, QStatusBar
except Exception:  # pragma: no cover - PySide6 not available
    QApplication = None  # type: ignore[assignment]
    QLabel = None  # type: ignore[assignment]
    QStatusBar = None  # type: ignore[assignment]


class StatusBar:
    """Status bar that mirrors the plan.md contract yet stays test friendly."""

    def __init__(self, parent: Any | None = None) -> None:
        self._message: str = ""
        self._message_timeout: Optional[int] = None
        self._cursor: tuple[int, int] = (1, 1)
        self._document_format: str = "plain"
        self._ai_state: str = "Idle"
        self._memory_usage: str = ""

        self._qt_bar = self._build_qt_status_bar(parent)
        self._cursor_label: Any = None
        self._format_label: Any = None
        self._ai_label: Any = None
        self._memory_label: Any = None

        if self._qt_bar is not None:
            self._init_widgets()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_message(self, message: str, *, timeout_ms: Optional[int] = None) -> None:
        """Show a primary status message, honoring optional timeouts."""

        self._message = message
        self._message_timeout = timeout_ms
        if self._qt_bar is not None:
            timeout = timeout_ms or 0
            try:
                self._qt_bar.showMessage(message, timeout)
            except Exception:
                pass

    def clear_message(self) -> None:
        """Clear the current status message."""

        self._message = ""
        self._message_timeout = None
        if self._qt_bar is not None:
            try:
                self._qt_bar.clearMessage()
            except Exception:
                pass

    def update_cursor(self, line: int, column: int) -> None:
        """Update the caret position indicator."""

        self._cursor = (max(1, line), max(1, column))
        self._update_label(self._cursor_label, self._format_cursor_text())

    def set_document_format(self, label: str) -> None:
        """Display the inferred document format (markdown/json/etc)."""

        self._document_format = label.strip() or "plain"
        self._update_label(self._format_label, self._document_format.upper())

    def set_ai_state(self, state: str | Enum) -> None:
        """Reflect the current AI controller state (Idle, Thinking, Streaming)."""

        self._ai_state = self._coerce_state(state)
        self._update_label(self._ai_label, self._ai_state)

    def set_memory_usage(self, usage: str) -> None:
        """Display the latest memory usage summary (e.g., autosave + tokens)."""

        self._memory_usage = usage.strip()
        self._update_label(self._memory_label, self._memory_usage)

    def widget(self) -> Any | None:
        """Return the underlying :class:`QStatusBar` when available."""

        return self._qt_bar

    # ------------------------------------------------------------------
    # Introspection helpers (handy for tests)
    # ------------------------------------------------------------------
    @property
    def message(self) -> str:
        return self._message

    @property
    def cursor_position(self) -> tuple[int, int]:
        return self._cursor

    @property
    def document_format(self) -> str:
        return self._document_format

    @property
    def ai_state(self) -> str:
        return self._ai_state

    @property
    def memory_usage(self) -> str:
        return self._memory_usage

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_widgets(self) -> None:
        if self._qt_bar is None or QLabel is None:
            return

        self._cursor_label = QLabel(self._format_cursor_text())
        self._cursor_label.setObjectName("tb-status-cursor")
        self._format_label = QLabel(self._document_format.upper())
        self._format_label.setObjectName("tb-status-format")
        self._ai_label = QLabel(self._ai_state)
        self._ai_label.setObjectName("tb-status-ai")
        self._memory_label = QLabel(self._memory_usage)
        self._memory_label.setObjectName("tb-status-memory")

        for label in (self._cursor_label, self._format_label, self._ai_label, self._memory_label):
            label.setContentsMargins(8, 0, 8, 0)
            try:
                self._qt_bar.addPermanentWidget(label)
            except Exception:
                break

    def _update_label(self, label: Any, text: str) -> None:
        if label is None:
            return
        try:
            label.setText(text)
        except Exception:
            pass

    def _format_cursor_text(self) -> str:
        line, column = self._cursor
        return f"Ln {line}, Col {column}"

    @staticmethod
    def _coerce_state(state: str | Enum) -> str:
        if isinstance(state, Enum):
            return str(getattr(state, "value", state.name))
        return str(state).strip() or "Idle"

    def _handle_qt_message_changed(self, text: str) -> None:
        self._message = text
        if not text:
            self._message_timeout = None

    def _build_qt_status_bar(self, parent: Any | None) -> Any | None:
        if QStatusBar is None or QApplication is None:
            return None
        try:
            if QApplication.instance() is None:
                return None
        except Exception:
            return None

        try:
            bar = QStatusBar(parent)
        except Exception:
            return None

        try:
            bar.setObjectName("tb-status-bar")
            bar.messageChanged.connect(self._handle_qt_message_changed)
        except Exception:
            pass
        return bar

