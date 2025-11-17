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


class ContextUsageWidget:
    """Helper component that displays rolling telemetry details."""

    def __init__(self) -> None:
        self.summary_text: str = ""
        self.totals_text: str = ""
        self.last_tool: str = ""
        self._summary_label: Any = None
        self._details_label: Any = None

    def install(self, status_bar: Any | None) -> None:
        if status_bar is None or QLabel is None:
            return
        self._summary_label = QLabel(self.summary_text)
        self._summary_label.setObjectName("tb-status-memory-summary")
        self._summary_label.setContentsMargins(8, 0, 8, 0)
        self._details_label = QLabel(self._details_text())
        self._details_label.setObjectName("tb-status-memory-details")
        self._details_label.setContentsMargins(0, 0, 8, 0)
        try:
            status_bar.addPermanentWidget(self._summary_label)
            status_bar.addPermanentWidget(self._details_label)
        except Exception:
            self._summary_label = None
            self._details_label = None

    def update(self, summary: str, totals: str | None = None, last_tool: str | None = None) -> None:
        self.summary_text = summary.strip()
        self.totals_text = (totals or "").strip()
        self.last_tool = (last_tool or "").strip()
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        if self._summary_label is not None:
            try:
                self._summary_label.setText(self.summary_text)
            except Exception:
                pass
        if self._details_label is not None:
            try:
                self._details_label.setText(self._details_text())
            except Exception:
                pass

    def _details_text(self) -> str:
        parts: list[str] = []
        if self.totals_text:
            parts.append(self.totals_text)
        if self.last_tool:
            parts.append(f"Last tool {self.last_tool}")
        return " · ".join(parts)


class StatusBar:
    """Status bar that mirrors the plan.md contract yet stays test friendly."""

    def __init__(self, parent: Any | None = None) -> None:
        self._message: str = ""
        self._message_timeout: Optional[int] = None
        self._cursor: tuple[int, int] = (1, 1)
        self._document_format: str = "plain"
        self._ai_state: str = "Idle"
        self._memory_usage: str = ""
        self._autosave_state: str = "Saved"
        self._autosave_detail: str = ""
        self._outline_status: str = ""
        self._outline_detail: str = ""
        self._embedding_status: str = ""
        self._embedding_detail: str = ""
        self._subagent_status: str = ""
        self._subagent_detail: str = ""
        self._context_widget = ContextUsageWidget()

        self._qt_bar = self._build_qt_status_bar(parent)
        self._cursor_label: Any = None
        self._format_label: Any = None
        self._ai_label: Any = None
        self._outline_label: Any = None
        self._embedding_label: Any = None
        self._autosave_label: Any = None
        self._subagent_label: Any = None

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

    def set_memory_usage(self, usage: str, *, totals: str | None = None, last_tool: str | None = None) -> None:
        """Display the latest memory usage summary (e.g., autosave + tokens)."""

        self._memory_usage = usage.strip()
        self._context_widget.update(self._memory_usage, totals, last_tool)

    def set_autosave_state(self, state: str, *, detail: str | None = None) -> None:
        """Update the autosave indicator text."""

        normalized = state.strip() if state else "Saved"
        self._autosave_state = normalized or "Saved"
        self._autosave_detail = (detail or "").strip()
        self._update_label(self._autosave_label, self._format_autosave_text())

    def set_outline_status(self, status: str | None, *, tooltip: str | None = None) -> None:
        """Update the outline freshness indicator."""

        self._outline_status = (status or "").strip()
        self._outline_detail = (tooltip or "").strip()
        if self._outline_label is not None:
            self._update_label(self._outline_label, self._format_outline_text())
            try:
                self._outline_label.setToolTip(self._outline_detail)
            except Exception:
                pass

    def set_embedding_status(self, status: str | None, *, detail: str | None = None) -> None:
        """Display the active embedding backend/provider status."""

        self._embedding_status = (status or "").strip()
        self._embedding_detail = (detail or "").strip()
        if self._embedding_label is not None:
            self._update_label(self._embedding_label, self._format_embedding_text())
            try:
                self._embedding_label.setToolTip(self._embedding_detail)
            except Exception:
                pass

    def set_subagent_status(self, status: str | None, *, detail: str | None = None) -> None:
        """Show whether the subagent sandbox is active plus optional detail text."""

        self._subagent_status = (status or "").strip()
        self._subagent_detail = (detail or "").strip()
        if self._subagent_label is not None:
            self._update_label(self._subagent_label, self._format_subagent_text())
            try:
                self._subagent_label.setToolTip(self._subagent_detail)
            except Exception:
                pass

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

    @property
    def context_widget(self) -> ContextUsageWidget:
        return self._context_widget

    @property
    def autosave_state(self) -> tuple[str, str]:
        return (self._autosave_state, self._autosave_detail)

    @property
    def outline_state(self) -> tuple[str, str]:
        return (self._outline_status, self._outline_detail)

    @property
    def embedding_state(self) -> tuple[str, str]:
        return (self._embedding_status, self._embedding_detail)

    @property
    def subagent_state(self) -> tuple[str, str]:
        return (self._subagent_status, self._subagent_detail)

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
        self._outline_label = QLabel(self._format_outline_text())
        self._outline_label.setObjectName("tb-status-outline")
        self._embedding_label = QLabel(self._format_embedding_text())
        self._embedding_label.setObjectName("tb-status-embedding")
        self._autosave_label = QLabel(self._format_autosave_text())
        self._autosave_label.setObjectName("tb-status-autosave")
        self._subagent_label = QLabel(self._format_subagent_text())
        self._subagent_label.setObjectName("tb-status-subagents")

        for label in (
            self._cursor_label,
            self._format_label,
            self._ai_label,
            self._outline_label,
            self._embedding_label,
            self._autosave_label,
            self._subagent_label,
        ):
            label.setContentsMargins(8, 0, 8, 0)
            try:
                self._qt_bar.addPermanentWidget(label)
            except Exception:
                break

        self._context_widget.install(self._qt_bar)

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

    def _format_autosave_text(self) -> str:
        detail = self._autosave_detail
        base = f"Autosave: {self._autosave_state}"
        return f"{base} · {detail}" if detail else base

    def _format_outline_text(self) -> str:
        return f"Outline: {self._outline_status}" if self._outline_status else ""

    def _format_embedding_text(self) -> str:
        return f"Embeddings: {self._embedding_status}" if self._embedding_status else ""

    def _format_subagent_text(self) -> str:
        return f"Subagents: {self._subagent_status}" if self._subagent_status else ""

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


__all__ = ["StatusBar", "ContextUsageWidget"]

