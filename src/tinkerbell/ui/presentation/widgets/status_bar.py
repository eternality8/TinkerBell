"""Custom status bar implementation with optional Qt widgets."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Callable

try:  # pragma: no cover - Qt imports are optional during tests
    from PySide6.QtWidgets import QApplication, QLabel, QStatusBar
    from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget
except Exception:  # pragma: no cover - PySide6 not available
    QApplication = None  # type: ignore[assignment]
    QLabel = None  # type: ignore[assignment]
    QWidget = None  # type: ignore[assignment]
    QPushButton = None  # type: ignore[assignment]
    QHBoxLayout = None  # type: ignore[assignment]
    QStatusBar = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


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
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
        if self._details_label is not None:
            try:
                self._details_label.setText(self._details_text())
            except Exception:  # pragma: no cover - Qt defensive guard
                pass

    def _details_text(self) -> str:
        parts: list[str] = []
        if self.totals_text:
            parts.append(self.totals_text)
        if self.last_tool:
            parts.append(f"Last tool {self.last_tool}")
        return " · ".join(parts)


class DiffReviewControls:
    """Inline accept/reject controls rendered inside the status bar."""

    def __init__(self) -> None:
        self.summary_text: str = ""
        self.accept_callback: Callable[[], None] | None = None
        self.reject_callback: Callable[[], None] | None = None
        self._status_bar: Any | None = None
        self._container: Any | None = None
        self._summary_label: Any | None = None
        self._accept_button: Any | None = None
        self._reject_button: Any | None = None
        self._visible = False

    def install(self, status_bar: Any | None) -> None:
        LOGGER.debug("DiffReviewControls.install: status_bar=%s", status_bar is not None)
        self._status_bar = status_bar
        if (
            status_bar is None
            or QWidget is None
            or QLabel is None
            or QPushButton is None
            or QHBoxLayout is None
        ):
            LOGGER.debug("DiffReviewControls.install: missing Qt widgets, skipping")
            return
        if self._container is not None:
            LOGGER.debug("DiffReviewControls.install: already installed")
            return
        try:
            from PySide6.QtWidgets import QSizePolicy
        except Exception:  # pragma: no cover - optional during tests
            QSizePolicy = None
        try:
            container = QWidget(status_bar)
            layout = QHBoxLayout(container)
            layout.setContentsMargins(4, 0, 4, 0)
            layout.setSpacing(6)
            summary_label = QLabel(self.summary_text or "")
            summary_label.setObjectName("tb-status-review-summary")
            summary_label.setContentsMargins(4, 0, 4, 0)
            accept_button = QPushButton("✓ Accept")
            accept_button.setObjectName("tb-status-review-accept")
            reject_button = QPushButton("✗ Reject")
            reject_button.setObjectName("tb-status-review-reject")
            # Set minimum sizes to ensure buttons are visible
            accept_button.setMinimumWidth(70)
            reject_button.setMinimumWidth(70)
            layout.addWidget(summary_label)
            layout.addWidget(accept_button)
            layout.addWidget(reject_button)
            try:
                accept_button.clicked.connect(self.trigger_accept)  # type: ignore[attr-defined]
                reject_button.clicked.connect(self.trigger_reject)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
            # Do NOT apply Ignored policy - we want the controls to be visible
            # The container should take its natural size from the buttons
            status_bar.addPermanentWidget(container)
            container.setVisible(False)
            LOGGER.debug("DiffReviewControls.install: container created successfully")
        except Exception:
            LOGGER.debug("DiffReviewControls.install: failed to create container", exc_info=True)
            return

        self._container = container
        self._summary_label = summary_label
        self._accept_button = accept_button
        self._reject_button = reject_button
        self._update_summary_label()
        self._set_visible(self._visible)

    def set_state(
        self,
        summary: str | None,
        *,
        accept_callback: Callable[[], None] | None = None,
        reject_callback: Callable[[], None] | None = None,
    ) -> None:
        normalized = (summary or "").strip()
        LOGGER.debug(
            "DiffReviewControls.set_state: summary=%r container=%s",
            normalized,
            self._container is not None,
        )
        self.summary_text = normalized
        self.accept_callback = accept_callback
        self.reject_callback = reject_callback
        self._update_summary_label()
        self._set_visible(bool(normalized))

    def clear(self) -> None:
        self.set_state(None)

    def trigger_accept(self) -> None:
        if self.accept_callback is not None:
            try:
                self.accept_callback()
            except Exception:  # pragma: no cover - Qt defensive guard
                pass

    def trigger_reject(self) -> None:
        if self.reject_callback is not None:
            try:
                self.reject_callback()
            except Exception:  # pragma: no cover - Qt defensive guard
                pass

    def _update_summary_label(self) -> None:
        if self._summary_label is None:
            return
        try:
            self._summary_label.setText(self.summary_text)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _set_visible(self, visible: bool) -> None:
        LOGGER.debug(
            "DiffReviewControls._set_visible: visible=%s container=%s",
            visible,
            self._container is not None,
        )
        self._visible = visible
        if self._container is None:
            LOGGER.debug("DiffReviewControls._set_visible: no container, skipping")
            return
        try:
            self._container.setVisible(visible)
            LOGGER.debug("DiffReviewControls._set_visible: setVisible(%s) called", visible)
        except Exception:
            LOGGER.debug("DiffReviewControls._set_visible: setVisible failed", exc_info=True)

    @property
    def visible(self) -> bool:
        return self._visible


class DocumentStatusIndicator:
    """Clickable badge reflecting document readiness."""

    def __init__(self) -> None:
        self._button: Any | None = None
        self._callback: Callable[[], None] | None = None
        self._text: str = ""
        self._detail: str = ""
        self._severity: str = ""

    def install(self, status_bar: Any | None) -> None:
        if status_bar is None or QPushButton is None:
            return
        if self._button is not None:
            return
        try:
            from PySide6.QtWidgets import QSizePolicy
        except Exception:  # pragma: no cover - optional during tests
            QSizePolicy = None
        try:
            button = QPushButton("Doc Status")
            button.setObjectName("tb-status-doc")
            button.setVisible(False)
            button.clicked.connect(self._handle_clicked)  # type: ignore[attr-defined]
            # Prevent the button from affecting the window's minimum width
            if QSizePolicy is not None:
                policy = button.sizePolicy()
                policy.setHorizontalPolicy(QSizePolicy.Ignored)
                button.setSizePolicy(policy)
            status_bar.addPermanentWidget(button)
        except Exception:
            return
        self._button = button
        self._refresh()

    def set_state(self, text: str, detail: str | None = None, *, severity: str | None = None) -> None:
        self._text = text.strip()
        self._detail = (detail or "").strip()
        self._severity = (severity or "").strip()
        self._refresh()

    def set_callback(self, callback: Callable[[], None] | None) -> None:
        self._callback = callback

    def _refresh(self) -> None:
        if self._button is None:
            return
        visible = bool(self._text)
        try:
            self._button.setVisible(visible)
            if visible:
                self._button.setText(self._text)
                self._button.setToolTip(self._detail)
                self._button.setProperty("data-severity", self._severity)
                style = getattr(self._button, "style", None)
                if callable(getattr(style, "unpolish", None)) and callable(getattr(style, "polish", None)):
                    try:
                        style.unpolish(self._button)
                        style.polish(self._button)
                    except Exception:  # pragma: no cover - Qt defensive guard
                        pass
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _handle_clicked(self) -> None:
        if self._callback is None:
            return
        try:
            self._callback()
        except Exception:  # pragma: no cover - Qt defensive guard
            pass


class EditorLockIndicator:
    """Visual indicator showing editor lock state with padlock symbol."""

    # Padlock symbols
    LOCKED_SYMBOL = "🔒 Locked"
    UNLOCKED_SYMBOL = "🔓"

    def __init__(self) -> None:
        self._label: Any | None = None
        self._is_locked: bool = False
        self._message: str = ""

    def install(self, status_bar: Any | None) -> None:
        """Install the lock indicator into the status bar."""
        import logging
        _logger = logging.getLogger(__name__)
        _logger.info("EditorLockIndicator.install called, status_bar=%s, QLabel=%s", status_bar, QLabel)
        if status_bar is None or QLabel is None:
            _logger.warning("EditorLockIndicator.install: early return (status_bar=%s, QLabel=%s)", status_bar, QLabel)
            return
        if self._label is not None:
            _logger.info("EditorLockIndicator.install: already installed")
            return
        try:
            from PySide6.QtWidgets import QSizePolicy
        except Exception:  # pragma: no cover - optional during tests
            QSizePolicy = None
        try:
            label = QLabel(self.UNLOCKED_SYMBOL)
            label.setObjectName("tb-status-lock")
            label.setContentsMargins(8, 0, 8, 0)
            # Always visible - shows lock state at all times
            label.setVisible(True)
            # Prevent the label from affecting window minimum width
            if QSizePolicy is not None:
                policy = label.sizePolicy()
                policy.setHorizontalPolicy(QSizePolicy.Ignored)
                label.setSizePolicy(policy)
            status_bar.addPermanentWidget(label)
            _logger.info("EditorLockIndicator.install: label added successfully, text=%r", label.text())
        except Exception as e:
            _logger.exception("EditorLockIndicator.install: failed to create/add label: %s", e)
            return
        self._label = label
        self._refresh()

    def set_locked(self, locked: bool, message: str = "") -> None:
        """Update the lock state and optional message."""
        self._is_locked = bool(locked)
        self._message = message.strip()
        self._refresh()

    @property
    def is_locked(self) -> bool:
        """Return current lock state."""
        return self._is_locked

    def _refresh(self) -> None:
        """Update the label display based on current state."""
        if self._label is None:
            return
        try:
            symbol = self.LOCKED_SYMBOL if self._is_locked else self.UNLOCKED_SYMBOL
            self._label.setText(symbol)
            tooltip = self._message or ("Editor locked" if self._is_locked else "Editor unlocked")
            self._label.setToolTip(tooltip)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

class StatusBar:
    """Status bar that mirrors the plan.md contract yet stays test friendly."""

    def __init__(self, parent: Any | None = None) -> None:
        self._message: str = ""
        self._message_timeout: int | None = None
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
        self._embedding_processing: bool = False
        self._embedding_processing_detail: str = ""
        self._review_controls = DiffReviewControls()
        self._subagent_status: str = ""
        self._subagent_detail: str = ""
        self._chunk_flow_status: str = ""
        self._chunk_flow_detail: str = ""
        self._guardrail_status: str = ""
        self._guardrail_detail: str = ""
        self._analysis_status: str = ""
        self._analysis_detail: str = ""
        self._context_widget = ContextUsageWidget()
        self._document_status_text: str = ""
        self._document_status_detail: str = ""
        self._document_status_severity: str = ""
        self._document_status_indicator = DocumentStatusIndicator()
        self._lock_indicator = EditorLockIndicator()

        self._qt_bar = self._build_qt_status_bar(parent)
        self._cursor_label: Any = None
        self._format_label: Any = None
        self._ai_label: Any = None
        self._outline_label: Any = None
        self._embedding_label: Any = None
        self._autosave_label: Any = None
        self._subagent_label: Any = None
        self._chunk_flow_label: Any = None
        self._guardrail_label: Any = None
        self._analysis_label: Any = None
        self._lock_label: Any = None

        if self._qt_bar is not None:
            self._init_widgets()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_message(self, message: str, *, timeout_ms: int | None = None) -> None:
        """Show a primary status message, honoring optional timeouts."""

        self._message = message
        self._message_timeout = timeout_ms
        if self._qt_bar is not None:
            timeout = timeout_ms or 0
            try:
                self._qt_bar.showMessage(message, timeout)
            except Exception:  # pragma: no cover - Qt defensive guard
                pass

    def clear_message(self) -> None:
        """Clear the current status message."""

        self._message = ""
        self._message_timeout = None
        if self._qt_bar is not None:
            try:
                self._qt_bar.clearMessage()
            except Exception:  # pragma: no cover - Qt defensive guard
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
        self._update_label(self._ai_label, self._format_ai_text())

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
            except Exception:  # pragma: no cover - Qt defensive guard
                pass

    def set_embedding_status(self, status: str | None, *, detail: str | None = None) -> None:
        """Display the active embedding backend/provider status."""

        self._embedding_status = (status or "").strip()
        self._embedding_detail = (detail or "").strip()
        self._refresh_embedding_label()

    def set_embedding_processing(self, active: bool, *, detail: str | None = None) -> None:
        """Toggle the embeddings processing indicator."""

        self._embedding_processing = bool(active)
        if detail is not None:
            self._embedding_processing_detail = detail.strip()
        elif not active:
            self._embedding_processing_detail = ""
        self._refresh_embedding_label()

    def set_subagent_status(self, status: str | None, *, detail: str | None = None) -> None:
        """Show whether the subagent sandbox is active plus optional detail text."""

        self._subagent_status = (status or "").strip()
        self._subagent_detail = (detail or "").strip()
        if self._subagent_label is not None:
            self._update_label(self._subagent_label, self._format_subagent_text())
            try:
                self._subagent_label.setToolTip(self._subagent_detail)
            except Exception:  # pragma: no cover - Qt defensive guard
                pass

    def set_chunk_flow_state(self, status: str | None, *, detail: str | None = None) -> None:
        """Reflect whether the AI stayed on the chunk-first path this turn."""

        self._chunk_flow_status = (status or "").strip()
        self._chunk_flow_detail = (detail or "").strip()
        label = self._chunk_flow_label
        if label is None:
            return
        self._update_label(label, self._format_chunk_flow_text())
        visible = bool(self._chunk_flow_status)
        try:
            label.setToolTip(self._chunk_flow_detail or self._chunk_flow_status)
            label.setVisible(visible)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def set_guardrail_notice(self, status: str | None, *, detail: str | None = None) -> None:
        """Show high-level guardrail notices (e.g., safe edit rejections)."""

        self._guardrail_status = (status or "").strip()
        self._guardrail_detail = (detail or "").strip()
        label = self._guardrail_label
        if label is None:
            return
        self._update_label(label, self._format_guardrail_text())
        visible = bool(self._guardrail_status)
        try:
            label.setToolTip(self._guardrail_detail or self._guardrail_status)
            label.setVisible(visible)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def set_analysis_state(self, status: str | None, *, detail: str | None = None) -> None:
        """Display the latest preflight analysis summary."""

        self._analysis_status = (status or "").strip()
        self._analysis_detail = (detail or "").strip()
        label = self._analysis_label
        if label is None:
            return
        self._update_label(label, self._format_analysis_text())
        visible = bool(self._analysis_status)
        try:
            label.setToolTip(self._analysis_detail or self._analysis_status)
            label.setVisible(visible)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def set_lock_state(self, locked: bool, *, reason: str = "") -> None:
        """Update the editor lock indicator (padlock symbol).
        
        Args:
            locked: Whether the editor is currently locked.
            reason: Optional reason explaining why (shown as tooltip).
        """
        self._lock_indicator.set_locked(locked, reason)

    def set_editor_lock_state(self, locked: bool, message: str = "") -> None:
        """Update the editor lock indicator (padlock symbol).
        
        Args:
            locked: Whether the editor is currently locked.
            message: Optional tooltip/message explaining why.
        
        Note:
            This is an alias for set_lock_state for backwards compatibility.
        """
        self._lock_indicator.set_locked(locked, message)

    @property
    def editor_lock_state(self) -> tuple[bool, str]:
        """Return the current editor lock state as (is_locked, message)."""
        return (self._lock_indicator.is_locked, self._lock_indicator._message)

    def set_document_status_badge(
        self,
        status: str | None,
        *,
        detail: str | None = None,
        severity: str | None = None,
    ) -> None:
        """Update the document status badge text."""

        self._document_status_text = (status or "").strip()
        self._document_status_detail = (detail or "").strip()
        self._document_status_severity = (severity or "").strip()
        self._document_status_indicator.set_state(
            self._document_status_text,
            self._document_status_detail,
            severity=self._document_status_severity,
        )

    def set_document_status_callback(self, callback: Callable[[], None] | None) -> None:
        """Register a handler invoked when the status badge is clicked."""

        self._document_status_indicator.set_callback(callback)

    def set_review_state(
        self,
        summary: str | None,
        *,
        accept_callback: Callable[[], None] | None = None,
        reject_callback: Callable[[], None] | None = None,
    ) -> None:
        """Show inline diff review summary and optional callbacks."""

        self._review_controls.set_state(
            summary,
            accept_callback=accept_callback,
            reject_callback=reject_callback,
        )

    def clear_review_state(self) -> None:
        self._review_controls.clear()

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

    @property
    def chunk_flow_state(self) -> tuple[str, str]:
        return (self._chunk_flow_status, self._chunk_flow_detail)

    @property
    def guardrail_notice_state(self) -> tuple[str, str]:
        return (self._guardrail_status, self._guardrail_detail)

    @property
    def analysis_state(self) -> tuple[str, str]:
        return (self._analysis_status, self._analysis_detail)

    @property
    def review_summary(self) -> str:
        return self._review_controls.summary_text

    @property
    def review_controls_visible(self) -> bool:
        return self._review_controls.visible

    @property
    def document_status_badge(self) -> tuple[str, str]:
        return (self._document_status_text, self._document_status_detail)

    @property
    def document_status_severity(self) -> str:
        return self._document_status_severity

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _init_widgets(self) -> None:
        if self._qt_bar is None or QLabel is None:
            return

        try:
            from PySide6.QtWidgets import QSizePolicy
        except Exception:  # pragma: no cover - optional during tests
            QSizePolicy = None

        self._cursor_label = QLabel(self._format_cursor_text())
        self._cursor_label.setObjectName("tb-status-cursor")
        self._format_label = QLabel(self._document_format.upper())
        self._format_label.setObjectName("tb-status-format")
        self._ai_label = QLabel(self._format_ai_text())
        self._ai_label.setObjectName("tb-status-ai")
        self._outline_label = QLabel(self._format_outline_text())
        self._outline_label.setObjectName("tb-status-outline")
        self._embedding_label = QLabel(self._format_embedding_text())
        self._embedding_label.setObjectName("tb-status-embedding")
        self._autosave_label = QLabel(self._format_autosave_text())
        self._autosave_label.setObjectName("tb-status-autosave")
        self._subagent_label = QLabel(self._format_subagent_text())
        self._subagent_label.setObjectName("tb-status-subagents")
        self._chunk_flow_label = QLabel(self._format_chunk_flow_text())
        self._chunk_flow_label.setObjectName("tb-status-chunk-flow")
        self._chunk_flow_label.setVisible(False)
        self._guardrail_label = QLabel(self._format_guardrail_text())
        self._guardrail_label.setObjectName("tb-status-guardrail")
        self._guardrail_label.setVisible(False)
        self._analysis_label = QLabel(self._format_analysis_text())
        self._analysis_label.setObjectName("tb-status-analysis")
        self._analysis_label.setVisible(False)
        
        # Create lock indicator label directly
        self._lock_label = QLabel(EditorLockIndicator.UNLOCKED_SYMBOL)
        self._lock_label.setObjectName("tb-status-lock")
        self._lock_indicator._label = self._lock_label  # Wire up to indicator

        # Apply Ignored horizontal policy to labels that start hidden
        # to prevent them from affecting window minimum width when shown
        hidden_labels = (
            self._chunk_flow_label,
            self._guardrail_label,
            self._analysis_label,
        )
        if QSizePolicy is not None:
            for label in hidden_labels:
                policy = label.sizePolicy()
                policy.setHorizontalPolicy(QSizePolicy.Ignored)
                label.setSizePolicy(policy)

        # Install review controls FIRST so they appear prominently on the left
        # of the permanent widget area (before all the status labels)
        self._review_controls.install(self._qt_bar)

        # Add AI label to the LEFT side of the status bar (non-permanent widget)
        self._ai_label.setContentsMargins(8, 0, 8, 0)
        try:
            self._qt_bar.addWidget(self._ai_label)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

        # Add remaining labels as permanent widgets (right side)
        for label in (
            self._cursor_label,
            self._format_label,
            self._outline_label,
            self._embedding_label,
            self._autosave_label,
            self._subagent_label,
            self._chunk_flow_label,
            self._guardrail_label,
            self._analysis_label,
            self._lock_label,  # Add lock label with other labels
        ):
            label.setContentsMargins(8, 0, 8, 0)
            try:
                self._qt_bar.addPermanentWidget(label)
            except Exception:
                break

        self._context_widget.install(self._qt_bar)
        self._document_status_indicator.install(self._qt_bar)

    def _update_label(self, label: Any, text: str) -> None:
        if label is None:
            return
        try:
            label.setText(text)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _refresh_embedding_label(self) -> None:
        label = self._embedding_label
        if label is None:
            return
        self._update_label(label, self._format_embedding_text())
        try:
            tooltip = self._format_embedding_tooltip()
            label.setToolTip(tooltip)
        except Exception:  # pragma: no cover - Qt defensive guard
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
        parts: list[str] = []
        if self._embedding_status:
            base = f"Embeddings: {self._embedding_status}"
        else:
            base = "Embeddings: Idle"
        parts.append(base)
        if self._embedding_processing:
            parts.append("Processing...")
        return " · ".join(parts)

    def _format_embedding_tooltip(self) -> str:
        parts: list[str] = []
        if self._embedding_detail:
            parts.append(self._embedding_detail)
        if self._embedding_processing:
            parts.append(self._embedding_processing_detail or "Embeddings are processing.")
        return " ".join(part for part in parts if part).strip()

    def _format_ai_text(self) -> str:
        return f"AI: {self._ai_state}"

    def _format_subagent_text(self) -> str:
        status = self._subagent_status or "Idle"
        return f"Subagents: {status}"

    def _format_chunk_flow_text(self) -> str:
        return f"Chunk Flow: {self._chunk_flow_status}" if self._chunk_flow_status else ""

    def _format_guardrail_text(self) -> str:
        return f"Guardrail: {self._guardrail_status}" if self._guardrail_status else ""

    def _format_analysis_text(self) -> str:
        return f"Analysis: {self._analysis_status}" if self._analysis_status else ""

    @staticmethod
    def _coerce_state(state: str | Enum) -> str:
        if isinstance(state, Enum):
            return str(getattr(state, "value", state.name))
        return str(state).strip() or "Idle"

    def trigger_accept_review(self) -> None:
        self._review_controls.trigger_accept()

    def trigger_reject_review(self) -> None:
        self._review_controls.trigger_reject()

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
        except Exception:  # pragma: no cover - Qt defensive guard
            pass
        return bar


__all__ = ["StatusBar", "ContextUsageWidget", "DiffReviewControls", "DocumentStatusIndicator", "EditorLockIndicator"]

