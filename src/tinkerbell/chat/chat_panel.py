"""Chat panel implementation with logic mirroring the architectural plan.

The panel keeps the actionable bits (message bookkeeping, composer state,
signal-style callbacks) independent from the actual Qt widgets so unit tests
can operate without a running ``QApplication``. When PySide6 is available, the
class eagerly builds a lightweight UI consisting of:

* A history list view mirroring the recorded :class:`ChatMessage` objects.
* A multi-line composer with a send button.
* A suggestion list that can pre-fill the composer.
* A tool-trace list that surfaces LangChain/agent tool calls for transparency.

Downstream components (main window, AI controller) interact with the panel via
listener registrations and explicit public methods instead of relying on Qt
signals. This mirrors the approach used by :mod:`editor.editor_widget` and keeps
the headless tests simple while remaining Qt-friendly for the full app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Protocol, Sequence

from .message_model import ChatMessage, ToolTrace

QApplication: Any = None
QFrame: Any = None
QHBoxLayout: Any = None
QLabel: Any = None
QListWidget: Any = None
QPushButton: Any = None
QTextEdit: Any = None
QVBoxLayout: Any = None
QWidgetBase: Any = None

try:  # pragma: no cover - PySide6 optional in CI
    from PySide6.QtWidgets import (
        QApplication as _QtApplication,
        QFrame as _QtFrame,
        QHBoxLayout as _QtHBoxLayout,
        QLabel as _QtLabel,
        QListWidget as _QtListWidget,
        QPushButton as _QtPushButton,
        QTextEdit as _QtTextEdit,
        QVBoxLayout as _QtVBoxLayout,
        QWidget as _QtWidget,
    )

    QApplication = _QtApplication
    QFrame = _QtFrame
    QHBoxLayout = _QtHBoxLayout
    QLabel = _QtLabel
    QListWidget = _QtListWidget
    QPushButton = _QtPushButton
    QTextEdit = _QtTextEdit
    QVBoxLayout = _QtVBoxLayout
    QWidgetBase = _QtWidget
except Exception:  # pragma: no cover - runtime fallback keeps dependencies optional

    class _StubQWidget:  # type: ignore[too-many-ancestors]
        """Runtime placeholder mirroring PySide6 QWidget signatures."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    QWidgetBase = _StubQWidget


class RequestListener(Protocol):
    """Protocol describing callbacks invoked on prompt submission."""

    def __call__(self, prompt: str, metadata: dict[str, Any]) -> None:
        ...


@dataclass(slots=True)
class ComposerContext:
    """Metadata passed alongside prompt submissions."""

    selection_summary: Optional[str] = None
    extras: dict[str, Any] | None = None

    def to_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.selection_summary:
            metadata["selection_summary"] = self.selection_summary
        if self.extras:
            metadata.update(self.extras)
        return metadata


class ChatPanel(QWidgetBase):
    """Pane showing chat history, composer controls, and tool traces."""

    MAX_HISTORY = 200

    def __init__(
        self,
        parent: Optional[Any] = None,
        *,
        history_limit: Optional[int] = None,
    ) -> None:
        super().__init__(parent)
        self._history_limit = max(1, history_limit or self.MAX_HISTORY)
        self._messages: List[ChatMessage] = []
        self._active_stream: Optional[ChatMessage] = None
        self._tool_traces: list[ToolTrace] = []
        self._suggestions: list[str] = []
        self._composer_text: str = ""
        self._composer_context = ComposerContext()
        self._request_listeners: list[RequestListener] = []

        # Qt widgets (optional; None when headless)
        self._history_widget: Any = None
        self._composer_widget: Any = None
        self._send_button: Any = None
        self._suggestion_widget: Any = None
        self._tool_trace_widget: Any = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API – history + composer
    # ------------------------------------------------------------------
    def history(self) -> List[ChatMessage]:
        """Return a copy of the recorded chat history."""

        return list(self._messages)

    def clear_history(self) -> None:
        """Remove all recorded messages and reset streaming state."""

        self._messages.clear()
        self._tool_traces.clear()
        self._active_stream = None
        self._refresh_history_widget()
        self._refresh_tool_trace_widget()

    @property
    def composer_text(self) -> str:
        """Return the current composer text buffer."""

        if self._composer_widget is not None:
            try:
                return self._composer_widget.toPlainText()
            except Exception:  # pragma: no cover - defensive
                pass
        return self._composer_text

    def set_composer_text(self, text: str, *, context: Optional[ComposerContext] = None) -> None:
        """Set the composer text and optional metadata context."""

        self._composer_text = text
        if context is not None:
            self._composer_context = context
        if self._composer_widget is not None:
            try:
                self._composer_widget.blockSignals(True)
                self._composer_widget.setPlainText(text)
            finally:  # pragma: no branch - ensure unblock
                self._composer_widget.blockSignals(False)

    def clear_composer(self) -> None:
        """Clear the composer text and metadata context."""

        self.set_composer_text("")
        self._composer_context = ComposerContext()

    # ------------------------------------------------------------------
    # Message append helpers
    # ------------------------------------------------------------------
    def append_user_message(self, content: str, selection_summary: Optional[str] = None) -> ChatMessage:
        """Add a user-authored message to the panel."""

        metadata: dict[str, Any] = {}
        if selection_summary:
            metadata["selection_summary"] = selection_summary.strip()
        message = ChatMessage(role="user", content=content, metadata=metadata)
        self._messages.append(message)
        self._trim_history()
        self._refresh_history_widget()
        return message

    def append_ai_message(self, message: ChatMessage, *, streaming: bool = False) -> ChatMessage:
        """Add an AI-authored message, optionally streaming partial chunks."""

        if streaming:
            target = self._active_stream
            if target is None:
                target = ChatMessage(role="assistant", content="")
                self._messages.append(target)
                self._active_stream = target
            if message.content:
                target.content += message.content
            if message.metadata:
                target.metadata.update(message.metadata)
            if message.tool_traces:
                target.tool_traces.extend(message.tool_traces)
            self._trim_history()
            self._refresh_history_widget()
            return target

        if self._active_stream is not None:
            target = self._active_stream
            if message.content:
                target.content += message.content
            if message.metadata:
                target.metadata.update(message.metadata)
            if message.tool_traces:
                target.tool_traces.extend(message.tool_traces)
            self._active_stream = None
            self._trim_history()
            self._refresh_history_widget()
            return target

        self._messages.append(message)
        self._trim_history()
        self._refresh_history_widget()
        return message

    def show_tool_trace(self, trace: ToolTrace) -> ToolTrace:
        """Attach a tool trace to the latest assistant message and cache it."""

        self._tool_traces.append(trace)
        target = None
        if self._messages and self._messages[-1].role == "assistant":
            target = self._messages[-1]
        elif self._active_stream is not None:
            target = self._active_stream
        if target is not None:
            target.tool_traces.append(trace)
        self._refresh_tool_trace_widget()
        return trace

    def load_transcript(self, messages: Iterable[ChatMessage]) -> None:
        """Replace the history with ``messages`` respecting the history limit."""

        self._messages = list(messages)
        self._trim_history()
        self._refresh_history_widget()

    # ------------------------------------------------------------------
    # Suggestions management
    # ------------------------------------------------------------------
    def set_suggestions(self, suggestions: Sequence[str]) -> None:
        """Populate the suggestion list used to pre-fill the composer."""

        deduped: list[str] = []
        for suggestion in suggestions:
            text = suggestion.strip()
            if not text or text in deduped:
                continue
            deduped.append(text)
        self._suggestions = deduped
        self._refresh_suggestion_widget()

    def suggestions(self) -> tuple[str, ...]:
        """Expose the currently configured suggestions."""

        return tuple(self._suggestions)

    def select_suggestion(self, index: int) -> str:
        """Apply a suggestion to the composer and return the text."""

        if not 0 <= index < len(self._suggestions):
            raise IndexError("Suggestion index out of range")
        text = self._suggestions[index]
        self.set_composer_text(text)
        return text

    # ------------------------------------------------------------------
    # Prompt submission + listeners
    # ------------------------------------------------------------------
    def add_request_listener(self, listener: RequestListener) -> None:
        """Register a callback fired when the user submits a prompt."""

        self._request_listeners.append(listener)

    def remove_request_listener(self, listener: RequestListener) -> None:
        """Remove a previously registered request listener."""

        try:
            self._request_listeners.remove(listener)
        except ValueError:  # pragma: no cover - defensive guard
            pass

    def send_prompt(
        self,
        prompt: Optional[str] = None,
        *,
        record_history: bool = True,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Submit a prompt to registered listeners and optionally log it."""

        text = prompt if prompt is not None else self.composer_text.strip()
        if not text:
            raise ValueError("Prompt cannot be empty")
        if record_history:
            self.append_user_message(text, self._composer_context.selection_summary)
        combined_metadata = self._composer_context.to_metadata()
        if metadata:
            combined_metadata.update(metadata)
        self._emit_request(text, combined_metadata)
        self.clear_composer()
        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Instantiate Qt widgets when the runtime supports them."""

        if (
            QApplication is None
            or QListWidget is None
            or QTextEdit is None
            or QPushButton is None
            or QVBoxLayout is None
            or QFrame is None
            or QHBoxLayout is None
        ):
            return
        try:
            if QApplication.instance() is None:
                return
        except Exception:  # pragma: no cover - Qt defensive guard
            return

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._history_widget = QListWidget(self)
        self._history_widget.setObjectName("tb-chat-history")
        layout.addWidget(self._history_widget)

        composer_frame = QFrame(self)
        composer_layout = QHBoxLayout(composer_frame)
        composer_layout.setContentsMargins(0, 0, 0, 0)
        self._composer_widget = QTextEdit(composer_frame)
        self._composer_widget.setObjectName("tb-chat-composer")
        composer_layout.addWidget(self._composer_widget, 1)
        self._send_button = QPushButton("Send", composer_frame)
        self._send_button.clicked.connect(self._handle_send_clicked)  # type: ignore[attr-defined]
        composer_layout.addWidget(self._send_button, 0)
        layout.addWidget(composer_frame)

        if QLabel is not None and QListWidget is not None:
            suggestion_label = QLabel("Suggestions", self)
            layout.addWidget(suggestion_label)
            self._suggestion_widget = QListWidget(self)
            self._suggestion_widget.setObjectName("tb-chat-suggestions")
            self._suggestion_widget.itemClicked.connect(self._handle_suggestion_clicked)  # type: ignore[attr-defined]
            layout.addWidget(self._suggestion_widget)

            trace_label = QLabel("Tool Activity", self)
            layout.addWidget(trace_label)
            self._tool_trace_widget = QListWidget(self)
            self._tool_trace_widget.setObjectName("tb-chat-traces")
            layout.addWidget(self._tool_trace_widget)

    def _emit_request(self, prompt: str, metadata: dict[str, Any]) -> None:
        for listener in list(self._request_listeners):
            listener(prompt, metadata)

    def _trim_history(self) -> None:
        overflow = len(self._messages) - self._history_limit
        if overflow > 0:
            del self._messages[0:overflow]

    def _refresh_history_widget(self) -> None:
        widget = self._history_widget
        if widget is None:
            return
        try:
            widget.blockSignals(True)
            widget.clear()
            for message in self._messages:
                widget.addItem(self._render_message_text(message))
        finally:  # pragma: no branch - ensure unblock
            widget.blockSignals(False)

    def _refresh_suggestion_widget(self) -> None:
        widget = self._suggestion_widget
        if widget is None:
            return
        try:
            widget.blockSignals(True)
            widget.clear()
            for suggestion in self._suggestions:
                widget.addItem(suggestion)
        finally:
            widget.blockSignals(False)

    def _refresh_tool_trace_widget(self) -> None:
        widget = self._tool_trace_widget
        if widget is None:
            return
        try:
            widget.blockSignals(True)
            widget.clear()
            for trace in self._tool_traces[-10:]:
                widget.addItem(f"{trace.name}: {trace.output_summary}")
        finally:
            widget.blockSignals(False)

    def _render_message_text(self, message: ChatMessage) -> str:
        prefix = message.role.upper()
        return f"[{prefix}] {message.content.strip()}"

    # Qt callbacks ----------------------------------------------------
    def _handle_send_clicked(self) -> None:
        try:
            self.send_prompt()
        except ValueError:
            pass

    def _handle_suggestion_clicked(self, item: Any) -> None:
        try:
            text = str(item.text())
        except Exception:  # pragma: no cover - Qt defensive guard
            return
        self.set_composer_text(text)

