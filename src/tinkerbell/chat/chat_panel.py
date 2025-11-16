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
from typing import Any, Callable, Iterable, List, Optional, Protocol, Sequence, cast

from .message_model import ChatMessage, ToolTrace

QApplication: Any = None
QFrame: Any = None
QHBoxLayout: Any = None
QLabel: Any = None
QListWidget: Any = None
QListWidgetItem: Any = None
QMenu: Any = None
QPushButton: Any = None
QTextEdit: Any = None
QToolButton: Any = None
QVBoxLayout: Any = None
QWidgetBase: Any = None
QAbstractItemView: Any = None
Qt: Any = None
QEvent: Any = None

# Fallback Qt constants for environments where PySide6 isn't available during tests.
FALLBACK_ENTER_KEYS = (0x01000004, 0x01000005)  # Qt.Key_Return, Qt.Key_Enter
FALLBACK_SHIFT_MODIFIER = 0x02000000  # Qt.ShiftModifier bit mask
FALLBACK_KEY_PRESS_EVENT = 6  # QEvent.KeyPress

try:  # pragma: no cover - PySide6 optional in CI
    from PySide6.QtCore import Qt as _Qt, QEvent as _QtEvent
    from PySide6.QtWidgets import (
        QApplication as _QtApplication,
        QFrame as _QtFrame,
        QHBoxLayout as _QtHBoxLayout,
        QLabel as _QtLabel,
    QListWidget as _QtListWidget,
    QListWidgetItem as _QtListWidgetItem,
    QMenu as _QtMenu,
    QPushButton as _QtPushButton,
        QTextEdit as _QtTextEdit,
        QToolButton as _QtToolButton,
        QVBoxLayout as _QtVBoxLayout,
        QWidget as _QtWidget,
        QAbstractItemView as _QtAbstractItemView,
    )

    QApplication = _QtApplication
    QFrame = _QtFrame
    QHBoxLayout = _QtHBoxLayout
    QLabel = _QtLabel
    QListWidget = _QtListWidget
    QListWidgetItem = _QtListWidgetItem
    QMenu = _QtMenu
    QPushButton = _QtPushButton
    QTextEdit = _QtTextEdit
    QToolButton = _QtToolButton
    QVBoxLayout = _QtVBoxLayout
    QWidgetBase = _QtWidget
    QAbstractItemView = _QtAbstractItemView
    Qt = _Qt
    QEvent = _QtEvent
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


class SessionResetListener(Protocol):
    """Protocol describing callbacks fired when a new chat session begins."""

    def __call__(self) -> None:
        ...


class SuggestionPanelListener(Protocol):
    """Protocol fired whenever the suggestion panel visibility changes."""

    def __call__(self, is_open: bool) -> None:
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
    HISTORY_BOTTOM_PADDING = 12
    SMOOTH_SCROLL_STEP = 12
    SMOOTH_SCROLL_PAGE = 72
    AUTOLOCK_THRESHOLD = 80

    def __init__(
        self,
        parent: Optional[Any] = None,
        *,
        history_limit: Optional[int] = None,
        show_tool_activity_panel: bool = False,
    ) -> None:
        super().__init__(parent)
        self._history_limit = max(1, history_limit or self.MAX_HISTORY)
        self._messages: List[ChatMessage] = []
        self._active_stream: Optional[ChatMessage] = None
        self._tool_traces: list[ToolTrace] = []
        self._visible_tool_traces: list[ToolTrace] = []
        self._suggestions: list[str] = []
        self._composer_text: str = ""
        self._composer_context = ComposerContext()
        self._request_listeners: list[RequestListener] = []
        self._session_reset_listeners: list[SessionResetListener] = []
        self._suggestion_panel_listeners: list[SuggestionPanelListener] = []
        self._tool_activity_visible = bool(show_tool_activity_panel)

        # Qt widgets (optional; None when headless)
        self._history_widget: Any = None
        self._composer_widget: Any = None
        self._send_button: Any = None
        self._new_chat_button: Any = None
        self._suggestion_widget: Any = None
        self._suggestion_toggle: Any = None
        self._tool_trace_widget: Any = None
        self._tool_trace_label: Any = None
        self._suggestion_panel_open = False
        self._last_copied_text: Optional[str] = None
        self._stop_ai_callback: Optional[Callable[[], None]] = None
        self._ai_running = False
        self._send_button_idle_text = "Send"
        self._send_button_idle_tooltip = "Send message"
        self._action_button_busy_text = "■"
        self._action_button_busy_tooltip = "Stop the current AI response"

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API – history + composer
    # ------------------------------------------------------------------
    def history(self) -> List[ChatMessage]:
        """Return a copy of the recorded chat history."""

        return list(self._messages)

    @property
    def tool_activity_visible(self) -> bool:
        """Return whether the tool activity panel is currently visible."""

        return self._tool_activity_visible

    @property
    def last_copied_text(self) -> Optional[str]:
        """Expose the last message text copied to the clipboard."""

        return self._last_copied_text

    def set_tool_activity_visibility(self, visible: bool) -> None:
        """Show or hide the tool activity panel in the Qt layout."""

        state = bool(visible)
        if state == self._tool_activity_visible:
            return
        self._tool_activity_visible = state
        self._sync_tool_trace_visibility()

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

    def start_new_chat(self) -> None:
        """Reset the chat conversation and composer for a fresh session."""

        self.clear_history()
        self.clear_composer()
        self._emit_session_reset()

    def set_selection_summary(
        self,
        summary: Optional[str],
        *,
        extras: Optional[dict[str, Any]] = None,
    ) -> None:
        """Update the composer metadata describing the current editor selection."""

        normalized = summary.strip() if summary else None
        self._composer_context.selection_summary = normalized or None
        if extras is not None:
            self._composer_context.extras = extras if extras else None

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
        should_autoscroll = self._should_auto_scroll()
        self._refresh_history_widget()
        if should_autoscroll:
            self._scroll_history_to_bottom(force=True)
        return message

    def append_ai_message(self, message: ChatMessage, *, streaming: bool = False) -> ChatMessage:
        """Add an AI-authored message, optionally streaming partial chunks."""

        should_autoscroll = self._should_auto_scroll()
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
            if should_autoscroll:
                self._scroll_history_to_bottom(force=True)
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
            if should_autoscroll:
                self._scroll_history_to_bottom(force=True)
            return target

        self._messages.append(message)
        self._trim_history()
        self._refresh_history_widget()
        if should_autoscroll:
            self._scroll_history_to_bottom(force=True)
        return message

    def show_tool_trace(self, trace: ToolTrace) -> ToolTrace:
        """Attach a tool trace to the latest assistant message and cache it."""

        self._tool_traces.append(trace)
        step_index = len(self._tool_traces)
        trace.step_index = step_index
        metadata = dict(getattr(trace, "metadata", None) or {})
        metadata.setdefault("step_index", step_index)
        trace.metadata = metadata
        target = None
        if self._messages and self._messages[-1].role == "assistant":
            target = self._messages[-1]
        elif self._active_stream is not None:
            target = self._active_stream
        if target is not None:
            target.tool_traces.append(trace)
        self._refresh_tool_trace_widget()
        return trace

    def update_tool_trace(self, trace: ToolTrace) -> None:
        """Refresh the tool trace list when an existing trace mutates."""

        if not any(existing is trace for existing in self._tool_traces):
            return
        self._refresh_tool_trace_widget()

    def load_transcript(self, messages: Iterable[ChatMessage]) -> None:
        """Replace the history with ``messages`` respecting the history limit."""

        self._messages = list(messages)
        self._trim_history()
        self._refresh_history_widget()
        self._scroll_history_to_bottom(force=True)

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

    def copy_text_to_clipboard(self, text: str) -> bool:
        """Copy ``text`` to the system clipboard when Qt is available."""

        normalized = text if isinstance(text, str) else str(text)
        self._last_copied_text = normalized
        if not QApplication:
            return False
        try:
            instance = QApplication.instance()
        except Exception:  # pragma: no cover - defensive guard
            return False
        if instance is None:
            return False
        try:
            clipboard = QApplication.clipboard()
        except Exception:  # pragma: no cover - clipboard access failed
            return False
        if clipboard is None:
            return False
        try:
            clipboard.setText(normalized)
        except Exception:  # pragma: no cover - clipboard write failure
            return False
        return True

    def copy_tool_trace_details(
        self,
        trace: Optional[ToolTrace] = None,
        *,
        index: Optional[int] = None,
    ) -> bool:
        """Copy a formatted summary of a recorded tool trace."""

        selected = trace or self._resolve_tool_trace(index)
        if selected is None:
            return False
        details = self._format_tool_trace_details(selected)
        return self.copy_text_to_clipboard(details)

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

    def add_session_reset_listener(self, listener: SessionResetListener) -> None:
        """Register a callback fired when a new chat session begins."""

        self._session_reset_listeners.append(listener)

    def remove_session_reset_listener(self, listener: SessionResetListener) -> None:
        """Remove a previously registered session reset listener."""

        try:
            self._session_reset_listeners.remove(listener)
        except ValueError:  # pragma: no cover - defensive guard
            pass

    def add_suggestion_panel_listener(self, listener: SuggestionPanelListener) -> None:
        """Register a callback fired when the suggestion panel toggles."""

        self._suggestion_panel_listeners.append(listener)

    def remove_suggestion_panel_listener(self, listener: SuggestionPanelListener) -> None:
        """Remove a previously registered suggestion panel listener."""

        try:
            self._suggestion_panel_listeners.remove(listener)
        except ValueError:  # pragma: no cover - defensive guard
            pass

    def set_stop_ai_callback(self, callback: Optional[Callable[[], None]]) -> None:
        """Configure a callable invoked when the user requests to stop the AI run."""

        self._stop_ai_callback = callback

    def set_ai_running(self, active: bool) -> None:
        """Toggle the composer/button state based on whether the AI is processing."""

        state = bool(active)
        if state == self._ai_running:
            return
        self._ai_running = state
        self._refresh_action_button_state()
        self._refresh_composer_enablement()

    def _emit_stop_ai_request(self) -> None:
        callback = self._stop_ai_callback
        if callback is None:
            return
        try:
            callback()
        except Exception:  # pragma: no cover - defensive guard
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
        layout.setSpacing(10)

        self._history_widget = QListWidget(self)
        self._configure_history_widget()
        layout.addWidget(self._history_widget, 1)

        composer_frame = QFrame(self)
        composer_frame.setObjectName("tb-chat-composer-frame")
        composer_layout = QVBoxLayout(composer_frame)
        composer_layout.setContentsMargins(0, 0, 0, 0)
        composer_layout.setSpacing(4)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(6)

        button_stack = QVBoxLayout()
        button_stack.setContentsMargins(0, 4, 6, 4)
        button_stack.setSpacing(6)
        input_row.addLayout(button_stack, 0)

        if QToolButton is not None:
            button_edge = 32
            self._suggestion_toggle = QToolButton(composer_frame)
            self._suggestion_toggle.setObjectName("tb-chat-suggestions-toggle")
            self._suggestion_toggle.setText("+")
            self._suggestion_toggle.setToolTip("Show suggestions")
            try:
                self._suggestion_toggle.setFixedSize(button_edge, button_edge)
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
            self._suggestion_toggle.clicked.connect(self._toggle_suggestion_panel)  # type: ignore[attr-defined]
            button_stack.addWidget(self._suggestion_toggle, 0)

            self._new_chat_button = QToolButton(composer_frame)
            self._new_chat_button.setObjectName("tb-chat-new")
            self._new_chat_button.setText("⟳")
            self._new_chat_button.setToolTip("Start a new chat")
            try:
                self._new_chat_button.setFixedSize(button_edge, button_edge)
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
            button_stack.addWidget(self._new_chat_button, 0)
        else:
            self._suggestion_toggle = None
            self._new_chat_button = QPushButton("New Chat", composer_frame)
            button_stack.addWidget(self._new_chat_button, 0)

        button_stack.addStretch(1)

        composer_shell = QFrame(composer_frame)
        composer_shell.setObjectName("tb-chat-composer-shell")
        composer_shell_layout = QHBoxLayout(composer_shell)
        composer_shell_layout.setContentsMargins(0, 0, 0, 0)
        composer_shell_layout.setSpacing(4)

        self._composer_widget = QTextEdit(composer_shell)
        self._composer_widget.setObjectName("tb-chat-composer")
        composer_fixed_height = 64
        try:
            self._composer_widget.setPlaceholderText("Message the assistant…")
            self._composer_widget.setFixedHeight(composer_fixed_height)
        except Exception:  # pragma: no cover - Qt defensive guard
            composer_fixed_height = None
        composer_shell_layout.addWidget(self._composer_widget, 1)
        self._install_composer_shortcuts()

        input_row.addWidget(composer_shell, 1)

        if QToolButton is not None:
            self._send_button = QToolButton(composer_frame)
            self._send_button.setObjectName("tb-chat-send")
            self._send_button.setText("➤")
            self._send_button.setToolTip("Send message")
            try:
                self._send_button.setFixedWidth(36)
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
        else:
            self._send_button = QPushButton("Send", composer_frame)
        self._send_button_idle_text = self._coerce_button_text(self._send_button) or self._send_button_idle_text
        self._send_button_idle_tooltip = (
            self._coerce_button_tooltip(self._send_button) or self._send_button_idle_tooltip
        )
        self._send_button.clicked.connect(self._handle_action_button_clicked)  # type: ignore[attr-defined]
        if composer_fixed_height:
            try:
                self._send_button.setFixedHeight(composer_fixed_height)
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
        self._new_chat_button.clicked.connect(self._handle_new_chat_clicked)  # type: ignore[attr-defined]
        input_row.addWidget(self._send_button, 0)

        composer_layout.addLayout(input_row)

        self._suggestion_widget = QListWidget(composer_frame)
        self._suggestion_widget.setObjectName("tb-chat-suggestions")
        self._suggestion_widget.itemClicked.connect(self._handle_suggestion_clicked)  # type: ignore[attr-defined]
        self._suggestion_widget.setVisible(False)
        composer_layout.addWidget(self._suggestion_widget)
        if self._suggestion_toggle is not None:
            try:
                self._suggestion_toggle.setEnabled(bool(self._suggestions))
            except Exception:  # pragma: no cover - Qt defensive guard
                pass

        layout.addWidget(composer_frame, 0)

        if QLabel is not None and QListWidget is not None:
            trace_label = QLabel("Tool Activity", self)
            self._tool_trace_label = trace_label
            layout.addWidget(trace_label)
            self._tool_trace_widget = QListWidget(self)
            self._tool_trace_widget.setObjectName("tb-chat-traces")
            self._configure_tool_trace_widget()
            layout.addWidget(self._tool_trace_widget)
            self._sync_tool_trace_visibility()

        self._refresh_action_button_state()
        self._refresh_composer_enablement()

    def _install_composer_shortcuts(self) -> None:
        widget = self._composer_widget
        if widget is None:
            return
        installer = getattr(widget, "installEventFilter", None)
        if not callable(installer):
            return
        try:
            installer(self)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _configure_history_widget(self) -> None:
        widget = self._history_widget
        if widget is None:
            return
        try:
            widget.setObjectName("tb-chat-history")
            widget.setAlternatingRowColors(False)
            widget.setSpacing(4)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass
        if QAbstractItemView is not None:
            try:
                widget.setSelectionMode(QAbstractItemView.NoSelection)
                widget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
        if Qt is not None:
            try:
                widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
        try:
            widget.setWordWrap(True)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass
        # Reserve a little empty space after the final chat bubble so the
        # scrollbar can travel far enough for the bubble to be fully visible.
        try:
            widget.setViewportMargins(0, 0, 0, self.HISTORY_BOTTOM_PADDING)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass
        self._configure_history_scrollbar(widget)

    def _emit_request(self, prompt: str, metadata: dict[str, Any]) -> None:
        for listener in list(self._request_listeners):
            listener(prompt, metadata)

    def _emit_session_reset(self) -> None:
        for listener in list(self._session_reset_listeners):
            listener()

    def _trim_history(self) -> None:
        overflow = len(self._messages) - self._history_limit
        if overflow > 0:
            del self._messages[0:overflow]

    def _refresh_history_widget(self) -> None:
        widget = self._history_widget
        if widget is None:
            return
        viewport_width = 0
        try:
            viewport = widget.viewport()
            if viewport is not None:
                viewport_width = viewport.width()
        except Exception:  # pragma: no cover - Qt defensive guard
            viewport_width = 0
        try:
            widget.blockSignals(True)
            widget.clear()
            for message in self._messages:
                bubble_widget = self._create_message_bubble(message, viewport_width)
                if bubble_widget is None:
                    widget.addItem(self._render_message_text(message))
                    continue
                item = QListWidgetItem(widget)
                item.setSizeHint(bubble_widget.sizeHint())
                item.setToolTip(self._tooltip_label_for_message(message))
                widget.setItemWidget(item, bubble_widget)
        finally:  # pragma: no branch - ensure unblock
            widget.blockSignals(False)

    def _configure_history_scrollbar(self, widget: Any) -> None:
        if widget is None:
            return
        getter = getattr(widget, "verticalScrollBar", None)
        if not callable(getter):
            return
        try:
            scrollbar = cast(Any, getter())
        except Exception:  # pragma: no cover - Qt defensive guard
            return
        if scrollbar is None:
            return
        try:
            scrollbar.setSingleStep(self.SMOOTH_SCROLL_STEP)
            scrollbar.setPageStep(self.SMOOTH_SCROLL_PAGE)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _should_auto_scroll(self) -> bool:
        widget = self._history_widget
        if widget is None:
            return True
        getter = getattr(widget, "verticalScrollBar", None)
        if not callable(getter):
            return True
        try:
            scrollbar = cast(Any, getter())
        except Exception:  # pragma: no cover - Qt defensive guard
            return True
        if scrollbar is None:
            return True
        try:
            maximum = int(scrollbar.maximum())
            value = int(scrollbar.value())
        except Exception:  # pragma: no cover - Qt defensive guard
            return True
        return (maximum - value) <= self.AUTOLOCK_THRESHOLD

    def _scroll_history_to_bottom(self, *, force: bool = False) -> None:
        widget = self._history_widget
        if widget is None:
            return
        if not force and not self._should_auto_scroll():
            return
        try:
            widget.scrollToBottom()
        except Exception:  # pragma: no cover - Qt defensive guard
            pass
        getter = getattr(widget, "verticalScrollBar", None)
        if not callable(getter):
            return
        try:
            scrollbar = cast(Any, getter())
        except Exception:  # pragma: no cover - Qt defensive guard
            return
        if scrollbar is None:
            return
        try:
            scrollbar.setValue(scrollbar.maximum())
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

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
        has_suggestions = bool(self._suggestions)
        if not has_suggestions:
            self._suggestion_panel_open = False
        should_show = has_suggestions and self._suggestion_panel_open
        try:
            widget.setVisible(should_show)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass
        toggle = self._suggestion_toggle
        if toggle is not None:
            try:
                toggle.setEnabled(bool(self._suggestions))
            except Exception:  # pragma: no cover - Qt defensive guard
                pass

    def _refresh_tool_trace_widget(self) -> None:
        widget = self._tool_trace_widget
        if widget is None:
            return
        try:
            widget.blockSignals(True)
            widget.clear()
            total_steps = len(self._tool_traces)
            visible_traces = list(self._tool_traces[-10:])
            self._visible_tool_traces = visible_traces
            start_index = max(0, total_steps - len(visible_traces))
            self._update_tool_trace_header(total_steps=total_steps, start_index=start_index)
            for offset, trace in enumerate(visible_traces, start=start_index + 1):
                trace.step_index = trace.step_index or offset
                name = trace.name or "tool"
                metadata = trace.metadata or {}
                badges: list[str] = []
                if str(name).endswith("patch"):
                    badges.append("patch")
                if metadata.get("compacted"):
                    badges.append("compacted")
                badge = f" [{', '.join(badges)}]" if badges else ""
                duration = f" · {trace.duration_ms} ms" if getattr(trace, "duration_ms", 0) else ""
                step_label = f"Step {trace.step_index or offset}"
                item_text = f"{step_label} · {name}{badge}{duration}: {trace.output_summary}"
                preview = metadata.get("diff_preview") if metadata else None
                if isinstance(preview, str) and preview.strip():
                    first_line = preview.strip().splitlines()[0]
                    snippet = first_line[:80]
                    item_text = f"{item_text} – {snippet}"
                if QListWidgetItem is not None:
                    QListWidgetItem(item_text, widget)
                else:
                    widget.addItem(item_text)
        finally:
            widget.blockSignals(False)

    def _update_tool_trace_header(self, *, total_steps: int, start_index: int) -> None:
        label = self._tool_trace_label
        if label is None:
            return
        if total_steps <= 0:
            text = "Tool Activity"
        elif total_steps == 1:
            text = "Tool Activity – Step 1"
        else:
            first_visible = start_index + 1
            if first_visible == total_steps:
                text = f"Tool Activity – Step {total_steps}"
            else:
                text = f"Tool Activity – Steps {first_visible}–{total_steps}"
        try:
            label.setText(text)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _refresh_composer_enablement(self) -> None:
        widget = self._composer_widget
        if widget is None:
            return
        try:
            widget.setReadOnly(self._ai_running)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass
        try:
            widget.setEnabled(not self._ai_running)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _refresh_action_button_state(self) -> None:
        button = self._send_button
        if button is None:
            return
        label = self._action_button_busy_text if self._ai_running else self._send_button_idle_text
        tooltip = self._action_button_busy_tooltip if self._ai_running else self._send_button_idle_tooltip
        try:
            button.setText(label)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass
        try:
            button.setToolTip(tooltip)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass
        try:
            button.setEnabled(True)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _coerce_button_text(self, button: Any) -> str:
        if button is None:
            return ""
        getter = getattr(button, "text", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:  # pragma: no cover - Qt defensive guard
                return ""
            if value is None:
                return ""
            return str(value)
        return ""

    def _coerce_button_tooltip(self, button: Any) -> str:
        if button is None:
            return ""
        getter = getattr(button, "toolTip", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:  # pragma: no cover - Qt defensive guard
                return ""
            if value is None:
                return ""
            return str(value)
        return ""

    def _configure_tool_trace_widget(self) -> None:
        widget = self._tool_trace_widget
        if widget is None or Qt is None or QMenu is None:
            return
        try:
            widget.setContextMenuPolicy(Qt.CustomContextMenu)
        except Exception:  # pragma: no cover - Qt defensive guard
            return

        try:
            widget.customContextMenuRequested.connect(self._handle_tool_trace_context_menu)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _sync_tool_trace_visibility(self) -> None:
        label = self._tool_trace_label
        widget = self._tool_trace_widget
        for target in (label, widget):
            if target is None:
                continue
            try:
                target.setVisible(self._tool_activity_visible)
            except Exception:  # pragma: no cover - depends on Qt availability
                pass

    def _handle_tool_trace_context_menu(self, point: Any) -> None:
        widget = self._tool_trace_widget
        if widget is None or QMenu is None:
            return
        try:
            item = widget.itemAt(point)
        except Exception:  # pragma: no cover - Qt defensive guard
            item = None
        if item is None:
            return
        try:
            row = widget.row(item)
        except Exception:  # pragma: no cover - Qt defensive guard
            row = -1
        trace = self._visible_tool_traces[row] if 0 <= row < len(self._visible_tool_traces) else None
        if trace is None:
            return
        details = self._format_tool_trace_details(trace)
        try:
            menu = QMenu(widget)
            copy_action = menu.addAction("Copy tool call details")
            global_pos = widget.mapToGlobal(point)
            selected = menu.exec(global_pos)
            if selected == copy_action:
                self.copy_text_to_clipboard(details)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _format_tool_trace_details(self, trace: ToolTrace) -> str:
        def _normalize(value: Optional[str]) -> str:
            text = (value or "").strip()
            return text if text else "(empty)"

        def _normalize_block(value: Optional[str]) -> str:
            if value is None:
                return "(empty)"
            text = str(value)
            return text if text.strip() else "(empty)"

        metadata = getattr(trace, "metadata", None) or {}
        raw_input = metadata.get("raw_input")
        raw_output = metadata.get("raw_output")

        def _format_input() -> str:
            if raw_input is not None:
                return _normalize_block(raw_input)
            return _normalize(trace.input_summary)

        def _format_output() -> str:
            if raw_output is not None:
                return _normalize_block(raw_output)
            return _normalize(trace.output_summary)

        lines = [
            f"Tool: {trace.name or '(unknown)'}",
            f"Input: {_format_input()}",
            f"Output: {_format_output()}",
            f"Duration: {trace.duration_ms} ms",
        ]

        step_index = trace.step_index or metadata.get("step_index")
        if isinstance(step_index, int) and step_index > 0:
            lines.insert(0, f"Step: {step_index}")

        before_text = metadata.get("text_before")
        after_text = metadata.get("text_after")
        if before_text is not None or after_text is not None:
            lines.extend(
                [
                    "",
                    "Replaced text:",
                    _normalize_block(before_text),
                    "",
                    "New text:",
                    _normalize_block(after_text),
                ]
            )

        diff_preview = metadata.get("diff_preview")
        if diff_preview:
            lines.extend(["", "Diff preview:", _normalize_block(str(diff_preview))])

        pointer = metadata.get("pointer")
        instructions = metadata.get("pointer_instructions")
        if isinstance(pointer, dict):
            pointer_id = pointer.get("pointer_id") or "(unknown)"
            lines.extend(["", f"Compacted pointer: {pointer_id}"])
            summary_text = pointer.get("display_text")
            if summary_text:
                lines.extend(["", "Pointer summary:", _normalize_block(str(summary_text))])
            instructions = instructions or pointer.get("rehydrate_instructions")
        if instructions:
            lines.extend(["", "Rehydrate instructions:", _normalize_block(str(instructions))])

        return "\n".join(lines)

    def _resolve_tool_trace(self, index: Optional[int]) -> Optional[ToolTrace]:
        if not self._tool_traces:
            return None
        if index is None:
            return self._tool_traces[-1]
        total = len(self._tool_traces)
        if -total <= index < total:
            return self._tool_traces[index]
        return None

    def _render_message_text(self, message: ChatMessage) -> str:
        prefix = message.role.upper()
        return f"[{prefix}] {message.content.strip()}"

    def _tooltip_label_for_message(self, message: ChatMessage) -> str:
        role = (message.role or "").lower()
        if role == "user":
            return "user message"
        if role == "assistant":
            return "AI message"
        if role == "system":
            return "system message"
        if role == "tool":
            return "tool message"
        return "message"

    def _create_message_bubble(self, message: ChatMessage, viewport_width: int | None = None) -> Any:
        if QFrame is None or QLabel is None or QHBoxLayout is None or QVBoxLayout is None:
            return None
        container = QFrame(self._history_widget)
        container.setObjectName("tb-chat-bubble-container")
        row_layout = QHBoxLayout(container)
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.setSpacing(4)

        bubble = QFrame(container)
        bubble.setObjectName(f"tb-chat-bubble-{message.role}")
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(12, 8, 12, 8)
        bubble_layout.setSpacing(4)

        text_label = QLabel(message.content.strip() or "\u200b", bubble)
        text_label.setWordWrap(True)
        bubble_layout.addWidget(text_label)

        if message.metadata:
            meta_text = ", ".join(f"{key}: {value}" for key, value in message.metadata.items())
            meta_label = QLabel(meta_text, bubble)
            meta_label.setStyleSheet("color: rgba(255, 255, 255, 0.65); font-size: 10px;")
            try:
                meta_label.setWordWrap(True)
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
            bubble_layout.addWidget(meta_label)

        bubble_max_width = 420
        if viewport_width:
            safe_width = max(120, viewport_width - 24)
            bubble_max_width = min(bubble_max_width, safe_width)
        bubble.setMaximumWidth(bubble_max_width)
        if message.role == "user":
            bubble.setStyleSheet("background-color: #2d7dff; color: white; border-radius: 16px;")
            row_layout.addStretch(1)
            row_layout.addWidget(bubble, 0)
        else:
            bubble.setStyleSheet("background-color: #2f333a; color: #f5f5f5; border-radius: 16px;")
            row_layout.addWidget(bubble, 0)
            row_layout.addStretch(1)

        self._attach_bubble_context_menu(bubble, message)

        return container

    def _attach_bubble_context_menu(self, bubble: Any, message: ChatMessage) -> None:
        if Qt is None or QMenu is None:
            return
        try:
            bubble.setContextMenuPolicy(Qt.CustomContextMenu)
        except Exception:  # pragma: no cover - Qt defensive guard
            return

        def _open_menu(point: Any, bubble_ref: Any = bubble, message_ref: ChatMessage = message) -> None:
            self._show_bubble_context_menu(bubble_ref, point, message_ref)

        try:
            bubble.customContextMenuRequested.connect(_open_menu)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _show_bubble_context_menu(self, bubble: Any, point: Any, message: ChatMessage | str) -> None:
        if QMenu is None:
            return
        text = message.content if isinstance(message, ChatMessage) else str(message)
        if text is None:
            text = ""
        try:
            menu = QMenu(bubble)
            copy_action = menu.addAction("Copy message")
            global_pos = bubble.mapToGlobal(point)
            selected = menu.exec(global_pos)
            if selected == copy_action:
                self.copy_text_to_clipboard(text)
        except Exception:  # pragma: no cover - Qt defensive guard
            pass

    def _toggle_suggestion_panel(self) -> None:
        widget = self._suggestion_widget
        has_suggestions = bool(self._suggestions)
        if not has_suggestions:
            if widget is not None:
                widget.setVisible(False)
            if self._suggestion_panel_open:
                self._suggestion_panel_open = False
                self._emit_suggestion_panel_event(False)
            return
        if widget is None:
            self._suggestion_panel_open = not self._suggestion_panel_open
            self._emit_suggestion_panel_event(self._suggestion_panel_open)
            return
        self._suggestion_panel_open = not widget.isVisible()
        widget.setVisible(self._suggestion_panel_open)
        self._emit_suggestion_panel_event(self._suggestion_panel_open)

    def _emit_suggestion_panel_event(self, is_open: bool) -> None:
        for listener in list(self._suggestion_panel_listeners):
            listener(is_open)

    def eventFilter(self, obj: Any, event: Any) -> bool:  # type: ignore[override]
        if obj is self._composer_widget:
            event_type = self._extract_event_type(event)
            keypress_code = self._coerce_int(getattr(QEvent, "KeyPress", None), FALLBACK_KEY_PRESS_EVENT)
            if event_type is not None and keypress_code is not None and event_type == keypress_code:
                key_code = self._extract_event_code(event, "key")
                modifier_code = self._extract_event_code(event, "modifiers")
                if self._handle_composer_key_event(key_code, modifier_code):
                    try:
                        event.accept()
                    except Exception:  # pragma: no cover - Qt defensive guard
                        pass
                    return True
        parent_filter = getattr(super(), "eventFilter", None)
        if callable(parent_filter):
            try:
                return bool(parent_filter(obj, event))
            except Exception:  # pragma: no cover - Qt defensive guard
                return False
        return False

    def _handle_composer_key_event(self, key: Optional[int], modifiers: Optional[int]) -> bool:
        if self._ai_running:
            return False
        if key is None or modifiers is None:
            return False

        enter_keys: set[int] = set(FALLBACK_ENTER_KEYS)
        qt_return = self._coerce_int(getattr(Qt, "Key_Return", None))
        if qt_return is not None:
            enter_keys.add(qt_return)
        qt_enter = self._coerce_int(getattr(Qt, "Key_Enter", None))
        if qt_enter is not None:
            enter_keys.add(qt_enter)

        if key not in enter_keys:
            return False

        shift_mask = self._coerce_int(getattr(Qt, "ShiftModifier", None), FALLBACK_SHIFT_MODIFIER)
        modifiers_int = self._coerce_int(modifiers, 0) or 0

        if shift_mask and (modifiers_int & shift_mask):
            return False

        self._handle_send_clicked()
        return True

    def _extract_event_type(self, event: Any) -> Optional[int]:
        if event is None:
            return None
        getter = getattr(event, "type", None)
        if not callable(getter):
            return None
        try:
            value = getter()
        except Exception:  # pragma: no cover - Qt defensive guard
            return None
        return self._coerce_int(value)

    def _extract_event_code(self, event: Any, attr: str) -> Optional[int]:
        if event is None:
            return None
        getter = getattr(event, attr, None)
        if not callable(getter):
            return None
        try:
            value = getter()
        except Exception:  # pragma: no cover - Qt defensive guard
            return None
        return self._coerce_int(value)

    @staticmethod
    def _coerce_int(value: Any, default: Optional[int] = None) -> Optional[int]:
        if value is None:
            return default
        try:
            return int(value)
        except Exception:
            pass
        candidate = getattr(value, "value", None)
        if candidate is not None:
            try:
                return int(candidate)
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
        return default

    def resizeEvent(self, event: Any) -> None:  # type: ignore[override]
        """Ensure chat bubbles adapt to the latest viewport width on resize."""

        parent_resize = getattr(super(), "resizeEvent", None)
        if callable(parent_resize):
            try:
                parent_resize(event)
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
        self._refresh_history_widget()

    # Qt callbacks ----------------------------------------------------
    def _handle_action_button_clicked(self) -> None:
        if self._ai_running:
            self._emit_stop_ai_request()
            return
        self._handle_send_clicked()

    def _handle_send_clicked(self) -> None:
        try:
            self.send_prompt()
        except ValueError:
            pass

    def _handle_new_chat_clicked(self) -> None:
        self.start_new_chat()

    def _handle_suggestion_clicked(self, item: Any) -> None:
        try:
            text = str(item.text())
        except Exception:  # pragma: no cover - Qt defensive guard
            return
        self.set_composer_text(text)
        if self._suggestion_widget is not None:
            try:
                self._suggestion_widget.setVisible(False)
            except Exception:  # pragma: no cover - Qt defensive guard
                pass
        self._suggestion_panel_open = False

