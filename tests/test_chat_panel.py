"""Chat panel behavior tests running in headless mode."""

from typing import Any
import types

import pytest

import tinkerbell.chat.chat_panel as chat_panel
from tinkerbell.chat.chat_panel import ChatPanel, ComposerContext
from tinkerbell.chat.message_model import ChatMessage, ToolTrace


def _ensure_qapp() -> None:
    """Initialize a QApplication when PySide6 is available."""

    try:
        from PySide6.QtWidgets import QApplication  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - PySide6 optional
        return

    if QApplication.instance() is None:  # pragma: no cover - depends on PySide6
        QApplication([])


def _make_panel() -> ChatPanel:
    _ensure_qapp()
    return ChatPanel()


def test_tool_activity_panel_hidden_by_default():
    panel = _make_panel()

    assert panel.tool_activity_visible is False


def test_tool_activity_visibility_toggle():
    panel = ChatPanel(show_tool_activity_panel=True)

    assert panel.tool_activity_visible is True

    panel.set_tool_activity_visibility(False)
    assert panel.tool_activity_visible is False


def test_chat_panel_appends_user_and_ai_messages():
    panel = _make_panel()
    panel.append_user_message("Hello", selection_summary="lines 1-2")
    assistant = ChatMessage(role="assistant", content="Hi there")
    panel.append_ai_message(assistant)

    history = panel.history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].metadata["selection_summary"] == "lines 1-2"
    assert history[1].content == "Hi there"


def test_chat_panel_streaming_ai_messages_merge_chunks():
    panel = _make_panel()
    panel.append_ai_message(ChatMessage(role="assistant", content="Hello"), streaming=True)
    panel.append_ai_message(ChatMessage(role="assistant", content=" world"), streaming=True)
    final_message = panel.append_ai_message(
        ChatMessage(role="assistant", content="!"), streaming=False
    )

    history = panel.history()
    assert len(history) == 1
    assert history[0].content == "Hello world!"
    assert final_message is history[0]


def test_chat_panel_message_tooltips_are_generic():
    panel = _make_panel()
    assert panel._tooltip_label_for_message(ChatMessage(role="user", content="Hello")) == "user message"
    assert (
        panel._tooltip_label_for_message(ChatMessage(role="assistant", content="Hi"))
        == "AI message"
    )
    assert (
        panel._tooltip_label_for_message(ChatMessage(role="system", content="Notice"))
        == "system message"
    )
    assert (
        panel._tooltip_label_for_message(ChatMessage(role="tool", content="details"))
        == "tool message"
    )


def test_chat_panel_notifies_request_listeners_and_clears_composer():
    panel = _make_panel()
    captured: list[tuple[str, dict]] = []

    def listener(prompt: str, metadata: dict) -> None:
        captured.append((prompt, metadata))

    panel.add_request_listener(listener)
    panel.set_composer_text("Summarize selection", context=ComposerContext("intro"))
    sent = panel.send_prompt()

    assert sent == "Summarize selection"
    assert panel.composer_text == ""
    assert captured == [("Summarize selection", {"selection_summary": "intro"})]


def test_chat_panel_show_tool_trace_attaches_to_latest_message():
    panel = _make_panel()
    msg = panel.append_ai_message(ChatMessage(role="assistant", content="Processing"))
    trace = ToolTrace(name="DocumentSnapshot", input_summary="full", output_summary="ok", duration_ms=12)
    panel.show_tool_trace(trace)

    assert msg.tool_traces[-1] is trace


def test_tool_traces_include_step_numbers_in_metadata():
    panel = _make_panel()
    first = panel.show_tool_trace(ToolTrace(name="snapshot", input_summary="all", output_summary="ok"))
    second = panel.show_tool_trace(ToolTrace(name="diff", input_summary="delta", output_summary="ready"))

    assert first.step_index == 1
    assert second.step_index == 2
    details = panel._format_tool_trace_details(second)
    assert "Step: 2" in details


def test_copy_tool_trace_details_includes_all_fields(monkeypatch):
    panel = _make_panel()
    trace = ToolTrace(
        name="search_api",
        input_summary="query=hello",
        output_summary="result=42",
        duration_ms=87,
    )
    panel.show_tool_trace(trace)

    class _Clipboard:
        def __init__(self) -> None:
            self.text = ""

        def setText(self, text: str) -> None:
            self.text = text

    class _QtAppStub:
        _clipboard = _Clipboard()

        @staticmethod
        def instance() -> object:
            return object()

        @staticmethod
        def clipboard() -> _Clipboard:
            return _QtAppStub._clipboard

    monkeypatch.setattr(chat_panel, "QApplication", _QtAppStub)

    copied = panel.copy_tool_trace_details(trace)

    assert copied is True
    text = panel.last_copied_text or ""
    assert "Tool: search_api" in text
    assert "Input: query=hello" in text
    assert "Output: result=42" in text
    assert "Duration: 87 ms" in text


def test_copy_tool_trace_details_includes_replacement_text(monkeypatch):
    panel = _make_panel()
    trace = ToolTrace(
        name="edit:replace",
        input_summary="range=(0, 5)",
        output_summary="Δ-2 chars",
        metadata={"text_before": "old text", "text_after": "new text"},
    )
    panel.show_tool_trace(trace)

    class _Clipboard:
        def __init__(self) -> None:
            self.text = ""

        def setText(self, text: str) -> None:
            self.text = text

    class _QtAppStub:
        _clipboard = _Clipboard()

        @staticmethod
        def instance() -> object:
            return object()

        @staticmethod
        def clipboard() -> _Clipboard:
            return _QtAppStub._clipboard

    monkeypatch.setattr(chat_panel, "QApplication", _QtAppStub)

    copied = panel.copy_tool_trace_details(trace)

    assert copied is True
    text = panel.last_copied_text or ""
    assert "Replaced text:" in text
    assert "old text" in text
    assert "New text:" in text
    assert "new text" in text


def test_copy_tool_trace_details_includes_diff_preview(monkeypatch):
    panel = _make_panel()
    trace = ToolTrace(
        name="edit:patch",
        input_summary="range=(0, 0)",
        output_summary="patch: +2 chars",
        metadata={"diff_preview": "@@ -1 +1 @@\n-old\n+new"},
    )
    panel.show_tool_trace(trace)

    class _Clipboard:
        def __init__(self) -> None:
            self.text = ""

        def setText(self, text: str) -> None:
            self.text = text

    class _QtAppStub:
        _clipboard = _Clipboard()

        @staticmethod
        def instance() -> object:
            return object()

        @staticmethod
        def clipboard() -> _Clipboard:
            return _QtAppStub._clipboard

    monkeypatch.setattr(chat_panel, "QApplication", _QtAppStub)

    copied = panel.copy_tool_trace_details(trace)

    assert copied is True
    assert "Diff preview:" in (panel.last_copied_text or "")


def test_copy_tool_trace_details_prefers_raw_metadata(monkeypatch):
    panel = _make_panel()
    long_input = '{"action":"replace","content":"' + ("A" * 60) + '"}'
    long_output = "Line 1\nLine 2 with details"
    trace = ToolTrace(
        name="edit:replace",
        input_summary="short",
        output_summary="summary",
        metadata={"raw_input": long_input, "raw_output": long_output},
    )
    panel.show_tool_trace(trace)

    class _Clipboard:
        def __init__(self) -> None:
            self.text = ""

        def setText(self, text: str) -> None:
            self.text = text

    class _QtAppStub:
        _clipboard = _Clipboard()

        @staticmethod
        def instance() -> object:
            return object()

        @staticmethod
        def clipboard() -> _Clipboard:
            return _QtAppStub._clipboard

    monkeypatch.setattr(chat_panel, "QApplication", _QtAppStub)

    copied = panel.copy_tool_trace_details(trace)

    assert copied is True
    text = panel.last_copied_text or ""
    assert long_input in text
    assert "Line 2 with details" in text


def test_copy_tool_trace_details_without_traces_returns_false():
    panel = _make_panel()

    assert panel.copy_tool_trace_details() is False


def test_chat_panel_suggestions_update_composer():
    panel = _make_panel()
    panel.set_suggestions(["Option A", "Option B"])

    selected = panel.select_suggestion(1)
    assert selected == "Option B"
    assert panel.composer_text == "Option B"

    with pytest.raises(IndexError):
        panel.select_suggestion(5)


def test_chat_panel_enter_key_sends_message():
    panel = _make_panel()
    captured: list[str] = []

    def _listener(prompt: str, metadata: dict[str, Any]) -> None:
        del metadata
        captured.append(prompt)

    panel.add_request_listener(_listener)
    panel.set_composer_text("Send via enter")

    handled = panel._handle_composer_key_event(chat_panel.FALLBACK_ENTER_KEYS[0], 0)

    assert handled is True
    assert captured == ["Send via enter"]
    assert panel.composer_text == ""


def test_chat_panel_shift_enter_inserts_newline():
    panel = _make_panel()
    panel.set_composer_text("First line")

    handled = panel._handle_composer_key_event(
        chat_panel.FALLBACK_ENTER_KEYS[0], chat_panel.FALLBACK_SHIFT_MODIFIER
    )

    assert handled is False
    assert panel.composer_text == "First line"


def test_chat_panel_set_selection_summary_updates_metadata():
    panel = _make_panel()
    panel.set_selection_summary("Intro", extras={"cursor": "top"})
    panel.set_composer_text("Summarize this")
    captured: list[dict[str, Any]] = []

    def _collect(prompt: str, metadata: dict[str, Any]) -> None:
        del prompt
        captured.append(metadata)

    panel.add_request_listener(_collect)
    panel.send_prompt()

    assert captured[0]["selection_summary"] == "Intro"
    assert captured[0]["cursor"] == "top"


def test_chat_panel_send_prompt_rejects_empty_text():
    panel = _make_panel()
    with pytest.raises(ValueError):
        panel.send_prompt("")


def test_chat_panel_start_new_chat_notifies_listeners():
    panel = _make_panel()
    called = []

    def _listener() -> None:
        called.append(True)

    panel.add_session_reset_listener(_listener)
    panel.start_new_chat()

    assert called == [True]


def test_chat_panel_start_new_chat_resets_state():
    panel = _make_panel()
    panel.append_user_message("Hi")
    panel.append_ai_message(ChatMessage(role="assistant", content="Hello again"))
    panel.set_composer_text("Draft reply", context=ComposerContext("intro"))
    panel.start_new_chat()

    assert panel.history() == []
    assert panel.composer_text == ""


def test_chat_panel_set_ai_running_updates_button_and_composer():
    panel = _make_panel()

    class _Composer:
        def __init__(self) -> None:
            self.readonly = False
            self.enabled = True

        def setReadOnly(self, value: bool) -> None:
            self.readonly = value

        def setEnabled(self, value: bool) -> None:
            self.enabled = value

    class _Button:
        def __init__(self) -> None:
            self._text = "Send"
            self._tooltip = "Send"
            self.enabled = True

        def text(self) -> str:
            return self._text

        def setText(self, value: str) -> None:
            self._text = value

        def toolTip(self) -> str:
            return self._tooltip

        def setToolTip(self, value: str) -> None:
            self._tooltip = value

        def setEnabled(self, value: bool) -> None:
            self.enabled = value

    panel._composer_widget = _Composer()
    panel._send_button = _Button()
    panel._send_button_idle_text = "Send"
    panel._send_button_idle_tooltip = "Send message"

    panel.set_ai_running(True)

    assert panel._composer_widget.readonly is True
    assert panel._composer_widget.enabled is False
    assert panel._send_button._text == "■"
    assert panel._send_button.enabled is True

    panel.set_ai_running(False)

    assert panel._composer_widget.readonly is False
    assert panel._composer_widget.enabled is True
    assert panel._send_button._text == "Send"


def test_chat_panel_stop_callback_invoked_when_running():
    panel = _make_panel()
    called: list[bool] = []

    panel.set_stop_ai_callback(lambda: called.append(True))
    panel.set_ai_running(True)
    panel._handle_action_button_clicked()

    assert called == [True]


def test_chat_panel_action_button_sends_when_idle(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel()
    called: list[bool] = []

    monkeypatch.setattr(panel, "_handle_send_clicked", lambda: called.append(True))

    panel._handle_action_button_clicked()

    assert called == [True]


def test_chat_panel_enter_key_ignored_when_ai_running(monkeypatch: pytest.MonkeyPatch) -> None:
    panel = _make_panel()
    called: list[bool] = []
    monkeypatch.setattr(panel, "_handle_send_clicked", lambda: called.append(True))

    panel.set_ai_running(True)

    handled = panel._handle_composer_key_event(chat_panel.FALLBACK_ENTER_KEYS[0], 0)

    assert handled is False
    assert called == []


def test_copy_text_to_clipboard_records_text_when_qt_missing(monkeypatch):
    monkeypatch.setattr(chat_panel, "QApplication", None)
    panel = ChatPanel()

    copied = panel.copy_text_to_clipboard("Example message")

    assert copied is False
    assert panel.last_copied_text == "Example message"


def test_chat_panel_resize_event_refreshes_history(monkeypatch):
    panel = ChatPanel()
    calls: list[bool] = []

    def _fake_refresh() -> None:
        calls.append(True)

    monkeypatch.setattr(panel, "_refresh_history_widget", _fake_refresh)

    panel.resizeEvent(object())

    assert calls == [True]


def test_copy_text_to_clipboard_uses_qt_clipboard(monkeypatch):
    panel = ChatPanel()

    class _Clipboard:
        def __init__(self) -> None:
            self.text = ""

        def setText(self, text: str) -> None:
            self.text = text

    class _QtAppStub:
        _clipboard = _Clipboard()

        @staticmethod
        def instance() -> object:
            return object()

        @staticmethod
        def clipboard() -> _Clipboard:
            return _QtAppStub._clipboard

    monkeypatch.setattr(chat_panel, "QApplication", _QtAppStub)

    assert panel.copy_text_to_clipboard("Persist me") is True
    assert _QtAppStub._clipboard.text == "Persist me"
    assert panel.last_copied_text == "Persist me"


def test_history_widget_reserves_bottom_padding(monkeypatch):
    panel = ChatPanel()

    class _StubHistory:
        def __init__(self) -> None:
            self.margins: tuple[int, int, int, int] | None = None
            self.scroll_mode: Any = None

        def setObjectName(self, name: str) -> None:
            self.object_name = name

        def setAlternatingRowColors(self, value: bool) -> None:
            self.alternating = value

        def setSpacing(self, spacing: int) -> None:
            self.spacing = spacing

        def setSelectionMode(self, mode: Any) -> None:
            self.selection_mode = mode

        def setHorizontalScrollBarPolicy(self, policy: Any) -> None:
            self.h_scroll_policy = policy

        def setWordWrap(self, enabled: bool) -> None:
            self.word_wrap = enabled

        def setViewportMargins(self, left: int, top: int, right: int, bottom: int) -> None:
            self.margins = (left, top, right, bottom)

        def setVerticalScrollMode(self, mode: Any) -> None:
            self.scroll_mode = mode

    panel._history_widget = _StubHistory()
    selection_stub = types.SimpleNamespace(NoSelection="noselect", ScrollPerPixel="per-pixel")
    qt_stub = types.SimpleNamespace(ScrollBarAlwaysOff="always-off")
    monkeypatch.setattr(chat_panel, "QAbstractItemView", selection_stub)
    monkeypatch.setattr(chat_panel, "Qt", qt_stub)

    panel._configure_history_widget()

    assert panel._history_widget.margins == (0, 0, 0, panel.HISTORY_BOTTOM_PADDING)
    assert panel._history_widget.scroll_mode == selection_stub.ScrollPerPixel
