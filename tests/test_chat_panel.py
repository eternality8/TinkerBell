"""Chat panel behavior tests running in headless mode."""

from typing import Any

import pytest

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


def test_chat_panel_suggestions_update_composer():
    panel = _make_panel()
    panel.set_suggestions(["Option A", "Option B"])

    selected = panel.select_suggestion(1)
    assert selected == "Option B"
    assert panel.composer_text == "Option B"

    with pytest.raises(IndexError):
        panel.select_suggestion(5)


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
