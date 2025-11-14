"""Chat panel widget stub."""

from __future__ import annotations

from typing import List, Optional

from .message_model import ChatMessage

class QWidget:  # pragma: no cover - placeholder base class
    """Fallback placeholder for PySide6 QWidget."""

    def __init__(self, *args, **kwargs) -> None:  # noqa: D401 - simple shim
        pass


class ChatPanel(QWidget):
    """Pane showing chat history and composer controls."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:  # type: ignore[name-defined]
        super().__init__(parent)
        self._messages: List[ChatMessage] = []

    def append_user_message(self, content: str) -> None:
        """Add a user-authored message to the panel."""

        self._messages.append(ChatMessage(role="user", content=content))

    def append_ai_message(self, message: ChatMessage) -> None:
        """Add an AI-authored message to the panel."""

        self._messages.append(message)

    def history(self) -> List[ChatMessage]:
        """Return the recorded message history."""

        return list(self._messages)
