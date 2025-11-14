"""Memory buffer stubs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class ConversationMemory:
    """Token-aware rolling buffer placeholder."""

    messages: List[str] = field(default_factory=list)
    max_messages: int = 20

    def add(self, content: str) -> None:
        self.messages.append(content)
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

