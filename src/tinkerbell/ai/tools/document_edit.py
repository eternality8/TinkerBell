"""Tool applying edits produced by the AI agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


class EditDirective(Protocol):
    """Protocol describing edit directives."""

    action: str
    content: str


class Bridge(Protocol):
    """Subset of the document bridge interface used by the tool."""

    def queue_edit(self, directive: EditDirective | Mapping[str, Any]) -> None:
        ...


@dataclass(slots=True)
class DocumentEditTool:
    """Apply validated edit directives via the bridge."""

    bridge: Bridge

    def run(self, directive: EditDirective | Mapping[str, Any]) -> str:
        self.bridge.queue_edit(directive)
        return "queued"

