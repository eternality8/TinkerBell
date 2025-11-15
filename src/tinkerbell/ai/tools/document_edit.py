"""Tool applying edits produced by the AI agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from ...chat.commands import parse_agent_payload
from ...chat.message_model import EditDirective


class Bridge(Protocol):
    """Subset of the document bridge interface used by the tool."""

    def queue_edit(self, directive: EditDirective | Mapping[str, Any]) -> None:
        ...

    @property
    def last_diff_summary(self) -> str | None:
        ...

    @property
    def last_snapshot_version(self) -> str | None:
        ...


DirectiveInput = EditDirective | Mapping[str, Any] | str | bytes


@dataclass(slots=True)
class DocumentEditTool:
    """Apply validated edit directives via the bridge."""

    bridge: Bridge

    def run(self, directive: DirectiveInput | None = None, **fields: Any) -> str:
        payload = self._coerce_input(self._resolve_input(directive, fields))
        self.bridge.queue_edit(payload)

        diff = getattr(self.bridge, "last_diff_summary", None)
        version = getattr(self.bridge, "last_snapshot_version", None)
        if diff and version:
            return f"applied: {diff} (version={version})"
        if diff:
            return f"applied: {diff}"
        if version:
            return f"queued (version={version})"
        return "queued"

    @staticmethod
    def _resolve_input(directive: DirectiveInput | None, fields: Mapping[str, Any]) -> DirectiveInput:
        if directive is not None and fields:
            raise ValueError("Provide either a directive argument or keyword fields, not both.")
        if directive is None:
            if not fields:
                raise ValueError("Directive payload is required.")
            return dict(fields)
        return directive

    @staticmethod
    def _coerce_input(directive: DirectiveInput) -> EditDirective | Mapping[str, Any]:
        if isinstance(directive, (str, bytes)):
            return parse_agent_payload(directive)
        if isinstance(directive, (Mapping, EditDirective)):
            return directive
        raise TypeError("Directive must be a mapping, EditDirective, or JSON string.")

