"""Tests for the simplified ToolProvider."""

from __future__ import annotations

from tinkerbell.editor.selection_gateway import SelectionSnapshot
from tinkerbell.ui.tools.provider import ToolProvider


class _GatewayStub:
    def capture(self, *, tab_id: str | None = None) -> SelectionSnapshot:
        return SelectionSnapshot(
            tab_id=tab_id,
            document_id="doc",
            content_hash="hash",
            selection_start=0,
            selection_end=0,
            length=0,
            line_start_offsets=(0,),
        )


class _WorkspaceStub:
    def find_document_by_id(self, document_id: str) -> None:
        return None

    def active_document(self) -> None:
        return None


def _make_provider() -> ToolProvider:
    return ToolProvider(
        controller_resolver=lambda: object(),
        bridge=object(),
        workspace=_WorkspaceStub(),
        selection_gateway=_GatewayStub(),
    )


def test_build_tool_context_exposes_dependencies() -> None:
    """ToolProvider builds a valid tool wiring context."""
    provider = _make_provider()

    context = provider.build_tool_wiring_context()

    assert context.controller is not None
    assert context.bridge is not None
    assert context.selection_gateway is provider.selection_gateway
