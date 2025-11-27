"""Helpers for building AI tool contexts and lazy tool factories."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ...ai.tools.tool_wiring import ToolWiringContext
from ...editor.selection_gateway import SelectionSnapshotProvider

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ToolProvider:
    """Lazily constructs document-aware AI tools and registry contexts."""

    controller_resolver: Callable[[], Any]
    bridge: Any
    workspace: Any
    selection_gateway: SelectionSnapshotProvider

    def build_tool_wiring_context(
        self,
    ) -> ToolWiringContext:
        """Return a tool wiring context reflecting the current runtime state."""

        return ToolWiringContext(
            controller=self.controller_resolver(),
            bridge=self.bridge,
            workspace=self.workspace,
            selection_gateway=self.selection_gateway,
        )


__all__ = ["ToolProvider"]
