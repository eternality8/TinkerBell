"""Helpers for building AI tool contexts and lazy tool factories."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ...ai.tools.tool_wiring import ToolWiringContext
from ...editor.selection_gateway import SelectionSnapshotProvider

LOGGER = logging.getLogger(__name__)


class _AIClientProviderAdapter:
    """Adapter that implements AIClientProvider protocol using controller.
    
    This allows the tool wiring system to access the AIClient for
    subagent execution (WS9).
    """
    
    def __init__(self, controller_resolver: Callable[[], Any]) -> None:
        self._controller_resolver = controller_resolver
    
    def get_ai_client(self) -> Any:
        """Get the AIClient from the controller."""
        controller = self._controller_resolver()
        if controller is None:
            return None
        return getattr(controller, "client", None)


class _DocumentCreatorAdapter:
    """Adapter that implements DocumentCreator protocol using editor/workspace.
    
    This adapter can work with either a TabbedEditorWidget (preferred, for full
    Qt UI integration) or a DocumentWorkspace (for tests without Qt).
    """
    
    def __init__(self, editor_or_workspace: Any) -> None:
        self._editor_or_workspace = editor_or_workspace
    
    def create_document(
        self,
        title: str,
        content: str = "",
        file_type: str | None = None,
    ) -> str:
        """Create a new document tab and return its tab_id."""
        from ...editor.document_model import DocumentState, DocumentMetadata
        
        # Create document state with content
        doc = DocumentState(
            text=content,
            metadata=DocumentMetadata(path=None, language=file_type or "markdown"),
        )
        
        # Create the tab using create_tab() - works for both
        # TabbedEditorWidget and DocumentWorkspace
        tab = self._editor_or_workspace.create_tab(document=doc, title=title, make_active=True)
        return tab.id
    
    def document_exists(self, title: str) -> tuple[bool, str | None]:
        """Check if a document with the given title already exists."""
        # Get workspace for iteration - either directly or via .workspace property
        workspace = getattr(self._editor_or_workspace, 'workspace', self._editor_or_workspace)
        for tab in workspace.iter_tabs():
            if tab.title == title:
                return (True, tab.id)
        return (False, None)


@dataclass(slots=True)
class ToolProvider:
    """Lazily constructs document-aware AI tools and registry contexts."""

    controller_resolver: Callable[[], Any]
    bridge: Any
    workspace: Any
    selection_gateway: SelectionSnapshotProvider
    editor: Any = None  # TabbedEditorWidget for Qt-aware tab creation

    def build_tool_wiring_context(
        self,
    ) -> ToolWiringContext:
        """Return a tool wiring context reflecting the current runtime state."""
        # Use editor (TabbedEditorWidget) for document creation if available,
        # otherwise fall back to workspace (for tests without Qt)
        doc_creator_target = self.editor if self.editor is not None else self.workspace

        return ToolWiringContext(
            controller=self.controller_resolver(),
            bridge=self.bridge,
            workspace=self.workspace,
            selection_gateway=self.selection_gateway,
            document_creator=_DocumentCreatorAdapter(doc_creator_target),
            ai_client_provider=_AIClientProviderAdapter(self.controller_resolver),
        )


__all__ = ["ToolProvider"]
