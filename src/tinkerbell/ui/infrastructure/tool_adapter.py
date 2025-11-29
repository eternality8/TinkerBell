"""Tool adapter for building tool wiring contexts and registering tools.

This module extracts tool registration logic from tools/provider.py into
a clean adapter that provides tool wiring context and registration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, TYPE_CHECKING

from ..events import EventBus

if TYPE_CHECKING:  # pragma: no cover - imports for type checking only
    from ...ai.tools.tool_wiring import ToolWiringContext, ToolRegistrationResult
    from ...editor.selection_gateway import SelectionSnapshotProvider

_LOGGER = logging.getLogger(__name__)


class _AIClientProviderAdapter:
    """Adapter that implements AIClientProvider protocol using controller.

    This allows the tool wiring system to access the AIClient for
    subagent execution.
    """

    __slots__ = ("_controller_resolver",)

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

    __slots__ = ("_editor_or_workspace",)

    def __init__(self, editor_or_workspace: Any) -> None:
        self._editor_or_workspace = editor_or_workspace

    def create_document(
        self,
        title: str,
        content: str = "",
        file_type: str | None = None,
    ) -> str:
        """Create a new document tab and return its tab_id."""
        try:
            from ...editor.document_model import DocumentMetadata, DocumentState
        except ImportError:  # pragma: no cover - dependency guard
            _LOGGER.warning("Document model not available")
            raise

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
        workspace = getattr(self._editor_or_workspace, "workspace", self._editor_or_workspace)
        for tab in workspace.iter_tabs():
            if tab.title == title:
                return (True, tab.id)
        return (False, None)


@dataclass(slots=True)
class ToolAdapter:
    """Adapter for building tool wiring contexts and registering tools.

    This adapter encapsulates the tool provider functionality, providing
    a clean interface for building tool wiring contexts and registering
    tools with the AI orchestrator.

    The adapter is designed to work with the event-driven architecture,
    though it doesn't emit events directly (tool registration results
    are returned synchronously).

    Attributes:
        controller_resolver: Callable that returns the current AI controller.
        bridge: The workspace bridge for document operations.
        workspace: The document workspace for tab management.
        selection_gateway: Provider for selection snapshots.
        editor: Optional TabbedEditorWidget for Qt-aware tab creation.
        event_bus: Optional event bus for future event emission.
    """

    controller_resolver: Callable[[], Any]
    bridge: Any
    workspace: Any
    selection_gateway: SelectionSnapshotProvider
    editor: Any = None
    event_bus: EventBus | None = None

    def build_wiring_context(self) -> ToolWiringContext:
        """Build a tool wiring context reflecting the current runtime state.

        Returns:
            A ToolWiringContext with all dependencies configured.
        """
        try:
            from ...ai.tools.tool_wiring import ToolWiringContext
        except ImportError:  # pragma: no cover - dependency guard
            _LOGGER.warning("Tool wiring module not available")
            raise

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

    def register_tools(self) -> ToolRegistrationResult:
        """Register all AI tools with the controller.

        This method builds the wiring context and registers all new tools
        (WS1-6 tools) with the AI controller.

        Returns:
            A ToolRegistrationResult indicating success/failure of registration.
        """
        try:
            from ...ai.tools.tool_wiring import register_new_tools
        except ImportError:  # pragma: no cover - dependency guard
            _LOGGER.warning("Tool wiring module not available")
            raise

        ctx = self.build_wiring_context()
        result = register_new_tools(ctx)

        _LOGGER.info(
            "Tool registration complete: %d registered, %d failed, %d skipped",
            len(result.registered),
            len(result.failed),
            len(result.skipped),
        )

        return result

    def unregister_tools(self, tool_names: list[str]) -> list[str]:
        """Unregister specific tools from the controller.

        Args:
            tool_names: Names of tools to unregister.

        Returns:
            List of successfully unregistered tool names.
        """
        try:
            from ...ai.tools.tool_wiring import unregister_tools
        except ImportError:  # pragma: no cover - dependency guard
            _LOGGER.warning("Tool wiring module not available")
            return []

        controller = self.controller_resolver()
        if controller is None:
            return []

        return unregister_tools(controller, tool_names)

    def configure_subagent_orchestrator(
        self,
        analyze_tool: Any | None = None,
        transform_tool: Any | None = None,
    ) -> Any | None:
        """Configure subagent tools with LLM orchestrator.

        This method can be called after initial tool registration to
        add LLM integration to subagent tools.

        Args:
            analyze_tool: Optional AnalyzeDocumentTool to configure.
            transform_tool: Optional TransformDocumentTool to configure.

        Returns:
            The created SubagentOrchestrator, or None if not available.
        """
        try:
            from ...ai.tools.tool_wiring import configure_subagent_orchestrator
        except ImportError:  # pragma: no cover - dependency guard
            _LOGGER.warning("Tool wiring module not available")
            return None

        # Get AI client from controller
        controller = self.controller_resolver()
        if controller is None:
            _LOGGER.debug("No controller available for subagent configuration")
            return None

        ai_client = getattr(controller, "client", None)
        if ai_client is None:
            _LOGGER.debug("No AI client available for subagent configuration")
            return None

        try:
            return configure_subagent_orchestrator(
                ai_client,
                analyze_tool=analyze_tool,
                transform_tool=transform_tool,
            )
        except Exception as exc:
            _LOGGER.warning("Failed to configure subagent orchestrator: %s", exc)
            return None


__all__ = ["ToolAdapter"]
