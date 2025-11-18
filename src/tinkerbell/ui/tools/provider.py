"""Helpers for building AI tool contexts and lazy tool factories."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, TYPE_CHECKING

from ...ai.tools.document_apply_patch import DocumentApplyPatchTool
from ...ai.tools.document_find_sections import DocumentFindSectionsTool
from ...ai.tools.document_outline import DocumentOutlineTool
from ...ai.tools.document_plot_state import DocumentPlotStateTool
from ...ai.tools.registry import ToolRegistryContext

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - typing aides only
    from ...ai.memory import DocumentEmbeddingIndex
    from ...ai.memory.plot_state import DocumentPlotStateStore
    from ...ai.services import OutlineBuilderWorker
    from ...editor.document_model import DocumentState

    OutlineWorker = OutlineBuilderWorker
    EmbeddingIndexType = DocumentEmbeddingIndex
    PlotStateStoreType = DocumentPlotStateStore
else:  # pragma: no cover - runtime fallback
    OutlineWorker = Any
    EmbeddingIndexType = Any
    PlotStateStoreType = Any


DocumentLookup = Callable[[str], Any]
DocumentProvider = Callable[[], Any]
EmbeddingIndexResolver = Callable[[], EmbeddingIndexType | None]
OutlineWorkerResolver = Callable[[], OutlineWorker | None]
OutlineMemoryResolver = Callable[[], Any]
PlotStateStoreResolver = Callable[[], PlotStateStoreType | None]


@dataclass(slots=True)
class ToolProvider:
    """Lazily constructs document-aware AI tools and registry contexts."""

    controller_resolver: Callable[[], Any]
    bridge: Any
    document_lookup: DocumentLookup
    active_document_provider: DocumentProvider
    outline_worker_resolver: OutlineWorkerResolver
    outline_memory_resolver: OutlineMemoryResolver
    embedding_index_resolver: EmbeddingIndexResolver
    outline_digest_resolver: Callable[[str | None], str | None]
    directive_schema_provider: Callable[[], Mapping[str, Any]]
    plot_state_store_resolver: PlotStateStoreResolver
    phase3_outline_enabled: bool = False
    plot_scaffolding_enabled: bool = False

    _outline_tool: DocumentOutlineTool | None = field(default=None, init=False)
    _find_sections_tool: DocumentFindSectionsTool | None = field(default=None, init=False)
    _plot_state_tool: DocumentPlotStateTool | None = field(default=None, init=False)

    def build_tool_registry_context(
        self,
        *,
        auto_patch_consumer: Callable[[DocumentApplyPatchTool], None] | None = None,
    ) -> ToolRegistryContext:
        """Return a tool registry context reflecting the current runtime state."""

        return ToolRegistryContext(
            controller=self.controller_resolver(),
            bridge=self.bridge,
            outline_digest_resolver=self.outline_digest_resolver,
            directive_schema_provider=self.directive_schema_provider,
            phase3_outline_enabled=self.phase3_outline_enabled,
            plot_scaffolding_enabled=self.plot_scaffolding_enabled,
            ensure_outline_tool=self.ensure_outline_tool,
            ensure_find_sections_tool=self.ensure_find_sections_tool,
            ensure_plot_state_tool=self.ensure_plot_state_tool,
            auto_patch_consumer=auto_patch_consumer,
        )

    # ------------------------------------------------------------------
    # Feature flag updates & cache management
    # ------------------------------------------------------------------
    def set_phase3_outline_enabled(self, enabled: bool) -> None:
        if self.phase3_outline_enabled == enabled:
            return
        self.phase3_outline_enabled = enabled
        if not enabled:
            self.reset_outline_tools()

    def set_plot_scaffolding_enabled(self, enabled: bool) -> None:
        if self.plot_scaffolding_enabled == enabled:
            return
        self.plot_scaffolding_enabled = enabled
        if not enabled:
            self.reset_plot_state_tool()

    def reset_outline_tools(self) -> None:
        self._outline_tool = None
        self._find_sections_tool = None

    def reset_plot_state_tool(self) -> None:
        self._plot_state_tool = None

    def peek_outline_tool(self) -> DocumentOutlineTool | None:
        return self._outline_tool

    # ------------------------------------------------------------------
    # Tool factories
    # ------------------------------------------------------------------
    def ensure_outline_tool(self) -> DocumentOutlineTool | None:
        if not self.phase3_outline_enabled:
            return None
        if self._outline_tool is not None:
            return self._outline_tool

        def _pending_outline(document_id: str) -> bool:
            worker = self.outline_worker_resolver()
            if worker is None:
                return False
            try:
                return bool(worker.is_rebuild_pending(document_id))
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.debug("Outline worker pending check failed", exc_info=True)
                return False

        try:
            tool = DocumentOutlineTool(
                memory_resolver=self.outline_memory_resolver,
                document_lookup=self.document_lookup,
                active_document_provider=self.active_document_provider,
                budget_policy=None,
                pending_outline_checker=_pending_outline,
            )
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Unable to initialize DocumentOutlineTool", exc_info=True)
            return None

        self._outline_tool = tool
        return tool

    def ensure_find_sections_tool(self) -> DocumentFindSectionsTool | None:
        if not self.phase3_outline_enabled:
            return None
        if self._find_sections_tool is not None:
            return self._find_sections_tool

        try:
            tool = DocumentFindSectionsTool(
                embedding_index_resolver=self.embedding_index_resolver,
                document_lookup=self.document_lookup,
                active_document_provider=self.active_document_provider,
                outline_memory=self.outline_memory_resolver,
            )
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Unable to initialize DocumentFindSectionsTool", exc_info=True)
            return None

        self._find_sections_tool = tool
        return tool

    def ensure_plot_state_tool(self) -> DocumentPlotStateTool | None:
        if not self.plot_scaffolding_enabled:
            return None
        if self._plot_state_tool is not None:
            return self._plot_state_tool

        try:
            tool = DocumentPlotStateTool(
                plot_state_resolver=self.plot_state_store_resolver,
                active_document_provider=self.active_document_provider,
                feature_enabled=lambda: self.plot_scaffolding_enabled,
            )
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Unable to initialize DocumentPlotStateTool", exc_info=True)
            return None

        self._plot_state_tool = tool
        return tool


__all__ = ["ToolProvider"]
