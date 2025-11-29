"""Bridge router that proxies DocumentBridge operations to the active tab."""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Iterable, List, Mapping, Sequence, Set

from ..chat.message_model import EditDirective
from ..editor.workspace import DocumentTab, DocumentWorkspace
from .bridge import DocumentBridge, EditAppliedListener


_LOGGER = logging.getLogger(__name__)


class WorkspaceBridgeRouter:
    """Routes bridge operations to the active tab while wiring listeners to all tabs."""

    def __init__(self, workspace: DocumentWorkspace) -> None:
        self._workspace = workspace
        self._wired_bridge_ids: Set[int] = set()
        self._edit_listeners: List[EditAppliedListener] = []
        self._failure_listeners: List[Callable[[EditDirective, str], None]] = []
        self._failure_listener_capabilities: dict[Callable[..., Any], bool] = {}
        self._bridge_failure_handlers: dict[int, Callable[..., Any]] = {}
        self._bridge_tab_index: dict[int, str | None] = {}
        self._main_thread_executor = None
        self._wire_existing_bridges()

    # ------------------------------------------------------------------
    # Listener management
    # ------------------------------------------------------------------
    def add_edit_listener(self, listener: EditAppliedListener) -> None:
        self._edit_listeners.append(listener)
        for bridge in self._iter_bridges():
            bridge.add_edit_listener(listener)

    def add_failure_listener(self, listener: Callable[[EditDirective, str], None]) -> None:
        self._failure_listeners.append(listener)
        self._failure_listener_capabilities[listener] = self._supports_failure_metadata(listener)

    def set_main_thread_executor(self, executor) -> None:
        self._main_thread_executor = executor
        for bridge in self._iter_bridges():
            bridge.set_main_thread_executor(executor)

    # ------------------------------------------------------------------
    # Tab lifecycle
    # ------------------------------------------------------------------
    def track_tab(self, tab: DocumentTab) -> None:
        self._wire_bridge(tab.bridge, tab_id=tab.id)

    # ------------------------------------------------------------------
    # Proxy helpers
    # ------------------------------------------------------------------
    def generate_snapshot(
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_open_documents: bool = False,
        window: Mapping[str, Any] | str | None = None,
        chunk_profile: str | None = None,
        max_tokens: int | None = None,
        include_text: bool = True,
    ) -> dict:
        tab = self._resolve_tab(tab_id)
        snapshot = tab.bridge.generate_snapshot(
            delta_only=delta_only,
            window=window,
            max_tokens=max_tokens,
            chunk_profile=chunk_profile,
            include_text=include_text,
        )
        return self._augment_snapshot(tab, snapshot, include_open_documents=include_open_documents)

    def generate_snapshots(
        self,
        tab_ids: Sequence[str],
        *,
        delta_only: bool = False,
        include_open_documents: bool = False,
        window: Mapping[str, Any] | str | None = None,
        chunk_profile: str | None = None,
        max_tokens: int | None = None,
        include_text: bool = True,
    ) -> list[dict[str, Any]]:
        return [
            self.generate_snapshot(
                delta_only=delta_only,
                tab_id=tab_id,
                include_open_documents=include_open_documents,
                window=window,
                chunk_profile=chunk_profile,
                max_tokens=max_tokens,
                include_text=include_text,
            )
            for tab_id in tab_ids
        ]

    def queue_edit(self, directive, *, tab_id: str | None = None) -> None:
        self._bridge_for_tab(tab_id).queue_edit(directive)

    @property
    def last_diff_summary(self):
        return self.get_last_diff_summary()

    def get_last_diff_summary(self, tab_id: str | None = None):
        return getattr(self._bridge_for_tab(tab_id), "last_diff_summary", None)

    @property
    def last_snapshot_version(self):
        return self.get_last_snapshot_version()

    def get_last_snapshot_version(self, tab_id: str | None = None):
        return getattr(self._bridge_for_tab(tab_id), "last_snapshot_version", None)

    @property
    def last_edit_context(self):
        return getattr(self._bridge_for_tab(None), "last_edit_context", None)

    def get_last_edit_context(self, tab_id: str | None = None):
        return getattr(self._bridge_for_tab(tab_id), "last_edit_context", None)

    @property
    def patch_metrics(self):
        return getattr(self._bridge_for_tab(None), "patch_metrics", None)

    def get_patch_metrics(self, tab_id: str | None = None):
        return getattr(self._bridge_for_tab(tab_id), "patch_metrics", None)

    def get_last_failure_metadata(self, tab_id: str | None = None):
        bridge = self._bridge_for_tab(tab_id)
        metadata = getattr(bridge, "last_failure_metadata", None)
        if metadata is None:
            return None
        payload = dict(metadata)
        if payload.get("tab_id") is None:
            if tab_id is not None:
                payload["tab_id"] = tab_id
            else:
                lookup = self._bridge_tab_index.get(id(bridge)) or self.active_tab_id()
                if lookup:
                    payload["tab_id"] = lookup
        return payload

    @property
    def editor(self):  # pragma: no cover - simple forwarding
        return getattr(self._bridge_for_tab(None), "editor", None)

    def list_tabs(self) -> list[dict[str, object]]:
        return self._workspace.serialize_tabs()

    def active_tab_id(self) -> str | None:
        return self._workspace.active_tab_id

    def get_active_tab_id(self) -> str | None:
        """Alias for active_tab_id() for ToolContextProvider compatibility."""
        return self._workspace.active_tab_id

    def get_tab_content(self, tab_id: str) -> str | None:
        """Get the content of a specific tab by ID.
        
        Required by TabListingProvider protocol for ListTabsTool.
        
        Args:
            tab_id: The tab identifier.
            
        Returns:
            Document text content, or None if tab not found.
        """
        try:
            tab = self._workspace.get_tab(tab_id)
            return tab.document().text
        except KeyError:
            return None

    def get_document_content(self, tab_id: str) -> str | None:
        """Get document content for a tab.
        
        Required by ToolContextProvider protocol for ToolDispatcher.
        
        Args:
            tab_id: The tab identifier.
            
        Returns:
            Document text content, or None if tab not found.
        """
        return self.get_tab_content(tab_id)

    def set_document_content(self, tab_id: str, content: str) -> None:
        """Set document content for a tab.
        
        Required by ToolContextProvider protocol for ToolDispatcher.
        
        Args:
            tab_id: The tab identifier.
            content: The new document content.
            
        Raises:
            KeyError: If the tab is not found.
        """
        tab = self._workspace.get_tab(tab_id)
        bridge = tab.bridge
        from ..editor.document_model import DocumentState
        current_doc = bridge.editor.to_document()
        new_doc = DocumentState(
            document_id=current_doc.document_id,
            text=content,
        )
        bridge.editor.load_document(new_doc)

    def get_version_token(self, tab_id: str) -> str | None:
        """Get current version token for a tab.
        
        Required by ToolContextProvider protocol for ToolDispatcher.
        
        Args:
            tab_id: The tab identifier.
            
        Returns:
            Version token string, or None if not available.
        """
        from ..ai.tools.version import get_version_manager
        try:
            token = get_version_manager().get_current_token(tab_id)
            return token.to_string() if token else None
        except Exception:
            return None

    def __getattr__(self, name: str):  # pragma: no cover - fallback forwarding
        return getattr(self._bridge_for_tab(None), name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _wire_existing_bridges(self) -> None:
        for tab in self._workspace.iter_tabs():
            self._wire_bridge(tab.bridge, tab_id=tab.id)

    def _wire_bridge(self, bridge: DocumentBridge, *, tab_id: str | None = None) -> None:
        ident = id(bridge)
        if tab_id is not None:
            self._bridge_tab_index[ident] = tab_id
        elif ident not in self._bridge_tab_index:
            self._bridge_tab_index[ident] = None
        if ident in self._wired_bridge_ids:
            return
        self._wired_bridge_ids.add(ident)
        for listener in self._edit_listeners:
            bridge.add_edit_listener(listener)
        handler = self._make_bridge_failure_handler(bridge)
        self._bridge_failure_handlers[ident] = handler
        bridge.add_failure_listener(handler)
        if self._main_thread_executor is not None:
            bridge.set_main_thread_executor(self._main_thread_executor)

    def _iter_bridges(self) -> Iterable[DocumentBridge]:
        for tab in self._workspace.iter_tabs():
            yield tab.bridge

    def _bridge_for_tab(self, tab_id: str | None) -> DocumentBridge:
        if tab_id is None:
            return self._workspace.active_bridge()
        return self._workspace.get_tab(tab_id).bridge

    def _resolve_tab(self, tab_id: str | None) -> DocumentTab:
        if tab_id is None:
            return self._workspace.require_active_tab()
        return self._workspace.get_tab(tab_id)

    def _augment_snapshot(
        self,
        tab: DocumentTab,
        snapshot: Mapping[str, Any],
        *,
        include_open_documents: bool,
    ) -> dict[str, Any]:
        payload = dict(snapshot)
        payload.setdefault("tab_id", tab.id)
        payload.setdefault("tab_title", tab.title)
        if include_open_documents:
            payload["open_tabs"] = self._workspace.serialize_tabs()
            payload["active_tab_id"] = self._workspace.active_tab_id
        return payload

    def _make_bridge_failure_handler(self, bridge: DocumentBridge) -> Callable[[EditDirective, str, Mapping[str, Any] | None], None]:
        def _handler(directive: EditDirective, message: str, metadata: Mapping[str, Any] | None = None) -> None:
            self._dispatch_failure_event(bridge, directive, message, metadata)

        return _handler

    def _dispatch_failure_event(
        self,
        bridge: DocumentBridge,
        directive: EditDirective,
        message: str,
        metadata: Mapping[str, Any] | None,
    ) -> None:
        payload = self._merge_failure_metadata(bridge, metadata)
        for listener in list(self._failure_listeners):
            accepts_metadata = self._failure_listener_capabilities.get(listener)
            if accepts_metadata is None:
                accepts_metadata = self._supports_failure_metadata(listener)
                self._failure_listener_capabilities[listener] = accepts_metadata
            try:
                if accepts_metadata:
                    listener(directive, message, dict(payload) if payload is not None else None)
                else:
                    listener(directive, message)
            except Exception:  # pragma: no cover - defensive guard
                _LOGGER.debug("Bridge failure listener raised", exc_info=True)

    def _merge_failure_metadata(
        self,
        bridge: DocumentBridge,
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {}
        if metadata:
            payload.update(metadata)
        tab_id = payload.get("tab_id") or self._bridge_tab_index.get(id(bridge))
        if tab_id:
            payload["tab_id"] = tab_id
        if "document_id" not in payload:
            version = getattr(bridge, "last_document_version", None)
            if version is not None:
                payload.setdefault("document_id", version.document_id)
                payload.setdefault("version_id", version.version_id)
                payload.setdefault("content_hash", version.content_hash)
        return payload or None

    @staticmethod
    def _supports_failure_metadata(listener: Callable[..., Any]) -> bool:
        try:
            signature = inspect.signature(listener)
        except (TypeError, ValueError):  # pragma: no cover - assume flexible
            return True
        params = list(signature.parameters.values())
        if not params:
            return False
        if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in params):
            return True
        positional = [
            param
            for param in params
            if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional) >= 3