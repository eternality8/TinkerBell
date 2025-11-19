"""Bridge router that proxies DocumentBridge operations to the active tab."""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Mapping, Sequence, Set

from ..chat.message_model import EditDirective
from ..editor.workspace import DocumentTab, DocumentWorkspace
from .bridge import DocumentBridge, EditAppliedListener


class WorkspaceBridgeRouter:
    """Routes bridge operations to the active tab while wiring listeners to all tabs."""

    def __init__(self, workspace: DocumentWorkspace) -> None:
        self._workspace = workspace
        self._wired_bridge_ids: Set[int] = set()
        self._edit_listeners: List[EditAppliedListener] = []
        self._failure_listeners: List[Callable[[EditDirective, str], None]] = []
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
        for bridge in self._iter_bridges():
            bridge.add_failure_listener(listener)

    def set_main_thread_executor(self, executor) -> None:
        self._main_thread_executor = executor
        for bridge in self._iter_bridges():
            bridge.set_main_thread_executor(executor)

    # ------------------------------------------------------------------
    # Tab lifecycle
    # ------------------------------------------------------------------
    def track_tab(self, tab: DocumentTab) -> None:
        self._wire_bridge(tab.bridge)

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

    @property
    def editor(self):  # pragma: no cover - simple forwarding
        return getattr(self._bridge_for_tab(None), "editor", None)

    def list_tabs(self) -> list[dict[str, object]]:
        return self._workspace.serialize_tabs()

    def active_tab_id(self) -> str | None:
        return self._workspace.active_tab_id

    def __getattr__(self, name: str):  # pragma: no cover - fallback forwarding
        return getattr(self._bridge_for_tab(None), name)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _wire_existing_bridges(self) -> None:
        for tab in self._workspace.iter_tabs():
            self._wire_bridge(tab.bridge)

    def _wire_bridge(self, bridge: DocumentBridge) -> None:
        ident = id(bridge)
        if ident in self._wired_bridge_ids:
            return
        self._wired_bridge_ids.add(ident)
        for listener in self._edit_listeners:
            bridge.add_edit_listener(listener)
        for listener in self._failure_listeners:
            bridge.add_failure_listener(listener)
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