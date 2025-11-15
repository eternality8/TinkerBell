"""Workspace models managing multiple document tabs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Protocol

from .document_model import DocumentMetadata, DocumentState
from .editor_widget import EditorWidget
from ..services.bridge import DocumentBridge

__all__ = ["DocumentTab", "DocumentWorkspace", "ActiveTabListener"]


class ActiveTabListener(Protocol):
    """Callback signature fired whenever the active tab changes."""

    def __call__(self, tab: Optional["DocumentTab"]) -> None:  # pragma: no cover - protocol
        ...


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _generate_tab_id() -> str:
    return uuid.uuid4().hex


def _normalize_path(path: Path | str | None) -> Path | None:
    if path is None:
        return None
    if isinstance(path, Path):
        return path.expanduser().resolve()
    return Path(path).expanduser().resolve()


@dataclass(slots=True)
class DocumentTab:
    """Container describing an open editor tab and related runtime metadata."""

    id: str
    editor: EditorWidget
    bridge: DocumentBridge
    created_at: datetime = field(default_factory=_utcnow)
    title: str = "Untitled"
    untitled_index: int | None = None
    last_snapshot_digest: str | None = None

    def document(self) -> DocumentState:
        """Return the current :class:`DocumentState` for this tab."""

        return self.editor.to_document()

    @property
    def path(self) -> Path | None:
        return self.document().metadata.path

    @property
    def dirty(self) -> bool:
        return self.document().dirty

    def update_title(self, fallback: str = "Untitled") -> None:
        """Refresh the human-friendly title used in the tab strip."""

        document = self.document()
        path = document.metadata.path
        if path is not None:
            candidate = path.name or str(path)
        else:
            suffix = f" {self.untitled_index}" if self.untitled_index else ""
            candidate = f"{fallback}{suffix}".strip()
        prefix = "*" if document.dirty else ""
        self.title = f"{prefix}{candidate}" if prefix else candidate


class DocumentWorkspace:
    """Manages multiple document tabs and their active selection."""

    def __init__(
        self,
        *,
        editor_factory: Callable[[], EditorWidget] | None = None,
        bridge_factory: Callable[[EditorWidget], DocumentBridge] | None = None,
    ) -> None:
        self._editor_factory = editor_factory or EditorWidget
        self._bridge_factory = bridge_factory or (lambda editor: DocumentBridge(editor=editor))
        self._tabs: Dict[str, DocumentTab] = {}
        self._order: List[str] = []
        self._active_tab_id: str | None = None
        self._listeners: List[ActiveTabListener] = []
        self._untitled_counter = 1

    # ------------------------------------------------------------------
    # Tab lifecycle helpers
    # ------------------------------------------------------------------
    def create_tab(
        self,
        *,
        document: DocumentState | None = None,
        path: Path | str | None = None,
        title: str | None = None,
        make_active: bool = True,
        tab_id: str | None = None,
        untitled_index: int | None = None,
    ) -> DocumentTab:
        """Create a new tab wrapping its own editor + bridge."""

        editor = self._editor_factory()
        resolved_path = _normalize_path(path)
        doc = document or DocumentState(metadata=DocumentMetadata(path=resolved_path))
        if resolved_path is not None:
            doc.metadata.path = resolved_path
        editor.load_document(doc)
        bridge = self._bridge_factory(editor)
        tab_id = tab_id or _generate_tab_id()
        untitled_idx: int | None = None
        if doc.metadata.path is None:
            if untitled_index is not None:
                untitled_idx = untitled_index
                self._observe_untitled_index(untitled_idx)
            else:
                untitled_idx = self._reserve_untitled_index()
        tab = DocumentTab(
            id=tab_id,
            editor=editor,
            bridge=bridge,
            untitled_index=untitled_idx,
        )
        tab.update_title(title or "Untitled")
        self._tabs[tab_id] = tab
        self._order.append(tab_id)
        if make_active or self._active_tab_id is None:
            self.set_active_tab(tab_id)
        return tab

    def close_tab(self, tab_id: str) -> DocumentTab:
        """Close and return the specified tab."""

        if tab_id not in self._tabs:
            raise KeyError(f"Unknown tab_id: {tab_id}")
        tab = self._tabs.pop(tab_id)
        try:
            index = self._order.index(tab_id)
        except ValueError:  # pragma: no cover - defensive, should not happen
            index = -1
        else:
            self._order.pop(index)

        if self._active_tab_id == tab_id:
            if self._order:
                fallback_index = index if 0 <= index < len(self._order) else len(self._order) - 1
                self._active_tab_id = self._order[fallback_index]
            else:
                self._active_tab_id = None
            self._notify_active_listeners()
        return tab

    def ensure_tab(self) -> DocumentTab:
        """Ensure at least one tab exists, returning it."""

        if self._order:
            return self._tabs[self._order[-1]]
        return self.create_tab()

    def set_active_tab(self, tab_id: str) -> DocumentTab:
        """Mark the provided tab as active and notify listeners."""

        if tab_id not in self._tabs:
            raise KeyError(f"Unknown tab_id: {tab_id}")
        if self._active_tab_id == tab_id:
            return self._tabs[tab_id]
        self._active_tab_id = tab_id
        self._notify_active_listeners()
        return self._tabs[tab_id]

    def add_active_listener(self, listener: ActiveTabListener) -> None:
        self._listeners.append(listener)

    def remove_active_listener(self, listener: ActiveTabListener) -> None:
        try:
            self._listeners.remove(listener)
        except ValueError:  # pragma: no cover - defensive
            pass

    def _notify_active_listeners(self) -> None:
        tab = self.active_tab
        for listener in list(self._listeners):
            listener(tab)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    @property
    def active_tab_id(self) -> str | None:
        return self._active_tab_id

    @property
    def active_tab(self) -> DocumentTab | None:
        if self._active_tab_id is None:
            return None
        return self._tabs.get(self._active_tab_id)

    def require_active_tab(self) -> DocumentTab:
        tab = self.active_tab
        if tab is None:
            raise RuntimeError("No active tab available")
        return tab

    def iter_tabs(self) -> Iterator[DocumentTab]:
        for tab_id in self._order:
            yield self._tabs[tab_id]

    def tab_ids(self) -> Iterable[str]:
        return tuple(self._order)

    def tab_count(self) -> int:
        return len(self._order)

    def get_tab(self, tab_id: str) -> DocumentTab:
        tab = self._tabs.get(tab_id)
        if tab is not None:
            return tab

        path_match = self.find_tab_by_path(tab_id)
        if path_match is not None:
            return path_match

        raise KeyError(f"Unknown tab_id: {tab_id}")

    def find_tab_by_path(self, path: Path | str) -> DocumentTab | None:
        normalized = _normalize_path(path)
        if normalized is None:
            return None
        for tab in self.iter_tabs():
            tab_path = tab.path
            if tab_path and _normalize_path(tab_path) == normalized:
                return tab
        return None

    def active_document(self) -> DocumentState:
        return self.require_active_tab().document()

    def active_editor(self) -> EditorWidget:
        return self.require_active_tab().editor

    def active_bridge(self) -> DocumentBridge:
        return self.require_active_tab().bridge

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def serialize_tabs(self) -> list[dict[str, object]]:
        """Return a simplified representation of open tabs for persistence."""

        payload: list[dict[str, object]] = []
        for tab in self.iter_tabs():
            document = tab.document()
            entry: dict[str, object] = {
                "tab_id": tab.id,
                "title": tab.title,
                "dirty": document.dirty,
                "language": document.metadata.language,
                "created_at": tab.created_at.isoformat(),
            }
            if document.metadata.path is not None:
                entry["path"] = str(document.metadata.path)
            if tab.untitled_index is not None:
                entry["untitled_index"] = tab.untitled_index
            payload.append(entry)
        return payload

    def serialize_state(self) -> dict[str, Any]:
        """Return a structured workspace snapshot for persistence layers."""

        return {
            "open_tabs": self.serialize_tabs(),
            "active_tab_id": self.active_tab_id,
            "untitled_counter": self._untitled_counter,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reserve_untitled_index(self) -> int:
        value = self._untitled_counter
        self._untitled_counter += 1
        return value

    def _observe_untitled_index(self, value: int | None) -> None:
        if value is None:
            return
        if value >= self._untitled_counter:
            self._untitled_counter = value + 1

    def set_next_untitled_index(self, value: int) -> None:
        """Force the next untitled counter to ``value`` when restoring state."""

        if value <= 0:
            value = 1
        self._untitled_counter = max(self._untitled_counter, value)
