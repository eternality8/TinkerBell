"""Tracks editor state updates and snapshot persistence for ``MainWindow``."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..chat.message_model import ToolTrace
from ..editor.document_model import DocumentState, SelectionRange
from ..editor.workspace import DocumentTab
from ..services.unsaved_cache import UnsavedCache
from ..utils import file_io
from ..ai.memory.cache_bus import DocumentCacheBus, DocumentChangedEvent, get_document_cache_bus


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DocumentStateMonitor:
    """Encapsulates document listeners, autosave, and snapshot persistence."""

    editor: Any
    workspace: Any
    chat_panel: Any
    status_bar: Any
    unsaved_cache_provider: Callable[[], UnsavedCache | None]
    unsaved_cache_persister: Callable[[UnsavedCache | None], None]
    refresh_window_title: Callable[[DocumentState | None], None]
    sync_workspace_state: Callable[[bool], None]
    current_path_getter: Callable[[], Path | None]
    current_path_setter: Callable[[Path | None], None]
    last_snapshot_setter: Callable[[dict[str, Any]], None]
    active_document_provider: Callable[[], DocumentState | None]
    maybe_clear_diff_overlay: Callable[[DocumentState], None]
    window_app_name: str
    untitled_document_name: str
    untitled_snapshot_key: str = "__untitled__"
    cache_bus_resolver: Callable[[], DocumentCacheBus | None] = get_document_cache_bus

    _outline_digest_cache: dict[str, str] = field(default_factory=dict)
    _unsaved_snapshot_digests: dict[str, str] = field(default_factory=dict)
    _published_versions: dict[str, int] = field(default_factory=dict)
    _snapshot_persistence_block: int = 0
    _last_autosave_at: datetime | None = None

    # ------------------------------------------------------------------
    # Editor/workspace listener entry points
    # ------------------------------------------------------------------
    def handle_editor_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Cache outline digest changes surfaced by the editor."""

        self.last_snapshot_setter(snapshot)
        digest = str(snapshot.get("outline_digest") or "").strip()
        document_id = str(snapshot.get("document_id") or "").strip()
        if not document_id:
            document = self.active_document_provider()
            if document is not None:
                document_id = document.document_id
        if not digest or not document_id:
            return
        if self._outline_digest_cache.get(document_id) == digest:
            return
        self._outline_digest_cache[document_id] = digest
        self._emit_outline_timeline_event(document_id, digest)

    def handle_editor_text_changed(self, text: str, state: DocumentState) -> None:
        """Respond to editor text or metadata edits."""

        del text  # text not needed; state already contains current snapshot
        self.refresh_window_title(state)
        self._refresh_chat_suggestions(state=state)
        if self._snapshot_persistence_block > 0:
            self.update_autosave_indicator(document=state)
            self.maybe_clear_diff_overlay(state)
            self._publish_document_change(state)
            return
        self._persist_unsaved_snapshot(state)
        self.update_autosave_indicator(document=state)
        self.maybe_clear_diff_overlay(state)
        self._publish_document_change(state)

    def handle_editor_selection_changed(self, selection: SelectionRange) -> None:
        self._refresh_chat_suggestions(selection=selection)

    def handle_active_tab_changed(self, tab: DocumentTab | None) -> None:
        if tab is None:
            self.current_path_setter(None)
            self.sync_workspace_state(True)
            return
        try:
            document = tab.document()
        except Exception:  # pragma: no cover - defensive guard
            self.current_path_setter(None)
            self.sync_workspace_state(True)
            return
        self.current_path_setter(document.metadata.path)
        self.refresh_window_title(document)
        self._refresh_chat_suggestions(state=document)
        self.update_autosave_indicator(document=document)
        self.sync_workspace_state(True)

    # ------------------------------------------------------------------
    # Public helpers consumed by MainWindow and tests
    # ------------------------------------------------------------------
    def update_autosave_indicator(
        self,
        *,
        autosaved: bool = False,
        document: DocumentState | None = None,
    ) -> None:
        doc = document or self.editor.to_document()
        if autosaved:
            self._last_autosave_at = datetime.now(timezone.utc)
        status, detail = self._format_autosave_label(doc)
        try:
            self.status_bar.set_autosave_state(status, detail=detail)
        except Exception:  # pragma: no cover - defensive guard for tests
            pass

    def refresh_chat_suggestions(
        self,
        *,
        state: DocumentState | None = None,
        selection: SelectionRange | None = None,
    ) -> None:
        self._refresh_chat_suggestions(state=state, selection=selection)

    def clear_unsaved_snapshot(
        self,
        *,
        cache: UnsavedCache | None = None,
        path: Path | str | None = None,
        tab_id: str | None = None,
        persist: bool = True,
    ) -> None:
        key = self.snapshot_key(path, tab_id=tab_id)
        target_cache = cache or self.unsaved_cache_provider()
        changed = False

        if target_cache is not None:
            if path is None and tab_id is not None:
                snapshots = dict(target_cache.untitled_snapshots or {})
                if snapshots.pop(tab_id, None) is not None:
                    target_cache.untitled_snapshots = snapshots
                    changed = True
            elif path is None and tab_id is None:
                if target_cache.unsaved_snapshot is not None:
                    target_cache.unsaved_snapshot = None
                    changed = True
            else:
                snapshots = dict(target_cache.unsaved_snapshots or {})
                if snapshots.pop(key, None) is not None:
                    target_cache.unsaved_snapshots = snapshots
                    changed = True

        self._unsaved_snapshot_digests.pop(key, None)
        if changed and persist and target_cache is not None:
            self.unsaved_cache_persister(target_cache)

    def suspend_snapshot_persistence(self) -> Any:
        @contextmanager
        def _manager() -> Any:
            self._snapshot_persistence_block += 1
            try:
                yield
            finally:
                self._snapshot_persistence_block = max(0, self._snapshot_persistence_block - 1)

        return _manager()

    def snapshot_key(self, path: Path | str | None, *, tab_id: str | None = None) -> str:
        if path is None:
            if tab_id:
                return f"{self.untitled_snapshot_key}:{tab_id}"
            return self.untitled_snapshot_key
        return str(Path(path).expanduser().resolve())

    def record_snapshot_digest(self, path: Path | None, tab_id: str | None, text: str) -> None:
        key = self.snapshot_key(path, tab_id=tab_id)
        self._unsaved_snapshot_digests[key] = file_io.compute_text_digest(text)

    def document_display_name(self, document: DocumentState) -> str:
        path = document.metadata.path or self.current_path_getter()
        if path is not None:
            candidate = Path(path)
            if candidate.name:
                return candidate.name
            return self.window_app_name
        return self.untitled_document_name

    def reset_outline_digest_cache(self) -> None:
        self._outline_digest_cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _refresh_chat_suggestions(
        self,
        *,
        state: DocumentState | None = None,
        selection: SelectionRange | None = None,
    ) -> None:
        document = state or self.editor.to_document()
        active_selection = selection or document.selection
        start, end = active_selection.as_tuple()
        text = document.text[start:end]
        summary = self._summarize_selection_text(text)
        self.chat_panel.set_selection_summary(summary)
        suggestions = self._build_chat_suggestions(document, text)
        self.chat_panel.set_suggestions(suggestions)

    def _build_chat_suggestions(self, document: DocumentState, selection_text: str) -> list[str]:
        has_selection = bool(selection_text.strip())
        has_document_text = bool(document.text.strip())
        if has_selection:
            suggestions = [
                "Summarize the selected text.",
                "Rewrite the selected text for clarity.",
                "Extract action items from the selection.",
            ]
        elif has_document_text:
            suggestions = [
                "Summarize the current document.",
                "Suggest improvements to the document structure.",
                "Highlight inconsistencies or missing sections.",
            ]
        else:
            name = self.document_display_name(document)
            suggestions = [
                f"Draft an outline for {name}.",
                "Propose a starter paragraph for this document.",
                "List the key points this document should cover.",
            ]
        suggestions.append("Help me plan the next edits.")
        return suggestions

    def _summarize_selection_text(self, selection_text: str) -> Optional[str]:
        condensed = self._condense_whitespace(selection_text)
        if not condensed:
            return None
        if len(condensed) > 80:
            condensed = f"{condensed[:77].rstrip()}…"
        return condensed

    def _emit_outline_timeline_event(self, document_id: str, outline_digest: str) -> None:
        try:
            document = self.workspace.find_document_by_id(document_id)
        except Exception:  # pragma: no cover - defensive guard
            document = None
        label = self.document_display_name(document) if document is not None else document_id
        trace = ToolTrace(
            name="document_outline",
            input_summary=label,
            output_summary="Outline updated",
            metadata={
                "document_id": document_id,
                "outline_digest": outline_digest,
                "timestamp": datetime.now(timezone.utc).timestamp(),
            },
        )
        try:
            self.chat_panel.show_tool_trace(trace)
        except Exception:  # pragma: no cover - UI optional in tests
            pass

    def _publish_document_change(self, state: DocumentState | None) -> None:
        if state is None:
            return
        bus = self._resolve_cache_bus()
        if bus is None:
            return
        try:
            version = state.version_info()
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Unable to resolve document version for cache event", exc_info=True)
            return
        last_version = self._published_versions.get(version.document_id)
        if last_version == version.version_id:
            return
        self._published_versions[version.document_id] = version.version_id
        event = DocumentChangedEvent(
            document_id=version.document_id,
            version_id=version.version_id,
            content_hash=version.content_hash,
            edited_ranges=((0, len(state.text or "")),),
            source="document-monitor",
        )
        try:
            bus.publish(event)
        except Exception:  # pragma: no cover - defensive guard
            LOGGER.debug("Document cache publish failed", exc_info=True)

    def _resolve_cache_bus(self) -> DocumentCacheBus | None:
        resolver = self.cache_bus_resolver
        if callable(resolver):
            try:
                return resolver()
            except Exception:  # pragma: no cover - defensive guard
                LOGGER.debug("Cache bus resolver failed", exc_info=True)
                return None
        return None

    def _persist_unsaved_snapshot(self, state: DocumentState | None = None) -> None:
        cache = self.unsaved_cache_provider()
        if cache is None:
            return

        document = state or self.editor.to_document()
        path = document.metadata.path or self.current_path_getter()
        active_tab = getattr(self.workspace, "active_tab", None)
        tab_id = getattr(active_tab, "id", None)
        key = self.snapshot_key(path, tab_id=tab_id)

        if not document.dirty:
            self.clear_unsaved_snapshot(cache=cache, path=path, tab_id=tab_id)
            return

        snapshot = {
            "text": document.text,
            "language": document.metadata.language,
            "selection": list(document.selection.as_tuple()),
        }
        digest = file_io.compute_text_digest(snapshot["text"])
        if self._unsaved_snapshot_digests.get(key) == digest:
            existing = self._get_snapshot_entry(cache, path=path, tab_id=tab_id)
            if existing == snapshot:
                return

        if path is None:
            if tab_id is not None:
                snapshots = dict(cache.untitled_snapshots or {})
                snapshots[tab_id] = snapshot
                cache.untitled_snapshots = snapshots
            else:
                cache.unsaved_snapshot = snapshot
        else:
            snapshots = dict(cache.unsaved_snapshots or {})
            snapshots[key] = snapshot
            cache.unsaved_snapshots = snapshots

        self._unsaved_snapshot_digests[key] = digest
        self.sync_workspace_state(False)
        self.unsaved_cache_persister(cache)
        self.update_autosave_indicator(autosaved=True, document=document)

    def _get_snapshot_entry(
        self,
        cache: UnsavedCache,
        *,
        path: Path | str | None,
        tab_id: str | None,
    ) -> dict[str, Any] | None:
        if path is None:
            if tab_id is not None:
                return (cache.untitled_snapshots or {}).get(tab_id)
            return cache.unsaved_snapshot
        key = self.snapshot_key(path)
        return (cache.unsaved_snapshots or {}).get(key)

    def _format_autosave_label(self, document: DocumentState) -> tuple[str, str]:
        name = self.document_display_name(document)
        if not document.dirty:
            return ("Saved", name)
        if self._last_autosave_at is None:
            return ("Unsaved changes", name)
        elapsed = datetime.now(timezone.utc) - self._last_autosave_at
        return (f"Autosaved {self._format_elapsed(elapsed)}", name)

    @staticmethod
    def _format_elapsed(delta: timedelta) -> str:
        seconds = max(0, int(delta.total_seconds()))
        if seconds < 5:
            return "just now"
        if seconds < 60:
            return f"{seconds}s ago"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours}h ago"
        days = hours // 24
        return f"{days}d ago"

    @staticmethod
    def _condense_whitespace(text: str) -> str:
        return " ".join(text.split())


__all__ = ["DocumentStateMonitor"]
