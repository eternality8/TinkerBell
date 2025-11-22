"""Persistence and file-dialog helpers extracted from :mod:`main_window`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..editor.document_model import DocumentMetadata, DocumentState, SelectionRange
from ..editor.workspace import DocumentTab, DocumentWorkspace
from ..services.settings import Settings
from ..services.unsaved_cache import UnsavedCache
from ..utils import file_io
from .ai_review_controller import AIReviewController
from .review_overlay_manager import ReviewOverlayManager
from .models.window_state import WindowContext


class DocumentSessionService:
    """Owns file prompts, workspace restore, and settings persistence."""

    def __init__(
        self,
        *,
        context: WindowContext,
        editor: Any,
        workspace: DocumentWorkspace,
        document_monitor_resolver: Callable[[], Any | None],
        open_document: Callable[[Path], None],
        status_updater: Callable[[str], None],
        qt_parent_provider: Callable[[], Any | None],
        untitled_document_name: str,
        untitled_snapshot_key: str,
    ) -> None:
        self._context = context
        self._editor = editor
        self._workspace = workspace
        self._document_monitor_resolver = document_monitor_resolver
        self._open_document = open_document
        self._status_updater = status_updater
        self._qt_parent_provider = qt_parent_provider
        self._untitled_document_name = untitled_document_name
        self._untitled_snapshot_key = untitled_snapshot_key
        self._current_path: Path | None = None
        self._review_controller: AIReviewController | None = None
        self._review_overlay_manager: ReviewOverlayManager | None = None
        self._import_dialog_filter_provider: Callable[[], str | None] = lambda: None
        self._restoring_workspace = False

    # ------------------------------------------------------------------
    # Dependency setters
    # ------------------------------------------------------------------
    def set_review_controller(self, controller: AIReviewController) -> None:
        self._review_controller = controller

    def set_review_overlay_manager(self, manager: ReviewOverlayManager | None) -> None:
        self._review_overlay_manager = manager

    def set_import_dialog_filter_provider(self, provider: Callable[[], str | None]) -> None:
        self._import_dialog_filter_provider = provider

    # ------------------------------------------------------------------
    # Current-path helpers exposed to DocumentStateMonitor
    # ------------------------------------------------------------------
    def get_current_path(self) -> Path | None:
        return self._current_path

    def set_current_path(self, path: Path | None) -> None:
        self._current_path = Path(path) if path is not None else None

    # ------------------------------------------------------------------
    # File dialog helpers
    # ------------------------------------------------------------------
    def prompt_for_open_path(self) -> Path | None:
        start_dir = self._resolve_open_start_dir(self._context.settings)
        try:
            from tinkerbell.widgets.dialogs import open_file_dialog
        except Exception as exc:  # pragma: no cover - optional PySide6 dependency
            raise RuntimeError("File dialogs require the optional PySide6 dependency.") from exc

        parent = self._qt_parent_provider()
        token_budget = self._resolve_token_budget()
        return open_file_dialog(parent=parent, start_dir=start_dir, token_budget=token_budget)

    def prompt_for_import_path(self) -> Path | None:
        start_dir = self._resolve_open_start_dir(self._context.settings)
        try:
            from tinkerbell.widgets.dialogs import open_file_dialog
        except Exception as exc:  # pragma: no cover - optional PySide6 dependency
            raise RuntimeError("File dialogs require the optional PySide6 dependency.") from exc

        parent = self._qt_parent_provider()
        file_filter = self._import_dialog_filter_provider()
        token_budget = self._resolve_token_budget()
        return open_file_dialog(
            parent=parent,
            caption="Import File",
            start_dir=start_dir,
            file_filter=file_filter,
            token_budget=token_budget,
            enable_samples=False,
        )

    def prompt_for_save_path(self, *, document: DocumentState | None = None) -> Path | None:
        start_dir = self._resolve_save_start_dir(self._context.settings)
        try:
            from tinkerbell.widgets.dialogs import save_file_dialog
        except Exception as exc:  # pragma: no cover - optional PySide6 dependency
            raise RuntimeError("File dialogs require the optional PySide6 dependency.") from exc

        parent = self._qt_parent_provider()
        token_budget = self._resolve_token_budget()
        document_text: str | None = None
        selection_text: str | None = None
        if document is not None:
            document_text = document.text
            selection = document.selection
            if selection.end > selection.start:
                text = document.text
                start = max(0, min(len(text), selection.start))
                end = max(start, min(len(text), selection.end))
                selection_text = text[start:end]
        return save_file_dialog(
            parent=parent,
            start_dir=start_dir,
            document_text=document_text,
            selection_text=selection_text,
            token_budget=token_budget,
        )

    # ------------------------------------------------------------------
    # Recent files & persistence helpers
    # ------------------------------------------------------------------
    def remember_recent_file(self, path: Path) -> None:
        settings = self._context.settings
        if settings is None:
            return

        normalized = str(path.expanduser().resolve())
        updated: list[str] = [normalized]
        for existing in settings.recent_files:
            candidate = str(Path(existing).expanduser().resolve())
            if candidate == normalized:
                continue
            updated.append(existing)
            if len(updated) >= 10:
                break
        settings.recent_files = updated
        settings.last_open_file = normalized
        self.persist_settings(settings)

    def restore_last_session_document(self) -> None:
        settings = self._context.settings
        if settings is None:
            return

        cache = self._unsaved_cache()
        if self._cleanup_orphan_snapshots(settings, cache=cache):
            self.persist_unsaved_cache(cache)

        if self._restore_workspace_tabs(settings):
            self.sync_workspace_state(persist=False)
            return

        next_index = getattr(settings, "next_untitled_index", None)
        if isinstance(next_index, int):
            self._workspace.set_next_untitled_index(next_index)

        if self._try_restore_last_file(settings):
            self.sync_workspace_state(persist=False)
            return

        if cache is not None:
            self._restore_unsaved_snapshot(cache)
        self.sync_workspace_state(persist=False)

    def persist_settings(self, settings: Settings | None) -> None:
        if settings is None:
            return
        store = self._context.settings_store
        if store is None:
            return
        try:
            store.save(settings)
        except Exception:
            pass

    def persist_unsaved_cache(self, cache: UnsavedCache | None) -> None:
        if cache is None:
            return
        store = getattr(self._context, "unsaved_cache_store", None)
        if store is None:
            return
        try:
            store.save(cache)
        except Exception:
            pass

    def sync_workspace_state(self, *, persist: bool = True) -> None:
        if self._restoring_workspace:
            return
        settings = self._context.settings
        if settings is None:
            return
        state = self._workspace.serialize_state()
        settings.open_tabs = state.get("open_tabs", [])
        settings.active_tab_id = state.get("active_tab_id")
        untitled_counter = state.get("untitled_counter")
        if isinstance(untitled_counter, int):
            settings.next_untitled_index = untitled_counter
        if persist:
            self.persist_settings(settings)

    def _unsaved_cache(self) -> UnsavedCache | None:
        return getattr(self._context, "unsaved_cache", None)

    # ------------------------------------------------------------------
    # Snapshot restore helpers surfaced to MainWindow
    # ------------------------------------------------------------------
    def apply_pending_snapshot_for_path(
        self,
        path: Path | str,
        *,
        tab_id: str | None = None,
        silent: bool = False,
    ) -> bool:
        cache = self._unsaved_cache()
        if cache is None:
            return False
        monitor = self._require_document_monitor()
        normalized_key = monitor.snapshot_key(path, tab_id=tab_id)
        snapshot = (cache.unsaved_snapshots or {}).get(normalized_key)
        if not isinstance(snapshot, dict) or "text" not in snapshot:
            return False
        resolved_path = Path(normalized_key)
        self._load_snapshot_document(snapshot, path=resolved_path, tab_id=tab_id)
        if not silent:
            self._status_updater(f"Restored unsaved changes for {resolved_path.name}")
        return True

    def apply_untitled_snapshot(self, tab_id: str) -> bool:
        cache = self._unsaved_cache()
        if cache is None:
            return False
        snapshot = None
        if cache.untitled_snapshots:
            snapshot = (cache.untitled_snapshots or {}).get(tab_id)
        if snapshot is None:
            snapshot = cache.unsaved_snapshot
        if not isinstance(snapshot, dict) or "text" not in snapshot:
            return False
        self._load_snapshot_document(snapshot, path=None, tab_id=tab_id)
        return True

    # ------------------------------------------------------------------
    # Internal helpers mirrored from the original MainWindow implementation
    # ------------------------------------------------------------------
    def _resolve_token_budget(self) -> int | None:
        settings = self._context.settings
        if settings is None:
            return None
        raw_budget = getattr(settings, "max_context_tokens", None)
        if isinstance(raw_budget, int):
            return raw_budget
        return None

    def _resolve_open_start_dir(self, settings: Settings | None) -> Path | None:
        if self._current_path is not None and self._current_path.parent.exists():
            return self._current_path.parent
        if settings:
            for entry in settings.recent_files:
                candidate = Path(entry).expanduser()
                if candidate.is_dir():
                    return candidate
                if candidate.exists():
                    return candidate.parent
        return Path.home()

    def _resolve_save_start_dir(self, settings: Settings | None) -> Path | None:
        return self._resolve_open_start_dir(settings)

    def _try_restore_last_file(self, settings: Settings) -> bool:
        last_path = (settings.last_open_file or "").strip()
        if not last_path:
            return False
        target = Path(last_path).expanduser()
        if not target.exists() or target.is_dir():
            self._handle_missing_last_file(settings)
            return False
        try:
            self._open_document(target)
            return True
        except FileNotFoundError:
            self._handle_missing_last_file(settings)
        except Exception:
            self._status_updater("Failed to restore last session file")
        return False

    def _restore_unsaved_snapshot(self, cache: UnsavedCache) -> bool:
        snapshot = cache.unsaved_snapshot
        if not isinstance(snapshot, dict) or "text" not in snapshot:
            return False
        monitor = self._require_document_monitor()
        tab = self._workspace.active_tab or self._workspace.ensure_tab()
        self._load_snapshot_document(snapshot, path=None, tab_id=tab.id)
        self._status_updater("Restored unsaved draft")
        monitor.update_autosave_indicator(document=self._editor.to_document())
        return True

    def _restore_workspace_tabs(self, settings: Settings) -> bool:
        entries = [entry for entry in (settings.open_tabs or []) if isinstance(entry, dict)]
        if not entries:
            return False
        review_controller = self._require_review_controller()
        self._restoring_workspace = True
        overlay_manager = self._review_overlay_manager
        if overlay_manager is not None:
            overlay_manager.reset_overlays()
        monitor = self._require_document_monitor()
        try:
            review_controller.abort_pending_review(
                reason="workspace-restore",
                status="AI edits discarded while restoring workspace",
                clear_overlays=True,
            )
            for tab_id in list(self._workspace.tab_ids()):
                review_controller.mark_pending_session_orphaned(tab_id, reason="workspace-restore")
                try:
                    self._editor.close_tab(tab_id)
                except KeyError:
                    continue

            restored_ids: list[str] = []
            for entry in entries:
                tab = self._create_tab_from_settings_entry(entry)
                if tab is not None:
                    restored_ids.append(tab.id)

            if not restored_ids:
                return False

            active_id = settings.active_tab_id or restored_ids[-1]
            if active_id not in self._workspace.tab_ids():
                active_id = restored_ids[-1]
            self._workspace.set_active_tab(active_id)
            next_index = getattr(settings, "next_untitled_index", None)
            if isinstance(next_index, int):
                self._workspace.set_next_untitled_index(next_index)
            monitor.update_autosave_indicator(document=self._editor.to_document())
            return True
        finally:
            self._restoring_workspace = False

    def _create_tab_from_settings_entry(self, entry: Mapping[str, Any]) -> DocumentTab | None:
        title = str(entry.get("title") or self._untitled_document_name)
        path_value = entry.get("path")
        path = Path(path_value).expanduser() if path_value else None
        language = str(entry.get("language") or (self._infer_language(path) if path else "markdown"))
        document = DocumentState(text="", metadata=DocumentMetadata(path=path, language=language))
        document.dirty = bool(entry.get("dirty", False))
        if path is not None:
            try:
                document.text = file_io.read_text(path)
            except Exception:
                document.text = ""
        untitled_index_value: int | None = None
        raw_index = entry.get("untitled_index")
        if raw_index is not None:
            try:
                untitled_index_value = int(raw_index)
            except (TypeError, ValueError):
                untitled_index_value = None

        monitor = self._require_document_monitor()
        with monitor.suspend_snapshot_persistence():
            tab = self._editor.create_tab(
                document=document,
                path=path,
                title=title,
                make_active=False,
                tab_id=entry.get("tab_id"),
                untitled_index=untitled_index_value,
            )

        if path is not None:
            self.apply_pending_snapshot_for_path(path, tab_id=tab.id, silent=True)
        else:
            self.apply_untitled_snapshot(tab.id)

        self._editor.refresh_tab_title(tab.id)
        return tab

    def _load_snapshot_document(self, snapshot: dict[str, Any], *, path: Path | None, tab_id: str | None) -> None:
        text = str(snapshot.get("text", ""))
        language = str(snapshot.get("language") or (self._infer_language(path) if path is not None else "markdown"))
        selection_raw = snapshot.get("selection")
        if isinstance(selection_raw, Sequence) and len(selection_raw) == 2:
            selection = SelectionRange(int(selection_raw[0]), int(selection_raw[1]))
        else:
            selection = SelectionRange()

        document = DocumentState(
            text=text,
            metadata=DocumentMetadata(path=path, language=language),
            selection=selection,
            dirty=True,
        )
        monitor = self._require_document_monitor()
        with monitor.suspend_snapshot_persistence():
            self._editor.load_document(document, tab_id=tab_id)

        if tab_id is None or self._workspace.active_tab_id == tab_id:
            self._current_path = path
        monitor.record_snapshot_digest(path, tab_id, text)
        if tab_id is not None:
            self._editor.refresh_tab_title(tab_id)
        monitor.update_autosave_indicator(document=document)

    def _handle_missing_last_file(self, settings: Settings) -> None:
        if settings.last_open_file:
            settings.last_open_file = None
            self.persist_settings(settings)
        self._status_updater("Last session file missing")

    def _cleanup_orphan_snapshots(self, settings: Settings, *, cache: UnsavedCache | None) -> bool:
        if cache is None:
            return False
        entries = [entry for entry in (settings.open_tabs or []) if isinstance(entry, Mapping)]
        active_paths: set[str] = set()
        active_tab_ids: set[str] = set()
        for entry in entries:
            tab_id = entry.get("tab_id")
            if tab_id:
                active_tab_ids.add(str(tab_id))
            path_value = entry.get("path")
            if path_value:
                normalized = self._normalize_snapshot_path(path_value)
                if normalized:
                    active_paths.add(normalized)

        snapshots = dict(cache.unsaved_snapshots or {})
        removed_file_snapshots = False
        for key in list(snapshots):
            if key not in active_paths:
                snapshots.pop(key, None)
                removed_file_snapshots = True
        if removed_file_snapshots:
            cache.unsaved_snapshots = snapshots

        untitled = dict(cache.untitled_snapshots or {})
        removed_untitled_snapshots = False
        for tab_id in list(untitled):
            if tab_id not in active_tab_ids:
                untitled.pop(tab_id, None)
                removed_untitled_snapshots = True
        if removed_untitled_snapshots:
            cache.untitled_snapshots = untitled

        return removed_file_snapshots or removed_untitled_snapshots

    @staticmethod
    def _normalize_snapshot_path(value: Any) -> str | None:
        try:
            return str(Path(value).expanduser().resolve())
        except Exception:
            try:
                return str(Path(str(value)).expanduser().resolve())
            except Exception:
                return None

    def _infer_language(self, path: Path | None) -> str:
        if path is None:
            return "markdown"
        suffix = path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            return "markdown"
        if suffix in {".yaml", ".yml"}:
            return "yaml"
        if suffix == ".json":
            return "json"
        if suffix in {".txt", ""}:
            return "text"
        return "plain"

    def _require_document_monitor(self) -> Any:
        monitor = self._document_monitor_resolver()
        if monitor is None:
            raise RuntimeError("Document monitor unavailable")
        return monitor

    def _require_review_controller(self) -> AIReviewController:
        if self._review_controller is None:
            raise RuntimeError("Review controller unavailable")
        return self._review_controller