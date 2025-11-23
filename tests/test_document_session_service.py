"""Unit tests for :mod:`tinkerbell.ui.document_session_service`."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tinkerbell.editor.document_model import DocumentMetadata, DocumentState
from tinkerbell.services.settings import Settings
from tinkerbell.services.unsaved_cache import UnsavedCache
from tinkerbell.ui.document_session_service import DocumentSessionService
from tinkerbell.ui.models.window_state import WindowContext


class _StubWorkspace:
    def __init__(self) -> None:
        self.active_tab = SimpleNamespace(id="tab-1")
        self.active_tab_id = "tab-1"
        self._state = {"open_tabs": [], "active_tab_id": None, "untitled_counter": 1}
        self.next_untitled_index: int | None = None
        self._ensure_calls = 0

    def serialize_state(self) -> dict[str, Any]:
        return dict(self._state)

    def set_next_untitled_index(self, value: int) -> None:
        self.next_untitled_index = value

    def ensure_tab(self) -> SimpleNamespace:
        self._ensure_calls += 1
        tab = SimpleNamespace(id=f"ensured-{self._ensure_calls}")
        self.active_tab = tab
        return tab

    def tab_ids(self) -> list[str]:  # pragma: no cover - provided for completeness
        return []

    def set_active_tab(self, tab_id: str) -> None:  # pragma: no cover - helper for future tests
        self.active_tab_id = tab_id


class _StubEditor:
    def __init__(self) -> None:
        self.loaded: list[tuple[DocumentState, str | None]] = []
        self._document = DocumentState(text="", metadata=DocumentMetadata())
        self.refreshed: list[str] = []

    def to_document(self) -> DocumentState:
        return self._document

    def load_document(self, document: DocumentState, tab_id: str | None = None) -> None:
        self._document = document
        self.loaded.append((document, tab_id))

    def refresh_tab_title(self, tab_id: str) -> None:
        self.refreshed.append(tab_id)


class _NoopContext:
    def __enter__(self) -> None:  # noqa: D401 - simple context shim
        return None

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - simple context shim
        return None


class _StubMonitor:
    def __init__(self) -> None:
        self.snapshot_keys: list[tuple[Path | str, str | None]] = []
        self.digests: list[tuple[Path | None, str | None, str]] = []
        self.autosave_updates: list[DocumentState] = []

    def snapshot_key(self, path: Path | str, *, tab_id: str | None = None) -> str:
        self.snapshot_keys.append((Path(path), tab_id))
        key = Path(path).expanduser()
        if tab_id:
            return f"{key}:{tab_id}"
        return str(key)

    def suspend_snapshot_persistence(self) -> _NoopContext:
        return _NoopContext()

    def record_snapshot_digest(self, path: Path | None, tab_id: str | None, text: str) -> None:
        self.digests.append((path, tab_id, text))

    def update_autosave_indicator(self, document: DocumentState) -> None:
        self.autosave_updates.append(document)


class _StubSettingsStore:
    def __init__(self) -> None:
        self.saved: list[Settings] = []

    def save(self, settings: Settings) -> Path:
        self.saved.append(settings)
        return Path("settings.json")


class _StubUnsavedCacheStore:
    def __init__(self) -> None:
        self.saved: list[UnsavedCache] = []

    def save(self, cache: UnsavedCache) -> Path:
        self.saved.append(cache)
        return Path("unsaved_cache.json")


def _make_service(settings: Settings | None = None) -> tuple[DocumentSessionService, SimpleNamespace]:
    workspace = _StubWorkspace()
    editor = _StubEditor()
    monitor = _StubMonitor()
    store = _StubSettingsStore()
    cache_store = _StubUnsavedCacheStore()
    settings_obj = settings or Settings()
    cache_obj = UnsavedCache()
    context = WindowContext(
        settings=settings_obj,
        settings_store=store,
        unsaved_cache=cache_obj,
        unsaved_cache_store=cache_store,
    )
    status_messages: list[str] = []
    opened_paths: list[Path] = []

    service = DocumentSessionService(
        context=context,
        editor=editor,
        workspace=workspace,
        document_monitor_resolver=lambda: monitor,
        open_document=lambda path: opened_paths.append(Path(path)),
        status_updater=status_messages.append,
        qt_parent_provider=lambda: object(),
        untitled_document_name="Untitled",
        untitled_snapshot_key="__untitled__",
    )

    tracker = SimpleNamespace(
        workspace=workspace,
        editor=editor,
        monitor=monitor,
        store=store,
        cache_store=cache_store,
        context=context,
        status_messages=status_messages,
        opened_paths=opened_paths,
    )
    return service, tracker


def test_remember_recent_file_updates_settings_and_persists(tmp_path: Path) -> None:
    settings = Settings()
    previous = tmp_path / "old.md"
    settings.recent_files = [str(previous)]
    service, tracker = _make_service(settings=settings)

    target = tmp_path / "notes.md"
    target.write_text("Hello", encoding="utf-8")
    service.remember_recent_file(target)

    normalized = str(target.expanduser().resolve())
    assert settings.recent_files[0] == normalized
    assert settings.last_open_file == normalized
    assert tracker.store.saved[-1] is settings


def test_sync_workspace_state_serializes_and_optionally_persists() -> None:
    settings = Settings()
    service, tracker = _make_service(settings=settings)
    tracker.workspace._state = {
        "open_tabs": [{"tab_id": "tab-1", "title": "Doc"}],
        "active_tab_id": "tab-1",
        "untitled_counter": 7,
    }

    service.sync_workspace_state()

    assert settings.open_tabs == tracker.workspace._state["open_tabs"]
    assert settings.active_tab_id == "tab-1"
    assert settings.next_untitled_index == 7
    assert tracker.store.saved[-1] is settings

    tracker.store.saved.clear()
    service.sync_workspace_state(persist=False)
    assert tracker.store.saved == []


def test_restore_last_session_document_reopens_previous_file(tmp_path: Path) -> None:
    target = tmp_path / "session.md"
    target.write_text("Recovered", encoding="utf-8")
    settings = Settings()
    settings.last_open_file = str(target)
    service, tracker = _make_service(settings=settings)

    sync_calls: list[bool] = []
    original_sync = service.sync_workspace_state

    def _record_sync(*, persist: bool = True) -> None:
        sync_calls.append(persist)
        original_sync(persist=persist)

    service.sync_workspace_state = _record_sync  # type: ignore[assignment]

    service.restore_last_session_document()

    assert tracker.opened_paths == [target]
    assert sync_calls == [False]


def test_restore_last_session_document_cleans_orphan_snapshots(tmp_path: Path) -> None:
    ghost = tmp_path / "ghost.md"
    ghost.write_text("ghost", encoding="utf-8")
    normalized = str(ghost.expanduser().resolve())
    settings = Settings()
    service, tracker = _make_service(settings=settings)
    cache = tracker.context.unsaved_cache
    cache.unsaved_snapshots = {
        normalized: {"text": "Lost", "language": "markdown"},
        str(tmp_path / "stale.md"): {"text": "Stale", "language": "markdown"},
    }
    cache.untitled_snapshots = {"tab-stale": {"text": "scratch", "language": "markdown"}}

    service.restore_last_session_document()

    assert cache.unsaved_snapshots == {}
    assert cache.untitled_snapshots == {}
    assert tracker.cache_store.saved  # cleanup persisted to disk


def test_apply_pending_snapshot_for_path_restores_document(tmp_path: Path) -> None:
    service, tracker = _make_service(settings=Settings())
    path = tmp_path / "draft.md"
    key = str(path.expanduser())
    cache = tracker.context.unsaved_cache
    cache.unsaved_snapshots[key] = {
        "text": "Recovered text",
        "language": "markdown",
    }

    restored = service.apply_pending_snapshot_for_path(path)

    assert restored is True
    assert tracker.editor.to_document().text == "Recovered text"
    assert tracker.monitor.digests[-1][0] == path
    assert tracker.status_messages[-1].startswith("Restored unsaved changes")


def test_apply_pending_snapshot_for_path_rejects_selection_payload(tmp_path: Path) -> None:
    service, tracker = _make_service(settings=Settings())
    path = tmp_path / "draft.md"
    key = str(path.expanduser())
    cache = tracker.context.unsaved_cache
    cache.unsaved_snapshots[key] = {
        "text": "Recovered text",
        "language": "markdown",
        "selection": (0, 8),
    }

    with pytest.raises(ValueError, match="selection"):
        service.apply_pending_snapshot_for_path(path)


def test_apply_untitled_snapshot_loads_tab_specific_snapshot() -> None:
    service, tracker = _make_service(settings=Settings())
    cache = tracker.context.unsaved_cache
    cache.untitled_snapshots["tab-7"] = {
        "text": "Untitled text",
        "language": "text",
    }

    applied = service.apply_untitled_snapshot("tab-7")

    assert applied is True
    assert tracker.editor.to_document().text == "Untitled text"


def test_prompt_for_save_path_omits_selection_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    service, tracker = _make_service(settings=Settings())
    document = DocumentState(text="Hello world", metadata=DocumentMetadata())
    captured: dict[str, Any] = {}

    def _fake_save_dialog(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return tmp_path / "output.md"

    fake_module = SimpleNamespace(save_file_dialog=_fake_save_dialog)
    fake_module.__spec__ = SimpleNamespace()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tinkerbell.widgets.dialogs", fake_module)

    path = service.prompt_for_save_path(document=document)

    assert path == tmp_path / "output.md"
    assert captured["document_text"] == "Hello world"
    assert "selection_text" not in captured
    assert captured["token_budget"] == tracker.context.settings.max_context_tokens
    assert captured["parent"] is not None