"""Main window behavior tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

from tinkerbell.editor.document_model import DocumentState
from tinkerbell.main_window import MainWindow, WindowContext
from tinkerbell.services.settings import Settings, SettingsStore


def _ensure_qapp() -> None:
    """Create a minimal QApplication when PySide6 is available."""

    try:
        from PySide6.QtWidgets import QApplication  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - PySide6 optional in tests
        return

    if QApplication.instance() is None:  # pragma: no cover - depends on PySide6
        QApplication([])


def _make_window(controller: Any | None = None) -> MainWindow:
    _ensure_qapp()
    return MainWindow(WindowContext(settings=Settings(), ai_controller=controller))


def test_main_window_registers_default_actions():
    window = _make_window()
    action_keys = set(window.actions.keys())
    assert {"file_open", "file_save", "file_save_as", "settings_open", "ai_snapshot"}.issubset(action_keys)
    assert window.last_status_message == "Ready"


def test_menu_specs_expose_file_and_settings_actions():
    window = _make_window()
    menus = {spec.name: spec for spec in window.menu_specs()}

    assert "file" in menus
    assert menus["file"].actions == ("file_open", "file_save", "file_save_as")

    assert "settings" in menus
    assert menus["settings"].actions == ("settings_open",)


def test_open_document_loads_editor_state(tmp_path: Path):
    target = tmp_path / "example.md"
    target.write_text("Hello world", encoding="utf-8")

    window = _make_window()
    window.open_document(target)

    document = window.editor_widget.to_document()
    assert document.text == "Hello world"
    assert document.metadata.path == target
    assert not document.dirty


def test_window_title_reflects_active_document(tmp_path: Path):
    target = tmp_path / "notes.md"
    target.write_text("Hello", encoding="utf-8")

    window = _make_window()
    window.open_document(target)

    assert window.windowTitle() == "notes.md - TinkerBell"


def test_window_title_dirty_indicator_clears_after_save(tmp_path: Path):
    target = tmp_path / "draft.md"
    target.write_text("seed", encoding="utf-8")

    window = _make_window()
    window.open_document(target)
    window.editor_widget.set_text("Updated content")

    assert window.windowTitle().startswith("*draft.md")

    window.save_document()

    assert window.windowTitle() == "draft.md - TinkerBell"


def test_save_document_persists_changes(tmp_path: Path):
    window = _make_window()
    target = tmp_path / "draft.md"
    document = DocumentState(text="Draft text")
    document.metadata.path = target
    document.dirty = True
    window.editor_widget.load_document(document)

    saved_path = window.save_document()

    assert saved_path == target
    assert target.read_text(encoding="utf-8") == "Draft text"
    assert not window.editor_widget.to_document().dirty


def test_save_document_prompts_for_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    window = _make_window()
    document = DocumentState(text="Draft text")
    window.editor_widget.load_document(document)
    target = tmp_path / "draft.md"

    monkeypatch.setattr(window, "_prompt_for_save_path", lambda: target)

    saved_path = window.save_document()

    assert saved_path == target
    assert target.read_text(encoding="utf-8") == "Draft text"


def test_save_document_cancelled(monkeypatch: pytest.MonkeyPatch):
    window = _make_window()
    window.editor_widget.load_document(DocumentState(text="content"))

    monkeypatch.setattr(window, "_prompt_for_save_path", lambda: None)

    with pytest.raises(RuntimeError):
        window.save_document()


def test_settings_dialog_updates_context_and_persists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    window = _make_window()

    class _Store(SettingsStore):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.saved: list[Settings] = []

        def save(self, settings: Settings) -> Path:  # type: ignore[override]
            self.saved.append(settings)
            return self._path

    window._context.settings_store = _Store(tmp_path / "settings.json")
    window._context.settings = Settings(model="gpt-4o-mini")
    updated = Settings(model="gpt-4.1-mini", base_url="https://example.com")
    result = SimpleNamespace(accepted=True, settings=updated)
    monkeypatch.setattr(window, "_show_settings_dialog", lambda current: result)

    window._handle_settings_requested()

    assert window._context.settings == updated
    assert window.last_status_message == "Settings updated"
    assert window._context.settings_store.saved[-1] == updated  # type: ignore[union-attr]


def test_settings_dialog_cancel_does_not_persist(monkeypatch: pytest.MonkeyPatch):
    window = _make_window()
    window._context.settings = Settings()
    monkeypatch.setattr(
        window,
        "_show_settings_dialog",
        lambda current: SimpleNamespace(accepted=False, settings=current),
    )

    window._handle_settings_requested()

    assert window.last_status_message == "Settings unchanged"


def test_update_status_tracks_message():
    window = _make_window()
    window.update_status("Testing status")
    assert window.last_status_message == "Testing status"


class _StubAIController:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.cancelled = False

    async def run_chat(self, prompt: str, snapshot: dict, *, metadata=None, on_event=None) -> dict:
        self.prompts.append(prompt)
        if on_event is not None:
            await on_event(SimpleNamespace(type="content.delta", content="Hello "))
            await on_event(SimpleNamespace(type="content.done", content="world!"))
        return {"response": "Hello world!"}

    def cancel(self) -> None:
        self.cancelled = True


def test_chat_prompt_without_controller_emits_notice():
    window = _make_window()
    panel = window.chat_panel
    panel.set_composer_text("Help me")
    panel.send_prompt()
    history = panel.history()
    assert history[-1].role == "assistant"
    assert "ai assistant" in history[-1].content.lower()


def test_chat_prompt_routes_to_ai_controller():
    controller = _StubAIController()
    window = _make_window(controller)
    panel = window.chat_panel
    panel.set_composer_text("Summarize this")
    panel.send_prompt()

    history = panel.history()
    assert history[-1].role == "assistant"
    assert "hello world" in history[-1].content.lower()
    assert controller.prompts == ["Summarize this"]
