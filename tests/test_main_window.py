"""Main window behavior tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinkerbell.editor.document_model import DocumentState
from tinkerbell.main_window import MainWindow, WindowContext
from tinkerbell.services.settings import Settings


def _ensure_qapp() -> None:
    """Create a minimal QApplication when PySide6 is available."""

    try:
        from PySide6.QtWidgets import QApplication  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - PySide6 optional in tests
        return

    if QApplication.instance() is None:  # pragma: no cover - depends on PySide6
        QApplication([])


def _make_window() -> MainWindow:
    _ensure_qapp()
    return MainWindow(WindowContext(settings=Settings(), ai_controller=None))


def test_main_window_registers_default_actions():
    window = _make_window()
    action_keys = set(window.actions.keys())
    assert {"file_open", "file_save", "ai_snapshot"}.issubset(action_keys)
    assert window.last_status_message == "Ready"


def test_open_document_loads_editor_state(tmp_path: Path):
    target = tmp_path / "example.md"
    target.write_text("Hello world", encoding="utf-8")

    window = _make_window()
    window.open_document(target)

    document = window.editor_widget.to_document()
    assert document.text == "Hello world"
    assert document.metadata.path == target
    assert not document.dirty


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


def test_save_document_without_path_raises(tmp_path: Path):
    del tmp_path  # unused fixture
    window = _make_window()
    window.editor_widget.load_document(DocumentState(text="content"))

    with pytest.raises(RuntimeError):
        window.save_document()


def test_update_status_tracks_message():
    window = _make_window()
    window.update_status("Testing status")
    assert window.last_status_message == "Testing status"
