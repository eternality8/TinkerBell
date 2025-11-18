"""Main window behavior tests."""

from __future__ import annotations

from copy import deepcopy
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from types import SimpleNamespace

import pytest

import tinkerbell.ui.main_window as main_window_module
from tinkerbell.chat.chat_panel import ChatTurnSnapshot
from tinkerbell.chat.message_model import ChatMessage, EditDirective, ToolTrace
from tinkerbell.editor.document_model import DocumentState, SelectionRange
from tinkerbell.ui.main_window import MainWindow, WindowContext
from tinkerbell.services.importers import FileImporter, ImportResult, ImporterError
from tinkerbell.services.settings import Settings, SettingsStore
from tinkerbell.services.telemetry import ContextUsageEvent
from tinkerbell.theme import theme_manager


def _ensure_qapp() -> None:
    """Create a minimal QApplication when PySide6 is available."""

    try:
        from PySide6.QtWidgets import QApplication  # type: ignore[import-not-found]
    except Exception:  # pragma: no cover - PySide6 optional in tests
        return

    if QApplication.instance() is None:  # pragma: no cover - depends on PySide6
        QApplication([])


def _make_window(controller: Any | None = None, settings: Settings | None = None) -> MainWindow:
    _ensure_qapp()
    resolved_settings = settings or Settings()
    if controller is not None and not (resolved_settings.api_key or "").strip():
        resolved_settings.api_key = "test-key"
    return MainWindow(WindowContext(settings=resolved_settings, ai_controller=controller))


def _install_fake_langchain_module(monkeypatch: pytest.MonkeyPatch, sink: dict[str, Any]) -> None:
    class _FakeEmbeddings:
        def __init__(self, **kwargs: Any) -> None:
            sink.clear()
            sink.update(kwargs)

    monkeypatch.setitem(sys.modules, "langchain_openai", SimpleNamespace(OpenAIEmbeddings=_FakeEmbeddings))


def test_main_window_registers_default_actions():
    window = _make_window()
    action_keys = set(window.actions.keys())
    assert {
        "file_new_tab",
        "file_open",
        "file_import",
        "file_save",
        "file_revert",
        "file_save_as",
        "file_close_tab",
        "settings_open",
        "ai_snapshot",
    }.issubset(action_keys)
    assert window.last_status_message == "Ready"


def test_close_event_requests_application_quit(monkeypatch: pytest.MonkeyPatch):
    window = _make_window()
    quit_calls = {"count": 0}

    class _FakeApp:
        def closingDown(self) -> bool:
            return False

        def quit(self) -> None:
            quit_calls["count"] += 1

    fake_app = _FakeApp()
    monkeypatch.setattr(
        main_window_module,
        "QApplication",
        SimpleNamespace(instance=lambda: fake_app),
    )

    event = SimpleNamespace(accept=lambda: None)
    window.closeEvent(event)

    assert quit_calls["count"] == 1


def test_close_event_stops_asyncio_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _make_window()

    class _FakeApp:
        def closingDown(self) -> bool:
            return False

        def quit(self) -> None:
            pass

    fake_app = _FakeApp()
    monkeypatch.setattr(
        main_window_module,
        "QApplication",
        SimpleNamespace(instance=lambda: fake_app),
    )

    stop_calls = {"scheduled": 0, "direct": 0}

    def _fake_stop() -> None:
        stop_calls["direct"] += 1

    def _fake_call_soon(callback):
        stop_calls["scheduled"] += 1
        assert callback is _fake_stop

    fake_loop = SimpleNamespace(
        is_running=lambda: True,
        stop=_fake_stop,
        call_soon=_fake_call_soon,
    )
    monkeypatch.setattr(main_window_module.asyncio, "get_running_loop", lambda: fake_loop)

    window.closeEvent(SimpleNamespace(accept=lambda: None))

    assert stop_calls == {"scheduled": 1, "direct": 0}


def test_chat_suggestions_initialized_on_startup():
    window = _make_window()
    suggestions = window.chat_panel.suggestions()
    assert "Draft an outline for Untitled." in suggestions


def test_chat_panel_visibility_controlled_by_settings():
    window = _make_window(settings=Settings(show_tool_activity_panel=True))

    assert window.chat_panel.tool_activity_visible is True


def test_document_edit_tool_runs_in_patch_only_mode():
    controller = _StubAIController()
    _make_window(controller, settings=Settings(api_key="key"))
    edit_tool = controller.registered_tools["document_edit"]["impl"]

    assert getattr(edit_tool, "patch_only", None) is True


def test_tool_traces_capture_compaction_metadata():
    window = _make_window()
    trace = ToolTrace(
        name="search",
        input_summary="term",
        output_summary="full output",
        metadata={"tool_call_id": "call-xyz"},
    )
    window.chat_panel.show_tool_trace(trace)
    window._tool_trace_index["call-xyz"] = trace
    pointer = {
        "pointer_id": "ptr-99",
        "display_text": "summary",
        "rehydrate_instructions": "run search",
    }

    window._annotate_tool_traces_with_compaction([{"id": "call-xyz", "pointer": pointer}])

    assert trace.metadata.get("compacted") is True
    assert trace.output_summary == "summary"
    details = window.chat_panel._format_tool_trace_details(trace)
    assert "Compacted pointer: ptr-99" in details
    assert "Rehydrate instructions:" in details


def test_settings_dialog_toggle_updates_tool_panel(monkeypatch: pytest.MonkeyPatch):
    initial = Settings(show_tool_activity_panel=False)
    window = _make_window(settings=initial)
    updated = Settings(show_tool_activity_panel=True)

    monkeypatch.setattr(
        window,
        "_show_settings_dialog",
        lambda current: SimpleNamespace(accepted=True, settings=updated),
    )

    window._handle_settings_requested()

    assert window.chat_panel.tool_activity_visible is True
    assert window._context.settings is updated


def test_selection_updates_chat_suggestions_and_metadata():
    window = _make_window()
    window.editor_widget.set_text("Alpha beta gamma")
    window.editor_widget.apply_selection(SelectionRange(0, 5))

    suggestions = window.chat_panel.suggestions()
    assert "Summarize the selected text." in suggestions
    assert "Rewrite the selected text for clarity." in suggestions

    captured: list[dict[str, Any]] = []

    def _collector(prompt: str, metadata: dict[str, Any]) -> None:
        del prompt
        captured.append(metadata)

    window.chat_panel.add_request_listener(_collector)
    window.chat_panel.set_composer_text("Please help")
    window.chat_panel.send_prompt()

    assert captured
    assert captured[0]["selection_summary"] == "Alpha"


def test_suggestion_panel_without_history_keeps_default_suggestions():
    controller = _StubAIController()
    window = _make_window(controller)
    baseline = window.chat_panel.suggestions()

    window._handle_suggestion_panel_toggled(True)

    updated = window.chat_panel.suggestions()
    assert baseline == updated
    assert controller.suggestion_calls == []


def test_suggestion_panel_with_history_requests_ai_suggestions():
    controller = _StubAIController()
    controller.suggestion_responses.append(["Idea A", "Idea B"])
    window = _make_window(controller)
    window.chat_panel.append_user_message("Hi there")
    window.chat_panel.append_ai_message(ChatMessage(role="assistant", content="Welcome back"))

    window._handle_suggestion_panel_toggled(True)

    suggestions = window.chat_panel.suggestions()
    assert suggestions == ("Idea A", "Idea B")
    assert controller.suggestion_calls
    payload = controller.suggestion_calls[-1]
    assert payload["max"] == 4
    assert payload["history"][0]["content"] == "Hi there"


def test_apply_embedding_metadata_updates_snapshot():
    window = _make_window()
    controller = window._embedding_controller
    controller._embedding_snapshot_metadata = {
        "embedding_backend": "langchain",
        "embedding_model": "deepseek-embedding",
        "embedding_status": "ready",
        "embedding_detail": "LangChain/DeepSeek",
    }
    snapshot: dict[str, Any] = {}

    window._apply_embedding_metadata(snapshot)

    assert snapshot["embedding_backend"] == "langchain"
    assert snapshot["embedding_model"] == "deepseek-embedding"
    assert snapshot["embedding_status"] == "ready"
    assert snapshot["embedding_detail"] == "LangChain/DeepSeek"

    controller._embedding_snapshot_metadata = {}
    snapshot = {
        "embedding_backend": "old",
        "embedding_model": "legacy",
        "embedding_status": "error",
        "embedding_detail": "failure",
    }

    window._apply_embedding_metadata(snapshot)

    for field in ("embedding_backend", "embedding_model", "embedding_status", "embedding_detail"):
        assert field not in snapshot


def test_langchain_autodetects_deepseek_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _make_window()
    controller = window._embedding_controller
    settings = Settings()
    sink: dict[str, Any] = {}
    _install_fake_langchain_module(monkeypatch, sink)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "secret-deepseek")

    embeddings, template = controller._instantiate_langchain_embeddings("deepseek-embedding", settings)

    assert embeddings is not None
    assert template.family == "deepseek"
    assert sink["model"] == "deepseek-embedding"
    assert sink["base_url"] == "https://api.deepseek.com/v1"
    assert sink["api_key"] == "secret-deepseek"
    assert sink["dimensions"] == 1536
    assert sink["tiktoken_model_name"] == "cl100k_base"


def test_langchain_autodetect_respects_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _make_window()
    controller = window._embedding_controller
    settings = Settings(base_url="https://api.custom.example/v1", api_key="glm-key")
    sink: dict[str, Any] = {}
    _install_fake_langchain_module(monkeypatch, sink)

    embeddings, template = controller._instantiate_langchain_embeddings("glm-4-embed", settings)

    assert embeddings is not None
    assert template.family == "glm"
    assert sink["base_url"] == "https://api.custom.example/v1"
    assert sink["api_key"] == "glm-key"
    assert sink["dimensions"] == 1024


def test_langchain_provider_forced_via_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _make_window()
    controller = window._embedding_controller
    settings = Settings()
    settings.metadata["langchain_provider_family"] = "moonshot"
    sink: dict[str, Any] = {}
    _install_fake_langchain_module(monkeypatch, sink)
    monkeypatch.setenv("MOONSHOT_API_KEY", "moon-key")

    embeddings, template = controller._instantiate_langchain_embeddings("custom-provider-model", settings)

    assert embeddings is not None
    assert template.family == "moonshot"
    assert sink["base_url"] == "https://api.moonshot.cn/v1"
    assert sink["api_key"] == "moon-key"


def test_suggestion_panel_reuses_cached_results_for_same_history():
    controller = _StubAIController()
    controller.suggestion_responses.append(["Idea A"])
    window = _make_window(controller)
    window.chat_panel.append_user_message("Hi there")
    window.chat_panel.append_ai_message(ChatMessage(role="assistant", content="Welcome back"))

    window._handle_suggestion_panel_toggled(True)
    assert window.chat_panel.suggestions() == ("Idea A",)
    assert len(controller.suggestion_calls) == 1

    window._handle_suggestion_panel_toggled(False)
    controller.suggestion_responses.append(["Idea B"])

    window._handle_suggestion_panel_toggled(True)

    assert window.chat_panel.suggestions() == ("Idea A",)
    assert len(controller.suggestion_calls) == 1


def test_context_usage_status_includes_compaction_stats():
    controller = SimpleNamespace(
        get_recent_context_events=lambda limit=None: [
            ContextUsageEvent(
                document_id="doc",
                model="gpt",
                prompt_tokens=100,
                tool_tokens=50,
                response_reserve=1_000,
                timestamp=0.0,
                conversation_length=2,
                tool_names=("search",),
                run_id="run-1",
            )
        ],
        get_budget_status=lambda: None,
    )
    settings = Settings()
    settings.debug.token_logging_enabled = True
    window = _make_window(controller=controller, settings=settings)
    telemetry = window._telemetry_controller
    telemetry.set_compaction_stats({"total_compactions": 2, "tokens_saved": 500})

    telemetry.refresh_context_usage_status()

    assert "Compactions 2" in window._status_bar.memory_usage


def test_subagent_indicator_off_when_feature_disabled() -> None:
    window = _make_window(settings=Settings(enable_subagents=False))

    status, detail = window._status_bar.subagent_state
    assert status == "Off"
    assert "subagents" in detail.lower()


def test_subagent_indicator_updates_from_telemetry_events() -> None:
    window = _make_window(settings=Settings(enable_subagents=True))

    # Initial state should be idle with a helpful tooltip message
    status, detail = window._status_bar.subagent_state
    assert status == "Idle"
    assert "telemetry" in detail.lower()

    telemetry = window._telemetry_controller

    telemetry.handle_subagent_telemetry(
        {
            "event": "subagent.job_started",
            "job_id": "job-1",
            "chunk_id": "chunk-xyz",
            "token_estimate": 512,
        }
    )

    status, detail = window._status_bar.subagent_state
    assert status == "Running (1)"
    assert "job-1" in detail

    telemetry.handle_subagent_telemetry(
        {
            "event": "subagent.job_completed",
            "job_id": "job-1",
            "chunk_id": "chunk-xyz",
            "tokens_used": 128,
            "latency_ms": 25,
        }
    )

    status, detail = window._status_bar.subagent_state
    assert status == "Idle"
    assert "Done 1" in detail


def test_suggestion_cache_invalidated_when_history_changes():
    controller = _StubAIController()
    controller.suggestion_responses.append(["Idea A"])
    controller.suggestion_responses.append(["Idea B"])
    window = _make_window(controller)
    window.chat_panel.append_user_message("Hi there")

    window._handle_suggestion_panel_toggled(True)
    assert len(controller.suggestion_calls) == 1

    window._handle_suggestion_panel_toggled(False)
    window.chat_panel.append_ai_message(ChatMessage(role="assistant", content="Adding detail"))

    window._handle_suggestion_panel_toggled(True)

    assert len(controller.suggestion_calls) == 2
    assert window.chat_panel.suggestions() == ("Idea B",)


def test_menu_specs_expose_file_and_settings_actions():
    window = _make_window()
    menus = {spec.name: spec for spec in window.menu_specs()}

    assert "file" in menus
    assert menus["file"].actions == (
        "file_new_tab",
        "file_open",
        "file_import",
        "file_save",
        "file_save_as",
        "file_close_tab",
        "file_revert",
    )

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


def test_import_action_creates_new_tab(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    window = _make_window()
    source = tmp_path / "sample.pdf"
    source.write_text("binary", encoding="utf-8")
    result = ImportResult(text="Converted PDF", title="Sample (imported)", language="markdown")
    stub_importer = FileImporter(handlers=())

    def _import(path: Path) -> ImportResult:
        assert path == source
        return result

    stub_importer.import_file = _import  # type: ignore[assignment]
    stub_importer.dialog_filter = lambda: "PDF (*.pdf)"  # type: ignore[assignment]
    window._file_importer = stub_importer
    monkeypatch.setattr(window, "_prompt_for_import_path", lambda: source)

    window._handle_import_requested()

    document = window.editor_widget.to_document()
    assert document.text == "Converted PDF"
    assert document.metadata.path is None
    assert document.metadata.language == "markdown"
    assert window.last_status_message.startswith("Imported sample.pdf")


def test_import_action_handles_handler_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    window = _make_window()
    source = tmp_path / "broken.pdf"
    source.write_text("binary", encoding="utf-8")

    failing_importer = FileImporter(handlers=())

    def _fail(_: Path) -> ImportResult:
        raise ImporterError("Cannot import")

    failing_importer.import_file = _fail  # type: ignore[assignment]
    failing_importer.dialog_filter = lambda: "PDF (*.pdf)"  # type: ignore[assignment]
    window._file_importer = failing_importer
    monkeypatch.setattr(window, "_prompt_for_import_path", lambda: source)

    window._handle_import_requested()

    assert window.last_status_message == "Cannot import"


def test_open_document_records_last_open_file(tmp_path: Path):
    target = tmp_path / "example.md"
    target.write_text("Hello world", encoding="utf-8")

    settings = Settings()
    window = _make_window(settings=settings)
    window.open_document(target)

    assert settings.last_open_file == str(target.resolve())
    assert settings.unsaved_snapshot is None
    assert not settings.untitled_snapshots


def test_unsaved_snapshot_persisted_for_untitled_document():
    settings = Settings()
    window = _make_window(settings=settings)

    tab_id = window.editor_widget.active_tab_id()
    assert tab_id is not None

    window.editor_widget.set_text("Scratch pad")

    snapshot = (settings.untitled_snapshots or {}).get(tab_id)
    assert snapshot is not None
    assert snapshot["text"] == "Scratch pad"
    assert snapshot["language"] == window.editor_widget.to_document().metadata.language
    assert settings.open_tabs[0]["tab_id"] == tab_id


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

    monkeypatch.setattr(window, "_prompt_for_save_path", lambda **_: target)

    saved_path = window.save_document()

    assert saved_path == target
    assert target.read_text(encoding="utf-8") == "Draft text"


def test_save_document_cancelled(monkeypatch: pytest.MonkeyPatch):
    window = _make_window()
    window.editor_widget.load_document(DocumentState(text="content"))

    monkeypatch.setattr(window, "_prompt_for_save_path", lambda **_: None)

    with pytest.raises(RuntimeError):
        window.save_document()


def test_main_window_diff_overlay_helpers() -> None:
    window = _make_window()
    document = window.editor_widget.to_document()
    trace = ToolTrace(name="edit:patch", input_summary="", output_summary="Δ", metadata={"spans": [(0, 1)], "diff_preview": "@@"})

    window._apply_diff_overlay(trace, document=document, range_hint=(0, 1))

    tab = window.editor_widget.workspace.active_tab
    assert tab is not None
    assert tab.editor.diff_overlay is not None

    window._clear_diff_overlay()

    assert tab.editor.diff_overlay is None


def test_main_window_autosave_indicator_updates(tmp_path: Path) -> None:
    window = _make_window()
    status = window._status_bar

    assert status.autosave_state[0] == "Saved"

    window.editor_widget.set_text("changed content")

    dirty_label = status.autosave_state[0]
    assert "Autosaved" in dirty_label or dirty_label == "Unsaved changes"

    target = tmp_path / "autosave.md"
    window.save_document(target)

    assert status.autosave_state[0] == "Saved"


def test_last_session_file_restored_on_startup(tmp_path: Path):
    target = tmp_path / "session.md"
    target.write_text("Restored text", encoding="utf-8")

    settings = Settings(last_open_file=str(target))
    window = _make_window(settings=settings)

    document = window.editor_widget.to_document()
    assert document.text == "Restored text"
    assert document.metadata.path == target
    assert window.last_status_message == f"Loaded {target.name}"


def test_missing_last_session_file_is_cleared(tmp_path: Path):
    missing = tmp_path / "ghost.md"
    settings = Settings(last_open_file=str(missing))

    window = _make_window(settings=settings)

    assert settings.last_open_file is None
    assert window.last_status_message == "Last session file missing"


def test_unsaved_snapshot_restores_when_available():
    settings = Settings(unsaved_snapshot={"text": "Draft", "language": "markdown", "selection": (1, 3)})

    window = _make_window(settings=settings)

    document = window.editor_widget.to_document()
    assert document.text == "Draft"
    assert document.metadata.path is None
    assert tuple(document.selection.as_tuple()) == (1, 3)
    assert window.last_status_message == "Restored unsaved draft"


def test_workspace_tabs_restore_from_settings(tmp_path: Path) -> None:
    doc_path = tmp_path / "notes.md"
    doc_path.write_text("Disk text", encoding="utf-8")
    normalized = str(doc_path.resolve())

    settings = Settings(
        open_tabs=[
            {
                "tab_id": "draft-1",
                "title": "Draft",
                "dirty": True,
                "language": "markdown",
                "untitled_index": 4,
            },
            {
                "tab_id": "file-1",
                "title": "notes.md",
                "dirty": True,
                "language": "markdown",
                "path": str(doc_path),
            },
        ],
        active_tab_id="file-1",
        untitled_snapshots={"draft-1": {"text": "Draft", "language": "markdown", "selection": [0, 5]}},
        unsaved_snapshots={normalized: {"text": "Edited", "language": "markdown", "selection": [0, 6]}},
        next_untitled_index=5,
    )

    window = _make_window(settings=settings)
    workspace = window.editor_widget.workspace

    assert workspace.tab_count() == 2
    assert workspace.active_tab_id == "file-1"

    untitled_tab = workspace.get_tab("draft-1")
    assert untitled_tab.document().text == "Draft"

    file_tab = workspace.get_tab("file-1")
    assert file_tab.document().text == "Edited"

    assert settings.open_tabs[0]["tab_id"] == "draft-1"


def test_unsaved_snapshot_clears_after_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings = Settings()
    window = _make_window(settings=settings)
    window.editor_widget.set_text("Unsaved draft")
    target = tmp_path / "saved.md"

    monkeypatch.setattr(window, "_prompt_for_save_path", lambda **_: target)

    window.save_document()

    assert not settings.untitled_snapshots
    assert target.read_text(encoding="utf-8") == "Unsaved draft"


def test_file_snapshot_persisted_for_dirty_document(tmp_path: Path):
    target = tmp_path / "notes.md"
    target.write_text("Hello", encoding="utf-8")

    settings = Settings()
    window = _make_window(settings=settings)
    window.open_document(target)
    window.editor_widget.set_text("Hello world")

    normalized = str(target.resolve())
    assert settings.unsaved_snapshots[normalized]["text"] == "Hello world"


def test_open_document_restores_file_snapshot(tmp_path: Path):
    target = tmp_path / "notes.md"
    target.write_text("Hello", encoding="utf-8")
    normalized = str(target.resolve())
    settings = Settings(
        unsaved_snapshots={normalized: {"text": "Draft", "language": "markdown", "selection": [0, 5]}}
    )
    window = _make_window(settings=settings)

    window.open_document(target)

    document = window.editor_widget.to_document()
    assert document.text == "Draft"
    assert document.metadata.path == target
    assert document.dirty is True
    assert window.last_status_message == f"Restored unsaved changes for {target.name}"


def test_save_document_clears_file_snapshot(tmp_path: Path):
    target = tmp_path / "notes.md"
    target.write_text("Hello", encoding="utf-8")
    settings = Settings()
    window = _make_window(settings=settings)
    window.open_document(target)
    window.editor_widget.set_text("Hello world")

    window.save_document()

    assert str(target.resolve()) not in settings.unsaved_snapshots


def test_revert_discards_unsaved_changes(tmp_path: Path):
    target = tmp_path / "story.md"
    target.write_text("Original text", encoding="utf-8")
    settings = Settings()
    window = _make_window(settings=settings)
    window.open_document(target)
    window.editor_widget.set_text("Edited text")

    normalized = str(target.resolve())
    assert settings.unsaved_snapshots is not None
    assert settings.unsaved_snapshots[normalized]["text"] == "Edited text"

    window._handle_revert_requested()

    document = window.editor_widget.to_document()
    assert document.text == "Original text"
    assert document.dirty is False
    assert normalized not in (settings.unsaved_snapshots or {})
    assert window.last_status_message == f"Reverted {target.name}"


def test_revert_without_path_updates_status():
    window = _make_window()
    window.editor_widget.set_text("Scratch")

    window._handle_revert_requested()

    assert window.last_status_message == "No file to revert"


def test_startup_restores_file_snapshot(tmp_path: Path):
    target = tmp_path / "story.md"
    target.write_text("Once", encoding="utf-8")
    normalized = str(target.resolve())
    settings = Settings(
        last_open_file=normalized,
        unsaved_snapshots={normalized: {"text": "Once upon", "language": "markdown", "selection": [0, 4]}},
    )

    window = _make_window(settings=settings)

    document = window.editor_widget.to_document()
    assert document.text == "Once upon"
    assert document.metadata.path == target
    assert window.last_status_message == f"Restored unsaved changes for {target.name}"


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


def test_debug_logging_toggle_reconfigures_logging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = _StubAIController()
    window = _make_window(controller, Settings(debug_logging=False))
    updated = Settings(debug_logging=True)
    monkeypatch.setattr(
        window,
        "_show_settings_dialog",
        lambda current: SimpleNamespace(accepted=True, settings=updated),
    )

    captured: dict[str, Any] = {}

    def _fake_setup(level: int, **kwargs: Any):
        captured["level"] = level
        captured["force"] = kwargs.get("force")
        return tmp_path / "log"

    monkeypatch.setattr("tinkerbell.ui.main_window.logging_utils.setup_logging", _fake_setup)

    window._handle_settings_requested()

    assert captured["level"] == logging.DEBUG
    assert captured["force"] is True
    assert controller.client.settings.debug_logging is True
    assert window._context.settings is not None
    assert window._context.settings.debug_logging is True


def test_debug_logging_disable_reconfigures_logging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    controller = _StubAIController()
    controller.client.settings.debug_logging = True
    window = _make_window(controller, Settings(debug_logging=True))
    updated = Settings(debug_logging=False)
    monkeypatch.setattr(
        window,
        "_show_settings_dialog",
        lambda current: SimpleNamespace(accepted=True, settings=updated),
    )

    captured: dict[str, Any] = {}

    def _fake_setup(level: int, **kwargs: Any):
        captured["level"] = level
        captured["force"] = kwargs.get("force")
        return tmp_path / "log"

    monkeypatch.setattr("tinkerbell.ui.main_window.logging_utils.setup_logging", _fake_setup)

    window._handle_settings_requested()

    assert captured["level"] == logging.INFO
    assert captured["force"] is True
    assert controller.client.settings.debug_logging is False
    assert window._context.settings is not None
    assert window._context.settings.debug_logging is False


def test_theme_change_applies_immediately(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _make_window()
    applied: dict[str, Any] = {}

    def _capture_theme(theme_obj):
        applied["theme"] = theme_obj

    monkeypatch.setattr(window._editor, "apply_theme", _capture_theme)
    updated = Settings(theme="midnight")
    monkeypatch.setattr(
        window,
        "_show_settings_dialog",
        lambda current: SimpleNamespace(accepted=True, settings=updated),
    )

    window._handle_settings_requested()

    assert applied["theme"].name == theme_manager.resolve("midnight").name


def test_ai_client_reconfigured_on_connection_change(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _StubAIController()
    window = _make_window(controller, Settings(api_key="old-key"))
    updated = Settings(api_key="new-key", model="gpt-4.1-mini", base_url="https://example.com/v1")
    new_client = SimpleNamespace(settings=SimpleNamespace(debug_logging=False))
    monkeypatch.setattr(window, "_build_ai_client_from_settings", lambda cfg: new_client)
    monkeypatch.setattr(
        window,
        "_show_settings_dialog",
        lambda current: SimpleNamespace(accepted=True, settings=updated),
    )

    window._handle_settings_requested()

    assert controller.updated_clients[-1] is new_client


def test_ai_controller_created_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _make_window(controller=None, settings=Settings(api_key="", base_url="https://api", model="gpt-4o-mini"))
    sentinel_controller = SimpleNamespace(client=SimpleNamespace(settings=SimpleNamespace(debug_logging=False)))
    monkeypatch.setattr(window, "_build_ai_controller_from_settings", lambda cfg: sentinel_controller)
    called: dict[str, bool] = {}
    monkeypatch.setattr(window, "_register_default_ai_tools", lambda: called.setdefault("wired", True))
    updated = Settings(api_key="live-key", base_url="https://api", model="gpt-4o-mini")
    monkeypatch.setattr(
        window,
        "_show_settings_dialog",
        lambda current: SimpleNamespace(accepted=True, settings=updated),
    )

    window._handle_settings_requested()

    assert window._context.ai_controller is sentinel_controller
    assert called["wired"] is True


def test_ai_controller_disabled_when_credentials_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _StubAIController()
    initial = Settings(api_key="valid-key", base_url="https://api", model="gpt-4o-mini")
    window = _make_window(controller, initial)
    updated = Settings(api_key="", base_url="https://api", model="gpt-4o-mini")
    monkeypatch.setattr(
        window,
        "_show_settings_dialog",
        lambda current: SimpleNamespace(accepted=True, settings=updated),
    )

    window._handle_settings_requested()

    assert window._context.ai_controller is None
    assert window._ai_client_signature is None


def test_update_status_tracks_message():
    window = _make_window()
    window.update_status("Testing status")
    assert window.last_status_message == "Testing status"


def test_new_chat_session_cancels_active_ai_request():
    controller = _StubAIController()
    window = _make_window(controller)

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            self.cancelled = True

    task = _FakeTask()
    window._ai_task = task  # type: ignore[assignment]
    window.chat_panel.start_new_chat()

    assert controller.cancelled is True
    assert task.cancelled is True
    assert window._ai_task is None
    assert window.last_status_message == "Chat reset"


def test_accept_ai_changes_clears_overlays_and_drops_turn():
    window = _make_window()
    tab = window._workspace.active_tab
    assert tab is not None
    tab.editor.set_text("Original text")

    _begin_test_review(window)
    _apply_fake_ai_edit(window, tab, "AI updated text", diff_label="Δ1")
    window._review_controller.finalize_pending_turn_review(success=True)

    assert tab.id in window._tabs_with_overlay

    window._handle_accept_ai_changes()

    assert window._review_controller.pending_turn_review is None
    assert not window._tabs_with_overlay
    assert window._status_bar.review_controls_visible is False
    assert "Accepted" in window.last_status_message


def test_reject_ai_changes_restores_documents_and_chat():
    window = _make_window()
    primary = window._workspace.active_tab
    assert primary is not None
    primary.editor.set_text("Base primary text")
    window._handle_new_tab_requested()
    secondary = window._workspace.active_tab
    assert secondary is not None
    secondary.editor.set_text("Base secondary text")
    window._workspace.set_active_tab(primary.id)

    prompt = "Rewrite both documents"
    _begin_test_review(window, prompt=prompt)
    window.chat_panel.set_composer_text("")

    base_primary = primary.document().text
    base_secondary = secondary.document().text

    _apply_fake_ai_edit(window, primary, "AI primary", diff_label="ΔP")
    _apply_fake_ai_edit(window, secondary, "AI secondary", diff_label="ΔS")
    window._review_controller.finalize_pending_turn_review(success=True)
    window.chat_panel.set_composer_text("post-AI")

    window._handle_reject_ai_changes()

    assert primary.document().text == base_primary
    assert secondary.document().text == base_secondary
    assert window.chat_panel.composer_text == prompt
    assert window._review_controller.pending_turn_review is None
    assert not window._tabs_with_overlay
    assert "Rejected" in window.last_status_message


def test_manual_edit_during_pending_review_aborts_envelope():
    window = _make_window()
    tab = window._workspace.active_tab
    assert tab is not None
    tab.editor.set_text("Draft copy")

    _begin_test_review(window)
    _apply_fake_ai_edit(window, tab, "AI variant", diff_label="Δdraft")
    window._review_controller.finalize_pending_turn_review(success=True)

    tab.editor._mark_change_source("user")
    window._maybe_clear_diff_overlay(tab.document())

    assert window._review_controller.pending_turn_review is None
    assert not window._tabs_with_overlay
    assert "discarded" in window.last_status_message.lower()


def test_closed_tab_marked_orphaned_and_skipped_on_accept():
    window = _make_window()
    tab = window._workspace.active_tab
    assert tab is not None
    tab.editor.set_text("Old text")

    _begin_test_review(window)
    _apply_fake_ai_edit(window, tab, "AI version", diff_label="Δold")
    window._review_controller.finalize_pending_turn_review(success=True)
    pending = window._review_controller.pending_turn_review
    assert pending is not None

    window._handle_close_tab_requested()

    session = pending.tab_sessions.get(tab.id)
    assert session is not None and session.orphaned is True

    window._handle_accept_ai_changes()

    assert window._review_controller.pending_turn_review is None
    assert "skipped 1 closed tab" in window.last_status_message


def test_cancel_active_ai_turn_restores_composer_and_drops_review():
    controller = _StubAIController()
    window = _make_window(controller)
    tab = window._workspace.active_tab
    assert tab is not None
    tab.editor.set_text("Original content")

    prompt = "Original prompt"
    _begin_test_review(window, prompt=prompt)
    _apply_fake_ai_edit(window, tab, "AI change", diff_label="Δcancel")
    window.chat_panel.set_composer_text("")

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            self.cancelled = True

    task = _FakeTask()
    window._ai_task = task  # type: ignore[assignment]

    window._cancel_active_ai_turn()

    assert controller.cancelled is True
    assert task.cancelled is True
    assert window._ai_task is None
    assert window._review_controller.pending_turn_review is None
    assert window.chat_panel.composer_text == prompt
    assert not window._tabs_with_overlay
    assert "Canceled AI request" in window.last_status_message


class _StubAIController:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.cancelled = False
        self.registered_tools: dict[str, dict[str, Any]] = {}
        self.client = SimpleNamespace(settings=SimpleNamespace(debug_logging=False))
        self.updated_clients: list[Any] = []
        self.client = SimpleNamespace(settings=SimpleNamespace(debug_logging=False))
        self.suggestion_calls: list[dict[str, Any]] = []
        self.suggestion_responses: list[list[str]] = []
        self.stream_scripts: list[list[Any]] = []
        self.response_texts: list[str] = []
        self.history_payloads: list[Sequence[Mapping[str, str]] | None] = []

    async def run_chat(
        self,
        prompt: str,
        snapshot: dict,
        *,
        metadata=None,
        history=None,
        on_event=None,
    ) -> dict:
        self.prompts.append(prompt)
        self.history_payloads.append(history)
        if on_event is not None:
            script = self.stream_scripts.pop(0) if self.stream_scripts else None
            events = script or [
                SimpleNamespace(type="content.delta", content="Hello "),
                SimpleNamespace(type="content.delta", content="world!"),
                SimpleNamespace(type="content.done", content="Hello world!"),
            ]
            for payload in events:
                await on_event(payload)
        response_text = self.response_texts.pop(0) if self.response_texts else "Hello world!"
        return {"response": response_text}

    def cancel(self) -> None:
        self.cancelled = True

    async def suggest_followups(self, history: Sequence[Mapping[str, str]], *, max_suggestions: int = 4) -> list[str]:
        payload = {"history": list(history), "max": max_suggestions}
        self.suggestion_calls.append(payload)
        if self.suggestion_responses:
            return self.suggestion_responses.pop(0)
        return [f"Suggestion {len(self.suggestion_calls)}"]

    def register_tool(
        self,
        name: str,
        impl: Any,
        *,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
        strict: bool | None = None,
    ) -> None:
        self.registered_tools[name] = {
            "impl": impl,
            "description": description,
            "parameters": parameters,
            "strict": True if strict is None else bool(strict),
        }

    def available_tools(self) -> tuple[str, ...]:
        return tuple(self.registered_tools.keys())

    def update_client(self, client: Any) -> None:
        self.client = client
        self.updated_clients.append(client)


def _begin_test_review(window: MainWindow, prompt: str = "Revise the document") -> None:
    window.chat_panel.set_composer_text(prompt)
    snapshot = window.chat_panel.capture_state()
    window._review_controller.begin_pending_turn_review(
        prompt=prompt,
        prompt_metadata={},
        chat_snapshot=snapshot,
    )


def _apply_fake_ai_edit(
    window: MainWindow,
    tab: Any,
    new_text: str,
    *,
    diff_label: str,
) -> None:
    before_state = deepcopy(tab.document())
    directive = EditDirective(action="replace", target_range=(0, len(before_state.text)), content=new_text)
    tab.editor.set_text(new_text)
    window._handle_edit_applied(directive, before_state, diff_label)


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
    assert controller.history_payloads[-1] == []


def test_handle_chat_request_uses_existing_chat_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _StubAIController()
    window = _make_window(controller)
    sentinel = window.chat_panel.capture_state()

    calls = {"consumed": 0}

    def _consume() -> ChatTurnSnapshot:
        calls["consumed"] += 1
        return sentinel

    def _fail_capture() -> ChatTurnSnapshot:
        raise AssertionError("capture_state should not be called when snapshot exists")

    monkeypatch.setattr(window._chat_panel, "consume_turn_snapshot", _consume)
    monkeypatch.setattr(window._chat_panel, "capture_state", _fail_capture)

    window._handle_chat_request("Summarize the document", {})

    assert calls["consumed"] == 1
    assert window._pending_turn_snapshot is sentinel


def test_handle_chat_request_captures_snapshot_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _StubAIController()
    window = _make_window(controller)
    sentinel = window.chat_panel.capture_state()

    records = {"consumed": 0, "captured": 0}

    def _consume() -> ChatTurnSnapshot | None:
        records["consumed"] += 1
        return None

    def _capture() -> ChatTurnSnapshot:
        records["captured"] += 1
        return sentinel

    monkeypatch.setattr(window._chat_panel, "consume_turn_snapshot", _consume)
    monkeypatch.setattr(window._chat_panel, "capture_state", _capture)

    window._handle_chat_request("Summarize the document", {})

    assert records["consumed"] == 1
    assert records["captured"] == 1
    assert window._pending_turn_snapshot is sentinel


def test_manual_outline_command_uses_outline_tool():
    window = _make_window(settings=Settings(phase3_outline_tools=True))
    document = window.editor_widget.to_document()

    class _StubOutlineTool:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def run(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            return {
                "status": "ok",
                "document_id": document.document_id,
                "nodes": [
                    {
                        "level": 1,
                        "text": "Intro",
                        "pointer_id": f"outline:{document.document_id}:intro",
                        "children": [],
                    }
                ],
            }

    stub_tool = _StubOutlineTool()
    window._outline_tool = stub_tool  # type: ignore[assignment]

    panel = window.chat_panel
    panel.set_composer_text("/outline")
    panel.send_prompt()

    assert stub_tool.calls == [{}]
    response = panel.history()[-1].content
    assert "Document outline" in response
    assert "Intro" in response


def test_manual_outline_command_disabled_without_flag():
    window = _make_window(settings=Settings(phase3_outline_tools=False))
    panel = window.chat_panel
    panel.set_composer_text("/outline")
    panel.send_prompt()

    response = panel.history()[-1].content.lower()
    assert "disabled" in response
    assert window.last_status_message == "Outline disabled"


def test_manual_find_sections_command_uses_tool():
    window = _make_window(settings=Settings(phase3_outline_tools=True))
    document = window.editor_widget.to_document()

    class _StubFindTool:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        def run(self, **kwargs: Any) -> dict[str, Any]:
            self.calls.append(kwargs)
            query = kwargs.get("query", "")
            return {
                "status": "ok",
                "document_id": document.document_id,
                "query": query,
                "strategy": "fallback",
                "pointers": [
                    {
                        "pointer_id": f"chunk:{document.document_id}:1",
                        "score": 0.9,
                        "preview": "Example paragraph about introductions.",
                    }
                ],
            }

    stub_tool = _StubFindTool()
    window._find_sections_tool = stub_tool  # type: ignore[assignment]

    panel = window.chat_panel
    panel.set_composer_text("/find introduction")
    panel.send_prompt()

    assert stub_tool.calls and stub_tool.calls[0]["query"] == "introduction"
    response = panel.history()[-1].content
    assert "Find sections" in response
    assert "introduction" in response.lower()
    assert "Example paragraph" in response


def test_manual_find_sections_command_disabled_without_flag():
    window = _make_window(settings=Settings(phase3_outline_tools=False))
    panel = window.chat_panel
    panel.set_composer_text("/find intro")
    panel.send_prompt()

    response = panel.history()[-1].content.lower()
    assert "disabled" in response
    assert window.last_status_message == "Retrieval disabled"


def test_manual_command_parse_error_skips_ai_controller():
    controller = _StubAIController()
    window = _make_window(controller)
    panel = window.chat_panel
    panel.set_composer_text("/find   ")
    panel.send_prompt()

    assert controller.prompts == []
    assert "requires a query" in panel.history()[-1].content.lower()


def test_chat_prompt_includes_prior_history_for_ai_requests():
    controller = _StubAIController()
    window = _make_window(controller)
    panel = window.chat_panel
    panel.append_user_message("Earlier question")
    panel.append_ai_message(ChatMessage(role="assistant", content="Earlier reply"))

    panel.set_composer_text("New request")
    panel.send_prompt()

    history_payload = controller.history_payloads[-1]
    assert history_payload == [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier reply"},
    ]


def test_stream_done_event_does_not_duplicate_text():
    controller = _StubAIController()
    controller.stream_scripts.append(
        [
            SimpleNamespace(
                type="content.delta",
                content="Ooh, I'd love to help you write a story! Let me take a peek… ",
            ),
            SimpleNamespace(
                type="content.delta",
                content="Ooh, I see we have a fresh, blank canvas to work with! ",
            ),
            SimpleNamespace(
                type="content.done",
                content=(
                    "Ooh, I'd love to help you write a story! Let me take a peek… "
                    "Ooh, I see we have a fresh, blank canvas to work with! "
                ),
            ),
        ]
    )
    controller.response_texts.append(
        "Ooh, I'd love to help you write a story! Let me take a peek… "
        "Ooh, I see we have a fresh, blank canvas to work with! "
    )

    window = _make_window(controller)
    panel = window.chat_panel
    panel.set_composer_text("Tell me a story")
    panel.send_prompt()

    assistant_messages = [msg for msg in panel.history() if msg.role == "assistant"]
    assert assistant_messages
    final_text = assistant_messages[-1].content
    assert final_text.count("Ooh, I'd love to help you write a story!") == 1
    assert final_text.count("Ooh, I see we have a fresh, blank canvas") == 1


def test_tool_call_stream_events_coalesce_into_single_trace():
    controller = _StubAIController()
    window = _make_window(controller)
    call_id = "tool-123"
    arguments_text = '{"action":"replace","content":"Better text"}'
    result_text = "Applied edit"

    events = [
        SimpleNamespace(
            type="tool_calls.function.arguments.delta",
            tool_name="document_edit",
            tool_index=0,
            tool_call_id=call_id,
            arguments_delta='{"action":"replace"',
        ),
        SimpleNamespace(
            type="tool_calls.function.arguments.delta",
            tool_name="document_edit",
            tool_index=0,
            tool_call_id=call_id,
            arguments_delta=',"content":"Better text"}',
        ),
        SimpleNamespace(
            type="tool_calls.function.arguments.done",
            tool_name="document_edit",
            tool_index=0,
            tool_call_id=call_id,
            tool_arguments=arguments_text,
        ),
        SimpleNamespace(
            type="tool_calls.result",
            tool_name="document_edit",
            tool_index=0,
            tool_call_id=call_id,
            content=result_text,
        ),
    ]

    for payload in events:
        window._process_stream_event(payload)  # type: ignore[arg-type]

    traces = getattr(window.chat_panel, "_tool_traces")
    assert len(traces) == 1
    trace = traces[0]
    assert trace.metadata["raw_input"] == arguments_text
    assert trace.metadata["raw_output"] == result_text
    assert trace.output_summary == result_text
    assert window._pending_tool_traces == {}


def test_tool_trace_completes_when_stream_events_lack_ids():
    controller = _StubAIController()
    window = _make_window(controller)

    events = [
        SimpleNamespace(
            type="tool_calls.function.arguments.delta",
            tool_name="document_edit",
            tool_index=0,
            arguments_delta='{"action":"patch"',
        ),
        SimpleNamespace(
            type="tool_calls.function.arguments.done",
            tool_name="document_edit",
            tool_index=0,
            tool_arguments='{"action":"patch","diff":"---"}',
        ),
        SimpleNamespace(
            type="tool_calls.result",
            tool_name="document_edit",
            tool_index=0,
            tool_call_id="document_edit:0",
            content="Applied patch",
        ),
    ]

    for payload in events:
        window._process_stream_event(payload)  # type: ignore[arg-type]

    traces = getattr(window.chat_panel, "_tool_traces")
    assert traces
    assert traces[-1].output_summary == "Applied patch"
    assert window._pending_tool_traces == {}


def test_default_ai_tools_register_when_controller_available():
    controller = _StubAIController()
    window = _make_window(controller)
    del window  # window retains references to editor/bridge but is unused in assertions

    tool_names = set(controller.registered_tools)
    assert {
        "document_snapshot",
        "document_edit",
        "document_apply_patch",
        "validate_snippet",
        "list_tabs",
    }.issubset(tool_names)

    assert controller.registered_tools["document_snapshot"]["strict"] is True

    snapshot_tool = controller.registered_tools["document_snapshot"]["impl"]
    snapshot = snapshot_tool.run()
    assert isinstance(snapshot, dict)
    assert "length" in snapshot

    validator = controller.registered_tools["validate_snippet"]["impl"]
    outcome = validator("foo: bar", "yaml")
    assert hasattr(outcome, "ok")

    tab_listing = controller.registered_tools["list_tabs"]["impl"].run()
    assert isinstance(tab_listing, dict)
    assert "tabs" in tab_listing and "active_tab_id" in tab_listing


def test_close_event_cancels_background_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    window = _make_window()
    calls: list[str] = []
    monkeypatch.setattr(window, "_cancel_active_ai_turn", lambda: calls.append("ai"))
    monkeypatch.setattr(window, "_cancel_dynamic_suggestions", lambda: calls.append("suggestions"))
    monkeypatch.setattr(window, "_clear_suggestion_cache", lambda: calls.append("cache"))

    accepted = {"value": False}

    def _accept() -> None:
        accepted["value"] = True

    event = SimpleNamespace(accept=_accept)

    window.closeEvent(event)  # type: ignore[arg-type]

    assert calls == ["ai", "suggestions", "cache"]
    assert accepted["value"] is True
