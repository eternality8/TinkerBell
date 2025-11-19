"""Bridge tests."""

from __future__ import annotations

import difflib

import pytest

from tinkerbell.ai.memory.cache_bus import (
    DocumentCacheBus,
    DocumentCacheEvent,
    DocumentChangedEvent,
    DocumentClosedEvent,
)
from tinkerbell.chat.message_model import EditDirective
from tinkerbell.editor.document_model import DocumentState, SelectionRange
from tinkerbell.editor.patches import PatchResult
from tinkerbell.services import bridge as bridge_module
from tinkerbell.services.bridge import DocumentBridge, DocumentVersionMismatchError


class RecordingEditor:
    def __init__(self) -> None:
        self.state = DocumentState(text="hello world")
        self.applied: list[EditDirective] = []

    def load_document(self, document: DocumentState) -> None:
        self.state = document

    def to_document(self) -> DocumentState:
        return self.state

    def apply_ai_edit(self, directive: EditDirective) -> DocumentState:
        self.applied.append(directive)
        start, end = directive.target_range
        text = self.state.text
        if directive.action == "insert":
            text = text[:start] + directive.content + text[start:]
        elif directive.action == "replace":
            text = text[:start] + directive.content + text[end:]
        elif directive.action == "annotate":
            annotation = f"\n[AI Note]: {directive.content.strip()}\n"
            text = text + annotation
        else:  # pragma: no cover - guard unexpected action
            raise ValueError(directive.action)

        self.state.update_text(text)
        return self.state

    def apply_patch_result(self, result: PatchResult) -> DocumentState:
        self.state.update_text(result.text)
        return self.state


def _make_diff(before: str, after: str, filename: str = "doc.txt") -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        lineterm="",
    )
    diff_text = "\n".join(diff)
    assert diff_text.strip(), "expected diff text"
    return diff_text


def test_generate_snapshot():
    editor = RecordingEditor()
    editor.state.selection = SelectionRange(0, 5)
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()
    assert snapshot["language"] == "markdown"
    assert "version" in snapshot
    assert snapshot["version_id"] >= 1
    assert snapshot["document_id"]
    assert snapshot["length"] == len(editor.state.text)
    assert snapshot["selection_text"] == editor.state.text[:5]


def test_queue_edit_applies_insert_and_tracks_diff():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)

    bridge.queue_edit(EditDirective(action="insert", target_range=(5, 5), content="!!!"))

    assert editor.state.text == "hello!!! world"
    assert bridge.last_diff_summary == "+3 chars"


def test_queue_edit_accepts_dict_and_defaults_to_selection():
    editor = RecordingEditor()
    editor.state.selection = SelectionRange(0, 5)
    bridge = DocumentBridge(editor=editor)

    bridge.queue_edit({"action": "replace", "content": "hi"})

    assert editor.state.text.startswith("hi")
    assert editor.applied[-1].target_range == (0, 5)


def test_queue_edit_rejects_invalid_payload():
    bridge = DocumentBridge(editor=RecordingEditor())

    with pytest.raises(ValueError):
        bridge.queue_edit({"content": "oops"})


def test_queue_edit_rejects_stale_snapshot_version():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    editor.state.update_text("something else")

    with pytest.raises(DocumentVersionMismatchError):
        bridge.queue_edit(
            {
                "action": "replace",
                "content": "patched",
                "target_range": (0, 5),
                "document_version": snapshot["version"],
            }
        )


def test_edit_listener_receives_diff_summary():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    events: list[tuple[str, str]] = []

    def _listener(directive: EditDirective, state: DocumentState, diff: str) -> None:
        events.append((directive.action, diff))

    bridge.add_edit_listener(_listener)
    bridge.queue_edit(EditDirective(action="insert", target_range=(5, 5), content="!!!"))

    assert events
    assert events[0][0] == "insert"
    assert events[0][1] == "+3 chars"


def test_last_edit_context_tracks_replacement_segments():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)

    bridge.queue_edit(EditDirective(action="replace", target_range=(0, 5), content="hi"))

    context = bridge.last_edit_context
    assert context is not None
    assert context.action == "replace"
    assert context.target_range == (0, 5)
    assert context.replaced_text == "hello"
    assert context.content == "hi"


def test_queue_edit_applies_patch_and_updates_metrics():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    diff = """--- a/doc.txt
+++ b/doc.txt
@@ -1 +1 @@
-hello world
+hello brave world
"""

    bridge.queue_edit({"action": "patch", "diff": diff, "document_version": snapshot["version"]})

    assert editor.state.text == "hello brave world"
    assert bridge.last_diff_summary and bridge.last_diff_summary.startswith("patch:")
    assert bridge.patch_metrics.total == 1
    context = bridge.last_edit_context
    assert context is not None and context.diff == diff


def test_queue_edit_patch_conflict_raises_and_tracks_failure():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    diff = (
        "--- a/doc.txt\n"
        "+++ b/doc.txt\n"
        "@@ -1 +1 @@\n"
        "-HELLO WORLD\n"
        "+HELLO BRAVE WORLD\n"
    )

    with pytest.raises(RuntimeError):
        bridge.queue_edit({"action": "patch", "diff": diff, "document_version": snapshot["version"]})

    assert editor.state.text == "hello world"
    assert bridge.patch_metrics.conflicts == 1


def test_patch_apply_emits_success_telemetry(monkeypatch: pytest.MonkeyPatch):
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    captured: list[tuple[str, dict | None]] = []

    def _emit(name: str, payload: dict | None = None) -> None:
        captured.append((name, payload))

    monkeypatch.setattr(bridge_module, "telemetry_emit", _emit)

    diff = _make_diff(editor.state.text, "hello brave world")
    bridge.queue_edit({"action": "patch", "diff": diff, "document_version": snapshot["version"]})

    event = next((payload for name, payload in captured if name == "patch.apply"), None)
    assert event is not None
    assert event["status"] == "success"
    assert event["range_count"] >= 1


def test_patch_apply_emits_conflict_telemetry(monkeypatch: pytest.MonkeyPatch):
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    captured: list[tuple[str, dict | None]] = []

    def _emit(name: str, payload: dict | None = None) -> None:
        captured.append((name, payload))

    monkeypatch.setattr(bridge_module, "telemetry_emit", _emit)

    diff = _make_diff("HELLO WORLD", "HELLO BRAVE WORLD")
    with pytest.raises(RuntimeError):
        bridge.queue_edit({"action": "patch", "diff": diff, "document_version": snapshot["version"]})

    event = next((payload for name, payload in captured if name == "patch.apply"), None)
    assert event is not None
    assert event["status"] == "conflict"
    assert event.get("reason")


def test_patch_apply_emits_stale_telemetry(monkeypatch: pytest.MonkeyPatch):
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    captured: list[tuple[str, dict | None]] = []

    def _emit(name: str, payload: dict | None = None) -> None:
        captured.append((name, payload))

    monkeypatch.setattr(bridge_module, "telemetry_emit", _emit)

    editor.state.update_text("HELLO WORLD")

    with pytest.raises(DocumentVersionMismatchError):
        bridge.queue_edit({"action": "patch", "diff": _make_diff("hello", "hola"), "document_version": snapshot["version"]})

    event = next((payload for name, payload in captured if name == "patch.apply"), None)
    assert event is not None
    assert event["status"] == "stale"


def test_queue_edit_applies_streamed_ranges():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    bridge.queue_edit(
        {
            "action": "patch",
            "ranges": [
                {
                    "start": 0,
                    "end": 5,
                    "replacement": "HELLO",
                    "match_text": "hello",
                },
            ],
            "document_version": snapshot["version"],
            "content_hash": snapshot["content_hash"],
        }
    )

    assert editor.state.text.startswith("HELLO")
    assert bridge.patch_metrics.total == 1


def test_queue_edit_range_patch_detects_mismatch():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()
    editor.state.update_text("HOLA world")

    with pytest.raises(RuntimeError):
        bridge.queue_edit(
            {
                "action": "patch",
                "ranges": [
                    {
                        "start": 0,
                        "end": 5,
                        "replacement": "HELLO",
                        "match_text": "hello",
                    }
                ],
                "document_version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
            }
        )

    assert editor.state.text.startswith("HOLA")
    assert bridge.patch_metrics.conflicts == 1


def test_queue_edit_applies_multiple_patch_directives():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    diff_one = _make_diff(editor.state.text, "hello brave world")
    bridge.queue_edit({"action": "patch", "diff": diff_one, "document_version": snapshot["version"]})

    assert editor.state.text == "hello brave world"
    assert bridge.patch_metrics.total == 1

    snapshot_two = bridge.generate_snapshot()
    diff_two = _make_diff(editor.state.text, "HELLO brave world!!!")
    bridge.queue_edit({"action": "patch", "diff": diff_two, "document_version": snapshot_two["version"]})

    assert editor.state.text == "HELLO brave world!!!"
    assert bridge.patch_metrics.total == 2
    summary = bridge.last_diff_summary
    assert summary is not None and summary.startswith("patch:")


def test_queue_edit_patch_handles_snippet_line_numbers():
    editor = RecordingEditor()
    editor.state.update_text("intro\nalpha\nbeta\ngamma\n")
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    diff = """--- a/snippet
+++ b/snippet
@@ -1,2 +1,2 @@
 alpha
-beta
+BETA
"""

    bridge.queue_edit({"action": "patch", "diff": diff, "document_version": snapshot["version"]})

    assert "BETA" in editor.state.text
    assert bridge.patch_metrics.total == 1


def test_bridge_publishes_document_changed_events() -> None:
    editor = RecordingEditor()
    bus = DocumentCacheBus()
    events: list[DocumentCacheEvent] = []
    bus.subscribe(DocumentChangedEvent, lambda event: events.append(event))

    bridge = DocumentBridge(editor=editor, cache_bus=bus)
    bridge.queue_edit(EditDirective(action="insert", target_range=(5, 5), content="!!!"))

    assert events
    changed = events[-1]
    assert isinstance(changed, DocumentChangedEvent)
    assert changed.document_id == editor.state.document_id
    assert changed.version_id == editor.state.version_id
    assert changed.edited_ranges[0] == (5, 5)


def test_bridge_notifies_document_closed() -> None:
    editor = RecordingEditor()
    bus = DocumentCacheBus()
    closed_events: list[DocumentClosedEvent] = []

    def on_closed(event: DocumentCacheEvent) -> None:
        assert isinstance(event, DocumentClosedEvent)
        closed_events.append(event)

    bus.subscribe(DocumentClosedEvent, on_closed)

    bridge = DocumentBridge(editor=editor, cache_bus=bus)
    bridge.notify_document_closed(reason="tab-closed")

    assert closed_events
    event = closed_events[-1]
    assert event.document_id == editor.state.document_id
    assert event.reason == "tab-closed"
