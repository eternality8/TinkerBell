"""Bridge tests."""

from __future__ import annotations

import difflib
from copy import deepcopy
from typing import Any, Mapping

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
from tinkerbell.editor.post_edit_inspector import InspectionResult
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

    def apply_ai_edit(self, directive: EditDirective, *, preserve_selection: bool = False) -> DocumentState:
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

    def apply_patch_result(self, result: PatchResult, selection_hint=None, *, preserve_selection: bool = False) -> DocumentState:
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


def test_queue_edit_rejects_invalid_payload():
    bridge = DocumentBridge(editor=RecordingEditor())

    with pytest.raises(ValueError):
        bridge.queue_edit({"content": "oops"})


def test_queue_edit_rejects_inline_directives():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)

    with pytest.raises(ValueError, match="patch directives"):
        bridge.queue_edit(EditDirective(action="insert", target_range=(5, 5), content="!!!"))


def test_range_patch_expands_to_word_boundaries():
    editor = RecordingEditor()
    editor.state.update_text("alpha bravo")
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    payload = {
        "action": "patch",
        "document_version": snapshot["version"],
        "content_hash": snapshot["content_hash"],
        "ranges": [
            {"start": 1, "end": 4, "replacement": "ALPHA", "match_text": "lph"},
        ],
    }

    bridge.queue_edit(payload)

    assert editor.state.text == "ALPHA bravo"
    context = bridge.last_edit_context
    assert context is not None
    assert context.target_range.to_tuple() == (0, 5)


def test_range_patch_expands_to_paragraph_boundaries():
    text = "Alpha one\nAlpha two\n\nSecond block\n"
    editor = RecordingEditor()
    editor.state.update_text(text)
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    start = 2
    end = 9
    payload = {
        "action": "patch",
        "document_version": snapshot["version"],
        "content_hash": snapshot["content_hash"],
        "ranges": [
            {
                "start": start,
                "end": end,
                "replacement": "Updated first paragraph.\n",
                "match_text": text[start:end],
            }
        ],
    }

    bridge.queue_edit(payload)

    expected = "Updated first paragraph.\n\nSecond block\n"
    assert editor.state.text == expected
    context = bridge.last_edit_context
    assert context is not None
    blank_line_index = text.find("\n\n")
    assert context.target_range.to_tuple() == (0, blank_line_index)


def test_patch_failure_metadata_includes_target_range():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    payload = {
        "action": "patch",
        "document_version": snapshot["version"],
        "content_hash": snapshot["content_hash"],
        "ranges": [
            {
                "start": 0,
                "end": 5,
                "replacement": "HELLO",
                "match_text": "hello",
            }
        ],
    }

    editor.state.update_text("HELLO world!!!")

    with pytest.raises(DocumentVersionMismatchError):
        bridge.queue_edit(payload)

    metadata = bridge.last_failure_metadata
    assert metadata is not None
    assert metadata["target_range"] == {"start": 0, "end": 5}


def test_queue_edit_rejects_stale_snapshot_version():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    editor.state.update_text("something else")

    with pytest.raises(DocumentVersionMismatchError):
        bridge.queue_edit(
            {
                "action": "patch",
                "diff": _make_diff("hello world", "patched world"),
                "document_version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
            }
        )


def test_queue_edit_rejects_content_hash_mismatch():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    payload = {
        "action": "patch",
        "diff": _make_diff(editor.state.text, "HELLO world"),
        "document_version": snapshot["version"],
        "content_hash": "bogus-hash",
    }

    with pytest.raises(DocumentVersionMismatchError) as exc:
        bridge.queue_edit(payload)

    assert "content_hash" in str(exc.value)


def test_edit_listener_receives_diff_summary():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    events: list[tuple[str, str]] = []

    def _listener(directive: EditDirective, state: DocumentState, diff: str) -> None:
        events.append((directive.action, diff))

    bridge.add_edit_listener(_listener)
    snapshot = bridge.generate_snapshot()
    diff = _make_diff(editor.state.text, "hello!!! world")
    bridge.queue_edit(
        {
            "action": "patch",
            "diff": diff,
            "document_version": snapshot["version"],
            "content_hash": snapshot["content_hash"],
        }
    )

    assert events
    assert events[0][0] == "patch"
    assert events[0][1].startswith("patch:")


def test_last_edit_context_tracks_patch_metadata():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()
    diff = _make_diff(editor.state.text, "HELLO world")

    bridge.queue_edit(
        {
            "action": "patch",
            "diff": diff,
            "document_version": snapshot["version"],
            "content_hash": snapshot["content_hash"],
        }
    )

    context = bridge.last_edit_context
    assert context is not None
    assert context.action == "patch"
    assert context.diff == diff
    assert context.spans


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

    bridge.queue_edit(
        {
            "action": "patch",
            "diff": diff,
            "document_version": snapshot["version"],
            "content_hash": snapshot["content_hash"],
        }
    )

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
        bridge.queue_edit(
            {
                "action": "patch",
                "diff": diff,
                "document_version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
            }
        )

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
    bridge.queue_edit(
        {
            "action": "patch",
            "diff": diff,
            "document_version": snapshot["version"],
            "content_hash": snapshot["content_hash"],
        }
    )

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
        bridge.queue_edit(
            {
                "action": "patch",
                "diff": diff,
                "document_version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
            }
        )

    event = next((payload for name, payload in captured if name == "patch.apply"), None)
    assert event is not None
    assert event["status"] == "conflict"
    assert event.get("reason")


def test_edit_rejected_event_emitted(monkeypatch: pytest.MonkeyPatch):
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    captured: list[tuple[str, dict | None]] = []

    def _emit(name: str, payload: dict | None = None) -> None:
        captured.append((name, payload))

    monkeypatch.setattr(bridge_module, "telemetry_emit", _emit)

    diff = _make_diff("HELLO WORLD", "HELLO BRAVE WORLD")
    with pytest.raises(RuntimeError):
        bridge.queue_edit(
            {
                "action": "patch",
                "diff": diff,
                "document_version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
            }
        )

    edit_event = next((payload for name, payload in captured if name == "edit_rejected"), None)
    assert edit_event is not None
    assert edit_event["document_id"] == snapshot["document_id"]
    assert edit_event["cause"] == bridge.CAUSE_HASH_MISMATCH


def test_patch_failure_provides_metadata():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    captured: list[Mapping[str, Any] | None] = []

    def _listener(_directive: EditDirective, _message: str, metadata: Mapping[str, Any] | None = None) -> None:
        captured.append(metadata)

    bridge.add_failure_listener(_listener)

    diff = _make_diff("HELLO WORLD", "HELLO BRAVE WORLD")
    with pytest.raises(RuntimeError):
        bridge.queue_edit(
            {
                "action": "patch",
                "diff": diff,
                "document_version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
            }
        )

    assert captured, "expected failure metadata"
    metadata = captured[-1]
    assert metadata is not None
    assert metadata["status"] == "conflict"
    assert metadata["cause"] == DocumentBridge.CAUSE_HASH_MISMATCH
    assert metadata["document_id"] == snapshot["document_id"]
    assert metadata["version_id"] == snapshot["version_id"]
    assert metadata.get("tab_id") is None
    assert bridge.last_failure_metadata == metadata


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
        bridge.queue_edit(
            {
                "action": "patch",
                "diff": _make_diff("hello", "hola"),
                "document_version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
            }
        )

    event = next((payload for name, payload in captured if name == "patch.apply"), None)
    assert event is not None
    assert event["status"] == "stale"


def test_hash_mismatch_event_emitted_on_rejection(monkeypatch: pytest.MonkeyPatch):
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()
    captured: list[tuple[str, dict | None]] = []

    def _emit(name: str, payload: dict | None = None) -> None:
        captured.append((name, payload))

    monkeypatch.setattr(bridge_module, "telemetry_emit", _emit)
    editor.state.update_text("HELLO WORLD")

    with pytest.raises(DocumentVersionMismatchError):
        bridge.queue_edit(
            {
                "action": "patch",
                "diff": _make_diff("alpha", "beta"),
                "document_version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
            }
        )

    event = next((payload for name, payload in captured if name == "hash_mismatch"), None)
    assert event is not None
    assert event["stage"] == "bridge"
    assert event["cause"] == bridge.CAUSE_HASH_MISMATCH


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


def test_range_patch_chunk_hash_mismatch_rejected():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()
    manifest = snapshot.get("chunk_manifest") or {}
    chunks = manifest.get("chunks") or []
    assert chunks, "expected chunk manifest entries for snapshot"
    first_chunk = chunks[0]
    chunk_id = first_chunk.get("id")
    assert chunk_id, "expected chunk id in manifest"
    tampered_hash = "deadbeef"

    with pytest.raises(DocumentVersionMismatchError) as exc:
        bridge.queue_edit(
            {
                "action": "patch",
                "ranges": [
                    {
                        "start": 0,
                        "end": 5,
                        "replacement": "HELLO",
                        "match_text": "hello",
                        "chunk_id": chunk_id,
                        "chunk_hash": tampered_hash,
                    }
                ],
                "document_version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
            }
        )

    assert exc.value.cause == bridge.CAUSE_CHUNK_HASH_MISMATCH
    assert bridge.patch_metrics.conflicts == 1


def test_post_edit_inspector_rejection_restores_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    bridge.configure_safe_editing(enabled=True)
    snapshot = bridge.generate_snapshot()
    original_text = editor.state.text

    captured: list[tuple[str, dict | None]] = []

    def _emit(name: str, payload: dict | None = None) -> None:
        captured.append((name, payload))

    monkeypatch.setattr(bridge_module, "telemetry_emit", _emit)

    diagnostics_payload = {"duplicate": {"normalized": "hello", "count": 2}}

    def _fake_inspect(**_: Any) -> InspectionResult:
        return InspectionResult(
            ok=False,
            reason="duplicate_paragraphs",
            detail="Paragraph repeated",
            diagnostics=diagnostics_payload,
        )

    monkeypatch.setattr(bridge._post_edit_inspector, "inspect", _fake_inspect)

    diff = _make_diff(original_text, "HELLO brave world")
    with pytest.raises(DocumentVersionMismatchError) as exc:
        bridge.queue_edit(
            {
                "action": "patch",
                "diff": diff,
                "document_version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
            }
        )

    assert exc.value.cause == bridge.CAUSE_INSPECTOR_FAILURE
    assert exc.value.details is not None
    assert exc.value.details.get("code") == "auto_revert"
    assert exc.value.details.get("reason") == "duplicate_paragraphs"
    assert editor.state.text == original_text, "inspector rejection should roll back the edit"

    metadata = bridge.last_failure_metadata
    assert metadata is not None
    assert metadata["status"] == "rejected"
    assert metadata["cause"] == bridge.CAUSE_INSPECTOR_FAILURE

    edit_event = next((payload for name, payload in captured if name == "edit_rejected"), None)
    assert edit_event is not None
    assert edit_event["cause"] == bridge.CAUSE_INSPECTOR_FAILURE

    auto_event = next((payload for name, payload in captured if name == "auto_revert"), None)
    assert auto_event is not None
    assert auto_event["reason"] == "duplicate_paragraphs"
    assert auto_event.get("diff_summary")

    duplicate_event = next((payload for name, payload in captured if name == "duplicate_detected"), None)
    assert duplicate_event is not None
    duplicate_details = duplicate_event.get("duplicate")
    assert isinstance(duplicate_details, dict)
    assert duplicate_details.get("count") == 2


def test_post_edit_inspector_auto_reverts_duplicate_inserts() -> None:
    class _InspectorEditor(RecordingEditor):
        def __init__(self) -> None:
            super().__init__()
            self.state.update_text("Alpha one.\n\nBeta two.\n\n")

        def to_document(self) -> DocumentState:  # type: ignore[override]
            return deepcopy(self.state)

    editor = _InspectorEditor()
    bridge = DocumentBridge(editor=editor)
    bridge.configure_safe_editing(enabled=True)
    before = editor.state.text
    snapshot = bridge.generate_snapshot()
    after = before + "Beta two.\n\nBeta two.\n\n"
    diff = _make_diff(before, after)

    with pytest.raises(DocumentVersionMismatchError) as exc:
        bridge.queue_edit(
            {
                "action": "patch",
                "diff": diff,
                "document_version": snapshot["version"],
                "content_hash": snapshot["content_hash"],
            }
        )

    assert exc.value.cause == bridge.CAUSE_INSPECTOR_FAILURE
    assert exc.value.details is not None
    assert exc.value.details.get("code") == "auto_revert"
    assert editor.state.text == before
    metadata = bridge.last_failure_metadata
    assert metadata is not None
    assert metadata["cause"] == bridge.CAUSE_INSPECTOR_FAILURE


def test_queue_edit_applies_multiple_patch_directives():
    editor = RecordingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()

    diff_one = _make_diff(editor.state.text, "hello brave world")
    bridge.queue_edit(
        {
            "action": "patch",
            "diff": diff_one,
            "document_version": snapshot["version"],
            "content_hash": snapshot["content_hash"],
        }
    )

    assert editor.state.text == "hello brave world"
    assert bridge.patch_metrics.total == 1

    snapshot_two = bridge.generate_snapshot()
    diff_two = _make_diff(editor.state.text, "HELLO brave world!!!")
    bridge.queue_edit(
        {
            "action": "patch",
            "diff": diff_two,
            "document_version": snapshot_two["version"],
            "content_hash": snapshot_two["content_hash"],
        }
    )

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

    bridge.queue_edit(
        {
            "action": "patch",
            "diff": diff,
            "document_version": snapshot["version"],
            "content_hash": snapshot["content_hash"],
        }
    )

    assert "BETA" in editor.state.text
    assert bridge.patch_metrics.total == 1


def test_bridge_publishes_document_changed_events() -> None:
    editor = RecordingEditor()
    bus = DocumentCacheBus()
    events: list[DocumentCacheEvent] = []
    bus.subscribe(DocumentChangedEvent, lambda event: events.append(event))

    bridge = DocumentBridge(editor=editor, cache_bus=bus)
    snapshot = bridge.generate_snapshot()
    diff = _make_diff(editor.state.text, "hello!!! world")
    bridge.queue_edit(
        {
            "action": "patch",
            "diff": diff,
            "document_version": snapshot["version"],
            "content_hash": snapshot["content_hash"],
        }
    )

    assert events
    changed = events[-1]
    assert isinstance(changed, DocumentChangedEvent)
    assert changed.document_id == editor.state.document_id
    assert changed.version_id == editor.state.version_id
    assert changed.edited_ranges[0][0] == 5
    assert changed.edited_ranges[0][1] >= 5


def test_bridge_applies_patches_without_moving_selection() -> None:
    class _SelectionTrackingEditor(RecordingEditor):
        def __init__(self) -> None:
            super().__init__()
            self.preserve_calls: list[bool] = []

        def apply_patch_result(self, result: PatchResult, selection_hint=None, *, preserve_selection: bool = False) -> DocumentState:  # type: ignore[override]
            self.preserve_calls.append(preserve_selection)
            return super().apply_patch_result(result, selection_hint, preserve_selection=preserve_selection)

    editor = _SelectionTrackingEditor()
    bridge = DocumentBridge(editor=editor)
    snapshot = bridge.generate_snapshot()
    diff = _make_diff(editor.state.text, "hello brave world")

    bridge.queue_edit(
        {
            "action": "patch",
            "diff": diff,
            "document_version": snapshot["version"],
            "content_hash": snapshot["content_hash"],
        }
    )

    assert editor.preserve_calls
    assert editor.preserve_calls[-1] is True


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
