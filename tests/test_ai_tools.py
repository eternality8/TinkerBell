"""Unit tests for AI tool modules."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from typing import Any, Mapping

import pytest

from tinkerbell.ai.memory.chunk_index import ChunkIndex
from tinkerbell.ai.analysis.models import AnalysisAdvice
from tinkerbell.ai.tools import document_apply_patch as document_apply_patch_module
from tinkerbell.ai.tools import document_edit as document_edit_module
from tinkerbell.ai.tools.registry import (
    ToolRegistrationError,
    ToolRegistryContext,
    register_default_tools,
)
from tinkerbell.ai.tools.diff_builder import DiffBuilderTool
from tinkerbell.ai.tools.document_apply_patch import DocumentApplyPatchTool
from tinkerbell.ai.tools.document_edit import DocumentEditTool
from tinkerbell.ai.tools.document_snapshot import DocumentSnapshotTool
from tinkerbell.ai.tools.list_tabs import ListTabsTool
from tinkerbell.ai.tools.search_replace import SearchReplaceTool
from tinkerbell.ai.tools.tool_usage_advisor import ToolUsageAdvisorTool
from tinkerbell.ai.tools.validation import validate_snippet
from tinkerbell.chat.commands import DIRECTIVE_SCHEMA
from tinkerbell.documents.range_normalizer import compose_normalized_replacement, normalize_text_range
from tinkerbell.editor.syntax import yaml_json
from tinkerbell.services.bridge import DocumentVersionMismatchError


class _EditBridgeStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.last_diff_summary: str | None = None
        self.last_snapshot_version: str | None = None
        self.snapshot = {"text": "", "selection": (0, 0), "version": "digest-base", "path": "doc.txt"}
        self.queue_tab_ids: list[str | None] = []
        self.snapshot_requests: list[dict[str, Any]] = []
        self.tabs: list[dict[str, Any]] = [
            {"tab_id": "tab-a", "title": "README.md", "path": "C:/repo/README.md", "dirty": False},
            {"tab_id": "tab-b", "title": "Notes.md", "path": "C:/repo/notes.md", "dirty": True},
        ]
        self._active_tab_id = "tab-a"

    def queue_edit(self, directive, *, tab_id: str | None = None) -> None:  # type: ignore[override]
        self.queue_tab_ids.append(tab_id)
        self.calls.append(directive)

    def generate_snapshot(  # type: ignore[override]
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_open_documents: bool = False,
        window=None,
        chunk_profile: str | None = None,
        max_tokens: int | None = None,
        include_text: bool = True,
    ):
        self.snapshot_requests.append(
            {
                "delta_only": delta_only,
                "tab_id": tab_id,
                "include_open_documents": include_open_documents,
                "window": window,
                "chunk_profile": chunk_profile,
                "max_tokens": max_tokens,
                "include_text": include_text,
            }
        )
        assert delta_only is False
        snapshot = dict(self.snapshot)
        text = snapshot.get("text", "")
        doc_text = text if isinstance(text, str) else ""
        doc_length = len(doc_text)
        snapshot["length"] = doc_length
        if isinstance(window, Mapping):
            start = max(0, min(int(window.get("start", 0)), doc_length))
            end = max(start, min(int(window.get("end", doc_length)), doc_length))
            snapshot["window"] = {"start": start, "end": end}
            snapshot["text_range"] = {"start": start, "end": end}
            if include_text:
                snapshot["text"] = doc_text[start:end]
            else:
                snapshot["text"] = ""
        elif not include_text:
            snapshot["text"] = ""
        else:
            snapshot.setdefault("window", {"start": 0, "end": doc_length})
            snapshot.setdefault("text_range", {"start": 0, "end": doc_length})
        return snapshot

    def get_last_diff_summary(self, tab_id: str | None = None) -> str | None:  # noqa: ARG002 - unused
        return self.last_diff_summary

    def get_last_snapshot_version(self, tab_id: str | None = None) -> str | None:  # noqa: ARG002 - unused
        return self.last_snapshot_version

    def list_tabs(self) -> list[dict[str, Any]]:
        return list(self.tabs)

    def active_tab_id(self) -> str | None:
        return self._active_tab_id


class _SnapshotProviderStub:
    def __init__(self) -> None:
        self.snapshot = {"text": "Hello", "selection": (0, 5), "version": "base", "document_id": "doc-stub"}
        self.delta_only_calls: list[bool] = []
        self.last_diff_summary: str | None = "+1 char"
        self.last_snapshot_version: str | None = "digest-1"

    def generate_snapshot(  # type: ignore[override]
        self,
        *,
        delta_only: bool = False,
        tab_id: str | None = None,
        include_open_documents: bool = False,
        **_: Any,
    ):
        self.delta_only_calls.append(delta_only)
        return dict(self.snapshot)

    def get_last_diff_summary(self, tab_id: str | None = None) -> str | None:  # noqa: ARG002 - unused
        return self.last_diff_summary

    def get_last_snapshot_version(self, tab_id: str | None = None) -> str | None:  # noqa: ARG002 - unused
        return self.last_snapshot_version


class _SearchReplaceBridgeStub:
    def __init__(self, *, text: str, selection: tuple[int, int] = (0, 0), version: str = "base") -> None:
        self.snapshot = {
            "text": text,
            "selection": selection,
            "version": version,
            "content_hash": self._hash_text(text),
            "document_id": "search-doc",
        }
        self.queue_calls: list[Mapping[str, Any]] = []
        self.last_snapshot_version: str | None = version

    def generate_snapshot(self, *, delta_only: bool = False):  # type: ignore[override]
        assert delta_only is False
        return dict(self.snapshot)

    def queue_edit(self, directive: Mapping[str, Any]):  # type: ignore[override]
        self.queue_calls.append(directive)
        self.last_snapshot_version = "updated"

    @staticmethod
    def _hash_text(value: str) -> str:
        return hashlib.sha1(value.encode("utf-8")).hexdigest()


class _PatchBridgeStub(_EditBridgeStub):
    def __init__(self, *, text: str = "Hello world", selection: tuple[int, int] = (0, 0), version: str = "digest-0") -> None:
        super().__init__()
        self.snapshot = {"text": text, "selection": selection, "version": version, "path": "doc.md"}
        self.last_snapshot_version = version


class _LegacyBridgeStub(_EditBridgeStub):
    def queue_edit(self, directive) -> None:  # type: ignore[override]
        # Legacy bridges do not accept tab_id keyword arguments
        self.calls.append(directive)


class _ControllerStub:
    def __init__(self, *, fail_on: set[str] | None = None) -> None:
        self.fail_on = fail_on or set()
        self.registered: list[str] = []
        self._chunk_index = ChunkIndex()

    def register_tool(self, name: str, tool: Any, **_: Any) -> None:
        if name in self.fail_on:
            raise RuntimeError(f"boom: {name}")
        self.registered.append(name)

    @contextmanager
    def suspend_graph_rebuilds(self):
        yield

    def ensure_chunk_index(self) -> ChunkIndex:
        return self._chunk_index

    def get_chunking_config(self) -> dict[str, Any]:
        return {
            "default_profile": "auto",
            "overlap_chars": 256,
            "max_inline_tokens": 1_800,
            "iterator_limit": 4,
        }


def _registry_context(controller: _ControllerStub) -> ToolRegistryContext:
    bridge = _EditBridgeStub()
    return ToolRegistryContext(
        controller=controller,
        bridge=bridge,
        outline_digest_resolver=lambda _doc_id=None: None,
        directive_schema_provider=lambda: DIRECTIVE_SCHEMA,
    )


def test_document_edit_tool_accepts_json_payload_and_reports_status():
    bridge = _EditBridgeStub()
    bridge.last_diff_summary = "+2 chars"
    bridge.last_snapshot_version = "digest-123"
    bridge.snapshot["text"] = "hello"
    tool = DocumentEditTool(bridge=bridge)

    status = tool.run('{"action":"insert","content":"Hi","target_range":[0,0]}')

    assert bridge.calls and isinstance(bridge.calls[0], dict)
    payload = bridge.calls[0]
    assert payload["action"] == "patch"
    ranges = payload["ranges"]
    assert ranges[0]["replacement"] == "Hi"
    assert ranges[0]["start"] == 0
    assert "applied: +2 chars" in status
    assert "digest-123" in status


def test_document_edit_tool_accepts_keyword_arguments():
    bridge = _EditBridgeStub()
    bridge.last_diff_summary = "Δ0"
    bridge.snapshot["text"] = "hello world"
    tool = DocumentEditTool(bridge=bridge)

    status = tool.run(action="replace", content="Hello", target_range=(0, 5))

    assert bridge.calls and bridge.calls[0]["action"] == "patch"
    assert status.startswith("applied") or status.startswith("queued")


def test_document_edit_tool_requires_payload():
    bridge = _EditBridgeStub()
    tool = DocumentEditTool(bridge=bridge)

    with pytest.raises(ValueError):
        tool.run()

    with pytest.raises(ValueError):
        tool.run('{"action":"insert","content":"Hi"}', action="insert")


def test_document_edit_tool_replaces_paragraph_with_formatting():
    bridge = _EditBridgeStub()
    bridge.last_diff_summary = "+12 chars"
    bridge.last_snapshot_version = "digest-abc"
    bridge.snapshot["text"] = (
        "Existing intro paragraph that will be replaced in this test to verify formatting handling."
    )
    tool = DocumentEditTool(bridge=bridge)

    replacement = "New intro paragraph.\n\n- bullet α\n- bullet β ✨"
    status = tool.run(
        action="replace",
        content=replacement,
        target_range={"start": 10, "end": 64},
        rationale="Refresh intro with bullets",
    )

    assert bridge.calls and bridge.calls[0]["action"] == "patch"
    ranges = bridge.calls[0]["ranges"]
    base_text = bridge.snapshot["text"]
    normalized = normalize_text_range(base_text, 10, 64, replacement=replacement)
    expected_replacement = compose_normalized_replacement(
        base_text,
        normalized,
        replacement,
        original_start=10,
        original_end=64,
    )
    assert ranges[0]["start"] == normalized.start
    assert ranges[0]["end"] == normalized.end
    assert ranges[0]["match_text"] == normalized.slice_text
    assert ranges[0]["replacement"] == expected_replacement
    assert "applied" in status and "digest-abc" in status

def test_document_edit_tool_supports_document_scope_literal():
    bridge = _PatchBridgeStub(text="Alpha\nBeta\n", selection=(0, 0), version="digest-42")
    tool = DocumentEditTool(bridge=bridge)

    status = tool.run(action="replace", content="Gamma", target_range="document")

    assert bridge.calls
    payload = bridge.calls[0]
    assert payload["action"] == "patch"
    assert payload["document_version"] == "digest-42"
    ranges = payload["ranges"]
    assert ranges[0]["start"] == 0
    assert ranges[0]["end"] == len("Alpha\nBeta\n")
    assert ranges[0]["match_text"] == "Alpha\nBeta\n"
    assert ranges[0]["replacement"] == "Gamma"
    assert status.startswith("applied") or status.startswith("queued")

def test_document_edit_tool_rejects_unknown_range_literal():
    bridge = _PatchBridgeStub(text="Hello", selection=(0, 0))
    tool = DocumentEditTool(bridge=bridge)

    with pytest.raises(ValueError, match="target_range string must be 'document'"):
        tool.run(action="replace", content="World", target_range="page")


def test_document_edit_tool_handles_special_characters_in_sentence():
    bridge = _EditBridgeStub()
    bridge.snapshot["text"] = "Original sentence spanning multiple characters for testing."
    tool = DocumentEditTool(bridge=bridge)

    replacement = "“smart quotes” & emojis 🎉\nNext line"
    payload = json.dumps(
        {
            "action": "replace",
            "target_range": [25, 52],
            "content": replacement,
        }
    )

    tool.run(payload)

    assert bridge.calls
    recorded = bridge.calls[-1]
    assert recorded["action"] == "patch"
    ranges = recorded["ranges"]
    base_text = bridge.snapshot["text"]
    normalized = normalize_text_range(base_text, 25, 52, replacement=replacement)
    expected_replacement = compose_normalized_replacement(
        base_text,
        normalized,
        replacement,
        original_start=25,
        original_end=52,
    )
    assert ranges[0]["start"] == normalized.start
    assert ranges[0]["end"] == normalized.end
    assert ranges[0]["match_text"] == normalized.slice_text
    assert ranges[0]["replacement"] == expected_replacement


def test_document_edit_tool_routes_to_specific_tab():
    bridge = _EditBridgeStub()
    bridge.snapshot["text"] = "Hi"
    tool = DocumentEditTool(bridge=bridge)

    tool.run(action="insert", content="Hi", target_range=(0, 0), tab_id="tab-123")

    assert bridge.queue_tab_ids[-1] == "tab-123"


def test_document_edit_tool_resolves_tab_reference_in_argument():
    bridge = _EditBridgeStub()
    bridge.snapshot["text"] = "Hi"
    tool = DocumentEditTool(bridge=bridge)

    tool.run(action="insert", content="Hi", target_range=(0, 0), tab_id="Tab 2")

    assert bridge.queue_tab_ids[-1] == "tab-b"


def test_document_edit_tool_resolves_tab_reference_from_payload():
    bridge = _EditBridgeStub()
    bridge.snapshot["text"] = "Hi"
    tool = DocumentEditTool(bridge=bridge)

    payload = {"action": "insert", "content": "Hi", "target_range": [0, 0], "tab": "README.md"}
    tool.run(payload)

    assert bridge.queue_tab_ids[-1] == "tab-a"


def test_document_edit_tool_embeds_tab_when_bridge_lacks_routing():
    bridge = _LegacyBridgeStub()
    bridge.snapshot["text"] = "Hi"
    tool = DocumentEditTool(bridge=bridge)

    tool.run(action="insert", content="Hi", target_range=(0, 0), tab_id="tab-legacy")

    payload = bridge.calls[-1]
    assert isinstance(payload, dict)
    assert payload["tab_id"] == "tab-legacy"
    metadata = payload.get("metadata") or {}
    assert metadata.get("tab_id") == "tab-legacy"


def test_document_edit_tool_accepts_patch_payload():
    bridge = _EditBridgeStub()
    bridge.last_diff_summary = "patch: +4 chars"
    bridge.last_snapshot_version = "digest-xyz"
    tool = DocumentEditTool(bridge=bridge)

    diff = """--- a/doc.txt
+++ b/doc.txt
@@ -1 +1 @@
-old
+new
"""

    status = tool.run(action="patch", diff=diff, document_version="digest-xyz", content_hash="hash-xyz")

    assert bridge.calls[-1]["action"] == "patch"
    assert "patch" in status and "digest-xyz" in status


def test_document_edit_tool_rejects_patch_without_snapshot_version():
    bridge = _EditBridgeStub()
    bridge.last_snapshot_version = "digest-present"
    tool = DocumentEditTool(bridge=bridge)
def test_document_edit_tool_normalizes_ranges_before_patch_application():
    bridge = _EditBridgeStub()
    bridge.snapshot["text"] = "alpha beta"
    tool = DocumentEditTool(bridge=bridge)

    tool.run(action="replace", content="BETA", target_range={"start": 2, "end": 5})

    assert bridge.calls, "patch payload was not enqueued"
    patch_range = bridge.calls[-1]["ranges"][0]
    assert patch_range["start"] == 0
    assert patch_range["end"] == 5
    assert patch_range["match_text"] == "alpha"
    assert patch_range["replacement"] == "alBETA"


    diff = """--- a/doc.txt\n+++ b/doc.txt\n@@ -1 +1 @@\n-old\n+new\n"""

    with pytest.raises(ValueError, match="document_snapshot"):
        tool.run(action="patch", diff=diff)


def test_document_edit_tool_rejects_inline_edits_when_not_allowed():
    bridge = _EditBridgeStub()
    tool = DocumentEditTool(bridge=bridge, patch_only=False)

    with pytest.raises(ValueError, match="diff-only"):
        tool.run(action="annotate", content="Note", target_range=(0, 0))

    assert bridge.calls == []


def test_document_edit_tool_auto_converts_replace_when_patch_only_enabled():
    bridge = _PatchBridgeStub(text="Goodbye moon", selection=(0, 7), version="digest-42")
    tool = DocumentEditTool(bridge=bridge, patch_only=True)

    status = tool.run(action="replace", content="Hello", target_range=(0, 7))

    assert bridge.calls and bridge.calls[-1]["action"] == "patch"
    assert bridge.calls[-1]["document_version"] == "digest-42"
    assert "digest-42" in status


def test_document_edit_tool_auto_convert_emits_anchor_success_event(monkeypatch: pytest.MonkeyPatch):
    bridge = _PatchBridgeStub(text="Alpha BETA", selection=(0, 5), version="digest-edit-anchor")
    tool = DocumentEditTool(bridge=bridge, patch_only=True)

    captured: list[tuple[str, dict | None]] = []

    def _emit(name: str, payload: dict | None = None) -> None:
        captured.append((name, payload))

    monkeypatch.setattr(document_edit_module, "telemetry_emit", _emit)

    tool.run(action="replace", content="beta", target_range=(0, 5), match_text="BETA")

    event = next((payload for name, payload in captured if name == "patch.anchor"), None)
    assert event is not None
    assert event["status"] == "success"
    assert event["source"] == "document_edit.auto_patch"


def test_document_edit_tool_auto_convert_emits_anchor_failure_event(monkeypatch: pytest.MonkeyPatch):
    bridge = _PatchBridgeStub(text="Alpha beta", selection=(0, 5), version="digest-edit-anchor-fail")
    bridge.snapshot["selection_text"] = "Alpha"
    bridge.snapshot["selection_hash"] = hashlib.sha1(b"Alpha").hexdigest()
    tool = DocumentEditTool(bridge=bridge, patch_only=True)

    captured: list[tuple[str, dict | None]] = []

    def _emit(name: str, payload: dict | None = None) -> None:
        captured.append((name, payload))

    monkeypatch.setattr(document_edit_module, "telemetry_emit", _emit)

    with pytest.raises(ValueError):
        tool.run(
            action="replace",
            content="ALPHA",
            target_range=(0, 5),
            selection_fingerprint="deadbeef",
        )

    event = next((payload for name, payload in captured if name == "patch.anchor"), None)
    assert event is not None
    assert event["status"] == "reject"
    assert event["phase"] == "alignment"


def test_document_edit_tool_requires_range_or_anchor_for_replace_when_selection_unknown():
    bridge = _PatchBridgeStub(text="Hello world", selection=(0, 0), version="digest-43")
    tool = DocumentEditTool(bridge=bridge, patch_only=True)

    with pytest.raises(ValueError, match="target_range or match_text"):
        tool.run(action="replace", content="Hi there")


def test_document_edit_tool_realigns_with_match_text_during_auto_convert():
    bridge = _PatchBridgeStub(text="Alpha BETA Gamma", selection=(0, 5), version="digest-44")
    tool = DocumentEditTool(bridge=bridge, patch_only=True)

    status = tool.run(action="replace", content="beta", target_range=(0, 5), match_text="BETA")

    assert "digest-44" in status
    payload = bridge.calls[-1]
    ranges = payload["ranges"]
    assert ranges[0]["start"] == 6
    assert ranges[0]["match_text"] == "BETA"


def test_document_edit_tool_rejects_selection_fingerprint_mismatch():
    bridge = _PatchBridgeStub(text="Alpha beta", selection=(0, 5), version="digest-45")
    bridge.snapshot["selection_text"] = "Alpha"
    bridge.snapshot["selection_hash"] = hashlib.sha1(b"Alpha").hexdigest()
    tool = DocumentEditTool(bridge=bridge, patch_only=True)

    with pytest.raises(ValueError, match="selection_fingerprint"):
        tool.run(
            action="replace",
            content="ALPHA",
            target_range=(0, 5),
            selection_fingerprint="mismatch",
        )


def test_document_edit_tool_blocks_caret_replace_without_insert_action():
    bridge = _PatchBridgeStub(text="Hello world", selection=(5, 5), version="digest-46")
    tool = DocumentEditTool(bridge=bridge, patch_only=True)

    with pytest.raises(ValueError, match="Caret inserts"):
        tool.run(action="replace", content="X", target_range=(5, 5))


def test_document_edit_tool_refreshes_snapshot_when_selection_missing():
    fingerprint = hashlib.sha1(b"Hello").hexdigest()

    class _LazySelectionBridge(_PatchBridgeStub):
        def generate_snapshot(self, *, include_text: bool | None = None, **kwargs):  # type: ignore[override]
            include_flag = bool(include_text)
            snapshot = super().generate_snapshot(include_text=include_flag, **kwargs)
            snapshot.pop("selection_text", None)
            snapshot.pop("selection_hash", None)
            if include_flag:
                snapshot["selection_text"] = "Hello"
                snapshot["selection_hash"] = fingerprint
            return snapshot

    bridge = _LazySelectionBridge(text="Hello world", selection=(0, 5), version="digest-refresh")
    tool = DocumentEditTool(bridge=bridge, patch_only=True)

    status = tool.run(
        action="replace",
        content="HELLO",
        target_range=(0, 5),
        selection_fingerprint=fingerprint,
    )

    assert "digest-refresh" in status
    assert len(bridge.snapshot_requests) >= 2
    assert bridge.snapshot_requests[-1]["include_text"] is True


def test_document_edit_tool_allows_patches_when_patch_only_enabled():
    bridge = _EditBridgeStub()
    bridge.last_diff_summary = "patch"
    bridge.last_snapshot_version = "digest"
    tool = DocumentEditTool(bridge=bridge, patch_only=True)

    diff = """--- a/doc.txt\n+++ b/doc.txt\n@@ -1 +1 @@\n-old\n+new\n"""
    status = tool.run(action="patch", diff=diff, document_version="digest", content_hash="hash-digest")

    assert bridge.calls and bridge.calls[-1]["action"] == "patch"
    assert "patch" in status


def test_document_apply_patch_tool_builds_and_applies_diff():
    bridge = _PatchBridgeStub(text="Alpha beta", selection=(6, 10), version="digest-7")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    status = tool.run(content="BETA", target_range=(6, 10))

    assert bridge.calls and bridge.calls[-1]["action"] == "patch"
    assert "digest-7" in status
    assert bridge.calls[-1]["document_version"] == "digest-7"


def test_document_apply_patch_tool_targets_specific_tab():
    bridge = _PatchBridgeStub(text="Alpha beta", selection=(0, 5), version="digest-8")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    tool.run(content="ALPHA", target_range=(0, 5), tab_id="tab-b")

    assert bridge.queue_tab_ids[-1] == "tab-b"
    assert bridge.snapshot_requests[-1]["tab_id"] == "tab-b"


def test_document_apply_patch_tool_validates_snapshot_version():
    bridge = _PatchBridgeStub(text="Hello world", selection=(0, 5), version="digest-1")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    with pytest.raises(DocumentVersionMismatchError):
        tool.run(content="Hola", target_range=(0, 5), document_version="digest-old")


def test_document_apply_patch_tool_skips_noop_edits():
    bridge = _PatchBridgeStub(text="Hello world", selection=(0, 5), version="digest-2")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    outcome = tool.run(content="Hello", target_range=(0, 5))

    assert outcome.startswith("skipped")
    assert bridge.calls == []


def test_document_apply_patch_tool_emits_anchor_success_event(monkeypatch: pytest.MonkeyPatch):
    bridge = _PatchBridgeStub(text="Alpha beta", selection=(0, 5), version="digest-anchor")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    captured: list[tuple[str, dict | None]] = []

    def _emit(name: str, payload: dict | None = None) -> None:
        captured.append((name, payload))

    monkeypatch.setattr(document_apply_patch_module, "telemetry_emit", _emit)

    tool.run(content="ALPHA", target_range=(0, 5), match_text="Alpha")

    event = next((payload for name, payload in captured if name == "patch.anchor"), None)
    assert event is not None
    assert event["status"] == "success"
    assert event["source"] == "document_apply_patch"
    assert event["anchor_source"] == "match_text"


def test_document_apply_patch_tool_emits_anchor_failure_event(monkeypatch: pytest.MonkeyPatch):
    bridge = _PatchBridgeStub(text="Alpha beta", selection=(0, 0), version="digest-anchor-fail")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    captured: list[tuple[str, dict | None]] = []

    def _emit(name: str, payload: dict | None = None) -> None:
        captured.append((name, payload))

    monkeypatch.setattr(document_apply_patch_module, "telemetry_emit", _emit)

    with pytest.raises(ValueError):
        tool.run(content="ALPHA")

    event = next((payload for name, payload in captured if name == "patch.anchor"), None)
    assert event is not None
    assert event["status"] == "reject"
    assert event["phase"] == "requirements"


def test_document_apply_patch_tool_handles_missing_rationale():
    bridge = _PatchBridgeStub(text="", selection=(0, 0), version="digest-9")
    edit_tool = DocumentEditTool(bridge=bridge)
    tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)

    status = tool.run(content="Hello world", target_range=(0, 0))

    assert bridge.calls and bridge.calls[-1]["action"] == "patch"
    assert "digest-9" in status


def test_register_default_tools_succeeds_when_controller_accepts_all():
    controller = _ControllerStub()
    context = _registry_context(controller)

    register_default_tools(context)

    registered = set(controller.registered)
    assert {
        "document_snapshot",
        "document_chunk",
        "document_edit",
        "document_apply_patch",
        "list_tabs",
    }.issubset(registered)


def test_register_default_tools_aggregates_failures_without_stopping_others():
    controller = _ControllerStub(fail_on={"document_edit", "list_tabs"})
    context = _registry_context(controller)

    with pytest.raises(ToolRegistrationError) as excinfo:
        register_default_tools(context)

    failure_names = {failure.name for failure in excinfo.value.failures}
    assert {"document_edit", "list_tabs"} <= failure_names
    # Remaining tools (like snapshot) still register successfully.
    assert "document_snapshot" in controller.registered


def test_diff_builder_tool_generates_diff_and_detects_noop():
    tool = DiffBuilderTool()

    diff = tool.run("alpha\n", "beta\n")

    assert diff.startswith("--- a/")
    assert "beta" in diff

    with pytest.raises(ValueError):
        tool.run("same", "same")


def test_document_snapshot_tool_includes_diff_summary_and_version():
    provider = _SnapshotProviderStub()
    tool = DocumentSnapshotTool(provider=provider)

    snapshot = tool.run(delta_only=True)

    assert provider.delta_only_calls == [True]
    assert snapshot["version"] == "base"
    assert snapshot["diff_summary"] == "+1 char"
    assert "diff_summary" not in provider.snapshot


def test_document_snapshot_tool_falls_back_to_provider_version():
    provider = _SnapshotProviderStub()
    provider.snapshot.pop("version", None)
    tool = DocumentSnapshotTool(provider=provider)

    snapshot = tool.run()

    assert snapshot["version"] == "digest-1"


def test_document_snapshot_tool_attaches_outline_digest_when_resolver_present():
    provider = _SnapshotProviderStub()
    calls: list[str | None] = []

    def resolver(document_id: str | None) -> str | None:
        calls.append(document_id)
        return "outline-hash" if document_id == "doc-stub" else None

    tool = DocumentSnapshotTool(provider=provider, outline_digest_resolver=resolver)

    snapshot = tool.run()

    assert snapshot["outline_digest"] == "outline-hash"
    assert calls == ["doc-stub"]


def test_tool_usage_advisor_tool_returns_serialized_advice():
    captured: dict[str, object] = {}

    def _advisor(**kwargs):
        captured.update(kwargs)
        return AnalysisAdvice(document_id="doc-1", document_version="v1", required_tools=("document_chunk",))

    tool = ToolUsageAdvisorTool(_advisor)
    response = tool.run(document_id="doc-1", selection_start=5, selection_end=25, force_refresh=True)
    assert response["status"] == "ok"
    assert response["advice"]["document_id"] == "doc-1"
    assert captured["force_refresh"] is True


def test_document_snapshot_tool_uses_tab_identifier_when_document_id_missing():
    provider = _SnapshotProviderStub()
    provider.snapshot.pop("document_id", None)

    tool = DocumentSnapshotTool(
        provider=provider,
        outline_digest_resolver=lambda doc_id: "tab-digest" if doc_id == "tab-extra" else None,
    )

    snapshot = tool.run(tab_id="tab-extra")

    assert snapshot["outline_digest"] == "tab-digest"


def test_search_replace_tool_updates_document_scope():
    bridge = _SearchReplaceBridgeStub(text="Hello world world", selection=(0, 0), version="digest-0")
    tool = SearchReplaceTool(bridge=bridge, preview_chars=100)

    result = tool.run(pattern="world", replacement="earth")

    assert result.replacements == 2
    assert result.applied is True
    assert result.scope == "document"
    assert bridge.queue_calls
    payload = bridge.queue_calls[-1]
    assert payload["action"] == "patch"
    assert payload["document_version"] == "digest-0"
    assert payload["content_hash"] == bridge.snapshot["content_hash"]
    assert payload["diff"].startswith("--- a/")
    assert "earth" in payload["diff"]
    assert result.document_version == "updated"
    assert "earth" in result.preview
    assert result.diff_preview and result.diff_preview.startswith("---")
    assert result.max_replacements == tool.default_max_replacements
    assert result.limited is False
    metadata = payload["metadata"]
    assert metadata["matches"] == 2
    assert metadata["limited"] is False
    assert metadata["scope"] == "document"
    assert metadata["target_range"] == {"start": 0, "end": len("Hello world world")}


def test_tools_expose_expected_summarizable_flags():
    bridge = _EditBridgeStub()
    edit_tool = DocumentEditTool(bridge=bridge)
    snapshot_tool = DocumentSnapshotTool(provider=_SnapshotProviderStub())
    patch_tool = DocumentApplyPatchTool(bridge=bridge, edit_tool=edit_tool)
    list_tool = ListTabsTool(provider=bridge)
    search_tool = SearchReplaceTool(bridge=_SearchReplaceBridgeStub(text="text"))
    diff_tool = DiffBuilderTool()

    assert snapshot_tool.summarizable is True
    assert list_tool.summarizable is True
    assert diff_tool.summarizable is True
    assert edit_tool.summarizable is False
    assert patch_tool.summarizable is False
    assert search_tool.summarizable is False
    assert getattr(validate_snippet, "summarizable", None) is False


def test_search_replace_tool_respects_selection_scope_and_dry_run():
    bridge = _SearchReplaceBridgeStub(text="Alpha Beta Gamma", selection=(6, 10), version="digest-1")
    tool = SearchReplaceTool(bridge=bridge, preview_chars=32)

    result = tool.run(
        pattern="beta",
        replacement="delta",
        scope="selection",
        dry_run=True,
        match_case=False,
        whole_word=True,
        max_replacements=1,
    )

    assert result.replacements == 1
    assert result.applied is False
    assert result.dry_run is True
    assert result.scope == "selection"
    assert result.target_range == (6, 10)
    assert bridge.queue_calls == []
    assert result.document_version == "digest-1"
    assert result.preview
    assert result.diff_preview and result.diff_preview.startswith("---")
    assert result.max_replacements == 1
    assert result.limited is False


def test_search_replace_tool_honors_replacement_cap_and_reports_limited():
    bridge = _SearchReplaceBridgeStub(text="aaaaa", selection=(0, 0), version="digest-cap")
    tool = SearchReplaceTool(bridge=bridge)

    result = tool.run(pattern="a", replacement="b", max_replacements=2)

    assert result.replacements == 2
    assert result.limited is True
    assert result.max_replacements == 2
    assert result.diff_preview
    assert bridge.queue_calls
    payload = bridge.queue_calls[-1]
    assert payload["action"] == "patch"
    metadata = payload["metadata"]
    assert metadata["limited"] is True
    assert metadata["matches"] == 2
    assert metadata["max_replacements"] == 2


def test_search_replace_tool_validates_inputs():
    bridge = _SearchReplaceBridgeStub(text="Hello", selection=(0, 0))
    tool = SearchReplaceTool(bridge=bridge)

    with pytest.raises(ValueError):
        tool.run(pattern="", replacement="x")

    with pytest.raises(ValueError):
        tool.run(pattern="x", replacement="y", max_replacements=0)


def test_validate_snippet_handles_json_and_unknown_formats():
    success = validate_snippet('{"foo": 1}', "json")
    assert success.ok is True

    failure = validate_snippet("{foo}", "json")
    assert failure.ok is False

    unsupported = validate_snippet("value", "toml")
    assert unsupported.ok is False
    assert "Unsupported format" in unsupported.message


def test_validate_snippet_reports_yaml_errors(monkeypatch: pytest.MonkeyPatch):
    from tinkerbell.ai.tools import validation as validation_module

    errors = [
        yaml_json.ValidationError(message="first", line=2),
        yaml_json.ValidationError(message="second", line=4),
    ]

    monkeypatch.setitem(validation_module._FORMAT_VALIDATORS, "yaml", lambda _text: errors)

    outcome = validation_module.validate_snippet("bad", "yaml")

    assert outcome.ok is False
    assert "Line 2: first (+1 more issue)." == outcome.message


def test_validate_snippet_success_path_can_be_stubbed(monkeypatch: pytest.MonkeyPatch):
    from tinkerbell.ai.tools import validation as validation_module

    monkeypatch.setitem(validation_module._FORMAT_VALIDATORS, "yaml", lambda _text: [])

    outcome = validation_module.validate_snippet("foo: bar", "yaml")

    assert outcome.ok is True


def test_validate_snippet_flags_markdown_heading_jumps():
    markdown = """# Title\n### Skipped level"""

    outcome = validate_snippet(markdown, "markdown")

    assert outcome.ok is False
    assert "Heading level" in outcome.message
    assert "Line 2" in outcome.message


def test_validate_snippet_detects_unclosed_markdown_fence():
    markdown = """```python\ncode sample"""

    outcome = validate_snippet(markdown, "md")

    assert outcome.ok is False
    assert "fenced code" in outcome.message.lower()


def test_register_snippet_validator_supports_custom_formats(monkeypatch: pytest.MonkeyPatch):
    from tinkerbell.ai.tools import validation as validation_module

    registry_copy = dict(validation_module._FORMAT_VALIDATORS)
    monkeypatch.setattr(validation_module, "_FORMAT_VALIDATORS", registry_copy)

    def _fake_validator(text: str):
        if "invalid" in text:
            return [yaml_json.ValidationError(message="custom failure", line=3)]
        return []

    validation_module.register_snippet_validator("custom", _fake_validator)

    failure = validation_module.validate_snippet("invalid", "custom")
    assert failure.ok is False
    assert "custom failure" in failure.message

    success = validation_module.validate_snippet("valid", "custom")
    assert success.ok is True
